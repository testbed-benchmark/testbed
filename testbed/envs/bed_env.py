"""Repo-native BED environment."""

from __future__ import annotations

from math import prod
from typing import Any

import torch
from torch import Tensor

from ..tasks.base_task import Task
from .design_spaces import DesignSpace
from .history import History
from .observations import ObservationFunction, PaddedHistoryObservation
from .rewards import RewardFunction, ZeroReward


class BEDEnv:
    """Stateful Bayesian experimental design rollout environment."""

    def __init__(
        self,
        task: Task,
        horizon: int,
        design_space: DesignSpace | None = None,
        reward_fn: RewardFunction | None = None,
        observation_fn: ObservationFunction | None = None,
        batch_shape: tuple[int, ...] = (),
        validate_actions: bool = True,
        project_actions: bool = True,
        include_theta_in_info: bool = False,
        track_grad: bool = False,
    ):
        """Initializes a BED environment."""
        self.task = task
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.design_space = (
            design_space if design_space is not None else task.design_space
        )
        self.reward_fn = reward_fn if reward_fn is not None else ZeroReward()
        self.observation_fn = (
            observation_fn
            if observation_fn is not None
            else PaddedHistoryObservation(self.horizon)
        )
        self.batch_shape = torch.Size(batch_shape)
        self.validate_actions = validate_actions
        self.project_actions = project_actions
        self.include_theta_in_info = include_theta_in_info
        self.track_grad = track_grad

        self.t = 0
        self._has_reset = False
        self._history: History | None = None
        self.theta_0: Tensor | None = None
        self.theta_all: Tensor | None = None

    @property
    def device(self) -> torch.device:
        """Returns the task device."""
        return self.task.device

    @property
    def history(self) -> History:
        """Returns the current history."""
        if self._history is None:
            raise RuntimeError("Environment has not been reset.")
        return self._history

    @property
    def terminated(self) -> Tensor:
        """Returns terminal flags for the current timestep."""
        return torch.full(
            self.batch_shape,
            self._has_reset and self.t >= self.horizon,
            device=self.device,
        )

    @property
    def truncated(self) -> Tensor:
        """Returns truncation flags."""
        return torch.zeros(self.batch_shape, device=self.device, dtype=torch.bool)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        theta: Tensor | None = None,
        theta_all: Tensor | None = None,
        batch_shape: tuple[int, ...] | None = None,
    ) -> tuple[Any, dict]:
        """Starts a new episode.

        Observation functions may return views or references to mutable environment
        state. In particular, ``PaddedHistoryObservation`` returns the tensors owned
        by ``env.history``. Callers that retain an observation across ``step()`` or
        ``reset()`` must clone it first.
        """
        if seed is not None:
            torch.manual_seed(seed)
        if batch_shape is not None:
            self.batch_shape = torch.Size(batch_shape)

        self.t = 0
        self.theta_all = self._init_theta_all(theta=theta, theta_all=theta_all)
        self.theta_0 = self.theta_all.select(dim=len(self.batch_shape), index=0)
        self._history = History.empty(
            self.batch_shape,
            self.horizon,
            self.task.event_shape_design,
            self.task.event_shape_y,
            device=self.device,
        )
        self._has_reset = True
        self.reward_fn.reset(self)
        self.observation_fn.reset(self)

        info: dict[str, Any] = {"t": self.t, "history": self.history}
        if options:
            info["options"] = options
        if self.include_theta_in_info:
            info["theta_0"] = self.theta_0
            info["theta_all"] = self.theta_all
        return self.observation_fn(self), info

    def step(self, action: Tensor) -> tuple[Any, Tensor, Tensor, Tensor, dict]:
        """Advances the environment by one design/outcome pair.

        This updates ``env.history`` in place before constructing the returned
        observation. Previously returned observations that reference the history,
        including those from ``PaddedHistoryObservation``, therefore change during
        this call. Clone observations before ``step()`` when storing transitions.
        """
        if not self._has_reset:
            raise RuntimeError("Environment must be reset before step().")
        if bool(torch.any(self.terminated).item()):
            raise RuntimeError("Cannot step a terminated environment; call reset().")

        raw_action = torch.as_tensor(action, device=self.device)
        if not self.track_grad:
            raw_action = raw_action.detach()

        grad_context = torch.enable_grad() if self.track_grad else torch.no_grad()
        with grad_context:
            design = self._prepare_design(raw_action)
            outcome = self.task.sample_outcome(self.theta_0, design)
            self.history.write(self.t, design, outcome)
            reward = self.reward_fn.step(self, design, outcome)
        self.t += 1

        info: dict[str, Any] = {
            "t": self.t,
            "design": design,
            "outcome": outcome,
            "history": self.history,
            "raw_action": raw_action,
            "projected_action": design,
        }
        if self.include_theta_in_info:
            info["theta_0"] = self.theta_0
            info["theta_all"] = self.theta_all
        return self.observation_fn(self), reward, self.terminated, self.truncated, info

    def _init_theta_all(
        self, *, theta: Tensor | None = None, theta_all: Tensor | None = None
    ) -> Tensor:
        """Samples or validates true and contrastive parameters."""
        n_contrastive = int(getattr(self.reward_fn, "n_contrastive", 0))
        L = n_contrastive + 1
        event_shape = self.task.event_shape_theta

        if theta_all is not None:
            theta_all = torch.as_tensor(theta_all, device=self.device)
            expected = (*self.batch_shape, L, *event_shape)
            if theta_all.shape != expected:
                raise ValueError(
                    f"theta_all must have shape {expected}; got {tuple(theta_all.shape)}."
                )
            return self._detach_if_needed(theta_all)

        if theta is not None:
            theta_0 = torch.as_tensor(theta, device=self.device)
            expected = (*self.batch_shape, *event_shape)
            if theta_0.shape != expected:
                raise ValueError(
                    f"theta must have shape {expected}; got {tuple(theta_0.shape)}."
                )
        else:
            n = prod(self.batch_shape)
            theta_0 = self.task.sample_current_prior(n).reshape(
                *self.batch_shape, *event_shape
            )

        if n_contrastive == 0:
            return self._detach_if_needed(theta_0.unsqueeze(len(self.batch_shape)))

        n = prod(self.batch_shape) * n_contrastive
        theta_contrastive = self.task.sample_current_prior(n).reshape(
            *self.batch_shape, n_contrastive, *event_shape
        )
        particle_dim = len(self.batch_shape)
        theta_all = torch.cat(
            [theta_0.unsqueeze(particle_dim), theta_contrastive], dim=particle_dim
        )
        return self._detach_if_needed(theta_all)

    def _detach_if_needed(self, tensor: Tensor) -> Tensor:
        """Detaches rollout tensors unless differentiable env interaction is enabled."""
        if self.track_grad:
            return tensor
        return tensor.detach()

    def _prepare_design(self, raw_action: Tensor) -> Tensor:
        """Converts, projects, validates, and shape-checks an action."""
        design = raw_action
        design_space = self.design_space
        if design_space is not None:
            if self.project_actions:
                design = design_space.project(design)
            if self.validate_actions and not design_space.contains(design):
                raise ValueError("Action is outside the design space.")

        expected_event_shape = (
            design_space.event_shape
            if design_space is not None
            else self.task.event_shape_design
        )
        expected = (*self.batch_shape, *expected_event_shape)
        if design.shape != expected:
            raise ValueError(
                f"Action must have shape {expected}; got {tuple(design.shape)}."
            )
        return design
