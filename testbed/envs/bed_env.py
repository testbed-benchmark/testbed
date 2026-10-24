"""Repo-native BED environment."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..tasks.base import Task
from .history import History
from .rewards import RewardFunction, ZeroReward


class BEDEnv:
    """Stateful Bayesian experimental design rollout environment."""

    def __init__(
        self,
        task: Task,
        horizon: int,
        reward_fn: RewardFunction | None = None,
        batch_shape: tuple[int, ...] = (),
        validate_designs: bool = True,
        project_designs: bool = True,
        include_theta_in_info: bool = False,
        track_grad: bool = False,
    ):
        """Initializes a BED environment."""
        self.task = task
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.reward_fn = reward_fn if reward_fn is not None else ZeroReward()
        self.batch_shape = torch.Size(batch_shape)
        self.validate_designs = validate_designs
        self.project_designs = project_designs
        self.include_theta_in_info = include_theta_in_info
        self.track_grad = track_grad

        self.t = 0
        self._has_reset = False
        self._history: History | None = None
        self.theta_0: Tensor | None = None

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

        Returns ``env.history`` as the observation. The returned ``History`` owns
        mutable tensor state; callers that retain an observation across ``step()``
        or ``reset()`` must clone its tensors first.
        """
        if seed is not None:
            torch.manual_seed(seed)
        if batch_shape is not None:
            self.batch_shape = torch.Size(batch_shape)

        self.t = 0
        event_shape = self.task.event_shape_theta

        theta_all_tensor: Tensor | None = None
        if theta_all is not None:
            theta_all_tensor = self._detach_if_needed(
                torch.as_tensor(theta_all, device=self.device)
            )
            L = int(getattr(self.reward_fn, "n_contrastive", 0)) + 1
            expected = (*self.batch_shape, L, *event_shape)
            if theta_all_tensor.shape != expected:
                raise ValueError(
                    f"theta_all must have shape {expected}; "
                    f"got {tuple(theta_all_tensor.shape)}."
                )
            self.theta_0 = theta_all_tensor.select(dim=len(self.batch_shape), index=0)
        elif theta is not None:
            theta_0 = self._detach_if_needed(torch.as_tensor(theta, device=self.device))
            expected = (*self.batch_shape, *event_shape)
            if theta_0.shape != expected:
                raise ValueError(
                    f"theta must have shape {expected}; got {tuple(theta_0.shape)}."
                )
            self.theta_0 = theta_0
        else:
            self.theta_0 = self._detach_if_needed(
                self.task.sample_current_prior(self.batch_shape)
            )

        self._history = History.empty(
            self.batch_shape,
            self.horizon,
            self.task.event_shape_design,
            self.task.event_shape_y,
            device=self.device,
        )
        self.reward_fn.reset(self, theta_all=theta_all_tensor)
        self._has_reset = True

        info: dict[str, Any] = {"t": self.t, "history": self.history}
        if options:
            info["options"] = options
        if self.include_theta_in_info:
            info["theta_0"] = self.theta_0
        return self.history, info

    def step(self, design: Tensor) -> tuple[Any, Tensor, Tensor, Tensor, dict]:
        """Advances the environment by one design/outcome pair.

        Returns the updated ``env.history`` as the observation. The ``History``
        tensors are mutated in place, so clone them before ``step()`` if you need
        to retain the pre-step values (e.g. when storing transitions).
        """
        if not self._has_reset:
            raise RuntimeError("Environment must be reset before step().")
        if self.t >= self.horizon:
            raise RuntimeError("Cannot step a terminated environment; call reset().")

        raw_design = self._detach_if_needed(torch.as_tensor(design, device=self.device))

        grad_context = torch.enable_grad() if self.track_grad else torch.no_grad()
        with grad_context:
            design = self._prepare_design(raw_design)
            # Slots [0, t) are the steps already taken; a memoryless task drops
            # them, one with memory conditions on them.
            past = self.history.prefix(self.t)
            outcome = self.task.sample_outcome(
                self.theta_0,
                design,
                y_history=past.outcomes,
                design_history=past.designs,
            )
            self.history.write(self.t, design, outcome)
            reward = self.reward_fn.step(self, design, outcome)
        self.t += 1

        info: dict[str, Any] = {
            "t": self.t,
            "design": design,
            "outcome": outcome,
            "history": self.history,
            "raw_design": raw_design,
        }
        if self.include_theta_in_info:
            info["theta_0"] = self.theta_0
        return self.history, reward, self.terminated, self.truncated, info

    def _detach_if_needed(self, tensor: Tensor) -> Tensor:
        """Detaches rollout tensors unless differentiable env interaction is enabled."""
        if self.track_grad:
            return tensor
        return tensor.detach()

    def _prepare_design(self, raw_design: Tensor) -> Tensor:
        """Converts, projects, validates, and shape-checks a design."""
        design = raw_design
        if self.project_designs:
            design = self.task.design_space.project(design)
        if self.validate_designs and not self.task.design_space.contains(design):
            raise ValueError("Design is outside the design space.")

        expected = (*self.batch_shape, *self.task.event_shape_design)
        if design.shape != expected:
            raise ValueError(
                f"Design must have shape {expected}; got {tuple(design.shape)}."
            )
        return design
