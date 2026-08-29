"""Composable rollout hooks: the "extra bits" that run alongside a rollout."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..inference.is_sampler import run_importance_sampling
from .rollout import RolloutState


def check_hooks(hooks: Sequence["RolloutHook"]) -> list["RolloutHook"]:
    """Returns the hooks as a list, raising if any cannot be registered.

    Both failures here are the same mistake -- a hook that isn't a usable
    nn.Module -- and both would otherwise surface much later as an obscure
    error from inside nn.ModuleList or .to().
    """
    hooks = list(hooks)
    for hook in hooks:
        if not isinstance(hook, nn.Module):
            raise TypeError(
                f"Hooks must subclass RolloutHook (an nn.Module); got {type(hook).__name__}."
            )
        if not hasattr(hook, "_parameters"):
            raise TypeError(
                f"{type(hook).__name__}.__init__ does not call super().__init__(); "
                "RolloutHook is an nn.Module and must be initialized as one."
            )
    return hooks


class RolloutHook(nn.Module):
    """Interface for rollout callbacks.

    All methods are no-ops by default; a concrete hook overrides only the
    callbacks it needs. Hooks must keep any per-call state on
    `state.scratch`, not on `self`, so one instance can be reused across
    rollouts (e.g. across train/eval/warmup steps) without cross-contaminating
    calls.

    A hook is an `nn.Module` so that anything it owns (an action net, a
    buffer) is registered on the `Designer` holding it, and moved by
    `designer.to(device)` along with the rest of the rollout. A subclass
    defining `__init__` must therefore call `super().__init__()` before
    assigning any module or tensor attribute -- `Designer` raises if it is
    handed a hook that did not.

    Hooks are called through their `before_propose`/`after_propose`/
    `after_step`/`on_rollout_end` callbacks, never through `forward()`.
    """

    def __init__(self):
        """Initializes the hook."""
        super().__init__()

    def before_propose(self, state: RolloutState) -> None:
        """Runs before the next design is proposed."""

    def after_propose(self, state: RolloutState) -> None:
        """Runs after a design is proposed, before the outcome is sampled."""

    def after_step(self, state: RolloutState) -> None:
        """Runs after the outcome is sampled and appended to the history."""

    def on_rollout_end(self, state: RolloutState) -> dict[str, Any] | None:
        """Runs once after all steps complete.

        Returns a dict merged into the final `Rollout.extra`, or None.
        """
        return None


class PosteriorUpdateHook(RolloutHook):
    """Runs posterior inference and updates `task.current_prior` after every step."""

    def __init__(
        self,
        inference_fn=run_importance_sampling,
        *,
        store_history: bool = False,
        **inference_kwargs,
    ):
        """inference_fn(task, y_history, design_history, **inference_kwargs) -> Distribution."""
        super().__init__()
        self.inference_fn = inference_fn
        self.store_history = store_history
        self.inference_kwargs = inference_kwargs

    def after_step(self, state: RolloutState) -> None:
        """Infers the posterior from the history so far and updates the task."""
        posterior = self.inference_fn(
            state.task, state.y_history, state.design_history, **self.inference_kwargs
        )
        state.task.update_prior(posterior)
        if self.store_history:
            state.scratch.setdefault("posterior_history", []).append(posterior)

    def on_rollout_end(self, state: RolloutState) -> dict[str, Any] | None:
        """Attaches the final (and optionally per-step) posterior to the rollout."""
        extra = {"posterior": state.task.current_prior}
        if self.store_history:
            extra["posterior_history"] = state.scratch["posterior_history"]
        return extra


class ActionHook(RolloutHook):
    """Runs an action net either once on the full history or after every step."""

    def __init__(self, action_net, *, every_step: bool = False, action_space=None):
        """action_net(y_history, design_history) -> action, [b, *event_shape_action]."""
        super().__init__()
        self.action_net = action_net
        self.every_step = every_step
        self.action_space = action_space

    def step_action(self, y_history: Tensor, design_history: Tensor) -> Tensor:
        """Computes the action by calling the action_net and possibly transforming via the action space."""
        action = self.action_net(y_history, design_history)
        if self.action_space:
            action = self.action_space.to_design(action)
        return action

    def after_step(self, state: RolloutState) -> None:
        """Computes and stashes an action from the history so far, if requested every step."""
        if not self.every_step:
            return
        action_t = self.step_action(state.y_history, state.design_history)
        state.scratch.setdefault("action_history", []).append(action_t)

    def on_rollout_end(self, state: RolloutState) -> dict[str, Any] | None:
        """Attaches the terminal action, or the stacked per-step action history."""
        if self.every_step:
            return {
                "action_history": torch.stack(state.scratch["action_history"], dim=1)
            }
        return {"action": self.step_action(state.y_history, state.design_history)}
