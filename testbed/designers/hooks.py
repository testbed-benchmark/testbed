"""Composable rollout hooks: the "extra bits" that run alongside a rollout."""

from typing import Any

import torch

from testbed.core.rollout import RolloutState

from ..inference.is_sampler import run_importance_sampling


class RolloutHook:
    """Interface for rollout callbacks.

    All methods are no-ops by default; a concrete hook overrides only the
    callbacks it needs. Hooks must keep any per-call state on
    `state.scratch`, not on `self`, so one instance can be reused across
    rollouts (e.g. across train/eval/warmup steps) without cross-contaminating
    calls.
    """

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

    def __init__(self, action_net, *, every_step: bool = False):
        """action_net(y_history, design_history) -> action, [b, *event_shape_action]."""
        self.action_net = action_net
        self.every_step = every_step

    def after_step(self, state: RolloutState) -> None:
        """Computes and stashes an action from the history so far, if requested every step."""
        if not self.every_step:
            return
        action_t = self.action_net(state.y_history, state.design_history)
        state.scratch.setdefault("action_history", []).append(action_t)

    def on_rollout_end(self, state: RolloutState) -> dict[str, Any] | None:
        """Attaches the terminal action, or the stacked per-step action history."""
        if self.every_step:
            return {
                "action_history": torch.stack(state.scratch["action_history"], dim=1)
            }
        return {"action": self.action_net(state.y_history, state.design_history)}
