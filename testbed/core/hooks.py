"""Composable rollout hooks: the "extra bits" that run alongside a rollout."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..inference.is_sampler import run_importance_sampling
from .rollout import RolloutState
from .util import squeeze_history


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


def apply_rollout_end(
    hooks: Sequence["RolloutHook"], state: "RolloutState", rollout
) -> None:
    """Runs every hook's ``on_rollout_end``, merging what they return into ``rollout.extra``."""
    for hook in hooks:
        extra_update = hook.on_rollout_end(state)
        if extra_update is None:
            continue
        colliding_keys = rollout.extra.keys() & extra_update.keys()
        if colliding_keys:
            raise ValueError(
                f"Multiple hooks wrote the same Rollout.extra key(s): {colliding_keys}"
            )
        rollout.extra.update(extra_update)


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
    """Runs posterior inference after every step, setting `task.current_prior` to the result.

    `inference(task, y_history, design_history) -> Distribution` conditions
    `task.prior` on the rollout's history so far, of shape `[t, *event_shape]`.
    Only batch size 1 is supported, since the task holds a single posterior.
    `task.current_prior` holds the previous step's posterior when it is called,
    which an engine may use, e.g. to warm-start. Configure an engine with
    `functools.partial`, e.g. `partial(run_importance_sampling, n_proposal=10_000)`.

    The task's `current_prior` is put back as it was when the rollout ends, and
    the final posterior is returned as `rollout.extra["posterior"]`.
    """

    def __init__(
        self, inference=run_importance_sampling, *, store_history: bool = False
    ):
        """Initializes the hook.

        store_history: also return the posterior after each step, as the list
            `rollout.extra["posterior_history"]` (default: False).
        """
        super().__init__()
        self.inference = inference
        self.store_history = store_history

    def after_step(self, state: RolloutState) -> None:
        """Infers the posterior from the history so far and updates the task."""
        task = state.task
        state.scratch.setdefault("prior_before_rollout", task.current_prior)
        posterior = self.inference(
            task,
            squeeze_history(state.y_history, task.event_shape_y),
            squeeze_history(state.design_history, task.event_shape_design),
        )
        task.update_prior(posterior)
        if self.store_history:
            state.scratch.setdefault("posterior_history", []).append(task.current_prior)

    def on_rollout_end(self, state: RolloutState) -> dict[str, Any] | None:
        """Returns the final (and optionally per-step) posterior and restores the prior."""
        extra = {"posterior": state.task.current_prior}
        if self.store_history:
            extra["posterior_history"] = state.scratch.get("posterior_history", [])
        if "prior_before_rollout" in state.scratch:
            state.task.update_prior(state.scratch["prior_before_rollout"])
        return extra


class ActionHook(RolloutHook):
    """Runs an action net either once on the full history or after every step."""

    def __init__(
        self, action_net, *, every_step: bool = False, use_raw_history: bool = False
    ):
        """action_net(y_history, design_history) -> action, [b, *event_shape_action].

        ``use_raw_history`` conditions on raw rather than physical designs, and
        should match the designer's own setting so that both nets see the same
        coordinates.
        """
        super().__init__()
        self.action_net = action_net
        self.every_step = every_step
        self.use_raw_history = bool(use_raw_history)

    def design_history(self, state: RolloutState) -> Tensor:
        """Returns the design history the action net conditions on.

        Raises rather than falling back to physical designs: a silent fallback
        would feed the action net two different coordinate systems within one
        run (a designer supplying no raw designs during warmup, raw ones after),
        which trains badly and reports nothing.
        """
        if not self.use_raw_history:
            return state.design_history
        if state.raw_design_history is None:
            raise ValueError(
                "ActionHook(use_raw_history=True) needs a raw design history, but "
                f"{type(state.task).__name__}'s designer produced none. Its design "
                "space likely implements no raw <-> design map."
            )
        return state.raw_design_history

    def step_action(self, y_history: Tensor, design_history: Tensor) -> Tensor:
        """Computes the action by calling the action_net."""
        return self.action_net(y_history, design_history)

    def after_step(self, state: RolloutState) -> None:
        """Computes and stashes an action from the history so far, if requested every step."""
        if not self.every_step:
            return
        action_t = self.step_action(state.y_history, self.design_history(state))
        state.scratch.setdefault("action_history", []).append(action_t)

    def on_rollout_end(self, state: RolloutState) -> dict[str, Any] | None:
        """Attaches the terminal action, or the stacked per-step action history."""
        if self.every_step:
            return {
                "action_history": torch.stack(state.scratch["action_history"], dim=1)
            }
        return {"action": self.step_action(state.y_history, self.design_history(state))}
