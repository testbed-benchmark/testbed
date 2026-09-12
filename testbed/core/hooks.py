"""Composable rollout hooks: the "extra bits" that run alongside a rollout."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ..inference.is_sampler import effective_sample_size, run_importance_sampling
from .rollout import RolloutState


def ess_record(posterior, step: int) -> dict[str, Any] | None:
    """ESS diagnostic for one inference call, or None if the posterior has no weights.

    A degenerate posterior is the quiet failure mode of importance sampling: the
    rollout finishes and the bounds look plausible off a handful of distinct
    particles. So every hook that runs inference records this.

    ``step`` is the number of steps the posterior conditions on, 1-indexed, so it
    joins against the other per-refinement records and ``posterior_history``.
    """
    ess = effective_sample_size(posterior)
    if ess is None:
        return None
    n_particles = posterior.locs.shape[0]
    return {
        "step": step,
        "ess": ess,
        "ess_fraction": ess / n_particles,
        "n_particles": n_particles,
    }


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
        eval_fn=None,
        store_posterior_history: bool = False,
        **inference_kwargs,
    ):
        """Initializes the hook.

        Args:
            inference_fn: ``(task, y_history, design_history, **kwargs) -> Distribution``.
            eval_fn: Optional ``(state, prior) -> dict | None``, run once per
                step on the design just proposed, under the measure it was
                chosen against. Collected as ``extra["step_evals"]``. See
                :meth:`after_propose` for why it runs there.
            store_posterior_history: Keep each posterior on the rollout as
                ``extra["posterior_history"]``. Off by default: a particle set
                is the largest thing a rollout can carry, and this hook infers
                one every step.
            **inference_kwargs: Forwarded to ``inference_fn``.
        """
        super().__init__()
        self.inference_fn = inference_fn
        self.eval_fn = eval_fn
        self.store_posterior_history = store_posterior_history
        self.inference_kwargs = inference_kwargs

    def after_propose(self, state: RolloutState) -> None:
        """Measures/ evalautes the design just proposed, before its outcome is drawn.

        Each call covers the one-step segment ``[t, t+1)``, measured under
        everything strictly before it: the posterior this hook installed last
        step, or the original prior at ``t = 0``. The segments therefore tile
        ``[0, T)`` on their own, and their sum is the whole chain rule with no
        separate opening term.

        Here rather than in ``after_step`` because this is the first moment the
        design exists alongside the measure it was chosen under. That matters
        for a designer that optimizes each design in turn, such as
        :class:`~testbed.designers.sequential.SequentialDesigner`: by
        ``after_step`` the next design has not been chosen, and re-deriving it
        would run the optimization again and land on a different design from the
        one the trajectory actually used. A designer whose designs come from a
        policy has no such problem, which is why
        :class:`RefinementHook` can evaluate whole segments from ``after_step``.

        The histories are also exactly right here: they hold the steps strictly
        before this one, which is the context the estimator conditions on. By
        ``after_step`` they include step ``t``, the very step being measured.
        """
        if self.eval_fn is None:
            return
        evaluation = self.eval_fn(state, state.task.current_prior)
        if evaluation is not None:
            state.scratch.setdefault("step_evals", []).append(
                {"step": state.t + 1, **evaluation}
            )

    def after_step(self, state: RolloutState) -> None:
        """Infers the posterior from the history so far and updates the task."""
        posterior = self.inference_fn(
            state.task, state.y_history, state.design_history, **self.inference_kwargs
        )
        state.task.update_prior(posterior)
        record = ess_record(posterior, state.t + 1)
        if record is not None:
            state.scratch.setdefault("ess_history", []).append(record)
        if self.store_posterior_history:
            # Keyed by how many steps it conditions on, not appended to a list:
            # a consumer scoring a subset of steps has to be able to ask for the
            # posterior at a given step rather than trust a position.
            state.scratch.setdefault("posterior_history", {})[state.t + 1] = posterior

    def on_rollout_end(self, state: RolloutState) -> dict[str, Any] | None:
        """Attaches the final (and optionally per-step) posterior to the rollout."""
        extra = {"posterior": state.task.current_prior}
        if "ess_history" in state.scratch:
            extra["ess_history"] = state.scratch["ess_history"]
        if "step_evals" in state.scratch:
            extra["step_evals"] = state.scratch["step_evals"]
        if self.store_posterior_history:
            extra["posterior_history"] = state.scratch["posterior_history"]
        return extra


class RefinementHook(RolloutHook):
    """Infers the posterior at scheduled steps and refines the designer on what remains.

    Step-DAD: at each scheduled step it runs inference on the
    history so far, installs the result as ``task.current_prior``, and hands it to
    ``refine_fn``. What refinement means is left to ``refine_fn`` (see
    :func:`testbed.designers.semi_amortized.make_refine_fn`), so the same hook
    serves an amortized policy or a static design.

    Installs the posterior itself, so it subsumes :class:`PosteriorUpdateHook` at
    the steps it fires on; do not compose both.
    """

    def __init__(
        self,
        refine_at,
        refine_fn,
        eval_fn=None,
        inference_fn=run_importance_sampling,
        *,
        store_posterior_history: bool = False,
        **inference_kwargs,
    ):
        """Initializes the hook.

        Args:
            refine_at: Step counts to refine after, 1-indexed, so ``[6]`` means
                after six steps have been observed. A list rather than a period:
                the single-intervention case is the common one.
            refine_fn: ``(state, posterior) -> dict | None`` of metrics.
            eval_fn: Optional ``(state, posterior) -> dict | None``, run right
                after the refinement. Evaluating the refined designer has to
                happen here rather than once the rollout is over: refinements
                accumulate and the stretch this refinement enabled would be measured
                using a designer that had already seen data from inside it.
            inference_fn: ``(task, y_history, design_history, **kwargs) -> Distribution``.
            store_posterior_history: Keep each posterior on the rollout as
                ``extra["posterior_history"]``, for a consumer that needs the
                particles themselves rather than a summary of them. Off by
                default, as on :class:`PosteriorUpdateHook`: a particle set is
                the largest thing a rollout can carry, and the rollouts of a run
                are typically all held at once.
            **inference_kwargs: Forwarded to ``inference_fn``.
        """
        super().__init__()
        refine_at = sorted({int(step) for step in refine_at})
        if refine_at and refine_at[0] < 1:
            raise ValueError(
                f"refine_at holds step counts (1-indexed); got {refine_at}. "
                "There is no history to infer from before the first step."
            )
        self.refine_at = refine_at
        self.refine_fn = refine_fn
        self.eval_fn = eval_fn
        self.inference_fn = inference_fn
        self.store_posterior_history = store_posterior_history
        self.inference_kwargs = inference_kwargs

    def after_step(self, state: RolloutState) -> None:
        """Refines if this step is scheduled, skipping the last step.

        ``eval_fn`` runs here, not in ``after_propose`` where
        :class:`PosteriorUpdateHook` runs its own. 

        It has to run after ``refine_fn``, since it measures the refined policy.

        And it does not need a design to have been proposed. The policy designs
        the whole stretch itself when the eval simulates it forward.

        This leaves the stretch before the first refinement uncovered, since the
        hook only fires at scheduled steps. The caller measures that one itself,
        under the original prior. Doing it there rather than here is better on
        two counts:

        - Cost. It is the same for every trajectory, because nothing has been
          observed and the policy has not been refined, so it is measured once
          instead of ``n_trajectories`` times.
        - Error bars. Adding the same number to every trajectory's total leaves
          the spread across trajectories measuring only real variation between
          trajectories. Re-estimating it each time would mix its own noise into
          that spread. Its error is added to the total separately.
        """
        n_observed = state.t + 1
        # Nothing to refine for at the end: the horizon it would optimize is empty.
        if n_observed not in self.refine_at or n_observed >= state.T:
            return

        posterior = self.inference_fn(
            state.task, state.y_history, state.design_history, **self.inference_kwargs
        )
        state.task.update_prior(posterior)

        record = ess_record(posterior, n_observed)
        if record is not None:
            state.scratch.setdefault("refinement_ess", []).append(record)
        if self.store_posterior_history:
            # Keyed by the step count it conditions on rather than appended to a
            # list, so a consumer can ask for the posterior at a given
            # refinement instead of trusting a position. Nothing in the
            # refinement itself needs this: eval_fn reads the posterior from
            # task.current_prior at the moment it runs. It is here for what
            # happens after the rollout, such as plotting the posteriors.
            state.scratch.setdefault("posterior_history", {})[n_observed] = posterior

        #refine the designnet conditoned on what has been observed thus far
        metrics = self.refine_fn(state, posterior)
        if metrics is not None:
            state.scratch.setdefault("refinement_metrics", []).append(
                {"step": n_observed, **metrics}
            )

        if self.eval_fn is not None:
            evaluation = self.eval_fn(state, posterior)
            if evaluation is not None:
                state.scratch.setdefault("refinement_evals", []).append(
                    {"step": n_observed, **evaluation}
                )

    def on_rollout_end(self, state: RolloutState) -> dict[str, Any] | None:
        """Attaches the final posterior and whatever the refinements recorded.

        ``posterior_history`` is among them only under ``store_posterior_history``.
        """
        extra = {"posterior": state.task.current_prior}
        for key in (
            "refinement_ess",
            "refinement_metrics",
            "refinement_evals",
            "posterior_history",
        ):
            if key in state.scratch:
                extra[key] = state.scratch[key]
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
