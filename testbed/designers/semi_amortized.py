"""Semi-amortized designers: an amortized policy that can be refined mid-rollout.

The extra capability over :class:`PolicyDesigner` is continuing a rollout from a
history that has already been observed, which is what Step-DAD's deployment-time
refinement optimizes over (Hedman et al., ICML 2025, https://arxiv.org/abs/2507.14057).
"""

import torch

from ..core.rollout import Rollout, RolloutState
from .amortized import PolicyDesigner


class SemiAmortizedDesigner(PolicyDesigner):
    """PolicyDesigner that can also continue a rollout from a realized history.

    ``rollout`` is inherited unchanged, so ordinary rollouts run the shared
    template loop in :meth:`Designer.rollout`. :meth:`continue_from` owns its own
    explicit loop instead of parameterizing that one.

    Histories are indexed as ``S`` observed steps followed by the simulated ones,
    matching the context convention in :meth:`Task.history_logp`.
    """

    def continue_from(
        self,
        theta,
        y_history,
        design_history,
        n_steps,
        raw_design_history=None,
    ):
        r"""Rolls the policy out for ``n_steps`` more, starting from an observed history.

        The policy conditions on ``[history, steps so far]`` at every step, so the
        history is never reset. The returned rollout holds **only the new steps**:
        the history is conditioning, not data to be scored again, and keeping it
        out is what lets the caller pass it to an objective as
        ``y_context``/``design_context`` (see :meth:`remaining_horizon_gain`).

        Runs without hooks. A refinement hook whose refinement ran continuations
        through the hooked path would recurse into itself.

        Args:
            theta: Parameters of shape ``[b, *event_shape_theta]``.
            y_history: Observed outcomes, ``[b or 1, S, *event_shape_y]``.
            design_history: The designs alongside them, ``[b or 1, S, *event_shape_design]``.
            n_steps: How many further steps to simulate.
            raw_design_history: Pre-transform designs, ``[b or 1, S, *event_shape_design]``.
                Required when ``use_raw_history`` is set.

        Returns:
            Rollout: ``y_history``/``design_history`` of length ``n_steps``, with
            ``extra`` carrying only the new steps as well.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1; got {n_steps}.")

        if self.use_raw_history and raw_design_history is None:
            raise ValueError(
                "use_raw_history=True but no raw_design_history was supplied. The "
                "policy would condition on transformed designs it was never "
                "trained on. Pass rollout.extra['raw_design_history']."
            )

        b = theta.shape[0]
        y_list = list(self._history_steps(y_history, b))
        design_list = list(self._history_steps(design_history, b))
        S = len(y_list)
        if S != len(design_list):
            raise ValueError(
                f"History lengths must agree; got {S} outcomes and "
                f"{len(design_list)} designs."
            )

        raw_design_list = (
            list(self._history_steps(raw_design_history, b))
            if raw_design_history is not None
            else [None] * S
        )
        # Not a history channel: nothing conditions on policy log probs, and an
        # observed prefix has no known ones, so there is no prefix to seed.
        policy_log_probs = []

        state = RolloutState(
            task=self.task,
            theta=theta,
            T=S + n_steps,
            y_history=torch.stack(y_list, dim=1),
            design_history=torch.stack(design_list, dim=1),
            raw_design_history=(
                torch.stack(raw_design_list, dim=1)
                if raw_design_history is not None
                else None
            ),
        )

        for t in range(S, state.T):
            state.t = t
            state.design = state.raw_design = state.policy_log_prob = state.y = None

            design, raw_design, policy_log_prob = self._step_design(state)
            state.design = design
            state.raw_design = raw_design
            state.policy_log_prob = policy_log_prob

            y = self.task.sample_outcome(
                state.theta,
                state.design,
                y_history=state.y_history,
                design_history=state.design_history,
            )
            state.y = y

            y_list.append(y)
            design_list.append(state.design)
            raw_design_list.append(state.raw_design)
            policy_log_probs.append(state.policy_log_prob)

            state.y_history = torch.stack(y_list, dim=1)
            state.design_history = torch.stack(design_list, dim=1)
            state.raw_design_history = (
                torch.stack(raw_design_list, dim=1)
                if all(rd is not None for rd in raw_design_list)
                else None
            )

        extra = {}
        # Checked over the simulated steps only: whether the prefix carried raw
        # designs says nothing about whether the policy just produced any.
        if all(rd is not None for rd in raw_design_list[S:]):
            extra["raw_design_history"] = torch.stack(raw_design_list[S:], dim=1)
        if all(log_prob is not None for log_prob in policy_log_probs):
            extra["policy_log_probs"] = torch.stack(policy_log_probs, dim=1)

        return Rollout(
            theta=state.theta,
            y_history=state.y_history[:, S:],
            design_history=state.design_history[:, S:],
            extra=extra,
        )

    def remaining_horizon_gain(
        self,
        objective,
        y_history,
        design_history,
        n_steps,
        batch_size,
        raw_design_history=None,
    ):
        """Gain over the policy's remaining ``n_steps``, given an observed history.

        The history enters through three channels:
        theta come from ``task.current_prior`` (the posterior, once a refinement
        hook has installed it); the design net conditions on the history via
        :meth:`continue_from`; and the objective is handed the same history as
        context, so it conditions on it without scoring it.

        Args:
            objective: A ``RolloutObjective`` taking ``y_context``/``design_context``.
            y_history: Observed outcomes, ``[b or 1, S, *event_shape_y]``.
            design_history: The designs alongside them.
            n_steps: How many further steps the gain covers.
            batch_size: Number of continuations to average over.
            raw_design_history: Pre-transform designs, required when
                ``use_raw_history`` is set.

        Returns:
            Tensor: Scalar gain estimate.
        """
        theta = self.task.current_prior.sample(n_samples=batch_size)
        rollout = self.continue_from(
            theta,
            y_history,
            design_history,
            n_steps,
            raw_design_history=raw_design_history,
        )
        return objective.estimate_gain(
            rollout,
            y_context=y_history,
            design_context=design_history,
        )

    def _history_steps(self, history, b):
        """Yields an observed history's per-step slices, detached and batched to ``b``.

        Detached because these are data, not something the refinement optimizes,
        and expanded so a single observed trajectory can seed a whole batch of
        hypothetical continuations.
        """
        history = history.detach()
        if history.shape[0] == 1 and b != 1:
            history = history.expand(b, *history.shape[1:])
        elif history.shape[0] != b:
            raise ValueError(
                f"History batch dim must be 1 or {b}; got {history.shape[0]}."
            )
        return history.unbind(dim=1)
