"""Abstract base classes for Designer objects."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from testbed.core.hooks import RolloutHook, apply_rollout_end, check_hooks
from testbed.core.rollout import Rollout, RolloutState


class Designer(ABC, nn.Module):
    r"""Base class for designers.

    Subclasses implement :meth:`_step_design`, which returns the design along
    with an optional raw (pre-transform) design and an optional log-prob.

    Args:
        task (Task):
            The :class:`Task` we are producing designs for.
        T (int):
            The number of steps in the experiment.
        hooks (iterable, optional):
            An optional iterable of ``RolloutHook`` objects used to modify the design process (default: ()).
        transform_designs (bool):
            If True, the designer will automatically reparametrise designs using the task's DesignSpace.
            Set to False if you would prefer to handle this manually (default: True).
    """

    def __init__(
        self,
        task,
        T,
        *,
        hooks: Sequence[RolloutHook] = (),
        transform_designs: bool = True,
    ):
        r"""Initializes the designer."""
        super().__init__()

        self.task = task
        self.T = T
        # A ModuleList, not a list, so hooks (and anything they own, e.g. an
        # ActionHook's net) are registered submodules and move with .to().
        # Appending to it later registers the new hook too.
        self.hooks = nn.ModuleList(check_hooks(hooks))
        self.transform_designs = bool(transform_designs)

    @property
    def device(self) -> torch.device:
        """Returns the device this designer produces designs on.

        Defaults to the task's device. Subclasses that own tensors of their own
        (a learnable design, a policy net) override this to report theirs, so
        that a designer moved independently of its task still answers correctly.
        """
        return self.task.device

    def _to_design(self, raw_design):
        r"""Maps a raw value into a valid design via task.design_space.to_design.

        Identity when self.transform_designs is False.
        """
        if not self.transform_designs:
            return raw_design
        return self.task.design_space.to_design(raw_design)

    def _to_raw(self, design):
        r"""Inverse of ``_to_design``, via task.design_space.to_raw.

        Identity when self.transform_designs is False.
        """
        if not self.transform_designs:
            return design
        return self.task.design_space.to_raw(design)

    @abstractmethod
    def _step_design(self, state: RolloutState):
        r"""Produces the next design from the current rollout state.

        The single extension point for a Designer. It receives the whole
        ``RolloutState`` rather than a fixed argument list, so a designer that
        comes to need something particular about the rollout can just read it,
        instead of every designer's signature having to grow.

        Keep this a thin adapter. Put the actual logic in a method of your own
        that takes explicit, semantic inputs and documents what it needs, then
        have this pull those pieces off the state. Otherwise "what does this
        designer read?" becomes unanswerable without reading the body. The
        existing designers each do exactly this.

        Args:
            state (RolloutState):
                Current rollout state. Its histories hold the steps completed
                strictly before ``state.t``, so at ``t=0`` they are empty.

        Returns:
            tuple[Tensor, Tensor | None, Tensor | None]: The physical design
            with shape ``[b, *event_shape_design]``, its optional raw
            pre-transform value, and its optional policy log probability with
            shape ``[b]``.
        """
        ...

    def rollout(self, theta: Tensor) -> Rollout:
        r"""Runs a rollout for batched theta values.

        Args:
            theta (Tensor):
                Parameter values in the likelihood used to produce the rollout. Shape ``[b, *event_shape_theta]``.

        Runs ``self.hooks`` (fixed at construction) alongside the rollout.

        Returns:
            Rollout:
                A :class:`testbed.core.rollout.Rollout`. ``extra`` carries
                ``raw_design_history`` whenever the designer produced a raw
                design at every step, and ``policy_log_probs`` whenever its
                proposal exposed a log-prob at every step; each key is simply
                absent otherwise.
        """
        hooks = self.hooks
        b = theta.shape[0]
        T = self.T

        # Initialize a mutable state; hooks will modify this in-place
        y_history, design_history = self.task.null_history(b=b)
        state = RolloutState(
            task=self.task,
            theta=theta,
            T=T,
            y_history=y_history,
            design_history=design_history,
        )

        y_list = []
        design_list = []
        raw_design_list = []
        policy_log_prob_list = []

        for t in range(T):
            state.t = t
            # Clear the previous step's per-step fields, so that anything reading
            # them before they are set this step sees None rather than step t-1's
            # values.
            state.design = state.raw_design = state.policy_log_prob = state.y = None

            for hook in hooks:
                hook.before_propose(state)

            # Produce the next design, possibly with raw/logprobs. The designer
            # reads whatever it needs off the state, including which of the
            # constrained/raw histories to condition on.
            design, raw_design, policy_log_prob = self._step_design(state)
            state.design = design
            state.raw_design = raw_design
            state.policy_log_prob = policy_log_prob

            for hook in hooks:
                hook.after_propose(state)

            # Sample observation; always using the (possibly transformed) design.
            y = state.task.sample_outcome(
                state.theta,
                state.design,
                y_history=state.y_history,
                design_history=state.design_history,
            )
            state.y = y  # [b, *event_shape_y]

            # Update state history using data that actually produced the observation.
            # A hook editing state.*_history in place is ovewritten.
            y_list.append(y)
            design_list.append(state.design)
            raw_design_list.append(state.raw_design)
            policy_log_prob_list.append(state.policy_log_prob)
            state.y_history = torch.stack(y_list, dim=1)
            state.design_history = torch.stack(design_list, dim=1)
            state.raw_design_history = (
                torch.stack(raw_design_list, dim=1)
                if all(rd is not None for rd in raw_design_list)
                else None
            )

            for hook in hooks:
                hook.after_step(state)

        policy_log_probs = None
        if policy_log_prob_list and all(
            log_prob is not None for log_prob in policy_log_prob_list
        ):
            policy_log_probs = torch.stack(policy_log_prob_list, dim=1)  # [b, T]

        extra = {}
        if state.raw_design_history is not None:
            extra["raw_design_history"] = state.raw_design_history
        if policy_log_probs is not None:
            extra["policy_log_probs"] = policy_log_probs

        rollout = Rollout(
            theta=state.theta,
            y_history=state.y_history,
            design_history=state.design_history,
            extra=extra,
        )

        apply_rollout_end(hooks, state, rollout)

        return rollout
