"""Abstract base classes for Designer objects."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from testbed.core.rollout import Rollout

from .hooks import RolloutHook, RolloutState


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
            An optional iterable of ``RolloutHook``s used to modify the design process (default: ()).
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
        self.hooks = list(hooks)
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
            Tensor:
                The design ``[b, *event_shape_design]``.
            Tensor | None:
                The raw (pre-transform) design, when the designer computes one.
            Tensor | None:
                The policy log-prob ``[b]`` when the designer is stochastic and calculates a log-prob.
        """
        ...

    def make_empty_history(self, b: int):
        r"""Returns an empty rollout (i.e., corresponding to t=0).

        Args:
            b (int):
                Batch size.

        Returns:
            Tensor:
                Empty y_history ``[b, 0, *event_shape_y]``.

            Tensor:
                Empty design_history ``[b, 0, *event_shape_design]``.
        """
        y_history = torch.empty(
            (b, 0, *self.task.event_shape_y), device=self.task.device
        )
        design_history = torch.empty(
            (b, 0, *self.task.event_shape_design), device=self.task.device
        )
        return y_history, design_history

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
        y_history, design_history = self.make_empty_history(b)
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

            # Sample observation; always using the (possibly transformed) design
            y = self.task.sample_outcome(theta, design)  # [b, *event_shape_y]
            state.y = y

            y_list.append(y)
            design_list.append(design)
            raw_design_list.append(raw_design)
            policy_log_prob_list.append(policy_log_prob)
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
        if all(log_prob is not None for log_prob in policy_log_prob_list):
            policy_log_probs = torch.stack(policy_log_prob_list, dim=1)  # [b, T]

        extra = {}
        if state.raw_design_history is not None:
            extra["raw_design_history"] = state.raw_design_history
        if policy_log_probs is not None:
            extra["policy_log_probs"] = policy_log_probs

        rollout = Rollout(
            theta=theta,
            y_history=state.y_history,
            design_history=state.design_history,
            extra=extra,
        )

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

        return rollout
