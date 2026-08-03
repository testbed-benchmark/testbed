"""The rollout produced by a Designer."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from testbed.core.util import shape_as_str

from ..tasks.base import Task


@dataclass
class Rollout:
    """A batched sequential-design rollout.

    theta/y_history/design_history are exactly the three quantities every
    design objective needs; anything method-specific (an action, a posterior,
    a per-step action history, ...) is contributed by a `RolloutHook` and
    lives in `extra` rather than growing this dataclass or any objective's
    signature.
    """

    theta: Tensor  # [b, *event_shape_theta]
    y_history: Tensor  # [b, T, *event_shape_y]
    design_history: Tensor  # [b, T, *event_shape_design] -- always fully constrained
    extra: dict[str, Any] = field(default_factory=dict)
    # Common optional keys a Designer.rollout() may populate (only when actually
    # present -- never as a key holding None):
    #   extra["raw_design_history"]: Tensor [b, T, *event_shape_design], whenever
    #       the designer computed a raw design at every step. Absent for designers
    #       that produce no raw design at all (e.g. RandomDesigner).
    #   extra["policy_log_probs"]: Tensor [b, T], when the designer's per-step
    #       proposal exposes a log prob (i.e. a stochastic policy).

    @property
    def b(self) -> int:
        """Returns the batch size."""
        return self.design_history.shape[0]

    @property
    def T(self) -> int:
        """Returns the number of steps."""
        return self.design_history.shape[1]

    def __repr__(self) -> str:
        """Compact representation of the Rollout."""
        return (
            "Rollout("
            f"theta={shape_as_str(self.theta)}, "
            f"y_history={shape_as_str(self.y_history)}, "
            f"design_history={shape_as_str(self.design_history)}, "
            f"extra={list(self.extra)}"
            ")"
        )


@dataclass
class RolloutState:
    """Mutable state threaded through a single `Designer.rollout()` call.

    Read by rollout hooks, and by `Designer._step_design`, which receives this
    whole object rather than a fixed argument list: everything a designer needs
    to choose the next design lives here, so a designer that comes to depend on
    something new (a hook's bookkeeping under `scratch`, say) needs no signature
    change anywhere.

    `y_history`/`design_history` hold the steps completed strictly before the
    current one in the `before_propose`/`after_propose`/`before_sample_outcome`
    callbacks; by `after_step`, they have been extended to include step `t`.
    `scratch` is hook-owned bookkeeping, freshly created for every rollout
    call, so a single hook instance is safe to reuse across many rollouts.

    `theta` is optional so a caller can build a minimal state for a one-off
    `_step_design` call outside a rollout.
    """

    task: Task
    theta: Tensor | None
    T: int
    t: int = 0
    y_history: Tensor | None = None
    design_history: Tensor | None = None
    raw_design_history: Tensor | None = None
    # Optional boolean mask [b, T] for padded histories.
    mask: Tensor | None = None
    # data for current step
    design: Tensor | None = None
    raw_design: Tensor | None = None
    y: Tensor | None = None
    policy_log_prob: Tensor | None = None
    scratch: dict[str, Any] = field(default_factory=dict)
