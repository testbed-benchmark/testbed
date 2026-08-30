"""ActionBED on the CES task, with a whitened theta loss and optional design normalization.

A CES-specific variant of :mod:`testbed.methods.actionbed`. The training loop
is reused unchanged; only two things differ, both motivated by the diagnosis in
``notes/actionbed_ces_diagnosis.md``:

1. **The loss is taken on a whitened theta.** Plain MSE in theta-space is
   >99.99% the prior variance of ``u ~ LogNormal(1, 3^2)``, so rho and alpha
   get no usable gradient; and because ``CESThetaSpace`` exponentiates ``u``,
   squared error on ``exp(raw)`` has a one-sided gradient that drives the ``u``
   head to ~1e-20 and kills it. :class:`CESWhitenedMSELoss` instead compares
   ``(log rho, alpha, (log u - u_loc) / u_scale)``, where every component is
   O(1). The action net's raw outputs are read directly as ``log rho`` and
   standardized ``log u``, so the exp/log round trip never happens.

   This requires ``CESThetaSpace`` to be *switched off* for this method, so
   that :class:`~testbed.core.hooks.ActionHook` leaves the action net's outputs
   untransformed. The class is untouched; the recipe simply points
   ``task.action_space`` at an unconstrained
   :class:`~testbed.core.design_spaces.DefaultDesignSpace` instead, and
   :func:`check_action_space_is_untransformed` fails loudly if it doesn't.

2. **Design histories are optionally normalized.** CES designs are baskets in
   ``[0, 100]^6``, and neither ``AttentivePolicyNet`` nor ``MLPPolicyNet``
   normalizes its inputs, so the O(1) outcome channel is swamped. The
   ``normalize_designs`` config flag wraps both nets in
   :class:`NormalizedHistoryNet`. It exists as a flag so the two arms can be
   compared directly; ``false`` reproduces the current behaviour.

Run with ``python testbed/methods/actionbed_ces.py [overrides...]``.

Note:
    ``final_eval`` / ``scripts/eval_run.py`` is known to be broken for
    ActionBED checkpoints (it rebuilds the policy with ``use_raw_history=True``
    because deterministic nets report ``produces_policy_log_probs = False``,
    while training rolls out on physical history). That is a separate issue and
    is not worked around here. Checkpoints written by this script have the same
    format as ``actionbed.py``'s, in both arms of the ablation --
    :class:`NormalizedHistoryNet` delegates ``state_dict`` to the net it wraps,
    so the normalization wrapper leaves no trace in the saved weights.
"""

import hydra
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from testbed.core.setup import setup
from testbed.design_nets.base import DesignNet, DesignProposal
from testbed.designers import PolicyDesigner, RandomDesigner
from testbed.methods.actionbed import Trainer, build_loss_fn, build_policy_nets

# Fallbacks for the CES prior on log u, used only if the task's prior does not
# expose them (see CESWhitenedMSELoss).
DEFAULT_U_LOC = 1.0
DEFAULT_U_SCALE = 3.0

THETA_NAMES = ("rho", "alpha_1", "alpha_2", "alpha_3", "u")


# ============================================================================
# Design-history normalization
# ============================================================================


class NormalizedHistoryNet(DesignNet):
    r"""Affinely rescales a physical design history before delegating to ``net``.

    Maps ``design -> (design - bias) / scale`` with ``bias = (high + low) / 2``
    and ``scale = (high - low) / 2``, i.e. the task's design box onto
    ``[-1, 1]``. Only the design history is touched; the outcome history is
    already O(1) for CES and is passed through.

    This is the *raw* design, squashed. ``BoxDesignSpace`` parametrizes designs
    as ``design = bias + scale * tanh(raw)``, so

    .. math::
        (\mathrm{design} - \mathrm{bias}) / \mathrm{scale} = \tanh(\mathrm{raw})

    exactly. Conditioning on this is therefore the same idea as conditioning on
    the policy's raw pre-squash outputs (what ``use_raw_history`` does, and what
    the CES DAD recipe relies on), with two practical advantages. It is bounded
    in ``[-1, 1]`` by construction, whereas ``raw`` is not -- ``tanh``
    saturates, so a policy pushing designs toward the corners of the box can
    let ``raw`` drift without limit over a long run. And it is computable from
    the physical history alone, so it is available during the action-warmup
    phase: ``RandomDesigner`` supplies no raw designs at all
    (``_step_design`` returns ``raw=None``, so ``Designer.rollout`` leaves
    ``state.raw_design_history`` as ``None``), and recovering them would mean
    inverting the squash with ``atanh``, which diverges at the box bounds that
    uniform sampling reaches.

    The wrapper is transparent in two respects that matter downstream: it
    delegates the :class:`~testbed.design_nets.base.DesignNet` capability flags
    to the wrapped net, so a :class:`~testbed.designers.PolicyDesigner` sees the
    same policy it otherwise would; and it delegates ``state_dict`` /
    ``load_state_dict``, so checkpoints match an unwrapped run's and can be
    loaded straight into a bare net.

    Args:
        net (DesignNet):
            The network to wrap. Its ``forward``/``propose`` are called with
            the rescaled design history.
        low (Tensor or float):
            Lower bound of the design box, broadcast against the design shape.
        high (Tensor or float):
            Upper bound of the design box.
    """

    def __init__(self, net: DesignNet, low, high) -> None:
        """Initializes the wrapper and precomputes the affine constants."""
        super().__init__()
        self.net = net

        low = torch.as_tensor(low, dtype=torch.get_default_dtype())
        high = torch.as_tensor(high, dtype=torch.get_default_dtype())
        scale = (high - low) / 2.0
        if not bool(torch.all(torch.isfinite(scale) & (scale > 0)).item()):
            raise ValueError(
                "NormalizedHistoryNet needs a finite, non-degenerate design box; "
                f"got low={low}, high={high}."
            )

        # Buffers rather than plain tensors, so .to(device) carries them.
        self.register_buffer("bias", (high + low) / 2.0)
        self.register_buffer("scale", scale)

        # Truthfully declare that physical histories are normalized here. This
        # is what makes PolicyDesigner reject use_raw_history=True, which would
        # otherwise double-transform the history.
        self.normalize_design_history = True

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"NormalizedHistoryNet({self.net!r})"

    @property
    def produces_policy_log_probs(self) -> bool:
        """Delegates to the wrapped net."""
        return self.net.produces_policy_log_probs

    @property
    def supports_reparameterized_proposals(self) -> bool:
        """Delegates to the wrapped net."""
        return self.net.supports_reparameterized_proposals

    @property
    def raw_designs_are_physical(self) -> bool:
        """Delegates to the wrapped net."""
        return self.net.raw_designs_are_physical

    def normalize(self, design_history: Tensor) -> Tensor:
        """Rescales a physical design history onto ``[-1, 1]``."""
        bias = self.bias.to(device=design_history.device, dtype=design_history.dtype)
        scale = self.scale.to(device=design_history.device, dtype=design_history.dtype)
        return (design_history - bias) / scale

    def forward(self, y_history, design_history, mask=None) -> Tensor:
        """Delegates ``forward`` with a normalized design history."""
        return self.net(y_history, self.normalize(design_history), mask=mask)

    def propose(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        reparameterize_policy: bool = False,
    ) -> DesignProposal:
        """Delegates ``propose`` with a normalized design history."""
        return self.net.propose(
            y_history,
            self.normalize(design_history),
            mask=mask,
            reparameterize_policy=reparameterize_policy,
        )

    def state_dict(self, *args, **kwargs):
        """Returns the wrapped net's state dict, so the wrapper leaves no trace."""
        return self.net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Loads into the wrapped net, matching :meth:`state_dict`."""
        return self.net.load_state_dict(state_dict, *args, **kwargs)


def maybe_normalize(net: DesignNet, task, enabled: bool) -> DesignNet:
    """Wraps ``net`` in a :class:`NormalizedHistoryNet` when ``enabled``.

    The bounds come from the task's design space, which must be a box (CES uses
    ``[0, 100]^6``).

    Args:
        net (DesignNet):
            Network whose design-history input should be normalized.
        task:
            Task supplying the design box.
        enabled (bool):
            Whether to normalize at all. When False, ``net`` is returned as-is.

    Returns:
        DesignNet: Either ``net`` or a wrapper around it.

    Raises:
        TypeError: If the task's design space has no ``low``/``high`` bounds.
    """
    if not enabled:
        return net

    design_space = task.design_space
    if not (hasattr(design_space, "low") and hasattr(design_space, "high")):
        raise TypeError(
            "normalize_designs=true needs a task whose design_space exposes "
            f"low/high bounds (e.g. BoxDesignSpace); got {design_space!r}."
        )
    return NormalizedHistoryNet(net, design_space.low, design_space.high)


# ============================================================================
# Loss
# ============================================================================


class CESWhitenedMSELoss(nn.Module):
    r"""MSE on a whitened CES theta, where all five components are O(1).

    The target is

    .. math::
        (\log \rho,\; \alpha,\; (\log u - \mu_u) / \sigma_u)

    and the estimate is read off the action net's raw output ``r`` as
    ``(r_0, \mathrm{softmax}(r_{1:4}), r_4)``. So channels 0 and 4 *are*
    ``log rho`` and standardized ``log u``; nothing is exponentiated on the way
    into the loss, which is what removes the one-sided gradient that kills the
    ``u`` head under a theta-space MSE.

    ``alpha`` keeps its softmax. It needs no rescaling (bounded in ``[0, 1]``,
    prior variance ~0.055 per dim), and unlike ``exp`` the softmax is a bounded
    map, so its only failure mode is saturation near the simplex vertices --
    where the loss contribution is near zero anyway. It is not strictly
    necessary (under MSE the optimum is the posterior mean, and the simplex is
    convex, so an unconstrained head lands in it regardless) but it is
    effectively free and keeps emitted actions valid.

    Args:
        task (optional):
            The CES task. Supplied automatically by
            :func:`~testbed.methods.actionbed.build_loss_fn`. Used only to read
            the prior's ``u`` location/scale; falls back to the standard prior's
            ``(1.0, 3.0)`` if the prior does not expose them.
        reduction (str):
            One of ``"sum"``, ``"mean"``, or ``"none"`` (default: ``"mean"``).
        u_loc (float, optional):
            Overrides the location used to standardize ``log u``.
        u_scale (float, optional):
            Overrides the scale used to standardize ``log u``.
    """

    def __init__(
        self,
        task=None,
        reduction: str = "mean",
        u_loc: float | None = None,
        u_scale: float | None = None,
    ) -> None:
        """Initializes the loss, reading the log-u standardization off the prior."""
        super().__init__()
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

        prior = getattr(task, "prior", None) if task is not None else None
        if u_loc is None:
            u_loc = float(getattr(prior, "u_loc", DEFAULT_U_LOC))
        if u_scale is None:
            u_scale = float(getattr(prior, "u_scale", DEFAULT_U_SCALE))
        if u_scale <= 0:
            raise ValueError(f"u_scale must be positive; got {u_scale}.")

        self.u_loc = u_loc
        self.u_scale = u_scale

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"CESWhitenedMSELoss(u_loc={self.u_loc}, u_scale={self.u_scale})"

    def whiten_target(self, theta: Tensor) -> Tensor:
        """Maps a physical theta to the whitened space the loss lives in.

        Args:
            theta (Tensor):
                Physical parameters ``(rho, alpha, u)`` of shape ``[..., 5]``.

        Returns:
            Tensor: Whitened parameters, shape ``[..., 5]``.
        """
        rho, alpha, u = theta[..., 0:1], theta[..., 1:4], theta[..., 4:5]
        return torch.cat(
            [rho.log(), alpha, (u.log() - self.u_loc) / self.u_scale], dim=-1
        )

    def whiten_estimate(self, action: Tensor) -> Tensor:
        """Maps the action net's raw output into the same whitened space.

        Args:
            action (Tensor):
                Untransformed action-net outputs of shape ``[..., 5]``.

        Returns:
            Tensor: Whitened estimate, shape ``[..., 5]``.
        """
        log_rho, alpha_logits, log_u = (
            action[..., 0:1],
            action[..., 1:4],
            action[..., 4:5],
        )
        return torch.cat([log_rho, torch.softmax(alpha_logits, dim=-1), log_u], dim=-1)

    def to_physical(self, action: Tensor) -> Tensor:
        """Maps a raw action to a physical theta, for reporting.

        Inverts the whitening, so the result is directly comparable to a
        prior draw. Not used in training.

        Args:
            action (Tensor):
                Untransformed action-net outputs of shape ``[..., 5]``.

        Returns:
            Tensor: Physical ``(rho, alpha, u)``, shape ``[..., 5]``.
        """
        whitened = self.whiten_estimate(action)
        log_rho, alpha, std_log_u = (
            whitened[..., 0:1],
            whitened[..., 1:4],
            whitened[..., 4:5],
        )
        u = (self.u_loc + self.u_scale * std_log_u).exp()
        return torch.cat([log_rho.exp(), alpha, u], dim=-1)

    def per_dim_error(self, action: Tensor, theta: Tensor) -> Tensor:
        """Returns the mean squared error per theta component, in whitened space.

        Args:
            action (Tensor):
                Untransformed action-net outputs of shape ``[batch, 5]``.
            theta (Tensor):
                Physical parameters of shape ``[batch, 5]``.

        Returns:
            Tensor: Per-component error, shape ``[5]``.
        """
        squared = (self.whiten_estimate(action) - self.whiten_target(theta)) ** 2
        return squared.reshape(-1, squared.shape[-1]).mean(dim=0)

    def forward(self, action: Tensor, theta: Tensor) -> Tensor:
        """Computes the whitened MSE.

        Args:
            action (Tensor):
                Untransformed action-net outputs of shape ``[..., 5]``.
            theta (Tensor):
                Physical parameters, the same shape as ``action``.

        Returns:
            Tensor: A scalar, or the same shape as ``action`` if
            ``reduction="none"``.

        Raises:
            ValueError: If ``action`` and ``theta`` have different shapes.
        """
        if action.shape != theta.shape:
            raise ValueError(
                f"action and theta must have the same shape, got "
                f"{tuple(action.shape)} and {tuple(theta.shape)}."
            )
        return F.mse_loss(
            self.whiten_estimate(action),
            self.whiten_target(theta),
            reduction=self.reduction,
        )


# ============================================================================
# Trainer
# ============================================================================


class CESTrainer(Trainer):
    """ActionBED trainer that also logs per-component errors at eval time.

    Identical to :class:`~testbed.methods.actionbed.Trainer` in what it
    optimizes. The scalar loss aggregates five parameters whose learnability
    differs a lot on CES, so the per-component breakdown is what actually shows
    whether a long run is making progress; each is directly comparable to the
    prior variance of that component.
    """

    def _per_dim_metrics(self, prefix: str, action: Tensor, theta: Tensor) -> dict:
        """Returns per-component whitened errors, keyed for the metric logger.

        Duck-typed rather than an ``isinstance`` check against
        :class:`CESWhitenedMSELoss`: running this file as a script imports it as
        ``__main__``, so the class defined here and the one Hydra instantiates
        from ``testbed.methods.actionbed_ces`` are different objects and
        ``isinstance`` would silently drop every metric.
        """
        per_dim_error = getattr(self.loss_fn, "per_dim_error", None)
        if per_dim_error is None:
            return {}
        errors = per_dim_error(action, theta)
        return {
            f"{prefix}/mse_{name}": value
            for name, value in zip(THETA_NAMES, errors.tolist())
        }

    def _evaluate_with(self, designer, prefix: str) -> dict:
        """Evaluates ``designer`` on a fresh batch, with per-component errors."""
        cfg = self.cfg

        policy_was_training = self.designer.design_net.training
        action_was_training = self.action_net.training
        self.designer.design_net.eval()
        self.action_net.eval()

        with torch.no_grad():
            theta = self.task.sample_prior(cfg.n_outer_eval)
            rollout = designer.rollout(theta)
            action = rollout.extra["action"]
            loss = self.loss_fn(action, theta)
            metrics = {f"{prefix}/loss": loss.item()}
            metrics.update(self._per_dim_metrics(prefix, action, theta))

        self.designer.design_net.train(policy_was_training)
        self.action_net.train(action_was_training)

        return metrics

    def evaluate_warmup(self) -> dict:
        """Evaluates warmup with a fresh batch of rollouts."""
        return self._evaluate_with(self.random_designer, "action_warmup/eval")

    def evaluate(self) -> dict:
        """Evaluates training with a fresh batch of rollouts."""
        return self._evaluate_with(self.designer, "eval")


# ============================================================================
# Helpers
# ============================================================================


def check_action_space_is_untransformed(task) -> None:
    """Raises unless the task's action space leaves action-net outputs alone.

    :class:`CESWhitenedMSELoss` reads the action net's raw outputs as
    ``log rho`` and standardized ``log u``, so a reparametrizing action space
    (notably ``CESThetaSpace``, the CES default) would exponentiate them before
    the loss ever saw them, silently reintroducing the pathology this method
    exists to avoid.

    Args:
        task: The task whose ``action_space`` is checked.

    Raises:
        ValueError: If the action space transforms its input.
    """
    action_space = task.action_space
    probe = torch.randn(8, *action_space.event_shape)
    try:
        transformed = action_space.to_design(probe)
    except NotImplementedError:
        # No reparametrization implemented at all, so nothing is applied.
        return

    if not torch.allclose(transformed, probe):
        raise ValueError(
            f"{type(action_space).__name__} reparametrizes the action net's "
            "outputs, but CESWhitenedMSELoss expects them raw. Point "
            "task.action_space at an unconstrained DefaultDesignSpace in the "
            "recipe (see experiments/constant_elasticity_of_substitution/"
            "actionbed_ces.yaml); CESThetaSpace itself needs no changes."
        )


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(
    version_base="1.3", config_path="../recipes", config_name="method/actionbed_ces"
)
def main(cfg):
    """Entry point for ActionBED training on CES.

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/actionbed_ces.py [overrides...]``.
    """
    manager, task = setup(cfg)

    check_action_space_is_untransformed(task)

    # Set seed
    torch.manual_seed(cfg.seed)

    design_net, action_net = build_policy_nets(cfg, task)
    design_net = maybe_normalize(design_net, task, cfg.normalize_designs)
    action_net = maybe_normalize(action_net, task, cfg.normalize_designs)
    design_net = design_net.to(cfg.device)
    action_net = action_net.to(cfg.device)

    if manager:
        manager.event_logger.info(
            f"Design-history normalization: {bool(cfg.normalize_designs)}"
        )

    designer = PolicyDesigner(task, cfg.T, design_net)

    # Setup for action-warmup; designs are drawn from the task's design space.
    random_designer = None
    if cfg.n_action_warmup_iter > 0:
        random_designer = RandomDesigner(task, cfg.T)

    # Create optimizer + LR scheduler
    all_policy_params = list(design_net.parameters()) + list(action_net.parameters())
    optimizer = torch.optim.Adam(all_policy_params, lr=cfg.lr, betas=cfg.betas)

    scheduler = None
    if cfg.lr_period is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_period, gamma=cfg.lr_gamma
        )

    loss_fn = build_loss_fn(cfg, task)

    trainer = CESTrainer(
        task=task,
        designer=designer,
        action_net=action_net,
        loss_fn=loss_fn,
        optimizer=optimizer,
        cfg=cfg,
        scheduler=scheduler,
        random_designer=random_designer,
        manager=manager,
    )

    trainer.train()


if __name__ == "__main__":
    main()
