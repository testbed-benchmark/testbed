"""Amortized variational posteriors q(theta | y_history, design_history)."""

from abc import ABC, abstractmethod
from math import atanh, log, prod

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
    Normal,
)

from testbed.core.util import sort_by_first_coordinate
from testbed.design_nets.encoder import AttentiveEncoder, DeepSetEncoder, HistoryEncoder
from testbed.design_nets.modules import MLP


class VariationalPosteriorNet(nn.Module, ABC):
    """Maps an experimental history to an amortized variational posterior over theta.

    Shape:
        ``y_history``: ``[b, T, *event_shape_y]``
        ``design_history``: ``[b, T, *event_shape_design]``
        ``mask``: ``[b, T]`` or None
        The returned distribution has ``batch_shape=[b]``, ``event_shape=[dim_theta]``.
    """

    def __init__(self, event_shape_theta):
        """Initializes the posterior net."""
        super().__init__()

        self.event_shape_theta = torch.Size(event_shape_theta)
        self.dim_theta = int(prod(self.event_shape_theta))

    @property
    def device(self) -> torch.device:
        """Returns the network device."""
        return next(self.parameters()).device

    @abstractmethod
    def distribution(self, y_history, design_history, mask=None) -> Distribution:
        """Returns the variational posterior for a history.

        Returns:
            Distribution: batch_shape ``[b]``, event_shape ``[dim_theta]``.
        """
        ...

    def log_prob(self, theta: Tensor, y_history, design_history, mask=None) -> Tensor:
        """Evaluates the variational posterior's log density at theta.

        Args:
            theta: Parameters, shape ``[b, *event_shape_theta]``.
            y_history: Outcomes so far, shape ``[b, T, *event_shape_y]``.
            design_history: Designs so far, shape ``[b, T, *event_shape_design]``.
            mask: Optional boolean mask for padded histories, shape ``[b, T]``.

        Returns:
            Tensor: Log density, shape ``[b]``.
        """
        trailing_shape = torch.Size(
            theta.shape[len(theta.shape) - len(self.event_shape_theta) :]
        )
        if trailing_shape != self.event_shape_theta:
            raise ValueError(
                f"theta must have trailing shape {tuple(self.event_shape_theta)}; "
                f"got {tuple(theta.shape)}."
            )
        b = theta.shape[0]
        theta_flat = theta.reshape(b, self.dim_theta)
        return self.distribution(y_history, design_history, mask=mask).log_prob(
            theta_flat
        )

    def sample(
        self, y_history, design_history, n_samples: int = 1, mask=None
    ) -> Tensor:
        """Draws samples from the variational posterior.

        Args:
            y_history: Outcomes so far, shape ``[b, T, *event_shape_y]``.
            design_history: Designs so far, shape ``[b, T, *event_shape_design]``.
            n_samples: How many samples to draw per batch element.
            mask: Optional boolean mask for padded histories, shape ``[b, T]``.

        Returns:
            Tensor: Shape ``[n_samples, b, *event_shape_theta]``.
        """
        dist = self.distribution(y_history, design_history, mask=mask)
        samples = dist.sample((n_samples,))  # [n_samples, b, dim_theta]
        return samples.reshape(n_samples, *dist.batch_shape, *self.event_shape_theta)

    def forward(self, y_history, design_history, mask=None) -> Distribution:
        """Returns the variational posterior for a history."""
        return self.distribution(y_history, design_history, mask=mask)


class GaussianMixtureVariationalPosteriorNet(VariationalPosteriorNet):
    """A Gaussian-mixture variational posterior read off a history encoding.

    The mixture is parametrized in a standardized internal space (shifted/scaled
    by ``theta_loc``/``theta_scale``) and mapped back to raw theta space; this is
    a cheap default fix for theta with large scale or constrained support.

    Known limitation: `event_shape_theta` is flattened before being modeled
    (see `VariationalPosteriorNet.log_prob`/`sample`), with no structural awareness of any
    permutation symmetry within that shape. For tasks like multi-source
    LocationFinding, whose likelihood is invariant to permuting the
    n_sources axis of theta, the true posterior is multimodal via this
    label-switching symmetry -- this class itself has no inductive bias
    encouraging it to spread mixture components across those symmetric
    modes. If using the Barber-Agakov bound it remains valid, since it holds
    for any posterior, but it may be needlessly loose on such tasks if used as-is.

    Limitation is addressed at the caller: pass
    `canonicalize_theta=True` to `VarPost`/`VarPostReinforce`/`VariationalPosteriorNLL` to
    sort theta by first coordinate before it ever reaches this posterior,
    collapsing the label-switching symmetry to one canonical representative
    per equivalence class.
    A posterior used without that flag still has this limitation.

    Args:
        encoder: HistoryEncoder object.
        event_shape_theta: Shape of theta.
        n_components: Number of mixture components (default: 10).
        dim_h: Per-layer trunk hidden widths, e.g. [128, 16]. An int specifies
            a single hidden layer. Defaults to the encoder's dim_emb.
        activation: Activation class for the trunk, e.g. nn.ReLU or nn.GELU.
        log_scale_min: Lower bound for component log standard deviations.
        log_scale_max: Upper bound for component log standard deviations.
        init_scale: Initial component standard deviation (in standardized space).
        theta_loc: Optional standardization shift, shape ``[dim_theta]``.
            Defaults to zeros.
        theta_scale: Optional standardization scale, shape ``[dim_theta]``.
            Defaults to ones.
    """

    def __init__(
        self,
        encoder: HistoryEncoder,
        event_shape_theta,
        n_components: int = 10,
        dim_h=None,
        activation=nn.GELU,
        log_scale_min: float = -7.0,
        log_scale_max: float = 3.0,
        init_scale: float = 1.0,
        theta_loc: Tensor | None = None,
        theta_scale: Tensor | None = None,
    ):
        """Initializes."""
        super().__init__(event_shape_theta)

        self.encoder = encoder
        self.n_components = int(n_components)
        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)

        dim_h = dim_h or encoder.dim_emb
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.dim_h = hidden_widths[-1]
        self.trunk = MLP(
            [encoder.dim_emb] + hidden_widths, activation=activation, activate_last=True
        )

        K, D = self.n_components, self.dim_theta
        self.loc_head = nn.Linear(self.dim_h, K * D)
        self.log_scale_head = nn.Linear(self.dim_h, K * D)
        self.logit_head = nn.Linear(self.dim_h, K)

        init_log_scale = log(init_scale)
        if not self.log_scale_min < init_log_scale < self.log_scale_max:
            raise ValueError(
                "init_scale must have a logarithm strictly between "
                "log_scale_min and log_scale_max."
            )
        nn.init.zeros_(self.log_scale_head.weight)
        scaled_log_scale = (
            2.0
            * (
                (init_log_scale - self.log_scale_min)
                / (self.log_scale_max - self.log_scale_min)
            )
            - 1.0
        )
        # back into unconstrained space
        nn.init.constant_(self.log_scale_head.bias, atanh(scaled_log_scale))

        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)

        self.register_buffer(
            "theta_loc", theta_loc if theta_loc is not None else torch.zeros(D)
        )
        self.register_buffer(
            "theta_scale", theta_scale if theta_scale is not None else torch.ones(D)
        )

    def distribution(self, y_history, design_history, mask=None) -> Distribution:
        """Returns the Gaussian-mixture variational posterior for a history."""
        emb = self.encoder(y_history, design_history, mask=mask)  # [b, dim_emb]
        b = emb.shape[0]
        K, D = self.n_components, self.dim_theta

        h = self.trunk(emb)  # [b, dim_h]
        loc = self.loc_head(h).reshape(b, K, D)
        raw_log_scale = self.log_scale_head(h).reshape(b, K, D)
        log_scale = self.log_scale_min + 0.5 * (
            self.log_scale_max - self.log_scale_min
        ) * (torch.tanh(raw_log_scale) + 1)
        scale = log_scale.exp()

        loc = self.theta_loc + self.theta_scale * loc
        scale = self.theta_scale * scale

        logits = self.logit_head(h)  # [b, K]
        component = Independent(Normal(loc, scale), 1)
        return MixtureSameFamily(Categorical(logits=logits), component)


class VariationalPosteriorNLL(nn.Module):
    """Negative log-likelihood loss for pretraining/fitting a posterior net.

    Minimizing this loss over posterior parameters is equivalent to maximizing
    any reasonable lower bound on EIG that depends on the posterior only through
    ``log q(theta | y, d)`` (e.g. ``testbed.design_objectives.var_post.VarPost``):
    whatever posterior-independent term such a bound also contains (``-log
    p(theta)`` in the Barber-Agakov bound) has zero gradient w.r.t. posterior
    parameters.
    """

    def __init__(
        self,
        posterior_net: VariationalPosteriorNet,
        reduction: str = "mean",
        canonicalize_theta: bool = False,
    ):
        """Initializes.

        canonicalize_theta: sort the leading axis of ``event_shape_theta`` by
        first coordinate before computing the NLL (default: False).
        Must be set consistently with whatever variational bound
        (e.g. `VarPost`/`VarPostReinforce`) this posterior is paired with downstream,
        since the posterior should be pretrained to fit whatever theta it will
        actually be evaluated against.
        """
        super().__init__()
        # Deferred import: avoids a design_nets <-> design_objectives import
        # cycle
        from testbed.design_objectives.loss import _check_reduction

        self.posterior_net = posterior_net
        self.reduction = _check_reduction(reduction)
        self.canonicalize_theta = canonicalize_theta

    def forward(self, theta: Tensor, y_history, design_history, mask=None) -> Tensor:
        """Computes the posterior NLL.

        Args:
            theta: Parameters, shape ``[b, *event_shape_theta]``.
            y_history: Outcomes so far, shape ``[b, T, *event_shape_y]``.
            design_history: Designs so far, shape ``[b, T, *event_shape_design]``.
            mask: Optional boolean mask for padded histories, shape ``[b, T]``.

        Returns:
            Tensor: If ``reduction`` is ``"none"``, shape ``[b]``. Otherwise, a scalar.
        """
        if self.canonicalize_theta:
            theta = sort_by_first_coordinate(theta)
        nll = -self.posterior_net.log_prob(theta, y_history, design_history, mask=mask)
        if self.reduction == "sum":
            return nll.sum()
        if self.reduction == "mean":
            return nll.mean()
        return nll

    def from_rollout(self, rollout) -> Tensor:
        """Computes the posterior NLL from a rollout's theta/y_history/design_history."""
        return self.forward(rollout.theta, rollout.y_history, rollout.design_history)


class AttentiveVariationalPosteriorNet(GaussianMixtureVariationalPosteriorNet):
    """Preset: AttentiveEncoder (transformer + CLS pooling) + Gaussian-mixture readout."""

    def __init__(
        self,
        event_shape_y,
        event_shape_design,
        event_shape_theta,
        n_components: int = 10,
        encoder_dim_h=[64],
        dim_emb=64,
        num_heads=4,
        num_blocks=2,
        trunk_dim_h=None,
        trunk_activation=nn.GELU,
        log_scale_min: float = -7.0,
        log_scale_max: float = 3.0,
        init_scale: float = 1.0,
        theta_loc: Tensor | None = None,
        theta_scale: Tensor | None = None,
    ):
        """Initializes."""
        encoder = AttentiveEncoder(
            event_shape_y,
            event_shape_design,
            dim_h=encoder_dim_h,
            dim_emb=dim_emb,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )

        super().__init__(
            encoder,
            event_shape_theta,
            n_components=n_components,
            dim_h=trunk_dim_h,
            activation=trunk_activation,
            log_scale_min=log_scale_min,
            log_scale_max=log_scale_max,
            init_scale=init_scale,
            theta_loc=theta_loc,
            theta_scale=theta_scale,
        )


class DeepSetVariationalPosteriorNet(GaussianMixtureVariationalPosteriorNet):
    """Preset: DeepSetEncoder (per-step MLP + sum pooling) + Gaussian-mixture readout."""

    def __init__(
        self,
        event_shape_y,
        event_shape_design,
        event_shape_theta,
        n_components: int = 10,
        encoder_dim_h=[128, 128],
        dim_emb=64,
        trunk_dim_h=None,
        trunk_activation=nn.GELU,
        log_scale_min: float = -7.0,
        log_scale_max: float = 3.0,
        init_scale: float = 1.0,
        theta_loc: Tensor | None = None,
        theta_scale: Tensor | None = None,
    ):
        """Initializes."""
        encoder = DeepSetEncoder(
            event_shape_y,
            event_shape_design,
            dim_h=encoder_dim_h,
            dim_emb=dim_emb,
        )

        super().__init__(
            encoder,
            event_shape_theta,
            n_components=n_components,
            dim_h=trunk_dim_h,
            activation=trunk_activation,
            log_scale_min=log_scale_min,
            log_scale_max=log_scale_max,
            init_scale=init_scale,
            theta_loc=theta_loc,
            theta_scale=theta_scale,
        )
