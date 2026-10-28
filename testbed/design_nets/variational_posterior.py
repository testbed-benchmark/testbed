"""Amortized variational posteriors q(theta | y_history, design_history)."""

from abc import ABC, abstractmethod
from math import atanh, log, prod

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from testbed.core.distributions import Distribution, TorchDistribution
from testbed.core.util import sort_by_first_coordinate
from testbed.design_nets.encoder import AttentiveEncoder, DeepSetEncoder, HistoryEncoder
from testbed.design_nets.modules import MLP, BlockTransform


class VariationalPosteriorNet(nn.Module, ABC):
    """Maps an experimental history to an amortized variational posterior over theta.

    Subclasses must implement :meth:`log_prob`, the only operation the
    variational bounds use. :meth:`sample` is optional and raises
    NotImplementedError until provided; it serves diagnostics and posterior
    proposals rather than training. A family that materializes an explicit
    distribution object may expose it as an optional ``distribution`` method
    (see :class:`GaussianMixtureVariationalPosteriorNet`), but nothing requires
    one.

    Shape:
        ``y_history``: ``[*batch_shape, T, *event_shape_y]``
        ``design_history``: ``[*batch_shape, T, *event_shape_design]``
        ``mask``: ``[*batch_shape, T]`` or None
        ``theta``: ``[*batch_shape, *event_shape_theta]``

    Args:
        event_shape_theta: Shape of theta.
        canonicalize_theta: Sort theta's leading (object) axis by first
            coordinate inside :meth:`log_prob` and :meth:`sample`, collapsing
            the label-switching symmetry of an exchangeable axis such as
            LocationFinding's ``n_sources``. Requires a theta with an
            object axis, i.e. ``len(event_shape_theta) >= 2`` (default: False).
    """

    def __init__(self, event_shape_theta, canonicalize_theta: bool = False):
        """Initializes the posterior net."""
        super().__init__()

        self.event_shape_theta = torch.Size(event_shape_theta)
        self.dim_theta = int(prod(self.event_shape_theta))
        self.canonicalize_theta = bool(canonicalize_theta)
        if self.canonicalize_theta and len(self.event_shape_theta) < 2:
            raise ValueError(
                "canonicalize_theta requires an exchangeable object axis, i.e. "
                f"len(event_shape_theta) >= 2; got {tuple(self.event_shape_theta)}."
            )

    def _maybe_canonicalize(self, theta: Tensor) -> Tensor:
        """Sorts theta's object axis by first coordinate, if canonicalizing.

        Accepts any number of leading batch dims, which sort_by_first_coordinate
        does not, so they are flattened into one before sorting.
        """
        if not self.canonicalize_theta:
            return theta
        batch_shape = theta.shape[: theta.ndim - len(self.event_shape_theta)]
        flat = theta.reshape(-1, *self.event_shape_theta)
        return sort_by_first_coordinate(flat).reshape(
            *batch_shape, *self.event_shape_theta
        )

    @property
    def device(self) -> torch.device:
        """Returns the network device."""
        return next(self.parameters()).device

    @abstractmethod
    def log_prob(self, theta: Tensor, y_history, design_history, mask=None) -> Tensor:
        """Evaluates the variational posterior's log density at theta.

        Args:
            theta: Parameters, shape ``[*batch_shape, *event_shape_theta]``.
            y_history: Outcomes so far, shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Designs so far, shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional boolean mask for padded histories, shape
                ``[*batch_shape, T]``.

        Returns:
            Tensor: Log density, shape ``[*batch_shape]``.
        """
        ...

    def sample(
        self, y_history, design_history, n_samples: int = 1, mask=None
    ) -> Tensor:
        """Draws samples from the variational posterior.

        Optional: raises NotImplementedError unless a subclass provides it.

        Args:
            y_history: Outcomes so far, shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Designs so far, shape
                ``[*batch_shape, T, *event_shape_design]``.
            n_samples: How many samples to draw per batch element.
            mask: Optional boolean mask for padded histories, shape
                ``[*batch_shape, T]``.

        Returns:
            Tensor: Shape ``[n_samples, *batch_shape, *event_shape_theta]``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement sampling.")

    def _flatten_theta(self, theta: Tensor, batch_shape: torch.Size) -> Tensor:
        """Checks theta's shape and flattens it to ``[*batch_shape, dim_theta]``.

        Raises:
            ValueError: If theta is not shaped
                ``[*batch_shape, *event_shape_theta]``.
        """
        if theta.shape != (*batch_shape, *self.event_shape_theta):
            raise ValueError(
                "theta must have shape [*batch_shape, *event_shape_theta] = "
                f"{(*batch_shape, *self.event_shape_theta)}; got {tuple(theta.shape)}."
            )
        return theta.reshape(*batch_shape, self.dim_theta)


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

    Limitation is addressed by `canonicalize_theta=True`, which sorts
    theta by first coordinate in `log_prob`/`sample`, collapsing the
    label-switching symmetry to one canonical representative per equivalence
    class. A posterior used without that flag still has this limitation.

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
        theta_loc: Optional standardization shift, shape ``[dim_latent]``.
            Defaults to zeros.
        theta_scale: Optional standardization scale, shape ``[dim_latent]``.
            Defaults to ones.
        dim_latent: Width of the space the mixture is built over. Defaults to
            ``dim_theta``, i.e. the mixture models theta directly. A subclass
            that models theta through a reparametrisation sets this to the
            width of *those* coordinates instead (see
            :class:`ConstrainedGaussianMixtureVariationalPosteriorNet`).
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
        canonicalize_theta: bool = False,
        dim_latent: int | None = None,
    ):
        """Initializes."""
        super().__init__(event_shape_theta, canonicalize_theta=canonicalize_theta)

        self.encoder = encoder
        self.n_components = int(n_components)
        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)
        self.dim_latent = self.dim_theta if dim_latent is None else int(dim_latent)

        dim_h = dim_h or encoder.dim_emb
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.dim_h = hidden_widths[-1]
        self.trunk = MLP(
            [encoder.dim_emb] + hidden_widths, activation=activation, activate_last=True
        )

        K, D = self.n_components, self.dim_latent
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
        if self.theta_loc.shape != (D,) or self.theta_scale.shape != (D,):
            raise ValueError(
                f"theta_loc and theta_scale must have shape ({D},), the mixture's "
                f"width; got {tuple(self.theta_loc.shape)} and "
                f"{tuple(self.theta_scale.shape)}."
            )

    def distribution(self, y_history, design_history, mask=None) -> Distribution:
        """Returns the Gaussian-mixture posterior with batch shape ``batch_shape``."""
        return TorchDistribution(
            self._mixture(y_history, design_history, mask=mask),
            reinterpreted_batch_ndims=0,
        )

    def log_prob(self, theta: Tensor, y_history, design_history, mask=None) -> Tensor:
        """Evaluates the mixture's log density at theta; shape ``[*batch_shape]``."""
        mixture = self._mixture(y_history, design_history, mask=mask)
        theta_flat = self._flatten_theta(
            self._maybe_canonicalize(theta), mixture.batch_shape
        )
        return mixture.log_prob(theta_flat)

    def sample(
        self, y_history, design_history, n_samples: int = 1, mask=None
    ) -> Tensor:
        """Draws samples; shape ``[n_samples, *batch_shape, *event_shape_theta]``."""
        mixture = self._mixture(y_history, design_history, mask=mask)
        samples = mixture.sample((n_samples,))  # [n_samples, *batch, dim_theta]
        samples = samples.reshape(
            n_samples, *mixture.batch_shape, *self.event_shape_theta
        )
        return self._maybe_canonicalize(samples)

    def _mixture(self, y_history, design_history, mask=None) -> MixtureSameFamily:
        """Builds the torch Gaussian mixture, with batch shape ``batch_shape``."""
        emb = self.encoder(y_history, design_history, mask=mask)  # [*batch, dim_emb]
        batch_shape = emb.shape[:-1]
        K, D = self.n_components, self.dim_latent

        h = self.trunk(emb)  # [b, dim_h]
        loc = self.loc_head(h).reshape(*batch_shape, K, D)
        raw_log_scale = self.log_scale_head(h).reshape(*batch_shape, K, D)
        log_scale = self.log_scale_min + 0.5 * (
            self.log_scale_max - self.log_scale_min
        ) * (torch.tanh(raw_log_scale) + 1)
        scale = log_scale.exp()

        loc = self.theta_loc + self.theta_scale * loc
        scale = self.theta_scale * scale

        logits = self.logit_head(h).reshape(*batch_shape, K)
        component = Independent(Normal(loc, scale), 1)
        return MixtureSameFamily(Categorical(logits=logits), component)


class ConstrainedGaussianMixtureVariationalPosteriorNet(
    GaussianMixtureVariationalPosteriorNet
):
    r"""Gaussian mixture in unconstrained coordinates for constrained ``theta``.

    The plain mixture models theta in its own coordinates, which fails when theta
    has constrained support. CES is the motivating case:
    ``theta = (rho, alpha_1, alpha_2, alpha_3, u)`` with ``rho`` in ``(0, 1)``,
    ``alpha`` on the 2-simplex and ``u > 0``, so theta lives on a **4-dimensional
    manifold in R^5**. A mixture over ``R^5`` is then wrong twice over: it puts
    mass outside the support, and its density contains a variance term for the
    direction normal to the lower dimensional simplex. Since every valid theta
    lies exactly on the simplex, there is never any error in that direction.
    Training can therefore shrink the corresponding sigma_normal toward zero,
    driving the log-likelihood arbitrarily toward +inf without fitting the
    distribution of theta along the simplex any better.

    This subclass instead builds the mixture over the ``transform.dim_raw``
    unconstrained coordinates and pushes it forward, so the support is respected
    by construction and the density is bounded.

    Args:
        encoder: HistoryEncoder object.
        event_shape_theta: Shape of theta, in constrained coordinates.
        transform: Maps unconstrained coordinates to theta. Its ``dim_theta``
            must equal ``prod(event_shape_theta)``.
    """

    def __init__(
        self,
        encoder: HistoryEncoder,
        event_shape_theta,
        transform: BlockTransform,
        n_components: int = 10,
        dim_h=None,
        activation=nn.GELU,
        log_scale_min: float = -7.0,
        log_scale_max: float = 3.0,
        init_scale: float = 1.0,
        theta_loc: Tensor | None = None,
        theta_scale: Tensor | None = None,
        canonicalize_theta: bool = False,
    ):
        """Initializes."""
        if canonicalize_theta:
            # Canonicalisation permutes an exchangeable object axis inside
            # log_prob. Composing it with a reparametrisation is only valid if
            # the transform is equivariant under that permutation, which nothing
            # checks and no task needs as yet e.g. CES has no exchangeable axis.
            raise ValueError(
                "canonicalize_theta is not supported alongside a reparametrising "
                "transform."
            )

        super().__init__(
            encoder,
            event_shape_theta,
            n_components=n_components,
            dim_h=dim_h,
            activation=activation,
            log_scale_min=log_scale_min,
            log_scale_max=log_scale_max,
            init_scale=init_scale,
            theta_loc=theta_loc,
            theta_scale=theta_scale,
            canonicalize_theta=False,
            dim_latent=transform.dim_raw,
        )

        if transform.dim_theta != self.dim_theta:
            raise ValueError(
                "transform must produce as many coordinates as theta has; "
                f"transform.dim_theta={transform.dim_theta} but "
                f"prod(event_shape_theta)={self.dim_theta}."
            )
        self.transform = transform

    def raw_distribution(self, y_history, design_history, mask=None) -> Distribution:
        """Returns the mixture over unconstrained coordinates for a history."""
        return TorchDistribution(
            self._mixture(y_history, design_history, mask=mask),
            reinterpreted_batch_ndims=0,
        )

    def distribution(self, y_history, design_history, mask=None) -> Distribution:
        """Unavailable: this family has no explicit distribution object over theta.

        The base class's ``distribution`` is a mixture in theta's own coordinates,
        which this net deliberately does not have. Use :meth:`raw_distribution`
        for the unconstrained mixture, or :meth:`log_prob` / :meth:`sample`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} models theta through a transform and exposes no "
            "distribution object over theta; use raw_distribution(), log_prob() or "
            "sample()."
        )

    def log_prob(self, theta: Tensor, y_history, design_history, mask=None) -> Tensor:
        """Evaluates the pushforward log density at theta; shape ``[*batch_shape]``."""
        mixture = self._mixture(y_history, design_history, mask=mask)
        # Before transform.inv, so a wrong-shaped theta still raises ValueError
        # rather than an IndexError from slicing inside the transform.
        raw = self.transform.inv(self._flatten_theta(theta, mixture.batch_shape))
        log_q_raw = mixture.log_prob(raw)

        # Change of variables into theta's coordinates. The correction lives here
        # rather than in the objective because the Barber-Agakov bound is
        # log q - log p: correcting the posterior and correcting the prior term
        # give the same value and the same gradients, and doing it here leaves
        # BarberAgakov, MyopicBarberAgakov, BarberAgakovReinforce and
        # posterior_nll untouched.
        #
        # It also cannot disturb training: the log-det is a function of theta
        # alone, and theta comes from the prior with no graph attached, so it is
        # constant w.r.t. both the posterior parameters and the design. The
        # correction makes the reported bound well defined without changing any
        # gradient.
        return log_q_raw - self.transform.log_abs_det_jacobian(raw, self.transform(raw))

    def sample(
        self, y_history, design_history, n_samples: int = 1, mask=None
    ) -> Tensor:
        """Draws theta samples; shape ``[n_samples, *batch_shape, *event_shape_theta]``.

        Every draw satisfies theta's constraints by construction, since it is the
        image of an unconstrained draw under ``transform``.
        """
        mixture = self._mixture(y_history, design_history, mask=mask)
        raw = mixture.sample((n_samples,))  # [n_samples, *batch_shape, dim_raw]
        theta = self.transform(raw)
        return theta.reshape(n_samples, *mixture.batch_shape, *self.event_shape_theta)


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
        canonicalize_theta: bool = False,
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
            canonicalize_theta=canonicalize_theta,
        )


class ConstrainedAttentiveVariationalPosteriorNet(
    ConstrainedGaussianMixtureVariationalPosteriorNet
):
    """Preset: AttentiveEncoder + Gaussian-mixture readout over transformed theta.

    The constrained counterpart of :class:`AttentiveVariationalPosteriorNet`, for
    a task whose theta has constrained support. The ``transform`` describing that
    support is task-specific and comes from the recipe currently.
    """

    def __init__(
        self,
        event_shape_y,
        event_shape_design,
        event_shape_theta,
        transform: BlockTransform,
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
        canonicalize_theta: bool = False,
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
            transform,
            n_components=n_components,
            dim_h=trunk_dim_h,
            activation=trunk_activation,
            log_scale_min=log_scale_min,
            log_scale_max=log_scale_max,
            init_scale=init_scale,
            theta_loc=theta_loc,
            theta_scale=theta_scale,
            canonicalize_theta=canonicalize_theta,
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
        canonicalize_theta: bool = False,
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
            canonicalize_theta=canonicalize_theta,
        )
