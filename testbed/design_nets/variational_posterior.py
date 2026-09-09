"""Amortized variational posteriors q(theta | y_history, design_history)."""

from abc import ABC, abstractmethod
from math import atanh, log, prod

from torch.distributions.transforms import Transform
from torch import Tensor, nn
import torch
from torch import Tensor
from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
    Normal,
    constraints,
)

from testbed.core.distributions import Distribution, TorchDistribution
from testbed.core.util import sort_by_first_coordinate
from testbed.design_nets.encoder import AttentiveEncoder, DeepSetEncoder, HistoryEncoder
from testbed.design_nets.modules import MLP


class BlockTransform(Transform):
    r"""Apply transforms to contiguous blocks of the last dimension.

    Each block ``i`` reads ``raw_sizes[i]`` unconstrained coordinates, applies
    ``transforms[i]``, and writes ``theta_sizes[i]`` constrained coordinates.
    Blocks remain in order, and any leading dimensions are treated as batch
    dimensions.

    Input and output widths are dim_raw = sum(raw_sizes) dim_theta = sum(theta_sizes)

    The separate size lists are necessary because a transform may change width. For
    example, stick-breaking maps ``R^2`` to a 3-component simplex.

    In the CES case::
        raw_sizes = [1, 2, 1]
        theta_sizes = [1, 3, 1]

        input: [z_rho | z_a1 z_a2 | z_u] # width 4
        output: [rho | a_1 a_2 a_3 | u] # width 5


    PyTorch's ``CatTransform`` and ``biject_to(constraints.cat(...))`` assume a
    single set of block lengths, so they cannot correctly represent this kind of
    width-changing concatenation. ``BlockTransform`` keeps input and output block
    widths separate to avoid that ambiguity.

    Args:
        transforms: One ``Transform`` per block.
        raw_sizes: Number of input coordinates consumed by each block.
        theta_sizes: Number of output coordinates produced by each block.
    """

    domain = constraints.independent(constraints.real, 1)
    # Under-declared on purpose: the honest codomain is a cat of interval /
    # simplex / positive constraints, but biject_to on that is the silent bug
    # above, so declaring it would invite the shortcut this class replaces.
    # Nothing consumes codomain here; log_prob and sample call the maps directly.
    codomain = constraints.independent(constraints.real, 1)
    bijective = True

    def __init__(self, transforms, raw_sizes, theta_sizes, cache_size: int = 0):
        """Initializes the block transform."""
        super().__init__(cache_size=cache_size)

        self.transforms = list(transforms)
        self.raw_sizes = [int(s) for s in raw_sizes]
        self.theta_sizes = [int(s) for s in theta_sizes]

        if not (len(self.transforms) == len(self.raw_sizes) == len(self.theta_sizes)):
            raise ValueError(
                "transforms, raw_sizes and theta_sizes must have equal length; got "
                f"{len(self.transforms)}, {len(self.raw_sizes)}, {len(self.theta_sizes)}."
            )
        if any(s <= 0 for s in self.raw_sizes + self.theta_sizes):
            raise ValueError("Block sizes must be positive.")

        self.dim_raw = sum(self.raw_sizes)
        self.dim_theta = sum(self.theta_sizes)

    def __repr__(self) -> str:
        """Returns a compact representation."""
        blocks = ", ".join(
            f"{type(t).__name__}({r}->{d})"
            for t, r, d in zip(self.transforms, self.raw_sizes, self.theta_sizes)
        )
        return f"{type(self).__name__}([{blocks}])"

    def _split(self, x: Tensor, sizes, name: str):
        """Splits the last axis into per-block slices, checking its width."""
        if x.shape[-1] != sum(sizes):
            raise ValueError(
                f"{name} must have trailing size {sum(sizes)}; got {tuple(x.shape)}."
            )
        return torch.split(x, sizes, dim=-1)

    def _call(self, raw: Tensor) -> Tensor:
        """Maps unconstrained ``[..., dim_raw]`` to constrained ``[..., dim_theta]``."""
        blocks = self._split(raw, self.raw_sizes, "raw")
        return torch.cat(
            [t(block) for t, block in zip(self.transforms, blocks)], dim=-1
        )

    def _inverse(self, theta: Tensor) -> Tensor:
        """Maps constrained ``[..., dim_theta]`` to unconstrained ``[..., dim_raw]``."""
        blocks = self._split(theta, self.theta_sizes, "theta")
        return torch.cat(
            [t.inv(block) for t, block in zip(self.transforms, blocks)], dim=-1
        )

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Log ``|det dtheta/draw|`` at ``x``, summed over blocks; shape ``[*batch]``.

        Summing block terms is valid because the Jacobian is
        block-diagonal: each block's output depends only on its own input, so the
        determinant factorises. A transform whose output read another block's
        input would make this wrong.
        """
        raw_blocks = self._split(x, self.raw_sizes, "x")
        theta_blocks = self._split(y, self.theta_sizes, "y")

        batch_ndim = x.ndim - 1
        total = x.new_zeros(x.shape[:batch_ndim])
        for transform, raw_block, theta_block in zip(
            self.transforms, raw_blocks, theta_blocks
        ):
            term = transform.log_abs_det_jacobian(raw_block, theta_block)
            # Elementwise blocks (sigmoid, exp) return [*batch, size]; blocks with
            # a vector domain (stick-breaking) have already consumed their event
            # dim and return [*batch]. Reduce whatever is left over the batch.
            while term.ndim > batch_ndim:
                term = term.sum(-1)
            total = total + term
        return total


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
        ``y_history``: ``[b, T, *event_shape_y]``
        ``design_history``: ``[b, T, *event_shape_design]``
        ``mask``: ``[b, T]`` or None

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
            theta: Parameters, shape ``[b, *event_shape_theta]``.
            y_history: Outcomes so far, shape ``[b, T, *event_shape_y]``.
            design_history: Designs so far, shape ``[b, T, *event_shape_design]``.
            mask: Optional boolean mask for padded histories, shape ``[b, T]``.

        Returns:
            Tensor: Log density, shape ``[b]``.
        """
        ...

    def sample(
        self, y_history, design_history, n_samples: int = 1, mask=None
    ) -> Tensor:
        """Draws samples from the variational posterior.

        Optional: raises NotImplementedError unless a subclass provides it.

        Args:
            y_history: Outcomes so far, shape ``[b, T, *event_shape_y]``.
            design_history: Designs so far, shape ``[b, T, *event_shape_design]``.
            n_samples: How many samples to draw per batch element.
            mask: Optional boolean mask for padded histories, shape ``[b, T]``.

        Returns:
            Tensor: Shape ``[n_samples, b, *event_shape_theta]``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement sampling.")

    def _check_theta_shape(self, theta: Tensor) -> None:
        """Raises if theta's trailing dims are not ``event_shape_theta``."""
        trailing_shape = torch.Size(
            theta.shape[len(theta.shape) - len(self.event_shape_theta) :]
        )
        if trailing_shape != self.event_shape_theta:
            raise ValueError(
                f"theta must have trailing shape {tuple(self.event_shape_theta)}; "
                f"got {tuple(theta.shape)}."
            )

    def _flatten_theta(self, theta: Tensor) -> Tensor:
        """Checks theta's trailing event shape and flattens it to ``[b, dim_theta]``."""
        self._check_theta_shape(theta)
        return theta.reshape(theta.shape[0], self.dim_theta)


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
        """Returns the Gaussian-mixture posterior for a history."""
        return TorchDistribution(self._mixture(y_history, design_history, mask=mask))

    def log_prob(self, theta: Tensor, y_history, design_history, mask=None) -> Tensor:
        """Evaluates the mixture's log density at theta; shape ``[b]``."""
        theta_flat = self._flatten_theta(self._maybe_canonicalize(theta))
        return self.distribution(y_history, design_history, mask=mask).log_prob(
            theta_flat
        )

    def sample(
        self, y_history, design_history, n_samples: int = 1, mask=None
    ) -> Tensor:
        """Draws mixture samples; shape ``[n_samples, b, *event_shape_theta]``."""
        dist = self.distribution(y_history, design_history, mask=mask)
        samples = dist.sample(n_samples)  # [n_samples, b, dim_theta]
        samples = samples.reshape(
            n_samples, y_history.shape[0], *self.event_shape_theta
        )
        return self._maybe_canonicalize(samples)

    def _mixture(self, y_history, design_history, mask=None) -> MixtureSameFamily:
        """Builds the underlying torch Gaussian mixture for a history."""
        emb = self.encoder(y_history, design_history, mask=mask)  # [b, dim_emb]
        b = emb.shape[0]
        K, D = self.n_components, self.dim_latent

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
        return TorchDistribution(self._mixture(y_history, design_history, mask=mask))

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
        """Evaluates the pushforward log density at theta; shape ``[b]``."""
        # Before transform.inv, so a wrong-shaped theta still raises ValueError
        # rather than an IndexError from slicing inside the transform.
        self._check_theta_shape(theta)

        raw = self.transform.inv(theta.reshape(theta.shape[0], self.dim_theta))
        log_q_raw = self.raw_distribution(
            y_history, design_history, mask=mask
        ).log_prob(raw)

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
        """Draws samples in theta space; shape ``[n_samples, b, *event_shape_theta]``.

        Every draw satisfies theta's constraints by construction, since it is the
        image of an unconstrained draw under ``transform``.
        """
        raw = self.raw_distribution(y_history, design_history, mask=mask).sample(
            n_samples
        )  # [n_samples, b, dim_raw]
        theta = self.transform(raw)
        return theta.reshape(n_samples, y_history.shape[0], *self.event_shape_theta)


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
