"""Stochastic policy networks for reinforcement learning BED agents."""

from collections.abc import Sequence
from math import atanh, log

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from testbed.core.constraints import interval_bounds, is_unconstrained
from testbed.core.distributions import Distribution, EmpiricalDistribution
from testbed.design_nets.base import DesignNet, DesignProposal
from testbed.design_nets.encoder import (
    DeepSetEncoder,
    HistoryEncoder,
    make_history_encoder,
)
from testbed.design_nets.history import normalize_history, prepare_history
from testbed.design_nets.modules import MLP, _initialize_linear_layers


class StochasticDesignNet(DesignNet):
    """Base network for sampled proposals with policy log probabilities.

    Subclasses implement :meth:`propose` and set the policy's history
    normalization contract. Their proposals participate in score-function
    training unless they support and are asked for reparameterized samples.
    """

    produces_policy_log_probs = True
    raw_designs_are_physical = False

    def __init__(
        self,
        design_prior: Distribution,
        *,
        normalize_design_history: bool,
        design_history_bounds: tuple[float, float],
    ):
        """Stores the design shape and history-normalization settings.

        Args:
            design_prior: Design prior whose event shape defines one design.
            normalize_design_history: Whether to rescale design histories.
            design_history_bounds: Range that design histories are rescaled to.

        Raises:
            ValueError: If ``design_history_bounds`` is not a ``(low, high)``
                pair with ``low < high``.
        """
        super().__init__()
        self.event_shape_design = torch.Size(design_prior.event_shape)
        self.dim_design = self.event_shape_design.numel()
        self.normalize_design_history = bool(normalize_design_history)
        self.design_history_bounds = tuple(float(x) for x in design_history_bounds)
        if (
            len(self.design_history_bounds) != 2
            or self.design_history_bounds[0] >= self.design_history_bounds[1]
        ):
            raise ValueError("design_history_bounds must satisfy low < high.")

    def _set_encoder_and_head(
        self,
        encoder: HistoryEncoder | None,
        head: nn.Module | None,
        *,
        event_shape_y: Sequence[int] | None,
        make_head,
    ) -> None:
        """Stores the history encoder and the policy head that reads it.

        Args:
            encoder: Optional history encoder. Defaults to a ``DeepSetEncoder``.
            head: Optional readout of the encoding. Defaults to
                ``make_head(encoder.dim_emb)``.
            event_shape_y: Outcome event shape for the default encoder.
            make_head: Builds the default head from an embedding width.

        Raises:
            ValueError: If a head is given without an encoder, the default
                encoder lacks ``event_shape_y``, or the embedding widths differ.
        """
        if encoder is None:
            if head is not None:
                raise ValueError("head requires an encoder.")
            if event_shape_y is None:
                raise ValueError(
                    "event_shape_y is required when creating the default encoder."
                )
            encoder = DeepSetEncoder(
                event_shape_y, self.event_shape_design, dim_h=(128, 128), dim_emb=64
            )
        if head is None:
            head = make_head(encoder.dim_emb)
        if encoder.dim_emb != head.dim_emb:
            raise ValueError("encoder and head embedding dimensions must match.")
        self.encoder = encoder
        self.head = head

    def _register_bounds(
        self,
        design_low: Tensor,
        design_high: Tensor,
        outcome_bounds: tuple[float, float] | None,
    ) -> None:
        """Registers the design and outcome bounds used to normalize histories.

        Args:
            design_low: Per-coordinate lower design bounds, flattened.
            design_high: Per-coordinate upper design bounds, flattened.
            outcome_bounds: Optional ``(low, high)`` outcome bounds, each a
                scalar or one entry per outcome coordinate. If omitted, outcomes
                are not normalized and no outcome buffers are stored.

        Raises:
            ValueError: If ``outcome_bounds`` does not satisfy ``low < high``.
        """
        self.register_buffer("design_low", design_low)
        self.register_buffer("design_high", design_high)
        outcome_low = outcome_high = None
        if outcome_bounds is not None:
            outcome_low, outcome_high = (
                torch.as_tensor(bound, dtype=design_low.dtype).reshape(-1)
                for bound in outcome_bounds
            )
            if torch.any(outcome_low >= outcome_high):
                raise ValueError("outcome_bounds must satisfy low < high.")
        self.register_buffer("outcome_low", outcome_low)
        self.register_buffer("outcome_high", outcome_high)

    def _encode(self, y_history, design_history, mask: Tensor | None) -> Tensor:
        """Validates, normalizes, and encodes policy histories.

        Args:
            y_history: Outcome histories in the public history format.
            design_history: Physical design histories in the public format.
            mask: Optional validity mask for padded history positions.

        Returns:
            Tensor: History encodings shaped ``[*batch_shape, dim_emb]``.
        """
        history = prepare_history(
            y_history,
            design_history,
            event_shape_y=self.encoder.event_shape_y,
            event_shape_design=self.event_shape_design,
            mask=mask,
            device=self.encoder.null_enc.device,
            dtype=self.encoder.null_enc.dtype,
        )
        history = normalize_history(
            history,
            design_bounds=(
                (self.design_low, self.design_high)
                if self.normalize_design_history
                else None
            ),
            design_target=self.design_history_bounds,
            outcome_bounds=(
                None
                if self.outcome_low is None
                else (self.outcome_low, self.outcome_high)
            ),
        )
        return self.encoder.encode(history)


class GaussianHead(nn.Module):
    """Maps a history encoding to Gaussian policy parameters."""

    def __init__(
        self,
        dim_emb: int,
        dim_design: int,
        dim_h=(128, 128),
        *,
        log_std_min: float = -20.0,
        log_std_max: float = 0.0,
        init_std: float = 3**-0.5,
    ):
        """Initializes the Gaussian readout.

        Args:
            dim_emb: Dimension of the input history encoding.
            dim_design: Flattened design dimension.
            dim_h: Hidden layer widths; an int specifies one layer.
            log_std_min: Lower bound for Gaussian log standard deviations.
            log_std_max: Upper bound for Gaussian log standard deviations.
            init_std: Initial unsquashed Gaussian standard deviation.
        """
        super().__init__()
        self.dim_emb = int(dim_emb)
        self.dim_design = int(dim_design)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.init_std = float(init_std)
        if self.init_std <= 0:
            raise ValueError("init_std must be positive.")
        init_log_std = log(self.init_std)
        if not self.log_std_min < init_log_std < self.log_std_max:
            raise ValueError(
                "init_std must have a logarithm strictly between "
                "log_std_min and log_std_max."
            )

        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.actor_net = MLP(
            [self.dim_emb] + hidden_widths,
            activation=nn.ReLU,
            activate_last=True,
        )
        self.mean_head = nn.Linear(hidden_widths[-1], self.dim_design)
        self.log_std_head = nn.Linear(hidden_widths[-1], self.dim_design)
        _initialize_linear_layers(self)
        nn.init.zeros_(self.log_std_head.weight)
        scaled_log_std = (
            2.0
            * (
                (init_log_std - self.log_std_min)
                / (self.log_std_max - self.log_std_min)
            )
            - 1.0
        )
        nn.init.constant_(self.log_std_head.bias, atanh(scaled_log_std))

    def forward(self, encoding: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the Gaussian mean and log standard deviation."""
        hidden = self.actor_net(encoding)
        mean = self.mean_head(hidden)
        log_std = torch.tanh(self.log_std_head(hidden))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        return mean, log_std


class CategoricalHead(nn.Module):
    """Maps a history encoding to categorical design logits."""

    def __init__(self, dim_emb: int, n_designs: int, dim_h=(128, 128)):
        """Initializes the categorical readout.

        Args:
            dim_emb: Width of each history embedding.
            n_designs: Number of designs represented by the logits.
            dim_h: Hidden layer widths; an integer specifies one layer.
        """
        super().__init__()
        self.dim_emb = int(dim_emb)
        self.n_designs = int(n_designs)
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.actor_net = MLP([self.dim_emb] + hidden_widths, activation=nn.ReLU)
        self.logits_head = nn.Linear(hidden_widths[-1], self.n_designs)
        _initialize_linear_layers(self)

    def forward(self, encoding: Tensor) -> Tensor:
        """Returns one categorical logit per design.

        Args:
            encoding: History encodings with shape ``[batch, dim_emb]``.

        Returns:
            Tensor: Logits with shape ``[batch, n_designs]``.
        """
        return self.logits_head(self.actor_net(encoding))


class GaussianPolicyNet(StochasticDesignNet):
    """Gaussian policy for continuous design spaces.

    The network proposes values in unconstrained Gaussian coordinates. The
    caller's design transform owns the constraint and density correction.
    """

    supports_reparameterized_proposals = True

    def __init__(
        self,
        design_prior: Distribution,
        encoder: HistoryEncoder | None = None,
        head: GaussianHead | None = None,
        *,
        event_shape_y: Sequence[int] | None = None,
        normalize_design_history: bool = False,
        design_history_bounds: tuple[float, float] = (-1.0, 1.0),
        outcome_bounds: tuple[float, float] | None = None,
    ):
        """Initializes the policy network.

        Args:
            design_prior: Design prior over a box or all of real space, whose
                shape defines the proposal. Box bounds support history
                normalization; an unconstrained space uses [-1, 1] as a
                reference scale only.
            encoder: Optional history encoder. Defaults to a ``DeepSetEncoder``.
            head: Optional Gaussian policy head paired with ``encoder``. If
                omitted, a default head is created for the encoder.
            event_shape_y: Shape of one outcome event. Required when creating
                the default encoder.
            normalize_design_history: If True, affinely map physical design
                histories to `design_history_bounds` before encoding them.
            design_history_bounds: Target bounds for normalized design history.
            outcome_bounds: Optional lower and upper bounds used to affinely map
                physical outcome histories to [0, 1] before encoding them.

        Raises:
            ValueError: If ``head`` does not output ``dim_design`` parameters.
        """
        super().__init__(
            design_prior,
            normalize_design_history=normalize_design_history,
            design_history_bounds=design_history_bounds,
        )
        self._set_encoder_and_head(
            encoder,
            head,
            event_shape_y=event_shape_y,
            make_head=lambda dim_emb: GaussianHead(dim_emb, self.dim_design),
        )
        if self.head.dim_design != self.dim_design:
            raise ValueError("head.dim_design must match the design space.")

        support = design_prior.support
        bounds = interval_bounds(support)
        if bounds is not None:
            low = bounds[0].reshape(self.dim_design)
            high = bounds[1].reshape(self.dim_design)
        elif is_unconstrained(support):
            # Reference scale for network inputs only; an unconstrained space
            # keeps policy proposals unconstrained.
            low = torch.full((self.dim_design,), -1.0)
            high = torch.full((self.dim_design,), 1.0)
        else:
            raise TypeError(
                "GaussianPolicyNet requires a design prior over a box or all of "
                "real space."
            )
        if torch.any(low >= high):
            raise ValueError(
                "GaussianPolicyNet requires finite-width reference bounds."
            )
        self._register_bounds(low.clone(), high.clone(), outcome_bounds)

    def forward(
        self, y_history, design_history, *, mask: Tensor | None = None
    ) -> Tensor:
        """Returns the deterministic raw Gaussian design for a history.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional boolean tensor for padded histories with shape
                ``[*batch_shape, T]``.

        Returns:
            Gaussian mean with shape
            ``[*batch_shape, *event_shape_design]``.
        """
        mean, _ = self._normal_params(y_history, design_history, mask)
        return self._as_design(mean)

    def propose(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        reparameterize_policy: bool = False,
    ) -> DesignProposal:
        """Samples an unconstrained Gaussian proposal and its raw density.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional validity mask with shape ``[*batch_shape, T]``.
            reparameterize_policy: Whether to sample with ``Normal.rsample``.

        Returns:
            DesignProposal: A raw sampled design, its summed raw log density,
            and its deterministic mean counterpart.
        """
        mean, log_std = self._normal_params(y_history, design_history, mask)
        normal = Normal(mean, log_std.exp())
        if reparameterize_policy:
            raw_design = normal.rsample()
        else:
            raw_design = normal.sample()

        return DesignProposal(
            raw_design=self._as_design(raw_design),
            log_prob=normal.log_prob(raw_design).sum(dim=-1),
            deterministic_raw_design=self._as_design(mean),
        )

    def _as_design(self, value: Tensor) -> Tensor:
        """Reshapes ``[*batch_shape, dim_design]`` to the design event shape."""
        return value.reshape(*value.shape[:-1], *self.event_shape_design)

    def _normal_params(
        self, y_history, design_history, mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Returns Gaussian parameters shaped ``[*batch_shape, dim_design]``."""
        encoding = self._encode(y_history, design_history, mask)
        mean, log_std = self.head(encoding)
        expected_shape = (*encoding.shape[:-1], self.dim_design)
        for name, value in (("mean", mean), ("log_std", log_std)):
            if value.shape != expected_shape:
                raise ValueError(
                    f"head {name} must have shape [*batch_shape, dim_design]; "
                    f"expected {expected_shape}, got {tuple(value.shape)}."
                )
        return mean, log_std


class CategoricalPolicyNet(StochasticDesignNet):
    """Categorical policy that maps design indices to physical designs.

    Attributes:
        event_shape_design: Shape of one physical design.
        dim_design: Flattened design dimension.
        n_designs: Number of categorical designs.
        normalize_design_history: Whether design histories are normalized.
        design_history_bounds: Target bounds for normalized design histories.
        encoder: History encoder read by ``head``.
        head: Readout producing one logit per categorical design.
        design_values: Physical designs indexed by categorical design.
        design_low: Per-coordinate minimum physical design values.
        design_high: Per-coordinate maximum physical design values.
        outcome_low: Lower bound used to normalize outcome histories, or
            ``None`` if outcomes are not normalized.
        outcome_high: Upper bound used to normalize outcome histories, or
            ``None`` if outcomes are not normalized.
    """

    raw_designs_are_physical = True

    def __init__(
        self,
        design_prior: EmpiricalDistribution,
        encoder: HistoryEncoder | None = None,
        head: CategoricalHead | None = None,
        *,
        event_shape_y: Sequence[int] | None = None,
        normalize_design_history: bool = False,
        design_history_bounds: tuple[float, float] = (0.0, 1.0),
        outcome_bounds: tuple[float, float] | None = None,
    ):
        """Initializes the categorical policy.

        Args:
            design_prior: Finite set of physical designs returned by policy
                proposals.
            encoder: Optional history encoder. Defaults to a ``DeepSetEncoder``.
            head: Optional categorical policy head paired with ``encoder``.
                If omitted, a default head is created for the encoder.
            event_shape_y: Shape of one outcome event. Required when creating
                the default encoder.
            normalize_design_history: Whether to rescale design histories.
            design_history_bounds: Bounds used when design histories are rescaled.
            outcome_bounds: Optional bounds used to rescale outcomes to [0, 1].

        Raises:
            ValueError: If history or outcome bounds are invalid, or ``head``
                does not output one logit per design.
        """
        super().__init__(
            design_prior,
            normalize_design_history=normalize_design_history,
            design_history_bounds=design_history_bounds,
        )
        self.n_designs = design_prior.locs.shape[0]
        self._set_encoder_and_head(
            encoder,
            head,
            event_shape_y=event_shape_y,
            make_head=lambda dim_emb: CategoricalHead(dim_emb, self.n_designs),
        )
        if self.head.n_designs != self.n_designs:
            raise ValueError("head.n_designs must match the design space.")

        self.register_buffer("design_values", design_prior.locs.clone())
        values = self.design_values.reshape(self.n_designs, self.dim_design)
        if not torch.is_floating_point(values):
            values = values.to(torch.get_default_dtype())
        self._register_bounds(values.amin(dim=0), values.amax(dim=0), outcome_bounds)

    def forward(
        self, y_history, design_history, *, mask: Tensor | None = None
    ) -> Tensor:
        """Returns the highest-probability physical design.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional padding mask with shape ``[*batch_shape, T]``.

        Returns:
            Most probable designs with shape
            ``[*batch_shape, *event_shape_design]``.
        """
        logits = self.logits(y_history, design_history, mask=mask)
        return self.design_values[logits.argmax(dim=-1)]

    def propose(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        reparameterize_policy: bool = False,
    ) -> DesignProposal:
        """Samples a physical design and its selected-design log probability.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional padding mask with shape ``[*batch_shape, T]``.
            reparameterize_policy: Part of the common proposal contract;
                categorical samples ignore it because they are not reparameterized.

        Returns:
            A proposal whose raw and deterministic values are physical designs,
            with the selected index in ``info["design_index"]``.
        """
        del reparameterize_policy
        logits = self.logits(y_history, design_history, mask=mask)
        distribution = Categorical(logits=logits)
        design_index = distribution.sample()
        return DesignProposal(
            raw_design=self.design_values[design_index],
            log_prob=distribution.log_prob(design_index),
            deterministic_raw_design=self.design_values[logits.argmax(dim=-1)],
            info={"design_index": design_index},
        )

    def logits(
        self, y_history, design_history, *, mask: Tensor | None = None
    ) -> Tensor:
        """Returns one action logit per categorical design.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional padding mask with shape ``[*batch_shape, T]``.

        Returns:
            Logits with shape ``[*batch_shape, n_designs]``.
        """
        encoding = self._encode(y_history, design_history, mask)
        logits = self.head(encoding)
        expected_shape = (*encoding.shape[:-1], self.n_designs)
        if logits.shape != expected_shape:
            raise ValueError(
                "head logits must have shape [*batch_shape, n_designs]; "
                f"expected {expected_shape}, got {tuple(logits.shape)}."
            )
        return logits

    def design_probabilities(
        self, y_history, design_history, *, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Returns exact categorical probabilities and log probabilities.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional padding mask with shape ``[*batch_shape, T]``.

        Returns:
            A ``(probabilities, log_probabilities)`` tuple, each with shape
                ``[*batch_shape, n_designs]``.
        """
        log_probs = torch.log_softmax(
            self.logits(y_history, design_history, mask=mask), dim=-1
        )
        return log_probs.exp(), log_probs


def make_stochastic_policy(
    *,
    design_prior: Distribution,
    event_shape_y,
    event_shape_design,
    dim_h,
    dim_emb,
    log_std_min,
    log_std_max,
    init_std,
    normalize_design_history,
    design_history_bounds,
    outcome_bounds,
    architecture="deepset",
    attention=None,
) -> StochasticDesignNet:
    """Composes a history encoder with the head required by the design prior."""
    encoder = make_history_encoder(
        event_shape_y=event_shape_y,
        event_shape_design=event_shape_design,
        architecture=architecture,
        dim_h=dim_h,
        dim_emb=dim_emb,
        attention=attention,
    )
    kwargs = {
        "encoder": encoder,
        "normalize_design_history": normalize_design_history,
        "design_history_bounds": tuple(design_history_bounds),
        "outcome_bounds": tuple(outcome_bounds),
    }
    support = design_prior.support
    if interval_bounds(support) is not None or is_unconstrained(support):
        head = GaussianHead(
            encoder.dim_emb,
            design_prior.event_shape.numel(),
            dim_h=dim_h,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            init_std=init_std,
        )
        return GaussianPolicyNet(design_prior, head=head, **kwargs)
    if isinstance(design_prior, EmpiricalDistribution):
        head = CategoricalHead(encoder.dim_emb, design_prior.locs.shape[0], dim_h=dim_h)
        return CategoricalPolicyNet(design_prior, head=head, **kwargs)
    raise TypeError(
        "Stochastic policies require a design prior over a box, all of real "
        "space, or a finite set (an EmpiricalDistribution)."
    )
