"""Stochastic policy networks for reinforcement learning BED agents."""

from collections.abc import Sequence
from math import atanh, log

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from testbed.core.design_spaces import (
    BoxDesignSpace,
    DefaultDesignSpace,
    FiniteDesignSpace,
)
from testbed.design_nets.base import DesignNet, DesignProposal
from testbed.design_nets.encoder import DeepSetEncoder, HistoryEncoder
from testbed.design_nets.history import PreparedHistory, prepare_history
from testbed.design_nets.modules import MLP, _initialize_linear_layers, affine_rescale


class StochasticDesignNet(DesignNet):
    """Base network for sampled proposals with policy log probabilities.

    Subclasses implement :meth:`propose` and set the policy's history
    normalization contract. Their proposals participate in score-function
    training unless they support and are asked for reparameterized samples.
    """

    produces_policy_log_probs = True
    raw_designs_are_physical = False

    def _prepare_histories(
        self, y_history, design_history, *, mask: Tensor | None = None
    ) -> PreparedHistory:
        """Validates, flattens, and optionally normalizes policy histories.

        Args:
            y_history: Outcome histories in the public history format.
            design_history: Physical design histories in the public format.
            mask: Optional validity mask for padded history positions.

        Returns:
            PreparedHistory: Flattened actor inputs and the flattened mask.
        """
        encoder = getattr(self.actor, "encoder", None)
        event_shape_y = getattr(self.actor, "event_shape_y", None)
        if event_shape_y is None and encoder is not None:
            event_shape_y = encoder.event_shape_y
        prepared = prepare_history(
            y_history,
            design_history,
            event_shape_y=event_shape_y,
            event_shape_design=self.event_shape_design,
            mask=mask,
            device=self.design_low.device,
            dtype=self.design_low.dtype,
        )

        y_flat = prepared.y_flat
        design_flat = prepared.design_flat
        if self.normalize_design_history:
            low, high = self.design_history_bounds
            width = self.design_high - self.design_low
            normalized = low + (design_flat - self.design_low) * (high - low) / width
            design_flat = torch.where(
                width == 0, torch.zeros_like(normalized), normalized
            )
        if self._normalize_outcomes:
            y_flat = affine_rescale(
                y_flat,
                self.outcome_low.reshape(-1),
                self.outcome_high.reshape(-1),
                0.0,
                1.0,
            )
        return PreparedHistory(
            prepared.batch_shape, y_flat, design_flat, prepared.mask_flat
        )


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
        self.actor_net = MLP([self.dim_emb] + hidden_widths, activation=nn.ReLU)
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


class GaussianPolicyActor(nn.Module):
    """History encoder followed by a Gaussian policy head."""

    def __init__(
        self,
        encoder: HistoryEncoder,
        head: GaussianHead,
    ):
        """Initializes an encoder and Gaussian head pair.

        The encoder and head must produce and consume the same embedding
        dimension.

        Args:
            encoder: History encoder that produces the Gaussian-head input.
            head: Readout that maps encoder outputs to Gaussian parameters.

        Raises:
            ValueError: If the encoder and head embedding widths differ.
        """
        super().__init__()
        if encoder.dim_emb != head.dim_emb:
            raise ValueError("encoder and head embedding dimensions must match.")
        self.encoder = encoder
        self.head = head

    def forward(
        self, y_history: Tensor, design_history: Tensor, *, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Computes Gaussian mean and log standard deviation.

        Args:
            y_history: Outcome history with shape ``[batch, T, dim_y]``.
            design_history: Design history with shape ``[batch, T, dim_design]``.
            mask: Optional boolean or float mask with shape ``[batch, T]``.

        Returns:
            Gaussian mean and log standard deviation, both with shape
            ``[batch, dim_design]``.
        """
        return self.head(self.encoder.encode_flat(y_history, design_history, mask=mask))


class DeepSetGaussianActor(GaussianPolicyActor):
    """Preset: ``DeepSetEncoder`` followed by a ``GaussianHead``."""

    def __init__(
        self,
        *,
        event_shape_y: Sequence[int],
        event_shape_design: Sequence[int],
        dim_h=(128, 128),
        dim_emb: int = 64,
        log_std_min: float = -20.0,
        log_std_max: float = 0.0,
        init_std: float = 3**-0.5,
    ):
        """Initializes the actor network.

        Args:
            event_shape_y: Shape of one outcome event.
            event_shape_design: Shape of one design event.
            dim_h: Per-layer hidden widths as a list; an int is a single layer.
            dim_emb: History summary dimension.
            log_std_min: Lower bound for Gaussian log standard deviations.
            log_std_max: Upper bound for Gaussian log standard deviations.
            init_std: Initial unsquashed Gaussian standard deviation.
        """
        event_shape_y = torch.Size(event_shape_y)
        event_shape_design = torch.Size(event_shape_design)
        encoder = DeepSetEncoder(
            event_shape_y,
            event_shape_design,
            dim_h=dim_h,
            dim_emb=dim_emb,
        )
        head = GaussianHead(
            encoder.dim_emb,
            event_shape_design.numel(),
            dim_h=dim_h,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            init_std=init_std,
        )
        super().__init__(encoder, head)
        self.event_shape_y = event_shape_y
        self.event_shape_design = event_shape_design
        self.dim_y = event_shape_y.numel()
        self.dim_design = event_shape_design.numel()


class CategoricalPolicyActor(nn.Module):
    """History encoder followed by a categorical policy head."""

    def __init__(self, encoder: HistoryEncoder, head: CategoricalHead):
        """Initializes an encoder and categorical head pair.

        Args:
            encoder: History encoder that produces the categorical-head input.
            head: Readout that maps encoder outputs to categorical logits.

        Raises:
            ValueError: If the encoder and head embedding widths differ.
        """
        super().__init__()
        if encoder.dim_emb != head.dim_emb:
            raise ValueError("encoder and head embedding dimensions must match.")
        self.encoder = encoder
        self.head = head

    def forward(
        self, y_history: Tensor, design_history: Tensor, *, mask: Tensor | None = None
    ) -> Tensor:
        """Computes design logits from a flattened history.

        Args:
            y_history: Outcomes with shape ``[batch, T, dim_y]``.
            design_history: Designs with shape ``[batch, T, dim_design]``.
            mask: Optional validity mask with shape ``[batch, T]``.

        Returns:
            Tensor: One logit per design with shape ``[batch, n_designs]``.
        """
        return self.head(self.encoder.encode_flat(y_history, design_history, mask=mask))


class DeepSetCategoricalActor(CategoricalPolicyActor):
    """DeepSets actor that returns categorical design logits.

    Attributes:
        event_shape_design: Shape of one design event.
        event_shape_y: Shape of one outcome event.
        dim_design: Flattened design dimension.
        dim_y: Flattened outcome dimension.
        n_designs: Number of categorical designs.
        dim_emb: History-summary dimension.
        encoder_net: Module that embeds each design-outcome pair.
        actor_net: Module that transforms the history summary.
        logits_head: Linear layer that produces categorical design logits.
    """

    def __init__(
        self,
        *,
        event_shape_y: Sequence[int],
        event_shape_design: Sequence[int],
        n_designs: int,
        dim_h=(128, 128),
        dim_emb: int = 64,
    ):
        """Initializes the actor.

        Args:
            event_shape_y: Shape of one outcome event.
            event_shape_design: Shape of one design event.
            n_designs: Number of categorical designs.
            dim_h: Per-layer hidden widths as a list; an int is a single layer.
            dim_emb: History summary dimension.
        """
        event_shape_y = torch.Size(event_shape_y)
        event_shape_design = torch.Size(event_shape_design)
        encoder = DeepSetEncoder(
            event_shape_y,
            event_shape_design,
            dim_h=dim_h,
            dim_emb=dim_emb,
        )
        head = CategoricalHead(encoder.dim_emb, n_designs, dim_h=dim_h)
        super().__init__(encoder, head)
        self.event_shape_y = event_shape_y
        self.event_shape_design = event_shape_design
        self.dim_y = event_shape_y.numel()
        self.dim_design = event_shape_design.numel()


class GaussianPolicyNet(StochasticDesignNet):
    """Gaussian policy for continuous box design spaces.

    The network proposes values in unconstrained Gaussian coordinates. The
    caller's design space owns the constraint transform and density correction.
    """

    supports_reparameterized_proposals = True

    def __init__(
        self,
        design_space: BoxDesignSpace,
        actor: nn.Module | None = None,
        *,
        encoder: HistoryEncoder | None = None,
        head: GaussianHead | None = None,
        event_shape_y: Sequence[int] | None = None,
        normalize_design_history: bool = False,
        design_history_bounds: tuple[float, float] = (-1.0, 1.0),
        outcome_bounds: tuple[float, float] | None = None,
    ):
        """Initializes the policy network.

        Args:
            design_space: Physical continuous design space whose shape defines
                the proposal and whose bounds support history normalization.
            actor: Optional module mapping flattened histories to Gaussian
                mean and log standard deviation.
            encoder: Optional history encoder used to build a default Gaussian
                actor. Cannot be combined with ``actor``.
            head: Optional Gaussian policy head paired with ``encoder``. If
                omitted, a default head is created for the encoder.
            event_shape_y: Shape of one outcome event. Required when creating
                the default actor without an explicit ``encoder``.
            normalize_design_history: If True, affinely map physical design
                histories to `design_history_bounds` before actor evaluation.
            design_history_bounds: Target bounds for normalized design history.
            outcome_bounds: Optional lower and upper bounds used to affinely map
                physical outcome histories to [0, 1] before actor evaluation.
        """
        super().__init__()

        self.design_space = design_space
        self.event_shape_design = torch.Size(design_space.event_shape)
        self.dim_design = self.event_shape_design.numel()
        self.normalize_design_history = bool(normalize_design_history)
        self.design_history_bounds = tuple(float(x) for x in design_history_bounds)
        if (
            len(self.design_history_bounds) != 2
            or self.design_history_bounds[0] >= self.design_history_bounds[1]
        ):
            raise ValueError("design_history_bounds must satisfy low < high.")
        if actor is not None and (encoder is not None or head is not None):
            raise ValueError("actor cannot be combined with encoder or head.")
        if actor is not None:
            self.actor = actor
        elif encoder is not None:
            self.actor = GaussianPolicyActor(
                encoder,
                head or GaussianHead(encoder.dim_emb, self.dim_design),
            )
        elif head is not None:
            raise ValueError("head requires an encoder.")
        else:
            if event_shape_y is None:
                raise ValueError(
                    "event_shape_y is required when creating the default actor."
                )
            self.actor = DeepSetGaussianActor(
                event_shape_y=event_shape_y,
                event_shape_design=self.event_shape_design,
            )

        low = design_space.low.reshape(self.dim_design)
        high = design_space.high.reshape(self.dim_design)
        if torch.any(low >= high):
            raise ValueError("GaussianPolicyNet requires finite-width design bounds.")
        self.register_buffer("design_low", low.clone())
        self.register_buffer("design_high", high.clone())

        if outcome_bounds is None:
            outcome_low = torch.tensor(float("nan"), dtype=low.dtype)
            outcome_high = torch.tensor(float("nan"), dtype=low.dtype)
        else:
            outcome_low, outcome_high = (
                torch.as_tensor(bound, dtype=low.dtype) for bound in outcome_bounds
            )
            if torch.any(outcome_low >= outcome_high):
                raise ValueError("outcome_bounds must satisfy low < high.")
        self.register_buffer("outcome_low", outcome_low)
        self.register_buffer("outcome_high", outcome_high)
        self._normalize_outcomes = outcome_bounds is not None

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
        history = self._prepare_histories(
            y_history,
            design_history,
            mask=mask,
        )
        mean, _ = self._normal_params(
            history.y_flat, history.design_flat, history.mask_flat
        )
        return mean.reshape(*history.batch_shape, *self.event_shape_design)

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
        history = self._prepare_histories(
            y_history,
            design_history,
            mask=mask,
        )
        mean, log_std = self._normal_params(
            history.y_flat, history.design_flat, history.mask_flat
        )
        normal = Normal(mean, log_std.exp())
        if reparameterize_policy:
            raw_design = normal.rsample()
        else:
            raw_design = normal.sample()

        log_prob = normal.log_prob(raw_design).sum(dim=-1)
        raw_design = raw_design.reshape(*history.batch_shape, *self.event_shape_design)
        deterministic_raw_design = mean.reshape(
            *history.batch_shape, *self.event_shape_design
        )

        return DesignProposal(
            raw_design=raw_design,
            log_prob=log_prob.reshape(history.batch_shape),
            deterministic_raw_design=deterministic_raw_design,
        )

    def _normal_params(
        self,
        y_flat: Tensor,
        design_flat: Tensor,
        mask_flat: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Returns validated Gaussian parameters in flattened actor coordinates."""
        mean, log_std = self.actor(y_flat, design_flat, mask=mask_flat)
        self._validate_actor_params(mean, log_std, batch_size=design_flat.shape[0])
        return mean, log_std

    def _validate_actor_params(
        self, mean: Tensor, log_std: Tensor, *, batch_size: int
    ) -> None:
        """Validates Gaussian parameter shapes returned by the actor."""
        expected_shape = (batch_size, self.dim_design)
        if mean.shape != expected_shape:
            raise ValueError(
                "actor mean must have shape "
                f"[batch, dim_design]; expected {expected_shape}, got "
                f"{tuple(mean.shape)}."
            )
        if log_std.shape != expected_shape:
            raise ValueError(
                "actor log_std must have shape "
                f"[batch, dim_design]; expected {expected_shape}, got "
                f"{tuple(log_std.shape)}."
            )


class CategoricalPolicyNet(StochasticDesignNet):
    """Categorical policy that maps design indices to physical designs.

    Attributes:
        design_space: Finite set of physical designs.
        event_shape_design: Shape of one physical design.
        dim_design: Flattened design dimension.
        normalize_design_history: Whether design histories are normalized.
        design_history_bounds: Target bounds for normalized design histories.
        actor: Module that produces one logit per categorical design.
        design_values: Physical designs indexed by categorical design.
        design_low: Per-coordinate minimum physical design values.
        design_high: Per-coordinate maximum physical design values.
        outcome_low: Lower bound used to normalize outcome histories.
        outcome_high: Upper bound used to normalize outcome histories.
    """

    raw_designs_are_physical = True

    def __init__(
        self,
        design_space: FiniteDesignSpace,
        actor: nn.Module | None = None,
        *,
        encoder: HistoryEncoder | None = None,
        head: CategoricalHead | None = None,
        event_shape_y: Sequence[int] | None = None,
        normalize_design_history: bool = False,
        design_history_bounds: tuple[float, float] = (0.0, 1.0),
        outcome_bounds: tuple[float, float] | None = None,
    ):
        """Initializes the categorical policy.

        Args:
            design_space: Finite physical designs returned by policy proposals.
            actor: Optional module returning one logit per action.
            encoder: Optional history encoder used to build a default
                categorical actor. Cannot be combined with ``actor``.
            head: Optional categorical policy head paired with ``encoder``.
                If omitted, a default head is created for the encoder.
            event_shape_y: Shape of one outcome event. Required when creating
                the default actor without an explicit ``encoder``.
            normalize_design_history: Whether to rescale design histories.
            design_history_bounds: Bounds used when design histories are rescaled.
            outcome_bounds: Optional bounds used to rescale outcomes to [0, 1].

        Raises:
            ValueError: If history or outcome bounds are invalid.
        """
        super().__init__()
        self.design_space = design_space
        self.event_shape_design = torch.Size(design_space.event_shape)
        self.dim_design = self.event_shape_design.numel()
        self.normalize_design_history = bool(normalize_design_history)
        self.design_history_bounds = tuple(float(x) for x in design_history_bounds)
        if self.design_history_bounds[0] >= self.design_history_bounds[1]:
            raise ValueError("design_history_bounds must satisfy low < high.")
        if actor is not None and (encoder is not None or head is not None):
            raise ValueError("actor cannot be combined with encoder or head.")
        if actor is not None:
            self.actor = actor
        elif encoder is not None:
            self.actor = CategoricalPolicyActor(
                encoder,
                head or CategoricalHead(encoder.dim_emb, design_space.n_designs),
            )
        elif head is not None:
            raise ValueError("head requires an encoder.")
        else:
            if event_shape_y is None:
                raise ValueError(
                    "event_shape_y is required when creating the default actor."
                )
            self.actor = DeepSetCategoricalActor(
                event_shape_y=event_shape_y,
                event_shape_design=self.event_shape_design,
                n_designs=design_space.n_designs,
            )

        self.register_buffer("design_values", design_space.values.clone())
        values = self.design_values.reshape(design_space.n_designs, self.dim_design)
        if not torch.is_floating_point(values):
            values = values.to(torch.get_default_dtype())
        self.register_buffer("design_low", values.amin(dim=0))
        self.register_buffer("design_high", values.amax(dim=0))
        if outcome_bounds is None:
            outcome_low = torch.tensor(float("nan"), dtype=values.dtype)
            outcome_high = torch.tensor(float("nan"), dtype=values.dtype)
        else:
            outcome_low, outcome_high = (
                torch.as_tensor(bound, dtype=values.dtype) for bound in outcome_bounds
            )
            if torch.any(outcome_low >= outcome_high):
                raise ValueError("outcome_bounds must satisfy low < high.")
        self.register_buffer("outcome_low", outcome_low)
        self.register_buffer("outcome_high", outcome_high)
        self._normalize_outcomes = outcome_bounds is not None

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
        history = self._prepare_histories(y_history, design_history, mask=mask)
        logits = self._validated_logits(
            history.y_flat, history.design_flat, history.mask_flat
        )
        design_index = logits.argmax(dim=-1)
        return self.design_values[design_index].reshape(
            *history.batch_shape, *self.event_shape_design
        )

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
        history = self._prepare_histories(y_history, design_history, mask=mask)
        logits = self._validated_logits(
            history.y_flat, history.design_flat, history.mask_flat
        )
        distribution = Categorical(logits=logits)
        design_index = distribution.sample()
        deterministic_design_index = logits.argmax(dim=-1)
        design = self.design_values[design_index].reshape(
            *history.batch_shape, *self.event_shape_design
        )
        deterministic_design = self.design_values[deterministic_design_index].reshape(
            *history.batch_shape, *self.event_shape_design
        )
        return DesignProposal(
            raw_design=design,
            log_prob=distribution.log_prob(design_index).reshape(history.batch_shape),
            deterministic_raw_design=deterministic_design,
            info={"design_index": design_index.reshape(history.batch_shape)},
        )

    def _validated_logits(
        self, y_flat: Tensor, design_flat: Tensor, mask_flat: Tensor | None
    ) -> Tensor:
        """Returns validated categorical logits in flattened actor coordinates."""
        logits = self.actor(y_flat, design_flat, mask=mask_flat)
        expected_shape = (design_flat.shape[0], self.design_space.n_designs)
        if logits.shape != expected_shape:
            raise ValueError(
                "actor logits must have shape "
                f"{expected_shape}; got {tuple(logits.shape)}."
            )
        return logits

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
            Logits with shape ``[batch, n_designs]`` after flattening batches.
        """
        history = self._prepare_histories(y_history, design_history, mask=mask)
        return self._validated_logits(
            history.y_flat, history.design_flat, history.mask_flat
        )

    def logits_from_observation(self, obs) -> Tensor:
        """Returns categorical logits for a ``History`` observation.

        Args:
            obs: History containing outcomes, designs, and an optional mask.

        Returns:
            Logits with shape ``[batch, n_designs]``.
        """
        return self.logits(obs.outcomes, obs.designs, mask=obs.mask)

    def design_probabilities_from_observation(self, obs) -> tuple[Tensor, Tensor]:
        """Returns exact categorical probabilities and log probabilities.

        Args:
            obs: History containing outcomes, designs, and an optional mask.

        Returns:
            A ``(probabilities, log_probabilities)`` tuple, each with shape
                ``[batch, n_designs]``.
        """
        log_probs = torch.log_softmax(self.logits_from_observation(obs), dim=-1)
        return log_probs.exp(), log_probs


def make_rlboed_policy(
    *,
    task,
    box_low,
    box_high,
    dim_h,
    dim_emb: int,
    log_std_min: float,
    log_std_max: float,
    init_std: float,
    normalize_design_history: bool,
    design_history_bounds,
    outcome_bounds,
    event_shape_y,
    event_shape_design,
    event_shape_out=None,
) -> StochasticDesignNet:
    """Builds the complete RLBOED policy described by a run configuration.

    A task-declared finite or box space takes precedence. Tasks carrying the
    unconstrained default space use the configured RLBOED box instead. Runtime
    event shapes are accepted for compatibility with the common policy loading
    path and checked against the task used to resolve the design space.
    """
    task_event_shape_y = torch.Size(task.event_shape_y)
    task_event_shape_design = torch.Size(task.event_shape_design)
    runtime_event_shape_y = torch.Size(event_shape_y)
    runtime_event_shape_design = torch.Size(event_shape_design)
    if runtime_event_shape_y != task_event_shape_y:
        raise ValueError(
            "event_shape_y must match the configured task; "
            f"expected {tuple(task_event_shape_y)}, got "
            f"{tuple(runtime_event_shape_y)}."
        )
    if runtime_event_shape_design != task_event_shape_design:
        raise ValueError(
            "event_shape_design must match the configured task; "
            f"expected {tuple(task_event_shape_design)}, got "
            f"{tuple(runtime_event_shape_design)}."
        )
    if (
        event_shape_out is not None
        and torch.Size(event_shape_out) != task_event_shape_design
    ):
        raise ValueError(
            "event_shape_out must match the configured task's design shape; "
            f"expected {tuple(task_event_shape_design)}, got "
            f"{tuple(torch.Size(event_shape_out))}."
        )

    if isinstance(task.design_space, DefaultDesignSpace):
        design_space = BoxDesignSpace(
            box_low,
            box_high,
            event_shape=task_event_shape_design,
        )
    else:
        design_space = task.design_space

    common_kwargs = {
        "normalize_design_history": normalize_design_history,
        "design_history_bounds": tuple(design_history_bounds),
        "outcome_bounds": tuple(outcome_bounds),
    }
    if isinstance(design_space, BoxDesignSpace):
        actor = DeepSetGaussianActor(
            event_shape_y=task_event_shape_y,
            event_shape_design=task_event_shape_design,
            dim_h=dim_h,
            dim_emb=dim_emb,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            init_std=init_std,
        )
        return GaussianPolicyNet(design_space, actor=actor, **common_kwargs)
    if isinstance(design_space, FiniteDesignSpace):
        actor = DeepSetCategoricalActor(
            event_shape_y=task_event_shape_y,
            event_shape_design=task_event_shape_design,
            n_designs=design_space.n_designs,
            dim_h=dim_h,
            dim_emb=dim_emb,
        )
        return CategoricalPolicyNet(design_space, actor=actor, **common_kwargs)
    raise TypeError(
        "RLBOED requires the configured task to resolve to a BoxDesignSpace "
        "or FiniteDesignSpace."
    )
