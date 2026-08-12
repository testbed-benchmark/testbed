"""Stochastic policy networks for reinforcement learning BED agents."""

from math import atanh, log, prod

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, Normal

from testbed.core.design_spaces import BoxDesignSpace, FiniteDesignSpace
from testbed.design_nets.base_design_net import DesignNet, DesignProposal
from testbed.design_nets.encoder import DeepSetEncoder, HistoryEncoder
from testbed.design_nets.modules import MLP, _initialize_linear_layers, affine_rescale


class StochasticDesignNet(DesignNet):
    """Design network that samples raw proposals with policy log probabilities."""

    produces_policy_log_probs = True
    raw_designs_are_physical = False


class GaussianHead(nn.Module):
    """Maps a history encoding to Gaussian policy parameters."""

    def __init__(
        self,
        dim_emb: int,
        dim_design: int,
        dim_h=[128, 128],
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
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        return mean, log_std


class CategoricalHead(nn.Module):
    """Maps a history encoding to categorical design logits."""

    def __init__(self, dim_emb: int, n_designs: int, dim_h=[128, 128]):
        """Initializes the categorical readout."""
        super().__init__()
        self.dim_emb = int(dim_emb)
        self.n_designs = int(n_designs)
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.actor_net = MLP([self.dim_emb] + hidden_widths, activation=nn.ReLU)
        self.logits_head = nn.Linear(hidden_widths[-1], self.n_designs)
        _initialize_linear_layers(self)

    def forward(self, encoding: Tensor) -> Tensor:
        """Returns one categorical logit per design."""
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
        return self.head(self.encoder(y_history, design_history, mask=mask))


class DeepSetGaussianActor(GaussianPolicyActor):
    """Preset: ``DeepSetEncoder`` followed by a ``GaussianHead``."""

    def __init__(
        self,
        *,
        dim_design: int,
        dim_y: int = 1,
        dim_h=[128, 128],
        dim_emb: int = 64,
        log_std_min: float = -20.0,
        log_std_max: float = 0.0,
        init_std: float = 3**-0.5,
    ):
        """Initializes the actor network.

        Args:
            dim_y: Flattened outcome dimension.
            dim_design: Flattened design dimension.
            dim_h: Per-layer hidden widths as a list; an int is a single layer.
            dim_emb: History summary dimension.
            log_std_min: Lower bound for Gaussian log standard deviations.
            log_std_max: Upper bound for Gaussian log standard deviations.
            init_std: Initial unsquashed Gaussian standard deviation.
        """
        self.dim_y = int(dim_y)
        encoder = DeepSetEncoder(
            (self.dim_y,),
            (int(dim_design),),
            dim_h=dim_h,
            dim_emb=dim_emb,
        )
        head = GaussianHead(
            encoder.dim_emb,
            dim_design,
            dim_h=dim_h,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            init_std=init_std,
        )
        super().__init__(encoder, head)


class CategoricalPolicyActor(nn.Module):
    """History encoder followed by a categorical policy head."""

    def __init__(self, encoder: HistoryEncoder, head: CategoricalHead):
        """Initializes an encoder and categorical head pair."""
        super().__init__()
        if encoder.dim_emb != head.dim_emb:
            raise ValueError("encoder and head embedding dimensions must match.")
        self.encoder = encoder
        self.head = head

    def forward(
        self, y_history: Tensor, design_history: Tensor, *, mask: Tensor | None = None
    ) -> Tensor:
        """Computes design logits from a history."""
        return self.head(self.encoder(y_history, design_history, mask=mask))


class DeepSetCategoricalActor(CategoricalPolicyActor):
    """DeepSets actor that returns categorical design logits.

    Attributes:
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
        dim_design: int,
        n_designs: int,
        dim_y: int = 1,
        dim_h=[128, 128],
        dim_emb: int = 64,
    ):
        """Initializes the actor.

        Args:
            dim_design: Flattened design dimension.
            n_designs: Number of categorical designs.
            dim_y: Flattened outcome dimension.
            dim_h: Per-layer hidden widths as a list; an int is a single layer.
            dim_emb: History summary dimension.
        """
        self.dim_design = int(dim_design)
        self.dim_y = int(dim_y)
        encoder = DeepSetEncoder(
            (self.dim_y,),
            (self.dim_design,),
            dim_h=dim_h,
            dim_emb=dim_emb,
        )
        head = CategoricalHead(encoder.dim_emb, n_designs, dim_h=dim_h)
        super().__init__(encoder, head)


class TanhGaussianPolicyNet(StochasticDesignNet):
    """Gaussian policy for continuous box design spaces.

    The network proposes values in unconstrained Gaussian coordinates. The
    caller's design space owns the constraint transform and density correction.
    """

    supports_reparameterized_proposals = True

    def __init__(
        self,
        design_space: BoxDesignSpace,
        actor: GaussianPolicyActor | None = None,
        *,
        encoder: HistoryEncoder | None = None,
        head: GaussianHead | None = None,
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
            self.actor = DeepSetGaussianActor(dim_design=self.dim_design)

        low = design_space.low.reshape(self.dim_design)
        high = design_space.high.reshape(self.dim_design)
        if torch.any(low >= high):
            raise ValueError(
                "TanhGaussianPolicyNet requires finite-width design bounds."
            )
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
        batch_shape, y_flat, design_flat, mask_flat = self._prepare_histories(
            y_history,
            design_history,
            mask=mask,
        )
        mean, _ = self._normal_params(y_flat, design_flat, mask_flat)
        return mean.reshape(*batch_shape, *self.event_shape_design)

    def propose(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        reparameterize_policy: bool = False,
    ) -> DesignProposal:
        """Samples an unconstrained Gaussian proposal and its raw density."""
        batch_shape, y_flat, design_flat, mask_flat = self._prepare_histories(
            y_history,
            design_history,
            mask=mask,
        )
        mean, log_std = self._normal_params(y_flat, design_flat, mask_flat)
        normal = Normal(mean, log_std.exp())
        if reparameterize_policy:
            raw_design = normal.rsample()
        else:
            raw_design = normal.sample()

        log_prob = normal.log_prob(raw_design).sum(dim=-1)
        raw_design = raw_design.reshape(*batch_shape, *self.event_shape_design)
        deterministic_raw_design = mean.reshape(*batch_shape, *self.event_shape_design)

        return DesignProposal(
            raw_design=raw_design,
            log_prob=log_prob.reshape(batch_shape),
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

    def _prepare_histories(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
    ) -> tuple[torch.Size, Tensor, Tensor, Tensor | None]:
        """Validates histories and flattens arbitrary batch shapes.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional boolean tensor for padded histories with shape
                ``[*batch_shape, T]``.

        Returns:
            The original batch shape, flattened outcome history, flattened design
            history, and optional flattened mask.
        """
        dtype = self.design_low.dtype
        device = self.design_low.device
        y_history = torch.as_tensor(y_history, device=device, dtype=dtype)
        design_history = torch.as_tensor(design_history, device=device, dtype=dtype)

        design_event_ndim = len(self.event_shape_design)
        design_t_dim = design_history.ndim - design_event_ndim - 1
        if design_t_dim < 0:
            raise ValueError(
                "design_history must have shape "
                f"[*batch_shape, T, {tuple(self.event_shape_design)}]."
            )

        batch_shape = torch.Size(design_history.shape[:design_t_dim])
        T = design_history.shape[design_t_dim]

        expected_design_suffix = (T, *self.event_shape_design)
        if (
            design_history.shape[-len(expected_design_suffix) :]
            != expected_design_suffix
        ):
            raise ValueError(
                "design_history must have shape "
                f"[*batch_shape, T, {tuple(self.event_shape_design)}]."
            )

        if (
            y_history.ndim < len(batch_shape) + 2
            or y_history.shape[: len(batch_shape)] != batch_shape
            or y_history.shape[len(batch_shape)] != T
        ):
            raise ValueError("y_history must have shape [*batch_shape, T, dim_y].")

        b_flat = prod(batch_shape) if batch_shape else 1
        dim_y = prod(y_history.shape[len(batch_shape) + 1 :])
        y_flat = y_history.reshape(b_flat, T, dim_y)
        design_flat = design_history.reshape(b_flat, T, self.dim_design)

        if self.normalize_design_history:
            target_low, target_high = self.design_history_bounds
            design_flat = affine_rescale(
                design_flat,
                self.design_low,
                self.design_high,
                target_low,
                target_high,
            )
        if self._normalize_outcomes:
            y_flat = affine_rescale(
                y_flat, self.outcome_low, self.outcome_high, 0.0, 1.0
            )

        mask_flat = None
        if mask is not None:
            mask = torch.as_tensor(mask, device=device)
            expected_mask_shape = (*batch_shape, T)
            if mask.shape != expected_mask_shape:
                raise ValueError(
                    "mask must have shape "
                    f"[*batch_shape, T]; expected {expected_mask_shape}."
                )
            mask_flat = mask.reshape(b_flat, T)

        return batch_shape, y_flat, design_flat, mask_flat

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
        actor: CategoricalPolicyActor | None = None,
        *,
        encoder: HistoryEncoder | None = None,
        head: CategoricalHead | None = None,
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
            normalize_design_history: Whether to rescale design histories.
            design_history_bounds: Bounds used when design histories are rescaled.
            outcome_bounds: Optional bounds used to rescale outcomes to [0, 1].

        Raises:
            ValueError: If history or outcome bounds are invalid, or the actor
                has a different number of actions than ``design_space``.
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
            self.actor = DeepSetCategoricalActor(
                dim_design=self.dim_design,
                n_designs=design_space.n_designs,
            )
        if self.actor.head.n_designs != design_space.n_designs:
            raise ValueError("actor n_designs must match the categorical design space.")

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
        batch_shape, y_flat, design_flat, mask_flat = self._prepare_histories(
            y_history, design_history, mask=mask
        )
        logits = self._validated_logits(y_flat, design_flat, mask_flat)
        design_index = logits.argmax(dim=-1)
        return self.design_values[design_index].reshape(
            *batch_shape, *self.event_shape_design
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
        batch_shape, y_flat, design_flat, mask_flat = self._prepare_histories(
            y_history, design_history, mask=mask
        )
        logits = self._validated_logits(y_flat, design_flat, mask_flat)
        distribution = Categorical(logits=logits)
        design_index = distribution.sample()
        deterministic_design_index = logits.argmax(dim=-1)
        design = self.design_values[design_index].reshape(
            *batch_shape, *self.event_shape_design
        )
        deterministic_design = self.design_values[deterministic_design_index].reshape(
            *batch_shape, *self.event_shape_design
        )
        return DesignProposal(
            raw_design=design,
            log_prob=distribution.log_prob(design_index).reshape(batch_shape),
            deterministic_raw_design=deterministic_design,
            info={"design_index": design_index.reshape(batch_shape)},
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
        _, y_flat, design_flat, mask_flat = self._prepare_histories(
            y_history, design_history, mask=mask
        )
        return self.actor(y_flat, design_flat, mask=mask_flat)

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

    def _prepare_histories(
        self, y_history, design_history, *, mask: Tensor | None = None
    ) -> tuple[torch.Size, Tensor, Tensor, Tensor | None]:
        """Validates, flattens, and optionally normalizes histories.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional padding mask with shape ``[*batch_shape, T]``.

        Returns:
            The batch shape, flattened outcome history, flattened design
            history, and flattened mask.

        Raises:
            ValueError: If a history or mask has an incompatible shape.
        """
        dtype = self.design_low.dtype
        device = self.design_low.device
        y_history = torch.as_tensor(y_history, device=device, dtype=dtype)
        design_history = torch.as_tensor(design_history, device=device, dtype=dtype)
        event_ndim = len(self.event_shape_design)
        t_dim = design_history.ndim - event_ndim - 1
        if t_dim < 0:
            raise ValueError(
                "design_history must have shape [*batch_shape, T, *event_shape]."
            )
        batch_shape = torch.Size(design_history.shape[:t_dim])
        history_length = design_history.shape[t_dim]
        if design_history.shape[-(event_ndim + 1) :] != (
            history_length,
            *self.event_shape_design,
        ):
            raise ValueError("design_history has the wrong shape.")
        if (
            y_history.ndim < len(batch_shape) + 2
            or y_history.shape[: len(batch_shape)] != batch_shape
            or y_history.shape[len(batch_shape)] != history_length
        ):
            raise ValueError("y_history must have shape [*batch_shape, T, dim_y].")
        batch_size = prod(batch_shape) if batch_shape else 1
        dim_y = prod(y_history.shape[len(batch_shape) + 1 :])
        y_flat = y_history.reshape(batch_size, history_length, dim_y)
        design_flat = design_history.reshape(
            batch_size, history_length, self.dim_design
        )
        if self.normalize_design_history:
            low, high = self.design_history_bounds
            width = self.design_high - self.design_low
            normalized = low + (design_flat - self.design_low) * (high - low) / width
            design_flat = torch.where(
                width == 0, torch.zeros_like(normalized), normalized
            )
        if self._normalize_outcomes:
            y_flat = affine_rescale(
                y_flat, self.outcome_low, self.outcome_high, 0.0, 1.0
            )
        mask_flat = None
        if mask is not None:
            mask = torch.as_tensor(mask, device=device)
            if mask.shape != (*batch_shape, history_length):
                raise ValueError("mask must have shape [*batch_shape, T].")
            mask_flat = mask.reshape(batch_size, history_length)
        return batch_shape, y_flat, design_flat, mask_flat
