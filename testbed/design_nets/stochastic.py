"""Stochastic policy networks for reinforcement learning BED agents."""

from dataclasses import dataclass
from math import atanh, log, prod

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from testbed.design_nets.base_design_net import DesignNet
from testbed.design_nets.modules import MLP, _initialize_linear_layers, affine_rescale
from testbed.envs.design_spaces import BoxDesignSpace


@dataclass
class PolicySample:
    """Sampled design and log-probability from a stochastic policy.

    Attributes:
        design: Sampled bounded design with shape
            ``[*batch_shape, *event_shape_design]``.
        log_prob: Summed design log probability with shape ``[*batch_shape]``.
        deterministic_design: Bounded deterministic design, typically the squashed
            policy mean, with shape ``[*batch_shape, *event_shape_design]``.
        raw_design: Optional unsquashed Gaussian design with shape
            ``[*batch_shape, *event_shape_design]``.
    """

    design: Tensor
    log_prob: Tensor
    deterministic_design: Tensor
    raw_design: Tensor | None = None


class StochasticDesignNet(DesignNet):
    """Design network with an RL-friendly stochastic design API."""

    def sample(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicySample:
        """Samples a design and returns its policy log probability.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional boolean mask for padded histories with shape
                ``[*batch_shape, T]``.
            deterministic: If True, return the deterministic design instead
                of a random sample.

        Returns:
            A policy sample containing the design, log probability, deterministic
            design, and optional raw design.
        """
        raise NotImplementedError

    def sample_from_observation(self, obs, *, deterministic: bool = False):
        """Samples from a ``History`` observation.

        Args:
            obs: ``History`` returned by ``BEDEnv.reset()`` or ``BEDEnv.step()``.
            deterministic: If True, return the deterministic design instead
                of a random sample.

        Returns:
            A policy sample for the observation.
        """
        return self.sample(
            obs.outcomes,
            obs.designs,
            mask=obs.mask,
            deterministic=deterministic,
        )


class GaussianPolicyActor(nn.Module):
    """Maps flattened experimental histories to Gaussian policy parameters."""

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
        raise NotImplementedError


class DeepSetGaussianActor(GaussianPolicyActor):
    """DeepSets-style Gaussian actor using masked sum pooling over history."""

    def __init__(
        self,
        *,
        dim_design: int,
        dim_y: int = 1,
        dim_h: int = 128,
        dim_emb: int = 64,
        log_std_min: float = -20.0,
        log_std_max: float = 0.0,
        init_std: float = 3**-0.5,
    ):
        """Initializes the actor network.

        Args:
            dim_y: Flattened outcome dimension.
            dim_design: Flattened design dimension.
            dim_h: Hidden layer width.
            dim_emb: History summary dimension.
            log_std_min: Lower bound for Gaussian log standard deviations.
            log_std_max: Upper bound for Gaussian log standard deviations.
            init_std: Initial unsquashed Gaussian standard deviation.
        """
        super().__init__()

        self.dim_y = int(dim_y)
        self.dim_design = int(dim_design)
        self.dim_emb = int(dim_emb)
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

        self.encoder_net = MLP(
            [self.dim_design + self.dim_y, dim_h, dim_h, dim_emb],
            activation=nn.ReLU,
        )
        self.actor_net = MLP([dim_emb, dim_h, dim_h], activation=nn.ReLU)
        self.mean_head = nn.Linear(dim_h, self.dim_design)
        self.log_std_head = nn.Linear(dim_h, self.dim_design)
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

    def forward(
        self, y_history: Tensor, design_history: Tensor, *, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Computes Gaussian parameters from flattened histories."""
        b_flat, T, _ = design_history.shape

        if T == 0:
            summary = torch.zeros(
                b_flat,
                self.dim_emb,
                device=design_history.device,
                dtype=design_history.dtype,
            )
        else:
            enc_input = torch.cat([y_history, design_history], dim=-1)
            enc = self.encoder_net(enc_input)

            if mask is not None:
                enc = enc * mask.to(dtype=enc.dtype).unsqueeze(-1)

            summary = enc.sum(dim=1)

        hidden = self.actor_net(summary)
        mean = self.mean_head(hidden)
        log_std = torch.tanh(self.log_std_head(hidden))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        return mean, log_std


class TanhGaussianPolicyNet(StochasticDesignNet):
    """Tanh-squashed Gaussian policy for continuous box design spaces."""

    def __init__(
        self,
        design_space: BoxDesignSpace,
        actor: GaussianPolicyActor | None = None,
        *,
        normalize_design_history: bool = False,
        design_history_bounds: tuple[float, float] = (-1.0, 1.0),
        outcome_bounds: tuple[float, float] | None = None,
    ):
        """Initializes the policy network.

        Args:
            design_space: Physical continuous design space. Public policy methods
                return designs in these coordinates.
            actor: Optional module mapping flattened histories to Gaussian
                mean and log standard deviation.
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
        self.actor = actor or DeepSetGaussianActor(
            dim_design=self.dim_design,
        )

        low = design_space.low.reshape(self.dim_design)
        high = design_space.high.reshape(self.dim_design)
        if torch.any(low >= high):
            raise ValueError(
                "TanhGaussianPolicyNet requires finite-width design bounds."
            )
        self.register_buffer("design_scale", (high - low) / 2.0)
        self.register_buffer("design_bias", (high + low) / 2.0)

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

    def forward(self, y_history, design_history) -> Tensor:
        """Returns the deterministic bounded design for a history.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.

        Returns:
            Deterministic bounded design with shape
            ``[*batch_shape, *event_shape_design]``.
        """
        return self.sample(y_history, design_history, deterministic=True).design

    def sample(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicySample:
        """Samples a tanh-squashed design and computes its log probability.

        Args:
            y_history: Outcome history with shape
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Design history with shape
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional boolean tensor for padded histories with shape
                ``[*batch_shape, T]``.
            deterministic: If True, use the Gaussian mean instead of drawing a
                reparameterized sample.

        Returns:
            A policy sample containing a bounded design, its corrected log
            probability, the deterministic bounded design, and the unsquashed
            Gaussian design.
        """
        batch_shape, y_flat, design_flat, mask_flat = self._prepare_histories(
            y_history,
            design_history,
            mask=mask,
        )
        return self._sample_prepared(
            batch_shape,
            y_flat,
            design_flat,
            mask_flat,
            deterministic=deterministic,
        )

    def _sample_prepared(
        self,
        batch_shape: torch.Size,
        y_flat: Tensor,
        design_flat: Tensor,
        mask_flat: Tensor | None,
        *,
        deterministic: bool,
    ) -> PolicySample:
        """Samples from validated histories in actor coordinates."""
        mean, log_std = self.actor(y_flat, design_flat, mask=mask_flat)
        self._validate_actor_params(mean, log_std, batch_size=design_flat.shape[0])
        std = log_std.exp()
        normal = Normal(mean, std)

        if deterministic:
            raw_design = mean
        else:
            raw_design = normal.rsample()

        squashed_flat = self._squash(raw_design)
        deterministic_flat = self._squash(mean)
        log_prob = (normal.log_prob(raw_design) - self._squash_log_det(raw_design)).sum(
            dim=-1
        )

        design = squashed_flat.reshape(*batch_shape, *self.event_shape_design)
        deterministic_design = deterministic_flat.reshape(
            *batch_shape, *self.event_shape_design
        )
        raw_design = raw_design.reshape(*batch_shape, *self.event_shape_design)

        return PolicySample(
            design=design,
            log_prob=log_prob.reshape(batch_shape),
            deterministic_design=deterministic_design,
            raw_design=raw_design,
        )

    def _squash(self, raw_design: Tensor) -> Tensor:
        """Maps an unsquashed Gaussian sample into the bounded design box."""
        return self.design_bias + self.design_scale * torch.tanh(raw_design)

    def _squash_log_det(self, raw_design: Tensor) -> Tensor:
        """Log absolute determinant of the tanh-then-affine squash, per dimension.

        Computed in the forward direction to stay finite for any finite
        ``raw_design``; inverting via ``atanh`` would diverge once ``tanh``
        saturates to +/-1 in floating point.
        """
        # log(1 - tanh(x)^2) in a numerically stable form (SpinUp / Denis Yarats).
        tanh_log_det = 2.0 * (
            log(2.0) - raw_design - nn.functional.softplus(-2.0 * raw_design)
        )
        return torch.log(self.design_scale) + tanh_log_det

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
        dtype = self.design_scale.dtype
        device = self.design_scale.device
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
                self.design_bias - self.design_scale,
                self.design_bias + self.design_scale,
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
