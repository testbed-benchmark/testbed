"""Stochastic policy networks for reinforcement learning BED agents."""

from dataclasses import dataclass
from math import prod

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from testbed.design_nets.base_design_net import DesignNet
from testbed.design_nets.modules import MLP
from testbed.envs.design_spaces import BoxDesignSpace


@dataclass
class PolicySample:
    """Sampled action and log-probability from a stochastic policy.

    Attributes:
        action: Sampled bounded design with shape
            [*batch_shape, *event_shape_design].
        log_prob: Summed action log probability with shape [*batch_shape].
        deterministic_action: Bounded deterministic action, typically the squashed
            policy mean, with shape [*batch_shape, *event_shape_design].
        raw_action: Optional unsquashed Gaussian action with shape
            [*batch_shape, *event_shape_design].
    """

    action: Tensor
    log_prob: Tensor
    deterministic_action: Tensor
    raw_action: Tensor | None = None


class StochasticDesignNet(DesignNet):
    """Design network with an RL-friendly stochastic action API."""

    def sample(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicySample:
        """Samples an action and returns its policy log probability.

        Args:
            y_history: Outcome history with shape
                [*batch_shape, T, *event_shape_y].
            design_history: Design history with shape
                [*batch_shape, T, *event_shape_design].
            mask: Optional boolean mask for padded histories with shape
                [*batch_shape, T].
            deterministic: If True, return the deterministic policy action instead
                of a random sample.

        Returns:
            A policy sample containing the action, log probability, deterministic
            action, and optional raw action.
        """
        raise NotImplementedError

    def sample_from_observation(self, obs, *, deterministic: bool = False):
        """Samples from a padded-history observation dictionary.

        Args:
            obs: Observation dictionary produced by `PaddedHistoryObservation`,
                containing `outcomes`, `designs`, and optionally `mask`.
            deterministic: If True, return the deterministic policy action instead
                of a random sample.

        Returns:
            A policy sample for the observation.
        """
        return self.sample(
            obs["outcomes"],
            obs["designs"],
            mask=obs.get("mask"),
            deterministic=deterministic,
        )


class TanhGaussianPolicyNet(StochasticDesignNet):
    """Tanh-squashed Gaussian policy for continuous box design spaces."""

    def __init__(
        self,
        design_space: BoxDesignSpace,
        dim_y: int = 1,
        dim_h: int = 128,
        dim_emb: int = 64,
        *,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        """Initializes the policy network.

        Args:
            design_space: Continuous box design space used to bound actions.
            dim_y: Flattened outcome dimension.
            dim_h: Hidden layer width.
            dim_emb: History summary dimension.
            log_std_min: Lower bound for Gaussian log standard deviations.
            log_std_max: Upper bound for Gaussian log standard deviations.
        """
        if not isinstance(design_space, BoxDesignSpace):
            raise TypeError("TanhGaussianPolicyNet requires a BoxDesignSpace.")
        super().__init__()

        self.design_space = design_space
        self.event_shape_design = torch.Size(design_space.event_shape)
        self.dim_design = self.event_shape_design.numel()
        self.dim_y = int(dim_y)
        self.dim_emb = int(dim_emb)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.encoder_net = MLP(
            [self.dim_design + self.dim_y, dim_h, dim_h, dim_emb],
            activation=nn.ReLU,
        )
        self.actor_net = MLP([dim_emb, dim_h, dim_h], activation=nn.ReLU)
        self.mean_head = nn.Linear(dim_h, self.dim_design)
        self.log_std_head = nn.Linear(dim_h, self.dim_design)

        low = design_space.low.reshape(self.dim_design)
        high = design_space.high.reshape(self.dim_design)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, y_history, design_history) -> Tensor:
        """Returns the deterministic bounded action for a history.

        Args:
            y_history: Outcome history with shape
                [*batch_shape, T, *event_shape_y].
            design_history: Design history with shape
                [*batch_shape, T, *event_shape_design].

        Returns:
            Deterministic bounded design with shape
            [*batch_shape, *event_shape_design].
        """
        return self.sample(y_history, design_history, deterministic=True).action

    def sample(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicySample:
        """Samples a tanh-squashed action and computes its log probability.

        Args:
            y_history: Outcome history with shape
                [*batch_shape, T, *event_shape_y].
            design_history: Design history with shape
                [*batch_shape, T, *event_shape_design].
            mask: Optional boolean tensor for padded histories with shape
                [*batch_shape, T].
            deterministic: If True, use the Gaussian mean instead of drawing a
                reparameterized sample.

        Returns:
            A policy sample containing a bounded action, its corrected log
            probability, the deterministic bounded action, and the unsquashed
            Gaussian action.
        """
        batch_shape, summary = self._summarize_history(
            y_history, design_history, mask=mask
        )
        mean, log_std = self._actor_params(summary)
        std = log_std.exp()
        normal, action_dist = self._action_distribution(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = normal.rsample()

        action_flat = self._transform_raw_action(raw_action, action_dist)
        deterministic_flat = self._transform_raw_action(mean, action_dist)
        log_prob = action_dist.log_prob(action_flat).sum(dim=-1)

        action = action_flat.reshape(*batch_shape, *self.event_shape_design)
        deterministic_action = deterministic_flat.reshape(
            *batch_shape, *self.event_shape_design
        )
        raw_action = raw_action.reshape(*batch_shape, *self.event_shape_design)

        return PolicySample(
            action=action,
            log_prob=log_prob.reshape(batch_shape),
            deterministic_action=deterministic_action,
            raw_action=raw_action,
        )

    def _action_distribution(
        self, mean: Tensor, std: Tensor
    ) -> tuple[Normal, TransformedDistribution]:
        """Builds the Gaussian base distribution and bounded action distribution."""
        normal = Normal(mean, std)
        action_dist = TransformedDistribution(
            normal,
            [
                TanhTransform(cache_size=1),
                AffineTransform(loc=self.action_bias, scale=self.action_scale),
            ],
        )
        return normal, action_dist

    @staticmethod
    def _transform_raw_action(raw_action: Tensor, action_dist) -> Tensor:
        """Applies the action distribution transforms to an unsquashed sample."""
        action = raw_action
        for transform in action_dist.transforms:
            action = transform(action)
        return action

    def _summarize_history(
        self, y_history, design_history, *, mask: Tensor | None = None
    ) -> tuple[torch.Size, Tensor]:
        """Encodes a possibly padded history with masked sum pooling.

        Args:
            y_history: Outcome history with shape
                [*batch_shape, T, *event_shape_y].
            design_history: Design history with shape
                [*batch_shape, T, *event_shape_design].
            mask: Optional boolean tensor for padded histories with shape
                [*batch_shape, T].

        Returns:
            A tuple containing the batch shape and encoded history summary with
            shape [prod(batch_shape), dim_emb].
        """
        dtype = self.action_scale.dtype
        y_history = torch.as_tensor(y_history, device=self.device, dtype=dtype)
        design_history = torch.as_tensor(
            design_history, device=self.device, dtype=dtype
        )

        design_event_ndim = len(self.event_shape_design)
        design_t_dim = design_history.ndim - design_event_ndim - 1
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

        b_flat = prod(batch_shape) if batch_shape else 1
        y_flat = y_history.reshape(b_flat, T, self.dim_y)
        design_flat = design_history.reshape(b_flat, T, self.dim_design)

        if T == 0:
            summary = torch.zeros(
                b_flat,
                self.dim_emb,
                device=y_flat.device,
                dtype=y_flat.dtype,
            )
            return batch_shape, summary

        enc_input = torch.cat([design_flat, y_flat], dim=-1)
        enc = self.encoder_net(enc_input)

        if mask is not None:
            mask = torch.as_tensor(mask, device=enc.device, dtype=enc.dtype)
            mask = mask.reshape(b_flat, T, 1)
            enc = enc * mask

        return batch_shape, enc.sum(dim=1)

    def _actor_params(self, summary: Tensor) -> tuple[Tensor, Tensor]:
        """Computes Gaussian mean and bounded log standard deviation.

        Args:
            summary: Encoded history summary with shape [batch, dim_emb].

        Returns:
            A tuple containing the Gaussian mean and log standard deviation, each
            with shape [batch, dim_design].
        """
        hidden = self.actor_net(summary)
        mean = self.mean_head(hidden)
        log_std = torch.tanh(self.log_std_head(hidden))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        return mean, log_std
