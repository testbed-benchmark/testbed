"""Analytic and hand-designed policies only; no neural imports or initialization."""

from dataclasses import asdict, dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Policy:
    kind: str
    beta: float = 0.0
    k: int = 1
    S: int = 1
    h: str = "mean"
    kappa: float = 0.0

    def validate(self, config):
        if self.kind not in ("ucb", "nonadaptive", "explore_certify", "eig", "random"):
            raise ValueError(f"Unknown non-neural policy {self.kind}")
        if not 1 <= self.k <= config.d or self.h not in ("mean", "margin"):
            raise ValueError("Invalid subset or leader rule")
        if self.kind == "explore_certify" and not self.k <= self.S < config.T:
            raise ValueError("Explore-then-certify requires k <= S < T")
        if self.beta < 0 or self.kappa < 0:
            raise ValueError("UCB and eligibility multipliers must be nonnegative")

    def to_dict(self):
        return asdict(self)


def ucb_choice(mean, sd, beta):
    return jnp.argmax(mean + beta * sd, axis=-1).astype(jnp.int32)


def leader_choice(mean, sd, k, kappa, margin):
    scores = mean - jnp.where(margin, kappa * sd, 0.0)
    return jnp.argmax(
        jnp.where(jnp.arange(mean.shape[-1]) < k, scores, -jnp.inf), axis=-1
    ).astype(jnp.int32)


def cyclic_counts(budget, k, d):
    indices = jnp.arange(d)
    return jnp.where(indices < k, budget // k + (indices < budget % k), 0).astype(
        jnp.int32
    )
