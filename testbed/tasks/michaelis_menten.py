"""Michaelis-Menten task."""

import torch

from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base_task import Task


class MichaelisMentenPrior(Distribution):
    """Uniform prior over Michaelis-Menten parameters.

    The default prior is: V_max, V_half ~ Unif(20, 200)
    Convention is: theta = (V_max, V_half) in that order
    TODO: taken from Alex's paper, is this the "usual" setup?
    """

    def __init__(self, vmax_lb=20, vmax_ub=200, vhalf_lb=20, vhalf_ub=200):
        """Initializes prior bounds."""
        super().__init__()

        # lower/upper bounds
        self.register_buffer("lbs", torch.tensor([vmax_lb, vhalf_lb]))
        self.register_buffer("ubs", torch.tensor([vmax_ub, vhalf_ub]))

        # log normalizing constant
        self.register_buffer("log_z", -torch.log(self.ubs - self.lbs).sum())

    def sample(self, n_samples: int = 1):
        """Draws parameter samples."""
        # [n_samples, 2]
        noise = torch.rand(n_samples, 2, device=self.lbs.device)
        samples = (self.ubs - self.lbs) * noise + self.lbs
        return samples

    def log_prob(self, theta):
        """Evaluates prior log probabilities."""
        # theta: [..., 2]
        in_support = ((theta >= self.lbs) & (theta <= self.ubs)).all(dim=-1)
        logp = torch.where(
            in_support, self.log_z, torch.tensor(-torch.inf, device=theta.device)
        )

        return logp


class MichaelisMenten(Task):
    """Implements the Michaelis Menten chemical kinetics model.

    Convention: theta = (V_max, V_half).
    """

    def __init__(self, noise_scale: float = 2.5, exponent: float = 1.0, prior=None):
        """Initializes the task."""
        # Default prior
        if prior is None:
            prior = MichaelisMentenPrior()

        self.event_shape_theta = torch.Size([2])
        self.event_shape_design = torch.Size([1])
        self.event_shape_y = torch.Size([1])

        # Standard deviation of observation noise (assumed Gaussian)
        self.noise_scale = noise_scale

        # Aka "s" -- s=1 is the standard Michaelis-Menten model
        # Other values give rise to the so-called Hill model
        self.exponent = exponent

        super().__init__(
            prior,
            discrete_y=False,
            discrete_theta=False,
            discrete_design=False,
        )

    def __repr__(self):
        """Returns a compact representation."""
        return (
            f"MichaelisMenten(noise_scale={self.noise_scale}, exponent={self.exponent})"
        )

    def make_mean(self, theta, design):
        """Computes the likelihood mean."""
        # theta:  [b, 2]
        # design: [b, 1]

        vmax = theta[:, 0].unsqueeze(-1)
        vhalf = theta[:, 1].unsqueeze(-1)

        num = vmax * design.pow(self.exponent)
        denom = vhalf.pow(self.exponent) + design.pow(self.exponent)
        return num / denom

    def _outcome_logp(self, y, theta, design):
        # y: [b, 1]
        # theta: [b, 2]
        # design: [b, 1]
        # returns: [b, ]
        loc = self.make_mean(theta, design)
        logp = normal_log_prob(y, loc, self.noise_scale)  # [b, 1]
        return logp.sum(dim=-1)

    def _sample_outcome(self, theta, design):
        # theta:   [b, 2]
        # design:  [b, 1]
        # returns: [b, 1]
        loc = self.make_mean(theta, design)  # [b, 1]
        return torch.randn_like(loc) * self.noise_scale + loc
