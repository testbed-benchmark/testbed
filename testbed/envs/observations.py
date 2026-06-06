"""Observation functions for BED environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from .history import History

if TYPE_CHECKING:
    from .bed_env import BEDEnv


class ObservationFunction(ABC):
    """Base class for environment observations."""

    def reset(self, env: "BEDEnv") -> None:
        """Resets observation state."""

    @abstractmethod
    def __call__(self, env: "BEDEnv"):
        """Builds the current observation."""
        ...


class HistoryObservation(ObservationFunction):
    """Returns the current History object."""

    def __call__(self, env: "BEDEnv") -> History:
        """Returns env.history."""
        return env.history


class PaddedHistoryObservation(ObservationFunction):
    """Returns fixed-length padded history tensors.

    The ``designs``, ``outcomes``, and ``mask`` values directly reference tensors
    owned by ``env.history``; they are not snapshots. ``BEDEnv.step()`` mutates
    these tensors in place, so callers must clone an observation before stepping
    if they need to retain its pre-step value, such as in an RL replay buffer.
    """

    def __init__(self, horizon: int | None = None, *, include_timestep: bool = True):
        """Initializes the observation function."""
        self.horizon = None if horizon is None else int(horizon)
        self.include_timestep = include_timestep

    def __call__(self, env: "BEDEnv") -> Any:
        """Returns padded designs, outcomes, and mask by reference."""
        history = env.history
        if self.horizon is not None and self.horizon != history.max_episode_length:
            raise ValueError(
                "PaddedHistoryObservation.horizon must match env.horizon when set."
            )

        obs = {
            "designs": history.designs,
            "outcomes": history.outcomes,
            "mask": history.mask,
        }
        if self.include_timestep:
            obs["t"] = torch.full(
                env.batch_shape, env.t, device=env.device, dtype=torch.long
            )
        return obs


class FlatPaddedHistoryObservation(PaddedHistoryObservation):
    """Returns one flat padded history tensor."""

    def __call__(self, env: "BEDEnv") -> torch.Tensor:
        """Returns flattened padded designs, outcomes, mask, and optional timestep."""
        obs = super().__call__(env)
        batch_shape = env.batch_shape
        parts = [
            obs["designs"].reshape(*batch_shape, -1).to(torch.get_default_dtype()),
            obs["outcomes"].reshape(*batch_shape, -1).to(torch.get_default_dtype()),
            obs["mask"].reshape(*batch_shape, -1).to(torch.get_default_dtype()),
        ]
        if self.include_timestep:
            t = obs["t"].unsqueeze(-1).to(torch.get_default_dtype())
            denom = max(env.horizon, 1)
            parts.append(t / denom)
        return torch.cat(parts, dim=-1)
