"""History containers for BED environment rollouts."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class History:
    """Design/outcome history with time before event dimensions.

    Attributes:
        max_episode_length: Maximum number of timesteps in the history.
        designs: Padded designs with shape
            ``[*batch_shape, max_episode_length, *event_shape_design]``.
        outcomes: Padded outcomes with shape
            ``[*batch_shape, max_episode_length, *event_shape_y]``.
        mask: Boolean mask with shape ``[*batch_shape, max_episode_length]``.
    """

    max_episode_length: int
    designs: Tensor
    outcomes: Tensor
    mask: Tensor

    @property
    def batch_shape(self) -> torch.Size:
        """Returns leading batch dimensions."""
        return self.mask.shape[:-1]

    @property
    def T(self) -> int:
        """Returns the number of recorded timesteps."""
        if self.mask.numel() == 0:
            return 0
        return int(self.mask.to(torch.long).sum(dim=-1).max().item())

    @classmethod
    def empty(
        cls,
        batch_shape,
        max_episode_length: int,
        event_shape_design,
        event_shape_y,
        *,
        device=None,
        design_dtype=None,
        outcome_dtype=None,
    ) -> "History":
        """Creates an empty padded history."""
        batch_shape = torch.Size(batch_shape)
        max_episode_length = int(max_episode_length)
        if max_episode_length <= 0:
            raise ValueError("max_episode_length must be positive.")
        event_shape_design = torch.Size(event_shape_design)
        event_shape_y = torch.Size(event_shape_y)
        design_dtype = design_dtype or torch.get_default_dtype()
        outcome_dtype = outcome_dtype or torch.get_default_dtype()
        designs = torch.zeros(
            *batch_shape,
            max_episode_length,
            *event_shape_design,
            device=device,
            dtype=design_dtype,
        )
        outcomes = torch.zeros(
            *batch_shape,
            max_episode_length,
            *event_shape_y,
            device=device,
            dtype=outcome_dtype,
        )
        mask = torch.zeros(
            *batch_shape, max_episode_length, device=device, dtype=torch.bool
        )
        return cls(
            max_episode_length=max_episode_length,
            designs=designs,
            outcomes=outcomes,
            mask=mask,
        )

    def write(self, t: int, design: Tensor, outcome: Tensor) -> None:
        """Writes one timestep into the preallocated history."""
        t = int(t)
        if t < 0 or t >= self.max_episode_length:
            raise IndexError(
                f"History timestep {t} is outside max_episode_length "
                f"{self.max_episode_length}."
            )

        batch_ndim = len(self.batch_shape)
        self.designs.narrow(batch_ndim, t, 1).copy_(design.unsqueeze(batch_ndim))
        self.outcomes.narrow(batch_ndim, t, 1).copy_(outcome.unsqueeze(batch_ndim))
        self.mask[..., t] = True

    def prefix(self, t: int) -> "History":
        """Returns a padded history containing only the first t slots."""
        batch_ndim = len(self.batch_shape)
        return History(
            max_episode_length=t,
            designs=self.designs.narrow(batch_ndim, 0, t),
            outcomes=self.outcomes.narrow(batch_ndim, 0, t),
            mask=self.mask[..., :t],
        )
