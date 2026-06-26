"""Shared interfaces for policy-training methods."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from testbed.designers import PolicyDesigner


@dataclass
class FitResult:
    """Result returned by a policy-training method.

    Attributes:
        designer: The `PolicyDesigner` that wraps the trained policy.
        metrics: A list of dictionaries containing training metrics, such as
            loss values and evaluation scores.
        checkpoint: A dictionary holding the trained weights and training state
            (optimizer states, step counts) needed to deploy the policy or to
            resume training. The methods do not yet provide a `load`/resume
            entry point, so consuming the checkpoint is currently the caller's
            responsibility.
    """

    designer: PolicyDesigner
    metrics: list[dict[str, float]] = field(default_factory=list)
    checkpoint: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Saves the training checkpoint."""
        torch.save(self.checkpoint, Path(path))
