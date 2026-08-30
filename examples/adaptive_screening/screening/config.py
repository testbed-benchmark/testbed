"""Immutable configuration and provenance for the non-neural pilot."""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Dataset:
    seed: int
    episodes: int


@dataclass(frozen=True)
class Config:
    d: int = 16
    T: int = 24
    tau: float = 1.0
    sigma: float = 0.75
    expected_positive_candidates: float = 2.0
    mu: float = -1.1503493803760083
    kappa_grid: tuple = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
    validation: Dataset = Dataset(271828, 20000)
    tests: tuple = (Dataset(314159, 50000), Dataset(161803, 50000))
    beta_grid: tuple = tuple(i / 5 for i in range(21))
    validation_tie_tolerance: float = 1e-4

    def __post_init__(self):
        if self.d < 2 or self.T < 2 or self.tau <= 0 or self.sigma <= 0:
            raise ValueError("Invalid dimensions or observation/prior scales")
        if not 0 < self.expected_positive_candidates < self.d:
            raise ValueError("Expected positive count must be between zero and d")
        calibrated = self.tau * NormalDist().inv_cdf(
            self.expected_positive_candidates / self.d
        )
        if not np.isclose(self.mu, calibrated, rtol=0, atol=1e-12):
            raise ValueError("mu does not match the specified prior calibration")
        numbers = [self.mu, self.tau, self.sigma, *self.beta_grid, *self.kappa_grid]
        if not np.isfinite(numbers).all():
            raise ValueError("Configuration values must be finite")
        for grid in (self.beta_grid, self.kappa_grid):
            if not grid or min(grid) < 0 or tuple(sorted(set(grid))) != tuple(grid):
                raise ValueError("Grids must be sorted, distinct and nonnegative")
        if self.kappa_grid[0] != 0 or self.validation_tie_tolerance < 0:
            raise ValueError(
                "The zero-mismatch baseline and a valid tie tolerance are required"
            )
        datasets = [self.validation, *self.tests]
        if len(self.tests) != 2 or len({x.seed for x in datasets}) != 3:
            raise ValueError("Three independent validation/test seeds are required")
        if any(x.episodes < 2 or x.seed < 0 for x in datasets):
            raise ValueError(
                "Datasets require at least two episodes and nonnegative seeds"
            )

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    def tuning_grids(self):
        return {
            "beta": list(self.beta_grid),
            "nonadaptive_k": list(range(1, self.d + 1)),
            "explore_certify": [
                {"k": k, "S": S, "h": h}
                for S in range(1, self.T)
                for k in range(1, min(self.d, S) + 1)
                for h in ("mean", "margin")
            ],
        }


def load_config(path):
    values = yaml.safe_load(Path(path).read_text())
    values["validation"] = Dataset(**values["validation"])
    values["tests"] = tuple(Dataset(**v) for v in values["tests"])
    for name in ("kappa_grid", "beta_grid"):
        values[name] = tuple(values[name])
    return Config(**values)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def code_identity():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    paths = sorted(
        [
            *PROJECT.glob("screening/**/*.py"),
            *PROJECT.glob("scripts/*.py"),
            PROJECT / "pyproject.toml",
            *PROJECT.glob("uv.lock"),
        ]
    )
    source_hash = digest(
        {
            str(path.relative_to(PROJECT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        }
    )
    return {
        "code_commit": commit,
        "source_hash": source_hash,
    }


def provenance(config, identity=None):
    identity = code_identity() if identity is None else identity
    payload = {
        "config": config.to_dict(),
        "tuning_grids": config.tuning_grids(),
        "code_commit": identity["code_commit"],
        "protocol_version": 2,
    }
    return {
        **payload,
        "source_hash": identity["source_hash"],
        "environment_hash": digest(payload),
    }
