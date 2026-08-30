"""All pilot tests run on the CPU, including on GPU-capable hosts."""

import os
import tempfile
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adaptive-screening-matplotlib")
)

import pytest  # noqa: E402

from screening.config import Config, Dataset  # noqa: E402


@pytest.fixture
def small_config():
    return Config(
        d=4,
        T=6,
        mu=0.0,
        validation=Dataset(271828, 256),
        tests=(Dataset(314159, 256), Dataset(161803, 256)),
    )
