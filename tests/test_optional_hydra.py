"""Hydra is an optional ``[train]`` dependency, so the core library must not need it."""

import subprocess
import sys

import pytest

# Imported in a fresh interpreter, with Hydra and OmegaConf made unimportable.
_PROBE = """
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("hydra", "omegaconf"):
            raise ImportError(f"{name} is blocked")

sys.meta_path.insert(0, _Block())

import testbed
from testbed.eval import chunked_spce_snmc, eval_spce_snmc
from testbed.designers import RandomDesigner
from testbed.tasks import LocationFinding
"""


def test_core_library_imports_without_hydra():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_eval_star_import_skips_hydra_evaluators():
    probe = _PROBE + "\nfrom testbed.eval import *\n"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def _hydra_cli_unsupported():
    import re
    from importlib.metadata import version

    major, minor = re.match(r"(\d+)\.(\d+)", version("hydra-core")).groups()
    return sys.version_info >= (3, 14) and (int(major), int(minor)) < (1, 4)


@pytest.mark.skipif(
    _hydra_cli_unsupported(), reason="hydra-core<1.4 CLI does not run on Python 3.14"
)
@pytest.mark.parametrize(
    "module",
    ["action", "actionbed", "aline", "dad", "rlboed", "static", "var_post"],
)
def test_cli_wraps_the_method_run(module):
    import importlib

    cli = importlib.import_module(f"testbed.cli.{module}")
    method = importlib.import_module(f"testbed.methods.{module}")
    assert cli.run is method.run
    assert callable(cli.main)


def test_lazy_submodules_resolve():
    import testbed

    assert testbed.methods.DADTrainer is not None
    assert callable(testbed.eval.evaluate_checkpoint)


def test_setup_works_outside_the_hydra_cli(tmp_path):
    from omegaconf import OmegaConf

    from testbed.core.setup import setup

    cfg = OmegaConf.create(
        {
            "save_dir": str(tmp_path),
            "device": "cpu",
            "task": {
                "_target_": "testbed.tasks.LocationFinding",
                "n_sources": 1,
                "dim_theta": 2,
            },
        }
    )
    manager, _ = setup(cfg)
    try:
        assert manager.dir.startswith(str(tmp_path))
    finally:
        manager.metric_logger.close()
