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
