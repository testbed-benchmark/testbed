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
from testbed.eval import *
"""


def test_core_library_imports_without_hydra():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("python", "hydra_version", "fails"),
    [
        ((3, 14, 0), "1.3.2", True),
        ((3, 14, 0), "1.4.0rc1", False),
        ((3, 13, 0), "1.3.2", False),
    ],
)
def test_main_refuses_hydra_cli_it_cannot_run(
    monkeypatch, python, hydra_version, fails
):
    from types import SimpleNamespace

    import testbed.core.setup as setup_module

    monkeypatch.setattr(setup_module, "sys", SimpleNamespace(version_info=python))
    monkeypatch.setattr(setup_module.hydra, "__version__", hydra_version)
    ran = []
    monkeypatch.setattr(
        setup_module.hydra, "main", lambda **_: lambda run: lambda: ran.append(run)
    )

    main = setup_module.make_main(object(), "method/dad")
    if fails:
        with pytest.raises(RuntimeError, match="Python 3.14"):
            main()
        assert not ran
    else:
        main()
        assert len(ran) == 1


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
