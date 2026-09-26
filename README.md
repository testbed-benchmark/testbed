# TestBED

TestBED is a PyTorch library for Bayesian Experimental Design (BED).

It is designed to support the development, evaluation, and application of BED methods by providing a set of modular building blocks that can be easily combined and extended.

Our goals are to:
- Provide simple, efficient implementations of core BED components
- Offer an extensible framework for developing new BED methods and applying them to new problems
- Enable reproducible training recipes for common and state-of-the-art BED approaches, supporting fair benchmarking and methodological development

TestBED is intentionally lightweight and highly customizable. Users are encouraged to roll their own training loops, logging, and experiment management pipelines towards their particular use cases. 

The current focus of TestBED is on policy-based approaches to Bayesian experimental design.

TestBED will be publicly released on PyPI after publication.

# Installation and Getting Started

Requirements:
- Python>=3.10
- PyTorch>=2.0

We recommend first installing a system-compatible version of PyTorch into your environment, before installing TestBED. You can install an editable version of the current library as follows:

```
git clone <repository URL anonymised for review>
cd testbed
pip install -e .[full,dev,docs]
```

To do a basic check of your installation, run `pytest`.

# Running experiments

Experiments in the paper are Hydra recipes under `testbed/recipes/`. The benchmark recipes live at `experiments/benchmarking/<task>/<method>.yaml`, e.g. `benchmarking/cart_pole/dad.yaml`.

To train one seed of a recipe, call its method's entry point from the repo root, then evaluate the final checkpoint (sPCE/sNMC bounds, sizes set by the recipe's `final_eval` block):

```sh
python testbed/methods/dad.py --config-name experiments/benchmarking/cart_pole/dad seed=0
python scripts/eval_run.py --run-dir <run_dir> --no-wandb
```

The entry point is `testbed/methods/<method>.py`, where `joint_var_post*` recipes use `var_post.py` and `actionbed_*` recipes use `actionbed.py`. Any config value can be overridden Hydra-style after the recipe, e.g. `lr=1e-3` or `final_eval.n_outer=32`. The run directory is printed at startup, and the bounds land in `<run_dir>/eval.json`.

Recipes run on `device=cuda` by default; add `device=cpu` to run without a GPU.

By default, each recipe trains for 12 hours (`train_time_budget_seconds=43200`). For a quick check that a recipe runs end to end, shorten training and the final evaluation:

```sh
python testbed/methods/dad.py --config-name experiments/benchmarking/cart_pole/dad seed=0 \
    train_time_budget_seconds=60 final_eval.n_contrastive=10000
```

The static baseline has no time budget and instead runs a fixed number of steps (`n_train_iter`).

Optionally, to log to W&B, `scripts/sweep_run.py` runs both steps in one go (set `WANDB_ENTITY` to choose the team; `final_eval.enabled=false` skips evaluation):

```sh
python scripts/sweep_run.py --recipe experiments/benchmarking/cart_pole/dad seed=0
```

`eval_run.py` can also re-score an existing run, e.g. at a larger contrastive sample size (`--n-contrastive`), or at an intermediate checkpoint (`--checkpoint step_00010000.pt`). See `python scripts/eval_run.py --help`.

## Docs

You can launch a local version of the documentation with `cd docs && make livehtml`. 

Check out the [Location Finding](docs/examples/location_finding.md) example for a quick overview of the package.

# Contributing

Please follow the following guidelines if you would like to contribute to TestBED:

1. Fork the repository and install the package in editable mode with dev
   dependencies:
```sh
pip install -e .[full,dev,docs]
```
Also install the pre-commit hooks once:
```sh
pre-commit install
```
2. Create a new branch for your feature or bug fix. Add your code, tests, and
   docs.
4. Run the tests and fix any issues before committing:
```sh
pytest
```
5. Submit a pull request with a clear description of your changes and why they
   are needed.
