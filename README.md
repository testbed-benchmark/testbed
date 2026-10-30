<p align="center">
  <img src="https://raw.githubusercontent.com/testbed-benchmark/testbed/main/docs/_static/testbed-logo.svg" alt="TestBED" width="360">
</p>

<p align="center">
  <strong>Modular PyTorch building blocks for Bayesian experimental design.</strong>
</p>

<p align="center">
  <a href="https://github.com/testbed-benchmark/testbed/actions/workflows/tests.yml"><img src="https://github.com/testbed-benchmark/testbed/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://torch-testbed.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/torch-testbed/badge/?version=latest" alt="Documentation"></a>
  <a href="https://pypi.org/project/torch-testbed/"><img src="https://img.shields.io/pypi/v/torch-testbed" alt="PyPI"></a>
  <a href="https://pypi.org/project/torch-testbed/"><img src="https://img.shields.io/pypi/pyversions/torch-testbed" alt="Python versions"></a>
  <a href="https://github.com/testbed-benchmark/testbed/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

<p align="center">
  <a href="https://torch-testbed.readthedocs.io">Documentation</a> ·
  <a href="https://torch-testbed.readthedocs.io/en/latest/examples/location_finding.html">Tutorial</a> ·
  <a href="#citation">Citation</a> ·
  <a href="https://github.com/testbed-benchmark/testbed/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

---

TestBED is a PyTorch library for developing, evaluating and applying Bayesian
experimental design (BED) methods. It provides tasks, design objectives,
designers and policy networks as small, composable pieces, so you can build
your own training loops on top of them rather than adopting a fixed pipeline.
It also ships reproducible training recipes for established methods, for fair
benchmarking.

## Installation

```bash
pip install torch-testbed
```

TestBED requires Python 3.10+ and PyTorch 2.5+. The core library needs only
PyTorch. The training methods and checkpoint evaluators use
[Hydra](https://hydra.cc/), which comes with the `train` extra; `full` adds
plotting, Pyro and Weights & Biases support:

```bash
pip install "torch-testbed[train]"
pip install "torch-testbed[full]"
```

> [!NOTE]
> Hydra 1.3's command line does not support Python 3.14, so launch training
> runs with Python 3.13 or earlier.

## Quickstart

The following code chooses the designs for a source location-finding problem by
optimising the prior contrastive estimation (PCE) bound on the expected
information gain at each round:

```python
from testbed.tasks import LocationFinding
from testbed.design_objectives import PCE
from testbed.designers import SequentialDesigner

task = LocationFinding(n_sources=1, dim_theta=2)
pce = PCE(task, n_outer=512, n_inner=512)
designer = SequentialDesigner(task, T=5, objective=pce, lr=5e-3, n_step_opt=1_000)

theta = task.sample_prior()
rollout = designer.rollout(theta)  # T designs and observations, plus a posterior estimate
```

The [Location Finding tutorial](https://torch-testbed.readthedocs.io/en/latest/examples/location_finding.html)
walks through this example and compares it with other designers.

## What's included

**Tasks.** The most common benchmark problems for BED with a
shared interface, such as location finding, CES, pharmacokinetics,
Michaelis–Menten, temporal discounting, etc. They cover continuous and discrete
designs and outcomes, and likelihoods that depend on the full history.

**Designers.** `PolicyDesigner`, `SequentialDesigner`, `StaticDesigner` and
`RandomDesigner` provide different recipes for choosing designs for any task,
and can be customized or replaced.

**Design objectives and networks.** Common bounds and estimators of the expected
information gain (PCE, sPCE, NMC, sNMC and Barber–Agakov), REINFORCE gradient
estimators, and policy networks for amortised learning.

**Training methods.** Reference implementations of state-of-the-art BED
algorithms for learning amortised policies, with Hydra recipes for most tasks.
For example, training a design policy with
[Action-BED](https://arxiv.org/pdf/2606.23662) on the temporal discounting task
is as simple as running:
```bash
python testbed/methods/actionbed.py --config-name experiments/temporal_discounting/actionbed
```

The following algorithms are implemented currently:
| Method | Reference |
|---|---|
| Variational posterior (Barber–Agakov) | [Foster et al. 2020](https://arxiv.org/pdf/1911.00294), [Phillips et al., 2026](https://arxiv.org/pdf/2607.08335) |
| DAD | [Foster et al., 2021](https://arxiv.org/pdf/2103.02438) |
| RL-BOED | [Blau et al., 2022](https://arxiv.org/pdf/2202.00821) |
| ALINE | [Huang et al., 2025](https://arxiv.org/pdf/2506.07259v2) |
| Action-BED | [Rossa et al., 2026](https://arxiv.org/pdf/2606.23662) |

See [Configuration recipes](https://torch-testbed.readthedocs.io/en/latest/guides/configuration.html)
for how recipes are organized and overridden.

## Documentation

The full documentation, including guides on
[using tasks](https://torch-testbed.readthedocs.io/en/latest/guides/using_tasks.html),
[using designers](https://torch-testbed.readthedocs.io/en/latest/guides/using_designers.html),
[implementing custom methods](https://torch-testbed.readthedocs.io/en/latest/guides/custom_methods.html)
and the API reference, is at
[torch-testbed.readthedocs.io](https://torch-testbed.readthedocs.io).

## Contributing

We welcome contributions. See
[CONTRIBUTING.md](https://github.com/testbed-benchmark/testbed/blob/main/CONTRIBUTING.md)
for how to set up a development install and open a pull request, and read our
[Code of
Conduct](https://github.com/testbed-benchmark/testbed/blob/main/CODE_OF_CONDUCT.md)
before taking part.

## Citation

If you use TestBED in your research, please cite:

```bibtex
@software{testbed,
  title  = {{TestBED}: A {PyTorch} Library for {Bayesian} Experimental Design},
  author = {Kerrigan, Gavin and Hedman, Marcel and Iqbal, Sahel and Rainforth, Tom},
  year   = {2026},
  url    = {https://github.com/testbed-benchmark/testbed},
}
```
