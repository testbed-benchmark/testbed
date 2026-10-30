<p align="center">
  <img src="https://raw.githubusercontent.com/testbed-benchmark/testbed/main/docs/_static/testbed-logo.svg" alt="TestBED" width="360">
</p>

<p align="center">
  <strong>Modular PyTorch building blocks for Bayesian experimental design.</strong>
</p>

<p align="center">
  <a href="https://torch-testbed.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/torch-testbed?style=for-the-badge&logo=readthedocs&logoColor=white&label=docs" alt="Documentation"></a>
  <a href="https://pypi.org/project/torch-testbed/"><img src="https://img.shields.io/pypi/v/torch-testbed?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/testbed-benchmark/testbed/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/testbed-benchmark/testbed/tests.yml?branch=main&style=for-the-badge&logo=github&logoColor=white&label=tests" alt="Tests"></a></p>

<p align="center">
  <a href="https://torch-testbed.readthedocs.io">Documentation</a> ·
  <a href="https://torch-testbed.readthedocs.io/en/latest/examples/location_finding.html">Tutorial</a> ·
  <a href="#citation">Citation</a> ·
  <a href="https://github.com/testbed-benchmark/testbed/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

---

TestBED is a PyTorch library for developing, evaluating and applying Bayesian
experimental design (BED) methods. Tasks, design objectives, designers and
policy networks come as small, composable pieces, so you can write your own
training loops on top of them instead of adopting a fixed pipeline. For fair
benchmarking, TestBED also ships reproducible training recipes for established
methods.

## Installation

```bash
pip install torch-testbed
```

TestBED requires Python 3.10+ and PyTorch 2.5+, and the core library depends
only on PyTorch. Two optional extras add more: `train` installs
[Hydra](https://hydra.cc/), which the training methods and checkpoint
evaluators use, and `full` also adds plotting, Pyro and Weights & Biases
support.

```bash
pip install "torch-testbed[train]"
pip install "torch-testbed[full]"
```

> [!NOTE]
> Hydra 1.3's command line does not yet support Python 3.14, so please use
> Python 3.13 or earlier to launch training runs.

## Quickstart

Here is how to design a source location-finding experiment, choosing each
design by optimising the prior contrastive estimation (PCE) bound on the
expected information gain:

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
walks through this example step by step and compares it with other designers.

## What's included

**Tasks.** Widely used BED benchmark problems behind a shared interface,
including location finding, CES, pharmacokinetics, Michaelis–Menten and
temporal discounting. The interface covers continuous and discrete designs and
outcomes, as well as likelihoods that depend on the full history.

**Designers.** `PolicyDesigner`, `SequentialDesigner`, `StaticDesigner` and
`RandomDesigner` offer different strategies for choosing designs. Each works
with any task, and you can customise or replace them as needed.

**Design objectives and networks.** Common bounds on and estimators of the
expected information gain (PCE, sPCE, NMC, sNMC and Barber–Agakov), REINFORCE
gradient estimators, and policy networks for amortised design.

**Training methods.** Reference implementations of state-of-the-art algorithms
for learning amortised design policies, with Hydra recipes for most tasks. For
example, training a policy with [Action-BED](https://arxiv.org/pdf/2606.23662)
on the temporal discounting task takes a single command:

```bash
python testbed/methods/actionbed.py --config-name experiments/temporal_discounting/actionbed
```

The following methods are currently available:

| Method | Reference |
|---|---|
| Variational posterior (Barber–Agakov) | [Foster et al., 2020](https://arxiv.org/pdf/1911.00294); [Phillips et al., 2026](https://arxiv.org/pdf/2607.08335) |
| DAD | [Foster et al., 2021](https://arxiv.org/pdf/2103.02438) |
| RL-BOED | [Blau et al., 2022](https://arxiv.org/pdf/2202.00821) |
| ALINE | [Huang et al., 2025](https://arxiv.org/pdf/2506.07259v2) |
| Action-BED | [Rossa et al., 2026](https://arxiv.org/pdf/2606.23662) |

See [Configuration recipes](https://torch-testbed.readthedocs.io/en/latest/guides/configuration.html)
to learn how recipes are organised and how to override them.

## Documentation

The full documentation lives at
[torch-testbed.readthedocs.io](https://torch-testbed.readthedocs.io). It
includes guides on
[using tasks](https://torch-testbed.readthedocs.io/en/latest/guides/using_tasks.html),
[using designers](https://torch-testbed.readthedocs.io/en/latest/guides/using_designers.html)
and
[implementing custom methods](https://torch-testbed.readthedocs.io/en/latest/guides/custom_methods.html),
as well as a complete API reference.

## Contributing

Contributions are very welcome! See
[CONTRIBUTING.md](https://github.com/testbed-benchmark/testbed/blob/main/CONTRIBUTING.md)
to set up a development install and open a pull request, and please read our
[Code of Conduct](https://github.com/testbed-benchmark/testbed/blob/main/CODE_OF_CONDUCT.md)
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
