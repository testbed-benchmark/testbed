# TestBED

> [!WARNING]
> TestBED is still early in its development.
> We welcome all suggestions and contributions!

TestBED is a PyTorch library for Bayesian Experimental Design (BED).

It is designed to support the development, evaluation, and application of BED methods by providing a set of modular building blocks that can be easily combined and extended.

Our goals are to:
- Provide simple, efficient implementations of core BED components
- Offer an extensible framework for developing new BED methods and applying them to new problems
- Enable reproducible training recipes for common and state-of-the-art BED approaches, supporting fair benchmarking and methodological development

TestBED is intentionally lightweight and highly customizable. Rather than prescribing a full pipeline, it provides core components that users can build on top of. Users are encouraged to roll their own training loops, logging, and experiment management pipelines towards their particular use cases. 

The current focus of TestBED is on policy-based approaches to Bayesian experimental design. Support for more traditional sequential BED methods may be added in future versions.

# Installation and Getting Started

TestBED is still in development. 

Requirements:
- Python>=3.10
- PyTorch>=2.0


You can install an editable version of the current library as follows:

```
git clone git@github.com:testbed-benchmark/testbed.git
cd testbed
pip install -e .[full,dev]
```
To do a basic check of your installation, run `pytest`. A more extensive test can be run with `python tests/location_finding.py` (nb: this second test requires the `stepdad` [package](https://github.com/marcelhedman/stepdad/tree/master)).

Check out `notebooks/introduction.ipynb` for a quick overview of the package.

# Recipes

> [!WARNING]
> These recipes are only meant as a guideline/illustration -- you will likely need to modify them for your use case.

While TestBED is intended as a general-purpose library, we have also included some basic recipes to help get you started:
- `recipes/train.py` provides a basic `Trainer` class for end-to-end training of policy networks
- `recipes/location_finding/train_lf.py` is an example script that glues together all of the TestBED components
- `recipes/location_finding/config.yaml` gives some suggested default hyperparameters which will yield a strong baseline
