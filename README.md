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
pip install -e .[full,dev,docs]
```
To do a basic check of your installation, run `pytest`. A more extensive test can be run with `python tests/location_finding.py` (nb: this second test requires the `stepdad` [package](https://github.com/marcelhedman/stepdad/tree/master)).

Check out `notebooks/introduction.ipynb` for a quick overview of the package.

## Docs

You can launch a local version of the documentation with `cd docs && make livehtml`. 

# Contributing

Please follow the following guidelines if you would like to contribute to TestBED:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Add your code. Add your tests.
3. Run the linter, formatter, and tests, and fix any issues:
```sh
ruff check
ruff format
pytest
```
4. Submit a pull request with a clear description of your changes and why they are needed.
