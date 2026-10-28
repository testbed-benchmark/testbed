# Contributing to TestBED

We welcome contributions to TestBED!

We especially welcome new tasks. Many of the standard benchmark tasks are now
too easy for current algorithms, so they no longer tell us much. Harder tasks
that break existing methods are what drive the field forward, so please
consider contributing one. See
[Implementing New Tasks](https://torch-testbed.readthedocs.io/en/latest/guides/custom_tasks.html)
to get started.

Bug reports, fixes, new methods and documentation improvements are also
welcome. For larger changes, please open an issue to discuss them first. Please
read our [Code of Conduct](CODE_OF_CONDUCT.md) before taking part.

## Making a pull request

1. Fork the repository and install the package in editable mode with all
   extras:
   ```sh
   pip install -e ".[full,dev,docs]"
   ```
   Also install the pre-commit hooks once:
   ```sh
   pre-commit install
   ```
2. Create a new branch for your feature or bug fix. Add your code, tests and
   docs.
3. Run the tests and fix any issues before committing:
   ```sh
   pytest
   ```
4. Submit a pull request with a clear description of your changes and why they
   are needed.

## Documentation

Docstrings follow the
[Google Python style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Guides live in `docs/guides/` and examples in `docs/examples/`; use an existing
page such as `docs/guides/using_tasks.md` as a template, and add new pages to
the table of contents in `docs/index.md`. To preview the docs while you edit
them, run:

```sh
cd docs && make livehtml
```
