# Contributing to vybir

Thanks for taking the time to improve vybir. Bug reports, reproducibility reports, documentation fixes, and focused pull requests are welcome.

## Development setup

```bash
git clone https://github.com/Galactic717/vybir
cd vybir
python -m venv .venv
```

Activate the environment, then install the development dependencies:

```bash
python -m pip install -e ".[ort,demo,server,dev]"
```

Run the same checks used in CI:

```bash
ruff check vybir tests scripts
pytest -q
```

The default test run uses tiny random models and does not download the Laya checkpoint. Integration tests that require cached real weights are skipped automatically.

## Before opening an issue

- Search existing issues first.
- Include the operating system, Python version, backend (`torch` or `ort`), device, and the full traceback.
- For performance reports, include the command, hardware, warm-up procedure, and whether model loading was excluded.
- For statistical or benchmark claims, include the dataset, split seed, sample count, and exact command needed to reproduce the result.

## Pull requests

Keep changes focused and explain why they are needed. Add or update tests when behavior changes. Please do not commit downloaded model weights, generated environments, caches, or benchmark results that cannot be reproduced by a script in the repository.

By contributing, you agree that your contribution is licensed under the Apache License 2.0 used by this repository.
