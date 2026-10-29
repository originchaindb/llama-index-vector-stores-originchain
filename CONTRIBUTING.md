# Contributing

Use Python 3.10 or newer. Create and activate a virtual environment, then run these commands from the repository root:

```sh
python -m pip install 'originchain @ git+https://github.com/originchaindb/originchaindb-python.git@944cd410ffb11110c00ce925c0c609e11862916b'
python -m pip install '.[dev]' build twine
python -m pip check
python -m pytest -q --import-mode=importlib tests
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m build
python -m twine check dist/*
```

The pinned core SDK matches the acceptance workflow. Installing this project without `-e` makes the tests exercise the installed package; reinstall after changing `src/`. The source layout follows the framework's import namespace. Keep validation helpers in the existing `_support.py` module unless a change actually needs a separate module.

Tests use the real framework types and Python SDK with local HTTP fixtures. The example test launches the documented command against a loopback server and checks its request, authentication and result. No provider key or live database is required. See [examples](examples/README.md) for a read-only query against your own prepared vector index.

The acceptance workflow covers minimum and current framework dependencies, builds both distributions, checks source archive completeness, and repeats tests with the wheel installed in a fresh environment. Keep examples and contributor files in the sdist; the wheel contains only the importable adapter. Preserve the existing proprietary license and source provenance. Do not include credentials or live customer data in fixtures.
