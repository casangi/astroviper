# AstroVIPER

AstroVIPER (Visibility and Image Parallel Execution Reduction) is in development.

[![Python 3.11 3.12 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/release/python-3130/)
[![Linux Tests](https://github.com/casangi/astroviper/actions/workflows/python-testing-linux.yml/badge.svg?branch=main)](https://github.com/casangi/astroviper/actions/workflows/python-testing-linux.yml?query=branch%3Amain)
[![macOS Tests](https://github.com/casangi/astroviper/actions/workflows/python-testing-macos.yml/badge.svg?branch=main)](https://github.com/casangi/astroviper/actions/workflows/python-testing-macos.yml?query=branch%3Amain)
[![ipynb Tests](https://github.com/casangi/astroviper/actions/workflows/run-ipynb.yml/badge.svg?branch=main)](https://github.com/casangi/astroviper/actions/workflows/run-ipynb.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/casangi/astroviper/branch/main/graph/badge.svg)](https://codecov.io/gh/casangi/astroviper/branch/main/astroviper)
<!-- [![Documentation Status](https://readthedocs.org/projects/astroviper/badge/?version=latest)](https://astroviper.readthedocs.io) -->
[![Version Status](https://img.shields.io/pypi/v/astroviper.svg)](https://pypi.python.org/pypi/astroviper/)

## Installation

### Linux

Create a virtual environment using any tool of your choice (`venv`, `uv`, `mamba`, `conda` etc.)
and run

```
pip install astroviper
```

### Mac

On macOS, `pip install python-casacore` does not work. Install it via
conda-forge first (this requires an active conda/mamba environment)

```bash
conda install -c conda-forge python-casacore
```
Then install astroviper:

```bash
pip install astroviper
```

### Developer Setup

```bash
git clone git@github.com:casangi/astroviper.git
cd astroviper
pip install '.[all]'
pre-commit install
```

On macOS, install python-casacore via conda-forge (as above) before running `pip install`.

The `pre-commit install` step sets up git hooks that automatically run code
formatting (black) and strip Jupyter notebook outputs (nbstripout) on every
commit. This keeps diffs clean and prevents large binary outputs from bloating
the repository.

If `pre-commit` detects and makes any changes, those files will need to be
re-staged before committing. This is to allow the developer to inspect the
modified file before committing.
