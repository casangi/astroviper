# AstroVIPER

AstroVIPER (Visibility and Image Parallel Execution Reduction) is in development.

[![Python 3.12 3.13 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/downloads/release/python-3140/)
[![Linux Tests](https://github.com/casangi/astroviper/actions/workflows/python-testing-linux.yml/badge.svg?branch=main)](https://github.com/casangi/astroviper/actions/workflows/python-testing-linux.yml?query=branch%3Amain)
[![macOS Tests](https://github.com/casangi/astroviper/actions/workflows/python-testing-macos.yml/badge.svg?branch=main)](https://github.com/casangi/astroviper/actions/workflows/python-testing-macos.yml?query=branch%3Amain)
[![ipynb Tests](https://github.com/casangi/astroviper/actions/workflows/run-ipynb.yml/badge.svg?branch=main)](https://github.com/casangi/astroviper/actions/workflows/run-ipynb.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/casangi/astroviper/branch/main/graph/badge.svg)](https://codecov.io/gh/casangi/astroviper/branch/main/astroviper)
<!-- [![Documentation Status](https://readthedocs.org/projects/astroviper/badge/?version=latest)](https://astroviper.readthedocs.io) -->
[![Version Status](https://img.shields.io/pypi/v/astroviper.svg)](https://pypi.python.org/pypi/astroviper/)

## Installation

Create a virtual environment using any tool of your choice (`venv`, `uv`, `mamba`, `conda` etc.)
and run

```
pip install astroviper
```

To convert Measurement Set v2 (MSv2) data or image it directly, also install the
casacore table backend, [arcae](https://github.com/ska-sa/arcae), which XRADIO uses
and which has wheels for Linux and macOS:

```
pip install "astroviper[casacore]"
```

### Developer Setup

```bash
git clone git@github.com:casangi/astroviper.git
cd astroviper
pip install -e '.[all]'     # builds the C++ extensions; '[all]' adds test/docs/interactive deps
pre-commit install          # installs ruff + nbstripout git hooks
```
- **After editing any `.cpp` / `.hpp` / `CMakeLists.txt`** you must rebuild for
  the change to take effect (the C++ is compiled at install time):
  ```bash
  pip install -e '.[all]'   # or: pip install .
  ```
  `scikit-build-core` caches in `build/{wheel_tag}/`; if a build behaves oddly,
  delete `build/` and reinstall. The -e creates an editable Python installation.
