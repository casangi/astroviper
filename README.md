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

## Imaging input validation

The cube and continuum workflows share three independent configuration checks
in `processing_functions/imaging/config_check`: residual settings, model-update
settings, and driver/output settings. They reject invalid explicit choices,
physical beam parameters, kernel recipes, control-array shapes, and unsupported
output representations before execution, without modifying caller parameters.
FITS output requires uniformly spaced channel centres; use Zarr for irregular
frequency axes.

Input metadata and generated task boundaries are checked before output writes
or graph execution when they become available. Continuum beam inference needs
only antenna metadata and now runs before the first output write. Kernel-design
preflight reuses the production kernel cache. Direct entry points use
`input_checked=False`; validated callers pass `True` through the stack. Configuration predicates live in the parameter-check package; legacy helpers
reuse those predicates. Actual-product, cache-consistency, and native-buffer
guards remain active even for checked calls. Both multifield modes remain explicitly unsupported.

See [the configuration-check tutorial](docs/processing_functions_tutorials/imaging/demo_config_check.ipynb)
for valid and invalid examples.
