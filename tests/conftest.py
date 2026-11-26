"""Shared test configuration and the input data of the continuum tests."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Tests run graphs on Dask's threaded scheduler, and OpenBLAS before 0.3.34
# returns wrong results when threads of one process call it at once
# (OpenMathLib/OpenBLAS#5825, #5836). BLAS reads these variables when NumPy
# loads it; pyproject.toml keeps zarr's pytest plugin, which imports NumPy,
# from loading before this file.
BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if "numpy" in sys.modules and any(
    os.environ.get(name) != "1" for name in BLAS_THREAD_VARIABLES
):
    raise pytest.UsageError(
        "NumPy was imported before tests/conftest.py could limit BLAS to one "
        "thread; set "
        + ", ".join(f"{name}=1" for name in BLAS_THREAD_VARIABLES)
        + " in the environment."
    )
os.environ.update(dict.fromkeys(BLAS_THREAD_VARIABLES, "1"))

TW_HYDRA_STORE_NAME = "twhya_selfcal_lsrk_5chans.ps.zarr"


@pytest.fixture(scope="session")
def tw_hydra_source_store(tmp_path_factory):
    """Download the five-channel processing set through ToolVIPER's manifest."""
    from toolviper.utils.data import download

    with TemporaryDirectory(
        prefix="tw_hydra_download_", dir=tmp_path_factory.getbasetemp()
    ) as directory:
        download(file=TW_HYDRA_STORE_NAME, folder=directory)
        store = Path(directory) / TW_HYDRA_STORE_NAME
        if not store.is_dir():
            raise RuntimeError("Could not download the shared TW Hydra reference data")
        yield store
