"""Shared input data for continuum tests."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from toolviper.utils.data import download

TW_HYDRA_STORE_NAME = "twhya_selfcal_lsrk_5chans.ps.zarr"


@pytest.fixture(scope="session")
def tw_hydra_source_store(tmp_path_factory):
    """Download the five-channel processing set through ToolVIPER's manifest."""
    with TemporaryDirectory(
        prefix="tw_hydra_download_", dir=tmp_path_factory.getbasetemp()
    ) as directory:
        download(file=TW_HYDRA_STORE_NAME, folder=directory)
        store = Path(directory) / TW_HYDRA_STORE_NAME
        if not store.is_dir():
            raise RuntimeError("Could not download the shared TW Hydra reference data")
        yield store
