"""Shared Google Drive input for continuum tests."""

from pathlib import Path
from tempfile import TemporaryDirectory

import gdown
import pytest


@pytest.fixture(scope="session")
def tw_hydra_archive(tmp_path_factory):
    """Download the same five-channel processing set used by the cube notebook."""
    with TemporaryDirectory(
        prefix="tw_hydra_download_", dir=tmp_path_factory.getbasetemp()
    ) as directory:
        archive = Path(directory) / "twhya_selfcal_lsrk_5chans.ps.zarr.zip"
        result = gdown.download(
            id="1BRe3cD6YAWkn-jSPbClGGM9VlbxHP_yn", output=str(archive), quiet=True
        )
        if result is None or not archive.is_file():
            raise RuntimeError("Could not download the shared TW Hydra reference data")
        yield archive
