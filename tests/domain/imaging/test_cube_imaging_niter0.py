import pytest
import numpy as np
import shutil
import os
from pandas import Index


@pytest.fixture
def antennae_from_s3():
    from astroviper.imaging.cube_imaging_niter0 import cube_imaging_niter0
    from xradio.vis.read_processing_set import read_processing_set

    ps_store = "s3://viper-test-data/Antennae_North.cal.lsrk.split.v3.vis.zarr"
    image_name = "Antennae_North_Cube.img.zarr"
    grid_params = {
        "image_size": [500, 500],
        "cell_size": np.array([-0.13, 0.13]) * np.pi / (180 * 3600),
        "fft_padding": 1.0,
        "phase_direction": 1,
    }
    ps_name = "Antennae_North.cal.lsrk.split_0"
    n_chunks = None
    data_variables = ["sky", "point_spread_function", "primary_beam"]

    ps = read_processing_set(ps_store, intents=["OBSERVE_TARGET#ON_SOURCE"])
    polarization_coord = ps[ps_name].polarization
    frequency_coord = ps[ps_name].frequency

    output = cube_imaging_niter0(
        ps_store,
        image_name,
        grid_params,
        polarization_coord=polarization_coord,
        frequency_coord=frequency_coord,
        n_chunks=n_chunks,
        data_variables=data_variables,
    )
    yield output

    # cleanup
    shutil.rmtree(image_name)


def test_file_creation(antennae_from_s3):
    assert os.path.exists("Antennae_North_Cube.img.zarr")


def test_shape_matches(antennae_from_s3):
    assert antennae_from_s3[0].shape == (8, 13)
