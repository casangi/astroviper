import numpy as np
from xradio.vis.read_processing_set import read_processing_set
from astroviper.imaging.cube_imaging_niter0 import cube_imaging_niter0


def test_cube_imaging_niter0():

    ps_store = "s3://viper-test-data/Antennae_North.cal.lsrk.split.v3.vis.zarr"
    ps = read_processing_set(ps_store, intents=["OBSERVE_TARGET#ON_SOURCE"])
    image_name = "Antennae_North_Cube.img.zarr"
    grid_params = {
        "image_size": [500, 500],
        "cell_size": np.array([-0.13, 0.13]) * np.pi / (180 * 3600),
        "fft_padding": 1.0,
        "phase_direction": 1,
    }
    polarization_coord = ps["Antennae_North.cal.lsrk.split_0"].polarization
    frequency_coord = ps["Antennae_North.cal.lsrk.split_0"].frequency
    n_chunks = None
    data_variables = ["sky", "point_spread_function", "primary_beam"]

    cube_imaging_niter0(
        ps_store,
        image_name,
        grid_params,
        polarization_coord=polarization_coord,
        frequency_coord=frequency_coord,
        n_chunks=None,
        data_variables=data_variables,
    )
