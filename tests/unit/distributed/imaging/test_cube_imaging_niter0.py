import pytest
import numpy as np
import shutil
import os
from pandas import Index


# @pytest.fixture
# def antennae_from_s3():
#     from toolviper.dask.client import local_client

#     viper_client = local_client(cores=4, memory_limit="4GB")

#     from astroviper.imaging.cube_imaging_niter0 import cube_imaging_niter0
#     from xradio.correlated_data import open_processing_set

#     ps_store = "s3://viper-test-data/Antennae_North.cal.lsrk.split.py39.v3.vis.zarr"
#     ps = open_processing_set(ps_store, intents=["OBSERVE_TARGET#ON_SOURCE"])

#     image_name = "Antennae_North_Cube.img.zarr"
#     grid_params = {
#         "image_size": [500, 500],
#         "cell_size": np.array([-0.13, 0.13]) * np.pi / (180 * 3600),
#         "fft_padding": 1.0,
#         "phase_direction": ps[
#             "Antennae_North.cal.lsrk.split_04"
#         ].VISIBILITY.field_and_source_xds.FIELD_PHASE_CENTER,
#     }
#     ms_name = "Antennae_North.cal.lsrk.split_00"
#     n_chunks = None
#     data_variables = ["sky", "point_spread_function", "primary_beam"]

#     polarization_coord = ps[ms_name].polarization
#     frequency_coord = ps[ms_name].frequency

#     output = cube_imaging_niter0(
#         ps_store,
#         image_name,
#         grid_params,
#         polarization_coord=polarization_coord,
#         frequency_coord=frequency_coord,
#         n_chunks=n_chunks,
#         data_variables=data_variables,
#     )
#     yield output

#     assert os.path.exists("Antennae_North_Cube.img.zarr")

#     # cleanup
#     shutil.rmtree(image_name)


# def test_file_creation(antennae_from_s3):
#     assert os.path.exists("Antennae_North_Cube.img.zarr")
