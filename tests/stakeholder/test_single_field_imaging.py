import pytest
import numpy as np
from toolviper import dask
import xarray as xr
import sys
import os
import shutil
import toolviper
from xradio.image import load_image

import os
import numpy as np
from xradio.measurement_set import open_processing_set


def test_single_field_imaging():
    from xradio.measurement_set import open_processing_set
    from xradio.image import load_image, write_image, open_image
    from toolviper.utils.data import download, update

    update()

    download(file="twhya_selfcal_5chans_lsrk_compare_weights.ps.zarr")
    download(file="twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.img.zarr")
    ps_xdt = open_processing_set("twhya_selfcal_5chans_lsrk_compare_weights.ps.zarr")
    img_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.img.zarr")

    # log_level = "DEBUG"
    log_level = "INFO"
    log_to_file = False
    log_params = {
        "logger_name": "main",
        "log_to_term": True,
        "log_level": log_level,
        "log_to_file": log_to_file,
        "log_file": "client.log",
    }

    worker_logs = {
        "logger_name": "worker",
        "log_to_term": True,
        "log_level": log_level,
        "log_to_file": log_to_file,
        "log_file": "client_worker.log",
    }

    from toolviper.dask.client import local_client

    # viper_client = local_client(cores=4, memory_limit="4GB", log_params=log_params, worker_log_params=worker_logs)
    # viper_client

    import dask

    dask.config.set(scheduler="synchronous")

    import os
    import numpy as np
    from xradio.measurement_set import open_processing_set
    from astroviper.distributed.imaging.image_cube_single_field import (
        image_cube_single_field,
    )
    from xradio.image import make_empty_sky_image

    ps_store = "twhya_selfcal_5chans_lsrk_compare_weights.ps.zarr"
    image_store = "twhya_selfcal_5chans_lsrk_compare_weights_astroviper.img.zarr"

    os.system("rm -rf ")
    ps_single_pol_xdt = open_processing_set(
        "twhya_selfcal_5chans_lsrk_compare_weights.ps.zarr"
    )
    combined_field_and_source_xds = (
        ps_single_pol_xdt.xr_ps.get_combined_field_and_source_xds()
    )
    center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
    phase_direction = combined_field_and_source_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=center_field_name
    )
    image_params = {
        "image_size": [250, 250],
        "cell_size": np.array([-0.1, 0.1]) * np.pi / (180 * 3600),
        "phase_direction": phase_direction.values,
        "frequency_coords": ps_single_pol_xdt.xr_ps.get_freq_axis().values,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }

    imaging_metadata_pd = image_cube_single_field(
        ps_store=ps_store,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params={
            "weighting": "briggs",
            "robust": 0.5,
        },
        # imaging_weights_params={
        #     "weighting": "natural",
        # },
        iteration_control_params={
            "niter": 0,
            "nmajor": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 10,
        },
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
        fft_padding="1.0",
        scan_intents="OBSERVE_TARGET#ON_SOURCE",
        # image_data_variables_keep=["sky", "point_spread_function", "primary_beam"],
        # image_data_variables_keep=["sky_model", "sky_residual", "sky_deconvolved", "point_spread_function", "primary_beam"],
        # image_data_variables_keep=["sky_residual", "point_spread_function", "primary_beam", "beam_fit_params_point_spread_function", "beam_fit_params_sky_residual"],
        # image_data_variables_keep=["sky_residual", "point_spread_function", "primary_beam"],
        image_data_variables_keep=[
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
        ],
        # image_data_variables_keep=[ "sky", "point_spread_function", "primary_beam"],
        processing_set_data_group_name="base",
        double_precision=True,
        thread_info=None,
        n_chunks=None,
        overwrite=True,
    )
    img_av_xds = xr.open_zarr(image_store)

    print("************" * 10)
    print(img_xds)
    print("************" * 10)
    print(img_av_xds)
    print("************" * 10)

    from matplotlib import pyplot as plt

    frequency = 4
    plt.figure()
    # plt.imshow((img_av_xds["SKY_RESIDUAL"].isel(frequency=frequency,time=0, polarization=0)+img_av_xds["SKY_RESIDUAL"].isel(frequency=frequency,time=0, polarization=1))/2)
    plt.imshow(
        img_av_xds["SKY_RESIDUAL"].isel(frequency=frequency, time=0, polarization=0)
    )
    plt.colorbar()

    plt.figure()
    plt.imshow(
        img_xds["SKY_RESIDUAL"].isel(frequency=frequency, time=0, polarization=0)
    )
    plt.colorbar()

    plt.figure()
    plt.imshow(
        img_av_xds["PRIMARY_BEAM"].isel(frequency=frequency, time=0, polarization=0)
    )
    plt.colorbar()

    I_av = img_av_xds["SKY_RESIDUAL"].isel(polarization=0)
    I = img_xds["SKY_RESIDUAL"].isel(polarization=0)

    print(np.abs(np.max(np.abs(I))).values)

    print(imaging_metadata_pd)
    print(imaging_metadata_pd.sum())

    print(I_av)
    print("************" * 10)
    print(I_av - I)

    plt.figure()
    plt.imshow(
        (
            I_av.isel(frequency=frequency, time=0).values
            - I.isel(frequency=frequency, time=0).values
        )
    )
    plt.colorbar()

    plt.show()

    for i_f in range(5):
        rel_diff = np.max(
            np.abs(I_av.isel(frequency=i_f).values - I.isel(frequency=i_f).values)
            / np.max(np.abs(I.isel(frequency=i_f).values))
        )
        print("Relative difference: ", rel_diff)
        if i_f == 0:
            assert (
                rel_diff < 1e-3
            ), "You broke something! Relative difference between the two images is larger than 1e-3."
        else:
            assert (
                rel_diff < 1e-4
            ), "You broke something! Relative difference between the two images is larger than 1e-5."

    psf_av = img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values
    psf = img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values

    print(img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values)
    print(
        img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values * 2
    )
    print(
        img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values * 2
        + 2 * np.pi
    )

    psf_ref = [
        [
            [6.88857774e-06, 5.11572600e-06, 4.12351878e00],
            [6.19846573e-06, 4.64803383e-06, 4.45109969e00],
            [6.19845440e-06, 4.64802511e-06, 4.45109823e00],
            [6.19841033e-06, 4.64803080e-06, 4.45112253e00],
            [6.19818779e-06, 4.64785618e-06, 4.45266980e00],
        ]
    ]

    # psf_ref = [[[6.77730687e-06, 5.06243222e-06, 4.12407830e+00],
    #             [6.07923064e-06, 4.58199058e-06, 4.45233484e+00],
    #             [6.07921959e-06, 4.58198214e-06, 4.45233349e+00],
    #             [6.07917812e-06, 4.58198792e-06, 4.45235747e+00],
    #             [6.07810246e-06, 4.58203903e-06, 4.45413812e+00]]]

    psf_ref = [
        [
            [3.44428887e-06, 2.55786300e-06, 2.06175939e00],
            [3.09923287e-06, 2.32401691e-06, 2.22554985e00],
            [3.09922720e-06, 2.32401255e-06, 2.22554912e00],
            [3.09920517e-06, 2.32401540e-06, 2.22556126e00],
            [3.09909389e-06, 2.32392809e-06, 2.22633490e00],
        ]
    ]
    assert np.allclose(
        img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values,
        psf_ref,
        rtol=1e-6,
    ), "You broke something! The beam fit parameters for the point spread function are not close enough to the reference values."

    # assert np.allclose(img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values, img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values, rtol=1e-2), "You broke something! The beam fit parameters for the point spread function are not close enough to the reference values."

    from astroviper.core.imaging.make_pb_symmetric import (
        airy_disk_rorder,
        airy_disk_rorder_v2,
    )
    import time

    pb_parms = {}
    pb_parms["list_dish_diameters"] = np.array([10.7])
    pb_parms["list_blockage_diameters"] = np.array([0.75])
    pb_parms["ipower"] = 1

    image_params["image_center"] = np.array(image_params["image_size"]) // 2

    start = time.time()
    PB_v2 = xr.DataArray(
        airy_disk_rorder_v2(
            img_xds.frequency.values,
            img_xds.polarization.values,
            pb_parms,
            image_params,
        )[0, ...][
            None, ...
        ],  # Select first since we only have one dish diameter and add time axis.
        dims=("time", "frequency", "polarization", "l", "m"),
    )
    print("PB_v2 time: ", time.time() - start)

    start = time.time()
    PB_v1 = xr.DataArray(
        airy_disk_rorder(
            img_xds.frequency.values,
            img_xds.polarization.values,
            pb_parms,
            image_params,
        )[0, ...][
            None, ...
        ],  # Select first since we only have one dish diameter and add time axis.
        dims=("time", "frequency", "polarization", "l", "m"),
    )
    print("PB_v1 time: ", time.time() - start)

    plt.figure()
    plt.imshow(PB_v2.isel(frequency=frequency, time=0, polarization=0))
    plt.colorbar()
    plt.figure()
    plt.imshow(PB_v1.isel(frequency=frequency, time=0, polarization=0))
    plt.colorbar()
    plt.figure()
    plt.imshow(
        np.abs(
            PB_v2.isel(frequency=frequency, time=0, polarization=0).values
            - PB_v1.isel(frequency=frequency, time=0, polarization=0).values
        )
    )
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    test_single_field_imaging()

    # # write_image(img_xds, "test_write", out_format="casa", overwrite=True)
    # casa_dir = "images_twhya_selfcal_5chans_lsrk"
    # # img_xds = load_image({"sky_deconvoled": casa_dir+ "/twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.image",
    # #                     "sky_residual": casa_dir + "/twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.residual",
    # #                     "sky_model": casa_dir + "/twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.model",
    # #                     "primary_beam": casa_dir + "/twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.pb",
    # #                     "point_spread_function": casa_dir + "/twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.psf"})
    # # img_xds = load_image({"visibility_normalization": casa_dir + "/twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.sumwt"})
    # # img_xds.to_zarr("twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.img.zarr", mode="w")
    # from xradio.image import load_image, write_image, open_image
    # img_xds  = open_image({"visibility_normalization": casa_dir + "/twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.sumwt"})
    # #img_xds  = load_image({"visibility_normalization": casa_dir + "/twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.sumwt"})
    # print(img_xds)
    # i = img_xds.load()
    # print("************"*10)
    # print(i)
