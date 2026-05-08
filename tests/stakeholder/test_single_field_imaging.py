import pytest
import numpy as np
from toolviper import dask
import xarray as xr
import sys
import os
import shutil
import toolviper
import os
import numpy as np
from xradio.measurement_set import open_processing_set


def make_corrugated_sinusoid(
    shape,
    amplitude=1.0,
    wavelength=16.0,
    angle_deg=45.0,
    phase=0.0,
):
    """Generate a 2D sinusoid matching the 'corrugated roof' artifact.

    Parameters
    ----------
    shape : tuple of int
        Output shape (ny, nx).
    amplitude : float
        Peak amplitude of the pattern.
    wavelength : float
        Wavelength in pixels along the propagation direction.
    angle_deg : float
        Angle (in degrees) of the wavefront normal relative to the x axis.
    phase : float
        Phase offset in radians.
    """
    ny, nx = shape
    theta = np.deg2rad(angle_deg)
    k = 2.0 * np.pi / wavelength
    kx = k * np.cos(theta)
    ky = k * np.sin(theta)

    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]

    return amplitude * np.sin(kx * x + ky * y + phase)


def fit_corrugated_sinusoid(
    image,
    amplitude=1.0,
    wavelength=16.0,
    angle_deg=45.0,
    phase=0.0,
):
    """Fit make_corrugated_sinusoid to a 2D image by least squares.

    Parameters
    ----------
    image : 2D ndarray
        Target image to fit.
    amplitude, wavelength, angle_deg, phase : float
        Initial guesses for the fit parameters.

    Returns
    -------
    params : dict
        Fitted {amplitude, wavelength, angle_deg, phase}.
    fit : 2D ndarray
        Best-fit sinusoid image (same shape as input).
    """
    from scipy.optimize import least_squares

    image = np.asarray(image)
    shape = image.shape

    def residuals(p):
        w, ang, ph = p
        return (
            make_corrugated_sinusoid(shape, amplitude, w, ang, ph) - image
        ).ravel()

    x0 = [wavelength, angle_deg, phase]
    result = least_squares(residuals, x0)
    w, ang, ph = result.x
    params = {
        "amplitude": amplitude,
        "wavelength": w,
        "angle_deg": ang,
        "phase": ph,
    }
    fit = make_corrugated_sinusoid(shape, amplitude, w, ang, ph)
    return params, fit


def sinusoid_power_at_frequency(image, wavelength, angle_deg):
    """Measure an image's FFT power at a specific spatial frequency.

    Returns (peak_power, noise_median, snr). A large snr means the sinusoid
    is present in this image above the local FFT background.
    """
    image = np.asarray(image)
    ny, nx = image.shape
    theta = np.deg2rad(angle_deg)
    fx_target = np.cos(theta) / wavelength
    fy_target = np.sin(theta) / wavelength

    f = np.fft.fftshift(np.fft.fft2(image - image.mean()))
    power = np.abs(f) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    iy = int(np.argmin(np.abs(fy - fy_target)))
    ix = int(np.argmin(np.abs(fx - fx_target)))

    peak = power[iy - 1 : iy + 2, ix - 1 : ix + 2].max()
    mask = np.ones_like(power, dtype=bool)
    mask[iy - 2 : iy + 3, ix - 2 : ix + 3] = False
    mask[ny - iy - 3 : ny - iy + 2, nx - ix - 3 : nx - ix + 2] = False
    noise = np.median(power[mask])
    return peak, noise, peak / noise


def test_single_field_imaging_niter0():
    from xradio.measurement_set import open_processing_set
    from toolviper.utils.data import download, update

    #update()

    # download(file="twhya_selfcal_5chans_lsrk_compare_weights.ps.zarr")
    # download(file="twhya_selfcal_5chans_lsrk_robust_0.5_niter_0_nmajor_1.img.zarr")
    ps_store = "twhya_selfcal_lsrk_5chans.ps.zarr"
    ps_xdt = open_processing_set("twhya_selfcal_lsrk_5chans.ps.zarr")
    #img_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_niter_0_nmajor_1_natural.img.zarr")
    img_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_niter_0_nmajor_1_briggs.img.zarr")

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

    ps_xdt = open_processing_set(
        ps_store
    )
    combined_field_and_source_xds = (
        ps_xdt.xr_ps.get_combined_field_and_source_xds()
    )
    center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
    phase_direction = combined_field_and_source_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=center_field_name
    )
    image_params = {
        "image_size": [250, 250],
        "cell_size": np.array([-0.1, 0.1]) * np.pi / (180 * 3600),
        "phase_direction": phase_direction.values,
        "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }
    image_store = "twhya_selfcal_lsrk_5chans_astroviper.img.zarr"
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
            "cycleniter": 1,
        },
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
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
        processing_function_threads=1,
        n_chunks=None,
        overwrite=True,
        disk_chunk_sizes={"frequency": 2},
        vizualize_graph=True
    )
    img_av_xds = xr.open_zarr(image_store)

    # print("************" * 10)
    # print(img_xds)
    # print("************" * 10)
    # print(img_av_xds)
    # print("************" * 10)

    from matplotlib import pyplot as plt

    frequency = 0
    # plt.figure()
    # # plt.imshow((img_av_xds["SKY_RESIDUAL"].isel(frequency=frequency,time=0, polarization=0)+img_av_xds["SKY_RESIDUAL"].isel(frequency=frequency,time=0, polarization=1))/2)
    # plt.imshow(
    #     img_av_xds["SKY_RESIDUAL"].isel(frequency=frequency, time=0, polarization=0)
    # )
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(
    #     img_xds["SKY_RESIDUAL"].isel(frequency=frequency, time=0, polarization=0)
    # )
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(
    #     img_av_xds["PRIMARY_BEAM"].isel(frequency=frequency, time=0, polarization=0)
    # )
    # plt.colorbar()

    polarization = 0
    I_av = img_av_xds["SKY_RESIDUAL"].isel(polarization=polarization,time=0)
    #I = img_xds["SKY_RESIDUAL"].isel(polarization=polarization,time=0)
    I = img_xds["SKY_DECONVOLVED"].isel(polarization=polarization,time=0)
    PSF_av = img_av_xds["POINT_SPREAD_FUNCTION"].isel(polarization=polarization,time=0)
    PSF = img_xds["POINT_SPREAD_FUNCTION"].isel(polarization=polarization,time=0)

    print(np.abs(np.max(np.abs(I))).values)

    # print(imaging_metadata_pd)
    # print(imaging_metadata_pd.sum())

    # plt.figure()
    # plt.imshow(
    #     (
    #         img_xds["POINT_SPREAD_FUNCTION"].isel(polarization=0,time=0,frequency=2)
    #         - img_xds["POINT_SPREAD_FUNCTION"].isel(polarization=1,time=0,frequency=2)
    #     )
    # )
    # plt.colorbar()

    # plt.show()
    
    print(I_av.frequency.values)
    print(I.frequency.values)
    print(I.frequency.values - I_av.frequency.values)
    
    I_dif = I_av.isel(frequency=1).values - I.isel(frequency=1).values
    fit_params, art = fit_corrugated_sinusoid(
        I_dif,
        amplitude=0.0001862015899329328,
        wavelength=14.907769026883434,
        angle_deg=160.85466157331112,
        phase=5.087640033309135,
    )
    # art = make_corrugated_sinusoid(
    #     I_dif.shape,
    #     amplitude=0.0001862015899329328,
    #     wavelength=15.0,
    #     angle_deg=158,
    #     phase=0.0
    # )
    print("Fitted sinusoid parameters:", fit_params)
    print(np.max(I_dif))

    wl = fit_params["wavelength"]
    ang = fit_params["angle_deg"]
    for label, arr in [
        ("I_av", I_av.isel(frequency=1).values),
        ("I   ", I.isel(frequency=1).values),
        ("diff", I_dif),
    ]:
        peak, noise, snr = sinusoid_power_at_frequency(arr, wl, ang)
        print(f"{label}: FFT peak={peak:.3e}  noise_med={noise:.3e}  snr={snr:.1f}")

    # Source-free cutout test: fit the sinusoid independently to a corner
    # of each image and see whose RMS drops more after subtracting the fit.
    # Whichever image has the larger RMS drop is where the artifact lives.
    def source_free_fit_drop(arr, label):
        cutout = arr[:80, :80]
        rms_before = np.std(cutout)
        _, fit = fit_corrugated_sinusoid(
            cutout,
            amplitude=fit_params["amplitude"],
            wavelength=wl,
            angle_deg=ang,
            phase=fit_params["phase"],
        )
        rms_after = np.std(cutout - fit)
        print(
            f"{label}: corner RMS {rms_before:.3e} -> {rms_after:.3e}  "
            f"(drop {100*(1 - rms_after/rms_before):+.1f}%)"
        )

    source_free_fit_drop(I_av.isel(frequency=1).values, "I_av")
    source_free_fit_drop(I.isel(frequency=1).values, "I   ")

    # PSF test: PSF is the pipeline's response to a point source, so it has
    # no source emission masking the artifact. If the sinusoid is present in
    # one PSF but not the other, the artifact belongs to that pipeline.
    # for label, arr in [
    #     ("PSF_av", PSF_av.isel(frequency=1).values),
    #     ("PSF   ", PSF.isel(frequency=1).values),
    # ]:
    #     peak, noise, snr = sinusoid_power_at_frequency(arr, wl, ang)
    #     print(f"{label}: FFT peak={peak:.3e}  noise_med={noise:.3e}  snr={snr:.1f}")

    # fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    # im0 = axes[0].imshow(I_dif)
    # axes[0].set_title("I_av - I (channel 1)")
    # fig.colorbar(im0, ax=axes[0])
    # im1 = axes[1].imshow(art)
    # axes[1].set_title("Fitted sinusoid")
    # fig.colorbar(im1, ax=axes[1])
    # im2 = axes[2].imshow(I_dif - art)
    # axes[2].set_title("Residual")
    # fig.colorbar(im2, ax=axes[2])
    # im3 = axes[3].imshow(I_av.isel(frequency=1).values / I.isel(frequency=1).values, vmin=0.99, vmax=1.01)
    # axes[3].set_title("Residual")
    # fig.colorbar(im3, ax=axes[3])


    for i_f in range(5): #Skip last since casa blanks this
        rel_diff = np.max(
            np.abs(I_av.isel(frequency=i_f).values - I.isel(frequency=i_f).values)
            / np.max(np.abs(I.isel(frequency=i_f).values))
        )
        # print("Relative difference: ", rel_diff)
        # fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        # fig.suptitle("Channel " + str(i_f) + " Frequency: " + str(I_av.frequency.values[i_f]))
        # im0 = axes[0,0].imshow(I_av.isel(frequency=i_f).values)
        # axes[0,0].set_title("I_AV")
        # fig.colorbar(im0, ax=axes[0,0])
        # im1 = axes[0,1].imshow(I.isel(frequency=i_f).values)
        # axes[0,1].set_title("I_CASA")
        # fig.colorbar(im1, ax=axes[0,1])
        # im2 = axes[0,2].imshow(
        #     I_av.isel(frequency=i_f).values - I.isel(frequency=i_f).values
        # )
        # axes[0,2].set_title("I_AV - I_CASA")
        # fig.colorbar(im2, ax=axes[0,2])
        
        # im3 = axes[1,0].imshow(PSF_av.isel(frequency=i_f).values)
        # axes[1,0].set_title("PSF_AV")
        # fig.colorbar(im3, ax=axes[1,0])
        # im4 = axes[1,1].imshow(PSF.isel(frequency=i_f).values)
        # axes[1,1].set_title("PSF_CASA")
        # fig.colorbar(im4, ax=axes[1,1])
        # im5 = axes[1,2].imshow(
        #     PSF_av.isel(frequency=i_f).values - PSF.isel(frequency=i_f).values
        # )
        # axes[1,2].set_title("PSF_AV - PSF_CASA")
        # fig.colorbar(im5, ax=axes[1,2])
        

        if i_f%2 == 1: #If odd
            print("Odd channel", i_f)
            assert (
                rel_diff < 1e-3
            ), "You broke something! Relative difference between the two images is larger than 1e-3."
        else:
            print("Even channel", i_f)
            assert (
                rel_diff < 1e-4
            ), "You broke something! Relative difference between the two images is larger than 1e-5."
            
# Relative difference:  3.649608446223135e-05
# Even channel 0
# Relative difference:  0.0005557717352557811
# Odd channel 1
# Relative difference:  3.9301162730436414e-05
# Even channel 2
# Relative difference:  0.00040725899929592245
# Odd channel 3
# Relative difference:  3.216840648012481e-05

    #plt.show()


    psf_av = img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values
    psf = img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values

    # print(img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values)
    # print(
    #     img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values * 2
    # )
    # print(
    #     img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values * 2
    #     + 2 * np.pi
    # )

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
    
    psf_ref = [[[3.09938237e-06, 2.32409432e-06, 2.22543281e+00],
                [3.09927976e-06, 2.32408319e-06, 2.22547993e+00],
                [3.09916079e-06, 2.32406462e-06, 2.22536627e+00],
                [3.09916724e-06, 2.32405466e-06, 2.22535443e+00],
                [3.09916185e-06, 2.32405054e-06, 2.22535528e+00]]]
    
    print(img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values)
    assert np.allclose(
        img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values,
        psf_ref,
        rtol=1e-6,
    ), "You broke something! The beam fit parameters for the point spread function are not close enough to the reference values."

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
    # print("PB_v2 time: ", time.time() - start)

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
    # print("PB_v1 time: ", time.time() - start)

    # plt.figure()
    # plt.imshow(PB_v2.isel(frequency=frequency, time=0, polarization=0))
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(PB_v1.isel(frequency=frequency, time=0, polarization=0))
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(
    #     np.abs(
    #         PB_v2.isel(frequency=frequency, time=0, polarization=0).values
    #         - PB_v1.isel(frequency=frequency, time=0, polarization=0).values
    #     )
    # )
    # plt.colorbar()
    # plt.show()
    
def test_single_field_imaging():
    from xradio.measurement_set import open_processing_set
    from toolviper.utils.data import download, update

    # update()

    # download(file="twhya_selfcal_5chans_lsrk_compare_weights.ps.zarr")
    # download(file="twhya_selfcal_5chans_lsrk_robust_0.5_niter_0.img.zarr")
    ps_store = "twhya_selfcal_lsrk_5chans.ps.zarr"
    ps_xdt = open_processing_set("twhya_selfcal_lsrk_5chans.ps.zarr")
    img_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_niter_99_nmajor_1_briggs.img.zarr")

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

    image_store = "twhya_selfcal_5chans_lsrk_niter_99_astroviper.img.zarr"
    combined_field_and_source_xds = (
        ps_xdt.xr_ps.get_combined_field_and_source_xds()
    )
    center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
    phase_direction = combined_field_and_source_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=center_field_name
    )
    image_params = {
        "image_size": [250, 250],
        "cell_size": np.array([-0.1, 0.1]) * np.pi / (180 * 3600),
        "phase_direction": phase_direction.values,
        "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
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
            "niter": 100,
            "nmajor": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 1,
        },
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
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
            "sky_model",
            "mask"
        ],
        # image_data_variables_keep=[ "sky", "point_spread_function", "primary_beam"],
        processing_set_data_group_name="base",
        double_precision=True,
        thread_info=None,
        processing_function_threads=12,
        n_chunks=1,
        overwrite=True,
        disk_chunk_sizes={"frequency": 5},
        vizualize_graph=True,
        write_visibility_model_to_ps=True
    )
    img_av_xds = xr.open_zarr(image_store)
    print("av image mask: ", img_av_xds["MASK"])

    from matplotlib import pyplot as plt
    frequency = 4
    polarization = 0
    I_av = img_av_xds.isel(polarization=polarization,time=0,l=slice(100,150), m=slice(100,150))
    I = img_xds.isel(polarization=polarization,time=0,l=slice(100,150), m=slice(100,150))

    # print(np.abs(np.max(np.abs(I))).values)

    # print(imaging_metadata_pd)
    # print(imaging_metadata_pd.sum())

    # print(I_av)
    # print("************" * 10)
    # print(I_av - I)
    
    print(I_av)
    print("************" * 10)
    print(I)
    
    max_per_dif_model= [2.1e-05, 0.001, 7.7e-06, 0.0022, 0.0013]
    min_per_dif_model = [2.0e-05, 0.0009, 7.6e-06, 0.0021, 0.0012]  
    
    max_per_dif_residual= [0.12, 0.07, 0.09, 0.051, 0.045]
    min_per_dif_residual = [0.11, 0.06, 0.08, 0.05, 0.04]  
    
    for i_f in range(5):
        # print("Channel ", i_f)
        
        fig, axes = plt.subplots(3, 3, figsize=(14, 14 ))
        fig.suptitle("Channel " + str(i_f) + " Frequency: " + str(I_av.frequency.values[i_f]))
        im0 = axes[0,0].imshow(I_av["SKY_MODEL"].isel(frequency=i_f).values)
        axes[0,0].set_title("I_AV SKY MODEL")
        fig.colorbar(im0, ax=axes[0,0])
        im1 = axes[0,1].imshow(I["SKY_MODEL"].isel(frequency=i_f).values)
        axes[0,1].set_title("I_CASA SKY MODEL")
        fig.colorbar(im1, ax=axes[0,1])
        im2 = axes[0,2].imshow(
            I_av["SKY_MODEL"].isel(frequency=i_f).values - I["SKY_MODEL"].isel(frequency=i_f).values
        )
        axes[0,2].set_title("SKY_MODEL_AV - SKY_MODEL_CASA")
        fig.colorbar(im2, ax=axes[0,2])
        
        
        im3 = axes[1,0].imshow(I_av["SKY_RESIDUAL"].isel(frequency=i_f).values)
        axes[1,0].set_title("SKY_RESIDUAL_AV")
        fig.colorbar(im3, ax=axes[1,0])
        im4 = axes[1,1].imshow(I["SKY_RESIDUAL"].isel(frequency=i_f).values)
        axes[1,1].set_title("SKY_RESIDUAL_CASA")
        fig.colorbar(im4, ax=axes[1,1])
        im5 = axes[1,2].imshow(
            (I_av["SKY_RESIDUAL"].isel(frequency=i_f).values - I["SKY_RESIDUAL"].isel(frequency=i_f).values)/np.max(np.abs(I["SKY_RESIDUAL"].isel(frequency=i_f).values))
        )
        axes[1,2].set_title("SKY_RESIDUAL_AV - SKY_RESIDUAL_CASA")
        fig.colorbar(im5, ax=axes[1,2])
        
        
        # im3 = axes[2,0].imshow(I_av["PRIMARY_BEAM"].isel(frequency=i_f).values)
        # axes[2,0].set_title("PRIMARY_BEAM_AV")
        # fig.colorbar(im3, ax=axes[2,0])
        # im4 = axes[2,1].imshow(I["PRIMARY_BEAM"].isel(frequency=i_f).values)
        # axes[2,1].set_title("PRIMARY_BEAM_CASA")
        # fig.colorbar(im4, ax=axes[2,1])
        # im5 = axes[2,2].imshow(
        #     (I_av["PRIMARY_BEAM"].isel(frequency=i_f).values - I["PRIMARY_BEAM"].isel(frequency=i_f).values)/np.max(np.abs(I["PRIMARY_BEAM"].isel(frequency=i_f).values))
        # )
        # axes[2,2].set_title("PRIMARY_BEAM_AV - PRIMARY_BEAM_CASA")
        # fig.colorbar(im5, ax=axes[2,2])
        
        im3 = axes[2,0].imshow(I_av["MASK"].isel(frequency=i_f).values)
        axes[2,0].set_title("MASK_AV")
        fig.colorbar(im3, ax=axes[2,0])
        im4 = axes[2,1].imshow(I["MASK"].isel(frequency=i_f).values)
        axes[2,1].set_title("MASK_CASA")
        fig.colorbar(im4, ax=axes[2,1])
        im5 = axes[2,2].imshow(
            I_av["MASK"].isel(frequency=i_f).values - I["MASK"].isel(frequency=i_f).values
        )
        axes[2,2].set_title("MASK_AV - MASK_CASA")
        fig.colorbar(im5, ax=axes[2,2])
        
        rel_diff_model = np.max(
            np.abs(I_av["SKY_MODEL"].isel(frequency=i_f).values - I["SKY_MODEL"].isel(frequency=i_f).values)
            / np.max(np.abs(I["SKY_MODEL"].isel(frequency=i_f).values))
        )
        
        rel_diff_residual = np.max(
            np.abs(I_av["SKY_RESIDUAL"].isel(frequency=i_f).values - I["SKY_RESIDUAL"].isel(frequency=i_f).values)  
            / np.max(np.abs(I["SKY_RESIDUAL"].isel(frequency=i_f).values))
        )
        
        print(f"Channel {i_f} Relative difference in SKY_MODEL: ", rel_diff_model)
        print(f"Channel {i_f} Relative difference in SKY_RESIDUAL: ", rel_diff_residual)
        

        if rel_diff_model > max_per_dif_model[i_f]:
            print(f"BAD MODEL ", rel_diff_model, " is greater than ", max_per_dif_model[i_f])
            
        if rel_diff_model < min_per_dif_model[i_f]:
            print(f"GOOD MODEL ", rel_diff_model, " is less than ", min_per_dif_model[i_f])
        
        if rel_diff_residual > max_per_dif_residual[i_f]:
            print(f"BAD RESIDUAL ", rel_diff_residual, " is greater than ", max_per_dif_residual[i_f])
        if rel_diff_residual < min_per_dif_residual[i_f]:
            print(f"GOOD RESIDUAL ", rel_diff_residual, " is less than ", min_per_dif_residual[i_f])
        
        
        
        # Channel 0 Relative difference in SKY_MODEL:  2.0388167938454347e-05
        # Channel 0 Relative difference in SKY_RESIDUAL:  0.49524705855669815
        
        # Channel 1 Relative difference in SKY_MODEL:  0.08863909101226053
        # Channel 1 Relative difference in SKY_RESIDUAL:  0.6390266758908688
        
        # Channel 2 Relative difference in SKY_MODEL:  1.0192190827199984e-05
        # Channel 2 Relative difference in SKY_RESIDUAL:  0.6264641337899804
        
        # Channel 3 Relative difference in SKY_MODEL:  0.002121664713110676
        # Channel 3 Relative difference in SKY_RESIDUAL:  0.5714527578778785
        
        # Channel 4 Relative difference in SKY_MODEL:  0.0012122482123595138
        # Channel 4 Relative difference in SKY_RESIDUAL:  0.5615437885823942
        
        
# Channel 0 Relative difference in SKY_MODEL:  2.0388167939156927e-05
# Channel 0 Relative difference in SKY_RESIDUAL:  0.11555623461263295
# Channel 1 Relative difference in SKY_MODEL:  0.0009789482204398345
# Channel 1 Relative difference in SKY_RESIDUAL:  0.06962187922617491
# Channel 2 Relative difference in SKY_MODEL:  7.67860174649184e-06
# Channel 2 Relative difference in SKY_RESIDUAL:  0.087915044478974
# Channel 3 Relative difference in SKY_MODEL:  0.0021216647131103673
# Channel 3 Relative difference in SKY_RESIDUAL:  0.05012888138713684
# Channel 4 Relative difference in SKY_MODEL:  0.0012122482123601821
# Channel 4 Relative difference in SKY_RESIDUAL:  0.04491952484766548

    #plt.show()

    # psf_av = img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values
    # psf = img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values

    # print(img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values)
    # print(
    #     img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values * 2
    # )
    # print(
    #     img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values * 2
    #     + 2 * np.pi
    # )

    # psf_ref = [
    #     [
    #         [3.44428887e-06, 2.55786300e-06, 2.06175939e00],
    #         [3.09923287e-06, 2.32401691e-06, 2.22554985e00],
    #         [3.09922720e-06, 2.32401255e-06, 2.22554912e00],
    #         [3.09920517e-06, 2.32401540e-06, 2.22556126e00],
    #         [3.09909389e-06, 2.32392809e-06, 2.22633490e00],
    #     ]
    # ]
    # assert np.allclose(
    #     img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values,
    #     psf_ref,
    #     rtol=1e-6,
    # ), "You broke something! The beam fit parameters for the point spread function are not close enough to the reference values."
    
    
    # assert np.allclose(
    #     img_av_xds.SKY_MODEL.values,
    #     img_av_compare_xds.SKY_MODEL.values,
    #     rtol=1e-6,
    # ), "You broke something! The sky model values are not close enough to the reference values."
    # img_av_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_niter_99_astroviper.img.zarr")
    # img_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_niter_99_casa.img.zarr")
    
    # for i_f in range(5):
    #     fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    #     im0 = axes[0,0].imshow(img_av_xds["SKY_MODEL"].isel(frequency=i_f, time=0, polarization=0).values)
    #     axes[0,0].set_title("SKY_MODEL Av")
    #     fig.colorbar(im0, ax=axes[0,0])
    #     im1 = axes[0,1].imshow(img_xds["SKY_MODEL"].isel(frequency=i_f, time=0, polarization=0).values)
    #     axes[0,1].set_title("SKY_MODEL Casa")
    #     fig.colorbar(im1, ax=axes[0,1])
    #     im2 = axes[0,2].imshow(
    #         img_av_xds["SKY_MODEL"].isel(frequency=i_f, time=0, polarization=0).values-img_xds["SKY_MODEL"].isel(frequency=i_f, time=0, polarization=0).values
    #     )
    #     axes[0,2].set_title("SKY_MODEL Av - Casa")
    #     fig.colorbar(im2, ax=axes[0,2]) 

    # plt.show()
    
    
    # print(ps_xdt.xr_ps.summary())
    
    
    # from astroviper.core.imaging.fft_normalize_prolate_spheriodal_gridder import (
    #     fft_lm_to_uv, ifft_uv_to_lm
    # )
    
    # psf = img_xds["POINT_SPREAD_FUNCTION"].values
    # uv_sampling = fft_lm_to_uv(
    #     psf
    # )
    # psf_rt = ifft_uv_to_lm(uv_sampling).real
    
    # plt.figure()
    # plt.imshow(
    #     psf[0,0,0,:,:]
    # )
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(
    #     psf_rt[0,0,0,:,:]
    # )
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(
    #     (psf_rt - psf)[0,0,0,:,:]
    # )
    # plt.colorbar()
    
    
    # plt.show()
    
    
    # plt.figure()
    # # plt.imshow((img_av_xds["SKY_RESIDUAL"].isel(frequency=frequency,time=0, polarization=0)+img_av_xds["SKY_RESIDUAL"].isel(frequency=frequency,time=0, polarization=1))/2)
    # plt.imshow(
    #     img_av_xds["SKY_MODEL"].isel(frequency=frequency, time=0, polarization=0)
    # )
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(
    #     img_av_xds["SKY_RESIDUAL"].isel(frequency=frequency, time=0, polarization=0)
    # )
    # plt.colorbar()
    
    
        
    # plt.figure()
    # plt.imshow(
    #     img_xds["SKY_RESIDUAL"].isel(frequency=frequency, time=0, polarization=0)
    # )
    # plt.colorbar()
    
    # plt.show()
    
    # print(img_av_xds.attrs["data_groups"].keys())
    # print(img_av_xds.attrs["data_groups"])
    # print(img_av_xds.data_vars)
    
    # plt.figure()
    # plt.imshow(
    #     img_av_xds["POINT_SPREAD_FUNCTION"].isel(frequency=frequency, time=0, polarization=0)
    # )
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(
    #     img_xds["POINT_SPREAD_FUNCTION"].isel(frequency=frequency, time=0, polarization=0)
    # )
    # plt.colorbar()
    
    # plt.show()

    # print(img_xds)


if __name__ == "__main__":
    test_single_field_imaging_niter0()
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"*10)
    test_single_field_imaging()
    
    # import xarray as xr
    # img_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_niter_99_astroviper.img.zarr")
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(img_xds["MASK"].isel(frequency=0, time=0, polarization=0).values)
    # plt.colorbar()
    # plt.show()  


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


