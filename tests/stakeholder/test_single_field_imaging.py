"""Stakeholder tests for single-field imaging.

These tests run ``image_cube_single_field`` and compare the astroviper output
against pre-computed CASA reference images that live alongside this file.
They must therefore be run from this directory so the relative ``*.zarr``
paths resolve.

Run with pytest::

    conda run -n zinc pytest test_single_field_imaging.py

Set ``SAVE_PLOTS=1`` to write the per-channel comparison plots to disk
(directory controlled by ``PLOT_DIR``, default ``plots``)::

    SAVE_PLOTS=1 PLOT_DIR=/tmp/plots conda run -n zinc pytest test_single_field_imaging.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so figures never block pytest

import dask
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from astroviper.distributed_graphs.imaging.image_cube_single_field import (
    image_cube_single_field,
)
from astroviper.processing_functions.imaging.make_pb_symmetric import (
    airy_disk_rorder,
    airy_disk_rorder_v2,
)
from xradio.measurement_set import open_processing_set

PS_STORE = "twhya_selfcal_lsrk_5chans.ps.zarr"


def make_plot_saver():
    """Build the ``save(fig, name)`` callable used by the tests to emit plots.

    Figures are written only when ``SAVE_PLOTS`` is truthy (``PLOT_DIR``
    selects the output directory, default ``plots``); they are always closed
    afterwards so a full run does not accumulate open figures.

    Exposed as a plain function (rather than only the ``plot_saver`` fixture)
    so the ``__main__`` block can run the tests under plain ``python`` too.
    """
    save = os.environ.get("SAVE_PLOTS", "").lower() in ("1", "true", "yes", "on")
    save = True
    out_dir = os.environ.get("PLOT_DIR", "plots")
    if save:
        os.makedirs(out_dir, exist_ok=True)

    def _save(fig, name):
        if save:
            fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return _save


@pytest.fixture
def plot_saver():
    """Pytest fixture providing the plot saver (see ``make_plot_saver``)."""
    return make_plot_saver()


def _image_params(ps_xdt):
    """Build the shared imaging parameters from the processing set."""
    combined_field_and_source_xds = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
    phase_direction = combined_field_and_source_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=center_field_name
    )
    return {
        "image_size": [250, 250],
        "cell_size": np.array([-0.1, 0.1]) * np.pi / (180 * 3600),
        "phase_direction": phase_direction.values,
        "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }


def test_single_field_imaging_niter0(plot_saver):
    dask.config.set(scheduler="synchronous")

    ps_xdt = open_processing_set(PS_STORE)
    img_xds = xr.open_zarr(
        "twhya_selfcal_5chans_lsrk_niter_0_nmajor_1_briggs.img.zarr"
    )

    image_params = _image_params(ps_xdt)
    image_store = "twhya_selfcal_lsrk_5chans_astroviper.img.zarr"

    imaging_metadata_pd = image_cube_single_field(
        ps_store=PS_STORE,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params={
            "weighting": "briggs",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        iteration_control_params={
            "niter": 0,
            "nmajor": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
        },
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
        scan_intents="OBSERVE_TARGET#ON_SOURCE",
        image_data_variables_keep=[
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
            "visibility_normalization",
            "uv_sampling_normalization",
        ],
        processing_set_data_group_name="base",
        double_precision=True,
        thread_info=None,
        processing_function_threads=1,
        n_chunks=None,
        overwrite=True,
        disk_chunk_sizes={"frequency": 2},
        vizualize_graph=True,
    )
    img_av_xds = xr.open_zarr(image_store)
    print("&&&&&&&&&"*10)
    print("imaging_metadata_pd", imaging_metadata_pd)


    # First pass: generate every channel's comparison plot and record the
    # relative differences. Assertions are deferred to a second pass below so
    # that a failure on one channel cannot prevent the remaining plots from
    # being saved.
    polarization = 0
    channel_rel_diffs = []
    for i_f in range(5):
        I = img_xds.isel(frequency=i_f, polarization=polarization, time=0)[
            "SKY_DECONVOLVED"
        ]
        PSF = img_xds.isel(frequency=i_f, polarization=polarization, time=0)[
            "POINT_SPREAD_FUNCTION"
        ]
        I_av = img_av_xds.isel(frequency=i_f, polarization=polarization, time=0)[
            "SKY_RESIDUAL"
        ]
        PSF_av = img_av_xds.isel(frequency=i_f, polarization=polarization, time=0)[
            "POINT_SPREAD_FUNCTION"
        ]

        rel_diff = np.max(np.abs(I_av.values - I.values) / np.max(np.abs(I.values)))
        channel_rel_diffs.append(rel_diff)
        print(f"Channel {i_f} relative difference: {rel_diff}")

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f"Channel {i_f} Frequency: {I_av.frequency.values}")
        im0 = axes[0, 0].imshow(I_av.values)
        axes[0, 0].set_title("I_AV")
        fig.colorbar(im0, ax=axes[0, 0])
        im1 = axes[0, 1].imshow(I.values)
        axes[0, 1].set_title("I_CASA")
        fig.colorbar(im1, ax=axes[0, 1])
        im2 = axes[0, 2].imshow(I_av.values - I.values)
        axes[0, 2].set_title("I_AV - I_CASA")
        fig.colorbar(im2, ax=axes[0, 2])
        im3 = axes[1, 0].imshow(PSF_av.values)
        axes[1, 0].set_title("PSF_AV")
        fig.colorbar(im3, ax=axes[1, 0])
        im4 = axes[1, 1].imshow(PSF.values)
        axes[1, 1].set_title("PSF_CASA")
        fig.colorbar(im4, ax=axes[1, 1])
        im5 = axes[1, 2].imshow(PSF_av.values - PSF.values)
        axes[1, 2].set_title("PSF_AV - PSF_CASA")
        fig.colorbar(im5, ax=axes[1, 2])
        plot_saver(fig, f"niter0_channel_{i_f}.png")

    # Second pass: assertions only (all plots have already been generated).
    # Even channels must match the reference more tightly than odd channels.
    for i_f, rel_diff in enumerate(channel_rel_diffs):
        tol = 1e-4 if i_f % 2 == 0 else 1e-3
        assert rel_diff < tol, (
            f"Channel {i_f}: relative difference {rel_diff} exceeds tolerance "
            f"{tol}. You broke something!"
        )

    psf_ref = [
        [
            [3.09938237e-06, 2.32409432e-06, 2.22543281e00],
            [3.09927976e-06, 2.32408319e-06, 2.22547993e00],
            [3.09916079e-06, 2.32406462e-06, 2.22536627e00],
            [3.09916724e-06, 2.32405466e-06, 2.22535443e00],
            [3.09916185e-06, 2.32405054e-06, 2.22535528e00],
        ]
    ]
    assert np.allclose(
        img_av_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I").values,
        psf_ref,
        rtol=1e-6,
    ), "Beam fit parameters for the point spread function differ from the reference."

    # The two airy-disk implementations should agree.
    pb_parms = {
        "list_dish_diameters": np.array([10.7]),
        "list_blockage_diameters": np.array([0.75]),
        "ipower": 1,
    }
    image_params["image_center"] = np.array(image_params["image_size"]) // 2

    # Select the first (only) dish diameter and add a leading time axis.
    PB_v2 = airy_disk_rorder_v2(
        img_xds.frequency.values,
        img_xds.polarization.values,
        pb_parms,
        image_params,
    )[0, ...][None, ...]
    PB_v1 = airy_disk_rorder(
        img_xds.frequency.values,
        img_xds.polarization.values,
        pb_parms,
        image_params,
    )[0, ...][None, ...]
    assert np.allclose(
        PB_v1, PB_v2
    ), "airy_disk_rorder and airy_disk_rorder_v2 produced different primary beams."


def test_single_field_imaging(plot_saver):
    dask.config.set(scheduler="synchronous")

    ps_xdt = open_processing_set(PS_STORE)
    img_xds = xr.open_zarr(
        "twhya_selfcal_5chans_lsrk_niter_99_nmajor_1_briggs.img.zarr"
    )

    image_params = _image_params(ps_xdt)
    image_store = "twhya_selfcal_5chans_lsrk_niter_99_astroviper.img.zarr"

    imaging_metadata_pd = image_cube_single_field(
        ps_store=PS_STORE,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params={
            "weighting": "briggs",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        iteration_control_params={
            "niter": 100,
            "nmajor": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.2,
        },
        # iteration_control_params={
        #     "niter": 10000,
        #     "nmajor": 10,
        #     "threshold": 0.0,
        #     "gain": 0.1,
        #     "cyclefactor": 1.5,
        #     "cycleniter": 1,
        #     "minpsffraction": 0.05,
        #     "maxpsffraction": 0.8,
        # },
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
        #deconvolver="asp",
        scan_intents="OBSERVE_TARGET#ON_SOURCE",
        image_data_variables_keep=[
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
            "sky_model",
            "mask",
        ],
        processing_set_data_group_name="base",
        double_precision=False,
        thread_info=None,
        processing_function_threads=12,
        n_chunks=1,
        overwrite=True,
        disk_chunk_sizes={"frequency": 5},
        vizualize_graph=True,
        write_visibility_model_to_ps=True,
        fft_backend="scipy",
    )
    img_av_xds = xr.open_zarr(image_store)
    print("&&&&&&&&&"*10)
    print("imaging_metadata_pd:")
    print(imaging_metadata_pd.to_string())

    polarization = 0
    region = dict(polarization=polarization, time=0, l=slice(100, 150), m=slice(100, 150))
    I_av = img_av_xds.isel(**region)
    I = img_xds.isel(**region)

    # Expected upper bound on the per-channel relative difference vs CASA.
    # Exceeding the maximum is a regression; dropping below the minimum is an
    # (informational) improvement and is reported but not treated as a failure.
    max_per_dif_model = [2.1e-05, 0.001, 7.7e-06, 0.0022, 0.0013]
    min_per_dif_model = [2.0e-05, 0.0009, 7.6e-06, 0.0021, 0.0012]
    max_per_dif_residual = [8.949e-05, 0.0331, 0.000122, 0.001684, 0.001123]
    min_per_dif_residual = [8.946e-05, 0.033, 0.000121, 0.001682, 0.001122]

    # First pass: generate every channel's comparison plot and record the
    # relative differences. Assertions are deferred to a second pass below so
    # that a failure on one channel cannot prevent the remaining plots from
    # being saved.
    channel_rel_diffs = []
    for i_f in range(5):
        model_av = I_av["SKY_MODEL"].isel(frequency=i_f).values
        model_casa = I["SKY_MODEL"].isel(frequency=i_f).values
        residual_av = I_av["SKY_RESIDUAL"].isel(frequency=i_f).values
        residual_casa = I["SKY_RESIDUAL"].isel(frequency=i_f).values
        pb_av = I_av["PRIMARY_BEAM"].isel(frequency=i_f).values
        pb_casa = I["PRIMARY_BEAM"].isel(frequency=i_f).values

        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f"Channel {i_f} Frequency: {I_av.frequency.values[i_f]}")
        im0 = axes[0, 0].imshow(model_av)
        axes[0, 0].set_title("MODEL_AV")
        fig.colorbar(im0, ax=axes[0, 0])
        im1 = axes[0, 1].imshow(model_casa)
        axes[0, 1].set_title("MODEL_CASA")
        fig.colorbar(im1, ax=axes[0, 1])
        im2 = axes[0, 2].imshow(
            100 * (model_av - model_casa) / np.max(np.abs(model_casa))
        )
        axes[0, 2].set_title("Percentage Difference 100%")
        fig.colorbar(im2, ax=axes[0, 2])

        im3 = axes[1, 0].imshow(residual_av)
        axes[1, 0].set_title("RESIDUAL_AV")
        fig.colorbar(im3, ax=axes[1, 0])
        im4 = axes[1, 1].imshow(residual_casa)
        axes[1, 1].set_title("RESIDUAL_CASA")
        fig.colorbar(im4, ax=axes[1, 1])
        im5 = axes[1, 2].imshow(
            100 * (residual_av - residual_casa) / np.max(np.abs(residual_casa))
        )
        axes[1, 2].set_title("Percentage Difference 100%")
        fig.colorbar(im5, ax=axes[1, 2])

        im6 = axes[2, 0].imshow(pb_av)
        axes[2, 0].set_title("PB_AV")
        fig.colorbar(im6, ax=axes[2, 0])
        im7 = axes[2, 1].imshow(pb_casa)
        axes[2, 1].set_title("PB_CASA")
        fig.colorbar(im7, ax=axes[2, 1])
        im8 = axes[2, 2].imshow(100 * (pb_av - pb_casa) / np.max(np.abs(pb_casa)))
        axes[2, 2].set_title("Percentage Difference 100%")
        fig.colorbar(im8, ax=axes[2, 2])
        plot_saver(fig, f"niter99_channel_{i_f}.png")

        rel_diff_model = np.max(
            np.abs(model_av - model_casa) / np.max(np.abs(model_casa))
        )
        rel_diff_residual = np.max(
            np.abs(residual_av - residual_casa) / np.max(np.abs(residual_casa))
        )
        channel_rel_diffs.append((rel_diff_model, rel_diff_residual))
        print(f"Channel {i_f} relative difference in SKY_MODEL: {rel_diff_model}")
        print(f"Channel {i_f} relative difference in SKY_RESIDUAL: {rel_diff_residual}")

        if rel_diff_model < min_per_dif_model[i_f]:
            print(
                f"Channel {i_f} SKY_MODEL improved: {rel_diff_model} is below the "
                f"expected minimum {min_per_dif_model[i_f]} (consider tightening)."
            )
        if rel_diff_residual < min_per_dif_residual[i_f]:
            print(
                f"Channel {i_f} SKY_RESIDUAL improved: {rel_diff_residual} is below "
                f"the expected minimum {min_per_dif_residual[i_f]} (consider tightening)."
            )

    # Second pass: assertions only (all plots have already been generated).
    for i_f, (rel_diff_model, rel_diff_residual) in enumerate(channel_rel_diffs):
        assert rel_diff_model <= max_per_dif_model[i_f], (
            f"Channel {i_f}: SKY_MODEL relative difference {rel_diff_model} exceeds "
            f"the maximum {max_per_dif_model[i_f]}. You broke something!"
        )
        assert rel_diff_residual <= max_per_dif_residual[i_f], (
            f"Channel {i_f}: SKY_RESIDUAL relative difference {rel_diff_residual} "
            f"exceeds the maximum {max_per_dif_residual[i_f]}. You broke something!"
        )



def test_single_field_imaging_multi_cycle(plot_saver):
    dask.config.set(scheduler="synchronous")

    ps_xdt = open_processing_set(PS_STORE)
    img_xds = xr.open_zarr(
        "twhya_selfcal_5chans_lsrk_niter_99_nmajor_1_briggs.img.zarr"
    )

    image_params = _image_params(ps_xdt)
    image_store = "twhya_selfcal_5chans_lsrk_niter_99_astroviper.img.zarr"

    imaging_metadata_pd = image_cube_single_field(
        ps_store=PS_STORE,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params={
            "weighting": "briggs",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        iteration_control_params={
            "niter": 1000,
            "nmajor": 10,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
        },
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
        #deconvolver="asp",
        scan_intents="OBSERVE_TARGET#ON_SOURCE",
        image_data_variables_keep=[
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
            "sky_model",
            "mask",
        ],
        processing_set_data_group_name="base",
        double_precision=False,
        thread_info=None,
        processing_function_threads=12,
        n_chunks=1,
        overwrite=True,
        disk_chunk_sizes={"frequency": 5},
        vizualize_graph=True,
        write_visibility_model_to_ps=True,
        fft_backend="scipy",
    )
    img_av_xds = xr.open_zarr(image_store)
    print("&&&&&&&&&"*10)
    print("imaging_metadata_pd:")
    print(imaging_metadata_pd.to_string())

    polarization = 0
    region = dict(polarization=polarization, time=0, l=slice(100, 150), m=slice(100, 150))
    I_av = img_av_xds.isel(**region)
    I = img_xds.isel(**region)

    # Expected upper bound on the per-channel relative difference vs CASA.
    # Exceeding the maximum is a regression; dropping below the minimum is an
    # (informational) improvement and is reported but not treated as a failure.
    max_per_dif_model = [2.1e-05, 0.001, 7.7e-06, 0.0022, 0.0013]
    min_per_dif_model = [2.0e-05, 0.0009, 7.6e-06, 0.0021, 0.0012]
    max_per_dif_residual = [8.949e-05, 0.0331, 0.000122, 0.001684, 0.001123]
    min_per_dif_residual = [8.946e-05, 0.033, 0.000121, 0.001682, 0.001122]

    # First pass: generate every channel's comparison plot and record the
    # relative differences. Assertions are deferred to a second pass below so
    # that a failure on one channel cannot prevent the remaining plots from
    # being saved.
    channel_rel_diffs = []
    for i_f in range(5):
        model_av = I_av["SKY_MODEL"].isel(frequency=i_f).values
        model_casa = I["SKY_MODEL"].isel(frequency=i_f).values
        residual_av = I_av["SKY_RESIDUAL"].isel(frequency=i_f).values
        residual_casa = I["SKY_RESIDUAL"].isel(frequency=i_f).values
        pb_av = I_av["PRIMARY_BEAM"].isel(frequency=i_f).values
        pb_casa = I["PRIMARY_BEAM"].isel(frequency=i_f).values

        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f"Channel {i_f} Frequency: {I_av.frequency.values[i_f]}")
        im0 = axes[0, 0].imshow(model_av)
        axes[0, 0].set_title("MODEL_AV")
        fig.colorbar(im0, ax=axes[0, 0])
        im1 = axes[0, 1].imshow(model_casa)
        axes[0, 1].set_title("MODEL_CASA")
        fig.colorbar(im1, ax=axes[0, 1])
        im2 = axes[0, 2].imshow(
            100 * (model_av - model_casa) / np.max(np.abs(model_casa))
        )
        axes[0, 2].set_title("Percentage Difference 100%")
        fig.colorbar(im2, ax=axes[0, 2])

        im3 = axes[1, 0].imshow(residual_av)
        axes[1, 0].set_title("RESIDUAL_AV")
        fig.colorbar(im3, ax=axes[1, 0])
        im4 = axes[1, 1].imshow(residual_casa)
        axes[1, 1].set_title("RESIDUAL_CASA")
        fig.colorbar(im4, ax=axes[1, 1])
        im5 = axes[1, 2].imshow(
            100 * (residual_av - residual_casa) / np.max(np.abs(residual_casa))
        )
        axes[1, 2].set_title("Percentage Difference 100%")
        fig.colorbar(im5, ax=axes[1, 2])

        im6 = axes[2, 0].imshow(pb_av)
        axes[2, 0].set_title("PB_AV")
        fig.colorbar(im6, ax=axes[2, 0])
        im7 = axes[2, 1].imshow(pb_casa)
        axes[2, 1].set_title("PB_CASA")
        fig.colorbar(im7, ax=axes[2, 1])
        im8 = axes[2, 2].imshow(100 * (pb_av - pb_casa) / np.max(np.abs(pb_casa)))
        axes[2, 2].set_title("Percentage Difference 100%")
        fig.colorbar(im8, ax=axes[2, 2])
        plot_saver(fig, f"multi_cycle_channel_{i_f}.png")

        rel_diff_model = np.max(
            np.abs(model_av - model_casa) / np.max(np.abs(model_casa))
        )
        rel_diff_residual = np.max(
            np.abs(residual_av - residual_casa) / np.max(np.abs(residual_casa))
        )
        channel_rel_diffs.append((rel_diff_model, rel_diff_residual))
        print(f"Channel {i_f} relative difference in SKY_MODEL: {rel_diff_model}")
        print(f"Channel {i_f} relative difference in SKY_RESIDUAL: {rel_diff_residual}")

        if rel_diff_model < min_per_dif_model[i_f]:
            print(
                f"Channel {i_f} SKY_MODEL improved: {rel_diff_model} is below the "
                f"expected minimum {min_per_dif_model[i_f]} (consider tightening)."
            )
        if rel_diff_residual < min_per_dif_residual[i_f]:
            print(
                f"Channel {i_f} SKY_RESIDUAL improved: {rel_diff_residual} is below "
                f"the expected minimum {min_per_dif_residual[i_f]} (consider tightening)."
            )

    # Second pass: assertions only (all plots have already been generated).
    for i_f, (rel_diff_model, rel_diff_residual) in enumerate(channel_rel_diffs):
        assert rel_diff_model <= max_per_dif_model[i_f], (
            f"Channel {i_f}: SKY_MODEL relative difference {rel_diff_model} exceeds "
            f"the maximum {max_per_dif_model[i_f]}. You broke something!"
        )
        assert rel_diff_residual <= max_per_dif_residual[i_f], (
            f"Channel {i_f}: SKY_RESIDUAL relative difference {rel_diff_residual} "
            f"exceeds the maximum {max_per_dif_residual[i_f]}. You broke something!"
        )





if __name__ == "__main__":
    test_single_field_imaging_niter0(make_plot_saver())
    print("************"*10)
    test_single_field_imaging(make_plot_saver())
    print("************"*10)
    test_single_field_imaging_multi_cycle(make_plot_saver())