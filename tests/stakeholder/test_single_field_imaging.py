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
from astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric import (
    airy_disk_rorder,
    airy_disk_rorder_v2,
)
from astroviper.processing_functions.imaging.iteration_control import (
    print_deconvolve_dict,
)
from xradio.measurement_set import open_processing_set

PS_STORE = "twhya_selfcal_lsrk_5chans.ps.zarr"

# Google Drive file id for the zipped input processing set (``PS_STORE``).
_PS_STORE_DRIVE_ID = "11rMeV1uWiFNL8hrLU3pozSTBjkCDfl_4"

# Google Drive file IDs for the zipped CASA reference (truth) images. Each is a
# zipped ``.img.zarr`` directory.
_TRUTH_IMAGE_DRIVE_IDS = {
    "twhya_selfcal_5chans_lsrk_multi_cycle_truth.img.zarr": "1lvMKzqgEcH6kRJjx-2U3MxA_Aj6uWPt_",
    "twhya_selfcal_5chans_lsrk_niter_0_nmajor_1_briggs.img.zarr": "12QF0hbfVcgY_Mo8jKRQC0FaAG3Q4pf8_",
    "twhya_selfcal_5chans_lsrk_niter_99_nmajor_1_briggs.img.zarr": "1SqTUd6V3pMRm07LM2dP2seawHb2jUFH_",
}


def _import_gdown():
    """Import :mod:`gdown`, installing it on first use if necessary."""
    try:
        import gdown
    except ImportError:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown

    return gdown


def _download_zarr(zarr_name, file_id):
    """Download and extract one zipped ``.zarr`` directory from Google Drive.

    Used for both the input processing set (``PS_STORE``) and the CASA
    reference (truth) images. The archives are uploaded as zipped ``.zarr``
    directories, but may arrive *double-zipped* (a ``.zip`` whose only member is
    another ``.zip``). This helper extracts into a scratch directory, recursively
    unpacks any nested ``.zip`` it finds, locates the ``<zarr_name>`` directory
    wherever it lands and moves it into place. It is a no-op when the directory
    already exists locally, so it never re-downloads or clobbers a local copy.
    """
    import glob
    import shutil
    import zipfile

    if os.path.isdir(zarr_name):
        return  # already present locally -- nothing to download.

    gdown = _import_gdown()
    zip_path = zarr_name + ".zip"
    # Use the file id (rather than the browser "view" URL) so gdown fetches the
    # binary directly instead of an HTML page.
    gdown.download(id=file_id, output=zip_path, quiet=False)

    work_dir = zarr_name + ".extract"
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)
    shutil.move(zip_path, os.path.join(work_dir, os.path.basename(zip_path)))

    # Unwrap nested zips until the target directory appears (handles single- and
    # double-zipped archives, with or without a leading folder).
    for _ in range(6):  # safety bound against malformed archives
        for root, dirs, _files in os.walk(work_dir):
            if zarr_name in dirs:
                shutil.move(os.path.join(root, zarr_name), zarr_name)
                shutil.rmtree(work_dir, ignore_errors=True)
                return
        nested_zips = glob.glob(os.path.join(work_dir, "**", "*.zip"), recursive=True)
        if not nested_zips:
            break
        for nested in nested_zips:
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(os.path.dirname(nested))
            os.remove(nested)

    shutil.rmtree(work_dir, ignore_errors=True)
    raise RuntimeError(
        f"Could not extract the '{zarr_name}' reference image from its archive."
    )


def _ensure_ps_store():
    """Ensure the input processing set (``PS_STORE``) is present, downloading once."""
    _download_zarr(PS_STORE, _PS_STORE_DRIVE_ID)


def _ensure_truth_images():
    """Ensure every reference (truth) image is present, downloading if absent."""
    for zarr_name, file_id in _TRUTH_IMAGE_DRIVE_IDS.items():
        _download_zarr(zarr_name, file_id)


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


def _check_deconvolve_dict(deconvolve_dict, expected, rtol=1e-5, atol=1e-8):
    """Assert every key and field of a deconvolve ReturnDict matches ``expected``.

    ``expected`` is keyed by ``(time, pol, chan)`` tuples, with ``stop_code``
    given as a ``(major, minor)`` tuple. Integer fields (niter, iter_done,
    masksum), string fields (stokes, stop_description) and the stop code are
    compared exactly; every other (floating-point) field -- scalar or per-cycle
    history list -- is compared with ``np.allclose``. Regenerate the expected
    literals if the imaging or iteration-control behaviour intentionally changes.
    """
    exact_int_fields = {"niter", "iter_done", "masksum"}
    string_fields = {"stokes", "stop_description"}

    actual = {tuple(k): v for k, v in deconvolve_dict.data.items()}
    assert set(actual) == set(
        expected
    ), f"deconvolve_dict planes {sorted(actual)} != expected {sorted(expected)}"
    for key, exp_fields in expected.items():
        got = actual[key]
        assert set(got) == set(
            exp_fields
        ), f"plane {key}: fields {sorted(got)} != expected {sorted(exp_fields)}"
        for field, exp_val in exp_fields.items():
            val = got[field]
            if field == "stop_code":
                assert (int(val.major), int(val.minor)) == tuple(
                    exp_val
                ), f"plane {key} stop_code {val} != {exp_val}"
            elif field in string_fields:
                assert (
                    str(val) == exp_val
                ), f"plane {key} {field}: {val!r} != {exp_val!r}"
            elif field in exact_int_fields:
                assert np.array_equal(
                    np.asarray(val), np.asarray(exp_val)
                ), f"plane {key} {field}: {val} != {exp_val}"
            else:
                assert np.allclose(
                    np.asarray(val, dtype=float),
                    np.asarray(exp_val, dtype=float),
                    rtol=rtol,
                    atol=atol,
                ), f"plane {key} {field}: {val} != {exp_val}"


def test_single_field_imaging_niter0(plot_saver):
    dask.config.set(scheduler="synchronous")

    _ensure_ps_store()
    _ensure_truth_images()
    ps_xdt = open_processing_set(PS_STORE)
    img_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_niter_0_nmajor_1_briggs.img.zarr")

    image_params = _image_params(ps_xdt)
    image_store = "twhya_selfcal_lsrk_5chans_astroviper.img.zarr"

    return_dict = image_cube_single_field(
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
    imaging_metadata_pd = return_dict["timing"]
    deconvolve_dict = return_dict["deconvolution"]
    img_av_xds = xr.open_zarr(image_store)
    print("&&&&&&&&&" * 10)
    print("imaging_metadata_pd", imaging_metadata_pd)
    print("deconvolve_dict (global channel numbering):")
    print_deconvolve_dict(deconvolve_dict)

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

    _ensure_ps_store()
    _ensure_truth_images()
    ps_xdt = open_processing_set(PS_STORE)
    img_xds = xr.open_zarr(
        "twhya_selfcal_5chans_lsrk_niter_99_nmajor_1_briggs.img.zarr"
    )

    image_params = _image_params(ps_xdt)
    image_store = "twhya_selfcal_5chans_lsrk_niter_99_astroviper.img.zarr"

    return_dict = image_cube_single_field(
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
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
        # deconvolver="asp",
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
    imaging_metadata_pd = return_dict["timing"]
    deconvolve_dict = return_dict["deconvolution"]
    img_av_xds = xr.open_zarr(image_store)
    print("&&&&&&&&&" * 10)
    print("imaging_metadata_pd:")
    print(imaging_metadata_pd.to_string())
    print("deconvolve_dict (global channel numbering):")
    print_deconvolve_dict(deconvolve_dict)

    # Per-plane deconvolution statistics for this niter=100, nmajor=0 run (one
    # major cycle, stopped on the iteration limit -> stop_code (1, 0)).
    expected_deconvolve_dict = {
        (0, 0, 0): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.3445475697517395,
            "stop_code": (1, 0),
            "stokes": "I",
            "frequency": 372762580492.5155,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.9266056708712089],
            "start_peakres": [0.35750773549079895],
            "start_peakres_nomask": [0.35750773549079895],
            "peakres": [0.09062794993320043],
            "peakres_nomask": [0.09062794993320043],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 0): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.3445475697517395,
            "stop_code": (1, 0),
            "stokes": "Q",
            "frequency": 372762580492.5155,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.06009334377062642],
            "start_peakres": [0.1205507144331932],
            "start_peakres_nomask": [0.1205507144331932],
            "peakres": [0.07445588224991798],
            "peakres_nomask": [0.07445588224991798],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 1): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.34453392028808594,
            "stop_code": (1, 0),
            "stokes": "I",
            "frequency": 372763190875.0631,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.8096181158874369],
            "start_peakres": [0.3351074755191803],
            "start_peakres_nomask": [0.3351074755191803],
            "peakres": [0.09003297085726052],
            "peakres_nomask": [0.09003297085726052],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 1): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.34453392028808594,
            "stop_code": (1, 0),
            "stokes": "Q",
            "frequency": 372763190875.0631,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.06434190144351142],
            "start_peakres": [0.1182534508407116],
            "start_peakres_nomask": [0.1182534508407116],
            "peakres": [0.07756869577706144],
            "peakres_nomask": [0.07756869577706144],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 2): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.34448084235191345,
            "stop_code": (1, 0),
            "stokes": "I",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.9100114705555141],
            "start_peakres": [0.3380203992128372],
            "start_peakres_nomask": [0.3380203992128372],
            "peakres": [0.0897947973663456],
            "peakres_nomask": [0.0897947973663456],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 2): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.34448084235191345,
            "stop_code": (1, 0),
            "stokes": "Q",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.07577367670663074],
            "start_peakres": [0.1301141269505024],
            "start_peakres_nomask": [0.1301141269505024],
            "peakres": [0.07658878708058861],
            "peakres_nomask": [0.07658878708058861],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 3): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.3444790840148926,
            "stop_code": (1, 0),
            "stokes": "I",
            "frequency": 372764411640.15845,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.7685137758888749],
            "start_peakres": [0.3250153809785843],
            "start_peakres_nomask": [0.3250153809785843],
            "peakres": [0.09214768756550326],
            "peakres_nomask": [0.09214768756550326],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 3): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.3444790840148926,
            "stop_code": (1, 0),
            "stokes": "Q",
            "frequency": 372764411640.15845,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.051890125145152254],
            "start_peakres": [0.13275842368602753],
            "start_peakres_nomask": [0.13275842368602753],
            "peakres": [0.07897557256149514],
            "peakres_nomask": [0.07897557256149514],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 4): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.3444778323173523,
            "stop_code": (1, 0),
            "stokes": "I",
            "frequency": 372765022022.7062,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.685908240144164],
            "start_peakres": [0.3533458411693573],
            "start_peakres_nomask": [0.3533458411693573],
            "peakres": [0.08926753026602027],
            "peakres_nomask": [0.08926753026602027],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 4): {
            "niter": 100,
            "threshold": 0.0715015470981598,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.3444778323173523,
            "stop_code": (1, 0),
            "stokes": "Q",
            "frequency": 372765022022.7062,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [-0.1185715380641539],
            "start_peakres": [0.11055046692490578],
            "start_peakres_nomask": [0.11055046692490578],
            "peakres": [0.07679465310366025],
            "peakres_nomask": [0.07679465310366025],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
    }
    _check_deconvolve_dict(deconvolve_dict, expected_deconvolve_dict)

    polarization = 0
    region = dict(
        polarization=polarization, time=0, l=slice(100, 150), m=slice(100, 150)
    )
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
                f"expected minimum {min_per_dif_model[i_f]}."
            )
        if rel_diff_residual < min_per_dif_residual[i_f]:
            print(
                f"Channel {i_f} SKY_RESIDUAL improved: {rel_diff_residual} is below "
                f"the expected minimum {min_per_dif_residual[i_f]}."
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

    _ensure_ps_store()
    _ensure_truth_images()
    ps_xdt = open_processing_set(PS_STORE)
    img_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_multi_cycle_truth.img.zarr")

    image_params = _image_params(ps_xdt)
    image_store = "twhya_selfcal_5chans_lsrk_multi_cycle_astroviper.img.zarr"

    return_dict = image_cube_single_field(
        ps_store=PS_STORE,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params={
            "weighting": "briggs",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        iteration_control_params={
            "niter": 10000,
            "nmajor": 4,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
        },
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
        # deconvolver="asp",
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
    imaging_metadata_pd = return_dict["timing"]
    deconvolve_dict = return_dict["deconvolution"]
    img_av_xds = xr.open_zarr(image_store)
    print("&&&&&&&&&" * 10)
    print("imaging_metadata_pd:")
    print(imaging_metadata_pd.to_string())
    print("deconvolve_dict (global channel numbering):")
    print_deconvolve_dict(deconvolve_dict)

    # Per-plane deconvolution statistics for this niter=10000, nmajor=4 run. Each
    # history list has one entry per major cycle; the run stops on the major
    # cycle limit -> stop_code (9, 0).
    expected_deconvolve_dict = {
        (0, 0, 0): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [3, 21, 273, 1517],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3445475697517395,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372762580492.5155,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.09718702917164579,
                0.4688002345071223,
                1.1537604726241149,
            ],
            "model_flux": [
                0.09718702917164579,
                0.4688002345071223,
                1.1537604726241149,
                1.847822861355533,
            ],
            "start_peakres": [
                0.35750773549079895,
                0.26311833411455154,
                0.13563209027051926,
                0.07156781107187271,
            ],
            "start_peakres_nomask": [
                0.35750773549079895,
                0.26311833411455154,
                0.13563209027051926,
                0.10233728587627411,
            ],
            "peakres": [
                0.26311837077153755,
                0.13563208829064496,
                0.06990318710485387,
                0.036122493679281074,
            ],
            "peakres_nomask": [
                0.26311837077153755,
                0.13732841007407787,
                0.11575909682422982,
                0.08907417228016686,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 0): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [0, 252, 1728, 2507],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3445475697517395,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372762580492.5155,
            "time": 0.0,
            "start_model_flux": [0.0, 0.0, -0.005321894089268711, -0.15096239242265286],
            "model_flux": [
                0.0,
                -0.005321894089268711,
                -0.15096239242265286,
                -0.34173595784959543,
            ],
            "start_peakres": [
                0.1205507144331932,
                0.1205507181584835,
                0.06295699812471867,
                0.04065505973994732,
            ],
            "start_peakres_nomask": [
                0.1205507144331932,
                0.1205507181584835,
                0.08899888396263123,
                0.06060711480677128,
            ],
            "peakres": [
                0.1205507144331932,
                0.062144659342189774,
                0.03209609787320367,
                0.016586460598119468,
            ],
            "peakres_nomask": [
                0.1205507144331932,
                0.09875828369965392,
                0.07549973026809857,
                0.05831613073743894,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 0, 1): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [2, 19, 217, 1410],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.34453392028808594,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372763190875.0631,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.06367041795176881,
                0.4067038674282753,
                0.9533928860362973,
            ],
            "model_flux": [
                0.06367041795176881,
                0.4067038674282753,
                0.9533928860362973,
                1.4102348849152202,
            ],
            "start_peakres": [
                0.3351074755191803,
                0.2719259709119797,
                0.14018256962299347,
                0.07534321397542953,
            ],
            "start_peakres_nomask": [
                0.3351074755191803,
                0.2719259709119797,
                0.14018256962299347,
                0.09314373135566711,
            ],
            "peakres": [
                0.27192600301764985,
                0.14018257657320918,
                0.07241199221474351,
                0.03734806570023123,
            ],
            "peakres_nomask": [
                0.27192600301764985,
                0.14018257657320918,
                0.09828627041416445,
                0.07833979352745836,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 1): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [0, 510, 1836, 2778],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.34453392028808594,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372763190875.0631,
            "time": 0.0,
            "start_model_flux": [0.0, 0.0, 0.2559758192787306, 0.23270246637926756],
            "model_flux": [
                0.0,
                0.2559758192787306,
                0.23270246637926756,
                -0.1537634680033626,
            ],
            "start_peakres": [
                0.10747269913554192,
                0.1074727214872837,
                0.05736164189875126,
                0.034725749865174294,
            ],
            "start_peakres_nomask": [
                0.1182534508407116,
                0.11825336143374443,
                0.10508286580443382,
                0.07211481314152479,
            ],
            "peakres": [
                0.10747269913554192,
                0.05539023078315345,
                0.02861225256105843,
                0.014761570474880244,
            ],
            "peakres_nomask": [
                0.1182534508407116,
                0.1104731137899703,
                0.08816010987821392,
                0.06365545932103889,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 0, 2): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [2, 21, 225, 1365],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.34448084235191345,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.06422387343272877,
                0.44228863426298626,
                1.0642817398022482,
            ],
            "model_flux": [
                0.06422387343272877,
                0.44228863426298626,
                1.0642817398022482,
                1.2768425511333288,
            ],
            "start_peakres": [
                0.3380203992128372,
                0.2737964391708374,
                0.1385836973786354,
                0.0724470242857933,
            ],
            "start_peakres_nomask": [
                0.3380203992128372,
                0.2737964391708374,
                0.1385836973786354,
                0.10458540543913841,
            ],
            "peakres": [
                0.2737964798436145,
                0.13858376370159314,
                0.07154864122829262,
                0.03695601525200815,
            ],
            "peakres_nomask": [
                0.2737964798436145,
                0.13858376370159314,
                0.12015870268768956,
                0.0819424536988555,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 2): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [0, 210, 1595, 2311],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.34448084235191345,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [0.0, 0.0, 0.29230648444672946, 1.0342716437161545],
            "model_flux": [
                0.0,
                0.29230648444672946,
                1.0342716437161545,
                1.0718694933458495,
            ],
            "start_peakres": [
                0.1301141269505024,
                0.1301141493022442,
                0.07082179840654135,
                0.03966241981834173,
            ],
            "start_peakres_nomask": [
                0.1301141269505024,
                0.1301141493022442,
                0.08626517280936241,
                0.06678120419383049,
            ],
            "peakres": [
                0.1301141269505024,
                0.06714578978305617,
                0.034671231530496435,
                0.017898155966177817,
            ],
            "peakres_nomask": [
                0.1301141269505024,
                0.08533931549999212,
                0.07839905071764713,
                0.06462196519349624,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 0, 3): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [2, 22, 279, 1552],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3444790840148926,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372764411640.15845,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.06208955190118765,
                0.41661112520218035,
                0.7711116959103461,
            ],
            "model_flux": [
                0.06208955190118765,
                0.41661112520218035,
                0.7711116959103461,
                1.246058923764651,
            ],
            "start_peakres": [
                0.3250153809785843,
                0.2669743224978447,
                0.13665873184800148,
                0.07552315667271614,
            ],
            "start_peakres_nomask": [
                0.3250153809785843,
                0.2669743224978447,
                0.13665873184800148,
                0.09676706790924072,
            ],
            "peakres": [
                0.2669743294114686,
                0.13665873894881198,
                0.07054322757269109,
                0.03642931892516965,
            ],
            "peakres_nomask": [
                0.2669743294114686,
                0.13665873894881198,
                0.10821098085022521,
                0.07827344667891167,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 3): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [0, 234, 1574, 2161],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3444790840148926,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372764411640.15845,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.0,
                -0.0019376348073805033,
                -0.32794513500782063,
            ],
            "model_flux": [
                0.0,
                -0.0019376348073805033,
                -0.32794513500782063,
                -0.40983942431381465,
            ],
            "start_peakres": [
                0.13275842368602753,
                0.13275842741131783,
                0.07041618973016739,
                0.04057924449443817,
            ],
            "start_peakres_nomask": [
                0.13275842368602753,
                0.13275842741131783,
                0.08207742124795914,
                0.06188509427011013,
            ],
            "peakres": [
                0.13275842368602753,
                0.0685731236197225,
                0.035427532969437514,
                0.018303977467112688,
            ],
            "peakres_nomask": [
                0.13275842368602753,
                0.09255358378616745,
                0.07877961247840846,
                0.05598988311518244,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 0, 4): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [3, 20, 269, 1455],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3444778323173523,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372765022022.7062,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.09632811689227336,
                0.44859344134508533,
                0.7196022894115142,
            ],
            "model_flux": [
                0.09632811689227336,
                0.44859344134508533,
                0.7196022894115142,
                0.8687526581169909,
            ],
            "start_peakres": [
                0.3533458411693573,
                0.2624829560518265,
                0.13386095687747002,
                0.07134852558374405,
            ],
            "start_peakres_nomask": [
                0.3533458411693573,
                0.2624829560518265,
                0.13386095687747002,
                0.10185753926634789,
            ],
            "peakres": [
                0.26248303728047495,
                0.1338610040358027,
                0.06916595739352831,
                0.03571509627867214,
            ],
            "peakres_nomask": [
                0.26248303728047495,
                0.1338610040358027,
                0.1137017434033023,
                0.08186949308006658,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 4): {
            "niter": 10000,
            "threshold": 0.03742406390770764,
            "iter_done": [0, 461, 1768, 2610],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3444778323173523,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372765022022.7062,
            "time": 0.0,
            "start_model_flux": [0.0, 0.0, 0.014776201410511294, 0.1883208494848036],
            "model_flux": [
                0.0,
                0.014776201410511294,
                0.1883208494848036,
                0.14211058918192704,
            ],
            "start_peakres": [
                0.11055046692490578,
                0.11055045202374458,
                0.06046096049249172,
                0.03794900490902364,
            ],
            "start_peakres_nomask": [
                0.11055046692490578,
                0.11055045202374458,
                0.08840354532003403,
                0.0653633363544941,
            ],
            "peakres": [
                0.11055046692490578,
                0.057018342864911066,
                0.02942986543593892,
                0.015200409354116482,
            ],
            "peakres_nomask": [
                0.11055046692490578,
                0.09952463161245279,
                0.07619173761696382,
                0.06074711394111128,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
    }
    _check_deconvolve_dict(deconvolve_dict, expected_deconvolve_dict)

    polarization = 0
    region = dict(
        polarization=polarization, time=0, l=slice(100, 150), m=slice(100, 150)
    )
    I_av = img_av_xds.isel(**region)
    I = img_xds.isel(**region)

    max_per_dif_model = [1e-06, 1e-06, 1e-06, 1e-06, 1e-06]
    min_per_dif_model = [1e-07, 1e-07, 1e-07, 1e-07, 1e-07]
    max_per_dif_residual = [1e-06, 1e-06, 1e-06, 1e-06, 1e-06]
    min_per_dif_residual = [1e-07, 1e-07, 1e-07, 1e-07, 1e-07]

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
                f"expected minimum {min_per_dif_model[i_f]}."
            )
        if rel_diff_residual < min_per_dif_residual[i_f]:
            print(
                f"Channel {i_f} SKY_RESIDUAL improved: {rel_diff_residual} is below "
                f"the expected minimum {min_per_dif_residual[i_f]}."
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
    print("************" * 10)
    test_single_field_imaging(make_plot_saver())
    print("************" * 10)
    test_single_field_imaging_multi_cycle(make_plot_saver())
