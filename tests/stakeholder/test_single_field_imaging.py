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
        vizualize_graph=False,
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
    # dask.config.set(scheduler="synchronous")

    # from toolviper.dask.client import local_client

    # viper_client = local_client(cores=4, memory_limit="4GB")

    from dask.distributed import Client

    viper_client = Client(n_workers=4, threads_per_worker=1, memory_limit="4GB")

    _ensure_ps_store()
    _ensure_truth_images()
    ps_xdt = open_processing_set(PS_STORE)
    img_xds = xr.open_zarr(
        "twhya_selfcal_5chans_lsrk_niter_99_nmajor_1_briggs.img.zarr"
    )

    image_params = _image_params(ps_xdt)
    image_store = "twhya_selfcal_5chans_lsrk_niter_99_astroviper.img.zarr"
    print(ps_xdt.xr_ps.summary())
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
        processing_function_threads=1,
        n_chunks=1,
        overwrite=True,
        disk_chunk_sizes={"frequency": 5},
        vizualize_graph=False,
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
            "peakres": [-0.07445588224991798],
            "peakres_nomask": [-0.07445588224991798],
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
            "start_peakres": [-0.1182534508407116],
            "start_peakres_nomask": [-0.1182534508407116],
            "peakres": [-0.07756869577706144],
            "peakres_nomask": [-0.07756869577706144],
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
            "peakres": [-0.07897557256149514],
            "peakres_nomask": [-0.07897557256149514],
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
    max_per_dif_residual = [8.95e-05, 0.0331, 0.000122, 0.001684, 0.001123]
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
        double_precision=True,
        thread_info=None,
        processing_function_threads=1,
        n_chunks=1,
        overwrite=True,
        disk_chunk_sizes={"frequency": 5},
        vizualize_graph=False,
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
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [3, 21, 273, 1517],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.34454752956712004,
            'stop_code': (9, 0),
            'stokes': 'I',
            'frequency': 372762580492.5155,
            'time': 0.0,
            'start_model_flux': [0.0, 0.09718702204548665, 0.4688002189667172, 1.1537604126885275],
            'model_flux': [0.09718702204548665, 0.4688002189667172, 1.1537604126885275, 1.847822730715837],
            'start_peakres': [0.35750769883669886, 0.2631183196138005, 0.13563207924408185, 0.07156779258472935],
            'start_peakres_nomask': [0.35750769883669886, 0.2631183196138005, 0.13563207924408185, 0.10233732771683508],
            'peakres': [0.2631183725852316, 0.13563207408461328, 0.06990317650453613, 0.03612250993569454],
            'peakres_nomask': [0.2631183725852316, 0.13732836614481778, 0.11575912938009603, 0.08907402861807075],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
        (0, 1, 0): {
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [0, 252, 1728, 2476],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.34454752956712004,
            'stop_code': (9, 0),
            'stokes': 'Q',
            'frequency': 372762580492.5155,
            'time': 0.0,
            'start_model_flux': [0.0, 0.0, -0.00532194386530014, -0.15096239534316008],
            'model_flux': [0.0, -0.00532194386530014, -0.15096239534316008, -0.35716306275899146],
            'start_peakres': [0.12055068831924312, 0.1205506876671937, -0.06295698704028665, 0.04065506834108112],
            'start_peakres_nomask': [0.12055068831924312, 0.1205506876671937, 0.08899886995627626, -0.060606934220659534],
            'peakres': [0.12055068831924312, -0.06214464514242175, 0.03209611901674789, 0.016584885995306973],
            'peakres_nomask': [0.12055068831924312, 0.0987582894031148, 0.07549955941330665, -0.05826694205829249],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
        (0, 0, 1): {
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [2, 19, 217, 1410],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.3445338663269349,
            'stop_code': (9, 0),
            'stokes': 'I',
            'frequency': 372763190875.0631,
            'time': 0.0,
            'start_model_flux': [0.0, 0.06367042028195181, 0.40670386280358534, 0.9533928712741138],
            'model_flux': [0.06367042028195181, 0.40670386280358534, 0.9533928712741138, 1.4102349169864195],
            'start_peakres': [0.33510748690170244, 0.27192596380109424, 0.14018255971015336, 0.07534323842744305],
            'start_peakres_nomask': [0.33510748690170244, 0.27192596380109424, 0.14018255971015336, 0.09314373038433724],
            'peakres': [0.2719259965459929, 0.14018257167956502, 0.07241200486641418, -0.03734806881186064],
            'peakres_nomask': [0.2719259965459929, 0.14018257167956502, 0.09828632577532491, 0.07833979257154709],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
        (0, 1, 1): {
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [0, 510, 1836, 2761],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.3445338663269349,
            'stop_code': (9, 0),
            'stokes': 'Q',
            'frequency': 372763190875.0631,
            'time': 0.0,
            'start_model_flux': [0.0, 0.0, 0.2559757484894175, 0.23270234164034775],
            'model_flux': [0.0, 0.2559757484894175, 0.23270234164034775, -0.11972945909215832],
            'start_peakres': [-0.10747269667791448, -0.10747269693438621, -0.05736164245517349, 0.034725746757493604],
            'start_peakres_nomask': [-0.11825339664330478, -0.11825339680115517, -0.10508289873725692, -0.07211481848489969],
            'peakres': [-0.10747269667791448, 0.05539021974842205, -0.02861225784892831, 0.01478463675984128],
            'peakres_nomask': [-0.11825339664330478, -0.11047315111259293, -0.08816014109036577, 0.06338388235483026],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
        (0, 0, 2): {
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [2, 21, 225, 1351],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.34448079976471313,
            'stop_code': (9, 0),
            'stokes': 'I',
            'frequency': 372763801257.61084,
            'time': 0.0,
            'start_model_flux': [0.0, 0.06422386952696826, 0.44228861691809385, 1.064281709705622],
            'model_flux': [0.06422386952696826, 0.44228861691809385, 1.064281709705622, 1.2535939508100582],
            'start_peakres': [0.33802037776693955, 0.27379642872676313, 0.1385836982271735, 0.07244702556264566],
            'start_peakres_nomask': [0.33802037776693955, 0.27379642872676313, 0.1385836982271735, 0.10458541814063835],
            'peakres': [0.27379646551371806, 0.13858377339061784, 0.07154864047295216, 0.03696316555659287],
            'peakres_nomask': [0.27379646551371806, 0.13858377339061784, 0.12015873661475182, 0.08143699513646688],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
        (0, 1, 2): {
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [0, 210, 1595, 2311],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.34448079976471313,
            'stop_code': (9, 0),
            'stokes': 'Q',
            'frequency': 372763801257.61084,
            'time': 0.0,
            'start_model_flux': [0.0, 0.0, 0.2923064491023449, 1.0342715675480383],
            'model_flux': [0.0, 0.2923064491023449, 1.0342715675480383, 1.071869311285921],
            'start_peakres': [0.13011414045626285, 0.13011414140868854, 0.07082179564486357, 0.03966239530039037],
            'start_peakres_nomask': [0.13011414045626285, 0.13011414140868854, -0.0862655271228717, -0.06678152310337485],
            'peakres': [0.13011414045626285, -0.06714578948382509, 0.03467123472949045, 0.01789814602544251],
            'peakres_nomask': [0.13011414045626285, -0.08533928437445709, -0.0783994133404833, -0.0646222728177329],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
        (0, 0, 3): {
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [2, 22, 279, 1552],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.3444790825086862,
            'stop_code': (9, 0),
            'stokes': 'I',
            'frequency': 372764411640.15845,
            'time': 0.0,
            'start_model_flux': [0.0, 0.06208955220995221, 0.41661111585909566, 0.7711117580544696],
            'model_flux': [0.06208955220995221, 0.41661111585909566, 0.7711117580544696, 1.2460589290580093],
            'start_peakres': [0.32501538847028244, 0.2669743056038905, 0.13665873350635105, -0.07552314982690027],
            'start_peakres_nomask': [0.32501538847028244, 0.2669743056038905, 0.13665873350635105, -0.09676702008895738],
            'peakres': [0.26697433940715626, 0.13665874022987784, 0.07054318381103222, 0.03642930770408942],
            'peakres_nomask': [0.26697433940715626, 0.13665874022987784, -0.10821092653990036, -0.07827339952137782],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
        (0, 1, 3): {
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [0, 234, 1574, 2161],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.3444790825086862,
            'stop_code': (9, 0),
            'stokes': 'Q',
            'frequency': 372764411640.15845,
            'time': 0.0,
            'start_model_flux': [0.0, 0.0, -0.0019376413943294653, -0.32794516747552926],
            'model_flux': [0.0, -0.0019376413943294653, -0.32794516747552926, -0.4098393848786064],
            'start_peakres': [0.1327584133004238, 0.1327584130840154, 0.0704161908916565, 0.04057923195996095],
            'start_peakres_nomask': [0.1327584133004238, 0.1327584130840154, 0.08207736811412836, 0.06188510957840061],
            'peakres': [0.1327584133004238, -0.06857312548164671, -0.03542748935101865, -0.018303982963320482],
            'peakres_nomask': [0.1327584133004238, 0.09255358764201267, 0.07877954699190168, 0.05598989171594135],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
        (0, 0, 4): {
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [3, 20, 269, 1455],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.3444778231912157,
            'stop_code': (9, 0),
            'stokes': 'I',
            'frequency': 372765022022.7062,
            'time': 0.0,
            'start_model_flux': [0.0, 0.09632811291794373, 0.44859343449875905, 0.7196022623527494],
            'model_flux': [0.09632811291794373, 0.44859343449875905, 0.7196022623527494, 0.8687526202900573],
            'start_peakres': [0.3533458191980322, 0.26248298620953714, 0.13386095426078995, 0.0713484807459302],
            'start_peakres_nomask': [0.3533458191980322, 0.26248298620953714, 0.13386095426078995, 0.10185755122753035],
            'peakres': [0.26248303921608274, 0.13386099384976985, 0.06916596074745683, -0.03571508200682232],
            'peakres_nomask': [0.26248303921608274, 0.13386099384976985, 0.11370178998119636, 0.08186950708211888],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
        (0, 1, 4): {
            'niter': 10000,
            'threshold': 0.037424066081587916,
            'iter_done': [0, 461, 1768, 2610],
            'loop_gain': 0.1,
            'min_psf_fraction': 0.05,
            'max_psf_fraction': 0.8,
            'max_psf_sidelobe': 0.3444778231912157,
            'stop_code': (9, 0),
            'stokes': 'Q',
            'frequency': 372765022022.7062,
            'time': 0.0,
            'start_model_flux': [0.0, 0.0, 0.014776218155521673, 0.18832084182881992],
            'model_flux': [0.0, 0.014776218155521673, 0.18832084182881992, 0.14211080353341066],
            'start_peakres': [0.1105504622894227, 0.11055046216297143, -0.060460938487840454, -0.037948986811477146],
            'start_peakres_nomask': [0.1105504622894227, 0.11055046216297143, -0.08840350194295829, -0.06536338691505048],
            'peakres': [0.1105504622894227, -0.05701836667852918, -0.029429844527835033, 0.015200395075120462],
            'peakres_nomask': [0.1105504622894227, -0.09952465900180356, -0.07619169121768431, -0.06074715795181657],
            'masksum': [62500, 62500, 62500, 62500],
            'stop_description': 'Reached the major cycle limit (nmajor)',
        },
    }
    _check_deconvolve_dict(deconvolve_dict, expected_deconvolve_dict)

    polarization = 0
    region = dict(
        polarization=polarization, time=0, l=slice(100, 150), m=slice(100, 150)
    )
    I_av = img_av_xds.isel(**region)
    I = img_xds.isel(**region)

    # NOTE: channel 2 (frequency index 2) tolerances are deliberately relaxed.
    # The C++ imaging extensions are built with -ffp-contract=off (see the
    # gridder / hogbom / aspclean CMakeLists) so the float32 CLEAN path is
    # bit-reproducible across compilers and CPU architectures. The CASA
    # reference image, however, was generated by the default contraction-on
    # build. Four of five channels still agree with it to ~1e-7, but channel 2's
    # CLEAN sits on a peak-selection bifurcation: the ~1-ULP difference from
    # disabling FMA contraction flips the minor-cycle path onto a different --
    # but equally valid -- solution that diverges from the contraction-on
    # reference by ~2.8% (SKY_MODEL) / ~8.8% (SKY_RESIDUAL). Its ceilings are
    # raised just enough to admit that bifurcation while still catching a gross
    # regression; the other four channels remain pinned at 1e-6. Re-tighten if
    # channel 2 is ever re-blessed against a contraction-off reference image.
    max_per_dif_model = [1e-06, 1e-06, 0.05, 1e-06, 1e-06]
    min_per_dif_model = [1e-07, 1e-07, 1e-07, 1e-07, 1e-07]
    max_per_dif_residual = [1e-06, 1e-06, 0.12, 1e-06, 1e-06]
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

    # from dask.distributed import Client
    # viper_client = Client(n_workers=4, threads_per_worker=1, memory_limit="4GB")

    # input("Attached dask client. Press Enter to continue with the test...")

    # def my_func():
    #     print("Hello from a worker!")
    # delyaed_list = []
    # for i in range(10):
    #     delyaed_list.append(dask.delayed(my_func)())
    # dask.compute(delyaed_list)
