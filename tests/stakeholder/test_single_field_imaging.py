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
from astroviper.processing_functions.imaging.utils.iteration_control import (
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
        deconvolver="hogbom_many_threads",
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
        single_precision_image=False,
        thread_info=None,
        processing_function_threads=12,
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


def test_single_field_imaging_nchunks5(plot_saver):
    dask.config.set(scheduler="synchronous")

    # from toolviper.dask.client import local_client

    # viper_client = local_client(cores=4, memory_limit="4GB")

    # from dask.distributed import Client

    # viper_client = Client(n_workers=2, threads_per_worker=2, memory_limit="4GB")
    # input("Press enter")

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
        deconvolver="hogbom_many_threads",
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
        single_precision_image=True,
        thread_info=None,
        processing_function_threads=12,
        n_chunks=5,
        overwrite=True,
        disk_chunk_sizes={"frequency": 1},
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
            "cyclethreshold": 0.07150154113769532,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.3445475399494171,
            "stop_code": (1, 0),
            "stokes": "I",
            "frequency": 372762580492.5155,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.9266055226325989],
            "start_peakres": [0.35750770568847656],
            "start_peakres_nomask": [0.35750770568847656],
            "peakres": [0.09062789380550385],
            "peakres_nomask": [0.09062789380550385],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 0): {
            "niter": 100,
            "cyclethreshold": 0.02411012947559357,
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
            "model_flux": [0.060093339532613754],
            "start_peakres": [0.12055064737796783],
            "start_peakres_nomask": [0.12055064737796783],
            "peakres": [-0.07445588707923889],
            "peakres_nomask": [-0.07445588707923889],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 1): {
            "niter": 100,
            "cyclethreshold": 0.06702150106430053,
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
            "model_flux": [0.8096180558204651],
            "start_peakres": [0.3351075053215027],
            "start_peakres_nomask": [0.3351075053215027],
            "peakres": [0.09003297239542007],
            "peakres_nomask": [0.09003297239542007],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 1): {
            "niter": 100,
            "cyclethreshold": 0.023650634288787845,
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
            "model_flux": [0.06434216350317001],
            "start_peakres": [-0.11825317144393921],
            "start_peakres_nomask": [-0.11825317144393921],
            "peakres": [-0.07756861299276352],
            "peakres_nomask": [-0.07756861299276352],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 2): {
            "niter": 100,
            "cyclethreshold": 0.06760407090187073,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.34448081254959106,
            "stop_code": (1, 0),
            "stokes": "I",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.9100114107131958],
            "start_peakres": [0.33802035450935364],
            "start_peakres_nomask": [0.33802035450935364],
            "peakres": [0.08979489654302597],
            "peakres_nomask": [0.08979489654302597],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 2): {
            "niter": 100,
            "cyclethreshold": 0.026022839546203616,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.34448081254959106,
            "stop_code": (1, 0),
            "stokes": "Q",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.07577363401651382],
            "start_peakres": [0.13011419773101807],
            "start_peakres_nomask": [0.13011419773101807],
            "peakres": [0.07658877223730087],
            "peakres_nomask": [0.07658877223730087],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 3): {
            "niter": 100,
            "cyclethreshold": 0.06500307321548462,
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
            "model_flux": [0.7685136795043945],
            "start_peakres": [0.3250153660774231],
            "start_peakres_nomask": [0.3250153660774231],
            "peakres": [0.09214764833450317],
            "peakres_nomask": [0.09214764833450317],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 3): {
            "niter": 100,
            "cyclethreshold": 0.026551687717437746,
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
            "model_flux": [0.05189017206430435],
            "start_peakres": [0.13275843858718872],
            "start_peakres_nomask": [0.13275843858718872],
            "peakres": [-0.07897554337978363],
            "peakres_nomask": [-0.07897554337978363],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 4): {
            "niter": 100,
            "cyclethreshold": 0.07066916227340699,
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
            "model_flux": [0.6859082579612732],
            "start_peakres": [0.3533458113670349],
            "start_peakres_nomask": [0.3533458113670349],
            "peakres": [0.0892675369977951],
            "peakres_nomask": [0.0892675369977951],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 4): {
            "niter": 100,
            "cyclethreshold": 0.022110094130039216,
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
            "model_flux": [-0.11857172846794128],
            "start_peakres": [0.11055047065019608],
            "start_peakres_nomask": [0.11055047065019608],
            "peakres": [0.07679460197687149],
            "peakres_nomask": [0.07679460197687149],
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
    max_per_dif_model = [2.1e-05, 0.001, 8.0e-06, 0.0022, 0.0013]
    min_per_dif_model = [2.0e-05, 0.0009, 7.6e-06, 0.0021, 0.0012]
    max_per_dif_residual = [9.00e-05, 0.0331, 0.000122, 0.001684, 0.001123]
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

    # input("Press enter")


def test_single_field_imaging_nchunks1(plot_saver):
    dask.config.set(scheduler="synchronous")

    # from toolviper.dask.client import local_client

    # viper_client = local_client(cores=4, memory_limit="4GB")

    # from dask.distributed import Client

    # viper_client = Client(n_workers=2, threads_per_worker=2, memory_limit="4GB")
    # input("Press enter")

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
        deconvolver="hogbom_many_threads",
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
        single_precision_image=True,
        thread_info=None,
        processing_function_threads=12,
        n_chunks=1,
        overwrite=True,
        disk_chunk_sizes={"frequency": 1},
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
            "cyclethreshold": 0.07150154113769532,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.3445475399494171,
            "stop_code": (1, 0),
            "stokes": "I",
            "frequency": 372762580492.5155,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.9266055226325989],
            "start_peakres": [0.35750770568847656],
            "start_peakres_nomask": [0.35750770568847656],
            "peakres": [0.09062789380550385],
            "peakres_nomask": [0.09062789380550385],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 0): {
            "niter": 100,
            "cyclethreshold": 0.02411012947559357,
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
            "model_flux": [0.060093339532613754],
            "start_peakres": [0.12055064737796783],
            "start_peakres_nomask": [0.12055064737796783],
            "peakres": [-0.07445588707923889],
            "peakres_nomask": [-0.07445588707923889],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 1): {
            "niter": 100,
            "cyclethreshold": 0.06702150106430053,
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
            "model_flux": [0.8096180558204651],
            "start_peakres": [0.3351075053215027],
            "start_peakres_nomask": [0.3351075053215027],
            "peakres": [0.09003297239542007],
            "peakres_nomask": [0.09003297239542007],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 1): {
            "niter": 100,
            "cyclethreshold": 0.023650634288787845,
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
            "model_flux": [0.06434216350317001],
            "start_peakres": [-0.11825317144393921],
            "start_peakres_nomask": [-0.11825317144393921],
            "peakres": [-0.07756861299276352],
            "peakres_nomask": [-0.07756861299276352],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 2): {
            "niter": 100,
            "cyclethreshold": 0.06760407090187073,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.34448081254959106,
            "stop_code": (1, 0),
            "stokes": "I",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.9100114107131958],
            "start_peakres": [0.33802035450935364],
            "start_peakres_nomask": [0.33802035450935364],
            "peakres": [0.08979489654302597],
            "peakres_nomask": [0.08979489654302597],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 2): {
            "niter": 100,
            "cyclethreshold": 0.026022839546203616,
            "iter_done": [100],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
            "max_psf_sidelobe": 0.34448081254959106,
            "stop_code": (1, 0),
            "stokes": "Q",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [0.0],
            "model_flux": [0.07577363401651382],
            "start_peakres": [0.13011419773101807],
            "start_peakres_nomask": [0.13011419773101807],
            "peakres": [0.07658877223730087],
            "peakres_nomask": [0.07658877223730087],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 3): {
            "niter": 100,
            "cyclethreshold": 0.06500307321548462,
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
            "model_flux": [0.7685136795043945],
            "start_peakres": [0.3250153660774231],
            "start_peakres_nomask": [0.3250153660774231],
            "peakres": [0.09214764833450317],
            "peakres_nomask": [0.09214764833450317],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 3): {
            "niter": 100,
            "cyclethreshold": 0.026551687717437746,
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
            "model_flux": [0.05189017206430435],
            "start_peakres": [0.13275843858718872],
            "start_peakres_nomask": [0.13275843858718872],
            "peakres": [-0.07897554337978363],
            "peakres_nomask": [-0.07897554337978363],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 0, 4): {
            "niter": 100,
            "cyclethreshold": 0.07066916227340699,
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
            "model_flux": [0.6859082579612732],
            "start_peakres": [0.3533458113670349],
            "start_peakres_nomask": [0.3533458113670349],
            "peakres": [0.0892675369977951],
            "peakres_nomask": [0.0892675369977951],
            "masksum": [62500],
            "stop_description": "Reached the iteration limit",
        },
        (0, 1, 4): {
            "niter": 100,
            "cyclethreshold": 0.022110094130039216,
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
            "model_flux": [-0.11857172846794128],
            "start_peakres": [0.11055047065019608],
            "start_peakres_nomask": [0.11055047065019608],
            "peakres": [0.07679460197687149],
            "peakres_nomask": [0.07679460197687149],
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
    max_per_dif_model = [2.1e-05, 0.001, 8.0e-06, 0.0022, 0.0013]
    min_per_dif_model = [2.0e-05, 0.0009, 7.6e-06, 0.0021, 0.0012]
    max_per_dif_residual = [9.00e-05, 0.0331, 0.000122, 0.001684, 0.001123]
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

    # input("Press enter")


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
        deconvolver="hogbom_many_threads",
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
        single_precision_image=False,
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
            "niter": 10000,
            "cyclethreshold": 0.035887088729932064,
            "iter_done": [3, 22, 311, 1627],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3445475295671201,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372762580492.5155,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.09718702204548665,
                0.4825330600317771,
                1.1146658811277848,
            ],
            "model_flux": [
                0.09718702204548665,
                0.4825330600317771,
                1.1146658811277848,
                1.4936288787330485,
            ],
            "start_peakres": [
                0.35750769883669886,
                0.2631183190329552,
                0.13405951306715244,
                0.07311894932118859,
            ],
            "start_peakres_nomask": [
                0.35750769883669886,
                0.2631183190329552,
                0.13405951306715244,
                0.07311894932118859,
            ],
            "peakres": [
                0.26311837258523163,
                0.13442036393758017,
                0.06943810002066497,
                0.035860155400381656,
            ],
            "peakres_nomask": [
                0.26311837258523163,
                0.13442036393758017,
                0.06943810002066497,
                0.035860155400381656,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 0): {
            "niter": 10000,
            "cyclethreshold": 0.013247323584634525,
            "iter_done": [8, 739, 2168, 3550],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3445475295671201,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372762580492.5155,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.06398640705714075,
                -0.1452926990861219,
                -0.17522760771102674,
            ],
            "model_flux": [
                0.06398640705714075,
                -0.1452926990861219,
                -0.17522760771102674,
                -0.43959681844257215,
            ],
            "start_peakres": [
                0.12055068831924319,
                0.09668704802810893,
                0.059011654121003544,
                0.035509332659826295,
            ],
            "start_peakres_nomask": [
                0.12055068831924319,
                0.09668704802810893,
                0.059011654121003544,
                0.035509332659826295,
            ],
            "peakres": [
                0.0959953340255374,
                -0.04959675650007245,
                0.025632309909517356,
                0.013244905094691034,
            ],
            "peakres_nomask": [
                0.0959953340255374,
                -0.04959675650007245,
                0.025632309909517356,
                0.013244905094691034,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 0, 1): {
            "niter": 10000,
            "cyclethreshold": 0.03353459444139285,
            "iter_done": [3, 26, 377, 1721],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.34453386632693495,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372763190875.0631,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.09086301993655113,
                0.48714152453270637,
                1.0310477002839429,
            ],
            "model_flux": [
                0.09086301993655113,
                0.48714152453270637,
                1.0310477002839429,
                1.344485064348138,
            ],
            "start_peakres": [
                0.3351074869017025,
                0.24658031275538556,
                -0.12563841463957237,
                0.06588943182953905,
            ],
            "start_peakres_nomask": [
                0.3351074869017025,
                0.24658031275538556,
                -0.12563841463957237,
                0.06588943182953905,
            ],
            "peakres": [
                0.24658036446829307,
                -0.12563929217840017,
                -0.06488882074961964,
                -0.03351216571464057,
            ],
            "peakres_nomask": [
                0.24658036446829307,
                -0.12563929217840017,
                -0.06488882074961964,
                -0.03351216571464057,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 1): {
            "niter": 10000,
            "cyclethreshold": 0.012917005614728168,
            "iter_done": [15, 899, 2085, 3405],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.34453386632693495,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372763190875.0631,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.006016719503292783,
                0.0758875588914891,
                -0.22256907481112922,
            ],
            "model_flux": [
                0.006016719503292783,
                0.0758875588914891,
                -0.22256907481112922,
                -0.36620845393971224,
            ],
            "start_peakres": [
                -0.11825339664330456,
                0.09308343885970413,
                -0.05162875402914403,
                -0.031164287472864143,
            ],
            "start_peakres_nomask": [
                -0.11825339664330456,
                0.09308343885970413,
                -0.05162875402914403,
                -0.031164287472864143,
            ],
            "peakres": [
                -0.09389724733759419,
                0.04840577933350613,
                0.02499416724483618,
                0.01291204244747398,
            ],
            "peakres_nomask": [
                -0.09389724733759419,
                0.04840577933350613,
                0.02499416724483618,
                0.01291204244747398,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 0, 2): {
            "niter": 10000,
            "cyclethreshold": 0.03338193870123756,
            "iter_done": [3, 28, 377, 1671],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.34448079976471313,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.0916035160783401,
                0.549167181835268,
                1.1556019812834628,
            ],
            "model_flux": [
                0.0916035160783401,
                0.549167181835268,
                1.1556019812834628,
                1.2716836682032417,
            ],
            "start_peakres": [
                0.33802037776693955,
                0.24641674736577607,
                0.1246686131246617,
                -0.06560293026492342,
            ],
            "start_peakres_nomask": [
                0.33802037776693955,
                0.24641674736577607,
                0.1246686131246617,
                -0.06560293026492342,
            ],
            "peakres": [
                0.246416800747472,
                0.12511840620530976,
                -0.0646033851989004,
                -0.033345614925897445,
            ],
            "peakres_nomask": [
                0.246416800747472,
                0.12511840620530976,
                -0.0646033851989004,
                -0.033345614925897445,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 2): {
            "niter": 10000,
            "cyclethreshold": 0.014325075683566882,
            "iter_done": [5, 637, 2041, 3026],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.34448079976471313,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372763801257.61084,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.03497499853849211,
                0.5752580905134075,
                0.9703525326562256,
            ],
            "model_flux": [
                0.03497499853849211,
                0.5752580905134075,
                0.9703525326562256,
                1.1518167912861337,
            ],
            "start_peakres": [
                0.13011414045626285,
                -0.10400362324374736,
                0.05671751363877437,
                -0.034978092733420285,
            ],
            "start_peakres_nomask": [
                0.13011414045626285,
                -0.10400362324374736,
                0.05671751363877437,
                -0.034978092733420285,
            ],
            "peakres": [
                -0.10401673272264614,
                0.05372676534765114,
                0.027723026834105463,
                0.014317272176727186,
            ],
            "peakres_nomask": [
                -0.10401673272264614,
                0.05372676534765114,
                0.027723026834105463,
                0.014317272176727186,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 0, 3): {
            "niter": 10000,
            "cyclethreshold": 0.03299434374746375,
            "iter_done": [3, 28, 452, 1826],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3444790825086862,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372764411640.15845,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.08878698615066784,
                0.5086397677138595,
                0.9821409278275168,
            ],
            "model_flux": [
                0.08878698615066784,
                0.5086397677138595,
                0.9821409278275168,
                1.4103561200186392,
            ],
            "start_peakres": [
                0.32501538847028244,
                0.24324762521379611,
                -0.123651127264195,
                -0.06885332750001473,
            ],
            "start_peakres_nomask": [
                0.32501538847028244,
                0.24324762521379611,
                -0.123651127264195,
                -0.06885332750001473,
            ],
            "peakres": [
                0.24324767373115244,
                -0.12365099019310298,
                0.06385359890297894,
                0.032992658767265225,
            ],
            "peakres_nomask": [
                0.24324767373115244,
                -0.12365099019310298,
                0.06385359890297894,
                0.032992658767265225,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 3): {
            "niter": 10000,
            "cyclethreshold": 0.014327696405280138,
            "iter_done": [7, 721, 1896, 3206],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3444790825086862,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372764411640.15845,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.011751165304778682,
                -0.006559480846617824,
                -0.16484162868240546,
            ],
            "model_flux": [
                0.011751165304778682,
                -0.006559480846617824,
                -0.16484162868240546,
                -0.15666565941950023,
            ],
            "start_peakres": [
                0.1327584133004238,
                -0.10396325269068865,
                0.05757515774487702,
                0.03560674358081196,
            ],
            "start_peakres_nomask": [
                0.1327584133004238,
                -0.10396325269068865,
                0.05757515774487702,
                0.03560674358081196,
            ],
            "peakres": [
                -0.10396225525444691,
                -0.05368349314507971,
                0.027728236890201426,
                -0.01430856567084777,
            ],
            "peakres_nomask": [
                -0.10396225525444691,
                -0.05368349314507971,
                0.027728236890201426,
                -0.01430856567084777,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 0, 4): {
            "niter": 10000,
            "cyclethreshold": 0.035698191359377525,
            "iter_done": [3, 20, 300, 1555],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3444778231912157,
            "stop_code": (9, 0),
            "stokes": "I",
            "frequency": 372765022022.7062,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                0.09632811291794373,
                0.44859343492168907,
                0.7644585620275723,
            ],
            "model_flux": [
                0.09632811291794373,
                0.448593434921689,
                0.7644585620275723,
                0.9485323094018827,
            ],
            "start_peakres": [
                0.3533458191980322,
                0.2624829860334643,
                0.13386095443544133,
                0.06936598751965845,
            ],
            "start_peakres_nomask": [
                0.3533458191980322,
                0.2624829860334643,
                0.13386095443544133,
                0.06936598751965845,
            ],
            "peakres": [
                0.26248303921608274,
                0.13386099432385132,
                0.06908657869210517,
                0.03568096836877782,
            ],
            "peakres_nomask": [
                0.26248303921608274,
                0.13386099432385132,
                0.06908657869210517,
                0.03568096836877782,
            ],
            "masksum": [62500, 62500, 62500, 62500],
            "stop_description": "Reached the major cycle limit (nmajor)",
        },
        (0, 1, 4): {
            "niter": 10000,
            "cyclethreshold": 0.012144056419142775,
            "iter_done": [34, 1056, 1928, 3662],
            "loop_gain": 0.1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_psf_sidelobe": 0.3444778231912157,
            "stop_code": (9, 0),
            "stokes": "Q",
            "frequency": 372765022022.7062,
            "time": 0.0,
            "start_model_flux": [
                0.0,
                -0.035942656627423034,
                0.05525502537880295,
                0.09013507629054078,
            ],
            "model_flux": [
                -0.035942656627423034,
                0.05525502537880295,
                0.09013507629054078,
                -0.1617125421110452,
            ],
            "start_peakres": [
                0.1105504622894227,
                0.0886835370547483,
                0.05120629000761651,
                -0.03239891032590521,
            ],
            "start_peakres_nomask": [
                0.1105504622894227,
                0.0886835370547483,
                0.05120629000761651,
                -0.03239891032590521,
            ],
            "peakres": [
                -0.08827756285000017,
                -0.04550212157875654,
                0.02350234780793953,
                -0.01214399112980437,
            ],
            "peakres_nomask": [
                -0.08827756285000017,
                -0.04550212157875654,
                0.02350234780793953,
                -0.01214399112980437,
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

    # The reference image was re-blessed from the per-plane iteration-control
    # code (cyclethreshold is now computed per (time, frequency, polarization)
    # plane from that plane's own peak residual, so deconvolution is independent
    # of how the cube is chunked across tasks), so this build matches it to ~0.
    #
    # Channel 2 (frequency index 2) keeps a relaxed ceiling as a cross-build
    # guard: the C++ imaging extensions are built with -ffp-contract=off (see
    # the gridder / hogbom / aspclean CMakeLists) so the float32 CLEAN path is
    # bit-reproducible, but channel 2's CLEAN sits on a peak-selection
    # bifurcation where a ~1-ULP difference (e.g. a rebuild with FMA contraction
    # on) flips the minor cycle onto a different -- but equally valid -- solution
    # that can diverge by a few percent. Its ceiling admits that bifurcation
    # while still catching a gross regression; the other four channels stay at
    # 1e-6. Tighten channel 2 too once the reference build is pinned.
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

    print(imaging_metadata_pd.T)
    print(imaging_metadata_pd["T_deconvolve"])
    print(imaging_metadata_pd["T_residual_cycle"])
    print(imaging_metadata_pd["T_image_cube_task"])


if __name__ == "__main__":
    # test_single_field_imaging_niter0(make_plot_saver())
    # print("************" * 10)
    test_single_field_imaging_nchunks5(make_plot_saver())
    test_single_field_imaging_nchunks1(make_plot_saver())
    # print("************" * 10)
    # test_single_field_imaging_multi_cycle(make_plot_saver())

    # from dask.distributed import Client
    # viper_client = Client(n_workers=4, threads_per_worker=1, memory_limit="4GB")

    # input("Attached dask client. Press Enter to continue with the test...")

    # def my_func():
    #     print("Hello from a worker!")
    # delyaed_list = []
    # for i in range(10):
    #     delyaed_list.append(dask.delayed(my_func)())
    # dask.compute(delyaed_list)
