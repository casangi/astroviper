"""Stakeholder tests for single-field imaging.

These tests run ``image_cube_single_field`` and compare the astroviper output
against pre-computed astroviper *truth* images that live alongside this file.
Each base test exists in several variants that differ only in knobs which must
*not* change the result (the processing-function thread count and, for the
multi-cycle test, the number of compute chunks); every variant must therefore
reproduce its truth image to within ``TRUTH_RTOL``.

The truth images are themselves astroviper output: a single canonical
reference per base test, regenerated at double precision,
``processing_function_threads=1``, ``n_mapping_parallelism=1`` and ``skunk_works=False``
(see ``_regenerate_truth_images``). The old CASA reference images are no longer
used.

The tests must be run from this directory so the relative ``*.zarr`` paths
resolve.

Run with pytest::

    conda run -n zinc pytest test_single_field_imaging.py

Regenerate the truth images -- then upload the resulting ``*.img.zarr``
directories to Google Drive and paste their file ids into
``_TRUTH_IMAGE_DRIVE_IDS`` -- with::

    conda run -n zinc python test_single_field_imaging.py

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
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging.image_cube_single_field import (
    image_cube_single_field,
)
from astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric import (
    airy_disk_rorder,
    airy_disk_rorder_v2,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    IMAGING_MAX_CYCLES,
    print_imaging_dict,
)

PS_STORE = "twhya_selfcal_lsrk_5chans.ps.zarr"

# Astroviper truth (reference) images. niter0 and niter100 each have a single
# double-precision truth; multi_cycle has both a double- and a single-precision
# truth because its deep CLEAN is precision-sensitive. All are regenerated at
# processing_function_threads=1, n_mapping_parallelism=1, skunk_works=False (see
# ``_regenerate_truth_images``); double-precision variants compare against the
# double truth, single-precision variants against the single truth.
TRUTH_IMAGE_NITER0 = "twhya_selfcal_5chans_lsrk_niter0_truth_v8.img.zarr"
TRUTH_IMAGE_NITER100 = "twhya_selfcal_5chans_lsrk_niter100_truth_v8.img.zarr"
TRUTH_IMAGE_MULTI_CYCLE_DOUBLE = (
    "twhya_selfcal_5chans_lsrk_multi_cycle_double_truth_v8.img.zarr"
)
TRUTH_IMAGE_MULTI_CYCLE_SINGLE = (
    "twhya_selfcal_5chans_lsrk_multi_cycle_single_truth_v8.img.zarr"
)

# Google Drive file IDs for the zipped truth images above. Each is a zipped
# ``.img.zarr`` directory. Regenerate the images locally with
# ``python test_single_field_imaging.py``, upload them to Google Drive and
# paste the resulting file ids here. (Until then, a locally regenerated copy is
# used directly -- the download is a no-op when the directory already exists.)
_TRUTH_IMAGE_DRIVE_IDS = {
    # v8, regenerated 2026-10-26 for the new default gridding kernel (RADPS
    # Memo 13a): no imager call here passes gridding_kernel_params, so the
    # images are gridded with the least misfit kernel of support 9,
    # oversampling 300 and linear lookup instead of CASA's prolate spheroidal
    # kernel (support 7, oversampling 100, nearest lookup). With CASA's kernel
    # the same code reproduces v6 to rounding (at most 3.1e-10 relative at
    # double precision). The kernel moves the Stokes I PSF by up to 8.7e-4 of
    # its peak (6.7e-5 in the inner half of the image) and the niter0 residual
    # by up to 5.4e-3 of its peak (1.5e-3), which changes the CLEAN of
    # niter100 and multi_cycle. (v7, made for threshold_tolerance, was never
    # uploaded.) Uploaded 2026-10-27.
    # v6, for the stop test on the fresh residual (issue 290): every
    # threshold_per_cycle is based on the peak residual inside the clean mask
    # after the residual update of its imaging cycle, which changes the CLEAN
    # of multi_cycle (niter0 and niter100 hold the same data as v5). Uploaded
    # 2026-10-17, regenerated from 290-imaging-accuracy at 86b9f12.
    # v5, regenerated 2026-10-13 for the shared beam model (issue 290): the
    # primary beam is now the textbook obscured Airy disk instead of CASA's
    # PBMath1DAiry formula, whose blockage term widens the beam, so the
    # primary_beam_limit mask shrinks and the CLEAN of niter100 and multi_cycle
    # changes. A new version keeps main's truths (the unversioned names)
    # untouched.
    TRUTH_IMAGE_NITER0: "1XZcfXiSs8n_dRmVrElOxiupCo5fd96Hd",
    TRUTH_IMAGE_NITER100: "16TDgMucqAFMINkX0PxnKbH6Qy92YQVxJ",
    TRUTH_IMAGE_MULTI_CYCLE_DOUBLE: "1vDINEZ5cewer7p0ncitmBFGdC6ej1WdD",
    TRUTH_IMAGE_MULTI_CYCLE_SINGLE: "1D_0KyA-2PlqOOSg-9JD40uMkYd0-ZtcI",
}

# Default (tight) per-channel relative-difference ceiling for the reproducible
# variants. Double-precision multi_cycle and the niter0/niter100 tests reproduce
# their truth to ~3e-12 (SKY_RESTORED ~4e-10 at processing_function_threads=12),
# so 1e-6 is a real regression guard with comfortable margin.
TRUTH_RTOL = 1e-6

# Deep float32 CLEAN may select different near-tied peaks after PSF rounding.
# Compare the observable restored Stokes-I image, rather than requiring the
# unconvolved component model or each residual pixel to reproduce its history.
# These are TW Hydra regression bounds, not general science/QA2 tolerances.
# The existing 15% peak-image ceiling is retained, with additional masked
# restored-image L2/flux checks at 15% and residual RMS agreement at 5%.
# Float64 references and shallow imaging retain the tight checks above.
# Float32 runs use one thread. With several threads the gridder adds into a
# float32 grid in an order that changes from run to run, and so does the deep
# CLEAN: with the default gridding kernel and 12 threads, the residual RMS
# change of the worst case comparison ranged from 2.9 to 5.3 percent in five
# repeats. The thread count is varied in double precision only.
MULTI_CYCLE_SINGLE_RTOL = 0.15
MULTI_CYCLE_DOUBLE_VS_SINGLE_RTOL = 0.15
MULTI_CYCLE_WORST_CASE_RTOL = 0.15
DEEP_CLEAN_RESIDUAL_RMS_RTOL = 0.05
# Integrated restored flux is compared inside a source aperture taken from the
# reference image (pixels above this fraction of the channel peak). The CLEAN
# mask itself is the whole primary-beam-limited field, whose net flux is a
# small difference of large positive and negative parts (about 5 percent of
# its absolute flux here), so a flux ratio normalised by it amplified the
# noise-level differences of a float32 CLEAN branch flip about twentyfold
# (CI 2026-09-29: restored L2 difference 3 percent, net-flux ratio 17 percent).
# Between CLEAN branches the source-aperture flux moves by under 0.3 percent.
DEEP_CLEAN_SOURCE_APERTURE_FRACTION = 0.1
DEEP_CLEAN_PSF_RTOL = 1e-4
# The float32 rounding of the PSF is divided by the image plane correction
# c(l) c(m) of the gridding kernel, which for the default kernel falls to
# 4.7e-5 in the image corners (CASA's kernel: 1.6e-3). There a float32 PSF
# differs from another float32 or a float64 PSF by up to 3.2e-4 of its peak
# (the difference times c(l) c(m) is at most 2.1e-7 everywhere), in the inner
# half of the image by at most 3.0e-7. The PSF is held to DEEP_CLEAN_PSF_RTOL
# in the inner half and to DEEP_CLEAN_PSF_EDGE_ATOL over the whole image.
DEEP_CLEAN_PSF_EDGE_ATOL = 1e-3

# Deconvolve-dict floats computed FROM the float32 gridded seed: the per-imaging-
# cycle CLEAN trajectory (model_flux, peakres, ...) plus the thresholds derived
# from it and from the PSF (threshold_per_cycle, max_psf_sidelobe). In single
# precision these differ cross-platform once the deep CLEAN tips a peak-selection
# tie (~few %) or simply inherits the platform's float32 seed noise (~1e-4) --
# see the TRUTH_RTOL comment above. For a single-precision dict they are compared
# by MAGNITUDE at MULTI_CYCLE_SINGLE_DICT_RTOL with a peak-scaled atol (the peak-
# residual sign flips at the bifurcation while its magnitude is stable). iter_done
# is included because the model update count to reach the cycle threshold also
# bifurcates (~few %) once a channel tips. Everything else stays tight: the
# remaining exact-int fields (max_iter, masksum), stop_code, the strings, and the
# config/coordinate floats (gain, min/max_psf_fraction, frequency, time) are
# all bit-reproducible. Double-precision dicts compare every field tightly.
_BIFURCATION_SENSITIVE_FIELDS = frozenset(
    {
        "model_flux",
        "start_model_flux",
        "peakres",
        "peakres_nomask",
        "start_peakres",
        "start_peakres_nomask",
        "threshold_per_cycle",
        "max_psf_sidelobe",
        "iter_done",
    }
)
MULTI_CYCLE_SINGLE_DICT_RTOL = 0.15

# Imaging weighting is identical for every base test.
IMAGING_WEIGHTS_PARAMS = {
    "weighting": "briggs",
    "robust": 0.5,
    "casa_weighting_implementation": True,
}

# Per-base-test imaging configuration. Everything fixed for a base test lives
# here; the per-variant knobs (``processing_function_threads``, ``n_mapping_parallelism`` and
# ``single_precision_image``) are supplied by each test. ``skunk_works`` is
# always False and the truth images are generated from these configs at double
# precision, threads=1, n_mapping_parallelism=1.
_CONFIGS = {
    "niter0": {
        "cycle_control_params": {
            "max_iter": 0,
            "max_cycles": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        "image_data_variables_keep": [
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
            "visibility_normalization",
            "uv_sampling_normalization",
        ],
        "single_precision_image": False,
        "extra_kwargs": {},
        "compare_variables": ["SKY_RESIDUAL", "POINT_SPREAD_FUNCTION", "PRIMARY_BEAM"],
    },
    "niter100": {
        "cycle_control_params": {
            "max_iter": 100,
            "max_cycles": 1,
            "threshold": 0.001,
            "primary_beam_limit": 0.2,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
        },
        "image_data_variables_keep": [
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
            "sky_model",
            "mask",
            "sky_restored",
        ],
        "single_precision_image": False,
        "extra_kwargs": {
            "write_visibility_model_to_ps": True,
            "fft_backend": "scipy",
            "restore": True,
        },
        "compare_variables": [
            "SKY_MODEL",
            "SKY_RESIDUAL",
            "PRIMARY_BEAM",
            "SKY_RESTORED",
            "MASK",
        ],
    },
    # Same numeric parameters as "niter100" except max_cycles=0, which must run
    # zero deconvolution -- so no MASK/SKY_MODEL/SKY_RESTORED is ever created.
    "max_cycles0": {
        "cycle_control_params": {
            "max_iter": 100,
            "max_cycles": 0,
            "threshold": 0.001,
            "primary_beam_limit": 0.2,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.2,
        },
        "image_data_variables_keep": [
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
        ],
        "single_precision_image": False,
        "extra_kwargs": {},
        "compare_variables": ["SKY_RESIDUAL", "POINT_SPREAD_FUNCTION", "PRIMARY_BEAM"],
    },
    "multi_cycle": {
        "cycle_control_params": {
            "max_iter": 10000,
            "max_cycles": 4,
            "threshold": 0.001,
            "primary_beam_limit": 0.2,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        "image_data_variables_keep": [
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
            "sky_model",
            "mask",
            "sky_restored",
        ],
        "single_precision_image": False,
        "extra_kwargs": {
            "write_visibility_model_to_ps": True,
            "fft_backend": "scipy",
            "restore": True,
        },
        "compare_variables": [
            "SKY_MODEL",
            "SKY_RESIDUAL",
            "PRIMARY_BEAM",
            "SKY_RESTORED",
            "MASK",
        ],
    },
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

    Used for astroviper truth images. The archives are uploaded as zipped
    ``.zarr`` directories, but may arrive *double-zipped* (a ``.zip`` whose only
    member is another ``.zip``). This helper extracts into a scratch directory,
    recursively unpacks any nested ``.zip`` it finds, locates the
    ``<zarr_name>`` directory wherever it lands and moves it into place. It is a
    no-op when the directory already exists locally, so it never re-downloads or
    clobbers a local copy.
    """
    import glob
    import shutil
    import zipfile

    if os.path.isdir(zarr_name):
        return  # already present locally -- nothing to download.
    if file_id is None:
        raise RuntimeError(
            f"{zarr_name} is not on Google Drive yet: regenerate it with "
            "'python test_single_field_imaging.py', or upload it and paste its "
            "file id into _TRUTH_IMAGE_DRIVE_IDS"
        )

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
    """Ensure the ToolVIPER-managed input processing set is present."""
    if os.path.isdir(PS_STORE):
        return

    from toolviper.utils.data import download

    download(file=PS_STORE, folder=".")
    if not os.path.isdir(PS_STORE):
        raise RuntimeError(f"Could not download the '{PS_STORE}' processing set")


def _ensure_truth_image(zarr_name):
    """Ensure one astroviper truth image is present, downloading if absent."""
    _download_zarr(zarr_name, _TRUTH_IMAGE_DRIVE_IDS[zarr_name])


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


def _check_imaging_dict(
    imaging_dict,
    expected,
    rtol=1e-5,
    atol=1e-8,
    loose_fields=frozenset(),
    loose_rtol=0.15,
):
    """Assert every key and field of a deconvolve ImagingDict matches ``expected``.

    ``expected`` is keyed by ``(time, pol, chan)`` tuples, with ``stop_code``
    given as an ``(imaging, model_update)`` tuple. Integer fields (max_iter, iter_done,
    masksum), string fields (stokes, stop_description) and the stop code are
    compared exactly; every other (floating-point) field -- scalar or per-cycle
    history list -- is compared with ``np.allclose`` at ``rtol``/``atol``.

    Fields named in ``loose_fields`` are instead compared by MAGNITUDE at
    ``loose_rtol`` with a peak-scaled ``atol`` (``loose_rtol * max|expected|``).
    This is for the single-precision multi_cycle dict, whose per-cycle
    flux/residual trajectory bifurcates cross-platform and above one thread (see
    the TRUTH_RTOL comment): the peak-residual sign flips while its magnitude is
    stable, and the model update count itself drifts, so those fields cannot be
    pinned tightly, while max_iter/stop_code/masksum still can. Regenerate the
    expected literals if the imaging or cycle control behaviour intentionally
    changes.
    """
    exact_int_fields = {"max_iter", "iter_done", "masksum", "model_update_stop_code"}
    string_fields = {"stokes", "stop_description"}

    actual = {tuple(k): v for k, v in imaging_dict.data.items()}
    assert set(actual) == set(expected), (
        f"imaging_dict planes {sorted(actual)} != expected {sorted(expected)}"
    )
    for key, exp_fields in expected.items():
        got = actual[key]
        assert set(got) == set(exp_fields), (
            f"plane {key}: fields {sorted(got)} != expected {sorted(exp_fields)}"
        )
        for field, exp_val in exp_fields.items():
            val = got[field]
            if field == "stop_code":
                assert (int(val.imaging), int(val.model_update)) == tuple(exp_val), (
                    f"plane {key} stop_code {val} != {exp_val}"
                )
            elif field in string_fields:
                assert str(val) == exp_val, (
                    f"plane {key} {field}: {val!r} != {exp_val!r}"
                )
            elif field in loose_fields:
                # Checked BEFORE exact_int_fields so a field marked loose (e.g.
                # iter_done for a single-precision dict) is compared loosely; for
                # a double-precision dict loose_fields is empty, so iter_done
                # falls through to the exact-int check below.
                # Single-precision deep-CLEAN trajectory: bifurcates cross-
                # platform / above one thread. Compare MAGNITUDES at a loose,
                # peak-scaled tolerance. The accumulated flux is positive so abs
                # is a no-op there, but the peak-residual SIGN flips at the
                # bifurcation (same |value|, opposite sign -- a positive vs
                # negative sidelobe wins); only its magnitude, which convergence
                # tests against the threshold, is stable. The peak-scaled atol
                # also keeps small values from false-failing on a per-element
                # relative test.
                exp_arr = np.abs(np.asarray(exp_val, dtype=float))
                act_arr = np.abs(np.asarray(val, dtype=float))
                scale = float(np.max(exp_arr)) if exp_arr.size else 0.0
                assert np.allclose(
                    act_arr,
                    exp_arr,
                    rtol=loose_rtol,
                    atol=max(atol, loose_rtol * scale),
                ), f"plane {key} {field} (loose |{loose_rtol}|): {val} != {exp_val}"
            elif field in exact_int_fields:
                assert np.array_equal(np.asarray(val), np.asarray(exp_val)), (
                    f"plane {key} {field}: {val} != {exp_val}"
                )
            else:
                assert np.allclose(
                    np.asarray(val, dtype=float),
                    np.asarray(exp_val, dtype=float),
                    rtol=rtol,
                    atol=atol,
                ), f"plane {key} {field}: {val} != {exp_val}"


def _check_image_statistics(imaging_dict, img_av_xds, expect_mask):
    """The gathered per-plane statistics must describe the written cube: full
    (time, frequency, polarization) extent in global frequency order, and the
    NaN-ignoring mean / signed peak of every SKY_RESIDUAL plane must match a
    direct recomputation from the image store (5 chunks -> exercises the
    frequency concatenation in the reduce)."""
    stats = imaging_dict["image_statistics"]
    assert "sky_residual" in stats
    residual_stats = stats["sky_residual"]
    assert residual_stats.sizes == {
        dim: img_av_xds.sizes[dim] for dim in ("time", "frequency", "polarization")
    }
    np.testing.assert_allclose(
        residual_stats["frequency"].values, img_av_xds["frequency"].values
    )
    residual = (
        img_av_xds["SKY_RESIDUAL"]
        .transpose("time", "frequency", "polarization", "l", "m")
        .values.astype(np.float64)
    )
    expected_mean = np.nanmean(residual, axis=(-2, -1))
    np.testing.assert_allclose(
        residual_stats["mean"].values, expected_mean, rtol=1e-6, atol=1e-12
    )
    flat = residual.reshape(residual.shape[:3] + (-1,))
    peak_index = np.nanargmax(np.abs(flat), axis=-1)
    expected_peak = np.take_along_axis(flat, peak_index[..., None], axis=-1)[..., 0]
    np.testing.assert_allclose(
        residual_stats["peak"].values, expected_peak, rtol=1e-6, atol=1e-12
    )
    # Masked statistics are always available: the clean MASK when the run
    # deconvolved, else the PRIMARY_BEAM > primary_beam_limit fallback.
    assert bool(residual_stats.attrs["mask_present"]) is True
    if expect_mask:
        assert residual_stats.attrs["mask_source"] == "MASK"
    else:
        assert residual_stats.attrs["mask_source"] == "PRIMARY_BEAM > 0.2"
    assert np.isfinite(residual_stats["peak_masked"].values).all()
    assert (
        residual_stats["n_pixels_masked"].values < residual_stats["n_pixels"].values
    ).all()
    assert "T_image_statistics" in imaging_dict["timing_node_tasks"].columns


def _run_image_cube(
    kind,
    image_store,
    *,
    processing_function_threads,
    n_mapping_parallelism,
    single_precision_image=None,
    skunk_works=False,
    image_sharding=None,
    image_chunking=None,
):
    """Run ``image_cube_single_field`` for one base test ``kind`` and variant.

    Returns ``(imaging_dict, img_av_xds, image_params)``. The per-variant knobs are
    ``processing_function_threads``, ``n_mapping_parallelism`` and optionally
    ``single_precision_image`` (which overrides the config value when not None),
    plus ``skunk_works`` / ``image_sharding`` /
    ``image_chunking`` to exercise the direct-write, sharded-write and
    sub-chunked-write paths. Everything else comes from ``_CONFIGS[kind]``, so the
    truth images (generated from the same configs at double precision, threads=1,
    n_mapping_parallelism=1) and every variant share an identical imaging setup.
    """
    dask.config.set(scheduler="synchronous")
    config = _CONFIGS[kind]
    if single_precision_image is None:
        single_precision_image = config["single_precision_image"]
    ps_xdt = open_processing_set(PS_STORE)
    image_params = _image_params(ps_xdt)
    imaging_dict = image_cube_single_field(
        ps_store=PS_STORE,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params=IMAGING_WEIGHTS_PARAMS,
        cycle_control_params=config["cycle_control_params"],
        gridder="anti_alias",
        deconvolver="hogbom_many_threads",
        scan_intents="OBSERVE_TARGET#ON_SOURCE",
        image_data_variables_keep=config["image_data_variables_keep"],
        processing_set_data_group_name="base",
        single_precision_image=single_precision_image,
        thread_info=None,
        processing_function_threads=processing_function_threads,
        # The variant knob is the frequency chunk COUNT; the driver takes the
        # {parallel_axis: n_chunks} dict form (cube imaging: frequency only).
        n_mapping_parallelism={"frequency": n_mapping_parallelism},
        overwrite=True,
        vizualize_graph=False,
        skunk_works=skunk_works,
        image_sharding=image_sharding,
        image_chunking=image_chunking,
        **config["extra_kwargs"],
    )
    img_av_xds = xr.open_zarr(image_store)
    return imaging_dict, img_av_xds, image_params


def _check_deep_clean_history(deconvolve_dict):
    """Check the deep fixture's stopping contract without pinning its path."""
    expected = EXPECTED_DECONVOLVE_DICT_MULTI_CYCLE
    assert set(deconvolve_dict.data) == set(expected)
    controls = _CONFIGS["multi_cycle"]["cycle_control_params"]
    for plane, fields in deconvolve_dict.data.items():
        counts = np.asarray(fields["iter_done"])
        # One entry per residual update: one per imaging cycle the plane's
        # channel ran, then the stop test that ended it, which iterates
        # nothing. A plane may spend its max_iter budget in fewer cycles than
        # max_cycles (cycle control is per plane), never in more.
        n_cycles = len(counts) - 1
        assert counts[-1] == 0 and 1 <= n_cycles <= controls["max_cycles"], (
            f"{plane}: iter_done {counts.tolist()}, expected at most "
            f"{controls['max_cycles']} imaging cycles and a closing 0"
        )
        assert np.all(counts >= 0)
        assert fields["max_iter"] == controls["max_iter"]
        code = (int(fields["stop_code"].imaging), int(fields["stop_code"].model_update))
        # The threshold lies below the residual this CLEAN reaches, so a plane
        # stops with its max_iter spent, the last model update that ran for it
        # having spent its max_iter_per_cycle_updated (1, 1), or with
        # iterations left after max_cycles imaging cycles, the last model
        # update having reached its threshold_per_cycle (9, 2).
        if counts.sum() == controls["max_iter"]:
            assert code == (1, 1), f"{plane}: expected iteration-limit stop, got {code}"
        else:
            assert (
                counts.sum() < controls["max_iter"]
                and n_cycles == controls["max_cycles"]
                and code == (9, 2)
            ), f"{plane}: {counts.sum()} iterations in {n_cycles} cycles, stop {code}"
        assert (
            np.isfinite(fields["threshold_per_cycle"])
            and fields["threshold_per_cycle"] > 0
        )
        # peakres has the closing entry too, start_peakres and model_flux one
        # entry per model update only.
        for key, length in (
            ("peakres", n_cycles + 1),
            ("start_peakres", n_cycles),
            ("model_flux", n_cycles),
        ):
            values = np.asarray(fields[key])
            assert len(values) == length and np.all(np.isfinite(values)), (
                f"{plane}: invalid {key} history"
            )


def _check_deep_clean_images(actual, reference, *, tol, polarization=0):
    """Bound restored-image agreement and residual RMS for deep CLEAN.

    The reference CLEAN mask defines one common comparison region for the L2
    and residual-RMS bounds; nothing depends on the image under test.
    Integrated restored flux is compared through image sums (the common
    pixel/beam area factors cancel after the beam check) inside a source
    aperture derived from the reference image alone (pixels above
    ``DEEP_CLEAN_SOURCE_APERTURE_FRACTION`` of the channel peak), because the
    net flux of the whole masked field is ill-conditioned (see the constant).
    Like the existing image assertions, these science bounds apply to Stokes I;
    Q remains visible in the diagnostic plots. The unconvolved component model
    is checked for finiteness, but its pixel differences are diagnostic only.
    The PSF is held to ``DEEP_CLEAN_PSF_RTOL`` in the inner half of the image
    and to ``DEEP_CLEAN_PSF_EDGE_ATOL`` over all of it (see the constant).
    """
    for coord in ("time", "frequency", "polarization", "l", "m"):
        np.testing.assert_array_equal(actual[coord].values, reference[coord].values)
    np.testing.assert_array_equal(actual["MASK"].values, reference["MASK"].values)
    for var in (
        "PRIMARY_BEAM",
        "POINT_SPREAD_FUNCTION",
        "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
    ):
        bound = TRUTH_RTOL if var == "PRIMARY_BEAM" else DEEP_CLEAN_PSF_RTOL
        atol = 1e-10 if var == "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION" else bound
        if var == "POINT_SPREAD_FUNCTION":
            inner = {
                dim: slice(reference.sizes[dim] // 4, -(reference.sizes[dim] // 4))
                for dim in ("l", "m")
            }
            np.testing.assert_allclose(
                actual[var].isel(inner).values,
                reference[var].isel(inner).values,
                rtol=bound,
                atol=bound,
                err_msg=f"{var} (inner half)",
            )
            atol = DEEP_CLEAN_PSF_EDGE_ATOL
        np.testing.assert_allclose(
            actual[var].values,
            reference[var].values,
            rtol=bound,
            atol=atol,
            err_msg=var,
        )
    for var in ("SKY_MODEL", "SKY_RESIDUAL", "SKY_RESTORED"):
        assert np.all(np.isfinite(actual[var].values)), f"nonfinite {var}"
        assert np.all(np.isfinite(reference[var].values)), f"nonfinite reference {var}"
    for channel in range(reference.sizes["frequency"]):
        selection = dict(time=0, frequency=channel, polarization=polarization)
        mask = reference["MASK"].isel(**selection).values.astype(bool)
        assert np.any(mask), f"channel {channel}: empty comparison aperture"

        def values(dataset, var, selection=selection, mask=mask):
            return np.asarray(dataset[var].isel(**selection).values, dtype=float)[mask]

        image = values(actual, "SKY_RESTORED")
        truth = values(reference, "SKY_RESTORED")
        residual = values(actual, "SKY_RESIDUAL")
        truth_residual = values(reference, "SKY_RESIDUAL")
        norm = np.linalg.norm(truth)
        source = truth > DEEP_CLEAN_SOURCE_APERTURE_FRACTION * truth.max()
        flux = truth[source].sum()
        rms = np.sqrt(np.mean(truth_residual**2))
        assert norm > 0 and flux > 0 and rms > 0, "degenerate reference"
        metrics = {
            "restored L2 difference": (np.linalg.norm(image - truth) / norm, tol),
            "restored source flux difference": (
                abs(image[source].sum() - flux) / flux,
                tol,
            ),
            "residual RMS change": (
                abs(np.sqrt(np.mean(residual**2)) / rms - 1),
                DEEP_CLEAN_RESIDUAL_RMS_RTOL,
            ),
        }
        for label, (value, limit) in metrics.items():
            print(f"deep CLEAN channel {channel} {label}: {value:.6g} (limit {limit})")
            assert value < limit, f"channel {channel}: {label} {value} exceeds {limit}"


def _compare_to_truth(
    img_av_xds,
    truth_xds,
    variables,
    *,
    plot_saver,
    plot_prefix,
    polarization=0,
    tol=TRUTH_RTOL,
    deep_clean=False,
):
    """Compare ``variables`` of ``img_av_xds`` against ``truth_xds`` per channel.

    For every frequency channel and every variable a 3-panel (AV / TRUTH /
    normalized difference) comparison plot is generated -- as both a full-image
    figure and a central-50x50 zoom saved separately -- with the peak relative
    difference ``max|av - truth| / max|truth|`` annotated under the difference
    panel. A summary figure plots that peak relative difference against frequency
    for Stokes I and Q. Assertions (on ``polarization``) are deferred to a final
    pass so a single failing channel cannot prevent the remaining plots from
    being saved. Every relative difference must be < ``tol`` by default.
    With ``deep_clean=True``, model/residual maps remain diagnostic; only the
    restored peak difference uses ``tol``. Additional observable and invariant
    checks are applied by ``_check_deep_clean_images``.
    """
    n_freq = img_av_xds.sizes["frequency"]
    n_pol = img_av_xds.sizes["polarization"]
    pol_labels = [str(p) for p in np.atleast_1d(img_av_xds.polarization.values)]
    # Central 50x50 region of the 250x250 image, for the zoomed comparison plot.
    central = (slice(100, 150), slice(100, 150))

    def _plane(xds, var, i_f, pol):
        # Cast to float so boolean variables (e.g. MASK, stored as bool) can be
        # differenced and imshow'd like the floating-point images.
        return np.asarray(
            xds[var].isel(frequency=i_f, polarization=pol, time=0).values,
            dtype=float,
        )

    def _rel_diff(av, truth):
        denom = np.max(np.abs(truth))
        return (
            np.max(np.abs(av - truth)) / denom if denom else np.max(np.abs(av - truth))
        )

    def _channel_figure(i_f, zoom):
        # constrained_layout keeps the suptitle snug against the panels and stops
        # adjacent column titles/colorbars from overlapping.
        fig, axes = plt.subplots(
            len(variables),
            3,
            figsize=(12, 4 * len(variables)),
            squeeze=False,
            constrained_layout=True,
        )
        freq = float(img_av_xds.frequency.values[i_f])
        ztag = "  (central 50x50)" if zoom else ""
        fig.suptitle(
            f"{plot_prefix}  channel {i_f}  Stokes {pol_labels[polarization]}  "
            f"frequency {freq:.6g}{ztag}"
        )
        sl = central if zoom else (slice(None), slice(None))
        diffs = {}
        for row, var in enumerate(variables):
            av = _plane(img_av_xds, var, i_f, polarization)
            truth = _plane(truth_xds, var, i_f, polarization)
            denom = np.max(np.abs(truth))
            rel = (
                np.max(np.abs(av - truth)) / denom
                if denom
                else np.max(np.abs(av - truth))
            )
            diffs[var] = rel
            norm_diff = (av - truth) / denom if denom else (av - truth)

            im0 = axes[row, 0].imshow(av[sl])
            axes[row, 0].set_title("AV")
            axes[row, 0].set_ylabel(var)  # variable name on the left, not the title
            fig.colorbar(im0, ax=axes[row, 0])
            im1 = axes[row, 1].imshow(truth[sl])
            axes[row, 1].set_title("TRUTH")
            fig.colorbar(im1, ax=axes[row, 1])
            im2 = axes[row, 2].imshow(norm_diff[sl])
            axes[row, 2].set_title("(AV - TRUTH) / max|TRUTH|")
            axes[row, 2].set_xlabel(f"peak rel diff: {rel:.3e}")
            fig.colorbar(im2, ax=axes[row, 2])
        return fig, diffs

    # Per-channel comparison figures: full image plus a central-50x50 zoom.
    per_channel_diffs = []
    for i_f in range(n_freq):
        fig_full, channel_diffs = _channel_figure(i_f, zoom=False)
        plot_saver(fig_full, f"{plot_prefix}_channel_{i_f}.png")
        fig_zoom, _ = _channel_figure(i_f, zoom=True)
        plot_saver(fig_zoom, f"{plot_prefix}_channel_{i_f}_zoom.png")
        per_channel_diffs.append(channel_diffs)
        for var, rel_diff in channel_diffs.items():
            print(f"{plot_prefix} channel {i_f} {var} relative difference: {rel_diff}")

    # Summary: peak relative difference as a function of frequency, Stokes I & Q.
    freqs = np.asarray(img_av_xds.frequency.values, dtype=float)
    fig, axes = plt.subplots(
        len(variables),
        1,
        figsize=(8, 2.6 * len(variables)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{plot_prefix}  peak relative difference vs frequency")
    for row, var in enumerate(variables):
        ax = axes[row, 0]
        for pol in range(n_pol):
            series = np.array(
                [
                    _rel_diff(
                        _plane(img_av_xds, var, i_f, pol),
                        _plane(truth_xds, var, i_f, pol),
                    )
                    for i_f in range(n_freq)
                ]
            )
            label = pol_labels[pol] if pol < len(pol_labels) else str(pol)
            # Floor exact zeros so they stay visible on the log axis.
            ax.plot(
                freqs, np.maximum(series, 1e-16), marker="o", label=f"Stokes {label}"
            )
        ax.set_ylabel(var)
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("frequency (Hz)")
    plot_saver(fig, f"{plot_prefix}_reldiff_vs_freq.png")

    # Final pass: assertions only (all plots have already been generated).
    if deep_clean:
        _check_deep_clean_images(
            img_av_xds, truth_xds, tol=tol, polarization=polarization
        )
    for i_f, channel_diffs in enumerate(per_channel_diffs):
        for var, rel_diff in channel_diffs.items():
            if deep_clean and var != "SKY_RESTORED":
                continue
            assert rel_diff < tol, (
                f"{plot_prefix} channel {i_f}: {var} relative difference "
                f"{rel_diff} exceeds tolerance {tol}. You broke something!"
            )


@pytest.mark.parametrize("processing_function_threads", [1, 12])
def test_single_field_imaging_niter0(plot_saver, processing_function_threads):
    _ensure_ps_store()
    _ensure_truth_image(TRUTH_IMAGE_NITER0)

    image_store = (
        "twhya_selfcal_5chans_lsrk_niter0_astroviper_"
        f"t{processing_function_threads}.img.zarr"
    )
    imaging_dict, img_av_xds, image_params = _run_image_cube(
        "niter0",
        image_store,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=5,
    )
    truth_xds = xr.open_zarr(TRUTH_IMAGE_NITER0)

    print("&&&&&&&&&" * 10)
    print("imaging_metadata_pd", imaging_dict["timing_node_tasks"])
    print("imaging_dict (global channel numbering):")
    print_imaging_dict(imaging_dict["deconvolution"])
    _check_image_statistics(imaging_dict, img_av_xds, expect_mask=False)

    _compare_to_truth(
        img_av_xds,
        truth_xds,
        _CONFIGS["niter0"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix=f"niter0_t{processing_function_threads}",
    )

    # Angular beam widths use the corrected resampled pixel-interval scale.
    psf_ref = [
        [
            [2.971086828910649e-06, 2.227885477633231e-06, 2.2254373922920276],
            [2.9709892169257772e-06, 2.227874275332764e-06, 2.225484062649881],
            [2.9708749766051937e-06, 2.227856105453191e-06, 2.2253698081236335],
            [2.970881030955093e-06, 2.2278465757259163e-06, 2.225357810302524],
            [2.9708760545035416e-06, 2.2278426266917785e-06, 2.2253578827415383],
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
        img_av_xds.frequency.values,
        img_av_xds.polarization.values,
        pb_parms,
        image_params,
    )[0, ...][None, ...]
    PB_v1 = airy_disk_rorder(
        img_av_xds.frequency.values,
        img_av_xds.polarization.values,
        pb_parms,
        image_params,
    )[0, ...][None, ...]
    assert np.allclose(PB_v1, PB_v2), (
        "airy_disk_rorder and airy_disk_rorder_v2 produced different primary beams."
    )


@pytest.mark.parametrize("processing_function_threads", [1, 12])
def test_single_field_imaging_niter0_sharded(plot_saver, processing_function_threads):
    """The sharded direct-write path (skunk_works=True + image_sharding)
    must produce the SAME image as the plain direct-write path while writing far
    fewer files -- many single-channel tasks writing into shared Zarr v3 shard
    files (the "single parallel file" pattern for metadata-server relief).

    Verifies three things: (1) the sharded output matches the unsharded
    direct-write output (numerically identical up to threaded-gridder float
    reordering), (2) it is a valid Zarr v3 sharded array that
    reproduces the niter0 truth image, and (3) it really is sharded -- 5 channels
    packed 2-per-shard give 3 shard files/variable vs 5 one-file-per-channel.
    """
    import glob

    _ensure_ps_store()
    _ensure_truth_image(TRUTH_IMAGE_NITER0)

    # A: plain direct write (one file per channel).
    store_plain = (
        "twhya_selfcal_5chans_lsrk_niter0_skunk_astroviper_"
        f"t{processing_function_threads}.img.zarr"
    )
    _, img_plain, _ = _run_image_cube(
        "niter0",
        store_plain,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=5,
        skunk_works=True,
        image_sharding=None,
    )
    # B: sharded direct write (2 channels per shard).
    store_sharded = (
        "twhya_selfcal_5chans_lsrk_niter0_sharded_astroviper_"
        f"t{processing_function_threads}.img.zarr"
    )
    _, img_sharded, _ = _run_image_cube(
        "niter0",
        store_sharded,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=5,
        skunk_works=True,
        image_sharding={"frequency": 2},
    )

    compare_vars = _CONFIGS["niter0"]["compare_variables"]

    # (1) sharded output matches the unsharded direct-write output. The sharded
    # vs unsharded difference is only the Zarr *file layout*, so the two must
    # carry the same data; but each store is computed by its own imaging run and
    # the threaded gridder accumulates visibilities in a nondeterministic order,
    # so the gridded variables (SKY_RESIDUAL, POINT_SPREAD_FUNCTION) differ run
    # to run at the ~1e-13 relative level (see: two identical unsharded runs are
    # not bit-equal above one thread). A real sharding write bug -- data landing
    # in the wrong shard/offset, truncation, byte corruption -- is orders of
    # magnitude larger, so allclose well above that float-reordering floor still
    # catches it while tolerating the benign nondeterminism.
    for var in compare_vars:
        assert np.allclose(
            img_sharded[var].values,
            img_plain[var].values,
            rtol=1e-8,
            atol=1e-10,
            equal_nan=True,
        ), f"{var}: sharded write differs from unsharded direct write."

    # (2) it reproduces the niter0 truth image (valid sharded array, read via xarray).
    truth_xds = xr.open_zarr(TRUTH_IMAGE_NITER0)
    _compare_to_truth(
        img_sharded,
        truth_xds,
        compare_vars,
        plot_saver=plot_saver,
        plot_prefix=f"niter0_sharded_t{processing_function_threads}",
    )

    # (3) it is actually sharded: ceil(5/2)=3 shard files/variable vs 5 unsharded.
    for var in compare_vars:
        n_sharded = len(
            [
                f
                for f in glob.glob(
                    os.path.join(store_sharded, var, "c", "**"), recursive=True
                )
                if os.path.isfile(f)
            ]
        )
        n_plain = len(
            [
                f
                for f in glob.glob(
                    os.path.join(store_plain, var, "c", "**"), recursive=True
                )
                if os.path.isfile(f)
            ]
        )
        assert n_sharded == 3 and n_plain == 5, (
            f"{var}: expected 3 shard files (sharded) and 5 (unsharded), "
            f"got {n_sharded} and {n_plain}"
        )


def test_single_field_imaging_niter0_image_chunking(plot_saver):
    """``image_chunking`` must change only the on-disk chunk layout,
    never the image: l/m sub-chunking of the 250x250 planes ({"l": 100, "m":
    125} -> 3x2 chunks per plane, including a padded 50-row edge chunk) through
    BOTH direct-write paths (plain and sharded) reproduces the plain
    direct-write image and the niter0 truth, with the expected Zarr layout."""
    import glob

    import zarr

    _ensure_ps_store()
    _ensure_truth_image(TRUTH_IMAGE_NITER0)

    chunking = {"l": 100, "m": 125}

    # Reference: plain direct write, no sub-chunking.
    store_plain = "twhya_selfcal_5chans_lsrk_niter0_skunk_ref_astroviper.img.zarr"
    _, img_plain, _ = _run_image_cube(
        "niter0",
        store_plain,
        processing_function_threads=4,
        n_mapping_parallelism=5,
        skunk_works=True,
    )
    # A: plain direct write with l/m sub-chunking.
    store_chunked = "twhya_selfcal_5chans_lsrk_niter0_chunked_astroviper.img.zarr"
    _, img_chunked, _ = _run_image_cube(
        "niter0",
        store_chunked,
        processing_function_threads=4,
        n_mapping_parallelism=5,
        skunk_works=True,
        image_chunking=chunking,
    )
    # B: sharded direct write with l/m sub-chunking (inner chunks).
    store_sharded = (
        "twhya_selfcal_5chans_lsrk_niter0_sharded_chunked_astroviper.img.zarr"
    )
    _, img_sharded, _ = _run_image_cube(
        "niter0",
        store_sharded,
        processing_function_threads=4,
        n_mapping_parallelism=5,
        skunk_works=True,
        image_sharding={"frequency": 2},
        image_chunking=chunking,
    )

    compare_vars = _CONFIGS["niter0"]["compare_variables"]

    # Same data as the un-chunked direct write (same allclose rationale as the
    # sharded test: separate runs of the threaded gridder differ at ~1e-13).
    for img_variant, label in ((img_chunked, "chunked"), (img_sharded, "sharded")):
        for var in compare_vars:
            assert np.allclose(
                img_variant[var].values,
                img_plain[var].values,
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
            ), f"{var}: {label} sub-chunked write differs from plain direct write."

    # Reproduces the truth image (valid stores, read via xarray).
    truth_xds = xr.open_zarr(TRUTH_IMAGE_NITER0)
    _compare_to_truth(
        img_sharded,
        truth_xds,
        compare_vars,
        plot_saver=plot_saver,
        plot_prefix="niter0_sharded_image_chunking",
    )

    # The requested layout is what landed on disk.
    for var in compare_vars:
        arr = zarr.open_array(os.path.join(store_chunked, var))
        assert arr.chunks[-2:] == (100, 125), f"{var}: unexpected l/m chunks"
        arr = zarr.open_array(os.path.join(store_sharded, var))
        assert arr.chunks[-2:] == (100, 125), f"{var}: unexpected inner chunks"
        # One shard spans the full l/m extent, rounded UP to a multiple of the
        # inner chunk as Zarr requires (l: ceil(250/100)*100 = 300).
        assert arr.shards[-2:] == (300, 250), f"{var}: shard must span full l/m"
        # Plain sub-chunked: 5 tasks x (3 l-chunks x 2 m-chunks) = 30 files;
        # sharded: still ceil(5/2) = 3 shard files.
        n_chunked = len(
            [
                f
                for f in glob.glob(
                    os.path.join(store_chunked, var, "c", "**"), recursive=True
                )
                if os.path.isfile(f)
            ]
        )
        n_sharded = len(
            [
                f
                for f in glob.glob(
                    os.path.join(store_sharded, var, "c", "**"), recursive=True
                )
                if os.path.isfile(f)
            ]
        )
        assert n_chunked == 30 and n_sharded == 3, (
            f"{var}: expected 30 chunk files (plain) and 3 shard files "
            f"(sharded), got {n_chunked} and {n_sharded}"
        )


@pytest.mark.parametrize("processing_function_threads", [1, 12])
def test_single_field_imaging_niter100(plot_saver, processing_function_threads):
    _ensure_ps_store()
    _ensure_truth_image(TRUTH_IMAGE_NITER100)

    image_store = (
        "twhya_selfcal_5chans_lsrk_niter100_astroviper_"
        f"t{processing_function_threads}.img.zarr"
    )
    imaging_dict, img_av_xds, _ = _run_image_cube(
        "niter100",
        image_store,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=5,
    )
    truth_xds = xr.open_zarr(TRUTH_IMAGE_NITER100)

    print("&&&&&&&&&" * 10)
    print("imaging_metadata_pd:")
    print(imaging_dict["timing_node_tasks"].to_string())
    print("imaging_dict (global channel numbering):")
    print_imaging_dict(imaging_dict["deconvolution"])
    _check_imaging_dict(
        imaging_dict["deconvolution"], EXPECTED_DECONVOLVE_DICT_NITER100
    )
    _check_image_statistics(imaging_dict, img_av_xds, expect_mask=True)

    _compare_to_truth(
        img_av_xds,
        truth_xds,
        _CONFIGS["niter100"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix=f"niter100_t{processing_function_threads}",
    )


@pytest.mark.parametrize("processing_function_threads", [1, 12])
def test_single_field_imaging_max_cycles0(plot_saver, processing_function_threads):
    """max_cycles=0 with max_iter>0 must run zero deconvolution.

    Regression test: max_cycles=0 used to still run one full deconvolution
    regardless of the setting. Per CASA's own documented ``max_cycles`` contract,
    0 means only the initial residual is computed -- no model update
    iterations, no model, no mask.
    """
    _ensure_ps_store()

    image_store = (
        "twhya_selfcal_5chans_lsrk_max_cycles0_astroviper_"
        f"t{processing_function_threads}.img.zarr"
    )
    imaging_dict, img_av_xds, _ = _run_image_cube(
        "max_cycles0",
        image_store,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=5,
    )

    print("imaging_dict (global channel numbering):")
    print_imaging_dict(imaging_dict["deconvolution"])

    _check_image_statistics(imaging_dict, img_av_xds, expect_mask=False)
    assert "SKY_MODEL" not in img_av_xds.data_vars
    assert "model" not in img_av_xds.attrs.get("data_groups", {})

    deconv = imaging_dict["deconvolution"]
    n_planes = (
        img_av_xds.sizes["time"]
        * img_av_xds.sizes["frequency"]
        * img_av_xds.sizes["polarization"]
    )
    assert len(deconv.data) == n_planes
    # No MASK exists (the model update makes it and never ran): the stop test
    # counts the pixels of the mask the first model update would have made.
    primary_beam = (
        img_av_xds["PRIMARY_BEAM"]
        .transpose("time", "polarization", "frequency", "l", "m")
        .values
    )
    primary_beam_limit = _CONFIGS["max_cycles0"]["cycle_control_params"][
        "primary_beam_limit"
    ]
    masksums = (primary_beam >= primary_beam_limit * np.nanmax(primary_beam)).sum(
        axis=(-2, -1)
    )
    for key, fields in deconv.data.items():
        expected_masksum = int(masksums[tuple(key)])
        assert fields["iter_done"] == [0], (
            f"plane {key}: iter_done {fields['iter_done']} != [0] -- a "
            "deconvolution ran despite max_cycles=0"
        )
        assert fields["stop_code"] == (
            IMAGING_MAX_CYCLES,
            0,
        ), f"plane {key}: stop_code {fields['stop_code']} != ({IMAGING_MAX_CYCLES}, 0)"
        assert fields["masksum"] == [expected_masksum], (
            f"plane {key}: masksum {fields['masksum']} != [{expected_masksum}]"
        )


@pytest.mark.parametrize(
    "processing_function_threads, n_mapping_parallelism, single_precision_image, tol, dict_kind",
    [
        (1, 5, False, TRUTH_RTOL, "double"),
        (12, 5, False, TRUTH_RTOL, "double"),
        (1, 1, False, TRUTH_RTOL, "double"),
        (12, 1, False, TRUTH_RTOL, "double"),
        (1, 1, True, MULTI_CYCLE_SINGLE_RTOL, None),
    ],
)
def test_single_field_imaging_multi_cycle(
    plot_saver,
    processing_function_threads,
    n_mapping_parallelism,
    single_precision_image,
    tol,
    dict_kind,
):
    _ensure_ps_store()
    precision_tag = "single" if single_precision_image else "double"
    truth_image = (
        TRUTH_IMAGE_MULTI_CYCLE_SINGLE
        if single_precision_image
        else TRUTH_IMAGE_MULTI_CYCLE_DOUBLE
    )
    _ensure_truth_image(truth_image)

    image_store = (
        "twhya_selfcal_5chans_lsrk_multi_cycle_astroviper_"
        f"t{processing_function_threads}_c{n_mapping_parallelism}_{precision_tag}.img.zarr"
    )
    imaging_dict, img_av_xds, _ = _run_image_cube(
        "multi_cycle",
        image_store,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=n_mapping_parallelism,
        single_precision_image=single_precision_image,
    )
    multi_cycle_params = _CONFIGS["multi_cycle"]["cycle_control_params"]
    for fields in imaging_dict["deconvolution"].data.values():
        psf_fraction = (
            fields["max_psf_sidelobe"] * multi_cycle_params["psf_sidelobe_factor"]
        )
        assert psf_fraction < multi_cycle_params["max_psf_fraction"], (
            f"psf_fraction {psf_fraction} >= max_psf_fraction "
            f"{multi_cycle_params['max_psf_fraction']}: this config no longer "
            "exercises a non-saturating threshold_per_cycle"
        )
    truth_xds = xr.open_zarr(truth_image)

    # Float64 pins the full trajectory; float32 checks the stopping contract
    # and observable images while allowing different component selections.
    _check_deep_clean_history(imaging_dict["deconvolution"])
    expected_dict = {
        "double": EXPECTED_DECONVOLVE_DICT_MULTI_CYCLE,
        None: None,
    }[dict_kind]
    if expected_dict is not None:
        loose_fields = (
            _BIFURCATION_SENSITIVE_FIELDS if dict_kind == "single" else frozenset()
        )
        _check_imaging_dict(
            imaging_dict["deconvolution"],
            expected_dict,
            loose_fields=loose_fields,
            loose_rtol=MULTI_CYCLE_SINGLE_DICT_RTOL,
        )

    _compare_to_truth(
        img_av_xds,
        truth_xds,
        _CONFIGS["multi_cycle"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix=(
            f"multi_cycle_t{processing_function_threads}_c{n_mapping_parallelism}_{precision_tag}"
        ),
        tol=tol,
        deep_clean=single_precision_image,
    )

    print(imaging_dict["timing_node_tasks"].T)
    print(imaging_dict["timing_node_tasks"]["T_deconvolve"])
    print(imaging_dict["timing_node_tasks"]["T_residual_update"])
    print(imaging_dict["timing_node_tasks"]["T_image_cube_task"])


def test_single_field_imaging_multi_cycle_double_vs_single(plot_saver):
    """Directly compare the double- and single-precision multi_cycle truths.

    A float32-vs-float64 comparison of the deep multi-cycle CLEAN. Both truths
    are generated identically except for precision (threads=1, n_mapping_parallelism=1), so
    this isolates the precision difference: the single-precision deep CLEAN can
    select different near-tied components. Restored-image and residual-RMS
    criteria therefore replace component-by-component agreement. Generates
    per-channel comparison plots (double / single / difference).
    """
    _ensure_truth_image(TRUTH_IMAGE_MULTI_CYCLE_DOUBLE)
    _ensure_truth_image(TRUTH_IMAGE_MULTI_CYCLE_SINGLE)

    double_xds = xr.open_zarr(TRUTH_IMAGE_MULTI_CYCLE_DOUBLE)
    single_xds = xr.open_zarr(TRUTH_IMAGE_MULTI_CYCLE_SINGLE)

    _compare_to_truth(
        double_xds,
        single_xds,
        _CONFIGS["multi_cycle"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix="multi_cycle_double_vs_single",
        tol=MULTI_CYCLE_DOUBLE_VS_SINGLE_RTOL,
        deep_clean=True,
    )


def test_single_field_imaging_multi_cycle_worst_case(plot_saver):
    """Worst-case multi_cycle cross-config comparison.

    Compares the two most-divergent valid multi_cycle runs -- double precision /
    12 threads / n_mapping_parallelism=5 against single precision / 1 thread / n_mapping_parallelism=1 --
    so every knob that can perturb the result (precision, thread count, chunking)
    differs at once. The 12 threads go to the double-precision run, because a
    float32 run with several threads is not reproducible (see
    MULTI_CYCLE_SINGLE_RTOL). Deep-CLEAN component selection can differ, so the
    test bounds restored-image agreement and residual RMS, plus stopping
    invariants. Generates the per-channel comparison plots.
    Unlike the double-vs-single test, which compares the on-disk truths, both
    runs are imaged here.
    """
    _ensure_ps_store()
    double_result, double_xds, _ = _run_image_cube(
        "multi_cycle",
        "twhya_selfcal_5chans_lsrk_multi_cycle_worstcase_double_t12_c5.img.zarr",
        processing_function_threads=12,
        n_mapping_parallelism=5,
        single_precision_image=False,
    )
    single_result, single_xds, _ = _run_image_cube(
        "multi_cycle",
        "twhya_selfcal_5chans_lsrk_multi_cycle_worstcase_single_t1_c1.img.zarr",
        processing_function_threads=1,
        n_mapping_parallelism=1,
        single_precision_image=True,
    )
    _check_deep_clean_history(double_result["deconvolution"])
    _check_deep_clean_history(single_result["deconvolution"])
    _compare_to_truth(
        double_xds,
        single_xds,
        _CONFIGS["multi_cycle"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix="multi_cycle_worst_case_double_t12_c5_vs_single_t1_c1",
        tol=MULTI_CYCLE_WORST_CASE_RTOL,
        deep_clean=True,
    )


def _regenerate_truth_images():
    """Regenerate every astroviper truth image at threads=1, n_mapping_parallelism=1.

    niter0 and niter100 get a single double-precision truth each. multi_cycle
    gets two: a double-precision truth and a single-precision truth (its deep
    CLEAN is precision-sensitive, so single-precision variants are compared
    like-for-like against a single-precision reference). Each truth is generated
    single-threaded and single-chunk -- free of thread/chunk reduction-order
    effects -- so a variant's divergence is purely the effect of the knob being
    varied. Run as ``python test_single_field_imaging.py``; each truth
    ``*.img.zarr`` directory is (over)written in place. Upload them to Google
    Drive afterwards and paste the file ids into ``_TRUTH_IMAGE_DRIVE_IDS``.
    """
    _ensure_ps_store()
    plan = [
        ("niter0", TRUTH_IMAGE_NITER0, False),
        ("niter100", TRUTH_IMAGE_NITER100, False),
        ("multi_cycle", TRUTH_IMAGE_MULTI_CYCLE_DOUBLE, False),
        ("multi_cycle", TRUTH_IMAGE_MULTI_CYCLE_SINGLE, True),
    ]
    for kind, truth_image, single_precision_image in plan:
        print(f"Regenerating truth image for {kind!r} -> {truth_image}")
        _run_image_cube(
            kind,
            truth_image,
            processing_function_threads=1,
            n_mapping_parallelism=1,
            single_precision_image=single_precision_image,
        )


# ---------------------------------------------------------------------------
# Expected per-plane deconvolution ImagingDicts.
#
# Keyed by ``(time, pol, chan)``; iter_done, peakres, peakres_nomask and
# masksum hold one entry per imaging cycle and a last one for the stop test
# that ended the run, the start_* histories and model_flux one entry per
# imaging cycle; checked by ``_check_imaging_dict``. Within a precision they
# are independent of thread count and chunking (cycle control is computed
# per (time, frequency, polarization) plane). Only the double-precision
# multi_cycle is pinned to a literal: the single-precision deep CLEAN is chaotic
# near the float32 peak-selection bifurcation (see MULTI_CYCLE_SINGLE_RTOL), so
# it is not checked against a literal. Regenerate if the imaging or
# cycle control behaviour intentionally changes.
# ---------------------------------------------------------------------------

# max_iter=100, max_cycles=1, threshold=0.001: deconvolves to the iteration limit
# (the threshold is not reached) -> stop_code (1, 1): max_iter is spent, and
# the model update spent its max_iter_per_cycle_updated.
# Regression values for corrected angular beams and Gaussian-subtracted PSF
# sidelobes. Captured at float64, one thread and one frequency chunk; checked
# across the thread/chunk variants above without changing their tolerances.
EXPECTED_DECONVOLVE_DICT_NITER100 = {
    (0, 0, 0): {
        "max_iter": 100,
        "threshold_per_cycle": 0.07150122138741274,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23110101503158764,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "I",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.9390706641300919],
        "start_peakres": [0.3575061069370637],
        "start_peakres_nomask": [0.3575061069370637],
        "peakres": [0.08644045867529111, 0.08649316238613147],
        "peakres_nomask": [0.13521733113176856, 0.1166377601969733],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 0): {
        "max_iter": 100,
        "threshold_per_cycle": 0.020741135906129672,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23110101503158764,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "Q",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.08619022371566301],
        "start_peakres": [0.10370567953064835],
        "start_peakres_nomask": [0.1210460658653319],
        "peakres": [-0.07035060136192678, 0.0715051277986202],
        "peakres_nomask": [0.11992265026750214, 0.11978129110999045],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 1): {
        "max_iter": 100,
        "threshold_per_cycle": 0.06702267893762809,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.2310617547246182,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "I",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.8082093624707314],
        "start_peakres": [0.33511339468814044],
        "start_peakres_nomask": [0.33511339468814044],
        "peakres": [-0.08644663221176482, 0.08598635613291097],
        "peakres_nomask": [-0.12025724449227156, 0.11986176872815969],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 1): {
        "max_iter": 100,
        "threshold_per_cycle": 0.0209653786693909,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.2310617547246182,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "Q",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.1608579688589704],
        "start_peakres": [-0.1048268933469545],
        "start_peakres_nomask": [-0.11798320767823073],
        "peakres": [-0.0739074391308455, 0.0742370417524203],
        "peakres_nomask": [-0.11598225039971831, 0.11513725650191489],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 2): {
        "max_iter": 100,
        "threshold_per_cycle": 0.06760463094046014,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23114656173820292,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "I",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.895231495485499],
        "start_peakres": [0.3380231547023007],
        "start_peakres_nomask": [0.3380231547023007],
        "peakres": [0.08506670463846394, 0.0852970748345912],
        "peakres_nomask": [0.13003207720644455, 0.11464107262564441],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 2): {
        "max_iter": 100,
        "threshold_per_cycle": 0.026065781222152296,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23114656173820292,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "Q",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.03066418846086614],
        "start_peakres": [0.13032890611076148],
        "start_peakres_nomask": [0.13032890611076148],
        "peakres": [0.07280236246230987, 0.07324559215790433],
        "peakres_nomask": [0.11232139366941123, 0.1080752904608871],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 3): {
        "max_iter": 100,
        "threshold_per_cycle": 0.06500482974730017,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.2311651664641835,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "I",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.7889469480031467],
        "start_peakres": [0.3250241487365008],
        "start_peakres_nomask": [0.3250241487365008],
        "peakres": [0.08917054450604678, 0.08972937377506535],
        "peakres_nomask": [-0.11322907284455908, 0.11006230430830721],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 3): {
        "max_iter": 100,
        "threshold_per_cycle": 0.02655809744310248,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.2311651664641835,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "Q",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [-0.0005805684143612572],
        "start_peakres": [0.1327904872155124],
        "start_peakres_nomask": [0.1327904872155124],
        "peakres": [-0.07614916354047258, 0.07742959298448036],
        "peakres_nomask": [-0.10088798841996376, 0.09829934038165908],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 4): {
        "max_iter": 100,
        "threshold_per_cycle": 0.07066931962312804,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23116499537752241,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "I",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.7365358943889082],
        "start_peakres": [0.35334659811564023],
        "start_peakres_nomask": [0.35334659811564023],
        "peakres": [0.08598384375372414, 0.08581034842095456],
        "peakres_nomask": [0.1173252667577916, 0.11400767762283018],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 4): {
        "max_iter": 100,
        "threshold_per_cycle": 0.02208376194446436,
        "iter_done": [100, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23116499537752241,
        "stop_code": (1, 1),
        "model_update_stop_code": [1],
        "stokes": "Q",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [-0.08452351974236866],
        "start_peakres": [0.11041880972232179],
        "start_peakres_nomask": [0.11041880972232179],
        "peakres": [-0.07327712162379275, 0.0734367261231992],
        "peakres_nomask": [-0.1047945889227964, 0.10081318721877064],
        "masksum": [42417, 42417],
        "stop_description": "Reached max_iter",
    },
}


# max_iter=10000, max_cycles=4, threshold=0.001: the threshold lies below the
# residual this deep clean reaches. threshold_per_cycle is psf_fraction (0.347)
# times the peak of the fresh residual a model update starts from, and most
# model updates stop on it: the Q planes spend max_iter in their fourth model
# update -> stop_code (1, 1); the I planes end their max_cycles imaging
# cycles with 802 to 1761 iterations left, the fourth model update having
# reached its threshold_per_cycle -> (9, 2). The divergence test of the Hogbom
# kernel (max_iter_divergence, default 1) stops no model update here: with it
# switched off the run is the same bit for bit. Regenerated with threads=1,
# n_mapping_parallelism=1.
EXPECTED_DECONVOLVE_DICT_MULTI_CYCLE = {
    (0, 0, 0): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.009577202018895784,
        "iter_done": [30, 1079, 3761, 3369, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23110101503158764,
        "stop_code": (9, 2),
        "model_update_stop_code": [2, 2, 2, 2],
        "stokes": "I",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.546664589741065,
            1.2698938904130708,
            1.3844367897740646,
        ],
        "model_flux": [
            0.546664589741065,
            1.2698938904130708,
            1.3844367897740646,
            1.5217108244377522,
        ],
        "start_peakres": [
            0.3575061069370637,
            0.12392896259279304,
            0.04944416843723031,
            -0.027627751202468007,
        ],
        "start_peakres_nomask": [
            0.3575061069370637,
            0.12592141963378103,
            -0.08527926939017402,
            -0.07196183351535514,
        ],
        "peakres": [
            0.12392896236201795,
            0.04284472619812726,
            0.01712892027589528,
            0.00956659928549736,
            0.019304831163476206,
        ],
        "peakres_nomask": [
            0.1378446899572045,
            0.10062690008666364,
            -0.08402888768248416,
            -0.06963620881424404,
            0.06277021170009625,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_cycles",
    },
    (0, 1, 0): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.005675545346962147,
        "iter_done": [1365, 5369, 1731, 1535, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23110101503158764,
        "stop_code": (1, 1),
        "model_update_stop_code": [2, 2, 2, 1],
        "stokes": "Q",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.36247469186198905,
            0.6881895755187206,
            0.7691080721677274,
        ],
        "model_flux": [
            0.36247469186198905,
            0.6881895755187206,
            0.7691080721677274,
            0.6556781430295731,
        ],
        "start_peakres": [
            0.10370567953064835,
            -0.04149804531075471,
            0.028781050139220596,
            -0.01637248065508322,
        ],
        "start_peakres_nomask": [
            0.1210460658653319,
            0.089586576202404,
            -0.060623611492372734,
            -0.057957950085608695,
        ],
        "peakres": [
            0.03594878156765423,
            -0.014370701569034167,
            -0.009974317917331322,
            -0.008241239819455631,
            0.012642124540733699,
        ],
        "peakres_nomask": [
            0.09535992339731539,
            -0.06351222539563504,
            -0.05947288694734069,
            -0.0570201832205762,
            0.05713777990527776,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 1): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.00928263924423757,
        "iter_done": [36, 1299, 2864, 4265, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.2310617547246182,
        "stop_code": (9, 2),
        "model_update_stop_code": [2, 2, 2, 2],
        "stokes": "I",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.572822487793245,
            1.1170084090118306,
            1.3226539388127079,
        ],
        "model_flux": [
            0.572822487793245,
            1.1170084090118306,
            1.3226539388127079,
            1.47720495451527,
        ],
        "start_peakres": [
            0.33511339468814044,
            0.11471007452933298,
            -0.0497636294659376,
            0.02678256369255257,
        ],
        "start_peakres_nomask": [
            0.33511339468814044,
            -0.1316859182824443,
            0.07762107623030685,
            0.05360490431465971,
        ],
        "peakres": [
            0.11471007449354141,
            0.039757373735000975,
            -0.01723414638855542,
            -0.00926938482440874,
            0.020557352052794534,
        ],
        "peakres_nomask": [
            -0.13193316125384436,
            0.09274191134784554,
            0.06956043331175767,
            -0.05415625935382237,
            0.051204917633235966,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_cycles",
    },
    (0, 1, 1): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.005988954646716409,
        "iter_done": [1414, 4408, 3302, 876, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.2310617547246182,
        "stop_code": (1, 1),
        "model_update_stop_code": [2, 2, 2, 1],
        "stokes": "Q",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.14091004156997236,
            -0.10233433897192534,
            -0.20619951127033231,
        ],
        "model_flux": [
            0.14091004156997236,
            -0.10233433897192534,
            -0.20619951127033243,
            -0.272079022311486,
        ],
        "start_peakres": [
            -0.1048268933469545,
            -0.04283585429387284,
            0.025310485243570078,
            -0.0172795209484267,
        ],
        "start_peakres_nomask": [
            -0.11798320767823073,
            -0.08031555282750644,
            -0.06253575438327774,
            0.05761407803517272,
        ],
        "peakres": [
            -0.03630458352082569,
            -0.014846536980428777,
            -0.008771202545593177,
            -0.00867329278062486,
            0.011648944019315002,
        ],
        "peakres_nomask": [
            -0.10106982581450329,
            -0.06657445005554377,
            -0.05447372727778187,
            0.05684581313529164,
            0.05662299996942072,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 2): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.008531069896067202,
        "iter_done": [38, 1204, 3663, 4293, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23114656173820292,
        "stop_code": (9, 2),
        "model_update_stop_code": [2, 2, 2, 2],
        "stokes": "I",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.6338894515289931,
            1.3526962423612063,
            1.5277392911979626,
        ],
        "model_flux": [
            0.6338894515289931,
            1.3526962423612063,
            1.5277392911979626,
            1.5317634104096394,
        ],
        "start_peakres": [
            0.3380231547023007,
            0.11593317000067577,
            -0.04613886838568987,
            0.024605081243445043,
        ],
        "start_peakres_nomask": [
            0.3380231547023007,
            0.1224029684042347,
            0.0893050295950745,
            0.06510481037976035,
        ],
        "peakres": [
            0.11593316884462447,
            0.0401531641218168,
            -0.015993563474186318,
            0.008530242422694224,
            0.019054286714679892,
        ],
        "peakres_nomask": [
            0.13257422705653088,
            0.09877592196246388,
            0.06867831205824926,
            0.060849313633612276,
            0.05813546571199034,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_cycles",
    },
    (0, 1, 2): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.006120220527490573,
        "iter_done": [789, 4522, 1864, 2825, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23114656173820292,
        "stop_code": (1, 1),
        "model_update_stop_code": [2, 2, 2, 1],
        "stokes": "Q",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.4176125929212012,
            0.4017788663469708,
            0.16292342985916242,
        ],
        "model_flux": [
            0.4176125929212012,
            0.40177886634697085,
            0.16292342985916242,
            0.07100285128829689,
        ],
        "start_peakres": [
            0.13032890611076148,
            -0.050262219384166584,
            -0.03159257913628608,
            -0.017651774647412802,
        ],
        "start_peakres_nomask": [
            0.13032890611076148,
            0.08980793368290488,
            -0.05774805505557466,
            -0.05487393037433369,
        ],
        "peakres": [
            -0.04507199177095059,
            -0.017405476089829033,
            -0.01095294777302111,
            -0.008039710682879512,
            0.01512579270917231,
        ],
        "peakres_nomask": [
            0.10353076771753963,
            -0.07641736332497781,
            -0.05572070356650241,
            -0.051475379452082706,
            0.050779941939859664,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 3): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.009066949045576039,
        "iter_done": [41, 1446, 3176, 4046, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.2311651664641835,
        "stop_code": (9, 2),
        "model_update_stop_code": [2, 2, 2, 2],
        "stokes": "I",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.5792088173276562,
            1.2068601182415022,
            1.0112262179736335,
        ],
        "model_flux": [
            0.5792088173276562,
            1.2068601182415022,
            1.0112262179736335,
            1.0886111648647863,
        ],
        "start_peakres": [
            0.3250241487365008,
            -0.11253208908162188,
            -0.04816794976094593,
            -0.026148544737544795,
        ],
        "start_peakres_nomask": [
            0.3250241487365008,
            -0.1159438653378909,
            0.07165325501583167,
            0.05829174791303345,
        ],
        "peakres": [
            -0.11253209740491682,
            -0.038969846579243246,
            -0.016701082333867915,
            0.009063819270679866,
            0.019075195796729408,
        ],
        "peakres_nomask": [
            -0.11602456415167178,
            -0.09854363049193167,
            0.0625343106649465,
            0.05759470756826135,
            0.05693552188944722,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_cycles",
    },
    (0, 1, 3): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.005725305541315224,
        "iter_done": [877, 4466, 2132, 2525, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.2311651664641835,
        "stop_code": (1, 1),
        "model_update_stop_code": [2, 2, 2, 1],
        "stokes": "Q",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.10811163962371234,
            -0.05457446662261589,
            -0.24983022129248375,
        ],
        "model_flux": [
            0.10811163962371234,
            -0.05457446662261589,
            -0.24983022129248375,
            -0.4253060610634984,
        ],
        "start_peakres": [
            0.1327904872155124,
            0.04887929277285675,
            -0.029562787506127876,
            0.016511442529418455,
        ],
        "start_peakres_nomask": [
            0.1327904872155124,
            -0.0775311416380942,
            0.05487439915828987,
            0.049925503151429226,
        ],
        "peakres": [
            0.04588077485118808,
            -0.016943656586862955,
            0.010247386924623394,
            0.007688200833925437,
            0.014141012146297942,
        ],
        "peakres_nomask": [
            -0.09223374555880132,
            0.06579395951879188,
            0.050223371441364247,
            0.045605364784928305,
            0.04782747540133447,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 4): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.00988305152971994,
        "iter_done": [30, 1126, 3533, 3564, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23116499537752241,
        "stop_code": (9, 2),
        "model_update_stop_code": [2, 2, 2, 2],
        "stokes": "I",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.5123562796656139,
            0.5182077537740228,
            0.6232779094102163,
        ],
        "model_flux": [
            0.5123562796656139,
            0.5182077537740231,
            0.6232779094102163,
            0.7495930166136969,
        ],
        "start_peakres": [
            0.35334659811564023,
            0.12011732538239724,
            -0.050053001049232645,
            -0.028502157123975847,
        ],
        "start_peakres_nomask": [
            0.35334659811564023,
            -0.1214172723458633,
            -0.07863903552904836,
            0.06892398303849331,
        ],
        "peakres": [
            0.12011732660205834,
            0.04164559733593213,
            0.017352274280011532,
            0.009881953076755686,
            0.024297984627168874,
        ],
        "peakres_nomask": [
            -0.12141727244579338,
            0.10538671679316734,
            -0.07116382531963589,
            0.06973534042857053,
            0.06436536017697597,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_cycles",
    },
    (0, 1, 4): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.006165381222119387,
        "iter_done": [1254, 3495, 4841, 410, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23116499537752241,
        "stop_code": (1, 1),
        "model_update_stop_code": [2, 2, 2, 1],
        "stokes": "Q",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            -0.0112688839099509,
            -0.2107245619522221,
            -0.42580430395346835,
        ],
        "model_flux": [
            -0.0112688839099509,
            -0.2107245619522221,
            -0.42580430395346835,
            -0.4178887181864355,
        ],
        "start_peakres": [
            0.11041880972232179,
            0.04589297626982401,
            -0.024286871392186517,
            0.017780607921917477,
        ],
        "start_peakres_nomask": [
            0.11041880972232179,
            -0.08365526218783903,
            -0.06318782686409044,
            -0.054358837748009536,
        ],
        "peakres": [
            -0.03826468580694416,
            0.0158932755969615,
            -0.008416667987463372,
            -0.009666331222128531,
            0.011792843976200226,
        ],
        "peakres_nomask": [
            0.09994482499949099,
            0.06729631018926867,
            -0.055079253135827025,
            -0.05330378394723589,
            0.05413119483147609,
        ],
        "masksum": [42417, 42417, 42417, 42417, 42417],
        "stop_description": "Reached max_iter",
    },
}


if __name__ == "__main__":
    _regenerate_truth_images()
