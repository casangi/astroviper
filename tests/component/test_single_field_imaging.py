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
from astroviper.processing_functions.imaging.utils.iteration_control import (
    IMAGING_MAX_CYCLES,
    print_imaging_dict,
)

PS_STORE = "twhya_selfcal_lsrk_5chans.ps.zarr"

# Google Drive file id for the zipped input processing set (``PS_STORE``).
_PS_STORE_DRIVE_ID = "1BRe3cD6YAWkn-jSPbClGGM9VlbxHP_yn"

# Astroviper truth (reference) images. niter0 and niter100 each have a single
# double-precision truth; multi_cycle has both a double- and a single-precision
# truth because its deep CLEAN is precision-sensitive. All are regenerated at
# processing_function_threads=1, n_mapping_parallelism=1, skunk_works=False (see
# ``_regenerate_truth_images``); double-precision variants compare against the
# double truth, single-precision variants against the single truth.
TRUTH_IMAGE_NITER0 = "twhya_selfcal_5chans_lsrk_niter0_truth.img.zarr"
TRUTH_IMAGE_NITER100 = "twhya_selfcal_5chans_lsrk_niter100_truth.img.zarr"
TRUTH_IMAGE_MULTI_CYCLE_DOUBLE = (
    "twhya_selfcal_5chans_lsrk_multi_cycle_double_truth.img.zarr"
)
TRUTH_IMAGE_MULTI_CYCLE_SINGLE = (
    "twhya_selfcal_5chans_lsrk_multi_cycle_single_truth.img.zarr"
)

# Google Drive file IDs for the zipped truth images above. Each is a zipped
# ``.img.zarr`` directory. Regenerate the images locally with
# ``python test_single_field_imaging.py``, upload them to Google Drive and
# paste the resulting file ids here. (Until then, a locally regenerated copy is
# used directly -- the download is a no-op when the directory already exists.)
_TRUTH_IMAGE_DRIVE_IDS = {
    # Regenerated 2026-09-28 (power-pattern primary beam, branch 126 iteration control).
    TRUTH_IMAGE_NITER0: "1-jHOP_CBuczRmtni-WHWqtjnO-SvDAqk",
    TRUTH_IMAGE_NITER100: "1CT8Car1x2M31btFc_e8gbZLK1HP4-UMb",
    TRUTH_IMAGE_MULTI_CYCLE_DOUBLE: "1yBe0dcwCuRayP6dq_BgFREJELGTALPRt",
    TRUTH_IMAGE_MULTI_CYCLE_SINGLE: "1RA64QYxchWutFvYtRwv8IOMwq8nQzKKv",
}

# Default (tight) per-channel relative-difference ceiling for the reproducible
# variants. Double-precision multi_cycle and the niter0/niter100 tests reproduce
# their truth to ~1e-13 (PRIMARY_BEAM ~5e-10 at n_mapping_parallelism=5), so 1e-6 is a real
# regression guard with comfortable margin.
TRUTH_RTOL = 1e-6

# Deep float32 CLEAN may select different near-tied peaks after PSF rounding.
# Compare the observable restored Stokes-I image, rather than requiring the
# unconvolved component model or each residual pixel to reproduce its history.
# These are TW Hydra regression bounds, not general science/QA2 tolerances.
# The existing 15% peak-image ceiling is retained, with additional masked
# restored-image L2/flux checks at 15% and residual RMS agreement at 5%.
# Float64 references and shallow imaging retain the tight checks above.
MULTI_CYCLE_SINGLE_RTOL = 0.15
MULTI_CYCLE_DOUBLE_VS_SINGLE_RTOL = 0.15
MULTI_CYCLE_WORST_CASE_RTOL = 0.15
DEEP_CLEAN_RESIDUAL_RMS_RTOL = 0.05
DEEP_CLEAN_PSF_RTOL = 1e-4

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
        "iteration_control_params": {
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
        "iteration_control_params": {
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
        "iteration_control_params": {
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
        "iteration_control_params": {
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

    Used for both the input processing set (``PS_STORE``) and the astroviper
    truth images. The archives are uploaded as zipped ``.zarr``
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
    expected literals if the imaging or iteration-control behaviour intentionally
    changes.
    """
    exact_int_fields = {"max_iter", "iter_done", "masksum"}
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
        iteration_control_params=config["iteration_control_params"],
        gridder="prolate_spheroidal",
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
    controls = _CONFIGS["multi_cycle"]["iteration_control_params"]
    for plane, fields in deconvolve_dict.data.items():
        counts = np.asarray(fields["iter_done"])
        # One entry per imaging cycle the plane's channel ran: a plane may spend
        # its max_iter budget in fewer cycles than max_cycles (iteration control
        # is per plane), never in more.
        assert 1 <= len(counts) <= controls["max_cycles"], (
            f"{plane}: {len(counts)} update cycles, expected at most "
            f"{controls['max_cycles']}"
        )
        assert np.all(counts >= 0)
        assert fields["max_iter"] == controls["max_iter"]
        assert counts.sum() == controls["max_iter"], f"{plane}: wrong iteration total"
        code = fields["stop_code"]
        assert (int(code.imaging), int(code.model_update)) == (1, 0), (
            f"{plane}: expected iteration-limit stop, got {code}"
        )
        assert (
            np.isfinite(fields["threshold_per_cycle"])
            and fields["threshold_per_cycle"] > 0
        )
        for key in ("peakres", "start_peakres", "model_flux"):
            values = np.asarray(fields[key])
            assert len(values) == len(counts) and np.all(np.isfinite(values)), (
                f"{plane}: invalid {key} history"
            )


def _check_deep_clean_images(actual, reference, *, tol, polarization=0):
    """Bound restored-image agreement and residual RMS for deep CLEAN.

    The reference CLEAN mask defines one common aperture; no data-dependent
    clipping or alignment is used. Integrated restored flux is compared through
    image sums (the common pixel/beam area factors cancel after the beam check).
    Like the existing image assertions, these science bounds apply to Stokes I;
    Q remains visible in the diagnostic plots. The unconvolved component model
    is checked for finiteness, but its pixel differences are diagnostic only.
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
        np.testing.assert_allclose(
            actual[var].values,
            reference[var].values,
            rtol=bound,
            atol=1e-10 if var == "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION" else bound,
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
        flux = abs(truth.sum())
        rms = np.sqrt(np.mean(truth_residual**2))
        assert norm > 0 and flux > 0 and rms > 0, "degenerate reference"
        metrics = {
            "restored L2 difference": (np.linalg.norm(image - truth) / norm, tol),
            "restored aperture flux difference": (
                abs(image.sum() - truth.sum()) / flux,
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
            [2.971085492209819e-06, 2.227889975208082e-06, 2.2254328140061412],
            [2.9709871340259003e-06, 2.2278793086101007e-06, 2.2254799335544933],
            [2.9708730854565744e-06, 2.2278615092386934e-06, 2.225366271858412],
            [2.970879276728119e-06, 2.227851957688052e-06, 2.225354427338833],
            [2.970874104952147e-06, 2.227848011221223e-06, 2.2253552845366853],
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
    expected_masksum = img_av_xds.sizes["l"] * img_av_xds.sizes["m"]
    for key, fields in deconv.data.items():
        assert fields["iter_done"] == [0], (
            f"plane {key}: iter_done {fields['iter_done']} != [0] -- a "
            "deconvolution ran despite max_cycles=0"
        )
        assert fields["stop_code"] == (
            IMAGING_MAX_CYCLES,
            0,
        ), f"plane {key}: stop_code {fields['stop_code']} != ({IMAGING_MAX_CYCLES}, 0)"
        # No MASK exists (it's made during the model update, which never
        # ran), so this is the full-pixel-count fallback, not a real mask sum.
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
        (12, 1, True, MULTI_CYCLE_SINGLE_RTOL, None),
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
    multi_cycle_params = _CONFIGS["multi_cycle"]["iteration_control_params"]
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
    1 thread / n_mapping_parallelism=5 against single precision / 12 threads / n_mapping_parallelism=1 --
    so every knob that can perturb the result (precision, thread count, chunking)
    differs at once. Deep-CLEAN component selection can differ, so the test
    bounds restored-image agreement and residual RMS, plus stopping invariants.
    Generates the per-channel comparison plots.
    Unlike the double-vs-single test, neither configuration is an on-disk truth,
    so both are imaged here.
    """
    _ensure_ps_store()
    double_result, double_xds, _ = _run_image_cube(
        "multi_cycle",
        "twhya_selfcal_5chans_lsrk_multi_cycle_worstcase_double_t1_c5.img.zarr",
        processing_function_threads=1,
        n_mapping_parallelism=5,
        single_precision_image=False,
    )
    single_result, single_xds, _ = _run_image_cube(
        "multi_cycle",
        "twhya_selfcal_5chans_lsrk_multi_cycle_worstcase_single_t12_c1.img.zarr",
        processing_function_threads=12,
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
        plot_prefix="multi_cycle_worst_case_double_t1_c5_vs_single_t12_c1",
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
# Keyed by ``(time, pol, chan)`` with each history list holding one entry per
# imaging cycle; checked by ``_check_imaging_dict``. Within a precision they
# are independent of thread count and chunking (iteration control is computed
# per (time, frequency, polarization) plane). Only the double-precision
# multi_cycle is pinned to a literal: the single-precision deep CLEAN is chaotic
# near the float32 peak-selection bifurcation (see MULTI_CYCLE_SINGLE_RTOL), so
# it is not checked against a literal. Regenerate if the imaging or
# iteration-control behaviour intentionally changes.
# ---------------------------------------------------------------------------

# max_iter=100, max_cycles=1, threshold=0.001: deconvolves to the iteration limit
# (the threshold is not reached) -> stop_code (1, 0).
# Regression values for corrected angular beams and Gaussian-subtracted PSF
# sidelobes. Captured at float64, one thread and one frequency chunk; checked
# across the thread/chunk variants above without changing their tolerances.
EXPECTED_DECONVOLVE_DICT_NITER100 = {
    (0, 0, 0): {
        "max_iter": 100,
        "threshold_per_cycle": 0.07150153976733978,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23110860147112905,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.9390590901412813],
        "start_peakres": [0.35750769883669886],
        "start_peakres_nomask": [0.35750769883669886],
        "peakres": [0.08642099906922444],
        "peakres_nomask": [0.1353075651374172],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 0): {
        "max_iter": 100,
        "threshold_per_cycle": 0.024110137663848633,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23110860147112905,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.13245397459361766],
        "start_peakres": [0.10430169385685946],
        "start_peakres_nomask": [0.12055068831924316],
        "peakres": [0.07041251025246369],
        "peakres_nomask": [0.1032123334607514],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 1): {
        "max_iter": 100,
        "threshold_per_cycle": 0.0670214973803405,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23107365193069979,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.8080084742083817],
        "start_peakres": [0.33510748690170244],
        "start_peakres_nomask": [0.33510748690170244],
        "peakres": [-0.08649813963264054],
        "peakres_nomask": [-0.12041248728003895],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 1): {
        "max_iter": 100,
        "threshold_per_cycle": 0.023650679328660964,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23107365193069979,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.15960419931812203],
        "start_peakres": [-0.10502057730946415],
        "start_peakres_nomask": [-0.11825339664330481],
        "peakres": [0.0742774307437133],
        "peakres_nomask": [-0.11617113784224149],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 2): {
        "max_iter": 100,
        "threshold_per_cycle": 0.06760407555338792,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23115665063155355,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.8952543936803149],
        "start_peakres": [0.3380203777669396],
        "start_peakres_nomask": [0.3380203777669396],
        "peakres": [-0.0850572812866544],
        "peakres_nomask": [0.13093647723004956],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 2): {
        "max_iter": 100,
        "threshold_per_cycle": 0.026022828091252573,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23115665063155355,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.030894695983302592],
        "start_peakres": [0.13011414045626285],
        "start_peakres_nomask": [0.13011414045626285],
        "peakres": [-0.07246370711058724],
        "peakres_nomask": [0.11245830713546653],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 3): {
        "max_iter": 100,
        "threshold_per_cycle": 0.06500307769405649,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23118065024762538,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.7892382411029459],
        "start_peakres": [0.3250153884702824],
        "start_peakres_nomask": [0.3250153884702824],
        "peakres": [0.08844051945783668],
        "peakres_nomask": [-0.11288792115316137],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 3): {
        "max_iter": 100,
        "threshold_per_cycle": 0.026551682660084775,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23118065024762538,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [-0.030733119090230663],
        "start_peakres": [0.13275841330042387],
        "start_peakres_nomask": [0.13275841330042387],
        "peakres": [0.07619168740494403],
        "peakres_nomask": [-0.1013834010764904],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 4): {
        "max_iter": 100,
        "threshold_per_cycle": 0.07066916383960643,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23118394049874885,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.7367661224836529],
        "start_peakres": [0.35334581919803215],
        "start_peakres_nomask": [0.35334581919803215],
        "peakres": [0.08590799811833363],
        "peakres_nomask": [0.11790231095564097],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 4): {
        "max_iter": 100,
        "threshold_per_cycle": 0.02211009245788454,
        "iter_done": [100],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.23118394049874885,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [-0.08371006772816465],
        "start_peakres": [0.1105504622894227],
        "start_peakres_nomask": [0.1105504622894227],
        "peakres": [0.07337304244458982],
        "peakres_nomask": [-0.10525405495764695],
        "masksum": [42921],
        "stop_description": "Reached max_iter",
    },
}


# max_iter=10000, max_cycles=4, threshold=0.001: every plane spends its whole
# max_iter budget before the threshold -> stop_code (1, 0). With the corrected
# PSF sidelobe the Stokes-I planes need three cycles and the Q planes two; a Q
# plane's trailing 0 is the cycle its channel's I plane still ran (the channel
# keeps cycling until all of its planes have stopped). Regenerated with
# threads=1, n_mapping_parallelism=1 from the merged iteration control.
EXPECTED_DECONVOLVE_DICT_MULTI_CYCLE = {
    (0, 0, 0): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.014871819064994624,
        "iter_done": [30, 1087, 8883],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23110860147112905,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [0.0, 0.5466700143981923, 1.2576197215819305],
        "model_flux": [0.5466700143981923, 1.2576197215819305, 1.2456289213795557],
        "start_peakres": [0.35750769883669886, 0.12392010971957867, 0.0505148175765573],
        "start_peakres_nomask": [
            0.35750769883669886,
            0.1262203916589705,
            -0.08494050122022936,
        ],
        "peakres": [0.12392019063087857, 0.042899943923412606, 0.017193042352814727],
        "peakres_nomask": [
            0.13796055832859327,
            0.099030029467624,
            -0.08329402619747603,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 0): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.006306899338778819,
        "iter_done": [911, 9089, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23110860147112905,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [0.0, 0.155840414839831, 0.8254673605911406],
        "model_flux": [0.155840414839831, 0.8254673605911406, 0.8254673605911406],
        "start_peakres": [
            0.10430169385685946,
            -0.045062137024791234,
            -0.03676621243296076,
        ],
        "start_peakres_nomask": [
            0.12055068831924316,
            0.07881074285001036,
            -0.06478539423470368,
        ],
        "peakres": [-0.041759656863974835, -0.018193176421913197, -0.03676621243296076],
        "peakres_nomask": [
            0.093212808922926,
            -0.06707311854210926,
            -0.06478539423470368,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 1): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.013792080078708196,
        "iter_done": [36, 1301, 8663],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23107365193069979,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [0.0, 0.5727494093261228, 1.040091941624565],
        "model_flux": [0.5727494093261228, 1.040091941624565, 0.8670076005972925],
        "start_peakres": [
            0.33510748690170244,
            0.11486959938902963,
            -0.05031742213097422,
        ],
        "start_peakres_nomask": [
            0.33510748690170244,
            -0.13183672100151,
            0.07715072445095911,
        ],
        "peakres": [0.11486276778081829, -0.03979129587318625, 0.01637880681520796],
        "peakres_nomask": [
            -0.13208707311632742,
            0.09183673552731667,
            -0.06479796120413152,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 1): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.005963434470245462,
        "iter_done": [1035, 8965, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23107365193069979,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [0.0, 0.21969042124587013, 0.036882287814695935],
        "model_flux": [0.21969042124587013, 0.036882287814695935, 0.036882287814695935],
        "start_peakres": [
            -0.10502057730946415,
            -0.044949888608820085,
            -0.03497373965624594,
        ],
        "start_peakres_nomask": [
            -0.11825339664330481,
            -0.08817812088447091,
            -0.05908285487801882,
        ],
        "peakres": [0.04094756348751961, 0.017205003456456175, -0.03497373965624594],
        "peakres_nomask": [
            -0.10440379322863674,
            -0.07116774269526827,
            -0.05908285487801882,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 2): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.013937565241293932,
        "iter_done": [38, 1203, 8759],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23115665063155355,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [0.0, 0.6338229476786227, 1.2959593463565882],
        "model_flux": [0.6338229476786227, 1.2959593463565882, 1.176710736234387],
        "start_peakres": [
            0.3380203777669396,
            0.11594504491518846,
            -0.04464114468144219,
        ],
        "start_peakres_nomask": [
            0.3380203777669396,
            0.12318955475224164,
            0.08810452659578305,
        ],
        "peakres": [0.11594516709923056, 0.04019659454087226, -0.015822360903460127],
        "peakres_nomask": [
            0.13344617614567184,
            0.09871579160995302,
            -0.0667337663823381,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 2): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.006681893698221591,
        "iter_done": [814, 9186, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23115665063155355,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [0.0, 0.4199100901574759, -0.1454798386059527],
        "model_flux": [0.4199100901574759, -0.1454798386059527, -0.1454798386059527],
        "start_peakres": [
            0.13011414045626285,
            -0.05354144656040486,
            0.04002442051848469,
        ],
        "start_peakres_nomask": [
            0.13011414045626285,
            0.08693749500629958,
            -0.061754638317740765,
        ],
        "peakres": [-0.045054951323585044, -0.019270896107223354, 0.04002442051848469],
        "peakres_nomask": [
            0.10288861104049368,
            -0.07534401242341368,
            -0.061754638317740765,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 3): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.013514577685608608,
        "iter_done": [41, 1451, 8508],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23118065024762538,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [0.0, 0.5791989355236492, 1.1888196900450798],
        "model_flux": [0.5791989355236492, 1.1888196900450798, 1.090408303387831],
        "start_peakres": [
            0.3250153884702824,
            -0.1125809543760414,
            -0.04733498182565653,
        ],
        "start_peakres_nomask": [
            0.3250153884702824,
            -0.11605430778290478,
            0.0727629650576304,
        ],
        "peakres": [-0.11258089288036224, 0.038972632213906275, 0.01679222598406665],
        "peakres_nomask": [
            -0.11612019015867855,
            -0.09872237252180248,
            0.06297524577506201,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 3): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.005522300573776167,
        "iter_done": [880, 6453, 2667],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23118065024762538,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [0.0, 0.03298583010139605, -0.3058254894695851],
        "model_flux": [0.03298583010139605, -0.3058254894695853, -0.31723889807352257],
        "start_peakres": [
            0.13275841330042387,
            0.04869234684844154,
            -0.02962248787457348,
        ],
        "start_peakres_nomask": [
            0.13275841330042387,
            -0.07740717891640513,
            -0.05300771106446301,
        ],
        "peakres": [0.04594640247427958, 0.015924921536068712, -0.009096157145210495],
        "peakres_nomask": [
            -0.09251233546979004,
            0.06151904550355304,
            0.04728997180159948,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
    (0, 0, 4): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.014429928780634527,
        "iter_done": [30, 1130, 8840],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23118394049874885,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [0.0, 0.5123393893968381, 0.6070476836687916],
        "model_flux": [0.5123393893968381, 0.6070476836687916, 0.5456928429545845],
        "start_peakres": [
            0.35334581919803215,
            0.12010882816451446,
            -0.04823612045891114,
        ],
        "start_peakres_nomask": [
            0.35334581919803215,
            -0.12115886623533803,
            -0.07784742143657576,
        ],
        "peakres": [0.12010885614110677, -0.04161168158856207, -0.01704552479901329],
        "peakres_nomask": [
            -0.12115987770296933,
            0.10630478353572127,
            0.06582235488647444,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
    (0, 1, 4): {
        "max_iter": 10000,
        "threshold_per_cycle": 0.005315512158509648,
        "iter_done": [1251, 8749, 0],
        "gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.23118394049874885,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [0.0, 0.04575815254849027, -0.5662781946961217],
        "model_flux": [0.04575815254849027, -0.5662781946961217, -0.5662781946961217],
        "start_peakres": [
            0.1105504622894227,
            -0.045337043653807745,
            0.033513833704590774,
        ],
        "start_peakres_nomask": [
            0.1105504622894227,
            -0.08396627619926997,
            -0.058694826515291426,
        ],
        "peakres": [-0.03826919842835053, -0.015328377761425618, 0.033513833704590774],
        "peakres_nomask": [
            0.09859090409741078,
            0.07247922581594626,
            -0.058694826515291426,
        ],
        "masksum": [42921, 42921, 42921],
        "stop_description": "Reached max_iter",
    },
}


if __name__ == "__main__":
    _regenerate_truth_images()
