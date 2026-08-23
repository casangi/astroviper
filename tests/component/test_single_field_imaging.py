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
    print_deconvolve_dict,
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
    TRUTH_IMAGE_NITER0: "1uZrW7xs0dkEs1MQx3d_7yY9tolNGIWq6",
    TRUTH_IMAGE_NITER100: "1pX1ME9zrDp82Sd71_nrm3yMOVZqwpG_2",
    TRUTH_IMAGE_MULTI_CYCLE_DOUBLE: "1KzjzH4rg4-UrXO2CZgcgri6KJWlH8qwR",
    TRUTH_IMAGE_MULTI_CYCLE_SINGLE: "1CiiOg-pfwS7gbrzhUPM37feM1jWIjq1e",
}

# Default (tight) per-channel relative-difference ceiling for the reproducible
# variants. Double-precision multi_cycle and the niter0/niter100 tests reproduce
# their truth to ~1e-13 (PRIMARY_BEAM ~5e-10 at n_mapping_parallelism=5), so 1e-6 is a real
# regression guard with comfortable margin.
TRUTH_RTOL = 1e-6

# Loose ceiling for the single-precision multi_cycle image comparison -- used by
# BOTH the 1- and 12-thread variants. At threshold=0.001 the float32 deep CLEAN
# sits on a peak-selection bifurcation: a tiny float32 difference sends a channel
# onto an alternate-but-valid branch that differs by ~10% in the image (and by a
# few percent in the per-plane deconvolution history). The single-threaded run is
# bit-identical to its truth in isolation, but floating-point / thread state
# accumulated across a full test session can occasionally tip a channel, so a
# 1e-6 image bound is flaky even single-threaded. Both single-precision variants
# therefore use this loose image bound AND skip the exact deconvolution-dict
# check (dict_kind=None); the double-precision variants are the tight ReturnDict
# regression guard.
MULTI_CYCLE_SINGLE_RTOL = 0.15

# Ceiling for the direct double-vs-single multi_cycle comparison
# (test_single_field_imaging_multi_cycle_double_vs_single). float32 vs float64
# differ by up to ~10% on the channel where the deep CLEAN tips a peak-selection
# bifurcation, so this is deliberately loose -- it bounds the precision spread
# while still catching a gross regression.
MULTI_CYCLE_DOUBLE_VS_SINGLE_RTOL = 0.15

# Ceiling for the worst-case multi_cycle cross-config comparison
# (test_single_field_imaging_multi_cycle_worst_case): double / 1 thread /
# n_mapping_parallelism=5 vs single / 12 threads / n_mapping_parallelism=1, so precision, thread count and
# chunking all differ at once. Still dominated by the float32 peak-selection
# bifurcation (~10% on one channel), so it shares the same loose ceiling.
MULTI_CYCLE_WORST_CASE_RTOL = 0.15

# Deconvolve-dict floats computed FROM the float32 gridded seed: the per-major-
# cycle CLEAN trajectory (model_flux, peakres, ...) plus the thresholds derived
# from it and from the PSF (cyclethreshold, max_psf_sidelobe). In single
# precision these differ cross-platform once the deep CLEAN tips a peak-selection
# tie (~few %) or simply inherits the platform's float32 seed noise (~1e-4) --
# see the TRUTH_RTOL comment above. For a single-precision dict they are compared
# by MAGNITUDE at MULTI_CYCLE_SINGLE_DICT_RTOL with a peak-scaled atol (the peak-
# residual sign flips at the bifurcation while its magnitude is stable). iter_done
# is included because the minor-cycle count to reach the cycle threshold also
# bifurcates (~few %) once a channel tips. Everything else stays tight: the
# remaining exact-int fields (niter, masksum), stop_code, the strings, and the
# config/coordinate floats (loop_gain, min/max_psf_fraction, frequency, time) are
# all bit-reproducible. Double-precision dicts compare every field tightly.
_BIFURCATION_SENSITIVE_FIELDS = frozenset(
    {
        "model_flux",
        "start_model_flux",
        "peakres",
        "peakres_nomask",
        "start_peakres",
        "start_peakres_nomask",
        "cyclethreshold",
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
            "niter": 0,
            "nmajor": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": -1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
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
            "niter": 100,
            "nmajor": 0,
            "threshold": 0.001,
            "primary_beam_limit": 0.2,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": -1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.2,
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
    "multi_cycle": {
        "iteration_control_params": {
            "niter": 10000,
            "nmajor": 4,
            "threshold": 0.001,
            "primary_beam_limit": 0.2,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": -1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
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


def _check_deconvolve_dict(
    deconvolve_dict,
    expected,
    rtol=1e-5,
    atol=1e-8,
    loose_fields=frozenset(),
    loose_rtol=0.15,
):
    """Assert every key and field of a deconvolve ReturnDict matches ``expected``.

    ``expected`` is keyed by ``(time, pol, chan)`` tuples, with ``stop_code``
    given as a ``(major, minor)`` tuple. Integer fields (niter, iter_done,
    masksum), string fields (stokes, stop_description) and the stop code are
    compared exactly; every other (floating-point) field -- scalar or per-cycle
    history list -- is compared with ``np.allclose`` at ``rtol``/``atol``.

    Fields named in ``loose_fields`` are instead compared by MAGNITUDE at
    ``loose_rtol`` with a peak-scaled ``atol`` (``loose_rtol * max|expected|``).
    This is for the single-precision multi_cycle dict, whose per-cycle
    flux/residual trajectory bifurcates cross-platform and above one thread (see
    the TRUTH_RTOL comment): the peak-residual sign flips while its magnitude is
    stable, and the minor-cycle count itself drifts, so those fields cannot be
    pinned tightly, while niter/stop_code/masksum still can. Regenerate the
    expected literals if the imaging or iteration-control behaviour intentionally
    changes.
    """
    exact_int_fields = {"niter", "iter_done", "masksum"}
    string_fields = {"stokes", "stop_description"}

    actual = {tuple(k): v for k, v in deconvolve_dict.data.items()}
    assert set(actual) == set(expected), (
        f"deconvolve_dict planes {sorted(actual)} != expected {sorted(expected)}"
    )
    for key, exp_fields in expected.items():
        got = actual[key]
        assert set(got) == set(exp_fields), (
            f"plane {key}: fields {sorted(got)} != expected {sorted(exp_fields)}"
        )
        for field, exp_val in exp_fields.items():
            val = got[field]
            if field == "stop_code":
                assert (int(val.major), int(val.minor)) == tuple(exp_val), (
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


def _check_image_statistics(return_dict, img_av_xds, expect_mask):
    """The gathered per-plane statistics must describe the written cube: full
    (time, frequency, polarization) extent in global frequency order, and the
    NaN-ignoring mean / signed peak of every SKY_RESIDUAL plane must match a
    direct recomputation from the image store (5 chunks -> exercises the
    frequency concatenation in the reduce)."""
    stats = return_dict["image_statistics"]
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
    assert bool(residual_stats.attrs["mask_present"]) is expect_mask
    if expect_mask:
        assert np.isfinite(residual_stats["peak_masked"].values).all()
        assert (
            residual_stats["n_pixels_masked"].values
            <= residual_stats["n_pixels"].values
        ).all()
    else:
        assert np.isnan(residual_stats["mean_masked"].values).all()
    assert "T_image_statistics" in return_dict["timing_node_tasks"].columns


def _run_image_cube(
    kind,
    image_store,
    *,
    processing_function_threads,
    n_mapping_parallelism,
    single_precision_image=None,
    skunk_works=False,
    output_shard_channels=None,
    node_task_image_chunking=None,
):
    """Run ``image_cube_single_field`` for one base test ``kind`` and variant.

    Returns ``(return_dict, img_av_xds, image_params)``. The per-variant knobs are
    ``processing_function_threads``, ``n_mapping_parallelism`` and optionally
    ``single_precision_image`` (which overrides the config value when not None),
    plus ``skunk_works`` / ``output_shard_channels`` /
    ``node_task_image_chunking`` to exercise the direct-write, sharded-write and
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
    return_dict = image_cube_single_field(
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
        output_shard_channels=output_shard_channels,
        node_task_image_chunking=node_task_image_chunking,
        **config["extra_kwargs"],
    )
    img_av_xds = xr.open_zarr(image_store)
    return return_dict, img_av_xds, image_params


def _compare_to_truth(
    img_av_xds,
    truth_xds,
    variables,
    *,
    plot_saver,
    plot_prefix,
    polarization=0,
    tol=TRUTH_RTOL,
):
    """Compare ``variables`` of ``img_av_xds`` against ``truth_xds`` per channel.

    For every frequency channel and every variable a 3-panel (AV / TRUTH /
    normalized difference) comparison plot is generated -- as both a full-image
    figure and a central-50x50 zoom saved separately -- with the peak relative
    difference ``max|av - truth| / max|truth|`` annotated under the difference
    panel. A summary figure plots that peak relative difference against frequency
    for Stokes I and Q. Assertions (on ``polarization``) are deferred to a final
    pass so a single failing channel cannot prevent the remaining plots from
    being saved. Every relative difference must be < ``tol``.
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
    for i_f, channel_diffs in enumerate(per_channel_diffs):
        for var, rel_diff in channel_diffs.items():
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
    return_dict, img_av_xds, image_params = _run_image_cube(
        "niter0",
        image_store,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=5,
    )
    truth_xds = xr.open_zarr(TRUTH_IMAGE_NITER0)

    print("&&&&&&&&&" * 10)
    print("imaging_metadata_pd", return_dict["timing_node_tasks"])
    print("deconvolve_dict (global channel numbering):")
    print_deconvolve_dict(return_dict["deconvolution"])
    _check_image_statistics(return_dict, img_av_xds, expect_mask=False)

    _compare_to_truth(
        img_av_xds,
        truth_xds,
        _CONFIGS["niter0"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix=f"niter0_t{processing_function_threads}",
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
    """The sharded direct-write path (skunk_works=True + output_shard_channels)
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
        output_shard_channels=None,
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
        output_shard_channels=2,
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


def test_single_field_imaging_niter0_node_task_image_chunking(plot_saver):
    """``node_task_image_chunking`` must change only the on-disk chunk layout,
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
        node_task_image_chunking=chunking,
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
        output_shard_channels=2,
        node_task_image_chunking=chunking,
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
        plot_prefix="niter0_sharded_node_task_image_chunking",
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
    return_dict, img_av_xds, _ = _run_image_cube(
        "niter100",
        image_store,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=5,
    )
    truth_xds = xr.open_zarr(TRUTH_IMAGE_NITER100)

    print("&&&&&&&&&" * 10)
    print("imaging_metadata_pd:")
    print(return_dict["timing_node_tasks"].to_string())
    print("deconvolve_dict (global channel numbering):")
    print_deconvolve_dict(return_dict["deconvolution"])
    _check_deconvolve_dict(
        return_dict["deconvolution"], EXPECTED_DECONVOLVE_DICT_NITER100
    )
    _check_image_statistics(return_dict, img_av_xds, expect_mask=True)

    _compare_to_truth(
        img_av_xds,
        truth_xds,
        _CONFIGS["niter100"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix=f"niter100_t{processing_function_threads}",
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
    return_dict, img_av_xds, _ = _run_image_cube(
        "multi_cycle",
        image_store,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=n_mapping_parallelism,
        single_precision_image=single_precision_image,
    )
    truth_xds = xr.open_zarr(truth_image)

    # Only the double-precision multi_cycle checks the deconvolution ReturnDict:
    # it is stable across thread count and chunking. The single-precision deep
    # CLEAN sits on a float32 peak-selection bifurcation (see
    # MULTI_CYCLE_SINGLE_RTOL), so its per-plane history is not reproducible
    # tightly enough to pin -- both single-precision variants pass dict_kind=None
    # and are validated only by the (loosely bounded) image comparison below.
    expected_dict = {
        "double": EXPECTED_DECONVOLVE_DICT_MULTI_CYCLE,
        None: None,
    }[dict_kind]
    if expected_dict is not None:
        loose_fields = (
            _BIFURCATION_SENSITIVE_FIELDS if dict_kind == "single" else frozenset()
        )
        _check_deconvolve_dict(
            return_dict["deconvolution"],
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
    )

    print(return_dict["timing_node_tasks"].T)
    print(return_dict["timing_node_tasks"]["T_deconvolve"])
    print(return_dict["timing_node_tasks"]["T_residual_cycle"])
    print(return_dict["timing_node_tasks"]["T_image_cube_task"])


def test_single_field_imaging_multi_cycle_double_vs_single(plot_saver):
    """Directly compare the double- and single-precision multi_cycle truths.

    A float32-vs-float64 comparison of the deep multi-cycle CLEAN. Both truths
    are generated identically except for precision (threads=1, n_mapping_parallelism=1), so
    this isolates the precision difference: the single-precision deep CLEAN can
    tip a peak-selection bifurcation on a channel, hence the deliberately loose
    ``MULTI_CYCLE_DOUBLE_VS_SINGLE_RTOL``. Generates the per-channel comparison
    plots (double / single / difference).
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
    )


def test_single_field_imaging_multi_cycle_worst_case(plot_saver):
    """Worst-case multi_cycle cross-config comparison.

    Compares the two most-divergent valid multi_cycle runs -- double precision /
    1 thread / n_mapping_parallelism=5 against single precision / 12 threads / n_mapping_parallelism=1 --
    so every knob that can perturb the result (precision, thread count, chunking)
    differs at once. The spread is dominated by the float32 deep-CLEAN
    peak-selection bifurcation, hence the deliberately loose
    ``MULTI_CYCLE_WORST_CASE_RTOL``. Generates the per-channel comparison plots.
    Unlike the double-vs-single test, neither configuration is an on-disk truth,
    so both are imaged here.
    """
    _ensure_ps_store()
    _, double_xds, _ = _run_image_cube(
        "multi_cycle",
        "twhya_selfcal_5chans_lsrk_multi_cycle_worstcase_double_t1_c5.img.zarr",
        processing_function_threads=1,
        n_mapping_parallelism=5,
        single_precision_image=False,
    )
    _, single_xds, _ = _run_image_cube(
        "multi_cycle",
        "twhya_selfcal_5chans_lsrk_multi_cycle_worstcase_single_t12_c1.img.zarr",
        processing_function_threads=12,
        n_mapping_parallelism=1,
        single_precision_image=True,
    )
    _compare_to_truth(
        double_xds,
        single_xds,
        _CONFIGS["multi_cycle"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix="multi_cycle_worst_case_double_t1_c5_vs_single_t12_c1",
        tol=MULTI_CYCLE_WORST_CASE_RTOL,
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
# Expected per-plane deconvolution ReturnDicts.
#
# Keyed by ``(time, pol, chan)`` with each history list holding one entry per
# major cycle; checked by ``_check_deconvolve_dict``. Within a precision they
# are independent of thread count and chunking (iteration control is computed
# per (time, frequency, polarization) plane). Only the double-precision
# multi_cycle is pinned to a literal: the single-precision deep CLEAN is chaotic
# near the float32 peak-selection bifurcation (see MULTI_CYCLE_SINGLE_RTOL), so
# it is not checked against a literal. Regenerate if the imaging or
# iteration-control behaviour intentionally changes.
# ---------------------------------------------------------------------------

# niter=100, nmajor=0, threshold=0.001: deconvolves to the iteration limit
# (the threshold is not reached) -> stop_code (1, 0).
EXPECTED_DECONVOLVE_DICT_NITER100 = {
    (0, 0, 0): {
        "niter": 100,
        "cyclethreshold": 0.07150153976733978,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.3445475295671201,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.9677845797776481],
        "start_peakres": [0.35750769883669886],
        "start_peakres_nomask": [0.35750769883669886],
        "peakres": [-0.08899075577822424],
        "peakres_nomask": [-0.11071206252751019],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
    (0, 1, 0): {
        "niter": 100,
        "cyclethreshold": 0.02411013766384864,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.3445475295671201,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.08916942945052633],
        "start_peakres": [0.12055068831924319],
        "start_peakres_nomask": [0.12055068831924319],
        "peakres": [0.07430471569361485],
        "peakres_nomask": [0.08951620816899497],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
    (0, 0, 1): {
        "niter": 100,
        "cyclethreshold": 0.06702149738034051,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.344533866326935,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.8096180886404875],
        "start_peakres": [0.3351074869017025],
        "start_peakres_nomask": [0.3351074869017025],
        "peakres": [0.09003297370692762],
        "peakres_nomask": [0.09003297370692762],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
    (0, 1, 1): {
        "niter": 100,
        "cyclethreshold": 0.023650679328660947,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.344533866326935,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.0806183920125066],
        "start_peakres": [-0.11825339664330473],
        "start_peakres_nomask": [-0.11825339664330473],
        "peakres": [0.07750591941443867],
        "peakres_nomask": [-0.08463490573442017],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
    (0, 0, 2): {
        "niter": 100,
        "cyclethreshold": 0.06760407555338792,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.34448079976471313,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.9100113948102967],
        "start_peakres": [0.3380203777669396],
        "start_peakres_nomask": [0.3380203777669396],
        "peakres": [0.08979484245919653],
        "peakres_nomask": [0.08979484245919653],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
    (0, 1, 2): {
        "niter": 100,
        "cyclethreshold": 0.026022828091252573,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.34448079976471313,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.09208456255193924],
        "start_peakres": [0.13011414045626285],
        "start_peakres_nomask": [0.13011414045626285],
        "peakres": [-0.07639895031706921],
        "peakres_nomask": [-0.08528560955343739],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
    (0, 0, 3): {
        "niter": 100,
        "cyclethreshold": 0.06500307769405649,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.34447908250868625,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.7685137907285896],
        "start_peakres": [0.3250153884702824],
        "start_peakres_nomask": [0.3250153884702824],
        "peakres": [0.09214764521763133],
        "peakres_nomask": [0.09214764521763133],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
    (0, 1, 3): {
        "niter": 100,
        "cyclethreshold": 0.026551682660084775,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.34447908250868625,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.03533077827276314],
        "start_peakres": [0.13275841330042387],
        "start_peakres_nomask": [0.13275841330042387],
        "peakres": [0.07901175475017379],
        "peakres_nomask": [0.08541584586642978],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
    (0, 0, 4): {
        "niter": 100,
        "cyclethreshold": 0.07066916383960643,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.3444778231912157,
        "stop_code": (1, 0),
        "stokes": "I",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [0.6857668900749301],
        "start_peakres": [0.35334581919803215],
        "start_peakres_nomask": [0.35334581919803215],
        "peakres": [0.08930366061252694],
        "peakres_nomask": [0.09027342549634497],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
    (0, 1, 4): {
        "niter": 100,
        "cyclethreshold": 0.02211009245788454,
        "iter_done": [100],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.2,
        "max_psf_sidelobe": 0.3444778231912157,
        "stop_code": (1, 0),
        "stokes": "Q",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [0.0],
        "model_flux": [-0.09793243292320182],
        "start_peakres": [0.1105504622894227],
        "start_peakres_nomask": [0.1105504622894227],
        "peakres": [0.07635328920165872],
        "peakres_nomask": [-0.09901618422777503],
        "masksum": [62500],
        "stop_description": "Reached the iteration limit",
    },
}


# niter=10000, nmajor=4, threshold=0.001: deconvolves to the major-cycle limit
# (the threshold is not reached) -> stop_code (9, 0).
EXPECTED_DECONVOLVE_DICT_MULTI_CYCLE = {
    (0, 0, 0): {
        "niter": 10000,
        "cyclethreshold": 0.035900881810030635,
        "iter_done": [3, 22, 290, 1595],
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
            1.159797427505857,
        ],
        "model_flux": [
            0.09718702204548665,
            0.4825330600317771,
            1.159797427505857,
            1.7520032099811906,
        ],
        "start_peakres": [
            0.35750769883669886,
            0.2631183190329552,
            0.13405951306715252,
            0.07343511202635758,
        ],
        "start_peakres_nomask": [
            0.35750769883669886,
            0.2631183190329552,
            0.13405951306715252,
            0.09198393537130387,
        ],
        "peakres": [
            0.26311837258523163,
            0.13442036393758017,
            0.06946478831746959,
            0.03585352130809877,
        ],
        "peakres_nomask": [
            0.26311837258523163,
            0.13442036393758017,
            -0.10285130475892759,
            -0.08985693562049257,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 0): {
        "niter": 10000,
        "cyclethreshold": 0.013208288568023155,
        "iter_done": [8, 715, 2066, 3653],
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
            0.06398640705714076,
            -0.13570365233397255,
            -0.11925250561112496,
        ],
        "model_flux": [
            0.06398640705714076,
            -0.13570365233397255,
            -0.11925250561112496,
            -0.2702666200375462,
        ],
        "start_peakres": [
            0.12055068831924319,
            0.09668704802810893,
            0.05755014392609231,
            0.0363014930639613,
        ],
        "start_peakres_nomask": [
            0.12055068831924319,
            0.09668704802810893,
            -0.06483231242826937,
            -0.056938615472674585,
        ],
        "peakres": [
            0.0959953340255374,
            -0.04950471250278252,
            -0.02555678086874822,
            -0.013208065044139546,
        ],
        "peakres_nomask": [
            0.0959953340255374,
            0.07869248490248072,
            -0.05892657840905252,
            -0.04888484815084779,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 0, 1): {
        "niter": 10000,
        "cyclethreshold": 0.03349452616839233,
        "iter_done": [3, 26, 372, 1637],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.344533866326935,
        "stop_code": (9, 0),
        "stokes": "I",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.09086301993655113,
            0.48714152453270637,
            1.055466672602224,
        ],
        "model_flux": [
            0.09086301993655113,
            0.48714152453270637,
            1.055466672602224,
            1.3929070027327735,
        ],
        "start_peakres": [
            0.3351074869017025,
            0.2465803127553856,
            -0.12563841463957237,
            0.06689310012081892,
        ],
        "start_peakres_nomask": [
            0.3351074869017025,
            0.2465803127553856,
            -0.12563841463957237,
            -0.07175340967366711,
        ],
        "peakres": [
            0.24658036446829307,
            -0.12563929217840017,
            -0.0648112893815567,
            -0.03349328036668022,
        ],
        "peakres_nomask": [
            0.24658036446829307,
            -0.12563929217840017,
            -0.07955797944228352,
            -0.06498572800663095,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 1): {
        "niter": 10000,
        "cyclethreshold": 0.01294509960945086,
        "iter_done": [15, 854, 2012, 3310],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.344533866326935,
        "stop_code": (9, 0),
        "stokes": "Q",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.006016719503292748,
            0.2056671885703777,
            0.08811593952745678,
        ],
        "model_flux": [
            0.006016719503292748,
            0.2056671885703777,
            0.08811593952745678,
            0.07855219224260607,
        ],
        "start_peakres": [
            -0.11825339664330473,
            0.09308343885970413,
            -0.052830647582281204,
            -0.03481885440117216,
        ],
        "start_peakres_nomask": [
            -0.11825339664330473,
            0.09308343885970413,
            -0.06419945960041408,
            -0.04972177480089822,
        ],
        "peakres": [
            -0.09389724733759419,
            0.04849935769766049,
            -0.02504852860563592,
            -0.01293494736806545,
        ],
        "peakres_nomask": [
            -0.09389724733759419,
            -0.07604044511732594,
            -0.05753203513591941,
            -0.04780307601839216,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 0, 2): {
        "niter": 10000,
        "cyclethreshold": 0.033319147131844896,
        "iter_done": [3, 28, 368, 1614],
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
            0.5491671818352681,
            1.1218611311657138,
        ],
        "model_flux": [
            0.0916035160783401,
            0.5491671818352681,
            1.1218611311657138,
            1.1287944257041806,
        ],
        "start_peakres": [
            0.3380203777669396,
            0.2464167473657761,
            0.12466861312466165,
            -0.06623974818319509,
        ],
        "start_peakres_nomask": [
            0.3380203777669396,
            0.2464167473657761,
            0.12466861312466165,
            0.06986476369698277,
        ],
        "peakres": [
            0.246416800747472,
            0.12511840620530953,
            0.0644818659551853,
            0.0332715946474775,
        ],
        "peakres_nomask": [
            0.246416800747472,
            0.12511840620530953,
            0.0776866192006112,
            0.0643197276130056,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 2): {
        "niter": 10000,
        "cyclethreshold": 0.014318313868060933,
        "iter_done": [5, 612, 1963, 3026],
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
            0.03497499853849212,
            0.6018001645297726,
            0.9931123967009676,
        ],
        "model_flux": [
            0.03497499853849212,
            0.6018001645297726,
            0.9931123967009676,
            1.007334469762727,
        ],
        "start_peakres": [
            0.13011414045626285,
            -0.10400362324374737,
            -0.05773290986642028,
            -0.03473492134482885,
        ],
        "start_peakres_nomask": [
            0.13011414045626285,
            -0.10400362324374737,
            -0.0816911358287811,
            -0.052956508157880555,
        ],
        "peakres": [
            -0.10401673272264614,
            -0.053702486822943284,
            -0.027709940830452882,
            -0.014317940136346425,
        ],
        "peakres_nomask": [
            -0.10401673272264614,
            -0.08080850066390213,
            -0.0717053634202924,
            -0.052833260398108546,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 0, 3): {
        "niter": 10000,
        "cyclethreshold": 0.03293603309341919,
        "iter_done": [3, 28, 441, 1772],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.34447908250868625,
        "stop_code": (9, 0),
        "stokes": "I",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.08878698615066782,
            0.5086397677138595,
            0.9458757420905952,
        ],
        "model_flux": [
            0.08878698615066782,
            0.5086397677138595,
            0.9458757420905952,
            1.5433574528710663,
        ],
        "start_peakres": [
            0.3250153884702824,
            0.24324762521379614,
            -0.12365112726419497,
            -0.06921296065803335,
        ],
        "start_peakres_nomask": [
            0.3250153884702824,
            0.24324762521379614,
            -0.12365112726419497,
            0.07271240581112574,
        ],
        "peakres": [
            0.24324767373115244,
            -0.12365099019310297,
            0.06374075092080266,
            0.03293408980973097,
        ],
        "peakres_nomask": [
            0.24324767373115244,
            -0.12365099019310297,
            0.07802848851809419,
            0.061966590096323146,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 3): {
        "niter": 10000,
        "cyclethreshold": 0.014335478343885431,
        "iter_done": [7, 694, 1861, 3082],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.34447908250868625,
        "stop_code": (9, 0),
        "stokes": "Q",
        "frequency": 372764411640.15845,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.011751165304778703,
            -0.03164703664044338,
            -0.2342840038181733,
        ],
        "model_flux": [
            0.011751165304778703,
            -0.03164703664044338,
            -0.2342840038181733,
            -0.32699774390757613,
        ],
        "start_peakres": [
            0.13275841330042387,
            -0.10396325269068865,
            0.058687765846087675,
            -0.035089432781414145,
        ],
        "start_peakres_nomask": [
            0.13275841330042387,
            -0.10396325269068865,
            -0.0631802942904398,
            -0.04830070435469852,
        ],
        "peakres": [
            -0.10396225525444691,
            0.05371269532409429,
            -0.02774329719236088,
            -0.01433033577629681,
        ],
        "peakres_nomask": [
            -0.10396225525444691,
            0.08186354676928359,
            -0.05910770216233156,
            -0.0437291158976326,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 0, 4): {
        "niter": 10000,
        "cyclethreshold": 0.03573911986540156,
        "iter_done": [3, 20, 290, 1497],
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
            0.09632811291794371,
            0.44859343492168907,
            0.7335580527200094,
        ],
        "model_flux": [
            0.09632811291794371,
            0.44859343492168907,
            0.7335580527200094,
            0.9831520950925193,
        ],
        "start_peakres": [
            0.35334581919803215,
            0.2624829860334644,
            0.13386095443544138,
            0.07033269743888945,
        ],
        "start_peakres_nomask": [
            0.35334581919803215,
            0.2624829860334644,
            0.13386095443544138,
            0.07927601244668904,
        ],
        "peakres": [
            0.2624830392160828,
            0.1338609943238513,
            -0.06916578747956766,
            0.03571772488882554,
        ],
        "peakres_nomask": [
            0.2624830392160828,
            0.1338609943238513,
            0.08967755696001663,
            0.07547805510487447,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 4): {
        "niter": 10000,
        "cyclethreshold": 0.012186263452403511,
        "iter_done": [32, 1003, 1932, 3569],
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
            -0.01697025554183381,
            0.1709945982838767,
            0.2896250041338442,
        ],
        "model_flux": [
            -0.01697025554183381,
            0.1709945982838767,
            0.2896250041338442,
            0.18891088243462262,
        ],
        "start_peakres": [
            0.1105504622894227,
            0.08913756341350447,
            0.05230757236200801,
            -0.031314655510733445,
        ],
        "start_peakres_nomask": [
            0.1105504622894227,
            -0.09824161233271012,
            -0.07306211302917887,
            -0.04473555043359731,
        ],
        "peakres": [
            0.08842375422031411,
            -0.04568214338258588,
            0.02358403092446981,
            -0.012163852873695542,
        ],
        "peakres_nomask": [
            -0.10032285401796909,
            -0.09196188495585823,
            -0.05946674193127636,
            -0.04122865035022493,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
}


if __name__ == "__main__":
    _regenerate_truth_images()
