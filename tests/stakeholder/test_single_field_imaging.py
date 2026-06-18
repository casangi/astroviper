"""Stakeholder tests for single-field imaging.

These tests run ``image_cube_single_field`` and compare the astroviper output
against pre-computed astroviper *truth* images that live alongside this file.
Each base test exists in several variants that differ only in knobs which must
*not* change the result (the processing-function thread count and, for the
multi-cycle test, the number of compute chunks); every variant must therefore
reproduce its truth image to within ``TRUTH_RTOL``.

The truth images are themselves astroviper output: a single canonical
reference per base test, regenerated at double precision,
``processing_function_threads=1``, ``n_chunks=1`` and ``skunk_works=False``
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
_PS_STORE_DRIVE_ID = "1BRe3cD6YAWkn-jSPbClGGM9VlbxHP_yn"

# Astroviper truth (reference) images. niter0 and niter100 each have a single
# double-precision truth; multi_cycle has both a double- and a single-precision
# truth because its deep CLEAN is precision-sensitive. All are regenerated at
# processing_function_threads=1, n_chunks=1, skunk_works=False (see
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

# Default (tight) per-channel relative-difference ceiling. Double-precision
# variants reproduce their truth to ~1e-13 (PRIMARY_BEAM ~5e-10 at n_chunks=5),
# so 1e-6 is a real regression guard with comfortable margin. The multi-threaded
# single-precision multi_cycle variant carries a much looser per-variant
# tolerance: at threshold=0.001 the deep CLEAN reaches the peak-selection
# bifurcation, where a float32 thread-ordering difference flips one channel onto
# an alternate-but-valid solution (~9%), so that variant is bounded at 0.15.
TRUTH_RTOL = 1e-6

# Ceiling for the direct double-vs-single multi_cycle comparison
# (test_single_field_imaging_multi_cycle_double_vs_single). float32 vs float64
# differ by up to ~10% on the channel where the deep CLEAN tips a peak-selection
# bifurcation, so this is deliberately loose -- it bounds the precision spread
# while still catching a gross regression.
MULTI_CYCLE_DOUBLE_VS_SINGLE_RTOL = 0.15

# Ceiling for the worst-case multi_cycle cross-config comparison
# (test_single_field_imaging_multi_cycle_worst_case): double / 1 thread /
# n_chunks=5 vs single / 12 threads / n_chunks=1, so precision, thread count and
# chunking all differ at once. Still dominated by the float32 peak-selection
# bifurcation (~10% on one channel), so it shares the same loose ceiling.
MULTI_CYCLE_WORST_CASE_RTOL = 0.15

# Imaging weighting is identical for every base test.
IMAGING_WEIGHTS_PARAMS = {
    "weighting": "briggs",
    "robust": 0.5,
    "casa_weighting_implementation": True,
}

# Per-base-test imaging configuration. Everything fixed for a base test lives
# here; the per-variant knobs (``processing_function_threads``, ``n_chunks`` and
# ``single_precision_image``) are supplied by each test. ``skunk_works`` is
# always False and the truth images are generated from these configs at double
# precision, threads=1, n_chunks=1.
_CONFIGS = {
    "niter0": {
        "iteration_control_params": {
            "niter": 0,
            "nmajor": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 1,
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
        "disk_chunk_sizes": {"frequency": 2},
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
            "cycleniter": 1,
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
        "disk_chunk_sizes": {"frequency": 1},
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
            "cycleniter": 1,
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
        "disk_chunk_sizes": {"frequency": 5},
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
    # save = True
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


def _run_image_cube(
    kind,
    image_store,
    *,
    processing_function_threads,
    n_chunks,
    single_precision_image=None,
):
    """Run ``image_cube_single_field`` for one base test ``kind`` and variant.

    Returns ``(return_dict, img_av_xds, image_params)``. ``skunk_works`` is
    always False; the per-variant knobs are ``processing_function_threads``,
    ``n_chunks`` and optionally ``single_precision_image`` (which overrides the
    config value when not None). Everything else comes from ``_CONFIGS[kind]``,
    so the truth images (generated from the same configs at double precision,
    threads=1, n_chunks=1) and every variant share an identical imaging setup.
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
        n_chunks=n_chunks,
        overwrite=True,
        disk_chunk_sizes=config["disk_chunk_sizes"],
        vizualize_graph=False,
        skunk_works=False,
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
        n_chunks=5,
    )
    truth_xds = xr.open_zarr(TRUTH_IMAGE_NITER0)

    print("&&&&&&&&&" * 10)
    print("imaging_metadata_pd", return_dict["timing_node_tasks"])
    print("deconvolve_dict (global channel numbering):")
    print_deconvolve_dict(return_dict["deconvolution"])

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
    assert np.allclose(
        PB_v1, PB_v2
    ), "airy_disk_rorder and airy_disk_rorder_v2 produced different primary beams."


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
        n_chunks=5,
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

    _compare_to_truth(
        img_av_xds,
        truth_xds,
        _CONFIGS["niter100"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix=f"niter100_t{processing_function_threads}",
    )


@pytest.mark.parametrize(
    "processing_function_threads, n_chunks, single_precision_image, tol, dict_kind",
    [
        (1, 5, False, TRUTH_RTOL, "double"),
        (12, 5, False, TRUTH_RTOL, "double"),
        (1, 1, False, TRUTH_RTOL, "double"),
        (12, 1, False, TRUTH_RTOL, "double"),
        (1, 1, True, TRUTH_RTOL, "single"),
        (12, 1, True, 0.15, None),
    ],
)
def test_single_field_imaging_multi_cycle(
    plot_saver,
    processing_function_threads,
    n_chunks,
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
        f"t{processing_function_threads}_c{n_chunks}_{precision_tag}.img.zarr"
    )
    return_dict, img_av_xds, _ = _run_image_cube(
        "multi_cycle",
        image_store,
        processing_function_threads=processing_function_threads,
        n_chunks=n_chunks,
        single_precision_image=single_precision_image,
    )
    truth_xds = xr.open_zarr(truth_image)

    print("&&&&&&&&&" * 10)
    print("imaging_metadata_pd:")
    print(return_dict["timing_node_tasks"].to_string())
    print("deconvolve_dict (global channel numbering):")
    print_deconvolve_dict(return_dict["deconvolution"])
    # The deconvolution ReturnDict is independent of threads and chunks within a
    # precision. The float fields differ slightly between float32 and float64, so
    # there is a separate single-precision literal. iter_done is an exact-match
    # field; with threshold=0.2 the run stops early and single/double iter_done
    # agree, but the multi-threaded single-precision variant skips the dict check
    # (dict_kind is None) so a float32 thread-ordering nudge near an iteration
    # boundary cannot make it flaky -- its image is still validated below.
    expected_dict = {
        "double": EXPECTED_DECONVOLVE_DICT_MULTI_CYCLE,
        "single": EXPECTED_DECONVOLVE_DICT_MULTI_CYCLE_SINGLE,
        None: None,
    }[dict_kind]
    if expected_dict is not None:
        _check_deconvolve_dict(return_dict["deconvolution"], expected_dict)

    _compare_to_truth(
        img_av_xds,
        truth_xds,
        _CONFIGS["multi_cycle"]["compare_variables"],
        plot_saver=plot_saver,
        plot_prefix=(
            f"multi_cycle_t{processing_function_threads}_c{n_chunks}_{precision_tag}"
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
    are generated identically except for precision (threads=1, n_chunks=1), so
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
    1 thread / n_chunks=5 against single precision / 12 threads / n_chunks=1 --
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
        n_chunks=5,
        single_precision_image=False,
    )
    _, single_xds, _ = _run_image_cube(
        "multi_cycle",
        "twhya_selfcal_5chans_lsrk_multi_cycle_worstcase_single_t12_c1.img.zarr",
        processing_function_threads=12,
        n_chunks=1,
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
    """Regenerate every astroviper truth image at threads=1, n_chunks=1.

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
            n_chunks=1,
            single_precision_image=single_precision_image,
        )


# ---------------------------------------------------------------------------
# Expected per-plane deconvolution ReturnDicts.
#
# Keyed by ``(time, pol, chan)`` with each history list holding one entry per
# major cycle; checked by ``_check_deconvolve_dict``. Within a precision they
# are independent of thread count and chunking (iteration control is computed
# per (time, frequency, polarization) plane), but single precision can take a
# different iteration count, so multi_cycle carries both a double- and a
# single-precision literal. Regenerate if the imaging or iteration-control
# behaviour intentionally changes.
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


# multi_cycle, single precision (float32), threshold=0.001. The iteration
# path matches the double-precision run (identical iter_done) but the float
# fields differ at the float32 level. Asserted only for the single-threaded
# single-precision variant.
EXPECTED_DECONVOLVE_DICT_MULTI_CYCLE_SINGLE = {
    (0, 0, 0): {
        "niter": 10000,
        "cyclethreshold": 0.03590087109364959,
        "iter_done": [3, 22, 290, 1595],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.3445475399494171,
        "stop_code": (9, 0),
        "stokes": "I",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.09718703478574753,
            0.4825330972671509,
            1.1597975492477417,
        ],
        "model_flux": [
            0.09718703478574753,
            0.4825330972671509,
            1.1597975492477417,
            1.7520498037338257,
        ],
        "start_peakres": [
            0.35750773549079895,
            0.26311829686164856,
            0.13405951857566833,
            0.07343511283397675,
        ],
        "start_peakres_nomask": [
            0.35750773549079895,
            0.26311829686164856,
            0.13405951857566833,
            0.09198445081710815,
        ],
        "peakres": [
            0.2631183862686157,
            0.13442035019397736,
            0.06946476548910141,
            0.03585322946310043,
        ],
        "peakres_nomask": [
            0.2631183862686157,
            0.13442035019397736,
            -0.10285143554210663,
            -0.08985655009746552,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 0): {
        "niter": 10000,
        "cyclethreshold": 0.013208423494340593,
        "iter_done": [8, 715, 2066, 3691],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.3445475399494171,
        "stop_code": (9, 0),
        "stokes": "Q",
        "frequency": 372762580492.5155,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.06398636102676392,
            -0.13570334017276764,
            -0.11925153434276581,
        ],
        "model_flux": [
            0.06398636102676392,
            -0.13570334017276764,
            -0.11925153434276581,
            -0.2717492878437042,
        ],
        "start_peakres": [
            0.12055067718029022,
            0.09668704122304916,
            0.05755025893449783,
            0.03630160167813301,
        ],
        "start_peakres_nomask": [
            0.12055067718029022,
            0.09668704122304916,
            -0.06483209878206253,
            -0.05693851038813591,
        ],
        "peakres": [
            0.09599532932043076,
            -0.04950482025742531,
            -0.02555704116821289,
            -0.013187837786972523,
        ],
        "peakres_nomask": [
            0.09599532932043076,
            0.07869172841310501,
            -0.058926090598106384,
            -0.04875190183520317,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 0, 1): {
        "niter": 10000,
        "cyclethreshold": 0.033494540031693165,
        "iter_done": [3, 26, 372, 1629],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.34453389048576355,
        "stop_code": (9, 0),
        "stokes": "I",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.09086302667856216,
            0.48714160919189453,
            1.055466651916504,
        ],
        "model_flux": [
            0.09086302667856216,
            0.48714160919189453,
            1.055466651916504,
            1.433363676071167,
        ],
        "start_peakres": [
            0.3351075053215027,
            0.2465803176164627,
            -0.1256384253501892,
            0.06689311563968658,
        ],
        "start_peakres_nomask": [
            0.3351075053215027,
            0.2465803176164627,
            -0.1256384253501892,
            -0.07175352424383163,
        ],
        "peakres": [
            0.24658039212226868,
            -0.12563930451869965,
            -0.06481131166219711,
            0.03347983956336975,
        ],
        "peakres_nomask": [
            0.24658039212226868,
            -0.12563930451869965,
            -0.0795576274394989,
            -0.0650465115904808,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 1): {
        "niter": 10000,
        "cyclethreshold": 0.012953287189158197,
        "iter_done": [15, 854, 2012, 3325],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.34453389048576355,
        "stop_code": (9, 0),
        "stokes": "Q",
        "frequency": 372763190875.0631,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.006016768515110016,
            0.20566707849502563,
            0.07336804270744324,
        ],
        "model_flux": [
            0.006016768515110016,
            0.20566707849502563,
            0.07336804270744324,
            0.07113844156265259,
        ],
        "start_peakres": [
            -0.11825323104858398,
            0.09308347851037979,
            -0.052830714732408524,
            -0.03469909355044365,
        ],
        "start_peakres_nomask": [
            -0.11825323104858398,
            0.09308347851037979,
            -0.06419844180345535,
            -0.04902895539999008,
        ],
        "peakres": [
            -0.09389723837375641,
            0.04849942401051521,
            -0.025064369663596153,
            -0.012949423864483833,
        ],
        "peakres_nomask": [
            -0.09389723837375641,
            -0.07604039460420609,
            -0.057214319705963135,
            -0.04710587114095688,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 0, 2): {
        "niter": 10000,
        "cyclethreshold": 0.033319138567790896,
        "iter_done": [3, 28, 368, 1601],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.34448081254959106,
        "stop_code": (9, 0),
        "stokes": "I",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.09160351008176804,
            0.5491672158241272,
            1.121861219406128,
        ],
        "model_flux": [
            0.09160351008176804,
            0.5491672158241272,
            1.121861219406128,
            1.1933925151824951,
        ],
        "start_peakres": [
            0.33802035450935364,
            0.24641676247119904,
            0.12466861307621002,
            -0.06623998284339905,
        ],
        "start_peakres_nomask": [
            0.33802035450935364,
            0.24641676247119904,
            0.12466861307621002,
            0.06986557692289352,
        ],
        "peakres": [
            0.24641677737236023,
            0.12511801719665527,
            0.06448184698820114,
            0.03331746906042099,
        ],
        "peakres_nomask": [
            0.24641677737236023,
            0.12511801719665527,
            0.07768689095973969,
            0.06477666646242142,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 2): {
        "niter": 10000,
        "cyclethreshold": 0.01433482534525199,
        "iter_done": [5, 612, 1962, 3012],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.34448081254959106,
        "stop_code": (9, 0),
        "stokes": "Q",
        "frequency": 372763801257.61084,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            0.034974999725818634,
            0.601800799369812,
            1.0022575855255127,
        ],
        "model_flux": [
            0.034974999725818634,
            0.601800799369812,
            1.0022575855255127,
            1.0473482608795166,
        ],
        "start_peakres": [
            0.1301141381263733,
            -0.10400363802909851,
            -0.05773300305008888,
            -0.03526977822184563,
        ],
        "start_peakres_nomask": [
            0.1301141381263733,
            -0.10400363802909851,
            -0.08169113099575043,
            -0.053010307252407074,
        ],
        "peakres": [
            -0.1040167510509491,
            -0.05370248854160309,
            -0.027741894125938416,
            -0.014333546161651611,
        ],
        "peakres_nomask": [
            -0.1040167510509491,
            -0.08080887049436569,
            -0.07194143533706665,
            -0.05283176526427269,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 0, 3): {
        "niter": 10000,
        "cyclethreshold": 0.03293603412442003,
        "iter_done": [3, 28, 441, 1761],
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
            0.08878699690103531,
            0.5086398124694824,
            0.9458757042884827,
        ],
        "model_flux": [
            0.08878699690103531,
            0.5086398124694824,
            0.9458757042884827,
            1.5192127227783203,
        ],
        "start_peakres": [
            0.32501542568206787,
            0.2432476431131363,
            -0.12365113198757172,
            -0.06921298056840897,
        ],
        "start_peakres_nomask": [
            0.32501542568206787,
            0.2432476431131363,
            -0.12365113198757172,
            0.07271213084459305,
        ],
        "peakres": [
            0.24324765801429749,
            -0.12365101277828217,
            0.06374075263738632,
            0.03292318433523178,
        ],
        "peakres_nomask": [
            0.24324765801429749,
            -0.12365101277828217,
            0.07802814990282059,
            0.062044233083724976,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 3): {
        "niter": 10000,
        "cyclethreshold": 0.014314553233285565,
        "iter_done": [7, 694, 1864, 3095],
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
            0.011751163750886917,
            -0.03219442069530487,
            -0.22498169541358948,
        ],
        "model_flux": [
            0.011751163750886917,
            -0.03219442069530487,
            -0.22498169541358948,
            -0.3274250328540802,
        ],
        "start_peakres": [
            0.13275839388370514,
            -0.10396327078342438,
            0.05845488980412483,
            -0.034695617854595184,
        ],
        "start_peakres_nomask": [
            0.13275839388370514,
            -0.10396327078342438,
            -0.06288439780473709,
            -0.04791884124279022,
        ],
        "peakres": [
            -0.1039622575044632,
            -0.05366649851202965,
            0.027702800929546356,
            -0.014308770187199116,
        ],
        "peakres_nomask": [
            -0.1039622575044632,
            0.08219315856695175,
            -0.05895431712269783,
            -0.04318733140826225,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 0, 4): {
        "niter": 10000,
        "cyclethreshold": 0.035739132935993734,
        "iter_done": [3, 20, 290, 1497],
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
            0.09632812440395355,
            0.44859346747398376,
            0.7335582971572876,
        ],
        "model_flux": [
            0.09632812440395355,
            0.44859346747398376,
            0.7335582971572876,
            0.9831525087356567,
        ],
        "start_peakres": [
            0.3533458709716797,
            0.26248297095298767,
            0.13386096060276031,
            0.07033269107341766,
        ],
        "start_peakres_nomask": [
            0.3533458709716797,
            0.26248297095298767,
            0.13386096060276031,
            0.07927602529525757,
        ],
        "peakres": [
            0.26248303055763245,
            0.1338609755039215,
            -0.06916581094264984,
            0.035717736929655075,
        ],
        "peakres_nomask": [
            0.26248303055763245,
            0.1338609755039215,
            0.08967732638120651,
            0.07547777891159058,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
    (0, 1, 4): {
        "niter": 10000,
        "cyclethreshold": 0.012188658191709445,
        "iter_done": [32, 1003, 1925, 3573],
        "loop_gain": 0.1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_psf_sidelobe": 0.3444778323173523,
        "stop_code": (9, 0),
        "stokes": "Q",
        "frequency": 372765022022.7062,
        "time": 0.0,
        "start_model_flux": [
            0.0,
            -0.0169703159481287,
            0.17099492251873016,
            0.30038413405418396,
        ],
        "model_flux": [
            -0.0169703159481287,
            0.17099492251873016,
            0.30038413405418396,
            0.17396986484527588,
        ],
        "start_peakres": [
            0.11055049300193787,
            0.08913756906986237,
            0.05230756476521492,
            -0.030638378113508224,
        ],
        "start_peakres_nomask": [
            0.11055049300193787,
            -0.09824150055646896,
            -0.07306230068206787,
            -0.04516427963972092,
        ],
        "peakres": [
            0.08842375129461288,
            -0.045682262629270554,
            -0.023588664829730988,
            -0.012185166589915752,
        ],
        "peakres_nomask": [
            -0.10032305121421814,
            -0.09196170419454575,
            -0.05932775139808655,
            -0.04091144725680351,
        ],
        "masksum": [62500, 62500, 62500, 62500],
        "stop_description": "Reached the major cycle limit (nmajor)",
    },
}


if __name__ == "__main__":
    _regenerate_truth_images()
