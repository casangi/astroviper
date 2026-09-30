"""Component test: memory of the cube imaging node task.

The node task images one channel of a simulated field whose image planes are
200 MB each and whose visibilities are 100 MB, while the memory of the process
is watched from start to end of the node task:

* a sampler thread reads the resident set size of the process every few
  milliseconds (psutil), which sees every allocation, Python or C++;
* tracemalloc follows the allocations made through Python, which includes
  every NumPy array (NumPy reports its data buffers to tracemalloc), so a
  copy made by ``np.abs``, ``np.where``, ``astype`` or a pybind11 cast shows
  up there exactly, on every platform;
* every phase of the node task (loading, imaging setup, residual update,
  model update, restore, primary beam correction, statistics, write) is
  wrapped, and the memory a phase adds at its own peak is compared with a
  budget for that phase. One global peak would hide a copy made in a phase
  that peaks below another one: the residual update allocates far more than
  the model update, so a copy of the residual cube in the model update, the
  fault of issue 289, only shows in the model update's own peak.

The budgets are in units of one image plane and were measured with the code
of issue 289, in which the model update and residual update paths were freed
of their cube and plane sized temporaries. The margin of half a plane is what
a copy of a plane, let alone of the cube, or of the visibilities exceeds. A
change that raises a phase above its budget is a copy that has to be
explained, or a budget that has to be raised knowingly.

The run is small in everything but the array sizes: one channel, one point
source, no noise, a few CLEAN iterations, two imaging cycles, restore and
primary beam correction on, the synchronous Dask scheduler so that the node
task runs in this process. Runtime is about half a minute.
"""

from __future__ import annotations

import functools
import importlib
import threading
import time
import tracemalloc

import dask
import numpy as np
import pytest
import xarray as xr

import astroviper.distributed_applications as distributed_applications
import astroviper.node_tasks.imaging as node_tasks_imaging
from astroviper.distributed_applications.imaging import image_cube_single_field
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

ARCSEC = np.pi / (180 * 3600)
MB = 1_000_000

# --- the field --------------------------------------------------------------
LAYOUT = "alma.cycle8.1"  # 43 antennas, 903 baselines
IMAGE_SIZE = [5000, 5000]  # 25e6 pixels x 8 bytes: 200 MB per double precision plane
CELL_SIZE = np.array([-0.05, 0.05]) * ARCSEC
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
# 3700 integrations x 903 baselines x 2 correlations x 16 bytes: 107 MB of visibilities
TIME_PARAMS = {
    "time_start": "2019-10-03T19:00:00.000",
    "time_delta": 1.0,
    "n_samples": 3700,
}
FREQUENCY_PARAMS = {
    "freq_start": 100e9,
    "freq_delta": 0.5e9,
    "n_channels": 1,
    "channel_width": 2e6,
    "spectral_window_name": "Band3",
}
POLARIZATION = ["XX", "YY"]
STOKES = ["I", "Q"]
MIN_PLANE_BYTES = 200 * MB
MIN_VISIBILITY_BYTES = 100 * MB

ITERATION_CONTROL_PARAMS = {
    "max_iter": 50,
    "max_cycles": 2,
    "threshold": 0.0,
    "primary_beam_limit": 0.2,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.5,
    "max_iter_per_cycle": 25,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}

# --- the phases and their budgets --------------------------------------------
# Every entry names a function the node task calls (module, attribute) and
# the memory, in image planes, the phase may add at its own peak above the
# traced memory at its start. Measured with the code of issue 289 (macOS
# arm64, NumPy 2, scipy FFT backend), largest call of each phase, plus a
# margin of half a plane:
#
#   load_processing_set          2.0 + 0.5   visibilities, weights, flags, uvw
#   imaging_setup_single_field  14.4 + 0.5   PSF, dirty image, primary beam and
#                                            mask of two correlations, with the
#                                            padded complex grids and FFTs
#   residual_update              7.8 + 0.5   degridding, gridding and FFTs
#   model_update                 2.5 + 0.5   the model cube of two planes is
#                                            created in the first call; later
#                                            calls add nothing
#   restore_image                5.0 + 0.5   restored image of two planes and
#                                            the FFT convolution
#   correct_sky_by_primary_beam  4.0 + 0.5   corrected image of two planes and
#                                            the temporaries of the division
#   calculate_plane_statistics   3.2 + 0.5   plane statistics temporaries
#   write_result_chunk           3.8 + 0.5   zarr encoding buffers
PHASES = [
    ("xradio.measurement_set.load_processing_set", "load_processing_set", 2.5),
    (
        "astroviper.processing_functions.imaging.residual_update",
        "imaging_setup_single_field",
        14.9,
    ),
    (
        "astroviper.processing_functions.imaging.residual_update",
        "residual_update_cube_single_field",
        8.3,
    ),
    (
        "astroviper.processing_functions.imaging.model_update",
        "model_update_cube_single_field",
        3.0,
    ),
    ("astroviper.processing_functions.imaging.restore", "restore_image", 5.5),
    (
        "astroviper.processing_functions.imaging.correct_sky_by_primary_beam",
        "correct_sky_by_primary_beam",
        4.5,
    ),
    (
        "astroviper.processing_functions.image_analysis.plane_statistics",
        "calculate_plane_statistics",
        3.7,
    ),
    ("astroviper.utils.io", "write_result_chunk_to_disk_using_zarr", 4.3),
]
# The whole node task: the peak of the traced memory (14.8 planes measured,
# in the imaging setup) and of the resident set size above its value at the
# start of the node task (12.7 planes measured; the resident set also holds
# the C++ buffers and depends on the allocator, hence the wider margin).
TRACEMALLOC_BUDGET_PLANES = 14.8 + 0.5
RSS_BUDGET_PLANES = 12.7 + 2.0


class MemoryMonitor:
    """Memory of the process while the node task runs.

    The node task itself is wrapped for the global numbers: the largest
    resident set size the sampler thread saw minus the value at the start
    (``rss_peak_above_start``) and the peak of the traced memory
    (``tracemalloc_peak``). Every phase in ``PHASES`` is wrapped too and
    ``phase_peaks`` holds, per phase, the largest amount of traced memory one
    call added above the traced memory at its start, in bytes.
    """

    def __init__(self, interval=0.005):
        import psutil

        self.process = psutil.Process()
        self.interval = interval
        self.samples = []
        self.rss_start = 0
        self.tracemalloc_peak = 0
        self.phase_peaks = {}
        self.phase_calls = {}
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            self.samples.append(self.process.memory_info().rss)
            time.sleep(self.interval)
        self.samples.append(self.process.memory_info().rss)

    def start(self):
        self.rss_start = self.process.memory_info().rss
        self.samples = [self.rss_start]
        tracemalloc.start()
        tracemalloc.reset_peak()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        _, self.tracemalloc_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    @property
    def rss_peak_above_start(self):
        return max(self.samples) - self.rss_start

    def watch_node_task(self, monkeypatch):
        original = node_tasks_imaging.image_cube_single_field

        @functools.wraps(original)
        def watched(*args, **kwargs):
            self.start()
            try:
                return original(*args, **kwargs)
            finally:
                self.stop()

        # the driver looks the node task up in the package at call time
        monkeypatch.setattr(node_tasks_imaging, "image_cube_single_field", watched)

    def watch_phase(self, monkeypatch, module_name, attribute):
        """Wrap one function the node task imports from ``module_name`` when
        it runs. The peak is measured from a reset at the start of the call,
        so the phases must not nest."""
        module = importlib.import_module(module_name)
        original = getattr(module, attribute)
        peaks = self.phase_peaks
        calls = self.phase_calls

        @functools.wraps(original)
        def watched(*args, **kwargs):
            before, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            try:
                return original(*args, **kwargs)
            finally:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
                peaks[attribute] = max(peaks.get(attribute, 0), peak - before)
                calls[attribute] = calls.get(attribute, 0) + 1

        monkeypatch.setattr(module, attribute, watched)


def simulate(work_dir):
    """One point source at the phase centre; returns the processing set path."""
    antenna_xds = read_telescope_layout(LAYOUT)
    ps_store = f"{work_dir}/memory.ps.zarr"
    distributed_applications.simulation.simulate_processing_set(
        ps_store=ps_store,
        antenna_xds=antenna_xds,
        time_params=TIME_PARAMS,
        frequency_params=FREQUENCY_PARAMS,
        polarization=POLARIZATION,
        sky_components=[
            {
                "kind": "point",
                "flux": np.array([[1.0, 0.0, 0.0, 1.0]]),
                "ra_dec": PHASE_CENTER,
                "name": "source_00",
            }
        ],
        phase_center_ra_dec=PHASE_CENTER[None, :],
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
        n_time_chunks=1,
        n_frequency_chunks=1,
        overwrite=True,
    )
    return ps_store


def visibility_bytes(ps_store):
    from xradio.measurement_set import open_processing_set

    ps_xdt = open_processing_set(ps_store)
    return sum(
        int(np.prod(ms.ds["VISIBILITY"].shape)) * ms.ds["VISIBILITY"].dtype.itemsize
        for ms in ps_xdt.values()
    )


def image(work_dir, ps_store):
    """Image the field the way the point source component test does."""
    from xradio.measurement_set import open_processing_set

    ps_xdt = open_processing_set(ps_store)
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    phase_direction = combined.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=combined.attrs["center_field_name"]
    ).values
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": CELL_SIZE,
        "phase_direction": phase_direction,
        "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
        "polarization_coords": STOKES,
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }
    image_store = f"{work_dir}/memory.img.zarr"
    with dask.config.set(scheduler="synchronous"):
        image_cube_single_field(
            ps_store=ps_store,
            image_store=image_store,
            image_params=image_params,
            imaging_weights_params={
                "weighting": "natural",
                "robust": 0.5,
                "casa_weighting_implementation": True,
            },
            iteration_control_params=ITERATION_CONTROL_PARAMS,
            instrument_polarization_basis="linear",
            gridder="prolate_spheroidal",
            deconvolver="hogbom",
            scan_intents="OBSERVE_TARGET#ON_SOURCE",
            image_data_variables_keep=[
                "sky_model",
                "sky_residual",
                "primary_beam",
                "point_spread_function",
                "beam_fit_params_point_spread_function",
                "mask",
            ],
            processing_set_data_group_name="base",
            single_precision_image=False,
            processing_function_threads=1,
            n_mapping_parallelism={"frequency": 1},
            fft_backend="scipy",
            restore=True,
            primary_beam_correction=True,
            overwrite=True,
        )
    return image_store


def test_node_task_memory(tmp_path, monkeypatch):
    pytest.importorskip("psutil")
    work_dir = str(tmp_path)
    ps_store = simulate(work_dir)
    n_visibility_bytes = visibility_bytes(ps_store)
    assert n_visibility_bytes >= MIN_VISIBILITY_BYTES, n_visibility_bytes

    monitor = MemoryMonitor()
    monitor.watch_node_task(monkeypatch)
    for module_name, attribute, _ in PHASES:
        monitor.watch_phase(monkeypatch, module_name, attribute)
    image_store = image(work_dir, ps_store)

    img = xr.open_zarr(image_store)
    plane_bytes = (
        int(np.prod(img.SKY_RESIDUAL.shape[-2:])) * img.SKY_RESIDUAL.dtype.itemsize
    )
    assert plane_bytes >= MIN_PLANE_BYTES, plane_bytes
    assert img.SKY_RESIDUAL.dtype == np.float64

    lines = [
        f"plane {plane_bytes / MB:.0f} MB, visibilities {n_visibility_bytes / MB:.0f} MB, "
        f"{len(monitor.samples)} samples of the resident set size",
        f"node task: traced peak {monitor.tracemalloc_peak / plane_bytes:.2f} planes "
        f"(budget {TRACEMALLOC_BUDGET_PLANES:.2f}), resident set size peak "
        f"{monitor.rss_peak_above_start / plane_bytes:.2f} planes above the start "
        f"(budget {RSS_BUDGET_PLANES:.2f})",
    ]
    failures = []
    for _module_name, attribute, budget in PHASES:
        calls = monitor.phase_calls.get(attribute, 0)
        added = monitor.phase_peaks.get(attribute, 0) / plane_bytes
        lines.append(
            f"{attribute}: {calls} call(s), adds at most {added:.2f} planes at its peak "
            f"(budget {budget:.2f})"
        )
        if calls == 0:
            failures.append(f"{attribute} was not called")
        elif added > budget:
            failures.append(f"{attribute} adds {added:.2f} planes, budget {budget:.2f}")
    if monitor.tracemalloc_peak / plane_bytes > TRACEMALLOC_BUDGET_PLANES:
        failures.append("node task traced peak above its budget")
    if monitor.rss_peak_above_start / plane_bytes > RSS_BUDGET_PLANES:
        failures.append("node task resident set size peak above its budget")
    report = "\n".join(lines)
    print(report)
    assert not failures, "\n".join(failures) + "\n" + report
