"""Component tests: node-task memory of cube imaging and find_continuum_ultra.

The field has three channels and the distributed application maps it onto
three node tasks of one channel each; image planes are 200 MB each and every
task loads 107 MB of visibilities. Each node task runs once, in this process,
so that each run is a task the process has not run before (its own channel,
visibility selection and task id), as on a worker: run 1 through the
distributed application (graph, synchronous Dask scheduler, graphviper's
per-task wrapper), as in production, with the garbage collector on; the
application's other two tasks are recorded instead of run (they return an
empty result, as a task whose data cannot be read does) and are then run
directly, runs 2 and 3, with the garbage collector off, so that only
reference counting frees memory. The memory of the process is watched
during all runs:

* tracemalloc follows the allocations made through Python, which includes
  every NumPy array (NumPy reports its data buffers to tracemalloc), so a
  copy made by ``np.abs``, ``np.where``, ``astype`` or a pybind11 cast shows
  up there exactly, on every platform;
* every phase of the node task (loading, imaging setup, residual update,
  model update, restore with primary beam correction, statistics, write) is
  wrapped, and the traced memory a phase adds at its own peak above its start
  is compared with a budget for that phase, in every run. One global peak
  would hide a copy made in a phase that peaks below another one: a copy of
  the residual cube in the model update, the fault of issue 289, only shows
  in the model update's own peak;
* the peak of the traced memory over the whole node task (traced memory held
  at the start of a phase plus what the phase adds) is compared with a
  budget. It sees memory one phase keeps alive into the next;
* all memory is released, and nothing ratchets from one node task to the
  next. Every direct run is checked on its own, against absolute budgets, so
  a leak that repeats in every node task fails it however steady it is, and
  since every run is a new task, so does a cache that grows with every task
  (keyed by the channel, the selection or the task id). After the run and its
  result are gone, first by reference counting alone (garbage collector
  still off; a garbage collection inside a direct run, a ``gc.collect()``
  call included, fails the test), then after one ``gc.collect()``:

  - the traced memory left by reference counting alone is within a small
    budget, and that collection finds no object in a reference cycle (the
    lazy tree xarray's ``open_datatree`` drops is the one known exception,
    see ``cyclic_garbage``): memory held in reference cycles fails;
  - the traced memory left after the collection is within a tight budget:
    memory still reachable (a cache, a module-level list) fails, even a
    fraction of a megabyte per node task;
  - on Linux with glibc, the bytes malloc has in use (``mallinfo2``, a
    read-only query) minus the traced memory and minus tracemalloc's own
    tables are within a budget: memory C or C++ code allocates with malloc or
    new and keeps, which tracemalloc does not see;
  - on Linux, the anonymous memory of the process (RssAnon + RssShmem +
    VmSwap) and the part of it malloc does not hold are within budgets:
    memory mapped directly and kept (an anonymous mmap, which neither
    tracemalloc nor malloc counts). The budgets are tight unless transparent
    huge pages are "always" for the process, and coarse with "always".

  The first direct run warms up what the direct call path touches once
  (tracemalloc's tables, the malloc arenas of threads, glibc's adaptive mmap
  threshold), so its untraced memory is checked against looser budgets than
  the second's. Run 1, the whole call of the distributed application, is
  checked for the traced memory it keeps, not counting modules imported for
  the first time (themselves within a budget), and on Linux with glibc,
  after one collection, for the malloc memory it keeps beyond that (a buffer
  a C or C++ library allocates on first use and keeps) and for the anonymous
  memory it keeps outside malloc (a buffer mapped directly);
* the traced peaks and the memory at return of the two direct runs agree;
* a sampler thread reads the resident set size, the anonymous memory, the
  traced memory and, on Linux with glibc, the malloc bytes in use every few
  milliseconds. Per phase, two measures of untraced memory are checked on
  the warmed run: the gross untraced malloc memory (the peak of malloc bytes
  in use minus traced memory above its value at the start of the phase; it
  does not depend on which pages are resident), and the net untraced memory
  (anonymous memory added at its peak minus traced memory added at its
  peak), which also sees memory mapped directly but is offset by traced
  memory that is not resident at the peak. On macOS the memory compressor
  takes idle pages out of the resident set under memory pressure, so there
  the resident set size is only reported;
* the minor page faults and the wall time of every run and phase are
  reported, not checked.

No allocator setting is made, no memory is trimmed and astroviper calls no
``gc.collect()``: the checks hold with and without graphviper's per-task
memory management (``GRAPHVIPER_TASK_MEMORY_MANAGEMENT``) and with
transparent huge pages "always", "madvise" or disabled. The test itself
collects garbage only after the reference counting measurement of a run.

The budgets are in units of one image plane (in MB for the release checks)
and were measured with the code of issue 289 on Linux x86-64 (Docker, glibc
2.41) with the versions CI installs, Python 3.11, 3.12 and 3.13; each
comment gives the measured range and the smallest leak the check catches.

The run is small in everything but the array sizes: one channel per task,
one point source, no noise, a few CLEAN iterations, two imaging cycles,
restore and primary beam correction on, the synchronous Dask scheduler so
that the node tasks run in this process. Runtime (three node tasks) is
about a minute and a quarter natively on Apple silicon and two and a half
minutes under x86-64 emulation.

find_continuum_ultra's tasks are checked at 2048 x 2048 against the memory
its pass planner gives them (``ultra_task_memory_bytes``): a frequency task
of the first pass without a first-round store, and the imager's first round
of one chunk, each in a new process, as on a worker. The resident set peak
of the process stays within 1.3 times the plan (on macOS the memory
compressor can only lower it), and the traced peak of the frequency task
does not grow from 16 to 64 channels per task. As the fixed baseline of the
plan dominates at that size, the bytes per pixel of a frequency task are
also checked, from its peaks at 512 x 512 and 1024 x 1024, against the
plan's, on the harness's sharded layout, in float64 and on the imager's
default layout of whole-plane chunks. The measured values were taken on
Apple silicon (macOS, Python 3.13).
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import functools
import gc
import importlib
import math
import os
import resource
import sys
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
from astroviper.utils.data_tree import release_data_tree
from astroviper.utils.telescope_layout import read_telescope_layout
from tests.unit.node_tasks.imaging.cycle_test_utils import cyclic_garbage

ARCSEC = np.pi / (180 * 3600)
MB = 1_000_000

# --- the field --------------------------------------------------------------
LAYOUT = "alma.cycle8.1"  # 43 antennas, 903 baselines
IMAGE_SIZE = [5000, 5000]  # 25e6 pixels x 8 bytes: 200 MB per double precision plane
CELL_SIZE = np.array([-0.05, 0.05]) * ARCSEC
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
# 3700 integrations x 903 baselines x 2 correlations x 16 bytes: 107 MB of
# visibilities per channel
TIME_PARAMS = {
    "time_start": "2019-10-03T19:00:00.000",
    "time_delta": 1.0,
    "n_samples": 3700,
}
# Node tasks run directly after run 1. Every run is a task of its own, so the
# field has one channel per run and the distributed application maps it onto
# one node task per channel.
N_DIRECT_RUNS = 2
N_TASKS = 1 + N_DIRECT_RUNS
FREQUENCY_PARAMS = {
    "freq_start": 100e9,
    "freq_delta": 0.5e9,
    "n_channels": N_TASKS,
    "channel_width": 2e6,
    "spectral_window_name": "Band3",
}
POLARIZATION = ["XX", "YY"]
STOKES = ["I", "Q"]
MIN_PLANE_BYTES = 200 * MB
MIN_VISIBILITY_BYTES = 100 * MB

CYCLE_CONTROL_PARAMS = {
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
# traced memory at its start. Measured on Linux x86-64 with the versions CI
# installs (Python 3.11, 3.12 and 3.13; NumPy 2, scipy FFT backend; zarr
# 3.1.6 with numcodecs 0.16.5, zarr 3.4.0 with numcodecs 0.17.0), largest
# call of each phase, plus a margin of half a plane, which a copy of a plane,
# let alone of the cube or of the visibilities, exceeds; macOS arm64 (Python
# 3.13, the same zarr and numcodecs) measured the same. The peaks do not
# depend on whether NumPy reuses the temporaries of chained expressions (it
# cannot on the GitHub Linux runner builds of Python 3.12 and 3.13): the steps
# at the peaks are done in place.
#
#   load_processing_set          2.0 + 0.5   visibilities, weights, flags, uvw
#   imaging_setup_single_field  10.0 + 0.5   PSF, primary beam and imaging
#                                            weights of two correlations, with
#                                            the padded complex grid and FFT of
#                                            the PSF; peaks in the main lobe
#                                            extraction of the PSF fit
#   residual_update              7.8 + 0.5   degridding, gridding and FFTs
#   model_update                 2.5 + 0.5   the model cube of two planes is
#                                            created in the first call; later
#                                            calls add nothing
#   restore_image                7.0 + 0.5   restored and primary beam
#                                            corrected images of two planes
#                                            each and the FFT convolution
#   calculate_plane_statistics   3.2 + 0.5   plane statistics temporaries
#   write_result_chunk           2.0 + 0.5   the Blosc output buffer of one
#                                            variable (its two planes are one
#                                            on-disk chunk), allocated at the
#                                            uncompressed size and shrunk in
#                                            place; a copy of the cube exceeds
#                                            it. numcodecs before 0.16 copies
#                                            the compressed bytes out of that
#                                            buffer instead (dest[:cbytes]),
#                                            which adds the compressed size:
#                                            3.8 planes here; CI installs
#                                            numcodecs 0.16.5 and 0.17.0
PHASES = [
    # xradio's loader, imported by the node task at every call, so the module
    # attribute wrapped here is what it calls
    (
        "xradio.measurement_set.load_processing_set",
        "load_processing_set",
        2.5,
    ),
    (
        "astroviper.processing_functions.imaging.residual_update",
        "imaging_setup_single_field",
        10.5,
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
    ("astroviper.processing_functions.imaging.restore", "restore_image", 7.5),
    (
        "astroviper.processing_functions.image_analysis.plane_statistics",
        "calculate_plane_statistics",
        3.7,
    ),
    ("astroviper.utils.io", "write_result_chunk_to_disk_using_zarr", 2.5),
]
# The whole node task: the peak of the traced memory above its value at the
# start of the node task, the largest of (traced memory held at the start of
# a phase + what the phase adds). Measured 17.83 to 17.84 planes, in the
# restore (10.83 held + 7.00); the third residual update (17.47), the
# statistics (12.26 + 3.22) and the write (12.26 + 2.00) stay below it.
TRACEMALLOC_BUDGET_PLANES = 18.0
# --- release budgets ---------------------------------------------------------
# Run 1, the whole call of the distributed application, garbage collector on
# as in production: traced memory kept, without the modules it imports for
# the first time (see ImportMemory). Measured 1.2 to 1.5 MB (first-use state:
# per-call-site interpreter state, pandas index and dask graph caches, zarr
# metadata, the driver's pending reference cycles, the recorded arguments of
# the two other tasks). The budget, 6 MB, is below the smallest array of the
# node task (the flags, 6.7 MB): 4.6 MB or more kept by the first node task
# of a process fails it, so a plane, the visibilities or the imaging weights
# do. A cache that grows with every task is caught in the direct runs.
DRIVER_RELEASE_BUDGET_PLANES = 0.03
# Run 1: traced memory allocated by the modules imported for the first time
# (left out above). Measured 2.2 to 2.3 MB when the test runs alone and 0.4
# to 0.8 MB in the order of the full test suite. A module-level table of 8 MB
# a lazily imported module allocates fails it.
FIRST_IMPORTS_BUDGET_MB = 10.0
# Run 1 on Linux with glibc, after one collection: malloc bytes in use kept by
# the call, minus the traced memory and tracemalloc's own memory it keeps,
# each without what first imports add: memory a C or C++ library allocates
# with malloc or new on first use and keeps (the pocketfft plan cache, zarr
# and Blosc state). Measured 2.2 to 3.0 MB when the test runs alone and 3.1
# MB in the order of the full test suite (it is lowered by about 1 MB of
# small Python objects first imports allocate in Python's own arenas, which
# malloc does not count). A buffer kept once per process fails it from 1.9 to
# 2.8 MB, depending on the configuration.
FIRST_TASK_NATIVE_BUDGET_MB = 5.0
# Run 1 on Linux with glibc, after one collection: anonymous memory kept
# outside malloc (memory mapped directly: the deeper parts of thread stacks
# the first task touches, Python's own arenas of the objects first imports
# create, a buffer a library maps on first use). Measured 4.6 to 6.5 MB; a
# directly mapped buffer of 14 MB or more kept once per process fails it.
# With transparent huge pages "always", a thread stack touched anywhere in a
# 2 MB region gets a whole huge page (about 50 threads here): measured 96 to
# 116 MB (a buffer of 44 MB or more fails it), and 162 to 167 MB without
# graphviper's per-task memory management, of which the next point explains
# 88 MB. Where the program break is emulated (x86-64 emulation on an Arm
# host), pages the main malloc heap gives back by lowering the break stay
# resident, which a Linux kernel does not do; without graphviper's memory
# management the heap holds large arrays and gives back up to 88 MB in run 1
# (measured 54 to 56 MB outside malloc with transparent huge pages
# disabled), so there the budget is raised by the distance from the highest
# program break of run 1's node task to the final one.
FIRST_TASK_OUTSIDE_MALLOC_BUDGET_MB = 20.0
THP_ALWAYS_FIRST_TASK_OUTSIDE_MALLOC_BUDGET_MB = 160.0
# Every direct run, garbage collector off: traced memory left by reference
# counting alone. Measured 0.40 MB: 0.19 MB of CPython's small object free
# lists (floats, tuples, lists, dicts), which refill during the run after the
# collection that ended the run before emptied them, and 0.21 MB in the lazy
# tree xarray's open_datatree drops when xradio's load_processing_set opens
# the chunk (see cyclic_garbage; 0.19 MB and nothing in reference cycles with
# a loader that opens the measurement set's group alone). A leak of 0.2 MB per
# node task, reachable or in reference cycles, fails it (the two checks below
# are finer: any object in a cycle outside that tree, and a reachable leak of
# 0.1 MB).
REFCOUNT_RELEASE_BUDGET_MB = 0.6
# Every direct run: objects the collection after the reference counting
# measurement finds in reference cycles, not counting the lazy tree xarray's
# open_datatree drops (see cyclic_garbage). Measured 0; any other object a
# node task leaves in a reference cycle fails it, whatever its size.
CYCLIC_GARBAGE_BUDGET_OBJECTS = 0
# Every direct run, after one gc.collect(): traced memory left, i.e. memory
# still reachable. Measured -0.007 to +0.016 MB. A reachable leak of 0.11 MB
# per node task fails it.
POST_GC_RELEASE_BUDGET_MB = 0.1
# Every direct run, Linux, after the collection, in MB, for the first direct
# run (warm-up) and the second (warmed); None: reported only.
#
# native: on Linux with glibc, malloc bytes in use left minus traced memory
#   left minus tracemalloc's own memory left: memory C or C++ code allocates
#   with malloc or new and keeps. Measured +0.69 to +0.77 MB in the first
#   direct run (one-time state of the direct call path; an allocation tracer
#   found 16 and 32 byte blocks, the size of tracemalloc's own records of
#   allocation sites seen for the first time, which
#   tracemalloc.get_tracemalloc_memory() does not count) and +0.003 to +0.011
#   MB in the second. A malloc leak of 0.25 MB per node task fails the warmed
#   run, of 1.3 MB the first.
# anonymous: anonymous memory left (RssAnon + RssShmem + VmSwap), which sees
#   memory mapped directly and kept (neither tracemalloc nor malloc counts
#   it), but also the growth of glibc's heaps: without graphviper's per-task
#   memory management, glibc's adaptive mmap threshold moves the large arrays
#   onto the heaps in the first direct run (+50 MB, reported only), and the
#   heaps grow by up to 6 MB of free memory in the warmed run, of which up to
#   3.5 MB is resident. Measured -0.3 to +3.5 MB in the warmed run.
# outside malloc: on Linux with glibc, the anonymous memory minus the bytes
#   malloc holds (in use or free); heap growth does not raise it, heap growth
#   that is not resident lowers it. Measured -3.4 to +1.6 MB in the first
#   direct run and -3.1 to +0.3 MB in the second.
#
# Together the anonymous checks catch memory mapped directly and kept from
# 1.6 to 4.6 MB per node task (budget minus the lowest value measured in the
# configuration). With transparent huge pages "always" for this process
# (Docker's default here; GitHub's runners use "madvise") khugepaged may at
# any time fill a partly used 2 MB region of a heap or a thread stack with a
# huge page, which made a warmed run gain up to 7.2 MB, so there every
# anonymous budget is at least THP_ALWAYS_ANON_BUDGET_MB: a steady directly
# mapped leak of 20 MB per node task still fails.
RELEASE_BUDGETS_MB = {
    "warm-up": {"native": 2.0, "anonymous": None, "outside malloc": 5.0},
    "warmed": {"native": 0.25, "anonymous": 5.0, "outside malloc": 1.5},
}
THP_ALWAYS_ANON_BUDGET_MB = 20.0
# No ratchet between the two direct runs: traced peak and traced memory at
# return, either direction. Measured at most 0.0004 planes (0.0000 with a
# task of its own in every run).
RATCHET_BUDGET_PLANES = 0.01

# --- untraced memory per phase ------------------------------------------------
# Warmed run, Linux with glibc: gross untraced malloc memory, the peak of
# (malloc bytes in use - traced memory) above its value at the start of the
# phase; it does not depend on which pages are resident. Measured at most
# 0.011 planes (2.2 MB: zarr and Blosc buffers in the load and the write; 0.002
# in the restore since its inverse FFT no longer makes scipy's hidden copy of
# the spectrum). An untraced malloc copy of 0.04 planes (8 MB) in any phase
# fails it.
GROSS_UNTRACED_BUDGET_PLANES = 0.05
# Warmed run, Linux: net untraced memory, the anonymous memory a phase adds
# at its peak minus the traced memory it adds at its peak. It sees memory
# mapped directly, which the gross measure does not, but traced memory that
# is not resident at the traced peak offsets it. Measured offsets (the most
# negative value of the warmed run over all configurations: Python 3.11,
# 3.12 and 3.13, with and without graphviper's per-task memory management,
# transparent huge pages always, madvise and disabled; no new extreme with a
# task of its own in every run), in planes, and the smallest untraced mapped
# copy that fails (budget minus offset):
#
#   load_processing_set          -0.40   0.65   zarr's read buffers
#   imaging_setup_single_field   -0.25   0.50   (only without graphviper's
#                                               memory management)
#   residual_update              -0.08   0.33
#   model_update                 -0.02   0.27
#   restore_image                -2.00   2.25   the restored and corrected
#                                               cubes (two planes each) are
#                                               allocated first and written
#                                               plane by plane after the clean
#                                               beam's FFT, where the traced
#                                               peak is
#   calculate_plane_statistics   -0.22   0.47   (only without graphviper's
#                                               memory management)
#   write_result_chunk           -0.36   0.61   the Blosc output buffer,
#                                               allocated at the uncompressed
#                                               size and shrunk
#
# An untraced copy made with malloc or new is caught by the gross measure in
# every phase from 0.04 planes.
UNTRACED_BUDGET_PLANES = 0.25
CHECK_RESIDENT_SET = sys.platform.startswith("linux")
# malloc bytes in use and traced memory are read again until neither changed
# by more than this between two reads (no allocation or free in between).
QUIESCENT_BYTES = 1_000_000


def _huge_pages_always():
    """True when transparent huge pages are "always" for this process."""
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as mode:
            always = "[always]" in mode.read()
        with open("/proc/self/status") as status:
            disabled = any(
                line.split()[1:2] == ["0"]
                for line in status
                if line.startswith("THP_enabled:")
            )
    except OSError:
        return False
    return always and not disabled


THP_ALWAYS = CHECK_RESIDENT_SET and _huge_pages_always()


def _malloc_in_use_reader():
    """glibc's bytes in use by malloc (``mallinfo2``: ``uordblks + hblkhd``),
    as a function, or None where there is no glibc 2.33 or later.

    A read-only query; no allocator setting is changed. It is called through
    ``ctypes.PyDLL``, which keeps the GIL during the call, so no Python
    thread allocates between it and a tracemalloc read made next to it.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        mallinfo2 = ctypes.PyDLL(None).mallinfo2
    except (OSError, AttributeError):
        return None

    class MallInfo2(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_size_t)
            for name in (
                "arena",
                "ordblks",
                "smblks",
                "hblks",
                "hblkhd",
                "usmblks",
                "fsmblks",
                "uordblks",
                "fordblks",
                "keepcost",
            )
        ]

    mallinfo2.restype = MallInfo2
    mallinfo2.argtypes = []

    def in_use():
        info = mallinfo2()
        return info.uordblks + info.hblkhd

    def held():
        """Bytes malloc holds from the system, in use or free: its heaps
        (``arena``) and its separately mapped blocks (``hblkhd``)."""
        info = mallinfo2()
        return info.arena + info.hblkhd

    in_use.held = held
    return in_use


MALLOC_IN_USE = _malloc_in_use_reader()


def _program_break_reader():
    """The program break (``sbrk(0)``, a read-only query: it moves nothing),
    as a function, or None outside Linux with glibc."""
    if MALLOC_IN_USE is None:
        return None
    try:
        sbrk = ctypes.CDLL(None).sbrk
    except AttributeError:
        return None
    sbrk.restype = ctypes.c_void_p
    sbrk.argtypes = [ctypes.c_long]
    return lambda: sbrk(0) or 0


PROGRAM_BREAK = _program_break_reader()


def _break_shrink_keeps_pages():
    """True when the process has no "[heap]" mapping although malloc has used
    the program break: the break is then emulated (x86-64 emulation on an Arm
    host, as Docker on Apple silicon does), and pages the main malloc heap
    gives back by lowering the break stay resident and counted, whereas a
    Linux kernel frees them. Measured there: lowering the break by 200 MB of
    written pages left the resident set unchanged."""
    if PROGRAM_BREAK is None:
        return False
    try:
        with open("/proc/self/maps") as maps:
            return not any(line.rstrip().endswith("[heap]") for line in maps)
    except OSError:
        return False


BREAK_SHRINK_KEEPS_PAGES = _break_shrink_keeps_pages()


def traced_and_in_use(tries=20):
    """(traced memory, malloc bytes in use), read with no allocation or free
    in between: a thread switch between the two reads would otherwise count
    an array allocated or freed meanwhile as untraced. The malloc bytes are
    nan where they cannot be read (no glibc), or when ``tries`` readings in a
    row were not quiet (then the release checks fail rather than pass); the
    reader sleeps a millisecond between tries, so other threads can finish."""
    traced = tracemalloc.get_traced_memory()[0]
    if MALLOC_IN_USE is None:
        return traced, math.nan
    for attempt in range(tries):
        if attempt:
            time.sleep(0.001)
        in_use = MALLOC_IN_USE()
        traced = tracemalloc.get_traced_memory()[0]
        if (
            abs(MALLOC_IN_USE() - in_use) < QUIESCENT_BYTES
            and abs(tracemalloc.get_traced_memory()[0] - traced) < QUIESCENT_BYTES
        ):
            return traced, in_use
    return traced, math.nan


def collect_garbage():
    """One full collection, made after a run's reference counting
    measurement. Returns the number of objects it found in reference cycles
    and :func:`cyclic_garbage` of them; they are freed before it returns."""
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        garbage = gc.garbage[:]
        gc.garbage.clear()
    finally:
        gc.set_debug(0)
    found = len(garbage), cyclic_garbage(garbage)
    del garbage
    gc.collect()  # frees what the first collection kept
    return found


class CollectorOff:
    """Keeps the garbage collector off, and sees every collection made while
    a run is watched.

    ``gc.enable`` does nothing while active, and calls of it are counted.
    dask's ``disable_gc`` decorator (``dask.array.slicing``) records
    ``gc.isenabled()`` when dask is imported and turns the collector back on
    after every dask slice, which would end reference counting alone in the
    middle of a run. Within :meth:`watching`, every garbage collection
    (``gc.collect()`` called from Python or C code, or an automatic one) is
    recorded, with the code that started it, through ``gc.callbacks``: memory
    such a collection frees would otherwise pass as released by reference
    counting."""

    def __init__(self):
        self.enable_calls = 0
        self.collections = collections.defaultdict(list)
        self._enable = None
        self._label = None

    def _ignore(self):
        self.enable_calls += 1

    def _on_collection(self, phase, info):
        if phase != "start" or self._label is None:
            return
        frame = sys._getframe(1)
        callers = []
        while frame is not None and len(callers) < 4:
            code = frame.f_code
            callers.append(
                f"{code.co_filename.rsplit('/', 3)[-1]}:{frame.f_lineno} {code.co_name}"
            )
            frame = frame.f_back
        self.collections[self._label].append(
            f"generation {info['generation']} from {' <- '.join(callers)}"
        )

    @contextlib.contextmanager
    def watching(self, label):
        """Record the collections made inside the block under ``label``."""
        self._label = label
        try:
            yield
        finally:
            self._label = None

    def __enter__(self):
        gc.disable()
        self._enable = gc.enable
        gc.enable = self._ignore
        gc.callbacks.append(self._on_collection)
        return self

    def __exit__(self, *exc_info):
        gc.callbacks.remove(self._on_collection)
        gc.enable = self._enable
        return False


def deferred_result(task_id):
    """What the driver gets from a node task that is recorded instead of run:
    the result of a task whose data cannot be read (one timing row, an empty
    imaging dict, no statistics), which the reduce is built to take."""
    import pandas as pd

    from astroviper.processing_functions.imaging.utils.imaging_dict import (
        ImagingDict,
    )

    return {
        "timing_node_tasks": pd.DataFrame({"task_id": [task_id]}),
        "deconvolution": ImagingDict(),
        "image_statistics": {},
    }


class MemoryMonitor:
    """Memory of the process while the node task runs.

    :meth:`run` brackets one call of the node task and appends a record to
    ``runs``; :meth:`left_behind` adds what the run left behind once its
    result is gone (reference counting alone), :meth:`after_collection` what
    is left after a ``gc.collect()``. Every phase in ``PHASES`` is wrapped
    (:meth:`watch_phase`). The sampler thread writes into a buffer allocated
    before tracing starts, so sampling allocates nothing traced that lives.
    All memory values are in bytes, relative to the start of the run.
    """

    CAPACITY = 100_000  # samples per run (several minutes at 5 ms)

    def __init__(self, interval=0.005):
        import psutil

        self.process = psutil.Process()
        self.interval = interval
        self.runs = []
        self.current = None
        # resident set size, anonymous memory, traced memory, malloc in use,
        # program break
        self.samples = np.zeros((self.CAPACITY, 5))
        self.n_samples = 0
        self._stop = threading.Event()
        self._thread = None
        self._started_tracing = False

    def resident(self):
        """(resident set size, anonymous memory). On Linux the second is
        RssAnon + RssShmem + VmSwap: every page of the process that is not
        file backed (an anonymous ``mmap.mmap(-1, n)`` is shared memory),
        resident or swapped out. File backed pages, which the kernel may
        drop and read back at any time, are left out. Elsewhere it is the
        resident set size."""
        if not CHECK_RESIDENT_SET:
            rss = self.process.memory_info().rss
            return rss, rss
        values = {}
        with open("/proc/self/status", "rb") as status:
            for line in status:
                if line.startswith((b"VmRSS:", b"RssAnon:", b"RssShmem:", b"VmSwap:")):
                    name, value = line.split()[:2]
                    values[name] = int(value) * 1024
        anonymous = sum(
            values.get(name, 0) for name in (b"RssAnon:", b"RssShmem:", b"VmSwap:")
        )
        return values[b"VmRSS:"], anonymous

    def state(self):
        rss, anon = self.resident()
        traced, in_use = traced_and_in_use()
        held = MALLOC_IN_USE.held() if MALLOC_IN_USE is not None else math.nan
        return {
            "traced": traced,
            # tracemalloc's own memory (its tables, allocated with malloc)
            "tracemalloc": tracemalloc.get_tracemalloc_memory(),
            "in_use": in_use,
            "rss": rss,
            "anon": anon,
            # anonymous memory malloc does not hold (in use or free): memory
            # mapped directly
            "anon_outside_malloc": anon - held,
            "break": PROGRAM_BREAK() if PROGRAM_BREAK is not None else math.nan,
        }

    def _sample(self):
        while not self._stop.is_set():
            if self.n_samples < self.CAPACITY:
                rss, anon = self.resident()
                traced, in_use = traced_and_in_use(tries=2)
                brk = PROGRAM_BREAK() if PROGRAM_BREAK is not None else math.nan
                self.samples[self.n_samples] = rss, anon, traced, in_use, brk
                self.n_samples += 1
            time.sleep(self.interval)

    def start_tracing(self):
        """Start tracemalloc unless it is already tracing (then it is left
        running, and only the differences it reports are used)."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracing = True

    def stop_tracing(self):
        """Stop tracemalloc if :meth:`start_tracing` started it."""
        if self._started_tracing:
            tracemalloc.stop()
            self._started_tracing = False

    def fold_peak(self):
        """Fold the traced peak since the last ``reset_peak`` into the run's
        peak and return it (absolute)."""
        _, peak = tracemalloc.get_traced_memory()
        run = self.current
        if run is not None:
            run["traced_peak"] = max(run["traced_peak"], peak - run["start"]["traced"])
        return peak

    def run(self, label, node_task, *args, **kwargs):
        """Call ``node_task`` and record the run; returns its result."""
        start = self.state()
        self.current = run = {
            "label": label,
            "start": start,
            "traced_peak": 0,
            "phases": {},
        }
        self.n_samples = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        tracemalloc.reset_peak()
        self._thread.start()
        t0 = time.perf_counter()
        try:
            return node_task(*args, **kwargs)
        finally:
            run["wall_seconds"] = time.perf_counter() - t0
            run["minor_faults"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults
            )
            self._stop.set()
            self._thread.join()
            self._thread = None
            self.fold_peak()
            at_return = self.state()
            run["traced_at_return"] = at_return["traced"] - start["traced"]
            run["rss_at_return"] = at_return["rss"] - start["rss"]
            samples = self.samples[: self.n_samples, 0]
            run["rss_peak"] = (
                float(samples.max()) if len(samples) else at_return["rss"]
            ) - start["rss"]
            # highest program break (absolute)
            run["break_peak"] = float(
                np.fmax.reduce(
                    self.samples[: self.n_samples, 4],
                    initial=max(start["break"], at_return["break"]),
                )
            )
            self.runs.append(run)
            self.current = None

    def _left(self, run, key):
        now = self.state()
        run[key] = {name: now[name] - run["start"][name] for name in now}

    def left_behind(self, run):
        """What ``run`` left behind once its result is gone, before any
        garbage collection."""
        self._left(run, "left")

    def after_collection(self, run):
        """Collect garbage (:func:`collect_garbage`), then record what ``run``
        left behind; call it after :meth:`left_behind`."""
        run["unreachable"], run["cyclic_garbage"] = collect_garbage()
        self._left(run, "left_after_gc")

    def watch_node_task(self, monkeypatch, captured):
        """Record every call of the node task the driver makes: its keyword
        arguments go to ``captured``. The first call runs the node task, as
        run 1; later calls are deferred (:func:`deferred_result`), so that
        they can be run directly, each as a task the process has not run."""
        original = node_tasks_imaging.image_cube_single_field

        @functools.wraps(original)
        def watched(*args, **kwargs):
            captured.append(kwargs)
            if len(captured) > 1:
                return deferred_result(kwargs.get("task_id"))
            return self.run("run 1", original, *args, **kwargs)

        # the driver looks the node task up in the package at call time
        monkeypatch.setattr(node_tasks_imaging, "image_cube_single_field", watched)
        return original

    def watch_phase(self, monkeypatch, module_name, attribute):
        """Wrap one function the node task looks up in ``module_name`` when it
        runs. The peak is measured from a reset at the start of the call, so
        the phases must not nest."""
        module = importlib.import_module(module_name)
        original = getattr(module, attribute)
        monitor = self

        @functools.wraps(original)
        def watched(*args, **kwargs):
            run = monitor.current
            monitor.fold_peak()
            rss_before, anon_before = monitor.resident()
            before, in_use_before = traced_and_in_use()
            tracemalloc.reset_peak()
            first_sample = monitor.n_samples
            faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                seconds = time.perf_counter() - start
                faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults
                peak = monitor.fold_peak()
                tracemalloc.reset_peak()
                rss_end, anon_end = monitor.resident()
                traced_end, in_use_end = traced_and_in_use()
                window = monitor.samples[first_sample : monitor.n_samples]
                rss_peak = max(float(window[:, 0].max(initial=0)), rss_end)
                anon_peak = max(float(window[:, 1].max(initial=0)), anon_end)
                # untraced malloc memory: a value counts only if it holds in two
                # consecutive samples, since NumPy 2.4 reports a calloc'd array
                # to tracemalloc only after taking the GIL back, which the
                # sampler may hold in between; nan where the malloc bytes in use
                # could not be read
                untraced_malloc = window[:, 3] - window[:, 2]
                untraced_malloc = float(
                    np.fmax.reduce(
                        np.minimum(untraced_malloc[:-1], untraced_malloc[1:]),
                        initial=-np.inf,
                    )
                )
                untraced_malloc = max(untraced_malloc, in_use_end - traced_end)
                if run is not None:
                    held = before - run["start"]["traced"]
                    added = peak - before
                    phase = run["phases"].setdefault(
                        attribute, {"calls": 0, "minor_faults": 0, "wall_seconds": 0}
                    )
                    phase["calls"] += 1
                    phase["minor_faults"] += faults
                    phase["wall_seconds"] += seconds
                    if phase["calls"] == 1 or held + added > phase["peak"]:
                        phase["peak"], phase["held"] = held + added, held
                    for key, value in (
                        ("added", added),
                        ("rss_added", rss_peak - rss_before),
                        ("untraced", anon_peak - anon_before - added),
                        (
                            "gross_untraced",
                            untraced_malloc - (in_use_before - before),
                        ),
                    ):
                        if math.isnan(value):
                            continue
                        phase[key] = max(phase.get(key, value), value)

        monkeypatch.setattr(module, attribute, watched)


class ImportMemory:
    """Memory allocated while modules are imported for the first time.

    The node task and the distributed application import some modules on
    their first call; those modules (code objects, functions, classes,
    module-level tables, the static state of extension modules) stay in
    memory, once per process. That is not memory the node task keeps, so it
    is taken out of what run 1 leaves behind. Every first import of a module
    goes through importlib's ``_find_and_load`` (for the import statement and
    ``importlib.import_module`` alike); :meth:`watch` wraps it and sums the
    traced memory, tracemalloc's own memory and, on Linux with glibc, the
    malloc bytes in use each outermost import adds. The garbage collector is
    held off while an import runs, so that a collection inside the window
    cannot free unrelated garbage and offset the sums.
    """

    def __init__(self):
        self.added = 0
        self.in_use_added = 0
        self.tracemalloc_added = 0
        self._local = threading.local()

    def watch(self, monkeypatch):
        import importlib._bootstrap as bootstrap

        original = bootstrap._find_and_load
        watcher = self

        @functools.wraps(original)
        def find_and_load(*args, **kwargs):
            depth = getattr(watcher._local, "depth", 0)
            watcher._local.depth = depth + 1
            if depth == 0:
                gc_was_enabled = gc.isenabled()
                gc.disable()
                before, in_use_before = traced_and_in_use()
                own_before = tracemalloc.get_tracemalloc_memory()
            try:
                return original(*args, **kwargs)
            finally:
                watcher._local.depth = depth
                if depth == 0:
                    after, in_use_after = traced_and_in_use()
                    watcher.added += after - before
                    watcher.tracemalloc_added += (
                        tracemalloc.get_tracemalloc_memory() - own_before
                    )
                    if not math.isnan(in_use_after - in_use_before):
                        watcher.in_use_added += in_use_after - in_use_before
                    if gc_was_enabled:
                        gc.enable()

        monkeypatch.setattr(bootstrap, "_find_and_load", find_and_load)


def numcodecs_hint():
    """Why the write may exceed its budget with an old numcodecs, or ''."""
    import numcodecs
    from packaging.version import Version

    if Version(numcodecs.__version__) >= Version("0.16"):
        return ""
    return (
        f" (numcodecs {numcodecs.__version__} copies the compressed bytes of every "
        "chunk out of the Blosc output buffer; 0.16 shrinks the buffer in place)"
    )


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
        # one channel per on-disk chunk, so that a task reads its channel only
        n_frequency_chunks=N_TASKS,
        overwrite=True,
    )
    return ps_store


def visibility_bytes(ps_store):
    """Bytes of the visibilities of one channel; read before any
    measurement."""
    from xradio.measurement_set import open_processing_set

    ps_xdt = open_processing_set(ps_store)
    n_bytes = sum(
        int(np.prod(ms.ds["VISIBILITY"].shape))
        * ms.ds["VISIBILITY"].dtype.itemsize
        // ms.ds.sizes["frequency"]
        for ms in ps_xdt.values()
    )
    release_data_tree(ps_xdt)
    return n_bytes


def image_params_of(ps_store):
    """Image parameters of the field; read before any measurement."""
    from xradio.measurement_set import open_processing_set

    ps_xdt = open_processing_set(ps_store)
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    phase_direction = combined.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=combined.attrs["center_field_name"]
    ).values.copy()
    frequency_coords = ps_xdt.xr_ps.get_freq_axis().values.copy()
    del combined
    release_data_tree(ps_xdt)
    return {
        "image_size": IMAGE_SIZE,
        "cell_size": CELL_SIZE,
        "phase_direction": phase_direction,
        "frequency_coords": frequency_coords,
        "polarization_coords": STOKES,
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }


def image(work_dir, ps_store, image_params):
    """Image the field the way the point source component test does, one
    node task per channel."""
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
            cycle_control_params=CYCLE_CONTROL_PARAMS,
            instrument_polarization_basis="linear",
            gridder="anti_alias",
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
            n_mapping_parallelism={"frequency": N_TASKS},
            fft_backend="scipy",
            restore=True,
            restoration_output="both",
            overwrite=True,
        )
    return image_store


def test_node_task_memory(tmp_path, monkeypatch):
    pytest.importorskip("psutil")
    work_dir = str(tmp_path)
    ps_store = simulate(work_dir)
    n_visibility_bytes = visibility_bytes(ps_store)
    assert n_visibility_bytes >= MIN_VISIBILITY_BYTES, n_visibility_bytes
    image_params = image_params_of(ps_store)

    monitor = MemoryMonitor()
    captured = []
    node_task = monitor.watch_node_task(monkeypatch, captured)
    for module_name, attribute, _ in PHASES:
        monitor.watch_phase(monkeypatch, module_name, attribute)
    imports = ImportMemory()
    gc_was_enabled = gc.isenabled()
    collector = CollectorOff()
    imaged = {}
    # Every object alive now, garbage of the setup (the simulation) and of
    # earlier tests included, is set aside: no collection inside run 1's
    # window frees it. Memory allocated before tracing starts is not traced,
    # so freeing it there would offset the malloc memory run 1 keeps. Nothing
    # is collected here.
    gc.freeze()
    monitor.start_tracing()
    try:
        # run 1: through the distributed application, garbage collector on;
        # the application's other node tasks are recorded, not run
        imports.watch(monkeypatch)
        driver_start = monitor.state()
        image_store = image(work_dir, ps_store, image_params)
        driver_left = monitor.state()
        assert len(captured) == N_TASKS, (
            f"{len(captured)} node tasks, expected {N_TASKS}"
        )
        monitor.left_behind(monitor.runs[0])
        # from here on the garbage collector only runs where it is called,
        # after the reference counting measurement of a run
        with collector:
            monitor.after_collection(monitor.runs[0])
            driver_after_gc = monitor.state()
            # runs 2 and 3: the recorded node tasks, called directly
            for index, kwargs in enumerate(captured[1:]):
                label = f"run {index + 2}"
                with collector.watching(label):
                    result = monitor.run(label, node_task, **kwargs)
                imaged[label] = bool(result["deconvolution"].data)
                del result
                monitor.left_behind(monitor.runs[-1])
                monitor.after_collection(monitor.runs[-1])
    finally:
        if gc_was_enabled:
            gc.enable()
        monitor.stop_tracing()
        gc.unfreeze()

    # every run is a task of its own, and every direct run imaged its channel
    tasks = [
        (
            kwargs.get("task_id"),
            tuple(
                float(frequency)
                for frequency in np.ravel(kwargs["task_coords"]["frequency"]["data"])
            ),
            repr(kwargs["data_selection"]),
        )
        for kwargs in captured
    ]
    for position in range(3):
        assert len({task[position] for task in tasks}) == N_TASKS, tasks
    assert all(imaged.values()), f"a direct run imaged nothing: {imaged}"

    img = xr.open_zarr(image_store)
    plane_bytes = (
        int(np.prod(img.SKY_RESIDUAL.shape[-2:])) * img.SKY_RESIDUAL.dtype.itemsize
    )
    assert plane_bytes >= MIN_PLANE_BYTES, plane_bytes
    assert img.SKY_RESIDUAL.dtype == np.float64

    def planes(n_bytes):
        return n_bytes / plane_bytes

    def mb(n_bytes):
        return n_bytes / MB

    lines = [
        f"plane {plane_bytes / MB:.0f} MB, visibilities {n_visibility_bytes / MB:.0f} "
        "MB per task; phase numbers in planes, relative to the start of the run; "
        "release numbers in MB; "
        + (
            "transparent huge pages "
            + (
                f"'always' (anonymous budgets at least {THP_ALWAYS_ANON_BUDGET_MB:.0f}"
                " MB)"
                if THP_ALWAYS
                else "not 'always'"
            )
            if CHECK_RESIDENT_SET
            else "not Linux: resident, malloc and anonymous memory reported only"
        )
        + (
            ""
            if MALLOC_IN_USE is not None or not CHECK_RESIDENT_SET
            else "; no glibc mallinfo2: malloc numbers reported only"
        )
        + (
            "; emulated program break (lowering it keeps the pages)"
            if BREAK_SHRINK_KEEPS_PAGES
            else ""
        )
        + f"; the garbage collector was kept off during the direct runs "
        f"({collector.enable_calls} calls of gc.enable ignored); tasks (task id, "
        f"channel): {[task[:2] for task in tasks]}"
    ]
    failures = []
    write_hint = numcodecs_hint()
    direct = monitor.runs[1:]
    warmed_labels = {run["label"] for run in direct[1:]}
    for run in monitor.runs:
        label = run["label"]
        lines.append(
            f"{label}: wall time {run['wall_seconds']:.1f} s, "
            f"{run['minor_faults']} minor page faults (reported only); traced peak "
            f"{planes(run['traced_peak']):.2f} (budget "
            f"{TRACEMALLOC_BUDGET_PLANES:.2f}), at return "
            f"{planes(run['traced_at_return']):.3f}; resident set size peak "
            f"{planes(run['rss_peak']):.2f}, at return "
            f"{planes(run['rss_at_return']):.3f}"
        )
        check_untraced = CHECK_RESIDENT_SET and run["label"] in warmed_labels
        check_gross = check_untraced and MALLOC_IN_USE is not None
        for _module_name, attribute, budget in PHASES:
            phase = run["phases"].get(attribute)
            if phase is None:
                failures.append(f"{label}: {attribute} was not called")
                continue
            added = planes(phase["added"])
            untraced = planes(phase["untraced"])
            gross = planes(phase.get("gross_untraced", math.nan))
            lines.append(
                f"  {attribute}: {phase['calls']} call(s), adds at most {added:.2f} "
                f"(budget {budget:.2f}); highest peak {planes(phase['peak']):.2f} "
                f"on {planes(phase['held']):.2f} held; resident set adds "
                f"{planes(phase['rss_added']):.2f}; untraced: net {untraced:+.3f}"
                + (
                    f" (budget {UNTRACED_BUDGET_PLANES:.2f})"
                    if check_untraced
                    else " (reported only)"
                )
                + f", gross malloc {gross:+.3f}"
                + (
                    f" (budget {GROSS_UNTRACED_BUDGET_PLANES:.2f})"
                    if check_gross
                    else ""
                )
                + f"; {phase['wall_seconds']:.1f} s, {phase['minor_faults']} minor "
                "page faults"
            )
            if added > budget:
                failures.append(
                    f"{label}: {attribute} adds {added:.2f} planes, budget {budget:.2f}"
                    + (write_hint if attribute.startswith("write_") else "")
                )
            if check_untraced and untraced > UNTRACED_BUDGET_PLANES:
                failures.append(
                    f"{label}: {attribute} adds {untraced:.2f} planes of untraced "
                    f"resident memory (net), budget {UNTRACED_BUDGET_PLANES:.2f}"
                )
            if check_gross and math.isnan(gross):
                failures.append(
                    f"{label}: the untraced malloc memory of {attribute} could not "
                    "be read (malloc bytes in use never quiet)"
                )
            elif check_gross and gross > GROSS_UNTRACED_BUDGET_PLANES:
                failures.append(
                    f"{label}: {attribute} allocates {gross:.3f} planes of untraced "
                    f"malloc memory at its peak, budget "
                    f"{GROSS_UNTRACED_BUDGET_PLANES:.2f}"
                )
        if planes(run["traced_peak"]) > TRACEMALLOC_BUDGET_PLANES:
            failures.append(
                f"{label}: node task traced peak {planes(run['traced_peak']):.2f} "
                f"planes, budget {TRACEMALLOC_BUDGET_PLANES:.2f}"
            )

    # run 1, the whole call of the distributed application
    driver_kept = driver_left["traced"] - driver_start["traced"] - imports.added
    lines.append(
        f"run 1, the whole call of the distributed application: traced memory "
        f"kept {planes(driver_kept):+.4f} planes ({mb(driver_kept):+.2f} MB) not "
        f"counting {mb(imports.added):.2f} MB of first imports (budgets "
        f"{DRIVER_RELEASE_BUDGET_PLANES:.4f} planes and {FIRST_IMPORTS_BUDGET_MB:.1f}"
        f" MB), {mb(driver_after_gc['traced'] - driver_start['traced'] - imports.added):+.2f}"
        f" MB after a collection that found {monitor.runs[0]['unreachable']} "
        f"objects in reference cycles "
        f"({monitor.runs[0]['cyclic_garbage'].most_common(4)} not counting "
        "xarray's dropped lazy trees, reported only)"
    )
    if planes(driver_kept) > DRIVER_RELEASE_BUDGET_PLANES:
        failures.append(
            f"run 1 (through the distributed application) keeps "
            f"{planes(driver_kept):.4f} planes of traced memory, not counting "
            f"first imports, budget {DRIVER_RELEASE_BUDGET_PLANES:.4f}: memory "
            "the first node task of a process does not release"
        )
    if mb(imports.added) > FIRST_IMPORTS_BUDGET_MB:
        failures.append(
            f"run 1 imports modules for the first time that allocate "
            f"{mb(imports.added):.2f} MB of traced memory, budget "
            f"{FIRST_IMPORTS_BUDGET_MB:.1f}: a module-level table of a lazily "
            "imported module"
        )
    # malloc bytes in use = traced memory allocated with malloc + tracemalloc's
    # own tables + untraced malloc memory; first imports taken out of each
    first_task_native = (
        driver_after_gc["in_use"]
        - driver_start["in_use"]
        - imports.in_use_added
        - (driver_after_gc["traced"] - driver_start["traced"] - imports.added)
        - (
            driver_after_gc["tracemalloc"]
            - driver_start["tracemalloc"]
            - imports.tracemalloc_added
        )
    )
    if MALLOC_IN_USE is not None:
        lines.append(
            f"run 1: malloc memory kept beyond traced memory and first imports "
            f"{mb(first_task_native):+.2f} MB (budget "
            f"{FIRST_TASK_NATIVE_BUDGET_MB:.2f}): malloc in use "
            f"{mb(driver_after_gc['in_use'] - driver_start['in_use']):+.2f}, traced "
            f"{mb(driver_after_gc['traced'] - driver_start['traced']):+.2f}, "
            "tracemalloc's own "
            f"{mb(driver_after_gc['tracemalloc'] - driver_start['tracemalloc']):+.2f}"
            f"; first imports {mb(imports.in_use_added):+.2f}, "
            f"{mb(imports.added):+.2f} and {mb(imports.tracemalloc_added):+.2f}; "
            f"malloc in use from the start of run 1 to the start of its node task "
            f"{mb(monitor.runs[0]['start']['in_use'] - driver_start['in_use']):+.2f}, "
            f"over the node task {mb(monitor.runs[0]['left_after_gc']['in_use']):+.2f}"
        )
        if math.isnan(first_task_native):
            failures.append(
                "run 1: the malloc memory it keeps could not be read (malloc bytes "
                "in use never quiet)"
            )
        elif mb(first_task_native) > FIRST_TASK_NATIVE_BUDGET_MB:
            failures.append(
                f"run 1 keeps {mb(first_task_native):.2f} MB of malloc memory "
                f"tracemalloc does not see, budget {FIRST_TASK_NATIVE_BUDGET_MB:.2f}"
                ": a C or C++ buffer the first node task of a process keeps"
            )
    if CHECK_RESIDENT_SET:
        first_task_outside = (
            driver_after_gc["anon_outside_malloc"] - driver_start["anon_outside_malloc"]
        )
        outside_budget = (
            THP_ALWAYS_FIRST_TASK_OUTSIDE_MALLOC_BUDGET_MB
            if THP_ALWAYS
            else FIRST_TASK_OUTSIDE_MALLOC_BUDGET_MB
        )
        # where lowering the break keeps the pages, the main heap's pages
        # given back during run 1 count as outside malloc: at most the
        # distance from the highest break of its node task to the final one
        # (the direct runs reuse those pages: their outside malloc memory is
        # not raised by it)
        returned = 0.0
        if BREAK_SHRINK_KEEPS_PAGES:
            returned = max(
                0.0, monitor.runs[0]["break_peak"] - driver_after_gc["break"]
            )
            outside_budget += mb(returned)
        check_outside = MALLOC_IN_USE is not None
        lines.append(
            f"run 1: anonymous memory kept "
            f"{mb(driver_after_gc['anon'] - driver_start['anon']):+.1f} MB (reported "
            f"only), of which outside malloc {mb(first_task_outside):+.1f} MB"
            + (
                f" (budget {outside_budget:.1f}"
                + (
                    f", of which {mb(returned):.1f} MB the main heap gave back by "
                    "lowering an emulated program break"
                    if BREAK_SHRINK_KEEPS_PAGES
                    else ""
                )
                + ")"
                if check_outside
                else " (reported only)"
            )
        )
        if check_outside and mb(first_task_outside) > outside_budget:
            failures.append(
                f"run 1 keeps {mb(first_task_outside):.1f} MB of anonymous memory "
                f"outside malloc, budget {outside_budget:.1f}: a buffer mapped "
                "directly that the first node task of a process keeps"
            )

    # every direct run releases its memory; nothing ratchets
    for position, run in enumerate(direct):
        label = run["label"]
        budgets = dict(RELEASE_BUDGETS_MB["warm-up" if position == 0 else "warmed"])
        if THP_ALWAYS:
            for key in ("anonymous", "outside malloc"):
                if budgets[key] is not None:
                    budgets[key] = max(budgets[key], THP_ALWAYS_ANON_BUDGET_MB)
        applies = {
            "native": MALLOC_IN_USE is not None,
            "anonymous": CHECK_RESIDENT_SET,
            "outside malloc": CHECK_RESIDENT_SET and MALLOC_IN_USE is not None,
        }
        left, after_gc = run["left"], run["left_after_gc"]
        cycles = run["cyclic_garbage"]
        ran = collector.collections.get(label, [])
        parts = [
            f"{label} leaves: by reference counting alone {mb(left['traced']):+.3f}"
            f" MB traced (budget {REFCOUNT_RELEASE_BUDGET_MB:.2f}); "
            f"{len(ran)} garbage collections inside the run (budget 0); "
            f"{run['unreachable']} objects in reference cycles, "
            f"{sum(cycles.values())} not counting xarray's dropped lazy trees "
            f"(budget "
            f"{CYCLIC_GARBAGE_BUDGET_OBJECTS}); after a collection "
            f"{mb(after_gc['traced']):+.3f} MB traced (budget "
            f"{POST_GC_RELEASE_BUDGET_MB:.2f})"
        ]
        for key, value, what in (
            (
                "native",
                after_gc["in_use"] - after_gc["traced"] - after_gc["tracemalloc"],
                "malloc memory tracemalloc does not see",
            ),
            ("anonymous", after_gc["anon"], "anonymous memory"),
            (
                "outside malloc",
                after_gc["anon_outside_malloc"],
                "anonymous memory outside malloc (mapped directly)",
            ),
        ):
            budget = budgets[key] if applies[key] else None
            parts.append(
                f"{key} {mb(value):+.3f} MB"
                + (" (reported only)" if budget is None else f" (budget {budget:.2f})")
            )
            if budget is None:
                continue
            if math.isnan(value):
                failures.append(
                    f"{label}: the {what} it leaves could not be read (malloc bytes "
                    "in use never quiet)"
                )
            elif mb(value) > budget:
                failures.append(
                    f"{label} leaves {mb(value):.3f} MB of {what}, budget {budget:.2f}"
                )
        lines.append(
            ", ".join(parts)
            + f"; resident set size {planes(after_gc['rss']):+.3f} planes (reported "
            "only)"
        )
        if ran:
            failures.append(
                f"{label}: {len(ran)} garbage collections ran inside the node task, "
                "where memory must be freed by reference counting alone (no "
                f"gc.collect() in astroviper or what it calls): {ran[:4]}"
            )
        if mb(left["traced"]) > REFCOUNT_RELEASE_BUDGET_MB:
            failures.append(
                f"{label} leaves {mb(left['traced']):.3f} MB of traced memory "
                f"behind with the garbage collector off, budget "
                f"{REFCOUNT_RELEASE_BUDGET_MB:.2f}"
            )
        if sum(cycles.values()) > CYCLIC_GARBAGE_BUDGET_OBJECTS:
            failures.append(
                f"{label} leaves {sum(cycles.values())} objects in reference cycles "
                f"(not released by reference counting), budget "
                f"{CYCLIC_GARBAGE_BUDGET_OBJECTS}: {cycles.most_common(12)}"
            )
        if mb(after_gc["traced"]) > POST_GC_RELEASE_BUDGET_MB:
            failures.append(
                f"{label} leaves {mb(after_gc['traced']):.3f} MB of traced "
                f"memory behind after a collection, budget "
                f"{POST_GC_RELEASE_BUDGET_MB:.2f}: memory still reachable"
            )
    previous, last = monitor.runs[-2:]
    for key, what in (
        ("traced_peak", "traced peak"),
        ("traced_at_return", "traced memory at return"),
    ):
        change = planes(last[key] - previous[key])
        lines.append(
            f"{last['label']} - {previous['label']}, {what}: {change:+.4f} (budget "
            f"+-{RATCHET_BUDGET_PLANES:.3f})"
        )
        if abs(change) > RATCHET_BUDGET_PLANES:
            failures.append(
                f"{last['label']} {what} differs from {previous['label']} by "
                f"{change:+.4f} planes, budget +-{RATCHET_BUDGET_PLANES:.3f}"
            )
    report = "\n".join(lines)
    print(report)
    assert not failures, "\n".join(failures) + "\n" + report


# --- find_continuum_ultra ----------------------------------------------------
#: Planes (px), their inner chunks (px) and the frequency shards (channels) of
#: the cube of the find_continuum tasks.
FIND_CONTINUUM_IMAGE_SIZE = 2048
FIND_CONTINUUM_TILE = 512
FIND_CONTINUUM_SHARD = 32
FIND_CONTINUUM_VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
#: The resident set of a task stays within this factor of
#: ``ultra_task_memory_bytes``.
FIND_CONTINUUM_RESIDENT_FACTOR = 1.3
#: Float32 planes by which the traced peak of a frequency task of 64 channels
#: may exceed that of a task of 16. Its per-channel rows add kilobytes; the
#: peak also depends on how the read of the next batch overlaps the
#: reductions of the current one (measured: -0.26 to +0.52 planes).
FIND_CONTINUUM_GROWTH_BUDGET_PLANES = 1.0


def resident_peak():
    """Largest resident set size of this process so far (bytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else 1024 * peak


def in_new_process(function, *args):
    """``function(*args)`` in a new process, as a worker runs a node task."""
    import concurrent.futures
    import multiprocessing

    with concurrent.futures.ProcessPoolExecutor(
        1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        return executor.submit(function, *args).result()


def find_continuum_cube(directory, n_channels):
    """Stokes I of a residual cube and of its primary beam, below 0.2 in the
    corner chunks, written channel by channel by the sharded skunk_works
    writer; returns the store path."""
    import zarr
    from xradio.image import make_empty_sky_image

    from astroviper.node_tasks.imaging.utils import (
        write_result_chunk_to_disk_sharded_skunk_works,
    )
    from astroviper.utils.io import (
        create_empty_data_variables_on_disk,
        write_zarr_image_store,
    )

    n_pixels = FIND_CONTINUUM_IMAGE_SIZE
    coords = make_empty_sky_image(
        phase_center=[1.2, -0.4],
        image_size=[n_pixels, n_pixels],
        cell_size=[0.1 * ARCSEC, 0.1 * ARCSEC],
        frequency_coords=230.0e9 + 0.5e6 * np.arange(n_channels),
        pol_coords=["I"],
        time_coords=[0.0],
    )
    store = write_zarr_image_store(coords, f"{directory}/find_continuum.img.zarr")
    create_empty_data_variables_on_disk(
        store,
        ["sky_residual", "primary_beam"],
        shape_dict=dict(coords.sizes),
        parallel_coords={
            "frequency": {"data_chunks": [[c] for c in range(n_channels)]}
        },
        compressor=zarr.codecs.BloscCodec(cname="lz4", clevel=5),
        double_precision=False,
        data_variable_definitions="imaging",
        image_chunking={
            "polarization": 1,
            "l": FIND_CONTINUUM_TILE,
            "m": FIND_CONTINUUM_TILE,
        },
        image_sharding={"frequency": FIND_CONTINUUM_SHARD},
    )
    offsets = np.indices((n_pixels, n_pixels)) - (n_pixels - 1) / 2
    primary_beam = np.exp(-np.square(offsets).sum(axis=0) / (2 * 400.0**2))
    # Multiples of 2**-10 keep the store small (lz4 ratio 0.08 against 0.56).
    primary_beam = (np.round(1024 * primary_beam) / 1024).astype(np.float32)
    dims = ["time", "frequency", "polarization", "l", "m"]
    chunk_xds = xr.Dataset(
        {
            name: (dims, plane[None, None, None])
            for name, plane in (
                ("SKY_RESIDUAL", primary_beam / 64),
                ("PRIMARY_BEAM", primary_beam),
            )
        }
    )
    for channel in range(n_channels):
        write_result_chunk_to_disk_sharded_skunk_works(
            store,
            ["sky_residual", "primary_beam"],
            {"frequency": {"slice": slice(channel, channel + 1)}},
            chunk_xds,
        )
    return store


def ultra_first_pass(store, work_directory, n_channels):
    """find_continuum_ultra's first pass over the first ``n_channels``
    channels of ``store`` as one frequency task, in this process (synchronous
    scheduler): the task's planned bytes, the peak of the memory traced
    during the pass and the resident set peak of the process."""
    from astroviper.distributed_applications.imaging.find_continuum.ultra_reductions import (
        UltraReductions,
    )
    from astroviper.processing_functions.imaging.find_continuum import (
        find_continuum_first_round_requests,
        linear_frequency_axis,
    )

    frequency = xr.open_zarr(store, chunks=None).frequency
    frequency_axis = linear_frequency_axis(frequency.values[:n_channels])
    rest_frequency = float(frequency.attrs["rest_frequency"]["data"])
    # The names findContinuum's requests give the cubes.
    paths = {role: f"{work_directory}/cube.{role}" for role in FIND_CONTINUUM_VARIABLES}
    ultra = UltraReductions(
        {
            paths[role]: {"variable": variable, "channel_offset": 0}
            for role, variable in FIND_CONTINUUM_VARIABLES.items()
        },
        store,
        FIND_CONTINUUM_VARIABLES,
        0,
        work_directory,
        0.2,
        None,
        1,
        16 * 2**30,
        frequency_axis,
        rest_frequency,
        None,
    )
    # Moments 0 and 8 of the cube and its whole-cube statistics, which the
    # first round answers: the pass reads the primary beam too.
    requests = find_continuum_first_round_requests(
        frequency_axis, rest_frequency, paths["sky"], paths["primary_beam"]
    )[1:]
    tracemalloc.start()
    with dask.config.set(scheduler="synchronous"):
        ultra.run_pass(requests, len(requests))
    traced_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return ultra.passes[0]["task_memory_plan_bytes"], traced_peak, resident_peak()


def first_round_chunk(work_directory, n_channels):
    """find_continuum's first round of a chunk the imager holds, in this
    process: ``n_channels`` channels of Stokes I and Q of the residual and
    the primary beam, in two spectral windows, so that the registry holds
    the moments of both. Returns the peak of the memory traced during the
    first round and the resident set peak of the process."""
    import uuid

    from astroviper.distributed_applications.imaging.find_continuum.first_round import (
        create_find_continuum_first_round_store,
        find_continuum_first_round_partials_path,
    )
    from astroviper.node_tasks.imaging.find_continuum import (
        find_continuum_first_round_chunk,
    )
    from astroviper.processing_functions.imaging.find_continuum import (
        linear_frequency_axis,
        moment_velocity_widths,
    )

    n_pixels, tile = FIND_CONTINUUM_IMAGE_SIZE, FIND_CONTINUUM_TILE
    shape = (1, n_channels, 2, n_pixels, n_pixels)
    rng = np.random.default_rng(1)
    planes = {
        "SKY_RESIDUAL": rng.standard_normal(shape, dtype=np.float32),
        "PRIMARY_BEAM": rng.random(shape, dtype=np.float32),
    }
    dims = ["time", "frequency", "polarization", "l", "m"]
    chunk_xds = xr.Dataset({name: (dims, cube) for name, cube in planes.items()})
    frequency = 230.0e9 + 0.5e6 * np.arange(n_channels)
    rest_frequency = 230.0e9
    windows = [
        {"channel_range": [start, stop], "frequency_axis": linear_frequency_axis(f)}
        for start, stop, f in (
            (0, 1, frequency[:1]),
            (1, n_channels, frequency[1:]),
        )
    ]
    store = f"{work_directory}/first_round.zarr"
    value_dtypes = {"sky": "float32", "primary_beam": "float32"}
    run_id = uuid.uuid4().hex
    create_find_continuum_first_round_store(
        first_round_store=store,
        n_channels=n_channels,
        image_shape=(n_pixels, n_pixels),
        tile_shape=(tile, tile),
        merge_tile_shape=(2 * tile, 2 * tile),
        primary_beam_limit=0.2,
        variables=FIND_CONTINUUM_VARIABLES,
        polarization_index=0,
        value_dtypes=value_dtypes,
        run_id=run_id,
        written_by="image_cube_single_field",
        windows=windows,
        rest_frequency=rest_frequency,
    )
    partials_directory = find_continuum_first_round_partials_path(store)
    os.makedirs(partials_directory)
    tracemalloc.start()
    find_continuum_first_round_chunk(
        chunk_xds,
        slice(0, n_channels),
        store,
        FIND_CONTINUUM_VARIABLES,
        0,
        0.2,
        (tile, tile),
        value_dtypes,
        run_id,
        [
            dict(
                window,
                velocity_widths=moment_velocity_widths(
                    window["frequency_axis"], rest_frequency
                ),
            )
            for window in windows
        ],
        partials_directory,
        rest_frequency,
    )
    traced_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return traced_peak, resident_peak()


def test_find_continuum_ultra_task_memory(tmp_path):
    """One frequency task of find_continuum_ultra's first pass without a
    first-round store (moments 0 and 8, and the first round from the primary
    beam), run in a new process as on a worker: its resident set stays within
    ``FIND_CONTINUUM_RESIDENT_FACTOR`` x the bytes the pass planned for it
    (``ultra_task_memory_bytes``), and, as it streams planes in batches, its
    traced peak does not grow from 16 to 64 channels per task."""
    store = find_continuum_cube(tmp_path, 64)
    runs = {
        n_channels: in_new_process(
            ultra_first_pass, store, f"{tmp_path}/work_{n_channels}", n_channels
        )
        for n_channels in (16, 64)
    }
    lines = [
        f"{n_channels} channels: planned {planned / 2**30:.3f} GiB, resident "
        f"set peak {resident / 2**30:.3f} GiB, traced peak {traced / MB:.1f} MB"
        for n_channels, (planned, traced, resident) in runs.items()
    ]
    print("\n".join(lines))
    for planned, _, resident in runs.values():
        assert resident <= FIND_CONTINUUM_RESIDENT_FACTOR * planned, lines
    growth = (runs[64][1] - runs[16][1]) / (4 * FIND_CONTINUUM_IMAGE_SIZE**2)
    assert growth <= FIND_CONTINUUM_GROWTH_BUDGET_PLANES, (growth, lines)


def test_find_continuum_first_round_chunk_memory(tmp_path):
    """The imager's first round of one chunk of two channels in two spectral
    windows, run in a new process: its resident set, the chunk's Stokes I
    and Q included, stays within ``FIND_CONTINUUM_RESIDENT_FACTOR`` x
    ``ultra_task_memory_bytes`` of a task holding as many float32 planes (two
    batches of two planes of the residual and the primary beam) and the
    moments of ``FIRST_ROUND_WINDOWS_IN_MEMORY`` windows."""
    from astroviper.distributed_applications.imaging.find_continuum.ultra_reductions import (
        ultra_task_memory_bytes,
    )
    from astroviper.node_tasks.imaging.find_continuum import (
        FIRST_ROUND_WINDOWS_IN_MEMORY,
    )

    n_channels = 2
    traced, resident = in_new_process(first_round_chunk, str(tmp_path), n_channels)
    planned = ultra_task_memory_bytes(
        FIND_CONTINUUM_IMAGE_SIZE,
        FIND_CONTINUUM_IMAGE_SIZE,
        n_channels,
        12 * FIRST_ROUND_WINDOWS_IN_MEMORY,
        0,
        0.0,
        True,
    )
    line = (
        f"planned {planned / 2**30:.3f} GiB, resident set peak "
        f"{resident / 2**30:.3f} GiB, traced peak {traced / MB:.1f} MB"
    )
    print(line)
    assert resident <= FIND_CONTINUUM_RESIDENT_FACTOR * planned, line


#: Layouts of the per-pixel check: the harness's (sharded, 1-channel inner
#: chunks of an eighth of the plane, float32), the same in float64, and the
#: imager's default (one unsharded chunk of two channels, Stokes I and Q and
#: the whole plane, float32).
FIND_CONTINUUM_LAYOUTS = ("sharded", "sharded_float64", "whole_plane")
#: Plane sizes (px) whose difference gives the bytes per pixel of a task.
FIND_CONTINUUM_SLOPE_SIZES = (512, 1024)
#: The traced bytes per pixel of a frequency task are within this fraction of
#: the plan's (measured: 1 to 6 % below it).
FIND_CONTINUUM_SLOPE_TOLERANCE = 0.1


def find_continuum_layout_cube(directory, n_pixels, layout, n_channels=8):
    """Stokes I and Q of a residual cube and its primary beam (below 0.2 in
    the corners) in one of ``FIND_CONTINUUM_LAYOUTS``; returns the store."""
    import zarr
    from xradio.image import make_empty_sky_image

    from astroviper.node_tasks.imaging.utils import (
        write_result_chunk_to_disk_sharded_skunk_works,
        write_result_chunk_to_disk_using_zarr_skunk_works,
    )
    from astroviper.utils.io import (
        create_empty_data_variables_on_disk,
        write_zarr_image_store,
    )

    dtype = np.float64 if layout == "sharded_float64" else np.float32
    coords = make_empty_sky_image(
        phase_center=[1.2, -0.4],
        image_size=[n_pixels, n_pixels],
        cell_size=[0.1 * ARCSEC, 0.1 * ARCSEC],
        frequency_coords=230.0e9 + 0.5e6 * np.arange(n_channels),
        pol_coords=["I", "Q"],
        time_coords=[0.0],
    )
    store = write_zarr_image_store(coords, f"{directory}/{layout}_{n_pixels}.img.zarr")
    sharded = layout != "whole_plane"
    task = 1 if sharded else 2
    create_empty_data_variables_on_disk(
        store,
        ["sky_residual", "primary_beam"],
        shape_dict=dict(coords.sizes),
        parallel_coords={
            "frequency": {
                "data_chunks": [
                    list(range(c, c + task)) for c in range(0, n_channels, task)
                ]
            }
        },
        compressor=zarr.codecs.BloscCodec(cname="lz4", clevel=5),
        double_precision=dtype == np.float64,
        data_variable_definitions="imaging",
        image_chunking={"polarization": 1, "l": n_pixels // 8, "m": n_pixels // 8}
        if sharded
        else None,
        image_sharding={"frequency": n_channels} if sharded else None,
    )
    write = (
        write_result_chunk_to_disk_sharded_skunk_works
        if sharded
        else write_result_chunk_to_disk_using_zarr_skunk_works
    )
    offsets = np.indices((n_pixels, n_pixels)) - (n_pixels - 1) / 2
    sigma = 0.45 * n_pixels / np.sqrt(2 * np.log(5.0))
    primary_beam = np.exp(-np.square(offsets).sum(axis=0) / (2 * sigma**2))
    primary_beam = (np.round(1024 * primary_beam) / 1024).astype(dtype)
    rng = np.random.default_rng(0)
    dims = ["time", "frequency", "polarization", "l", "m"]
    for start in range(0, n_channels, task):
        sky = np.empty((1, task, 2, n_pixels, n_pixels), dtype=dtype)
        for k in range(task):
            sky[0, k, 0] = 0.01 * primary_beam + 1e-3 * rng.standard_normal(
                (n_pixels, n_pixels)
            )
            sky[0, k, 1] = 0.1 * sky[0, k, 0]
        chunk_xds = xr.Dataset(
            {
                "SKY_RESIDUAL": (dims, sky),
                "PRIMARY_BEAM": (dims, np.broadcast_to(primary_beam, sky.shape).copy()),
            }
        )
        write(
            store,
            ["sky_residual", "primary_beam"],
            {"frequency": {"slice": slice(start, start + task)}},
            chunk_xds,
        )
    return store


def ultra_pass_memory(store, work_directory):
    """find_continuum_ultra's first pass without a first-round store over the
    whole cube as one frequency task, then the moments again with the store
    it made, in this process (synchronous scheduler): per pass the bytes
    planned for a task and the traced peak of the frequency task, and the
    resident set peak of the first pass above the resident set before it."""
    import psutil

    from astroviper.distributed_applications.imaging.find_continuum import (
        ultra_reductions,
    )
    from astroviper.processing_functions.imaging.find_continuum import (
        find_continuum_first_round_requests,
        linear_frequency_axis,
    )

    peaks = []
    task = ultra_reductions.find_continuum_streamed_reductions

    @functools.wraps(task)
    def traced(**kwargs):
        tracemalloc.start()
        try:
            return task(**kwargs)
        finally:
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()

    ultra_reductions.find_continuum_streamed_reductions = traced
    frequency = xr.open_zarr(store, chunks=None).frequency
    frequency_axis = linear_frequency_axis(frequency.values)
    rest_frequency = float(frequency.attrs["rest_frequency"]["data"])
    paths = {role: f"{work_directory}/cube.{role}" for role in FIND_CONTINUUM_VARIABLES}
    ultra = ultra_reductions.UltraReductions(
        {
            paths[role]: {"variable": variable, "channel_offset": 0}
            for role, variable in FIND_CONTINUUM_VARIABLES.items()
        },
        store,
        FIND_CONTINUUM_VARIABLES,
        0,
        work_directory,
        0.2,
        None,
        1,
        16 * 2**30,
        frequency_axis,
        rest_frequency,
        None,
    )
    requests = find_continuum_first_round_requests(
        frequency_axis, rest_frequency, paths["sky"], paths["primary_beam"]
    )[1:]
    resident = psutil.Process().memory_info().rss
    with dask.config.set(scheduler="synchronous"):
        ultra.run_pass(requests, len(requests))
        first_pass_resident = resident_peak() - resident
        ultra.run_pass(requests, len(requests))
    return {
        "planned": [row["task_memory_plan_bytes"] for row in ultra.passes],
        "traced": peaks,
        "resident": first_pass_resident,
    }


@pytest.mark.parametrize("layout", FIND_CONTINUUM_LAYOUTS)
def test_find_continuum_ultra_task_memory_per_pixel(tmp_path, layout):
    """The bytes per pixel of a frequency task, from its peaks at two plane
    sizes (free of the fixed baseline), against the plan's: the traced ones
    within ``FIND_CONTINUUM_SLOPE_TOLERANCE``, the resident ones within
    ``FIND_CONTINUUM_RESIDENT_FACTOR``, for the first pass without a store
    (which reads the primary beam and makes the first round) and with the
    store it made."""
    runs = {}
    for n_pixels in FIND_CONTINUUM_SLOPE_SIZES:
        store = find_continuum_layout_cube(tmp_path, n_pixels, layout)
        runs[n_pixels] = in_new_process(
            ultra_pass_memory, store, f"{tmp_path}/work_{layout}_{n_pixels}"
        )
    small, large = (runs[n] for n in FIND_CONTINUUM_SLOPE_SIZES)
    pixels = np.diff(np.square(FIND_CONTINUUM_SLOPE_SIZES))[0]
    lines, failures = [], []
    for index, name in enumerate(("without a store", "with the store")):
        planned = (large["planned"][index] - small["planned"][index]) / pixels
        traced = (large["traced"][index] - small["traced"][index]) / pixels
        lines.append(
            f"{layout}, {name}: planned {planned:.1f} B/px, traced {traced:.1f} B/px"
        )
        if abs(traced - planned) > FIND_CONTINUUM_SLOPE_TOLERANCE * planned:
            failures.append(lines[-1])
        if index == 0:
            resident = (large["resident"] - small["resident"]) / pixels
            lines[-1] += f", resident {resident:.1f} B/px"
            if resident > FIND_CONTINUUM_RESIDENT_FACTOR * planned:
                failures.append(lines[-1])
    print("\n".join(lines))
    assert not failures, failures
