"""Benchmark the per-task data load of skunk-works cube imaging from a Measurement Set v2.

Not collected by pytest (like the other ``benchmark_*.py``): run it as a
script. It builds the node-task mapping exactly as the distributed
application ``image_cube_single_field`` does for a Measurement Set v2
``ps_store`` (lazy open with XRADIO's ``xradio_msv2`` engine, one task per
``--n-tasks`` frequency chunk of the image axis, the correlation selection of
``--stokes``, ``add_lazy_input_data``) and then times the load of every task
in this process, without imaging:

* ``engine``: ``load_processing_set_skunk_works_msv2`` (what a node task does
  with its ``lazy_input_data``: the values are read by the engine);
* ``zarr:<path>``: ``load_processing_set_skunk_works`` (the Zarr skunk-works
  loader) on each processing set given with ``--zarr``, or made with
  ``--convert`` (the converter with its default chunks and with one channel
  per chunk), for the same task selections. Their values are compared with
  the engine's (``values_equal_engine``).

Per task it records the load time (``T_load``) and the bytes the process read
(``rchar`` of ``/proc/self/io``, Linux; counted also when served from the page
cache); ``--cold`` evicts the files read from the page cache before every
task. It also reports

* the graph payload: the pickled size of a task's ``lazy_input_data``;
* the storage tile shapes of the Measurement Set's data columns (with
  python-casacore): a task reads whole tiles, so when a tile spans every
  channel of a spectral window every task reads whole rows, and converting
  the Measurement Set first is faster for an at-scale run;
* ``channel_sliced_reads``: whether the installed XRADIO reads only the tiles
  of the selected channels (XRADIO's tile-aligned channel-sliced reads),
  measured by reading one channel and then every channel of a time window of
  the first MSv4 (``not applicable`` when the column's tiles span every
  channel);
* ``ms_unchanged``: the Measurement Set's files (names, sizes, mtimes) are the
  same after the benchmark (the engine is opened with
  ``partition_cache="read"`` unless ``XRADIO_MSV2_PARTITION_CACHE`` is set).

Requirements: XRADIO with the ``xradio_msv2`` engine and python-casacore.
Data: a Measurement Set v2 path, or the name of a ToolVIPER test dataset
(downloaded into ``--work-dir`` with ``toolviper.utils.data.download``). The
benchmark never writes into the Measurement Set; ``--convert`` writes the two
processing sets into ``--work-dir``.

Examples::

    python benchmark_skunk_works_msv2_load.py Antennae_North.cal.lsrk.split.ms \\
        --work-dir /scratch/bench --convert --stokes I,Q --basis linear
    python benchmark_skunk_works_msv2_load.py /data/3c391_ctm_mosaic_10s_spw0.ms \\
        --data-group corrected --stokes I,V --basis circular --max-tasks 8 --cold

Each run prints one JSON record (and appends it to ``--output`` when given).
"""

import argparse
import copy
import hashlib
import json
import os
import pickle
import resource
import sys
import time

#: MSv2 data columns whose storage tile shapes are reported.
_DATA_COLUMNS = (
    "DATA",
    "CORRECTED_DATA",
    "MODEL_DATA",
    "FLOAT_DATA",
    "FLAG",
    "WEIGHT_SPECTRUM",
)

#: MSv4 correlated-data variable -> the MSv2 column it is read from.
_MSV2_COLUMN = {
    "VISIBILITY": "DATA",
    "VISIBILITY_CORRECTED": "CORRECTED_DATA",
    "VISIBILITY_MODEL": "MODEL_DATA",
    "SPECTRUM": "FLOAT_DATA",
}


def read_chars():
    """Bytes this process has read so far (``rchar`` of ``/proc/self/io``)."""
    try:
        with open("/proc/self/io") as io:
            for line in io:
                if line.startswith("rchar"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def evict_page_cache(path):
    """Drop the files under ``path`` from the page cache (read-only opens)."""
    for directory, _, files in os.walk(path):
        for name in files:
            descriptor = os.open(os.path.join(directory, name), os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)


def file_state(path):
    """Every file under ``path``: its relative path, size and mtime."""
    state = {}
    for directory, _, files in os.walk(path):
        for name in files:
            full = os.path.join(directory, name)
            stat = os.stat(full)
            state[os.path.relpath(full, path)] = (stat.st_size, stat.st_mtime_ns)
    return state


def column_tile_shapes(ms_path):
    """Storage tile shapes of the Measurement Set's data columns.

    Returns ``{column: [tile shape, ...]}`` (casacore order: polarization,
    channel, rows), or ``None`` without python-casacore.
    """
    try:
        from casacore import tables
    except ImportError:
        return None
    shapes = {}
    with tables.table(
        ms_path, ack=False, readonly=True, lockoptions="usernoread"
    ) as main:
        for manager in main.getdminfo().values():
            for column in manager["COLUMNS"]:
                if column not in _DATA_COLUMNS:
                    continue
                spec = manager.get("SPEC", {})
                hypercubes = spec.get("HYPERCUBES") or {}
                tiles = {
                    tuple(int(extent) for extent in cube.get("TileShape", []))
                    for cube in hypercubes.values()
                }
                if not tiles and "DEFAULTTILESHAPE" in spec:
                    tiles = {tuple(int(e) for e in spec["DEFAULTTILESHAPE"])}
                shapes[column] = {
                    "manager": manager["TYPE"],
                    "tile_shapes": sorted(tiles),
                }
    return shapes


def channel_slice_probe(ps_xdt, data_group_name, tile_shapes, n_times=16):
    """Whether the engine reads only the tiles of the selected channels.

    Reads the correlated data of the first MSv4 with more than one channel
    over ``n_times`` times: one channel, then every channel (in this order, so
    that a tile cache of the process can only make the ratio larger).
    """
    names = [
        name for name, node in ps_xdt.items() if node.sizes.get("frequency", 0) > 1
    ]
    if not names:
        return {"channel_sliced_reads": "not applicable (one channel)"}
    ms_name = names[0]
    ms_xdt = ps_xdt[ms_name]
    variable = ms_xdt.attrs["data_groups"][data_group_name]["correlated_data"]
    window = ms_xdt.ds[variable].isel(time=slice(0, min(n_times, ms_xdt.sizes["time"])))
    start = read_chars()
    window.isel(frequency=slice(0, 1)).load()
    one_channel = read_chars() - start
    start = read_chars()
    window.load()
    all_channels = read_chars() - start
    n_channels = int(ms_xdt.sizes["frequency"])
    column = _MSV2_COLUMN.get(variable)
    tiles = ((tile_shapes or {}).get(column) or {}).get("tile_shapes") or []
    tile_channels = sorted({tile[1] for tile in tiles if len(tile) == 3})
    ratio = one_channel / all_channels if all_channels else None
    if not tile_channels:
        verdict = "unknown (no tile shape)"
    elif min(tile_channels) >= n_channels:
        verdict = "not applicable (tiles span every channel)"
    elif ratio is not None and ratio < 0.75:
        verdict = "yes"
    else:
        verdict = "no"
    return {
        "channel_sliced_reads": verdict,
        "probe_msv4": ms_name,
        "probe_variable": variable,
        "probe_column": column,
        "probe_n_channels": n_channels,
        "probe_tile_channels": tile_channels,
        "probe_rchar_one_channel_MiB": round(one_channel / 2**20, 3),
        "probe_rchar_all_channels_MiB": round(all_channels / 2**20, 3),
        "probe_ratio": None if ratio is None else round(ratio, 3),
    }


def build_mapping(ps_xdt, stokes, basis, n_tasks, data_group_name):
    """The node-task mapping of the distributed application, with
    ``lazy_input_data``; returns ``(mapping, timings)``."""
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )

    from astroviper.node_tasks.imaging.utils import add_lazy_input_data
    from astroviper.processing_functions.imaging.utils.imaging_polarization import (
        correlation_selection,
        correlations_for_stokes,
    )

    timings = {}
    frequencies = ps_xdt.xr_ps.get_freq_axis()
    n_chunks = n_tasks if n_tasks > 0 else int(frequencies.size)
    start = time.perf_counter()
    parallel_coords = {
        "frequency": make_parallel_coord(coord=frequencies, n_chunks=n_chunks)
    }
    mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords, ps_xdt)
    correlations = correlations_for_stokes(stokes, basis)
    for ms_name, ms_xdt in ps_xdt.items():
        selection = correlation_selection(
            ms_xdt.polarization.values, correlations, ms_name
        )
        if selection is None:
            continue
        for task in mapping.values():
            if task["data_selection"].get(ms_name) is not None:
                task["data_selection"][ms_name]["polarization"] = selection
    timings["T_mapping_s"] = time.perf_counter() - start
    start = time.perf_counter()
    add_lazy_input_data(ps_xdt, mapping, data_group_name)
    timings["T_add_lazy_input_data_s"] = time.perf_counter() - start
    return mapping, timings


def payload(mapping):
    """Pickled size of the tasks' ``lazy_input_data`` and the deep-copy time
    of a task (GraphVIPER copies every task's parameters)."""
    import numpy as np

    sizes = [
        len(pickle.dumps(task["lazy_input_data"], protocol=5))
        for task in mapping.values()
    ]
    n_msv4 = sum(len(task["lazy_input_data"]) for task in mapping.values())
    start = time.perf_counter()
    for task in mapping.values():
        copy.deepcopy(task)
    return {
        "payload_task_bytes_mean": int(np.mean(sizes)),
        "payload_task_bytes_max": int(np.max(sizes)),
        "payload_per_msv4_bytes_mean": int(sum(sizes) / max(1, n_msv4)),
        "deepcopy_task_ms": round(
            1000 * (time.perf_counter() - start) / len(mapping), 3
        ),
    }


def digest(ps_xdt):
    """SHA-1 of the data variables (names, dtypes, shapes, values) of every
    MSv4 of a loaded processing set."""
    import numpy as np

    sha = hashlib.sha1()
    for ms_name in sorted(ps_xdt.children):
        dataset = ps_xdt[ms_name].ds
        for name in sorted(dataset.data_vars):
            values = np.asarray(dataset[name].values)
            sha.update(f"{ms_name}/{name}{values.dtype}{values.shape}".encode())
            sha.update(np.ascontiguousarray(values).tobytes())
    return sha.hexdigest()


def numpy_ready(ps_xdt):
    """Number of data variables that are not C-contiguous writeable NumPy."""
    import numpy as np

    bad = 0
    for ms_name in ps_xdt.children:
        for variable in ps_xdt[ms_name].ds.data_vars.values():
            values = variable.values
            if not (
                isinstance(values, np.ndarray)
                and values.flags.c_contiguous
                and values.flags.writeable
            ):
                bad += 1
    return bad


def time_loads(mapping, loader, evict_path, task_ids):
    """Load every task of ``task_ids``; returns ``(summary, digests)``."""
    import numpy as np

    seconds, read_bytes, digests, not_ready = [], [], [], 0
    for task_id in task_ids:
        task = mapping[task_id]
        if evict_path:
            evict_page_cache(evict_path)
        before, start = read_chars(), time.perf_counter()
        ps_xdt = loader(task)
        seconds.append(time.perf_counter() - start)
        read_bytes.append(read_chars() - before)
        digests.append(digest(ps_xdt))
        not_ready += numpy_ready(ps_xdt)
        del ps_xdt
    return {
        "n_tasks": len(task_ids),
        "T_load_mean_ms": round(1000 * float(np.mean(seconds)), 2),
        "T_load_median_ms": round(1000 * float(np.median(seconds)), 2),
        "T_load_max_ms": round(1000 * float(np.max(seconds)), 2),
        "T_load_sum_s": round(float(np.sum(seconds)), 3),
        "rchar_task_mean_MiB": round(float(np.mean(read_bytes)) / 2**20, 3),
        "rchar_sum_MiB": round(float(np.sum(read_bytes)) / 2**20, 1),
        "variables_not_numpy_ready": not_ready,
    }, digests


def resolve_ms(ms, work_dir):
    """The Measurement Set path; a name that is not a path is downloaded."""
    if os.path.exists(ms):
        return os.path.abspath(ms)
    from toolviper.utils.data import download

    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, os.path.basename(ms))
    if not os.path.exists(path):
        download(ms, folder=work_dir)
    return path


def convert(ms_path, work_dir):
    """Convert the Measurement Set twice (converter's default chunks, and one
    channel per chunk) into ``work_dir``; returns the two store paths."""
    from xradio.measurement_set import convert_msv2_to_processing_set

    stores = []
    name = os.path.basename(ms_path.rstrip("/")).removesuffix(".ms")
    for label, options in (
        ("default_chunks", {}),
        ("1_channel_chunks", {"main_chunksize": {"frequency": 1}}),
    ):
        store = os.path.join(work_dir, f"{name}.{label}.ps.zarr")
        if not os.path.exists(store):
            convert_msv2_to_processing_set(
                ms_path, store, partition_scheme=[], persistence_mode="w", **options
            )
        stores.append(store)
    return stores


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ms", help="Measurement Set v2 path, or a ToolVIPER dataset name"
    )
    parser.add_argument("--data-group", default="base")
    parser.add_argument(
        "--stokes", default="I,Q", help="comma list (image polarization_coords)"
    )
    parser.add_argument("--basis", default="linear", choices=["linear", "circular"])
    parser.add_argument(
        "--scan-intents", default=None, help="comma list; default every intent"
    )
    parser.add_argument(
        "--n-tasks", type=int, default=0, help="frequency chunks (0: one per channel)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="time at most this many tasks, evenly spaced (0: all)",
    )
    parser.add_argument("--zarr", nargs="*", default=[], help="processing sets")
    parser.add_argument(
        "--convert", action="store_true", help="convert into --work-dir"
    )
    parser.add_argument("--work-dir", default=".")
    parser.add_argument(
        "--cold", action="store_true", help="evict the page cache before each task"
    )
    parser.add_argument("--output", default=None, help="append the JSON record")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    import numpy as np
    import xradio

    from astroviper.node_tasks.imaging.utils import (
        load_processing_set_skunk_works,
        load_processing_set_skunk_works_msv2,
        open_processing_set_skunk_works_msv2,
        require_msv2_engine,
    )

    require_msv2_engine()
    ms_path = resolve_ms(arguments.ms, arguments.work_dir)
    state_before = file_state(ms_path)
    stokes = arguments.stokes.split(",")
    scan_intents = arguments.scan_intents.split(",") if arguments.scan_intents else None
    record = {
        "ms": os.path.basename(ms_path.rstrip("/")),
        "data_group": arguments.data_group,
        "stokes": stokes,
        "cold": arguments.cold,
        "xradio_version": getattr(xradio, "__version__", None),
        "xradio_path": os.path.dirname(xradio.__file__),
    }
    record["tile_shapes"] = column_tile_shapes(ms_path)

    start = time.perf_counter()
    ps_xdt = open_processing_set_skunk_works_msv2(ms_path, scan_intents=scan_intents)
    record["T_open_s"] = round(time.perf_counter() - start, 3)
    record["n_msv4"] = len(ps_xdt.children)
    record.update(
        channel_slice_probe(ps_xdt, arguments.data_group, record["tile_shapes"])
    )
    mapping, timings = build_mapping(
        ps_xdt, stokes, arguments.basis, arguments.n_tasks, arguments.data_group
    )
    record.update({key: round(value, 4) for key, value in timings.items()})
    record["n_tasks_mapped"] = len(mapping)
    record.update(payload(mapping))

    task_ids = sorted(mapping)
    if 0 < arguments.max_tasks < len(task_ids):
        picks = np.linspace(0, len(task_ids) - 1, arguments.max_tasks)
        task_ids = [task_ids[int(round(pick))] for pick in picks]
    record["task_ids"] = task_ids

    record["engine"], engine_digests = time_loads(
        mapping,
        lambda task: load_processing_set_skunk_works_msv2(task["lazy_input_data"]),
        ms_path if arguments.cold else None,
        task_ids,
    )

    stores = list(arguments.zarr)
    if arguments.convert:
        stores += convert(ms_path, arguments.work_dir)
    for store in stores:
        from xradio.measurement_set import open_processing_set

        first = next(iter(open_processing_set(store).values()))
        data_group = dict(first.attrs["data_groups"][arguments.data_group])

        def zarr_loader(task, store=store, data_group=data_group):
            return load_processing_set_skunk_works(
                store,
                sel_parms=task["data_selection"],
                data_group=data_group,
                processing_set_data_group_name=arguments.data_group,
                frequency_coords=task["task_coords"]["frequency"]["data"],
                instrument_polarization_basis=arguments.basis,
            )

        key = "zarr:" + os.path.basename(store.rstrip("/"))
        record[key], digests = time_loads(
            mapping, zarr_loader, store if arguments.cold else None, task_ids
        )
        record[key]["values_equal_engine"] = digests == engine_digests

    record["peak_rss_MiB"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )
    record["ms_unchanged"] = file_state(ms_path) == state_before
    line = json.dumps(record, default=str)
    print(line, flush=True)
    if arguments.output:
        with open(arguments.output, "a") as output:
            output.write(line + "\n")
    return record


if __name__ == "__main__":
    main()
