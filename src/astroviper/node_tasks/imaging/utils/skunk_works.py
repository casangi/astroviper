"""Performance-oriented (``_skunk_works``) I/O variants for cube imaging.

These are **experimental test variants** of the two I/O bottlenecks in the
single-field cube imaging node task, written for running at scale (thousands of
cores, ~15000 single/few-channel ``image_cube_single_field`` tasks). They trade
generality for speed:

* :func:`load_processing_set_skunk_works` reads **only** the data variables of
  the requested data group, going straight to the on-disk Zarr chunk blobs
  instead of opening a dataset/datatree through the asyncio Zarr array API. For
  a sharded array it reads the shard index and then ``pread``\\ s only the byte
  ranges of the inner chunks the task's frequency selection touches -- it does
  not read the whole shard (a single shard often spans every channel of the
  array). The data-group arrays are read concurrently across ``num_threads``
  threads. Every other coordinate is *reconstructed* from the node-task inputs
  (the processing set and the image are assumed to share the same frequency
  coordinate); all sub-datasets are ignored.
* :func:`write_result_chunk_to_disk_using_zarr_skunk_works` writes **only** this
  task's chunk by encoding each kept variable's array to its compressed blob and
  writing the file directly -- no ``open_group``/``open_zarr`` and no metadata
  round trip. The empty image (and its Zarr metadata) was already created by the
  distributed graph. The variables are compressed and written concurrently
  across ``num_threads`` threads (the compression is the dominant write cost).

Nothing here uses dask: the node task is already wrapped in dask, and each task
owns a disjoint set of chunks, so reads and writes are embarrassingly parallel.

The functions deliberately do not remove or replace the production
:func:`astroviper.utils.io.write_result_chunk_to_disk_using_zarr` /
``load_processing_set`` paths; they are selected by the ``skunk_works`` flag.
"""

from __future__ import annotations

import json
import os
import struct
from itertools import product

import numcodecs
import numpy as np

# Zarr v3 ``data_type`` -> little-endian NumPy dtype string.
_V3_DTYPE = {
    "bool": "|b1",
    "int8": "|i1",
    "int16": "<i2",
    "int32": "<i4",
    "int64": "<i8",
    "uint8": "|u1",
    "uint16": "<u2",
    "uint32": "<u4",
    "uint64": "<u8",
    "float16": "<f2",
    "float32": "<f4",
    "float64": "<f8",
    "complex64": "<c8",
    "complex128": "<c16",
}

# zarr v3 BloscCodec shuffle name -> numcodecs Blosc shuffle int.
_BLOSC_SHUFFLE = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2, None: 1}

_UINT64_MAX = (1 << 64) - 1

# Correlation polarization labels by instrument basis and number of correlations.
_POL_LABELS = {
    "linear": {1: ["XX"], 2: ["XX", "YY"], 4: ["XX", "XY", "YX", "YY"]},
    "circular": {1: ["RR"], 2: ["RR", "LL"], 4: ["RR", "RL", "LR", "LL"]},
}


# ---------------------------------------------------------------------------
# Low-level metadata + codec helpers (below the asyncio Zarr array API)
# ---------------------------------------------------------------------------
def _np_dtype(meta_dtype):
    """Return a NumPy dtype for a v2 (``"<c8"``) or v3 (``"complex64"`` /
    ``{"name": "fixed_length_utf32", ...}``) spec."""
    if isinstance(meta_dtype, dict):
        name = meta_dtype.get("name", "")
        nbytes = meta_dtype.get("configuration", {}).get("length_bytes", 0)
        if name == "fixed_length_utf32":  # UTF-32-LE == NumPy unicode (UCS4)
            return np.dtype(f"<U{nbytes // 4}")
        if name in ("fixed_length_ascii", "fixed_length_bytes"):
            return np.dtype(f"|S{nbytes}")
        raise ValueError(f"Unsupported Zarr v3 string data_type: {name!r}")
    if isinstance(meta_dtype, str) and meta_dtype in _V3_DTYPE:
        return np.dtype(_V3_DTYPE[meta_dtype])
    return np.dtype(meta_dtype)


def _read_array_meta(array_path):
    """Read just one array's Zarr metadata (no data, no group/tree open).

    Supports Zarr v3 (``zarr.json``, optionally ``sharding_indexed``) and v2
    (``.zarray`` + ``.zattrs``). Returns a small plain dict.
    """
    v3_path = os.path.join(array_path, "zarr.json")
    if os.path.exists(v3_path):
        with open(v3_path) as fh:
            m = json.load(fh)
        codecs = m["codecs"]
        sharding = next((c for c in codecs if c["name"] == "sharding_indexed"), None)
        key = m.get("chunk_key_encoding", {}).get("configuration", {})
        return {
            "format": 3,
            "shape": tuple(m["shape"]),
            "dims": list(m["dimension_names"]),
            "dtype": _np_dtype(m["data_type"]),
            "outer_chunks": tuple(m["chunk_grid"]["configuration"]["chunk_shape"]),
            "sep": key.get("separator", "/"),
            "codecs": codecs,
            "sharding": sharding["configuration"] if sharding else None,
            "fill_value": m.get("fill_value"),
        }

    with open(os.path.join(array_path, ".zarray")) as fh:
        m = json.load(fh)
    with open(os.path.join(array_path, ".zattrs")) as fh:
        dims = json.load(fh)["_ARRAY_DIMENSIONS"]
    return {
        "format": 2,
        "shape": tuple(m["shape"]),
        "dims": list(dims),
        "dtype": np.dtype(m["dtype"]),
        "outer_chunks": tuple(m["chunks"]),
        "sep": m.get("dimension_separator", "."),
        "sharding": None,
        "compressor": m.get("compressor"),
        "filters": m.get("filters"),
        "order": m.get("order", "C"),
        "fill_value": m.get("fill_value"),
    }


def _chunk_path(array_path, meta, chunk_index):
    """On-disk path of one (outer) chunk / shard for ``chunk_index``."""
    parts = [str(i) for i in chunk_index]
    if meta["format"] == 3:
        if meta["sep"] == "/":
            return os.path.join(array_path, "c", *parts)
        return os.path.join(array_path, "c", meta["sep"].join(parts))
    if meta["sep"] == "/":
        return os.path.join(array_path, *parts)
    return os.path.join(array_path, meta["sep"].join(parts))


def _decode_bytes_bytes(blob, codecs_after_array):
    """Undo the byte->byte codecs (zstd/blosc/gzip/crc32c) of a v3 chunk."""
    b = bytes(blob)
    for codec in reversed(codecs_after_array):
        name = codec["name"]
        if name == "zstd":
            b = numcodecs.Zstd().decode(b)
        elif name == "blosc":
            b = numcodecs.Blosc().decode(b)
        elif name in ("gzip", "gz"):
            b = numcodecs.GZip().decode(b)
        elif name == "crc32c":
            b = b[:-4]  # strip the trailing 4-byte checksum
        elif name == "bytes":
            pass  # array<->bytes, handled by frombuffer
        else:
            raise ValueError(f"Unsupported Zarr v3 codec for decode: {name!r}")
    return b


def _decode_unit(raw, shape, dtype, meta):
    """Decode a whole (non-sharded) chunk file ``raw`` into an array."""
    if meta["format"] == 3:
        after = [
            c for c in meta["codecs"] if c["name"] not in ("bytes", "sharding_indexed")
        ]
        buf = _decode_bytes_bytes(raw, after)
        return np.frombuffer(buf, dtype=dtype).reshape(shape)
    comp = meta.get("compressor")
    buf = numcodecs.get_codec(comp).decode(raw) if comp else bytes(raw)
    return np.frombuffer(buf, dtype=dtype).reshape(shape, order=meta.get("order", "C"))


def _read_shard_index(fd, file_size, meta, inner_per_shard):
    """Read just a shard's inner-chunk byte index (its small header/footer).

    Returns a flat tuple of ``(offset, nbytes)`` pairs in C-order over the
    inner-chunk grid.  Only the index bytes are read -- not the chunk data --
    so a task that wants one channel does not pull the whole shard off disk.
    """
    sh = meta["sharding"]
    n_inner = int(np.prod(inner_per_shard))
    has_crc = any(c["name"] == "crc32c" for c in sh.get("index_codecs", []))
    index_nbytes = n_inner * 16 + (4 if has_crc else 0)
    if sh.get("index_location", "end") == "start":
        index_blob = os.pread(fd, index_nbytes, 0)
    else:
        index_blob = os.pread(fd, index_nbytes, file_size - index_nbytes)
    if has_crc:
        index_blob = index_blob[:-4]
    return struct.unpack("<" + "Q" * (2 * n_inner), index_blob)


def _read_inner_chunk(fd, offsets, flat, inner_shape, sharding_cfg, dtype):
    """``pread`` + decode one inner chunk (``flat`` C-order index) from a shard.

    Reads only that inner chunk's byte range from ``fd``; returns ``None`` for
    an empty (all-fill) inner chunk so the caller can fill it.  ``os.pread`` is
    positional -- it does not use or move the shared file offset -- so several
    threads may read disjoint inner chunks from the same ``fd`` concurrently.
    """
    off = offsets[2 * flat]
    nbytes = offsets[2 * flat + 1]
    if off == _UINT64_MAX or nbytes == _UINT64_MAX:
        return None  # empty inner chunk -> caller fills
    after = [c for c in sharding_cfg["codecs"] if c["name"] != "bytes"]
    buf = _decode_bytes_bytes(os.pread(fd, nbytes, off), after)
    return np.frombuffer(buf, dtype=dtype).reshape(inner_shape)


def _axis_overlaps(start, stop, csize):
    """Chunks along one axis overlapping ``[start, stop)``.

    Yields ``(chunk_index, local_lo, local_hi, out_lo, out_hi)`` where
    ``[local_lo:local_hi]`` indexes inside the chunk and ``[out_lo:out_hi]``
    indexes inside the output (relative to ``start``).
    """
    out = []
    for ci in range(start // csize, (stop - 1) // csize + 1):
        cstart = ci * csize
        lo = max(start, cstart)
        hi = min(stop, cstart + csize)
        out.append((ci, lo - cstart, hi - cstart, lo - start, hi - start))
    return out


def read_array_region(array_path, sel):
    """Read ``array[sel]`` for one Zarr array via direct chunk-blob reads.

    ``sel`` maps dimension names to slices; missing dims select the full axis.
    Handles Zarr v3 (sharded and plain) and v2; supports selections spanning
    several chunks / inner chunks (e.g. multiple frequency channels). Returns
    ``(ndarray, dims)``.
    """
    meta = _read_array_meta(array_path)
    shape, dims, dtype = meta["shape"], meta["dims"], meta["dtype"]
    oc = meta["outer_chunks"]
    nd = len(shape)

    ranges = []
    for ax, dim in enumerate(dims):
        s = sel.get(dim, slice(None))
        a, b, _ = s.indices(shape[ax])
        ranges.append((a, b))
    out = np.empty(tuple(b - a for a, b in ranges), dtype=dtype)
    fill = meta["fill_value"] if meta["fill_value"] is not None else 0

    if meta["sharding"] is not None:
        inner = tuple(meta["sharding"]["chunk_shape"])
        inner_per_shard = tuple(oc[ax] // inner[ax] for ax in range(nd))
        per_axis = [
            _axis_overlaps(ranges[ax][0], ranges[ax][1], inner[ax]) for ax in range(nd)
        ]
        # Group the needed inner chunks by their shard so each shard file is
        # opened and its index read exactly once.  A single shard can span the
        # whole array (one inner chunk per channel), so reading just the wanted
        # inner-chunk byte ranges -- rather than the whole shard file -- avoids
        # over-reading by the channel count.
        by_shard = {}
        for combo in product(*per_axis):
            gi = tuple(c[0] for c in combo)  # global inner-chunk index
            shard_index = tuple(gi[ax] // inner_per_shard[ax] for ax in range(nd))
            within = tuple(gi[ax] % inner_per_shard[ax] for ax in range(nd))
            out_sl = tuple(slice(c[3], c[4]) for c in combo)
            in_sl = tuple(slice(c[1], c[2]) for c in combo)
            by_shard.setdefault(shard_index, []).append((within, out_sl, in_sl))

        for shard_index, items in by_shard.items():
            path = _chunk_path(array_path, meta, shard_index)
            if not os.path.exists(path):
                for _within, out_sl, _in_sl in items:
                    out[out_sl] = fill
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                offsets = _read_shard_index(
                    fd, os.fstat(fd).st_size, meta, inner_per_shard
                )
                for within, out_sl, in_sl in items:
                    flat = int(np.ravel_multi_index(within, inner_per_shard))
                    arr = _read_inner_chunk(
                        fd, offsets, flat, inner, meta["sharding"], dtype
                    )
                    out[out_sl] = fill if arr is None else arr[in_sl]
            finally:
                os.close(fd)
        return out, dims

    per_axis = [
        _axis_overlaps(ranges[ax][0], ranges[ax][1], oc[ax]) for ax in range(nd)
    ]
    for combo in product(*per_axis):
        ci = tuple(c[0] for c in combo)
        out_sl = tuple(slice(c[3], c[4]) for c in combo)
        path = _chunk_path(array_path, meta, ci)
        if not os.path.exists(path):
            out[out_sl] = fill
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        arr = _decode_unit(raw, oc, dtype, meta)
        out[out_sl] = arr[tuple(slice(c[1], c[2]) for c in combo)]
    return out, dims


# ---------------------------------------------------------------------------
# Public skunk-works load
# ---------------------------------------------------------------------------
def _read_arrays_concurrently(reads, num_threads):
    """Read several Zarr arrays at once via :func:`read_array_region`.

    ``reads`` maps an arbitrary key to ``(array_path, sel)``; the return value
    maps the same keys to ``(ndarray, dims)``.  With ``num_threads <= 1`` (or a
    single array) the reads run serially; otherwise each array is read on its
    own thread.  The reads touch disjoint files and return independent arrays,
    so there is nothing to synchronise.
    """
    items = list(reads.items())
    if num_threads <= 1 or len(items) <= 1:
        return {key: read_array_region(path, sel) for key, (path, sel) in items}

    from concurrent.futures import ThreadPoolExecutor

    results = {}
    with ThreadPoolExecutor(max_workers=min(num_threads, len(items))) as executor:
        futures = {
            executor.submit(read_array_region, path, sel): key
            for key, (path, sel) in items
        }
        for future in futures:
            results[futures[future]] = future.result()
    return results


def load_processing_set_skunk_works(
    input_data_store,
    sel_parms,
    data_group,
    processing_set_data_group_name,
    frequency_coords,
    instrument_polarization_basis="linear",
    num_threads=1,
):
    """Reconstruct a minimal processing set for cube imaging from chunk blobs.

    Only the data variables of ``data_group`` (correlated data, uvw, weight,
    flag) are read, straight from the Zarr chunk files for this task's frequency
    selection. Coordinates are reconstructed from the node-task inputs:

    * ``frequency`` from ``frequency_coords`` (the image and the processing set
      share the same frequency coordinate);
    * ``polarization`` from ``instrument_polarization_basis`` and the data's
      correlation count;
    * ``time`` / ``baseline_id`` / ``uvw_label`` as plain index ranges (the
      imaging only uses their sizes and the ``UVW`` values).

    All sub-datasets are ignored. Returns an :class:`xarray.DataTree` whose
    children are the selected measurement sets, with ``attrs["data_groups"]``
    set so the science path is indistinguishable from a normal load.

    Parameters
    ----------
    input_data_store : str
        Path to the processing-set Zarr store.
    sel_parms : dict
        ``{ms_name: {dim: slice}}`` selection for this task (from the graph).
        The ``frequency`` slice gives the channel range to read.
    data_group : dict
        Resolved role->variable mapping for ``processing_set_data_group_name``
        (e.g. ``{"correlated_data": "VISIBILITY", "uvw": "UVW", ...}``), passed
        down from the distributed graph.
    processing_set_data_group_name : str
        Name to register the data group under in ``attrs["data_groups"]``.
    frequency_coords : array-like
        Frequency values for this task's channels (``task_coords["frequency"]["data"]``).
    instrument_polarization_basis : str
        ``"linear"`` or ``"circular"``; used to label the polarization axis.
    num_threads : int, optional
        Maximum number of threads used to read this MS's arrays concurrently.
        Default ``1`` (serial).  File reads and the numcodecs decode both
        release the GIL, so threading overlaps the per-array I/O latency.
    """
    import xarray as xr

    freq_values = np.asarray(frequency_coords)
    nodes = {}
    for ms_name, ms_sel in sel_parms.items():
        ms_path = os.path.join(input_data_store, ms_name)
        freq_sel = (
            ms_sel.get("frequency", slice(None))
            if isinstance(ms_sel, dict)
            else slice(None)
        )

        # Collect every array this MS needs -- the four data-group variables and
        # the two tiny antenna-name coordinates (needed to drop auto-correlations
        # and not reconstructable from the imaging inputs) -- and read them all
        # concurrently.  Both the file I/O and the numcodecs decode release the
        # GIL, so threads overlap the per-array read latency instead of paying
        # it one array at a time.
        reads = {
            "correlated_data": (
                os.path.join(ms_path, data_group["correlated_data"]),
                {"frequency": freq_sel},
            ),
            "uvw": (os.path.join(ms_path, data_group["uvw"]), {}),
            "weight": (
                os.path.join(ms_path, data_group["weight"]),
                {"frequency": freq_sel},
            ),
            "flag": (
                os.path.join(ms_path, data_group["flag"]),
                {"frequency": freq_sel},
            ),
        }
        for coord_name in ("baseline_antenna1_name", "baseline_antenna2_name"):
            if os.path.isdir(os.path.join(ms_path, coord_name)):
                reads[coord_name] = (os.path.join(ms_path, coord_name), {})

        results = _read_arrays_concurrently(reads, num_threads)

        vis, vis_dims = results["correlated_data"]
        uvw, uvw_dims = results["uvw"]
        weight, w_dims = results["weight"]
        flag, f_dims = results["flag"]

        npol = vis.shape[vis_dims.index("polarization")]
        pol_labels = _POL_LABELS[instrument_polarization_basis].get(
            npol, [f"P{i}" for i in range(npol)]
        )

        data_vars = {
            data_group["correlated_data"]: (vis_dims, vis),
            data_group["uvw"]: (uvw_dims, uvw),
            data_group["weight"]: (w_dims, weight),
            data_group["flag"]: (f_dims, flag),
        }
        coords = {
            "frequency": ("frequency", freq_values),
            "polarization": ("polarization", pol_labels),
        }
        # The per-baseline antenna names cannot be reconstructed from the imaging
        # inputs and were read above (when present) to drop auto-correlations.
        for coord_name in ("baseline_antenna1_name", "baseline_antenna2_name"):
            if coord_name in results:
                values, c_dims = results[coord_name]
                coords[coord_name] = (c_dims, values)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.attrs["data_groups"] = {processing_set_data_group_name: dict(data_group)}
        nodes[ms_name] = ds

    return xr.DataTree.from_dict(nodes)


# ---------------------------------------------------------------------------
# Public skunk-works write
# ---------------------------------------------------------------------------
def _encode_chunk_blob(chunk, meta):
    """Encode a full-chunk array to the on-disk blob for ``meta``'s codecs."""
    if meta["format"] == 3:
        b = np.ascontiguousarray(chunk, dtype=meta["dtype"])
        out = b.tobytes()
        for codec in meta["codecs"]:
            name = codec["name"]
            if name in ("bytes", "sharding_indexed"):
                continue
            cfg = codec.get("configuration", {})
            if name == "zstd":
                out = numcodecs.Zstd(level=cfg.get("level", 0)).encode(out)
            elif name == "blosc":
                out = numcodecs.Blosc(
                    cname=cfg.get("cname", "lz4"),
                    clevel=cfg.get("clevel", 5),
                    shuffle=_BLOSC_SHUFFLE.get(cfg.get("shuffle"), 1),
                ).encode(
                    b
                )  # ndarray -> correct typesize for shuffle
            elif name in ("gzip", "gz"):
                out = numcodecs.GZip(level=cfg.get("level", 5)).encode(out)
            elif name == "crc32c":
                from numcodecs import checksum32

                out = checksum32.CRC32C().encode(out)
            else:
                raise ValueError(f"Unsupported Zarr v3 codec for encode: {name!r}")
        return out

    order = meta.get("order", "C")
    arr = np.asfortranarray(chunk) if order == "F" else np.ascontiguousarray(chunk)
    comp = meta.get("compressor")
    return numcodecs.get_codec(comp).encode(arr) if comp else arr.tobytes()


def _encode_one_variable(dv, image_store, task_coords, img_xds):
    """Encode (compress) this task's chunk for one image variable to its on-disk
    blob, WITHOUT writing it.

    The chunk grid index is reconstructed from ``task_coords`` (parallel dims ->
    ``slice.start // chunk_size``; other dims -> 0), the array is encoded with the
    variable's on-disk codecs, and the ``(path, blob)`` for
    ``<store>/<VAR>/c/<i0>/.../<iN>`` is returned.  Partial (edge) chunks are
    padded to the full chunk shape with the array's fill value so the blob
    round-trips through Zarr.  Touches only this variable's own metadata, so it is
    safe to run for several variables concurrently.  Splitting encode from the
    disk write lets the CPU-bound compression run concurrently while the writes
    stay serial (see :func:`write_result_chunk_to_disk_using_zarr_skunk_works`).
    """
    name = dv.upper()
    array_path = os.path.join(image_store, name)
    meta = _read_array_meta(array_path)
    oc = meta["outer_chunks"]
    dims = list(img_xds[name].dims)

    chunk_index = []
    for ax, dim in enumerate(dims):
        if dim in task_coords and isinstance(task_coords[dim].get("slice"), slice):
            start = task_coords[dim]["slice"].start or 0
            chunk_index.append(start // oc[ax])
        else:
            chunk_index.append(0)

    values = np.asarray(img_xds[name].values)
    if values.shape != tuple(oc):
        fill = meta["fill_value"]
        if fill is None or (isinstance(fill, float) and np.isnan(fill)):
            fill = np.nan if meta["dtype"].kind in "fc" else 0
        chunk = np.full(oc, fill, dtype=meta["dtype"])
        chunk[tuple(slice(0, s) for s in values.shape)] = values
    else:
        chunk = values

    blob = _encode_chunk_blob(chunk, meta)
    path = _chunk_path(array_path, meta, tuple(chunk_index))
    return path, blob


def _write_blob(path, blob):
    """Write one pre-encoded chunk blob to disk (create its parent dir first)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)


def write_result_chunk_to_disk_using_zarr_skunk_works(
    image_store, image_data_variables_keep, task_coords, img_xds, num_threads=1
):
    """Write this task's image chunk(s) directly to their Zarr chunk file(s).

    Two phases, deliberately decoupled:

    1. **Encode (compress)** each kept variable's chunk with its on-disk codecs.
       This is the dominant, CPU-bound, GIL-releasing cost and runs concurrently
       across up to ``num_threads`` threads (one per variable).
    2. **Write** the resulting blobs to disk **serially** -- one open file at a
       time for this task.

    Keeping the compression threaded but the writes serial is intentional for
    running at scale on a shared parallel filesystem (Lustre): the per-variable
    concurrency previously opened one file per variable *per task* simultaneously
    (e.g. 3 concurrent ~1 GB writes x thousands of tasks), which floods the OSTs /
    metadata server.  Serial writes cut a task's concurrent open files to one
    while the (larger) compression cost still overlaps.  No ``open_group`` /
    ``open_zarr`` is used -- the empty image and its metadata were created by the
    distributed graph.

    Parameters
    ----------
    image_store : str
        Path of the pre-created on-disk Zarr image store.
    image_data_variables_keep : list of str
        Logical image-variable keys to write (e.g. ``"sky_residual"``); each is
        upper-cased to the on-disk array name.
    task_coords : dict
        Per-chunk coordinate mapping; the parallel dims' ``slice`` give the chunk
        grid index this task owns.
    img_xds : xarray.Dataset
        The computed image holding this task's chunk for each variable.
    num_threads : int, optional
        Maximum number of threads used to *encode/compress* the variables
        concurrently (the disk writes are always serial).  Default ``1`` (serial
        encode).
    """
    variables = list(image_data_variables_keep)

    # Phase 1: encode/compress (optionally concurrent across variables).
    if num_threads <= 1 or len(variables) <= 1:
        encoded = [
            _encode_one_variable(dv, image_store, task_coords, img_xds)
            for dv in variables
        ]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(num_threads, len(variables))
        ) as executor:
            futures = [
                executor.submit(
                    _encode_one_variable, dv, image_store, task_coords, img_xds
                )
                for dv in variables
            ]
            encoded = [
                f.result() for f in futures
            ]  # order matches variables; re-raises

    # Phase 2: write the blobs serially (one open file at a time for this task).
    for path, blob in encoded:
        _write_blob(path, blob)


# ---------------------------------------------------------------------------
# Sharded write: many single-chunk tasks write into shared, pre-created shard
# files (the TACC "single parallel file" pattern -> far fewer files, MDS relief)
# ---------------------------------------------------------------------------
# Each shard file packs ``n_inner`` inner chunks (one frequency channel each) at
# FIXED, disjoint "slots" plus a Zarr v3 sharding index.  Because slots are
# disjoint and pre-sized, independent per-channel tasks ``pwrite`` their inner
# chunk and its 16-byte index entry with NO locking and NO file creation (the
# shard files are created once, sparse, by :func:`precreate_sharded_files` in the
# driver).  A slot's unused tail is a filesystem hole (Lustre is sparse), so
# fixed slots waste no real space while keeping compression (the index records
# the true byte length, so reads and stored bytes are the compressed size).
#
# The output is a spec-compliant Zarr v3 sharded array (verified readable by both
# stock zarr and :func:`read_array_region`); the arrays are created WITHOUT an
# index CRC so independent writers can each set their own index entry.
_SHARD_INDEX_EMPTY = (1 << 64) - 1


def _shard_inner_per_shard(meta):
    """``(inner_per_shard tuple, n_inner)`` for a sharded array's metadata."""
    shard = meta["outer_chunks"]
    inner = meta["sharding"]["chunk_shape"]
    ips = tuple(shard[i] // inner[i] for i in range(len(shard)))
    return ips, int(np.prod(ips))


def _assert_sharded_no_index_crc(meta, array_path):
    """The fixed-slot writer/precreator hard-code a CRC-free shard index (bare
    ``n_inner * 16`` bytes). Fail fast if the array was created with a ``crc32c``
    index codec (e.g. zarr's default), which would shift the index by 4 bytes and
    both corrupt our reads and make stock zarr reject the shard. GraphVIPER's
    creation path (io.create_empty_data_variables_on_disk) sets
    ``index_codecs=[BytesCodec()]``, so this only guards against a store created
    another way.
    """
    index_codecs = (meta.get("sharding") or {}).get("index_codecs", [])
    if any(c.get("name") == "crc32c" for c in index_codecs):
        raise ValueError(
            f"{array_path}: sharded array has a crc32c index codec, which the "
            "concurrent fixed-slot writer cannot maintain. Create it with "
            "index_codecs=[BytesCodec()] (no CRC)."
        )


def _shard_slot_size(meta):
    """Fixed per-inner-chunk slot size (bytes): the uncompressed inner-chunk size
    **floored to a whole MiB plus a 2 MiB margin** (so always >= 1 MiB above the
    raw size). This is a comfortable upper bound on the compressed inner-chunk
    size (compression essentially never exceeds the raw size + a tiny header); the
    unused tail of each slot is a sparse hole, so the generous bound costs no real
    disk space, and :func:`_encode_one_variable_sharded` hard-fails if a blob ever
    exceeds it (rather than corrupting the shard).
    """
    inner = meta["sharding"]["chunk_shape"]
    nbytes = int(np.prod(inner)) * meta["dtype"].itemsize
    return ((nbytes >> 20) + 2) << 20


def _encode_inner_chunk(chunk, meta):
    """Encode one inner chunk to its on-disk bytes using the SHARDING inner codecs
    (``meta["sharding"]["codecs"]`` -- e.g. bytes -> blosc), not the array's
    top-level codec (which for a sharded array is just the sharding codec)."""
    b = np.ascontiguousarray(chunk, dtype=meta["dtype"])
    out = b.tobytes()
    did_transform = False  # a byte-transform codec has run
    for codec in meta["sharding"]["codecs"]:
        name = codec["name"]
        if name == "bytes":
            continue
        cfg = codec.get("configuration", {})
        if name == "blosc":
            # blosc encodes the ndarray directly (to infer the shuffle typesize),
            # so it must be the FIRST transform codec (only 'bytes' may precede it),
            # else a preceding transform's output would be silently discarded.
            if did_transform:
                raise ValueError(
                    "blosc must be the first transform codec in the sharding inner "
                    "codec chain (it re-encodes the raw array)."
                )
            out = numcodecs.Blosc(
                cname=cfg.get("cname", "lz4"),
                clevel=cfg.get("clevel", 5),
                shuffle=_BLOSC_SHUFFLE.get(cfg.get("shuffle"), 1),
            ).encode(b)
            did_transform = True
        elif name == "zstd":
            out = numcodecs.Zstd(level=cfg.get("level", 0)).encode(out)
            did_transform = True
        elif name in ("gzip", "gz"):
            out = numcodecs.GZip(level=cfg.get("level", 5)).encode(out)
            did_transform = True
        elif name == "crc32c":
            from numcodecs import checksum32

            out = checksum32.CRC32C().encode(out)
        else:
            raise ValueError(f"Unsupported sharding inner codec for encode: {name!r}")
    return out


def precreate_sharded_files(array_path):
    """Pre-create every shard file of a sharded array as a sparse file with an
    all-empty index. Run ONCE in the driver (empty-image creation) so the
    concurrent write phase creates NO files -- eliminating the per-chunk
    file-create storm on the metadata server.

    Each shard file is ``n_inner * slot_size + index_size`` bytes; only the small
    index tail (and, later, the written inner chunks) materialise on disk. The
    index is initialised to the empty marker so unwritten channels read back as
    the array's fill value.
    """
    from itertools import product

    meta = _read_array_meta(array_path)
    if meta.get("sharding") is None:
        raise ValueError(
            f"{array_path} is not a sharded array; cannot pre-create shards."
        )
    _assert_sharded_no_index_crc(meta, array_path)
    ips, n_inner = _shard_inner_per_shard(meta)
    slot = _shard_slot_size(meta)
    index_size = n_inner * 16
    total = n_inner * slot + index_size
    shard = meta["outer_chunks"]
    n_shards = tuple(
        (meta["shape"][i] + shard[i] - 1) // shard[i] for i in range(len(shard))
    )
    empty_index = struct.pack(
        "<" + "Q" * (2 * n_inner), *([_SHARD_INDEX_EMPTY] * (2 * n_inner))
    )
    for shard_index in product(*[range(n) for n in n_shards]):
        path = _chunk_path(array_path, meta, shard_index)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, total)  # sparse
            os.pwrite(fd, empty_index, n_inner * slot)  # index at end, all-empty
        finally:
            os.close(fd)


def _encode_one_variable_sharded(dv, image_store, task_coords, img_xds):
    """Encode this task's inner chunk for one variable and resolve where it goes in
    its (pre-created) shard file.

    Returns ``(shard_path, slot_offset, index_offset, blob)``. Does not write.
    """
    name = dv.upper()
    array_path = os.path.join(image_store, name)
    meta = _read_array_meta(array_path)
    _assert_sharded_no_index_crc(meta, array_path)
    inner = meta["sharding"]["chunk_shape"]
    dims = list(img_xds[name].dims)

    # Global inner-chunk index (parallel dims -> slice.start // inner; else 0).
    chunk_index = []
    for ax, dim in enumerate(dims):
        if dim in task_coords and isinstance(task_coords[dim].get("slice"), slice):
            start = task_coords[dim]["slice"].start or 0
            chunk_index.append(start // inner[ax])
        else:
            chunk_index.append(0)

    values = np.asarray(img_xds[name].values)
    if values.shape != tuple(inner):
        fill = meta["fill_value"]
        if fill is None or (isinstance(fill, float) and np.isnan(fill)):
            fill = np.nan if meta["dtype"].kind in "fc" else 0
        chunk = np.full(inner, fill, dtype=meta["dtype"])
        chunk[tuple(slice(0, s) for s in values.shape)] = values
    else:
        chunk = values

    blob = _encode_inner_chunk(chunk, meta)
    slot = _shard_slot_size(meta)
    ips, n_inner = _shard_inner_per_shard(meta)
    if len(blob) > slot:
        raise ValueError(
            f"{name}: encoded inner chunk is {len(blob)} B but the shard slot is "
            f"{slot} B. Increase the slot margin in _shard_slot_size."
        )
    shard_index = tuple(chunk_index[i] // ips[i] for i in range(len(chunk_index)))
    within = tuple(chunk_index[i] % ips[i] for i in range(len(chunk_index)))
    flat = int(np.ravel_multi_index(within, ips))
    shard_path = _chunk_path(array_path, meta, shard_index)
    slot_offset = flat * slot
    index_offset = n_inner * slot + flat * 16
    return shard_path, slot_offset, index_offset, blob


def write_result_chunk_to_disk_sharded_skunk_works(
    image_store, image_data_variables_keep, task_coords, img_xds, num_threads=1
):
    """Sharded direct-write: write this task's inner chunk for each kept variable
    into its SHARED, pre-created shard file at a fixed disjoint slot, and set its
    Zarr v3 index entry.

    Many single-channel tasks write into one shard file concurrently (disjoint
    byte ranges -> lock-free, no file creation). Compression of the variables runs
    concurrently (``num_threads``); the ``pwrite``\\ s are serial per task. The
    shard files must already exist (:func:`precreate_sharded_files`).

    Parameters mirror :func:`write_result_chunk_to_disk_using_zarr_skunk_works`.
    """
    variables = list(image_data_variables_keep)

    # Phase 1: encode/compress each variable's inner chunk (optionally concurrent).
    if num_threads <= 1 or len(variables) <= 1:
        encoded = [
            _encode_one_variable_sharded(dv, image_store, task_coords, img_xds)
            for dv in variables
        ]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(num_threads, len(variables))
        ) as executor:
            futures = [
                executor.submit(
                    _encode_one_variable_sharded, dv, image_store, task_coords, img_xds
                )
                for dv in variables
            ]
            encoded = [f.result() for f in futures]

    # Phase 2: pwrite each inner chunk + its index entry into the shared shard
    # file (serial per task; disjoint offsets across tasks -> concurrency-safe).
    for shard_path, slot_offset, index_offset, blob in encoded:
        try:
            fd = os.open(shard_path, os.O_WRONLY)  # pre-created; no O_CREAT
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"shard file {shard_path} not found; precreate_sharded_files must "
                "run in the driver before the sharded writer."
            ) from exc
        try:
            os.pwrite(fd, blob, slot_offset)
            os.pwrite(fd, struct.pack("<QQ", slot_offset, len(blob)), index_offset)
        finally:
            os.close(fd)
