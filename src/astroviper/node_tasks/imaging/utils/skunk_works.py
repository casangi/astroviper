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

import numpy as np
import numcodecs

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
            "flag": (os.path.join(ms_path, data_group["flag"]), {"frequency": freq_sel}),
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


def _write_one_variable(dv, image_store, task_coords, img_xds):
    """Encode and write this task's chunk for one image variable.

    The chunk grid index is reconstructed from ``task_coords`` (parallel dims ->
    ``slice.start // chunk_size``; other dims -> 0), the array is encoded with the
    variable's on-disk codecs, and the blob is written straight to
    ``<store>/<VAR>/c/<i0>/.../<iN>``.  Partial (edge) chunks are padded to the
    full chunk shape with the array's fill value so the blob round-trips through
    Zarr.  Touches only this variable's own array/files, so it is safe to run for
    several variables concurrently.
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)


def write_result_chunk_to_disk_using_zarr_skunk_works(
    image_store, image_data_variables_keep, task_coords, img_xds, num_threads=1
):
    """Write this task's image chunk(s) directly to their Zarr chunk file(s).

    Each kept variable's chunk is encoded (compressed) and written by
    :func:`_write_one_variable`; no ``open_group`` or ``open_zarr`` is used -- the
    empty image and its metadata were created by the distributed graph.  The
    variables are encoded and written concurrently across ``num_threads`` threads:
    each touches a distinct array/file and the numcodecs compression (blosc/zstd)
    and the file write both release the GIL, so the per-variable compression --
    the dominant write cost -- overlaps instead of running one variable at a time.

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
        Maximum number of threads used to encode/write the variables
        concurrently.  Default ``1`` (serial).
    """
    variables = list(image_data_variables_keep)
    if num_threads <= 1 or len(variables) <= 1:
        for dv in variables:
            _write_one_variable(dv, image_store, task_coords, img_xds)
        return

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(num_threads, len(variables))) as executor:
        futures = [
            executor.submit(_write_one_variable, dv, image_store, task_coords, img_xds)
            for dv in variables
        ]
        for future in futures:
            future.result()  # re-raise any per-variable write error
