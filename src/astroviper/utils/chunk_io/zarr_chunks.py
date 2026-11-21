"""Zarr reads and writes of the chunk I/O engine (``io_engine="chunk"``).

Node tasks read and write Zarr chunk blobs directly by byte offset, written for
running at scale (thousands of cores, ~15000 single/few-channel
``image_cube_single_field`` tasks). They trade generality for speed:

* :func:`load_processing_set` reads **only** the data variables of
  the requested data group, going straight to the on-disk Zarr chunk blobs
  instead of opening a dataset/datatree through the asyncio Zarr array API. For
  a sharded array it reads the shard index and then ``pread``\\ s only the byte
  ranges of the inner chunks the task's frequency selection touches -- it does
  not read the whole shard (a single shard often spans every channel of the
  array). The data-group arrays are read concurrently across ``processing_function_threads``
  threads. Every other coordinate is *reconstructed* from the node-task inputs
  (the processing set and the image are assumed to share the same frequency
  coordinate); all sub-datasets are ignored.
* :func:`write_zarr` writes **only** this
  task's chunk by encoding each kept variable's array to its compressed blob and
  writing the file directly -- no ``open_group``/``open_zarr`` and no metadata
  round trip. The empty image (and its Zarr metadata) was already created by the
  distributed graph. The variables are compressed and written concurrently
  across ``processing_function_threads`` threads (the compression is the dominant write cost).

Nothing here uses dask: the node task is already wrapped in dask, and each task
owns a disjoint set of chunks, so reads and writes are embarrassingly parallel.

The xradio I/O engine (``io_engine="xradio"``) uses
:func:`astroviper.utils.io.write_result_chunk_to_disk_using_zarr` and XRADIO's
``load_processing_set`` instead.

A Measurement Set v2 is read through XRADIO's ``xradio_msv2`` engine
(:mod:`astroviper.utils.measurement_set_v2_input`) by both engines; the writers
here serve both inputs.
"""

from __future__ import annotations

import base64
import errno
import json
import os
import struct
from itertools import product

import numcodecs
import numpy as np
from xarray.backends import BackendArray

from astroviper.utils.param_docs import shares_param_docs

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
# Inner chunks of one shard read at the same time (zarr's default async
# concurrency): ``os.pread`` on one descriptor is thread-safe and the decode
# releases the GIL.
_READ_CONCURRENCY = 10

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


def _decode_float_fill_value(fill):
    """Decode one JSON float, including Zarr v3 big-endian IEEE hex bits."""
    if isinstance(fill, str) and fill.startswith("0x"):
        formats = {4: ">e", 8: ">f", 16: ">d"}
        fmt = formats.get(len(fill) - 2)
        if fmt is None:
            raise ValueError(
                f"Invalid hexadecimal float fill {fill!r}: expected 4, 8 or 16 hex digits."
            )
        return struct.unpack(fmt, bytes.fromhex(fill[2:]))[0]
    return float(fill)


def _decode_fill_value(fill, dtype, *, zarr_format):
    """Decode JSON numeric/v2 byte fills to a scalar, without array copies."""
    if fill is None:
        return None
    if zarr_format == 2 and dtype.kind in "SV":
        return dtype.type(base64.b64decode(fill, validate=True))
    if dtype.kind == "c" and isinstance(fill, (list, tuple)):
        # Decode each component independently, preserving NaNs, infinities and
        # signed zeros without complex arithmetic or data-sized temporaries.
        fill = complex(
            _decode_float_fill_value(fill[0]), _decode_float_fill_value(fill[1])
        )
    elif dtype.kind == "f":
        fill = _decode_float_fill_value(fill)
    return dtype.type(fill) if dtype.kind in "biufc" else fill


def _check_v3_codec_endianness(codecs, array_path):
    """Reject unsupported big-endian data and shard-index bytes before I/O."""
    for codec in codecs:
        config = codec.get("configuration", {})
        if codec["name"] == "bytes" and config.get("endian") == "big":
            raise NotImplementedError(
                f"Big-endian Zarr v3 bytes codecs are not supported by the "
                f"chunk I/O engine: {array_path}. Use a little-endian store."
            )
        if codec["name"] == "sharding_indexed":
            _check_v3_codec_endianness(config.get("codecs", []), array_path)
            _check_v3_codec_endianness(config.get("index_codecs", []), array_path)


def _read_array_meta(array_path):
    """Read just one array's Zarr metadata (no data, no group/tree open).

    Supports Zarr v3 (``zarr.json``, optionally ``sharding_indexed``) and v2
    (``.zarray`` + ``.zattrs``). Returns a small plain dict.
    Big-endian v3 data/index codecs raise ``NotImplementedError`` before I/O.
    """
    v3_path = os.path.join(array_path, "zarr.json")
    if os.path.exists(v3_path):
        with open(v3_path) as fh:
            m = json.load(fh)
        codecs = m["codecs"]
        _check_v3_codec_endianness(codecs, array_path)
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
            "fill_value": _decode_fill_value(
                m.get("fill_value"), _np_dtype(m["data_type"]), zarr_format=3
            ),
            "bool_encoded": m.get("attributes", {}).get("dtype") == "bool",
        }

    with open(os.path.join(array_path, ".zarray")) as fh:
        m = json.load(fh)
    with open(os.path.join(array_path, ".zattrs")) as fh:
        attrs = json.load(fh)
    dims = attrs["_ARRAY_DIMENSIONS"]
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
        "fill_value": _decode_fill_value(
            m.get("fill_value"), np.dtype(m["dtype"]), zarr_format=2
        ),
        "bool_encoded": attrs.get("dtype") == "bool",
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


def _read_shard_index(fd, file_size, meta, inner_per_shard, path=""):
    """Read just a shard's inner-chunk byte index (its small header/footer).

    Returns a flat tuple of ``(offset, nbytes)`` pairs in C-order over the
    inner-chunk grid and the number of index bytes read.  Only the index bytes
    are read -- not the chunk data -- so a task that wants one channel does not
    pull the whole shard off disk.

    A file smaller than its index, or a short ``pread`` of the index (Lustre
    can return fewer bytes than requested WITHOUT raising during an eviction
    window; both are also the signature of a truncated shard), raises
    ``OSError(EIO)`` so :func:`_retry_transient_io` retries the whole shard
    with a fresh descriptor; persistent truncation exhausts the schedule and
    surfaces naming the file.
    """
    sh = meta["sharding"]
    n_inner = int(np.prod(inner_per_shard))
    has_crc = any(c["name"] == "crc32c" for c in sh.get("index_codecs", []))
    index_nbytes = n_inner * 16 + (4 if has_crc else 0)
    if file_size < index_nbytes:
        raise OSError(
            errno.EIO,
            f"shard {path or fd} smaller than its index "
            f"({file_size} < {index_nbytes} bytes)",
        )
    if sh.get("index_location", "end") == "start":
        index_blob = os.pread(fd, index_nbytes, 0)
    else:
        index_blob = os.pread(fd, index_nbytes, file_size - index_nbytes)
    if len(index_blob) != index_nbytes:
        raise OSError(
            errno.EIO,
            f"short read of shard index of {path or fd} "
            f"({len(index_blob)}/{index_nbytes} bytes)",
        )
    if has_crc:
        index_blob = index_blob[:-4]
    return struct.unpack("<" + "Q" * (2 * n_inner), index_blob), index_nbytes


def _read_inner_chunk(fd, offsets, flat, inner_shape, sharding_cfg, dtype, path=""):
    """``pread`` + decode one inner chunk (``flat`` C-order index) from a shard.

    Reads only that inner chunk's byte range from ``fd``; returns ``None`` for
    an empty (all-fill) inner chunk so the caller can fill it.  ``os.pread`` is
    positional -- it does not use or move the shared file offset -- so several
    threads may read disjoint inner chunks from the same ``fd`` concurrently.

    A short ``pread`` (Lustre can return fewer bytes than requested WITHOUT
    raising during an eviction window; an index entry pointing past EOF reads
    short too) and a blob that fails to decode/reshape both raise
    ``OSError(EIO)`` naming the shard and inner chunk, so
    :func:`_retry_transient_io` retries the whole shard with a fresh
    descriptor. Genuine on-disk corruption fails every attempt and surfaces as
    that labeled error -- previously it escaped as a bare, unattributable,
    never-retried ``Zstd decompression error: invalid input data`` (jobs
    7861335 + 2026-07-20).
    """
    off = offsets[2 * flat]
    nbytes = offsets[2 * flat + 1]
    if off == _UINT64_MAX or nbytes == _UINT64_MAX:
        return None  # empty inner chunk -> caller fills
    blob = os.pread(fd, nbytes, off)
    if len(blob) != nbytes:
        raise OSError(
            errno.EIO,
            f"short read of inner chunk {flat} of {path or fd} "
            f"({len(blob)}/{nbytes} bytes at offset {off})",
        )
    after = [c for c in sharding_cfg["codecs"] if c["name"] != "bytes"]
    try:
        buf = _decode_bytes_bytes(blob, after)
        return np.frombuffer(buf, dtype=dtype).reshape(inner_shape)
    except Exception as exc:
        raise OSError(
            errno.EIO,
            f"decode of inner chunk {flat} of {path or fd} failed "
            f"(offset {off}, {nbytes} bytes): {exc!r}",
        ) from exc


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


def read_array_region(
    array_path, sel, chunk_mask=None, counters=None, out=None, concurrency=None
):
    """Read ``array[sel]`` for one Zarr array via direct chunk-blob reads.

    Handles Zarr v3 (sharded and plain) and v2; supports selections spanning
    several chunks / inner chunks (e.g. multiple frequency channels). The read
    units are the inner chunks of a sharded array and the chunks otherwise;
    a shard is opened, and its index read, once per call and only when it
    holds a unit to read.

    Every filesystem unit (metadata, one shard's index + inner chunks, one
    plain chunk file) runs under :func:`_retry_transient_io`, so a Lustre
    client eviction mid-read is retried with a fresh descriptor instead of
    failing the task (the read-side counterpart of the sharded writer's EIO
    retry; job 7856457 died on exactly this).

    Parameters
    ----------
    array_path : str
        Path of the Zarr array.
    sel : dict
        Maps dimension names to slices or one-dimensional integer arrays;
        missing dims select the full axis. Integer arrays are read through
        their contiguous bounding range and reordered in memory.
    chunk_mask : numpy.ndarray of bool, optional
        Broadcast against the grid of the read units that the bounding range
        of ``sel`` overlaps, one axis per dimension. Units where it is False
        are not read and hold the array's fill value. None reads every unit.
    counters : dict, optional
        Accumulates ``bytes`` (stored bytes of the units read),
        ``index_bytes``, ``units_read`` (units not skipped, absent or empty
        ones included), ``units_skipped`` and ``shard_opens``. Concurrent
        calls need separate dicts.
    out : numpy.ndarray, optional
        Array of the selection's shape and the dtype of the values (boolean
        for an array stored with xarray's encoding of booleans); it receives
        the values and is returned.
    concurrency : int, optional
        Inner chunks of a shard read at the same time; 1 reads them one at a
        time. None is ``_READ_CONCURRENCY``.

    Returns
    -------
    values : numpy.ndarray
        ``array[sel]`` (``out`` when given).
    dims : list of str
        The array's dimension names.

    Raises
    ------
    TypeError
        If a selection is neither a slice nor a one-dimensional integer
        array, or ``chunk_mask`` is not boolean.
    ValueError
        If a selection is empty, ``chunk_mask`` does not broadcast against
        the unit grid, or ``out`` has another shape or dtype.
    IndexError
        If an integer selection is out of bounds.
    """
    meta = _retry_transient_io(
        f"metadata of {array_path}", lambda: _read_array_meta(array_path)
    )
    shape, dims, dtype = meta["shape"], meta["dims"], meta["dtype"]
    oc = meta["outer_chunks"]
    nd = len(shape)

    ranges = []
    relative_indices = {}
    for ax, dim in enumerate(dims):
        selection = sel.get(dim, slice(None))
        if isinstance(selection, slice):
            a, b, step = selection.indices(shape[ax])
            if step != 1:
                indices = np.arange(a, b, step, dtype=np.int64)
                if indices.size == 0:
                    raise ValueError(f"Empty selection for dimension {dim!r}.")
                a, b = int(indices.min()), int(indices.max()) + 1
                relative_indices[ax] = indices - a
        else:
            indices = np.asarray(selection)
            if indices.ndim != 1 or indices.dtype.kind not in "iu":
                raise TypeError(
                    f"Selection for dimension {dim!r} must be a slice or a "
                    "one-dimensional integer array."
                )
            indices = indices.astype(np.int64, copy=True)
            indices[indices < 0] += shape[ax]
            if indices.size == 0:
                raise ValueError(f"Empty selection for dimension {dim!r}.")
            if np.any((indices < 0) | (indices >= shape[ax])):
                raise IndexError(f"Selection for dimension {dim!r} is out of bounds.")
            a, b = int(indices.min()), int(indices.max()) + 1
            relative_indices[ax] = indices - a
        ranges.append((a, b))
    bounds = tuple(b - a for a, b in ranges)
    # An array stored as integers with the attribute dtype "bool" (xarray's
    # encoding of booleans) is read as booleans, as xarray decodes it.
    values_dtype = np.dtype(bool) if meta["bool_encoded"] else dtype
    if out is not None:
        selection_shape = tuple(
            relative_indices[ax].size if ax in relative_indices else n
            for ax, n in enumerate(bounds)
        )
        if out.shape != selection_shape or out.dtype != values_dtype:
            raise ValueError(
                f"out has shape {out.shape} and dtype {out.dtype}; the selection "
                f"of {array_path} has shape {selection_shape} and dtype "
                f"{values_dtype}."
            )
    buffer = (
        out
        if out is not None and not relative_indices
        else np.empty(bounds, dtype=values_dtype)
    )
    fill = _fill_value(meta)

    sharding = meta["sharding"]
    unit_shape = tuple(sharding["chunk_shape"]) if sharding else oc
    per_axis = [
        _axis_overlaps(ranges[ax][0], ranges[ax][1], unit_shape[ax]) for ax in range(nd)
    ]
    grid = tuple(len(overlaps) for overlaps in per_axis)
    if chunk_mask is not None:
        chunk_mask = np.asarray(chunk_mask)
        if chunk_mask.dtype != bool:
            raise TypeError(f"chunk_mask must be boolean, not {chunk_mask.dtype}.")
        chunk_mask = np.broadcast_to(chunk_mask, grid)
    units = []
    n_skipped = 0
    for position in np.ndindex(grid):
        combo = [per_axis[ax][i] for ax, i in enumerate(position)]
        out_sl = tuple(slice(c[3], c[4]) for c in combo)
        if chunk_mask is None or chunk_mask[position]:
            in_sl = tuple(slice(c[1], c[2]) for c in combo)
            units.append((tuple(c[0] for c in combo), out_sl, in_sl))
        else:
            buffer[out_sl] = fill
            n_skipped += 1
    tally = {
        "bytes": 0,
        "index_bytes": 0,
        "units_read": len(units),
        "units_skipped": n_skipped,
        "shard_opens": 0,
    }

    def _finish():
        if counters is not None:
            for key, value in tally.items():
                counters[key] = counters.get(key, 0) + value
        values = buffer
        for axis, indices in relative_indices.items():
            values = np.take(values, indices, axis=axis)
        if out is None or values is out:
            return values, dims
        out[...] = values
        return out, dims

    if sharding is not None:
        inner_per_shard = tuple(oc[ax] // unit_shape[ax] for ax in range(nd))
        # Group the needed inner chunks by their shard so each shard file is
        # opened and its index read exactly once.  A single shard can span the
        # whole array (one inner chunk per channel), so reading just the wanted
        # inner-chunk byte ranges -- rather than the whole shard file -- avoids
        # over-reading by the channel count.
        by_shard = {}
        for gi, out_sl, in_sl in units:  # gi: global inner-chunk index
            shard_index = tuple(gi[ax] // inner_per_shard[ax] for ax in range(nd))
            within = tuple(gi[ax] % inner_per_shard[ax] for ax in range(nd))
            by_shard.setdefault(shard_index, []).append((within, out_sl, in_sl))

        from concurrent.futures import ThreadPoolExecutor

        # chunk_mask can leave no shard to read.
        n_threads = min(
            _READ_CONCURRENCY if concurrency is None else concurrency,
            max((len(i) for i in by_shard.values()), default=0),
        )
        executor = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None
        for shard_index, items in by_shard.items():
            path = _chunk_path(array_path, meta, shard_index)

            # The whole open -> index -> pread unit is one retryable action so
            # every attempt gets a fresh descriptor. A truly absent shard
            # (never written -> all fill) is detected via the open's ENOENT,
            # NOT an os.path.exists pre-check: exists() reports False on ANY
            # error, so during an eviction window it would silently turn real
            # data into fill values.
            def _read_shard(path=path, items=items):
                try:
                    fd = os.open(path, os.O_RDONLY)
                except FileNotFoundError:
                    for _within, out_sl, _in_sl in items:
                        buffer[out_sl] = fill
                    return None
                try:
                    offsets, index_nbytes = _read_shard_index(
                        fd, os.fstat(fd).st_size, meta, inner_per_shard, path
                    )

                    def _read_item(item):
                        """Read one inner chunk; returns its stored bytes."""
                        within, out_sl, in_sl = item
                        flat = int(np.ravel_multi_index(within, inner_per_shard))
                        arr = _read_inner_chunk(
                            fd, offsets, flat, unit_shape, sharding, dtype, path
                        )
                        if arr is None:
                            buffer[out_sl] = fill
                            return 0
                        buffer[out_sl] = arr[in_sl]
                        return offsets[2 * flat + 1]

                    if executor is None or len(items) == 1:
                        nbytes = sum(_read_item(item) for item in items)
                    else:
                        nbytes = sum(executor.map(_read_item, items))
                    return index_nbytes, nbytes
                finally:
                    os.close(fd)

            try:
                # Tallied after the retried unit succeeds, so a retry counts once.
                read = _retry_transient_io(f"shard {path}", _read_shard)
            except BaseException:
                if executor is not None:
                    executor.shutdown(wait=True)
                raise
            if read is not None:
                tally["shard_opens"] += 1
                tally["index_bytes"] += read[0]
                tally["bytes"] += read[1]
        if executor is not None:
            executor.shutdown(wait=True)
        return _finish()

    for ci, out_sl, in_sl in units:
        path = _chunk_path(array_path, meta, ci)

        # Absent chunk (never written) detected via ENOENT, not an exists()
        # pre-check -- see the sharded branch for why. The decode runs INSIDE
        # the retried unit as OSError(EIO), so a blob that fails to decode
        # (stale/short page content during an eviction window, or real
        # corruption) is re-read fresh and, if persistent, surfaces naming the
        # chunk file instead of as a bare decompression error.
        def _read_chunk(path=path):
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
            except FileNotFoundError:
                return None
            try:
                return _decode_unit(raw, oc, dtype, meta), len(raw)
            except Exception as exc:
                raise OSError(
                    errno.EIO,
                    f"decode of chunk {path} failed ({len(raw)} bytes): {exc!r}",
                ) from exc

        read = _retry_transient_io(f"chunk {path}", _read_chunk)
        if read is None:
            buffer[out_sl] = fill
            continue
        buffer[out_sl] = read[0][in_sl]
        tally["bytes"] += read[1]
        # Freed before the next chunk is read and decoded.
        read = None
    return _finish()


class _ChunkArray(BackendArray):
    """Lazily indexed Zarr array whose reads go through :func:`read_array_region`."""

    def __init__(self, array_path, shape, dtype, dims):
        self.array_path = array_path
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.dims = tuple(dims)

    def __getitem__(self, key):
        from xarray.core import indexing

        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )

    def _getitem(self, key):
        sel, out_shape, squeeze = {}, [], []
        for axis, (dim, n, k) in enumerate(
            zip(self.dims, self.shape, key, strict=True)
        ):
            if isinstance(k, slice):
                size = len(range(*k.indices(n)))
            elif np.ndim(k) == 0:
                start = int(k) + n if int(k) < 0 else int(k)
                k, size = slice(start, start + 1), 1
                squeeze.append(axis)
            else:
                size = np.size(k)
            sel[dim] = k
            out_shape.append(size)
        if 0 in out_shape:
            values = np.empty(out_shape, dtype=self.dtype)
        else:
            values, _ = read_array_region(self.array_path, sel)
        return np.squeeze(values, axis=tuple(squeeze)) if squeeze else values


def open_image(image_store):
    """Open a Zarr image store lazily, its data variables read by byte offset.

    The dataset is the one ``xarray.open_zarr(image_store, chunks=None)``
    gives (coordinates, attributes, encoding), without dask; each data
    variable is read with :func:`read_array_region` when it is indexed or
    loaded. A variable that xarray decodes to another dtype keeps xarray's
    reader.

    Parameters
    ----------
    image_store : str
        Path of the Zarr image store.

    Returns
    -------
    xarray.Dataset
    """
    import xarray as xr
    from xarray.core import indexing

    xds = xr.open_zarr(image_store, chunks=None)
    for name, var in xds.data_vars.items():
        array_path = os.path.join(image_store, name)
        meta = _read_array_meta(array_path)
        if (np.dtype(bool) if meta["bool_encoded"] else meta["dtype"]) != var.dtype:
            continue
        data = indexing.LazilyIndexedArray(
            _ChunkArray(array_path, var.shape, var.dtype, var.dims)
        )
        xds[name] = xr.Variable(var.dims, data, var.attrs, var.encoding)
    return xds


def load_image(image_store, selection=None):
    """Select part of a Zarr image lazily, with XRADIO's image conventions.

    The counterpart of :func:`xradio.image.load_image` for a Zarr store:
    :func:`open_image`, the selection (an integer keeps its dimension), then
    XRADIO's upgrade of legacy attributes and canonical polarization order.

    Parameters
    ----------
    image_store : str
        Path of the Zarr image store.
    selection : dict, optional
        ``{dimension: int, slice or integer array}`` in the order of the store.

    Returns
    -------
    xarray.Dataset
    """
    from xradio.image._util.common import _to_canonical_polarization_order
    from xradio.image._util.legacy import upgrade_legacy_image_attrs

    selection = {
        dim: (
            slice(int(sel), int(sel) + 1 or None)
            if isinstance(sel, int | np.integer)
            else sel
        )
        for dim, sel in (selection or {}).items()
    }
    xds = open_image(image_store).isel(selection)
    return _to_canonical_polarization_order(upgrade_legacy_image_attrs(xds))


# ---------------------------------------------------------------------------
# Public chunk-engine load
# ---------------------------------------------------------------------------
def _read_arrays_concurrently(reads, processing_function_threads):
    """Read several Zarr arrays at once via :func:`read_array_region`.

    ``reads`` maps an arbitrary key to ``(array_path, sel)``; the return value
    maps the same keys to ``(ndarray, dims)``.  With ``processing_function_threads <= 1`` (or a
    single array) the reads run serially; otherwise each array is read on its
    own thread.  The reads touch disjoint files and return independent arrays,
    so there is nothing to synchronise.
    """
    items = list(reads.items())
    if processing_function_threads <= 1 or len(items) <= 1:
        return {key: read_array_region(path, sel) for key, (path, sel) in items}

    from concurrent.futures import ThreadPoolExecutor

    results = {}
    with ThreadPoolExecutor(
        max_workers=min(processing_function_threads, len(items))
    ) as executor:
        futures = {
            executor.submit(read_array_region, path, sel): key
            for key, (path, sel) in items
        }
        for future in futures:
            results[futures[future]] = future.result()
    return results


@shares_param_docs
def load_processing_set(
    input_data_store,
    sel_parms,
    data_group,
    processing_set_data_group_name,
    frequency_coords,
    instrument_polarization_basis="linear",
    processing_function_threads=1,
    load_correlated_data=True,
    *,
    map_frequencies,
    full_image_frequencies=None,
    frequency_boundaries=None,
    observation_frequency_boundaries=None,
):
    """Reconstruct a minimal processing set for cube imaging from chunk blobs.

    Only the data variables of ``data_group`` (correlated data, uvw, weight,
    flag) are read, straight from the Zarr chunk files for this task's frequency
    selection. Coordinates are reconstructed from the node-task inputs:

    * ``frequency`` from each measurement set's own on-disk ``frequency``
      coordinate (from ``frequency_coords`` for a single measurement set
      without one); every selected channel must lie within half a channel of
      a task frequency, onto the nearest of which it is imaged
      (``map_frequencies``); a task frequency may receive none;
    * ``polarization`` from ``instrument_polarization_basis`` and the data's
      correlation count;
    * ``time`` and ``baseline_id`` from the stored coordinates (when present);
      ``uvw_label`` as a plain index range.

    All sub-datasets are ignored. Returns an :class:`xarray.DataTree` whose
    children are the selected measurement sets, with ``attrs["data_groups"]``
    set so the science path is indistinguishable from a normal load.

    Parameters
    ----------
    input_data_store : str
        Path to the processing-set Zarr store.
    sel_parms : dict
        ``{ms_name: {dim: selector}}`` selection for this task (from the graph).
        The ``frequency`` selector is a slice or one-dimensional integer array,
        and an optional ``polarization`` list gives the index positions of the
        correlations to keep.
    data_group : dict
        Resolved role->variable mapping for ``processing_set_data_group_name``
        (e.g. ``{"correlated_data": "VISIBILITY", "uvw": "UVW", ...}``), passed
        down from the distributed graph.
    processing_set_data_group_name : str
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    frequency_coords : array-like
        Frequency values for this task's channels (``task_coords["frequency"]["data"]``).
    instrument_polarization_basis : str
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` or ``"circular"``. The residual update grids and degrids the
        correlations of this basis and the model update deconvolves in the
        Stokes basis, in which the image is written. The Stokes planes requested
        in ``image_params["polarization_coords"]`` fix the correlations that are
        loaded and gridded: the two parallel hands give ``I, Q`` (linear) or
        ``I, V`` (circular), all four correlations give ``I, Q, U, V``. A sample
        is used only if none of its loaded correlations is flagged.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    load_correlated_data : bool, optional
        If false, omit the observed correlated-data array. This is used by
        cached-grid MFS residual updates, which require only UVW, flags, and
        imaging weights to predict and grid the model contribution.
    frequency_boundaries : dict, optional
        Precomputed image-channel acceptance intervals (Hz), keyed by channel
        centre. Each pair gives the lowest and highest accepted float64
        frequency, both inclusive. Only task-local channel records are needed.
        When omitted, use the supplied image axis to infer channel widths.
    observation_frequency_boundaries : dict, optional
        Single-channel image boundaries keyed by observation name, computed
        from each observation's full frequency axis before selection.
    full_image_frequencies : array-like, optional
        Original, unpartitioned image-frequency axis (Hz), retained for legacy
        callers. Mapping and boundary checks use this axis, with observations
        required to belong to the local image-frequency subset. Precomputed
        ``frequency_boundaries`` take precedence when supplied.
    map_frequencies : callable
        ``map_frequencies(visibility_frequencies, task_frequencies)`` returns
        the index of the task frequency each selected channel is imaged onto
        and raises ``ValueError`` for a channel that maps onto none (for
        imaging,
        :func:`~astroviper.processing_functions.imaging.utils.frequency_mapping.map_visibility_frequencies_to_image`).
    """
    import xarray as xr

    task_frequency_values = np.asarray(frequency_coords)
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
        frequency_path = os.path.join(ms_path, "frequency")
        if os.path.isdir(frequency_path):
            reads["frequency_coordinate"] = (
                frequency_path,
                {"frequency": freq_sel},
            )
        if load_correlated_data:
            reads["correlated_data"] = (
                os.path.join(ms_path, data_group["correlated_data"]),
                {"frequency": freq_sel},
            )
        for coord_name in (
            "baseline_antenna1_name",
            "baseline_antenna2_name",
            "time",
            "baseline_id",
        ):
            if os.path.isdir(os.path.join(ms_path, coord_name)):
                reads[coord_name] = (os.path.join(ms_path, coord_name), {})

        results = _read_arrays_concurrently(reads, processing_function_threads)

        uvw, uvw_dims = results["uvw"]
        weight, w_dims = results["weight"]
        flag, f_dims = results["flag"]

        if "frequency_coordinate" in results:
            child_frequency_values, frequency_dims = results["frequency_coordinate"]
            if frequency_dims != ["frequency"]:
                raise ValueError(
                    f"Frequency coordinate for {ms_name!r} has dimensions "
                    f"{frequency_dims}; expected ['frequency']."
                )
            child_frequency_values = np.asarray(child_frequency_values)
        elif len(sel_parms) == 1:
            child_frequency_values = task_frequency_values
        else:
            raise KeyError(
                f"Processing-set child {ms_name!r} has no on-disk frequency "
                "coordinate, so a multi-child task cannot label it safely."
            )

        expected_frequency_size = weight.shape[w_dims.index("frequency")]
        if child_frequency_values.size != expected_frequency_size:
            raise ValueError(
                f"Frequency coordinate for {ms_name!r} has length "
                f"{child_frequency_values.size}; selected data have length "
                f"{expected_frequency_size}."
            )

        # Each selected channel maps onto its nearest task frequency within half
        # a channel, the rule the gridders and the graph's channel selection
        # use: execution blocks converted to LSRK have channel centres a few
        # kHz apart, so their frequencies never equal the image's exactly.
        try:
            mapping_context = (
                {"full_image_frequencies": full_image_frequencies}
                if full_image_frequencies is not None
                else {}
            )
            if frequency_boundaries is not None:
                mapping_context["frequency_boundaries"] = frequency_boundaries
            if observation_frequency_boundaries is not None:
                if ms_name not in observation_frequency_boundaries:
                    raise ValueError(
                        f"Missing frequency boundaries for observation {ms_name!r}."
                    )
                mapping_context["frequency_boundaries"] = (
                    observation_frequency_boundaries[ms_name]
                )
            map_frequencies(
                child_frequency_values, task_frequency_values, **mapping_context
            )
        except ValueError as exc:
            raise ValueError(
                f"Selected frequencies for {ms_name!r} do not map onto the task "
                f"frequencies: {exc}"
            ) from exc

        # Select the requested correlations even when cached-grid cycles omit
        # observed visibilities. Weights and flags are always loaded.
        pol_sel = ms_sel.get("polarization") if isinstance(ms_sel, dict) else None
        if pol_sel is not None:
            weight = np.take(weight, pol_sel, axis=w_dims.index("polarization"))
            flag = np.take(flag, pol_sel, axis=f_dims.index("polarization"))

        npol = weight.shape[w_dims.index("polarization")]
        pol_labels = _POL_LABELS[instrument_polarization_basis].get(
            npol, [f"P{i}" for i in range(npol)]
        )

        data_vars = {
            data_group["uvw"]: (uvw_dims, uvw),
            data_group["weight"]: (w_dims, weight),
            data_group["flag"]: (f_dims, flag),
        }
        if load_correlated_data:
            vis, vis_dims = results["correlated_data"]
            if pol_sel is not None:
                vis = np.take(vis, pol_sel, axis=vis_dims.index("polarization"))
            data_vars[data_group["correlated_data"]] = (vis_dims, vis)
        coords = {
            "frequency": ("frequency", child_frequency_values),
            "polarization": ("polarization", pol_labels),
        }
        # The per-baseline antenna names cannot be reconstructed from the imaging
        # inputs and were read above (when present) to drop auto-correlations;
        # the time and baseline_id values index the in-place weight cache.
        for coord_name in (
            "baseline_antenna1_name",
            "baseline_antenna2_name",
            "time",
            "baseline_id",
        ):
            if coord_name in results:
                values, c_dims = results[coord_name]
                coords[coord_name] = (c_dims, values)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        if observation_frequency_boundaries is not None:
            ds.frequency.attrs["image_frequency_boundaries"] = (
                observation_frequency_boundaries[ms_name]
            )
        ds.attrs["data_groups"] = {processing_set_data_group_name: dict(data_group)}
        nodes[ms_name] = ds

    return xr.DataTree.from_dict(nodes)


# ---------------------------------------------------------------------------
# Public chunk-engine write
# ---------------------------------------------------------------------------
def _encode_chunk_blob(chunk, meta):
    """Encode a full-chunk array to the on-disk blob for ``meta``'s codecs."""
    if meta["format"] == 3:
        b = np.ascontiguousarray(chunk, dtype=meta["dtype"])
        out = memoryview(b).cast("B")
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
                ).encode(b)  # ndarray -> correct typesize for shuffle
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


def _fill_value(meta):
    """Return the explicit fill or a dtype-specific compatibility default.

    Zarr v2 leaves missing data with a null fill undefined. Match zarr-python's
    deterministic defaults for reads, skipped chunks and edge padding: zero
    (including raw bytes), empty strings, or NaT for datetimes/timedeltas.
    Only a scalar is allocated; explicit fills are returned unchanged.
    """
    fill = meta["fill_value"]
    if fill is not None:
        return fill
    dtype = meta["dtype"]
    if dtype.kind in "Mm":
        return np.full((), "NaT", dtype=dtype)[()]
    return np.zeros((), dtype=dtype)[()]


def _task_chunk_grid(dims, shape, chunk_shape, task_coords, array_path):
    """Per-axis chunk overlaps of this task's write region with a chunk grid.

    The task's region is the ``task_coords`` slice on each parallel dim and the
    full axis on every other dim.  Returns the per-axis :func:`_axis_overlaps`
    lists over a grid of ``chunk_shape``-sized chunks -- one entry per chunk the
    region touches on that axis.  The region must be chunk-grid **aligned** on
    every sliced dim: each written chunk file / shard slot is wholly owned by
    one task, so a region starting or ending inside a chunk (other than the
    array-edge partial chunk, which is padded) would clobber a neighbouring
    task's data -- that is rejected here rather than corrupting the store.
    """
    per_axis = []
    for ax, dim in enumerate(dims):
        if dim in task_coords and isinstance(task_coords[dim].get("slice"), slice):
            sel = task_coords[dim]["slice"]
            start = sel.start or 0
            stop = sel.stop if sel.stop is not None else shape[ax]
        else:
            start, stop = 0, shape[ax]
        overlaps = _axis_overlaps(start, stop, chunk_shape[ax])
        for ci, lo, hi, _out_lo, _out_hi in overlaps:
            if lo != 0 or (
                hi != chunk_shape[ax] and ci * chunk_shape[ax] + hi != shape[ax]
            ):
                raise ValueError(
                    f"{array_path}: task region [{start}, {stop}) on dimension "
                    f"{dim!r} is not aligned to the on-disk chunk grid (chunk "
                    f"size {chunk_shape[ax]}); a direct chunk write would "
                    "clobber a neighbouring task's data."
                )
        per_axis.append(overlaps)
    return per_axis


def _encode_one_variable(dv, image_store, task_coords, img_xds):
    """Encode (compress) this task's region for one image variable to its
    on-disk chunk blob(s), WITHOUT writing them.

    The task's write region -- the ``task_coords`` slice on each parallel dim,
    the full axis elsewhere -- may cover **several** chunks when the store was
    created with ``image_chunking`` (e.g. l/m sub-chunking, or a
    frequency chunk finer than the per-task chunk); without it the region is
    exactly one chunk, as before.  Each covered chunk is encoded with the
    variable's on-disk codecs and returned as ``(path, blob)`` for
    ``<store>/<VAR>/c/<i0>/.../<iN>``; the region must be chunk-grid aligned
    (guarded in :func:`_task_chunk_grid`).  Partial (array-edge) chunks are
    padded to the full chunk shape with the array's fill value so the blob
    round-trips through Zarr.  Touches only this variable's own metadata, so it
    is safe to run for several variables concurrently.  Splitting encode from
    the disk write lets the CPU-bound compression run concurrently while the
    writes stay serial (see
    :func:`write_zarr`).
    """
    name = dv.upper()
    array_path = os.path.join(image_store, name)
    meta = _read_array_meta(array_path)
    oc = meta["outer_chunks"]
    dims = list(img_xds[name].dims)

    per_axis = _task_chunk_grid(dims, meta["shape"], oc, task_coords, array_path)
    values = np.asarray(img_xds[name].values)
    fill = _fill_value(meta)

    encoded = []
    for combo in product(*per_axis):
        chunk_index = tuple(c[0] for c in combo)
        piece = values[tuple(slice(c[3], c[4]) for c in combo)]
        if piece.shape != tuple(oc):
            chunk = np.full(oc, fill, dtype=meta["dtype"])
            chunk[tuple(slice(0, s) for s in piece.shape)] = piece
        else:
            chunk = piece
        blob = _encode_chunk_blob(chunk, meta)
        encoded.append((_chunk_path(array_path, meta, chunk_index), blob))
    return encoded


def _write_blob(path, blob):
    """Write one pre-encoded chunk blob to disk (create its parent dir first)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)


def _encode_and_write(variables, encode, write, processing_function_threads):
    """Encode and write ``variables`` in batches of ``processing_function_threads``.

    A batch is encoded concurrently and written serially before the next one
    is encoded, so a task holds the blobs of one batch at a time, not of every
    variable.
    """
    batch = max(1, int(processing_function_threads))
    if batch == 1 or len(variables) <= 1:
        for dv in variables:
            write(encode(dv))
        return
    from concurrent.futures import ThreadPoolExecutor

    def write_batch(executor, batch_variables):
        for encoded in executor.map(encode, batch_variables):
            write(encoded)

    with ThreadPoolExecutor(max_workers=min(batch, len(variables))) as executor:
        for start in range(0, len(variables), batch):
            write_batch(executor, variables[start : start + batch])


def write_zarr(
    image_store,
    image_data_variables_keep,
    task_coords,
    img_xds,
    processing_function_threads=1,
):
    """Write this task's image chunk(s) directly to their Zarr chunk file(s).

    The variables are taken in batches of ``processing_function_threads``,
    each in two phases, deliberately decoupled, so a task holds the blobs of
    one batch at a time:

    1. **Encode (compress)** each variable's chunk of the batch with its on-disk
       codecs. This is the dominant, CPU-bound, GIL-releasing cost and runs
       concurrently (one thread per variable).
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
    processing_function_threads : int, optional
        Maximum number of threads used to *encode/compress* the variables
        concurrently (the disk writes are always serial).  Default ``1`` (serial
        encode).
    """

    def write(encoded):
        for path, blob in encoded:
            _write_blob(path, blob)

    _encode_and_write(
        list(image_data_variables_keep),
        lambda dv: _encode_one_variable(dv, image_store, task_coords, img_xds),
        write,
        processing_function_threads,
    )


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
    out = memoryview(b).cast("B")
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
    """Encode this task's inner chunk(s) for one variable and resolve where each
    goes in its (pre-created) shard file.

    The task's write region may cover **several** inner chunks when the store
    was created with ``image_chunking`` (e.g. l/m sub-chunking, or a
    frequency inner chunk finer than the per-task chunk); without it the region
    is exactly one inner chunk, as before.  The region must be aligned to the
    inner-chunk grid (guarded in :func:`_task_chunk_grid`).  Returns a list of
    ``(shard_path, slot_offset, index_offset, blob)``. Does not write.
    """
    name = dv.upper()
    array_path = os.path.join(image_store, name)
    meta = _read_array_meta(array_path)
    _assert_sharded_no_index_crc(meta, array_path)
    inner = tuple(meta["sharding"]["chunk_shape"])
    dims = list(img_xds[name].dims)

    per_axis = _task_chunk_grid(dims, meta["shape"], inner, task_coords, array_path)
    values = np.asarray(img_xds[name].values)
    fill = _fill_value(meta)
    slot = _shard_slot_size(meta)
    ips, n_inner = _shard_inner_per_shard(meta)

    encoded = []
    for combo in product(*per_axis):
        # Global inner-chunk index of this piece of the region.
        chunk_index = tuple(c[0] for c in combo)
        piece = values[tuple(slice(c[3], c[4]) for c in combo)]
        if piece.shape != inner:
            chunk = np.full(inner, fill, dtype=meta["dtype"])
            chunk[tuple(slice(0, s) for s in piece.shape)] = piece
        else:
            chunk = piece

        blob = _encode_inner_chunk(chunk, meta)
        if len(blob) > slot:
            raise ValueError(
                f"{name}: encoded inner chunk is {len(blob)} B but the shard "
                f"slot is {slot} B. Increase the slot margin in "
                "_shard_slot_size."
            )
        shard_index = tuple(chunk_index[i] // ips[i] for i in range(len(chunk_index)))
        within = tuple(chunk_index[i] % ips[i] for i in range(len(chunk_index)))
        flat = int(np.ravel_multi_index(within, ips))
        shard_path = _chunk_path(array_path, meta, shard_index)
        encoded.append((shard_path, flat * slot, n_inner * slot + flat * 16, blob))
    return encoded


def write_zarr_sharded(
    image_store,
    image_data_variables_keep,
    task_coords,
    img_xds,
    processing_function_threads=1,
):
    """Sharded direct-write: write this task's inner chunk for each kept variable
    into its SHARED, pre-created shard file at a fixed disjoint slot, and set its
    Zarr v3 index entry.

    Many single-channel tasks write into one shard file concurrently (disjoint
    byte ranges -> lock-free, no file creation). Compression of the variables runs
    concurrently (``processing_function_threads``); the ``pwrite``\\ s are serial per task. The
    shard files must already exist (:func:`precreate_sharded_files`).

    Parameters mirror :func:`write_zarr`.
    """

    def write(encoded):
        # pwrite each inner chunk + its index entry into the shared shard file
        # (serial per task; disjoint offsets across tasks -> concurrency-safe).
        for shard_path, slot_offset, index_offset, blob in encoded:
            _pwrite_shard_slot_with_eio_retry(
                shard_path, slot_offset, index_offset, blob
            )

    _encode_and_write(
        list(image_data_variables_keep),
        lambda dv: _encode_one_variable_sharded(dv, image_store, task_coords, img_xds),
        write,
        processing_function_threads,
    )


# Transient-I/O retry schedule for the direct chunk reads and sharded pwrites:
# seconds to wait before each retry. A Lustre client eviction / OST hiccup
# fails ALL in-flight I/O of a client at once (observed 2026-07-12/13: sweep
# combo 23 and job 7855526, six tasks within 35 ms); the client usually
# reconnects within seconds, after which a fresh file descriptor works again.
# Worst case adds ~21 s to the task -- well under the 400 s task watchdog.
_EIO_RETRY_BACKOFF_SECONDS = (1.0, 5.0, 15.0)

# Errnos an eviction surfaces as: EIO is the classic one (the writes of job
# 7855526); ESHUTDOWN (108, "Cannot send after transport endpoint shutdown")
# is what in-flight preads got when the client was evicted mid-read (job
# 7856457); ENOTCONN is its close sibling. Everything else (ENOENT, EACCES,
# ...) is a real error and propagates immediately.
_TRANSIENT_IO_ERRNOS = frozenset({errno.EIO, errno.ESHUTDOWN, errno.ENOTCONN})


def _retry_transient_io(description, func):
    """Run ``func()`` -- one self-contained open->read->close unit -- retrying
    the transient Lustre errnos above with the shared backoff schedule.

    ``func`` MUST open its own fresh file descriptor on every call: after a
    client eviction the old descriptor stays poisoned, so the reopen is what
    makes the retry effective (same reason the sharded writer reopens).
    ``func`` must also be idempotent -- redoing the whole unit is safe for
    reads. Returns ``func()``'s value; raises the last transient error when
    the schedule is exhausted.
    """
    import time

    import toolviper.utils.logger as logger

    n_attempts = 1 + len(_EIO_RETRY_BACKOFF_SECONDS)
    last_exc = None
    for attempt, delay in enumerate((0.0,) + _EIO_RETRY_BACKOFF_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            result = func()
        except OSError as exc:
            if exc.errno not in _TRANSIENT_IO_ERRNOS:
                raise
            last_exc = exc
            logger.warning(
                f"chunk reader: transient "
                f"{errno.errorcode.get(exc.errno, exc.errno)} on {description} "
                f"(attempt {attempt + 1}/{n_attempts}); reopening and retrying."
            )
            continue
        if attempt:
            logger.warning(
                f"chunk reader: {description} succeeded on attempt "
                f"{attempt + 1} after transient I/O error."
            )
        return result
    raise last_exc


def _pwrite_shard_slot_with_eio_retry(shard_path, slot_offset, index_offset, blob):
    """Write one encoded inner chunk + its Zarr v3 index entry into the shared
    shard file, retrying transient ``EIO`` with a FRESH file descriptor.

    The reopen is essential: after a Lustre client eviction the old descriptor
    stays poisoned, so retrying on the same fd cannot succeed. Redoing both
    pwrites after a partial failure is safe -- same bytes at the same disjoint
    offsets (idempotent), and the index entry is written last, so a reader
    never sees an index pointing at an unwritten blob. The ``os.close`` is part
    of the retried unit too: an eviction can surface a failed writeback as EIO
    at close() after apparently-successful pwrites (observed: Frontera job
    7860369), and then the data may not be durable. Only the transient eviction
    errnos (``_TRANSIENT_IO_ERRNOS``) are retried; every other error propagates
    immediately.
    """
    import time

    import toolviper.utils.logger as logger

    last_exc = None
    for attempt, delay in enumerate((0.0,) + _EIO_RETRY_BACKOFF_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            fd = os.open(shard_path, os.O_WRONLY)  # pre-created; no O_CREAT
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"shard file {shard_path} not found; precreate_sharded_files must "
                "run in the driver before the sharded writer."
            ) from exc
        except OSError as exc:
            if exc.errno not in _TRANSIENT_IO_ERRNOS:
                raise
            last_exc = exc
            logger.warning(
                f"sharded writer: {errno.errorcode.get(exc.errno, exc.errno)} "
                f"opening {shard_path} (attempt "
                f"{attempt + 1}/{1 + len(_EIO_RETRY_BACKOFF_SECONDS)}); retrying."
            )
            continue
        try:
            os.pwrite(fd, blob, slot_offset)
            os.pwrite(fd, struct.pack("<QQ", slot_offset, len(blob)), index_offset)
        except OSError as exc:
            try:
                os.close(fd)
            except OSError:
                pass  # the write already failed; a close error adds nothing
            failure = exc
        else:
            try:
                # close() is part of the retried unit: during an eviction
                # window it can surface a failed writeback as EIO even though
                # the pwrites above "succeeded", in which case the data may not
                # be durable and the whole slot write must be redone. Linux
                # releases the fd even when close() errors, so no second close.
                os.close(fd)
            except OSError as exc:
                failure = exc
            else:
                if attempt:
                    logger.warning(
                        f"sharded writer: pwrite to {shard_path} succeeded on "
                        f"attempt {attempt + 1} after transient I/O error."
                    )
                return
        if failure.errno not in _TRANSIENT_IO_ERRNOS:
            raise failure
        last_exc = failure
        logger.warning(
            f"sharded writer: {errno.errorcode.get(failure.errno, failure.errno)} "
            f"writing slot at offset {slot_offset} of {shard_path} (attempt "
            f"{attempt + 1}/{1 + len(_EIO_RETRY_BACKOFF_SECONDS)}); "
            "reopening and retrying."
        )
    raise last_exc


# ---------------------------------------------------------------------------
# Shard-aware task scheduling priorities (spread concurrent writers over
# shards, and hence over OSTs) + shard -> OST placement diagnostic
# ---------------------------------------------------------------------------


def compute_shard_task_priorities(image_store, data_variable, node_task_data_mapping):
    """Per-task scheduling priorities that spread concurrently running tasks
    across shard files (higher priority = dispatched earlier).

    With the sharded writer, every task whose chunk falls in the same shard
    file writes to the same Lustre OST (shard files are created stripe-1).
    Dispatching tasks in plain task_id order concentrates all in-flight tasks
    on a handful of consecutive shards -- a few hot OSTs while the rest sit
    idle. These priorities order the tasks in "waves" instead: consecutive
    priorities step through EVERY shard once before revisiting one, so at any
    instant the in-flight tasks are spread over all shard files/OSTs:

        rank(task)     = flat(within-shard index) * n_shards + flat(shard index)
        priority(task) = -rank

    (Negated so the first wave keeps Dask's default priority 0 and task_id
    order breaks ties.) The shard layout is read from the variable's own
    on-disk Zarr v3 metadata, and the task -> inner-chunk mapping uses the same
    arithmetic as the sharded writer (``task_coords[dim]["slice"].start //
    inner_chunk`` on every array axis), so the priorities agree with where the
    writer will actually write for ANY combination of sharded dimensions --
    nothing here is specific to frequency.

    Parameters
    ----------
    image_store : str
        Path of the (already created) on-disk Zarr image store.
    data_variable : str
        Logical image-variable key (e.g. ``"sky_residual"``); upper-cased to
        the on-disk array name exactly like the writers. Its shard layout
        stands in for all kept variables (they share the same layout on the
        parallel dims).
    node_task_data_mapping : dict
        ``{task_id: {"task_coords": {dim: {"slice": slice, ...}}, ...}}`` as
        produced by graphviper's
        ``interpolate_data_coords_onto_parallel_coords``.

    Returns
    -------
    dict or None
        ``{task_id: int}`` suitable for graphviper's
        ``map(task_priorities=...)``, or None when the variable is not sharded
        (no reordering to do).
    """
    array_path = os.path.join(image_store, data_variable.upper())
    meta = _read_array_meta(array_path)
    if meta.get("sharding") is None:
        return None

    inner = meta["sharding"]["chunk_shape"]
    ips, _ = _shard_inner_per_shard(meta)  # inner chunks per shard, per axis
    dims = meta["dims"]
    n_shards_per_axis = tuple(
        -(-meta["shape"][ax] // meta["outer_chunks"][ax]) for ax in range(len(dims))
    )
    n_shards = int(np.prod(n_shards_per_axis))

    priorities = {}
    for task_id, task_params in node_task_data_mapping.items():
        task_coords = task_params["task_coords"]
        shard_index, within = [], []
        for ax, dim in enumerate(dims):
            if dim in task_coords and isinstance(task_coords[dim].get("slice"), slice):
                start = task_coords[dim]["slice"].start or 0
                g = start // inner[ax]
            else:
                g = 0
            shard_index.append(g // ips[ax])
            within.append(g % ips[ax])
        flat_within = int(np.ravel_multi_index(within, ips))
        flat_shard = int(np.ravel_multi_index(shard_index, n_shards_per_axis))
        priorities[task_id] = -(flat_within * n_shards + flat_shard)
    return priorities


def _parse_lfs_getstripe(text, paths):
    """Parse ``lfs getstripe <file> ...`` output into
    ``{path: {"stripe_count": int or None, "ost_indices": [int, ...]}}``.

    Handles the classic per-file layout report: the file's path on its own
    line, ``lmm_stripe_count:  N``, then an ``obdidx objid objid group`` table
    with one indented row per stripe whose first token is the OST index. Only
    lines matching a requested path start a new file block; unrecognized lines
    are ignored, so layout/version variations degrade to partial info rather
    than a parse error.
    """
    wanted = set(paths)
    result = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if line in wanted:
            current = {"stripe_count": None, "ost_indices": []}
            result[line] = current
            continue
        if current is None or not line:
            continue
        if line.startswith("lmm_stripe_count:"):
            # PFL/composite layouts print one lmm_stripe_count PER component;
            # keep the first (the instantiated head component), not the last.
            if current["stripe_count"] is None:
                try:
                    current["stripe_count"] = int(line.split(":", 1)[1])
                except ValueError:
                    pass
        elif raw[:1].isspace():
            tokens = line.split()
            # obdidx table row: indented, first token is the (integer) OST index.
            if len(tokens) >= 2 and tokens[0].isdigit():
                current["ost_indices"].append(int(tokens[0]))
            elif "l_ost_idx:" in tokens:
                # PFL/composite object row: '- 0: { l_ost_idx: 5, l_fid: ... }'.
                nxt = tokens.index("l_ost_idx:") + 1
                if nxt < len(tokens):
                    try:
                        current["ost_indices"].append(int(tokens[nxt].rstrip(",}")))
                    except ValueError:
                        pass
    return result


def record_shard_ost_map(zarr_store, array_names, out_path=None):
    """Record which Lustre OST holds each shard file of the given sharded
    arrays and save the table as its own feather DataFrame.

    Run in the driver right after :func:`precreate_sharded_files`: with
    stripe-count 1 every shard file lives on exactly ONE OST, chosen by
    Lustre's weighted round-robin allocator with NO distinctness guarantee --
    two shards on one OST halve that OST's share of the write bandwidth. This
    table is the instrument for checking the actual placement and for joining
    per-task write times to OSTs in the benchmark analysis (task chunk ->
    shard -> OST).

    Best-effort by design: on hosts without Lustre (no ``lfs`` on PATH) or
    when ``lfs getstripe`` yields nothing usable, it logs and returns None
    without writing anything.

    Parameters
    ----------
    zarr_store : str
        Path of the image store holding the arrays.
    array_names : list of str
        On-disk array names (e.g. ``["SKY_RESIDUAL", "PRIMARY_BEAM"]``) whose
        shard files to record; non-sharded arrays are skipped.
    out_path : str, optional
        Feather file to write; default ``<zarr_store>_shard_osts.ft`` (a
        SIBLING of the store, so readers of the store never encounter a
        foreign file inside it, and the benchmark harness can find it from
        the store path alone).

    Returns
    -------
    pandas.DataFrame or None
        Columns ``variable``, ``shard_file`` (path relative to the store),
        ``shard_index`` (per-axis shard indices), ``ost_index`` (first/only
        OST, nullable), ``ost_indices`` (all OSTs of the file), and
        ``stripe_count``; or None when nothing could be recorded.
    """
    import shutil
    import subprocess

    import pandas as pd
    import toolviper.utils.logger as logger

    if shutil.which("lfs") is None:
        logger.info(
            "record_shard_ost_map: 'lfs' not found (not a Lustre filesystem?); "
            "skipping shard->OST recording."
        )
        return None

    rows, abs_paths = [], []
    for name in array_names:
        array_path = os.path.join(zarr_store, name)
        meta = _read_array_meta(array_path)
        if meta.get("sharding") is None:
            continue
        n_shards = tuple(
            -(-meta["shape"][ax] // meta["outer_chunks"][ax])
            for ax in range(len(meta["shape"]))
        )
        for shard_index in product(*[range(n) for n in n_shards]):
            path = _chunk_path(array_path, meta, shard_index)
            abs_paths.append(path)
            rows.append(
                {
                    "variable": name,
                    "shard_file": os.path.relpath(path, zarr_store),
                    "shard_index": list(shard_index),
                }
            )
    if not rows:
        logger.info("record_shard_ost_map: no sharded arrays found; nothing to record.")
        return None

    try:
        proc = subprocess.run(
            ["lfs", "getstripe"] + abs_paths,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        logger.warning(f"record_shard_ost_map: lfs getstripe failed ({exc}); skipping.")
        return None
    # getstripe exits nonzero if ANY path failed; still parse what it printed.
    stripe_info = _parse_lfs_getstripe(proc.stdout, abs_paths)
    if not stripe_info:
        logger.warning(
            "record_shard_ost_map: could not parse any layout from lfs getstripe "
            f"(rc={proc.returncode}); skipping."
        )
        return None

    for row, path in zip(rows, abs_paths, strict=False):
        info = stripe_info.get(path, {})
        osts = info.get("ost_indices", [])
        row["ost_index"] = osts[0] if osts else None
        row["ost_indices"] = osts
        row["stripe_count"] = info.get("stripe_count")

    df = pd.DataFrame(rows)
    df["ost_index"] = df["ost_index"].astype("Int64")
    df["stripe_count"] = df["stripe_count"].astype("Int64")

    ost_counts = df["ost_index"].value_counts()
    if len(ost_counts):
        logger.info(
            f"record_shard_ost_map: {len(df)} shard files on "
            f"{len(ost_counts)} distinct OSTs (most-loaded OST holds "
            f"{int(ost_counts.iloc[0])} shard files)."
        )

    if out_path is None:
        out_path = zarr_store.rstrip("/") + "_shard_osts.ft"
    try:
        df.to_feather(out_path)
        logger.info(f"record_shard_ost_map: wrote shard->OST table to {out_path}")
    except Exception as exc:
        logger.warning(f"record_shard_ost_map: could not write {out_path} ({exc}).")
    return df
