"""Unit tests for the ``_skunk_works`` performance I/O variants.

Builds tiny synthetic Zarr stores (v3 sharded, v3 plain, v2, v3 fixed-length
strings) and checks that the low-level reader matches ``zarr.open_array`` for
multi-channel selections, that the direct chunk writer round-trips through Zarr
(uncompressed, Blosc, and partial/edge chunks), and that
``load_processing_set_skunk_works`` reconstructs a usable, mutable processing
set from the chunk blobs alone.
"""

from __future__ import annotations

import errno
import os
import time

import numpy as np
import pytest
import xarray as xr
import zarr

from astroviper.node_tasks.imaging.utils import (
    load_processing_set_skunk_works,
    read_array_region,
    write_result_chunk_to_disk_sharded_skunk_works,
    write_result_chunk_to_disk_using_zarr_skunk_works,
)
from astroviper.utils.io import create_empty_data_variables_on_disk

DIMS = ("time", "baseline_id", "frequency", "polarization")


def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "fc":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Low-level reader
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fsel", [slice(0, 1), slice(1, 4), slice(0, 4), slice(2, 4)])
def test_read_v3_sharded_multichannel(tmp_path, fsel):
    """Sharded array, inner chunk = 1 channel: reader == zarr for any slice."""
    shape = (3, 5, 4, 2)
    data = (np.random.default_rng(0).standard_normal(shape)).astype("<c8")
    path = str(tmp_path / "VIS")
    arr = zarr.create_array(
        path,
        shape=shape,
        chunks=(3, 5, 1, 2),  # inner (read) chunk: one channel
        shards=(3, 5, 4, 2),  # one shard for the whole array
        dtype="complex64",
        dimension_names=DIMS,
        compressors=zarr.codecs.ZstdCodec(),
    )
    arr[:] = data
    got, dims = read_array_region(path, {"frequency": fsel})
    assert dims == list(DIMS)
    assert _equal(got, zarr.open_array(path)[:, :, fsel, :])


def test_read_v3_plain(tmp_path):
    shape = (3, 5, 3)
    data = np.random.default_rng(1).standard_normal(shape).astype("<f8")
    path = str(tmp_path / "UVW")
    arr = zarr.create_array(
        path,
        shape=shape,
        chunks=shape,
        dtype="float64",
        dimension_names=("time", "baseline_id", "uvw_label"),
        compressors=zarr.codecs.ZstdCodec(),
    )
    arr[:] = data
    got, _ = read_array_region(path, {})
    assert _equal(got, data)


@pytest.mark.parametrize("fsel", [slice(0, 2), slice(1, 4)])
def test_read_v3_multi_shard(tmp_path, fsel):
    """Several shards along frequency, inner chunk = 1 channel."""
    shape = (2, 4, 4, 2)
    data = np.random.default_rng(2).standard_normal(shape).astype("<f4")
    path = str(tmp_path / "W")
    arr = zarr.create_array(
        path,
        shape=shape,
        chunks=(2, 4, 1, 2),
        shards=(2, 4, 2, 2),  # 2 shards of 2 channels each
        dtype="float32",
        dimension_names=DIMS,
    )
    arr[:] = data
    got, _ = read_array_region(path, {"frequency": fsel})
    assert _equal(got, zarr.open_array(path)[:, :, fsel, :])


def test_read_v3_fixed_length_utf32(tmp_path):
    names = np.array(
        ["DA42_W203", "DV01_A001", "PM03_T704", "DA50", "DV22"], dtype="<U9"
    )
    path = str(tmp_path / "baseline_antenna1_name")
    arr = zarr.create_array(
        path,
        shape=names.shape,
        chunks=names.shape,
        dtype=names.dtype,
        dimension_names=("baseline_id",),
        compressors=zarr.codecs.ZstdCodec(),
    )
    arr[:] = names
    got, dims = read_array_region(path, {})
    assert dims == ["baseline_id"]
    assert list(got) == list(names)


def test_read_v2_multichannel(tmp_path):
    pytest.importorskip("zarr")
    shape = (3, 5, 4, 2)
    data = np.random.default_rng(3).standard_normal(shape).astype("<f4")
    data[0, 0, 0, 0] = np.nan  # ensure NaN handling is exercised
    path = str(tmp_path / "v2arr")
    z = zarr.open(
        path,
        mode="w",
        shape=shape,
        chunks=shape,
        dtype="<f4",
        zarr_format=2,
        dimension_separator=".",
    )
    z[:] = data
    # zarr v2 needs _ARRAY_DIMENSIONS for the reader to map dims.
    z.attrs["_ARRAY_DIMENSIONS"] = list(DIMS)
    got, dims = read_array_region(path, {"frequency": slice(1, 3)})
    assert dims == list(DIMS)
    assert _equal(got, data[:, :, 1:3, :])


# --------------------------------------------------------------------------- #
# Transient-I/O (Lustre client eviction) retry in the reader
# --------------------------------------------------------------------------- #
def _sharded_store(tmp_path):
    shape = (3, 5, 4, 2)
    data = np.random.default_rng(7).standard_normal(shape).astype("<f4")
    path = str(tmp_path / "EV")
    arr = zarr.create_array(
        path,
        shape=shape,
        chunks=(3, 5, 1, 2),
        shards=(3, 5, 4, 2),
        dtype="float32",
        dimension_names=DIMS,
        compressors=zarr.codecs.ZstdCodec(),
    )
    arr[:] = data
    return path, data


def test_read_retries_transient_eviction(tmp_path, monkeypatch):
    """First preads fail like an in-flight read on an evicted Lustre client
    (ESHUTDOWN -- the job 7856457 failure mode); the reader must reopen,
    retry, and still return the correct data."""
    path, data = _sharded_store(tmp_path)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    real_pread = os.pread
    fails = {"n": 2}

    def flaky_pread(fd, n, off):
        if fails["n"] > 0:
            fails["n"] -= 1
            raise OSError(
                errno.ESHUTDOWN, "Cannot send after transport endpoint shutdown"
            )
        return real_pread(fd, n, off)

    monkeypatch.setattr(os, "pread", flaky_pread)
    got, _ = read_array_region(path, {"frequency": slice(0, 4)})
    assert fails["n"] == 0
    assert _equal(got, data)


def test_read_retry_exhaustion_raises(tmp_path, monkeypatch):
    """A persistent EIO (client never reconnects) exhausts the schedule and
    surfaces the original error."""
    path, _ = _sharded_store(tmp_path)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(
        os,
        "pread",
        lambda fd, n, off: (_ for _ in ()).throw(OSError(errno.EIO, "I/O error")),
    )
    with pytest.raises(OSError) as excinfo:
        read_array_region(path, {"frequency": slice(0, 4)})
    assert excinfo.value.errno == errno.EIO


def test_read_non_transient_errno_propagates_immediately(tmp_path, monkeypatch):
    """Non-eviction errnos (here EBADF) must not be retried or slept on."""
    path, _ = _sharded_store(tmp_path)
    calls = {"n": 0}

    def bad_pread(fd, n, off):
        calls["n"] += 1
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(os, "pread", bad_pread)
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    with pytest.raises(OSError):
        read_array_region(path, {"frequency": slice(0, 4)})
    assert calls["n"] == 1 and not slept


def test_read_missing_shard_still_fills(tmp_path):
    """Absent chunk blobs are now detected via the open's ENOENT (not an
    os.path.exists pre-check); a never-written shard must still read as the
    fill value."""
    path, _ = _sharded_store(tmp_path)
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f != "zarr.json":
                os.remove(os.path.join(root, f))
    got, _ = read_array_region(path, {"frequency": slice(0, 4)})
    assert _equal(got, np.zeros_like(got))


# --------------------------------------------------------------------------- #
# Direct chunk writer
# --------------------------------------------------------------------------- #
def _make_image_store(
    tmp_path, compressor, freq_chunks, variables=("sky_residual",), shard_channels=None
):
    store = str(tmp_path / "img.zarr")
    zarr.open_group(store, mode="w")
    shape_dict = {
        "time": 1,
        "frequency": sum(len(c) for c in freq_chunks),
        "polarization": 2,
        "l": 16,
        "m": 16,
    }
    parallel_coords = {"frequency": {"data_chunks": freq_chunks}}
    create_empty_data_variables_on_disk(
        store,
        list(variables),
        shape_dict=shape_dict,
        parallel_coords=parallel_coords,
        compressor=compressor,
        double_precision=False,
        data_variable_definitions="imaging",
        shard_channels=shard_channels,
    )
    return store


@pytest.mark.parametrize("processing_function_threads", [1, 4])
@pytest.mark.parametrize(
    "compressor",
    [None, zarr.codecs.BloscCodec(cname="lz4", clevel=5)],
    ids=["uncompressed", "blosc"],
)
def test_write_roundtrip(tmp_path, compressor, processing_function_threads):
    # 3 equal chunks of 2 channels; write three variables so the threaded
    # (per-variable) path is exercised.
    freq_chunks = [list(range(2)), list(range(2)), list(range(2))]
    variables = ["sky_residual", "sky_model", "point_spread_function"]
    store = _make_image_store(tmp_path, compressor, freq_chunks, variables)
    rng = np.random.default_rng(4)
    vals = {v: rng.standard_normal((1, 2, 2, 16, 16)).astype("<f4") for v in variables}
    img = xr.Dataset(
        {
            v.upper(): (["time", "frequency", "polarization", "l", "m"], vals[v])
            for v in variables
        }
    )
    task_coords = {"frequency": {"slice": slice(2, 4), "data": [0, 1]}}
    write_result_chunk_to_disk_using_zarr_skunk_works(
        store,
        variables,
        task_coords,
        img,
        processing_function_threads=processing_function_threads,
    )
    for v in variables:
        back = zarr.open_array(store + "/" + v.upper())[:, 2:4, :, :, :]
        assert _equal(back, vals[v])


def test_write_roundtrip_edge_chunk(tmp_path):
    """Last chunk is partial (5 channels, chunk size 2): pad to full chunk."""
    freq_chunks = [list(range(2)), list(range(2)), list(range(1))]
    store = _make_image_store(tmp_path, None, freq_chunks)
    vals = np.random.default_rng(5).standard_normal((1, 1, 2, 16, 16)).astype("<f4")
    img = xr.Dataset(
        {"SKY_RESIDUAL": (["time", "frequency", "polarization", "l", "m"], vals)}
    )
    task_coords = {"frequency": {"slice": slice(4, 5), "data": [0]}}
    write_result_chunk_to_disk_using_zarr_skunk_works(
        store, ["sky_residual"], task_coords, img
    )
    back = zarr.open_array(store + "/SKY_RESIDUAL")[:, 4:5, :, :, :]
    assert _equal(back, vals)


@pytest.mark.parametrize("processing_function_threads", [1, 4])
@pytest.mark.parametrize(
    "compressor",
    [None, zarr.codecs.BloscCodec(cname="lz4", clevel=5)],
    ids=["uncompressed", "blosc"],
)
def test_write_roundtrip_sharded(tmp_path, compressor, processing_function_threads):
    """Sharded writer: many single-channel tasks write concurrently into shared,
    pre-created shard files; read back correctly by stock zarr AND our reader;
    the last (partial) shard and an unwritten channel are handled; the file count
    is reduced (fewer shard files than channels)."""
    import glob
    from concurrent.futures import ThreadPoolExecutor

    nfreq, shard = 5, 2  # 5 channels, 2/shard -> ceil(5/2) = 3 shard files/variable
    freq_chunks = [[c] for c in range(nfreq)]  # 1 channel per imaging chunk (inner=1)
    variables = ["sky_residual", "point_spread_function"]
    store = _make_image_store(
        tmp_path, compressor, freq_chunks, variables, shard_channels=shard
    )
    rng = np.random.default_rng(11)
    vals = {
        v: rng.standard_normal((1, nfreq, 2, 16, 16)).astype("<f4") for v in variables
    }

    def _write(k):
        img = xr.Dataset(
            {
                v.upper(): (
                    ["time", "frequency", "polarization", "l", "m"],
                    vals[v][:, k : k + 1, :, :, :],
                )
                for v in variables
            }
        )
        write_result_chunk_to_disk_sharded_skunk_works(
            store,
            variables,
            {"frequency": {"slice": slice(k, k + 1)}},
            img,
            processing_function_threads=processing_function_threads,
        )

    # Write channels 0..3 concurrently; leave channel 4 UNWRITTEN.
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_write, range(nfreq - 1)))

    for v in variables:
        apath = store + "/" + v.upper()
        z = zarr.open_array(apath)[:]  # stock zarr reader
        ours, _ = read_array_region(apath, {})  # our reader
        for k in range(nfreq - 1):
            assert _equal(z[:, k], vals[v][:, k]) and _equal(ours[:, k], vals[v][:, k])
        # Unwritten channel 4 reads back as fill (nan) both ways.
        assert np.isnan(z[:, 4]).all() and np.isnan(ours[:, 4]).all()
        # Actually sharded: 3 shard files, not 5 one-file-per-channel.
        n_files = len(
            [
                f
                for f in glob.glob(apath + "/c/**", recursive=True)
                if __import__("os").path.isfile(f)
            ]
        )
        assert n_files == 3, f"{v}: expected 3 shard files, got {n_files}"


def test_sharded_slot_overflow_raises(tmp_path):
    """A compressed inner chunk that would exceed its fixed slot raises clearly
    rather than corrupting the shard (guarded in _encode_one_variable_sharded)."""
    from astroviper.node_tasks.imaging.utils import skunk_works as sw

    store = _make_image_store(
        tmp_path, None, [[0], [1]], ["sky_residual"], shard_channels=2
    )
    vals = np.random.default_rng(12).standard_normal((1, 1, 2, 16, 16)).astype("<f4")
    img = xr.Dataset(
        {"SKY_RESIDUAL": (["time", "frequency", "polarization", "l", "m"], vals)}
    )
    # Force a tiny slot so the (uncompressed) chunk cannot fit.
    orig = sw._shard_slot_size
    sw._shard_slot_size = lambda meta: 16
    try:
        with pytest.raises(ValueError, match="shard slot"):
            write_result_chunk_to_disk_sharded_skunk_works(
                store, ["sky_residual"], {"frequency": {"slice": slice(0, 1)}}, img
            )
    finally:
        sw._shard_slot_size = orig


# --------------------------------------------------------------------------- #
# Full processing-set reconstruction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("processing_function_threads", [1, 4])
def test_load_processing_set_reconstruction(tmp_path, processing_function_threads):
    ntime, nbl, nfreq, npol = 2, 6, 4, 2
    rng = np.random.default_rng(6)
    store = str(tmp_path / "ps.zarr")
    ms_name = "ms0"
    ms_path = f"{store}/{ms_name}"

    def _write(name, data, dims, **kw):
        kw.setdefault("chunks", data.shape)
        a = zarr.create_array(
            f"{ms_path}/{name}",
            shape=data.shape,
            dtype=data.dtype,
            dimension_names=dims,
            **kw,
        )
        a[:] = data

    vis = rng.standard_normal((ntime, nbl, nfreq, npol)).astype("<c8")
    uvw = rng.standard_normal((ntime, nbl, 3)).astype("<f8")
    weight = rng.standard_normal((ntime, nbl, nfreq, npol)).astype("<f4")
    flag = (rng.random((ntime, nbl, nfreq, npol)) > 0.5).astype("|i1")
    _write("VISIBILITY", vis, DIMS, shards=vis.shape, chunks=(ntime, nbl, 1, npol))
    _write("UVW", uvw, ("time", "baseline_id", "uvw_label"))
    _write("WEIGHT", weight, DIMS, shards=weight.shape, chunks=(ntime, nbl, 1, npol))
    _write("FLAG", flag, DIMS)
    ant1 = np.array(["A", "A", "B", "B", "C", "C"], dtype="<U4")
    ant2 = np.array(["A", "B", "B", "C", "C", "A"], dtype="<U4")
    _write("baseline_antenna1_name", ant1, ("baseline_id",))
    _write("baseline_antenna2_name", ant2, ("baseline_id",))

    data_group = {
        "correlated_data": "VISIBILITY",
        "uvw": "UVW",
        "weight": "WEIGHT",
        "flag": "FLAG",
    }
    freq_values = np.linspace(1.0e9, 1.3e9, nfreq)[1:3]  # task selects channels 1:3
    sel = {ms_name: {"frequency": slice(1, 3)}}

    ps = load_processing_set_skunk_works(
        store,
        sel,
        data_group,
        "base",
        freq_values,
        "linear",
        processing_function_threads=processing_function_threads,
    )
    ms = ps[ms_name]
    assert _equal(ms["VISIBILITY"].values, vis[:, :, 1:3, :])
    assert _equal(ms["UVW"].values, uvw)
    assert _equal(ms["WEIGHT"].values, weight[:, :, 1:3, :])
    assert _equal(ms["FLAG"].values, flag[:, :, 1:3, :])
    assert _equal(ms.frequency.values, freq_values)
    assert list(ms.polarization.values) == ["XX", "YY"]
    assert list(ms["baseline_antenna1_name"].values) == list(ant1)
    assert ms.attrs["data_groups"]["base"]["correlated_data"] == "VISIBILITY"

    # The science writes WEIGHT_IMAGING back; the reconstructed tree must allow it.
    ps[ms_name]["WEIGHT_IMAGING"] = xr.ones_like(ms["WEIGHT"])
    assert "WEIGHT_IMAGING" in ps[ms_name].ds.data_vars

    # drop_auto_correlations: antenna1 != antenna2 keeps the 3 cross-corrs.
    mask = ms["baseline_antenna1_name"] != ms["baseline_antenna2_name"]
    assert int(mask.sum()) == 3


# --------------------------------------------------------------------------- #
# Transient-EIO retry in the sharded pwrite path
# --------------------------------------------------------------------------- #
def _make_flaky_pwrite(monkeypatch, fail_first_n, errno_value):
    """Patch os.pwrite to raise OSError(errno_value) for the first N calls."""
    import errno as _errno  # noqa: F401
    import os

    real_pwrite = os.pwrite
    calls = {"n": 0}

    def flaky(fd, data, offset):
        calls["n"] += 1
        if calls["n"] <= fail_first_n:
            raise OSError(errno_value, "Input/output error")
        return real_pwrite(fd, data, offset)

    monkeypatch.setattr(os, "pwrite", flaky)
    return calls


def test_pwrite_eio_retry_recovers(tmp_path, monkeypatch):
    """A transient EIO burst (first two pwrite calls) heals on retry with a
    fresh fd, and both the blob and its index entry land correctly."""
    import errno
    import struct

    from astroviper.node_tasks.imaging.utils.skunk_works import (
        _pwrite_shard_slot_with_eio_retry,
    )

    monkeypatch.setattr("time.sleep", lambda s: None)  # no real backoff in tests
    shard = tmp_path / "c0"
    shard.write_bytes(b"\0" * 64)
    calls = _make_flaky_pwrite(monkeypatch, 2, errno.EIO)

    _pwrite_shard_slot_with_eio_retry(str(shard), 0, 32, b"payload!")

    raw = shard.read_bytes()
    assert raw[:8] == b"payload!"
    assert struct.unpack("<QQ", raw[32:48]) == (0, 8)
    assert calls["n"] > 2  # retried past the failures


def test_pwrite_eio_retry_exhausts_and_raises(tmp_path, monkeypatch):
    """Persistent EIO raises after the retry schedule is exhausted."""
    import errno

    from astroviper.node_tasks.imaging.utils.skunk_works import (
        _EIO_RETRY_BACKOFF_SECONDS,
        _pwrite_shard_slot_with_eio_retry,
    )

    monkeypatch.setattr("time.sleep", lambda s: None)
    shard = tmp_path / "c0"
    shard.write_bytes(b"\0" * 64)
    calls = _make_flaky_pwrite(monkeypatch, 10_000, errno.EIO)

    with pytest.raises(OSError) as excinfo:
        _pwrite_shard_slot_with_eio_retry(str(shard), 0, 32, b"payload!")
    assert excinfo.value.errno == errno.EIO
    # one pwrite attempt per open (the first of the pair fails each time)
    assert calls["n"] == 1 + len(_EIO_RETRY_BACKOFF_SECONDS)


def test_pwrite_non_eio_oserror_propagates_immediately(tmp_path, monkeypatch):
    """Only EIO is retried; e.g. ENOSPC must fail fast."""
    import errno

    from astroviper.node_tasks.imaging.utils.skunk_works import (
        _pwrite_shard_slot_with_eio_retry,
    )

    monkeypatch.setattr("time.sleep", lambda s: None)
    shard = tmp_path / "c0"
    shard.write_bytes(b"\0" * 64)
    calls = _make_flaky_pwrite(monkeypatch, 10_000, errno.ENOSPC)

    with pytest.raises(OSError) as excinfo:
        _pwrite_shard_slot_with_eio_retry(str(shard), 0, 32, b"payload!")
    assert excinfo.value.errno == errno.ENOSPC
    assert calls["n"] == 1  # no retry


def test_pwrite_missing_shard_keeps_actionable_error(tmp_path):
    from astroviper.node_tasks.imaging.utils.skunk_works import (
        _pwrite_shard_slot_with_eio_retry,
    )

    with pytest.raises(FileNotFoundError, match="precreate_sharded_files"):
        _pwrite_shard_slot_with_eio_retry(str(tmp_path / "missing"), 0, 32, b"x")


def _make_flaky_close(monkeypatch, fail_first_n, errno_value):
    """Patch os.close to ACTUALLY close the fd, then raise OSError for the
    first N calls — mirroring Linux semantics where a failing close() (e.g. a
    Lustre writeback EIO after an eviction) still releases the descriptor."""
    import os

    real_close = os.close
    calls = {"n": 0}

    def flaky(fd):
        calls["n"] += 1
        real_close(fd)
        if calls["n"] <= fail_first_n:
            raise OSError(errno_value, "Input/output error")

    monkeypatch.setattr(os, "close", flaky)
    return calls


def test_close_eio_after_successful_pwrites_retries(tmp_path, monkeypatch):
    """An eviction can surface failed writeback as EIO at close() even though
    both pwrites succeeded (Frontera job 7860369). The data may not be durable,
    so the slot write must be redone on a fresh descriptor — not swallowed, and
    not allowed to escape the retry loop."""
    import errno
    import struct

    from astroviper.node_tasks.imaging.utils.skunk_works import (
        _pwrite_shard_slot_with_eio_retry,
    )

    monkeypatch.setattr("time.sleep", lambda s: None)
    shard = tmp_path / "c0"
    shard.write_bytes(b"\0" * 64)
    calls = _make_flaky_close(monkeypatch, 1, errno.EIO)

    _pwrite_shard_slot_with_eio_retry(str(shard), 0, 32, b"payload!")

    raw = shard.read_bytes()
    assert raw[:8] == b"payload!"
    assert struct.unpack("<QQ", raw[32:48]) == (0, 8)
    assert calls["n"] == 2  # first close failed transiently; retry's close won


def test_close_non_transient_error_propagates_immediately(tmp_path, monkeypatch):
    """A non-eviction close error (e.g. EDQUOT) must fail fast, not retry."""
    import errno

    from astroviper.node_tasks.imaging.utils.skunk_works import (
        _pwrite_shard_slot_with_eio_retry,
    )

    monkeypatch.setattr("time.sleep", lambda s: None)
    shard = tmp_path / "c0"
    shard.write_bytes(b"\0" * 64)
    calls = _make_flaky_close(monkeypatch, 10_000, errno.EDQUOT)

    with pytest.raises(OSError) as excinfo:
        _pwrite_shard_slot_with_eio_retry(str(shard), 0, 32, b"payload!")
    assert excinfo.value.errno == errno.EDQUOT
    assert calls["n"] == 1  # no retry


# --------------------------------------------------------------------------- #
# Shard-aware task priorities
# --------------------------------------------------------------------------- #
def test_compute_shard_task_priorities_wave_order(tmp_path):
    """6 single-channel chunks, 2 channels/shard -> 3 shards (ips=2). Ranks must
    interleave shards: rank = within * n_shards + shard, priority = -rank, so
    the dispatch order (descending priority) steps through every shard once
    before revisiting one -- exactly the writer's shard arithmetic."""
    from astroviper.node_tasks.imaging.utils import compute_shard_task_priorities

    freq_chunks = [[c] for c in range(6)]
    store = _make_image_store(tmp_path, None, freq_chunks, shard_channels=2)

    node_task_data_mapping = {
        i: {"task_coords": {"frequency": {"slice": slice(i, i + 1), "data": [i]}}}
        for i in range(6)
    }
    priorities = compute_shard_task_priorities(
        store, "sky_residual", node_task_data_mapping
    )
    # task i: shard = i // 2, within = i % 2, rank = (i % 2) * 3 + i // 2.
    assert priorities == {0: 0, 1: -3, 2: -1, 3: -4, 4: -2, 5: -5}
    # Dispatch order (higher priority first): each wave of 3 hits all 3 shards.
    order = sorted(priorities, key=lambda t: -priorities[t])
    assert order == [0, 2, 4, 1, 3, 5]
    shards_first_wave = {t // 2 for t in order[:3]}
    assert shards_first_wave == {0, 1, 2}


def test_compute_shard_task_priorities_unsharded_returns_none(tmp_path):
    """A plain (non-sharded) array has no shard layout -> None (no reordering)."""
    from astroviper.node_tasks.imaging.utils import compute_shard_task_priorities

    freq_chunks = [[c] for c in range(4)]
    store = _make_image_store(tmp_path, None, freq_chunks, shard_channels=None)
    mapping = {
        i: {"task_coords": {"frequency": {"slice": slice(i, i + 1)}}} for i in range(4)
    }
    assert compute_shard_task_priorities(store, "sky_residual", mapping) is None


# --------------------------------------------------------------------------- #
# Shard -> OST recording (lfs getstripe)
# --------------------------------------------------------------------------- #
def _fake_getstripe_output(paths, ost_of):
    """Canonical ``lfs getstripe`` layout report for each path."""
    blocks = []
    for p in paths:
        ost = ost_of(p)
        blocks.append(
            f"{p}\n"
            "lmm_stripe_count:  1\n"
            "lmm_stripe_size:   1048576\n"
            "lmm_pattern:       raid0\n"
            "lmm_layout_gen:    0\n"
            f"lmm_stripe_offset: {ost}\n"
            "\tobdidx\t\t objid\t\t objid\t\t group\n"
            f"\t    {ost}\t      1234567\t    0x12d687\t             0\n"
        )
    return "\n".join(blocks)


def test_parse_lfs_getstripe():
    from astroviper.node_tasks.imaging.utils.skunk_works import _parse_lfs_getstripe

    paths = ["/scratch/store/VAR/c/0/0", "/scratch/store/VAR/c/0/1"]
    text = _fake_getstripe_output(paths, ost_of=lambda p: 40 + int(p[-1]))
    parsed = _parse_lfs_getstripe(text, paths)
    assert parsed == {
        paths[0]: {"stripe_count": 1, "ost_indices": [40]},
        paths[1]: {"stripe_count": 1, "ost_indices": [41]},
    }


def test_record_shard_ost_map(tmp_path, monkeypatch):
    """With a fake ``lfs``: one row per pre-created shard file per sharded
    variable, OSTs parsed from getstripe, saved as the sibling feather."""
    import subprocess
    from types import SimpleNamespace

    import pandas as pd

    from astroviper.node_tasks.imaging.utils import record_shard_ost_map

    def fake_run(cmd, **kwargs):
        assert cmd[:2] == ["lfs", "getstripe"]
        # Deterministic OST per file: its index in the argument list, mod 4.
        ost_of = {p: i % 4 for i, p in enumerate(cmd[2:])}
        return SimpleNamespace(
            returncode=0,
            stdout=_fake_getstripe_output(cmd[2:], ost_of=ost_of.get),
            stderr="",
        )

    # Patch BEFORE creating the store: create_empty_data_variables_on_disk
    # itself calls record_shard_ost_map after pre-creating the shard files, and
    # on a real Lustre host that call would otherwise use the real lfs.
    monkeypatch.setattr("shutil.which", lambda name: "/fake/lfs")
    monkeypatch.setattr(subprocess, "run", fake_run)

    nfreq, shard = 5, 2  # ceil(5/2) = 3 shard files per variable
    freq_chunks = [[c] for c in range(nfreq)]
    variables = ["sky_residual", "point_spread_function"]
    store = _make_image_store(
        tmp_path, None, freq_chunks, variables, shard_channels=shard
    )

    df = record_shard_ost_map(store, ["SKY_RESIDUAL", "POINT_SPREAD_FUNCTION"])
    assert df is not None
    assert sorted(df["variable"].unique()) == ["POINT_SPREAD_FUNCTION", "SKY_RESIDUAL"]
    assert len(df) == 6  # 3 shard files x 2 variables
    assert df["ost_index"].tolist() == [0, 1, 2, 3, 0, 1]
    assert (df["stripe_count"] == 1).all()
    # Shard indices enumerate the frequency-axis shard grid (axis 1 of 5 axes).
    freq_shard_indices = [
        si[1] for si in df[df["variable"] == "SKY_RESIDUAL"]["shard_index"]
    ]
    assert freq_shard_indices == [0, 1, 2]
    # Saved as a sibling feather of the store, round-trippable.
    sibling = store.rstrip("/") + "_shard_osts.ft"
    assert os.path.exists(sibling)
    back = pd.read_feather(sibling)
    assert back["ost_index"].tolist() == [0, 1, 2, 3, 0, 1]


def test_record_shard_ost_map_without_lfs(tmp_path, monkeypatch):
    """No ``lfs`` on PATH (any non-Lustre host) -> returns None, writes nothing."""
    from astroviper.node_tasks.imaging.utils import record_shard_ost_map

    # Patch BEFORE creating the store (store creation triggers the io-level
    # record_shard_ost_map call, which on a Lustre host would write the sibling).
    monkeypatch.setattr("shutil.which", lambda name: None)
    freq_chunks = [[c] for c in range(4)]
    store = _make_image_store(tmp_path, None, freq_chunks, shard_channels=2)
    assert record_shard_ost_map(store, ["SKY_RESIDUAL"]) is None
    assert not os.path.exists(store.rstrip("/") + "_shard_osts.ft")


def test_parse_lfs_getstripe_composite_pfl_layout():
    """PFL/composite output: OSTs live in ``l_ost_idx:`` object rows and
    ``lmm_stripe_count`` appears once per component (first one must win)."""
    from astroviper.node_tasks.imaging.utils.skunk_works import _parse_lfs_getstripe

    path = "/scratch/store/VAR/c/0/0"
    text = (
        f"{path}\n"
        "  lcm_layout_gen:    3\n"
        "  lcm_mirror_count:  1\n"
        "  lcm_entry_count:   2\n"
        "    lcme_id:             1\n"
        "    lcme_extent.e_start: 0\n"
        "      lmm_stripe_count:  1\n"
        "      lmm_stripe_size:   1048576\n"
        "      lmm_pattern:       raid0\n"
        "      lmm_stripe_offset: 5\n"
        "      lmm_objects:\n"
        "      - 0: { l_ost_idx: 5, l_fid: [0x100050000:0x20f0:0x0] }\n"
        "    lcme_id:             2\n"
        "    lcme_extent.e_start: 1073741824\n"
        "      lmm_stripe_count:  4\n"
        "      lmm_stripe_size:   1048576\n"
        "      lmm_pattern:       raid0\n"
        "      lmm_stripe_offset: -1\n"
    )
    parsed = _parse_lfs_getstripe(text, [path])
    assert parsed[path]["ost_indices"] == [5]
    assert parsed[path]["stripe_count"] == 1
