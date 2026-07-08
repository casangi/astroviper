"""Unit tests for the ``_skunk_works`` performance I/O variants.

Builds tiny synthetic Zarr stores (v3 sharded, v3 plain, v2, v3 fixed-length
strings) and checks that the low-level reader matches ``zarr.open_array`` for
multi-channel selections, that the direct chunk writer round-trips through Zarr
(uncompressed, Blosc, and partial/edge chunks), and that
``load_processing_set_skunk_works`` reconstructs a usable, mutable processing
set from the chunk blobs alone.
"""

from __future__ import annotations

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


@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize(
    "compressor",
    [None, zarr.codecs.BloscCodec(cname="lz4", clevel=5)],
    ids=["uncompressed", "blosc"],
)
def test_write_roundtrip(tmp_path, compressor, num_threads):
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
        store, variables, task_coords, img, num_threads=num_threads
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


@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize(
    "compressor",
    [None, zarr.codecs.BloscCodec(cname="lz4", clevel=5)],
    ids=["uncompressed", "blosc"],
)
def test_write_roundtrip_sharded(tmp_path, compressor, num_threads):
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
            num_threads=num_threads,
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
@pytest.mark.parametrize("num_threads", [1, 4])
def test_load_processing_set_reconstruction(tmp_path, num_threads):
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
        store, sel, data_group, "base", freq_values, "linear", num_threads=num_threads
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
