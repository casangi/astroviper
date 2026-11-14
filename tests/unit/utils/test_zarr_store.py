"""Unit tests for :mod:`astroviper.utils.zarr_store`."""

import asyncio

import pytest
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import LocalStore

from astroviper.utils.zarr_store import PreadLocalStore

BYTE_RANGES = [
    None,
    RangeByteRequest(3, 11),
    RangeByteRequest(90, 200),
    RangeByteRequest(150, 160),
    OffsetByteRequest(7),
    SuffixByteRequest(5),
    SuffixByteRequest(500),
]


@pytest.fixture
def stores(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "blob").write_bytes(bytes(range(100)))
    return LocalStore(tmp_path, read_only=True), PreadLocalStore(
        tmp_path, read_only=True
    )


def _bytes(buffer):
    return None if buffer is None else buffer.to_bytes()


@pytest.mark.parametrize("byte_range", BYTE_RANGES)
@pytest.mark.parametrize("key", ["a/blob", "a", "missing", "a/blob/x"])
def test_get_matches_local_store(stores, key, byte_range):
    local, pread = stores
    prototype = default_buffer_prototype()
    expected = _bytes(asyncio.run(local.get(key, prototype, byte_range)))
    assert _bytes(asyncio.run(pread.get(key, prototype, byte_range))) == expected
    assert (
        _bytes(pread.get_sync(key, prototype=prototype, byte_range=byte_range))
        == expected
    )


def test_get_partial_values_matches_local_store(stores):
    local, pread = stores
    prototype = default_buffer_prototype()
    key_ranges = [("a/blob", byte_range) for byte_range in BYTE_RANGES]
    expected = asyncio.run(local.get_partial_values(prototype, key_ranges))
    got = asyncio.run(pread.get_partial_values(prototype, key_ranges))
    assert [_bytes(b) for b in got] == [_bytes(b) for b in expected]


def test_unexpected_byte_range_raises(stores):
    _, pread = stores
    with pytest.raises(TypeError, match="Unexpected byte_range"):
        pread.get_sync("a/blob", byte_range=(0, 4))
