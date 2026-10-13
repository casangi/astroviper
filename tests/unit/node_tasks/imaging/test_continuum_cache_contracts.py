"""Persistent continuum caches: channel ownership, dtype and failure handling."""

import numpy as np
import pytest
import xarray as xr
import zarr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    _create_wideband_primary_beam_cache_store,
    _remove_wideband_primary_beam_cache,
)
from astroviper.node_tasks.imaging.image_continuum_single_field import (
    _load_mfs_visibility_grid_in_place,
    _load_mvc_visibility_grid_in_place,
    _load_wideband_primary_beam_in_place,
    _stored_coordinate_indexer,
    _write_wideband_primary_beam_in_place,
)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("single_precision", [False, True])
def test_primary_beam_cache_roundtrip_preserves_disjoint_reordered_channels(
    tmp_path, zarr_format, single_precision
):
    store = str(tmp_path / "cache.zarr")
    zarr.open_group(store, mode="w", zarr_format=zarr_format)
    coords = dict(
        time=[0.0],
        frequency=[100.0, 102.0, 101.0],
        polarization=["XX", "YY"],
        l=[-1.0, 1.0],
        m=[-1.0, 1.0],
    )
    dtype = np.float32 if single_precision else np.float64
    beam = xr.DataArray(
        np.arange(24, dtype=dtype).reshape(1, 3, 2, 2, 2),
        dims=tuple(coords),
        coords=coords,
    )
    image = xr.Dataset({"PRIMARY_BEAM": beam})
    _create_wideband_primary_beam_cache_store(
        store, image, image.frequency.values, "linear", single_precision, None
    )
    first = image.isel(frequency=[2, 0])
    _write_wideband_primary_beam_in_place(first, store)
    # Writing one partition must not overwrite another partition's channels.
    unwritten = _load_wideband_primary_beam_in_place(image.isel(frequency=[1]), store)
    assert np.isnan(unwritten.PRIMARY_BEAM).all()
    _write_wideband_primary_beam_in_place(image.isel(frequency=[1]), store)
    for selection in ([2, 0], [1], [0, 1, 2]):
        subset = image.isel(frequency=selection)
        loaded = _load_wideband_primary_beam_in_place(subset, store)
        xr.testing.assert_equal(loaded.PRIMARY_BEAM, subset.PRIMARY_BEAM)
        assert loaded.PRIMARY_BEAM.dtype == dtype
    _remove_wideband_primary_beam_cache(store)
    with pytest.raises(KeyError, match="cache is missing"):
        _load_wideband_primary_beam_in_place(image, store)


@pytest.mark.parametrize("kind", ["mfs", "mvc", "primary_beam"])
def test_missing_disk_cache_fails_explicitly(tmp_path, kind):
    store = str(tmp_path / "cache.zarr")
    zarr.open_group(store, mode="w")
    with pytest.raises(KeyError, match="cache is missing"):
        if kind == "mfs":
            _load_mfs_visibility_grid_in_place(store)
        elif kind == "mvc":
            _load_mvc_visibility_grid_in_place(xr.Dataset(), store)
        else:
            _load_wideband_primary_beam_in_place(xr.Dataset(), store)


def test_cache_coordinate_indexer_rejects_ambiguous_stored_channels():
    with pytest.raises(ValueError, match="duplicate"):
        _stored_coordinate_indexer([1.0, 1.0, 2.0], [1.0], "frequency")
