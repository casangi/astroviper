import numpy as np
import pytest
import xarray as xr
from dask import array as da

from astroviper.node_tasks.image_analysis.image_statistics import (
    _compact_positions,
    _index_expression,
    _positions,
    _region_boxes,
    _take_indexer,
    build_image_selection,
    image_statistics,
)


@pytest.fixture
def image_cube():
    """Return a canonical five-axis image with a named broadcastable mask."""
    values = np.arange(2 * 4 * 2 * 4 * 5.0).reshape(2, 4, 2, 4, 5)
    image = xr.DataArray(
        values,
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={
            "time": [0, 1],
            "frequency": np.arange(4),
            "polarization": ["I", "Q"],
            "l": np.arange(4),
            "m": np.arange(5),
        },
        name="SKY",
        attrs={"units": "Jy/beam"},
    )
    mask = xr.DataArray(np.eye(4, 5, dtype=bool), dims=("l", "m"))
    return xr.Dataset({"SKY": image, "MASK_SKY": mask})


@pytest.fixture
def image_store(image_cube, tmp_path):
    """Write a synthetic image to Zarr using the image-suite storage pattern."""
    path = tmp_path / "statistics.img.zarr"
    image_cube.chunk({"frequency": 2, "l": 2, "m": 3}).to_zarr(path)
    return str(path), image_cube


def test_mask_multiple_axes_shape_units_and_positions():
    """Verify masked spatial reductions, output shape, units, and absolute extrema."""
    image = xr.DataArray(
        np.arange(18.0).reshape(2, 3, 3),
        dims=("frequency", "l", "m"),
        coords={"frequency": [100.0, 200.0]},
        attrs={"units": "Jy/beam"},
        name="SKY",
    )
    mask = xr.DataArray(
        [[False, True, True], [True, True, True], [True, True, False]],
        dims=("l", "m"),
    )

    result = image_statistics(
        image,
        axes=("l", "m"),
        chans="1",
        mask=mask,
        stretch=True,
        statistics=(
            "mean",
            "median",
            "max",
            "maxpos",
            "min",
            "minpos",
            "sigma",
            "rms",
            "medabsdevmed",
            "sum",
            "npts",
        ),
    )

    assert result.sizes == {"frequency": 1, "statistics_axis": 2}
    assert result["npts"].item() == 7
    assert result["min"].item() == 10
    assert result["max"].item() == 16
    np.testing.assert_array_equal(result["minpos"], [[0, 1]])
    np.testing.assert_array_equal(result["maxpos"], [[2, 1]])
    assert result["median"].item() == 13
    assert result["medabsdevmed"].item() == 2
    assert result["mean"].attrs["units"] == "Jy/beam"


def test_regular_large_selection_and_partition_remain_lazy_slices():
    """Keep large regular user selections and worker partitions lazy and compact."""
    image = xr.DataArray(da.empty((10_000_000,), chunks=(100_000,)), dims="frequency")

    selection = build_image_selection(
        image,
        chans="1000000~8999999",
        partition={"frequency": slice(2_000_000, 3_000_000)},
    )

    assert selection.user_indexers["frequency"] == slice(1_000_000, 9_000_000, 1)
    assert selection.effective_indexers["frequency"] == slice(3_000_000, 4_000_000, 1)
    selected = image.isel(selection.effective_indexers)
    assert isinstance(selected.data, da.Array)
    assert selected.sizes["frequency"] == 1_000_000

    union = build_image_selection(image, chans="0~4999999,5000000~9999999")
    assert union.effective_indexers["frequency"] == slice(0, 10_000_000, 1)


def test_irregular_selection_uses_integer_array():
    """Represent a genuinely irregular channel union with exact integer positions."""
    image = xr.DataArray(np.zeros(20), dims="frequency")

    selection = build_image_selection(image, chans="1,4,8~10")

    np.testing.assert_array_equal(
        selection.effective_indexers["frequency"], [1, 4, 8, 9, 10]
    )


def test_regular_union_is_compacted_after_parsing():
    """Collapse adjacent channel ranges into one storage-efficient slice."""
    image = xr.DataArray(np.zeros(20), dims="frequency")

    selection = build_image_selection(image, chans="1~3,4~6")

    assert selection.effective_indexers["frequency"] == slice(1, 7, 1)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("", slice(0, 20, 1)),
        ("2~8^2", slice(2, 9, 2)),
        ("8~2^2", slice(2, 9, 2)),
        ("<3", slice(0, 3, 1)),
        ("<=3", slice(0, 4, 1)),
        (">3", slice(4, 20, 1)),
        (">=3", slice(3, 20, 1)),
        ("-1", slice(19, 20, 1)),
        ("-8~-1^3", slice(12, 19, 3)),
    ],
)
def test_regular_index_expressions_produce_slices(expression, expected):
    """Parse full, stepped, descending, comparison, and negative selections as slices."""
    assert _index_expression(expression, 20, "chans") == expected


def test_on_disk_selection_loads_only_required_variables(image_store, monkeypatch):
    """Read a real Zarr subimage while excluding unrelated stored variables."""
    path, image_cube = image_store
    with_extra = image_cube.assign(EXTRA=xr.ones_like(image_cube.SKY))
    with_extra.to_zarr(path, mode="w")
    loaded = {}

    def load_selected(store, block_des):
        loaded["indexers"] = block_des
        return xr.open_zarr(store).isel(block_des)

    monkeypatch.setattr("xradio.image.load_image", load_selected)
    result = image_statistics(
        path,
        data_variable="SKY",
        chans="1~2",
        axes=("l", "m"),
        mask="MASK_SKY",
        stretch=True,
        statistics=("mean", "npts"),
    )

    assert loaded["indexers"]["frequency"] == slice(1, 3, 1)
    assert set(result.data_vars) == {"mean", "npts"}
    expected = image_cube.SKY.isel(frequency=slice(1, 3)).where(image_cube.MASK_SKY)
    xr.testing.assert_allclose(result["mean"], expected.mean(("l", "m")))


def test_region_union_and_value_filters(image_cube):
    """Apply a region-record union and inclusion/exclusion filters before reduction."""
    result = image_statistics(
        image_cube,
        data_variable="SKY",
        region={"blc": [1, 1], "trc": [3, 4]},
        includepix=(0, 300),
        excludepix=(100, 199),
        statistics=("min", "max", "npts"),
    )
    selected = image_cube.SKY.isel(l=slice(1, 4), m=slice(1, 5))
    expected = selected.where((selected <= 300) & ((selected < 100) | (selected > 199)))
    assert result["min"].item() == expected.min().item()
    assert result["max"].item() == expected.max().item()
    assert result["npts"].item() == expected.count().item()


def test_multiple_boxes_exclude_bounding_rectangle_gaps(image_cube):
    """Count only pixels in a disjoint box union, not gaps in its bounding box."""
    result = image_statistics(
        image_cube.SKY.isel(time=0, frequency=0, polarization=0, drop=False),
        box="0,0,0,0,3,4,3,4",
        statistics=("sum", "npts"),
    )
    assert result["npts"].item() == 2
    assert result["sum"].item() == 0 + 19


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"box": "0,0,1,1", "region": "0,0,1,1"}, ValueError, "either"),
        ({"box": "2,2,1,3"}, ValueError, "lower-left"),
        ({"box": "0,0,9,9"}, IndexError, "outside"),
        ({"chans": "bad"}, ValueError, "Unsupported"),
        ({"chans": "99"}, IndexError, "outside"),
        ({"stokes": "V"}, ValueError, "Unknown"),
        ({"axes": "bad"}, ValueError, "Unknown statistics axis"),
        ({"partition": {"bad": slice(0, 1)}}, ValueError, "not present"),
        ({"partition": {"frequency": slice(0, 0)}}, ValueError, "empty"),
    ],
)
def test_selection_validation_errors(image_cube, kwargs, error, message):
    """Reject conflicting, malformed, out-of-bounds, and empty selectors."""
    with pytest.raises(error, match=message):
        image_statistics(image_cube, data_variable="SKY", **kwargs)


def test_dataset_variable_and_mask_validation(image_cube):
    """Require unambiguous data variables and valid named or shaped masks."""
    ambiguous = image_cube.assign(SKY_COPY=image_cube.SKY)
    with pytest.raises(ValueError, match="data_variable is required"):
        image_statistics(ambiguous)
    with pytest.raises(KeyError, match="MISSING"):
        image_statistics(image_cube, data_variable="SKY", mask="MISSING")
    with pytest.raises(ValueError, match="stretch=True"):
        image_statistics(image_cube, data_variable="SKY", mask="MASK_SKY")
    with pytest.raises(ValueError, match="array mask must match"):
        image_statistics(image_cube.SKY, mask=np.ones(3, dtype=bool), stretch=True)


def test_partition_and_graph_selection_are_mutually_exclusive(image_cube):
    """Reject two competing worker-selection sources in one node invocation."""
    with pytest.raises(ValueError, match="cannot both"):
        image_statistics(
            image_cube,
            data_variable="SKY",
            partition={"frequency": slice(0, 1)},
            data_selection={"image": {"frequency": slice(0, 1)}},
        )


def test_indexer_helpers_cover_arrays_booleans_and_validation():
    """Normalize compact, Boolean, negative, irregular, and invalid index arrays."""
    assert _compact_positions([7]) == slice(7, 8, 1)
    assert _compact_positions([1, 3, 5]) == slice(1, 6, 2)
    np.testing.assert_array_equal(_compact_positions([]), [])
    assert _positions([True, False, True], 3) == slice(0, 3, 2)
    assert _positions(-1, 3) == slice(2, 3, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        _compact_positions([[1]])
    with pytest.raises(ValueError, match="wrong length"):
        _positions([True], 3)
    with pytest.raises(IndexError, match="outside"):
        _positions([4], 3)


def test_worker_partition_composes_with_irregular_user_selection():
    """Subset irregular absolute selections with regular and irregular workers."""
    source = np.array([1, 4, 8, 9, 10])
    np.testing.assert_array_equal(_take_indexer(source, slice(1, 4)), [4, 8, 9])
    np.testing.assert_array_equal(_take_indexer(source, [0, 2, 4]), [1, 8, 10])
    np.testing.assert_array_equal(
        _take_indexer(slice(10, 20, 2), [0, 2, 3]), [10, 14, 16]
    )


def test_crtf_text_and_file_regions(tmp_path):
    """Parse inline and file-based CRTF pixel boxes and reject unsupported shapes."""
    text = "#CRTFv0\nbox[[1pix,2pix],[3pix,4pix]]"
    assert _region_boxes(text, 6, 7) == ((1, 2, 3, 4),)
    path = tmp_path / "region.crtf"
    path.write_text(text)
    assert _region_boxes(path, 6, 7) == ((1, 2, 3, 4),)
    with pytest.raises(ValueError, match="blc"):
        _region_boxes({"blc": [0, 0]}, 6, 7)
    with pytest.raises(ValueError, match="pixel-coordinate"):
        _region_boxes("circle[[1pix,1pix],2pix]", 6, 7)


def test_multicharacter_polarizations_and_integer_axes():
    """Parse comma-separated multi-character labels and normalize integer axes."""
    image = xr.DataArray(
        np.zeros((3, 2)),
        dims=("polarization", "pixel"),
        coords={"polarization": ["XX", "XY", "YY"]},
    )
    selection = build_image_selection(image, stokes="XX,YY", axes=(1, 1))
    assert selection.effective_indexers["polarization"] == slice(0, 3, 2)
    assert selection.reduction_dims == ("pixel",)
    with pytest.raises(ValueError, match="Cannot parse"):
        build_image_selection(image, stokes="XXYYBAD")


def test_full_shape_and_spatial_array_masks(image_cube):
    """Accept both full-dimensional and stretched two-dimensional array masks."""
    full_mask = np.ones(image_cube.SKY.shape, dtype=bool)
    full = image_statistics(image_cube.SKY, mask=full_mask, statistics=("npts",))
    assert full["npts"].item() == image_cube.SKY.size

    spatial_mask = np.eye(4, 5, dtype=bool)
    spatial = image_statistics(
        image_cube.SKY,
        mask=spatial_mask,
        stretch=True,
        statistics=("npts",),
    )
    assert spatial["npts"].item() == 2 * 4 * 2 * 4
