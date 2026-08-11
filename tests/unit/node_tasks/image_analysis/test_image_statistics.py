import numpy as np
import pytest
import xarray as xr
from dask import array as da

from astroviper.node_tasks.image_analysis.image_statistics import (
    _index_expression,
    build_image_selection,
    image_statistics,
)


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
