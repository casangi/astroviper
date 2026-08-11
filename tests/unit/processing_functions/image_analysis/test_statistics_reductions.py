import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.statistics import (
    create_statistics_state,
    finalize_statistics_state,
    merge_statistics_states,
)


def test_second_moment_statistics_ignore_nan_and_preserve_retained_dims():
    """Verify sumsq, RMS, and sigma exclude NaNs and retain plane coordinates."""
    data = xr.DataArray(
        [[1.0, 2.0, np.nan], [3.0, 3.0, 3.0]],
        dims=("channel", "pixel"),
        coords={"channel": [10, 11]},
    )

    result = finalize_statistics_state(
        create_statistics_state(data, "pixel"),
        ("sumsq", "rms", "sigma"),
    )

    expected_coords = {"channel": [10, 11]}
    xr.testing.assert_allclose(
        result["sumsq"],
        xr.DataArray([5.0, 27.0], dims="channel", coords=expected_coords),
    )
    xr.testing.assert_allclose(
        result["rms"],
        xr.DataArray([np.sqrt(2.5), 3.0], dims="channel", coords=expected_coords),
    )
    xr.testing.assert_allclose(
        result["sigma"],
        xr.DataArray([np.sqrt(0.5), 0.0], dims="channel", coords=expected_coords),
    )


def test_second_moment_statistics_merge_from_unequal_partitions():
    """Match global second-moment statistics after merging unequal partitions."""
    data = xr.DataArray([1.0, 2.0, 8.0, np.nan], dims="pixel")
    states = [
        create_statistics_state(data.isel(pixel=slice(0, 1)), "pixel"),
        create_statistics_state(data.isel(pixel=slice(1, None)), "pixel"),
    ]

    result = finalize_statistics_state(
        merge_statistics_states(
            states, partition_dim="pixel", reduction_dims=("pixel",)
        ),
        ("mean", "sumsq", "rms", "sigma", "npts"),
    )

    assert result["mean"].item() == pytest.approx(11 / 3)
    assert result["sumsq"].item() == pytest.approx(69)
    assert result["rms"].item() == pytest.approx(np.sqrt(23))
    assert result["sigma"].item() == pytest.approx(np.std([1.0, 2.0, 8.0], ddof=1))
    assert result["npts"].item() == 3


@pytest.mark.parametrize("count", [0, 1])
def test_sigma_edge_cases(count):
    """Define second-moment outputs for empty and single-sample reductions."""
    values = [4.0] * count + [np.nan] * (1 - count)
    state = create_statistics_state(xr.DataArray(values, dims="pixel"), "pixel")

    result = finalize_statistics_state(state, ("sumsq", "rms", "sigma"))

    if count == 0:
        assert all(np.isnan(result[name].item()) for name in result.data_vars)
    else:
        assert result["sumsq"].item() == 16
        assert result["rms"].item() == 4
        assert result["sigma"].item() == 0


def test_order_statistics_and_absolute_extrema_positions():
    """Verify exact median/MAD, absolute positions, tie-breaking, and units."""
    data = xr.DataArray(
        [[5.0, 1.0, 9.0, np.nan], [2.0, 8.0, 8.0, 4.0]],
        dims=("channel", "pixel"),
        coords={"channel": [100, 101]},
        attrs={"units": "Jy/beam"},
    )
    state = create_statistics_state(
        data,
        "pixel",
        statistics=("median", "medabsdevmed"),
        positions={"pixel": [10, 11, 12, 13]},
    )

    result = finalize_statistics_state(
        state, ("median", "medabsdevmed", "minpos", "maxpos")
    )

    np.testing.assert_allclose(result["median"], [5.0, 6.0])
    np.testing.assert_allclose(result["medabsdevmed"], [4.0, 2.0])
    np.testing.assert_array_equal(result["minpos"], [[11], [10]])
    # The equal maxima in channel 101 use the first absolute pixel position.
    np.testing.assert_array_equal(result["maxpos"], [[12], [11]])
    assert result["median"].attrs["units"] == "Jy/beam"


def test_positions_and_exact_median_merge_across_reduced_partition():
    """Preserve exact robust statistics and extrema positions across state merging."""
    data = xr.DataArray([8.0, 1.0, 8.0, 3.0], dims="pixel")
    states = [
        create_statistics_state(
            data.isel(pixel=slice(0, 2)),
            "pixel",
            statistics=("median", "medabsdevmed"),
            positions={"pixel": [20, 21]},
        ),
        create_statistics_state(
            data.isel(pixel=slice(2, 4)),
            "pixel",
            statistics=("median", "medabsdevmed"),
            positions={"pixel": [22, 23]},
        ),
    ]
    merged = merge_statistics_states(
        states, partition_dim="pixel", reduction_dims=("pixel",)
    )

    result = finalize_statistics_state(
        merged, ("median", "medabsdevmed", "minpos", "maxpos")
    )

    assert result["median"].item() == 5.5
    assert result["medabsdevmed"].item() == 2.5
    np.testing.assert_array_equal(result["minpos"], [21])
    np.testing.assert_array_equal(result["maxpos"], [20])


def test_extrema_positions_unravel_without_coordinate_mesh(monkeypatch):
    """Map flat extrema to slice/array positions without allocating a coordinate mesh."""
    data = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=("channel", "frequency", "pixel"),
        coords={"channel": [0, 1]},
    )

    def reject_meshgrid(*args, **kwargs):
        raise AssertionError("extrema positions must not construct a coordinate mesh")

    monkeypatch.setattr(np, "meshgrid", reject_meshgrid)
    state = create_statistics_state(
        data,
        ("frequency", "pixel"),
        positions={
            "frequency": slice(10, 16, 2),
            "pixel": np.array([100, 105, 109, 120]),
        },
    )
    result = finalize_statistics_state(state, ("minpos", "maxpos"))

    np.testing.assert_array_equal(result["minpos"], [[10, 100], [10, 100]])
    np.testing.assert_array_equal(result["maxpos"], [[14, 120], [14, 120]])


def test_empty_extrema_positions_are_negative_one():
    """Use the documented -1 position sentinel when every selected sample is NaN."""
    data = xr.DataArray(np.full((2, 3), np.nan), dims=("frequency", "pixel"))

    result = finalize_statistics_state(
        create_statistics_state(data, ("frequency", "pixel")),
        ("minpos", "maxpos"),
    )

    np.testing.assert_array_equal(result["minpos"], [-1, -1])
    np.testing.assert_array_equal(result["maxpos"], [-1, -1])
