import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.statistics import (
    create_statistics_state,
    finalize_statistics_state,
    merge_statistics_states,
)


def test_second_moment_statistics_ignore_nan_and_preserve_retained_dims():
    """Verify sumsq, RMS, and std exclude NaNs and retain plane coordinates."""
    data = xr.DataArray(
        [[1.0, 2.0, np.nan], [3.0, 3.0, 3.0]],
        dims=("channel", "pixel"),
        coords={"channel": [10, 11]},
    )

    result = finalize_statistics_state(
        create_statistics_state(data, "pixel"),
        ("sumsq", "rms", "std"),
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
        result["std"],
        xr.DataArray([0.5, 0.0], dims="channel", coords=expected_coords),
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
        ("mean", "sumsq", "rms", "std", "n_pixels"),
    )

    assert result["mean"].item() == pytest.approx(11 / 3)
    assert result["sumsq"].item() == pytest.approx(69)
    assert result["rms"].item() == pytest.approx(np.sqrt(23))
    assert result["std"].item() == pytest.approx(np.std([1.0, 2.0, 8.0], ddof=0))
    assert result["n_pixels"].item() == 3


@pytest.mark.parametrize("count", [0, 1])
def test_std_edge_cases(count):
    """Define second-moment outputs for empty and single-sample reductions."""
    values = [4.0] * count + [np.nan] * (1 - count)
    state = create_statistics_state(xr.DataArray(values, dims="pixel"), "pixel")

    result = finalize_statistics_state(state, ("sumsq", "rms", "std"))

    if count == 0:
        assert all(np.isnan(result[name].item()) for name in result.data_vars)
    else:
        assert result["sumsq"].item() == 16
        assert result["rms"].item() == 4
        assert result["std"].item() == 0


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
        statistics=("median", "mad_sigma"),
        positions={"pixel": [10, 11, 12, 13]},
    )

    result = finalize_statistics_state(
        state, ("median", "mad_sigma", "minpos", "maxpos")
    )

    np.testing.assert_allclose(result["median"], [5.0, 6.0])
    np.testing.assert_allclose(result["mad_sigma"], 1.4826 * np.array([4.0, 2.0]))
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
            statistics=("median", "mad_sigma"),
            positions={"pixel": [20, 21]},
        ),
        create_statistics_state(
            data.isel(pixel=slice(2, 4)),
            "pixel",
            statistics=("median", "mad_sigma"),
            positions={"pixel": [22, 23]},
        ),
    ]
    merged = merge_statistics_states(
        states, partition_dim="pixel", reduction_dims=("pixel",)
    )

    result = finalize_statistics_state(
        merged, ("median", "mad_sigma", "minpos", "maxpos")
    )

    assert result["median"].item() == 5.5
    assert result["mad_sigma"].item() == pytest.approx(1.4826 * 2.5)
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


def test_retained_partition_dimension_is_concatenated_and_sorted():
    """Concatenate disjoint retained-axis states and restore coordinate order."""
    data = xr.DataArray(
        [[3.0, 4.0], [1.0, 2.0]],
        dims=("frequency", "pixel"),
        coords={"frequency": [20, 10]},
    )
    states = [
        create_statistics_state(data.isel(frequency=slice(0, 1)), "pixel"),
        create_statistics_state(data.isel(frequency=slice(1, 2)), "pixel"),
    ]

    result = finalize_statistics_state(
        merge_statistics_states(
            states, partition_dim="frequency", reduction_dims=("pixel",)
        ),
        ("mean", "n_pixels"),
    )

    np.testing.assert_array_equal(result.frequency, [10, 20])
    np.testing.assert_allclose(result["mean"], [1.5, 3.5])


def test_statistics_state_and_finalizer_validation():
    """Reject lazy/non-DataArray inputs, unknown dimensions, and unknown statistics."""
    data = xr.DataArray([1.0, 2.0], dims="pixel")
    with pytest.raises(TypeError, match="DataArray"):
        create_statistics_state(np.array([1.0]), "pixel")
    with pytest.raises(TypeError, match="loaded NumPy"):
        create_statistics_state(data.chunk(), "pixel")
    with pytest.raises(ValueError, match="Unknown reduction dimensions"):
        create_statistics_state(data, "bad")
    with pytest.raises(ValueError, match="Unknown statistics"):
        finalize_statistics_state(create_statistics_state(data, "pixel"), ("bad",))


def test_merge_state_validation_and_coordinate_alignment():
    """Reject empty, incomplete, sample-inconsistent, and misaligned state lists."""
    with pytest.raises(ValueError, match="At least one"):
        merge_statistics_states([])

    data = xr.DataArray(
        [[1.0], [2.0]], dims=("channel", "pixel"), coords={"channel": [0, 1]}
    )
    complete = create_statistics_state(data, "pixel")
    with pytest.raises(ValueError, match="missing"):
        merge_statistics_states([complete.drop_vars("sumsq")])

    sampled = create_statistics_state(data, "pixel", statistics=("median",))
    with pytest.raises(ValueError, match="Only some"):
        merge_statistics_states([sampled, complete], reduction_dims=("pixel",))

    shifted = create_statistics_state(data.assign_coords(channel=[10, 11]), "pixel")
    with pytest.raises(ValueError, match="align"):
        merge_statistics_states([complete, shifted], reduction_dims=("pixel",))


def test_median_requires_samples_in_state():
    """Explain that exact robust finalization requires opting into sample storage."""
    state = create_statistics_state(xr.DataArray([1.0, 2.0], dims="pixel"), "pixel")
    with pytest.raises(ValueError, match="does not contain samples"):
        finalize_statistics_state(state, ("median",))


def test_position_indexer_length_validation():
    """Reject absolute slice and array metadata that do not match selected data."""
    data = xr.DataArray([1.0, 2.0], dims="pixel")
    with pytest.raises(ValueError, match="wrong length"):
        create_statistics_state(data, "pixel", positions={"pixel": slice(5, 6)})
    with pytest.raises(ValueError, match="wrong length"):
        create_statistics_state(data, "pixel", positions={"pixel": [5]})


@pytest.mark.parametrize("values", [[1.0, 2.0, 8.0, np.nan], [4.0], [np.nan]])
@pytest.mark.parametrize("statistic", ["std", "mad_sigma"])
def test_spread_matches_plane_statistics(values, statistic):
    """Both independent implementations use the same spread normalization."""
    from astroviper.processing_functions.image_analysis.plane_statistics import (
        calculate_plane_statistics,
    )

    data = xr.DataArray(
        np.array(values).reshape(1, 1, 1, 1, -1),
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={"time": [0], "frequency": [100], "polarization": ["I"]},
        name="SKY_RESIDUAL",
    )
    result = finalize_statistics_state(
        create_statistics_state(data, ("l", "m"), statistics=(statistic,)),
        (statistic,),
    )
    plane_result = calculate_plane_statistics(data.to_dataset())["sky_residual"]
    xr.testing.assert_allclose(result[statistic], plane_result[statistic])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([2.0, -9.0, 4.0], -9.0),
        ([-2.0, 9.0], 9.0),
        ([-9.0, 9.0], -9.0),
        ([9.0, -9.0], 9.0),
        ([np.nan, -3.0], -3.0),
        ([np.nan, np.nan], np.nan),
        ([0.0, 0.0], 0.0),
        ([4.0, 4.0], 4.0),
    ],
)
def test_peak_preserves_sign_and_ties_after_merge(values, expected):
    """Peak agrees with the first largest-magnitude pixel, even with reversed chunks."""
    data = xr.DataArray(values, dims="pixel", attrs={"units": "Jy/beam"})
    states = [
        create_statistics_state(
            data.isel(pixel=slice(i, i + 1)), "pixel", positions={"pixel": [i]}
        )
        for i in range(len(values))
    ]
    merged = merge_statistics_states(
        states[::-1], partition_dim="pixel", reduction_dims=("pixel",)
    )
    for state in (create_statistics_state(data, "pixel"), merged):
        result = finalize_statistics_state(state, ("peak",))
        np.testing.assert_allclose(result["peak"], expected)
        assert result["peak"].attrs["units"] == "Jy/beam"


@pytest.mark.parametrize(
    "values, expected", [([1.0, np.nan, 3.0], 2.0), ([np.nan], 0.0)]
)
def test_pixel_count_is_float64(values, expected):
    """Expose floating-point counts, including zero for an all-NaN selection."""
    from astroviper.processing_functions.image_analysis.statistics import (
        statistics_n_pixels,
    )

    state = create_statistics_state(xr.DataArray(values, dims="pixel"), "pixel")
    for result in (
        statistics_n_pixels(state),
        finalize_statistics_state(state, ("n_pixels",))["n_pixels"],
    ):
        assert result.dtype == np.dtype("float64")
        assert result.item() == expected


@pytest.mark.parametrize(
    "dtype, offset", [(np.float32, 1e4), (np.float64, 1e8), (np.float32, 1e20)]
)
@pytest.mark.parametrize("merge_mode", ["direct", "flat", "tree"])
def test_precision_matches_plane_statistics(dtype, offset, merge_mode):
    """Preserve small spreads, float64 outputs, and empty planes through merges."""
    from astroviper.processing_functions.image_analysis.plane_statistics import (
        PLANE_STATISTIC_NAMES,
        calculate_plane_statistics,
    )

    values = np.full((1, 2, 1, 1, 6), np.nan, dtype=dtype)
    values[0, 0, 0, 0, 2:] = offset + np.arange(4, dtype=dtype)
    data = xr.DataArray(
        values,
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={"time": [0], "frequency": [100, 101], "polarization": ["I"]},
        name="SKY_RESIDUAL",
    )
    kwargs = dict(statistics=PLANE_STATISTIC_NAMES)
    if merge_mode == "direct":
        state = create_statistics_state(data, ("l", "m"), **kwargs)
    else:
        states = [
            create_statistics_state(
                data.isel(m=slice(start, stop)),
                ("l", "m"),
                positions={"m": slice(start, stop)},
                **kwargs,
            )
            for start, stop in [(0, 2), (2, 3), (3, 6)]
        ]
        merge_kwargs = dict(partition_dim="m", reduction_dims=("l", "m"))
        if merge_mode == "tree":
            states = [merge_statistics_states(states[1:], **merge_kwargs), states[0]]
        state = merge_statistics_states(states, **merge_kwargs)
    result = finalize_statistics_state(state, PLANE_STATISTIC_NAMES)
    expected = calculate_plane_statistics(data.to_dataset())["sky_residual"]
    for name in PLANE_STATISTIC_NAMES:
        assert result[name].dtype == np.dtype("float64")
        np.testing.assert_allclose(result[name], expected[name], rtol=1e-12, atol=1e-12)
    assert data.dtype == dtype
