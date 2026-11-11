"""Independent numerical checks for the shared, xarray-free statistics backend."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.statistics import (
    _array_statistics,
    create_statistics_state,
    finalize_statistics_state,
)

VALUE_NAMES = (
    "min",
    "max",
    "peak",
    "sum",
    "sumsq",
    "n_pixels",
    "mean",
    "rms",
    "std",
    "median",
    "mad_sigma",
)


def _numpy_reference(values):
    valid = values[~np.isnan(values)].astype(np.float64)
    if not valid.size:
        return {name: 0.0 if name == "n_pixels" else np.nan for name in VALUE_NAMES}
    return {
        "min": valid.min(),
        "max": valid.max(),
        "peak": valid[np.abs(valid).argmax()],
        "sum": valid.sum(),
        "sumsq": (valid * valid).sum(),
        "n_pixels": float(valid.size),
        "mean": valid.mean(),
        "rms": np.sqrt(np.mean(valid * valid)),
        "std": valid.std(),
        "median": np.median(valid),
        "mad_sigma": 1.4826 * np.median(np.abs(valid - np.median(valid))),
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(0,), (1,), (2, 0), (2, 3, 7)])
@pytest.mark.parametrize("with_nan", [False, True])
def test_grouped_statistics_match_numpy(dtype, shape, with_nan):
    """Arbitrary retained groups and empty sample axes follow NumPy definitions."""
    samples = np.random.default_rng(25).normal(size=shape).astype(dtype)
    if with_nan and samples.size:
        samples[..., ::2] = np.nan
    original = samples.copy()
    summary = _array_statistics.summarize_samples(samples)
    result = _array_statistics.finalize_summary(summary, VALUE_NAMES, samples=samples)
    for index in np.ndindex(shape[:-1]):
        expected = _numpy_reference(samples[index])
        for name in VALUE_NAMES:
            assert result[name].dtype == np.dtype("float64")
            np.testing.assert_allclose(
                result[name][index], expected[name], rtol=1e-12, atol=1e-12
            )
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("values", [[], [1.0, -5.0, 5.0], [5.0, -5.0, 1.0]])
def test_prefiltered_plane_path(values):
    """The fast path for already filtered planes keeps empty and tie semantics."""
    samples = np.asarray(values)
    summary = _array_statistics.summarize_samples(samples, assume_valid=True)
    result = _array_statistics.finalize_summary(summary, VALUE_NAMES, samples=samples)
    for name, expected in _numpy_reference(samples).items():
        np.testing.assert_allclose(result[name], expected)


def test_backend_requires_samples_and_known_names():
    """Exact robust statistics cannot be recovered from compact moments alone."""
    summary = _array_statistics.summarize_samples(np.array([1.0, 3.0]))
    with pytest.raises(ValueError, match="retained samples"):
        _array_statistics.finalize_summary(summary, ("median",))
    with pytest.raises(ValueError, match="Unknown statistic"):
        _array_statistics.finalize_summary(summary, ("unknown",))


@pytest.mark.parametrize("dims", [("x", "y"), ("y", "x"), ()])
def test_coordinate_adapter_with_noncontiguous_data(dims):
    """Axis permutation, scalar coordinates and strided inputs survive adaptation."""
    data = xr.DataArray(
        np.arange(48, dtype=np.float32).reshape(2, 4, 6)[:, ::2, ::-2],
        dims=("channel", "x", "y"),
        coords={"channel": [10, 20], "x": [0, 2], "y": [5, 3, 1], "observer": "test"},
        attrs={"units": "Jy/beam"},
    )
    result = finalize_statistics_state(
        create_statistics_state(data, dims), ("mean", "std", "minpos")
    )
    xr.testing.assert_allclose(result["mean"], data.astype(np.float64).mean(dim=dims))
    xr.testing.assert_allclose(result["std"], data.astype(np.float64).std(dim=dims))
    assert result["observer"].item() == "test"
    assert result["mean"].attrs["units"] == "Jy/beam"
    assert result["minpos"].sizes["statistics_axis"] == len(dims)


@pytest.mark.parametrize("extreme", [-np.inf, np.inf])
def test_extrema_positions_skip_nan_before_infinity(extreme):
    """A missing sample cannot win a tie with an infinite valid extreme."""
    state = create_statistics_state(
        xr.DataArray([np.nan, extreme], dims="pixel"), "pixel"
    )
    result = finalize_statistics_state(state, ("minpos", "maxpos", "peak"))
    np.testing.assert_array_equal(result["minpos"], [1])
    np.testing.assert_array_equal(result["maxpos"], [1])
    assert result["peak"].item() == extreme


def test_exported_functions_match_bulk_finalization():
    """Individual processing-function entry points share the bulk numerical path."""
    from astroviper.processing_functions.image_analysis.statistics import (
        STATISTIC_FUNCTIONS,
    )

    data = xr.DataArray([np.nan, -7.0, 3.0, 5.0], dims="pixel")
    state = create_statistics_state(data, "pixel", statistics=VALUE_NAMES)
    result = finalize_statistics_state(state, tuple(STATISTIC_FUNCTIONS))
    for name, function in STATISTIC_FUNCTIONS.items():
        np.testing.assert_allclose(function(state), result[name])


def test_empty_summary_merging_does_not_mutate_inputs():
    """Combining an empty partition with data preserves inputs and numerical values."""
    samples = np.array([1e8, 1e8 + 1, 1e8 + 2])
    empty = _array_statistics.summarize_samples(np.array([]))
    summary = _array_statistics.summarize_samples(samples)
    snapshots = {name: np.asarray(value).copy() for name, value in summary.items()}
    for parts in ([empty, summary], [summary, empty], [empty, empty]):
        merged = _array_statistics.merge_summaries(parts)
        expected = _numpy_reference(samples)
        if parts[0] is empty and parts[1] is empty:
            expected = _numpy_reference(np.array([]))
        result = _array_statistics.finalize_summary(merged, ("std", "mean", "n_pixels"))
        for name in result:
            np.testing.assert_allclose(result[name], expected[name])
    for name in summary:
        np.testing.assert_array_equal(summary[name], snapshots[name])
