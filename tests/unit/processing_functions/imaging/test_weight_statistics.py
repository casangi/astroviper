"""Unit tests for the weight statistics
(``astroviper.processing_functions.imaging.weight_statistics``) on tiny
synthetic arrays: the closed forms for equal weights, the Stokes combination
of unequal correlation noise, flagged and NaN weights, the visibility to image
channel mapping, the uv extent and the xarray-level function (autocorrelations
left out)."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging import weight_statistics
from astroviper.processing_functions.imaging.weight_statistics import (
    SPEED_OF_LIGHT,
    accumulate_correlation_weight_sums,
    accumulate_uv_extent,
    calculate_weight_statistics_single_field,
    stokes_weight_statistics,
    valid_sample_mask,
)

N_TIME, N_BASELINE = 3, 4


def _statistics(weight, weight_imaging, flag, correlations, stokes, channel_map=None):
    if channel_map is None:
        channel_map = np.zeros(weight.shape[2], dtype=int)
    sums = accumulate_correlation_weight_sums(
        weight, weight_imaging, flag, channel_map, int(channel_map.max()) + 1
    )
    return sums, stokes_weight_statistics(sums, correlations, stokes)


def test_equal_weights_closed_form():
    """Natural weighting with equal weights: sigma_I = 1 / sqrt(2 sum w) with
    sum w over one correlation, N_eff the number of samples."""
    w0 = 2.5
    weight = np.full((N_TIME, N_BASELINE, 1, 2), w0)
    flag = np.zeros(weight.shape, dtype=bool)
    sums, stats = _statistics(weight, weight, flag, ["XX", "YY"], ["I", "Q"])
    n = N_TIME * N_BASELINE
    expected = 1.0 / np.sqrt(2 * n * w0)
    np.testing.assert_allclose(stats["sensitivity"], [[expected, expected]])
    np.testing.assert_allclose(stats["sensitivity_natural"], [[expected, expected]])
    np.testing.assert_allclose(stats["sum_of_weights"], [[n * w0, n * w0]])
    np.testing.assert_allclose(stats["effective_number_of_visibilities"], [[n, n]])
    np.testing.assert_array_equal(sums["n_valid"], [[n, n]])


def test_stokes_combination_of_unequal_correlations():
    """sigma_S**2 = sum_c |M_Sc|**2 sigma_c**2 for the parallel hands and for
    four correlations (V has imaginary matrix entries)."""
    n = N_TIME * N_BASELINE
    w = np.array([1.0, 4.0, 9.0, 16.0])  # XX, XY, YX, YY
    weight = np.broadcast_to(w, (N_TIME, N_BASELINE, 1, 4)).copy()
    flag = np.zeros(weight.shape, dtype=bool)
    variance = 1.0 / (n * w)
    _, stats = _statistics(
        weight, weight, flag, ["XX", "XY", "YX", "YY"], ["I", "Q", "U", "V"]
    )
    expected_i = np.sqrt((variance[0] + variance[3]) / 4)
    expected_u = np.sqrt((variance[1] + variance[2]) / 4)
    np.testing.assert_allclose(
        stats["sensitivity"], [[expected_i, expected_i, expected_u, expected_u]]
    )
    np.testing.assert_allclose(
        stats["sum_of_weights"],
        [[n * (w[0] + w[3]) / 2] * 2 + [n * (w[1] + w[2]) / 2] * 2],
    )

    # Two circular hands: I = (RR + LL) / 2, V = (RR - LL) / 2; the columns
    # follow the data order, not the matrix order.
    weight = np.broadcast_to(w[[3, 0]], (N_TIME, N_BASELINE, 1, 2)).copy()
    _, stats = _statistics(
        weight, weight, np.zeros(weight.shape, bool), ["LL", "RR"], ["V", "I"]
    )
    expected = np.sqrt((variance[0] + variance[3]) / 4)
    np.testing.assert_allclose(stats["sensitivity"], [[expected, expected]])


def test_arbitrary_imaging_weights_against_brute_force():
    """Briggs-like imaging weights: the closed form of the module docstring,
    and never below the natural sensitivity (Cauchy-Schwarz)."""
    rng = np.random.default_rng(3)
    weight = rng.uniform(0.5, 2.0, size=(N_TIME, N_BASELINE, 2, 2))
    weight_imaging = rng.uniform(0.1, 3.0, size=weight.shape)
    flag = np.zeros(weight.shape, dtype=bool)
    _, stats = _statistics(weight, weight_imaging, flag, ["XX", "YY"], ["I", "Q"])
    sigma_c2 = (weight_imaging**2 / weight).sum(axis=(0, 1, 2)) / (
        weight_imaging.sum(axis=(0, 1, 2)) ** 2
    )
    expected = np.sqrt(sigma_c2.sum() / 4)
    np.testing.assert_allclose(stats["sensitivity"], [[expected, expected]])
    assert np.all(stats["sensitivity"] >= stats["sensitivity_natural"])
    neff = weight_imaging.sum(axis=(0, 1, 2)) ** 2 / (weight_imaging**2).sum(
        axis=(0, 1, 2)
    )
    np.testing.assert_allclose(
        stats["effective_number_of_visibilities"], [[neff.mean(), neff.mean()]]
    )


def test_flagged_and_nan_weights_are_left_out():
    """Flagged samples, NaN or zero natural weights and NaN imaging weights do
    not enter the sums; the flagged fraction counts every excluded sample
    except a NaN imaging weight (a weighting outcome, not a flag)."""
    rng = np.random.default_rng(5)
    weight = rng.uniform(0.5, 2.0, size=(N_TIME, N_BASELINE, 1, 2))
    weight_imaging = weight.copy()
    flag = np.zeros(weight.shape, dtype=bool)
    flag[0, 0, 0, 0] = True
    weight[1, 1, 0, 1] = np.nan
    weight[2, 2, 0, 0] = 0.0
    weight_imaging[0, 3, 0, 1] = np.nan
    valid = valid_sample_mask(weight, weight_imaging, flag)
    assert valid.sum() == weight.size - 4
    sums, stats = _statistics(weight, weight_imaging, flag, ["XX", "YY"], ["I", "Q"])
    kept_w = np.where(valid, weight, 0.0)
    np.testing.assert_allclose(sums["sum_weight"][0], kept_w.sum(axis=(0, 1, 2)))
    np.testing.assert_allclose(
        sums["sum_weight_imaging"][0], kept_w.sum(axis=(0, 1, 2))
    )
    expected = np.sqrt((1.0 / kept_w.sum(axis=(0, 1, 2))).sum() / 4)
    np.testing.assert_allclose(stats["sensitivity_natural"], [[expected, expected]])
    np.testing.assert_allclose(stats["sensitivity"], [[expected, expected]])
    assert sums["n_flagged"][0] == 3
    assert sums["n_samples"][0] == weight.size
    assert np.isfinite(stats["sensitivity"]).all()


def test_fully_flagged_channel_and_channel_mapping():
    """Visibility channels add onto their image channel; a channel without
    valid samples has an infinite sensitivity and no effective visibilities."""
    weight = np.ones((N_TIME, N_BASELINE, 3, 2))
    flag = np.zeros(weight.shape, dtype=bool)
    flag[:, :, 2] = True
    channel_map = np.array([0, 0, 1])
    sums, stats = _statistics(
        weight, weight, flag, ["XX", "YY"], ["I", "Q"], channel_map
    )
    n = N_TIME * N_BASELINE
    np.testing.assert_allclose(stats["sensitivity"][0], 1.0 / np.sqrt(2 * 2 * n))
    assert np.isinf(stats["sensitivity"][1]).all()
    np.testing.assert_array_equal(stats["effective_number_of_visibilities"][1], 0.0)
    np.testing.assert_array_equal(sums["n_flagged"], [0, 2 * n])


def test_uv_extent_in_wavelengths():
    """uv distance and |w| scale with each visibility channel's frequency;
    samples without a valid correlation are left out."""
    uvw = np.zeros((1, 3, 3))
    uvw[0, 0] = [3.0, 4.0, -7.0]  # 5 m
    uvw[0, 1] = [6.0, 8.0, 1.0]  # 10 m
    uvw[0, 2] = [30.0, 40.0, 100.0]  # 50 m, flagged below
    frequency = np.array([1.0e9, 2.0e9])
    valid = np.ones((1, 3, 2, 2), dtype=bool)
    valid[0, 2] = False
    extent = accumulate_uv_extent(uvw, frequency, valid, np.array([0, 1]), 2)
    per_metre = frequency / SPEED_OF_LIGHT
    np.testing.assert_allclose(extent["uv_distance_max"], 10.0 * per_metre)
    np.testing.assert_allclose(extent["uv_distance_min"], 5.0 * per_metre)
    np.testing.assert_allclose(extent["abs_w_max"], 7.0 * per_metre)


def _measurement_set(
    weight, flag, uvw, antenna1, antenna2, frequency, pols, weight_imaging=None
):
    ds = xr.Dataset(
        {
            "WEIGHT": (("time", "baseline_id", "frequency", "polarization"), weight),
            "WEIGHT_IMAGING": (
                ("time", "baseline_id", "frequency", "polarization"),
                weight.copy() if weight_imaging is None else weight_imaging,
            ),
            "FLAG": (("time", "baseline_id", "frequency", "polarization"), flag),
            "UVW": (("time", "baseline_id", "uvw_label"), uvw),
        },
        coords={
            "frequency": frequency,
            "polarization": pols,
            "baseline_antenna1_name": ("baseline_id", antenna1),
            "baseline_antenna2_name": ("baseline_id", antenna2),
        },
        attrs={
            "data_groups": {
                "base": {
                    "weight": "WEIGHT",
                    "weight_imaging": "WEIGHT_IMAGING",
                    "flag": "FLAG",
                    "uvw": "UVW",
                }
            }
        },
    )
    return xr.DataTree(ds)


def test_calculate_weight_statistics_single_field_leaves_out_autocorrelations():
    """Two measurement sets (different correlation order) onto one image
    channel; the autocorrelation baseline, with a huge weight, is ignored."""
    weight = np.full((2, 3, 1, 2), 4.0)
    weight[:, 0] = 1.0e6  # the autocorrelation
    flag = np.zeros(weight.shape, dtype=bool)
    uvw = np.zeros((2, 3, 3))
    uvw[:, 1] = [100.0, 0.0, 5.0]
    uvw[:, 2] = [0.0, 200.0, -10.0]
    antenna1 = ["a", "a", "b"]
    antenna2 = ["a", "b", "c"]
    ps = {
        "ms_0": _measurement_set(
            weight, flag, uvw, antenna1, antenna2, [1.0e11], ["XX", "YY"]
        ),
        "ms_1": _measurement_set(
            weight, flag, uvw, antenna1, antenna2, [1.0e11], ["YY", "XX"]
        ),
    }
    img_xds = xr.Dataset(coords={"frequency": [1.0e11], "polarization": ["I", "Q"]})
    stats = calculate_weight_statistics_single_field(ps, img_xds, "base")
    n = 2 * 2 * 2  # measurement sets x times x cross baselines
    expected = 1.0 / np.sqrt(2 * n * 4.0)
    np.testing.assert_allclose(stats.SENSITIVITY.values, [[expected, expected]])
    np.testing.assert_allclose(stats.SENSITIVITY_NATURAL.values, [[expected] * 2])
    np.testing.assert_allclose(stats.SUM_OF_WEIGHTS.values, [[n * 4.0] * 2])
    np.testing.assert_allclose(stats.EFFECTIVE_NUMBER_OF_VISIBILITIES.values, [[n] * 2])
    per_metre = 1.0e11 / SPEED_OF_LIGHT
    np.testing.assert_allclose(stats.UV_DISTANCE_MAX.values, [200.0 * per_metre])
    np.testing.assert_allclose(stats.UV_DISTANCE_MIN.values, [100.0 * per_metre])
    np.testing.assert_allclose(stats.ABS_W_MAX.values, [10.0 * per_metre])
    np.testing.assert_array_equal(stats.FLAGGED_FRACTION.values, [0.0])
    assert list(stats.polarization.values) == ["I", "Q"]


def _blocking_processing_set():
    """7 integrations of three cross baselines and 5 of one (correlations
    reversed), each with an autocorrelation, three visibility channels onto
    two image channels; NaN, zero and negative weights, NaN imaging weights
    and flags."""
    rng = np.random.default_rng(11)
    ps = {}
    for name, n_time, antenna1, antenna2, pols in (
        ("ms_0", 7, ["a", "a", "a", "b"], ["a", "b", "c", "c"], ["XX", "YY"]),
        ("ms_1", 5, ["a", "a"], ["a", "b"], ["YY", "XX"]),
    ):
        shape = (n_time, len(antenna1), 3, 2)
        weight = rng.uniform(0.5, 2.0, shape)
        weight_imaging = weight * rng.uniform(0.1, 3.0, shape)
        weight.flat[rng.choice(weight.size, 3, replace=False)] = [np.nan, 0.0, -1.0]
        weight_imaging.flat[rng.integers(weight.size)] = np.nan
        flag = rng.random(shape) < 0.1
        uvw = rng.normal(0.0, 300.0, (n_time, len(antenna1), 3))
        frequency = [1.0e11, 1.005e11, 1.02e11]
        ps[name] = _measurement_set(
            weight, flag, uvw, antenna1, antenna2, frequency, pols, weight_imaging
        )
    img_xds = xr.Dataset(
        coords={"frequency": [1.0e11, 1.02e11], "polarization": ["I", "Q"]}
    )
    return ps, img_xds


@pytest.mark.parametrize(
    "block_samples, blocks",
    [
        (1, [1] * 12),
        (20, [1] * 7 + [3, 2]),
        (40, [2, 2, 2, 1, 5]),
        (125, [6, 1, 5]),
        (126, [7, 5]),
        (weight_statistics.BLOCK_SAMPLES, [7, 5]),
    ],
)
def test_weight_statistics_in_blocks_of_integrations(
    monkeypatch, block_samples, blocks
):
    """Blocks of whole integrations (18 samples per integration in ms_0, 6 in
    ms_1, where a block of one integration is one row) against one block per
    measurement set: the same statistics bit for bit when every measurement
    set fits in one block, otherwise the uv extent and the flagged fraction
    bit for bit and the sums to rounding, also for blocks that do not divide
    the integrations."""
    ps, img_xds = _blocking_processing_set()
    monkeypatch.setattr(weight_statistics, "BLOCK_SAMPLES", 2**40)
    reference = calculate_weight_statistics_single_field(ps, img_xds, "base")

    accumulate = weight_statistics.accumulate_correlation_weight_sums
    block_times = []

    def counting_accumulate(weight, *args):
        block_times.append(weight.shape[0])
        return accumulate(weight, *args)

    monkeypatch.setattr(
        weight_statistics, "accumulate_correlation_weight_sums", counting_accumulate
    )
    monkeypatch.setattr(weight_statistics, "BLOCK_SAMPLES", block_samples)
    stats = calculate_weight_statistics_single_field(ps, img_xds, "base")
    assert block_times == blocks
    if len(blocks) == len(ps):
        xr.testing.assert_identical(stats, reference)
        return
    for name in ("UV_DISTANCE_MAX", "UV_DISTANCE_MIN", "ABS_W_MAX", "FLAGGED_FRACTION"):
        np.testing.assert_array_equal(stats[name].values, reference[name].values)
    for name in (
        "SENSITIVITY",
        "SENSITIVITY_NATURAL",
        "SUM_OF_WEIGHTS",
        "EFFECTIVE_NUMBER_OF_VISIBILITIES",
    ):
        np.testing.assert_allclose(
            stats[name].values, reference[name].values, rtol=1e-13, atol=0
        )


@pytest.mark.parametrize("bad", ["UU"])
def test_unknown_correlation_raises(bad):
    weight = np.ones((1, 1, 1, 2))
    with pytest.raises(ValueError):
        _statistics(weight, weight, np.zeros(weight.shape, bool), ["XX", bad], ["I"])
