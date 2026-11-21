"""Unit tests for astroviper.processing_functions.image_analysis.statistics.chauvenet."""

import numpy as np

from astroviper.processing_functions.image_analysis.statistics.chauvenet import (
    CHAUVENET_MAX_ITERATIONS,
    chauvenet_sample,
)


def test_chauvenet_z_score_near_casacore_table_points():
    # casacore's table: 185 points reach a z-score of 3, 7893 one of 4.
    rng = np.random.default_rng(3)
    for n_points, z_score in ((185, 3.0), (7893, 4.0)):
        values = rng.normal(0.0, 1.0, n_points)
        values[0] = (z_score - 0.1) * values.std(ddof=1) + values.mean()
        assert values[0] in chauvenet_sample(values)
        values[0] = (z_score + 0.2) * values.std(ddof=1) + values.mean()
        assert values[0] not in chauvenet_sample(values)


def test_chauvenet_removes_a_source():
    i, j = np.mgrid[:96, :96]
    source = 300.0 * np.exp(-4 * np.log(2) * ((i - 40) ** 2 + (j - 40) ** 2) / 36.0)
    plane = np.random.default_rng(1).normal(0.0, 1.0, (96, 96)) + source
    sample = chauvenet_sample(plane.ravel())
    assert sample.size < plane.size
    assert sample.max() < 6.0


def test_fewer_than_two_values_are_returned_unchanged():
    for values in (np.array([]), np.array([5.0])):
        assert chauvenet_sample(values) is values


def test_values_without_outliers_are_returned_unchanged():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert chauvenet_sample(values) is values


def test_kept_values_keep_their_order_and_dtype():
    values = np.random.default_rng(2).normal(0.0, 1.0, 50).astype(np.float32)
    values[[7, 31]] = [40.0, -35.0]
    kept = chauvenet_sample(values)
    assert kept.dtype == np.float32
    np.testing.assert_array_equal(kept, np.delete(values, [7, 31]))


def test_at_most_six_passes():
    noise = np.random.default_rng(0).normal(0.0, 1.0, 200)
    # Each pass removes only the largest of these outliers.
    values = np.concatenate([noise, 10.0 ** np.arange(1, 10)])
    kept = chauvenet_sample(values)
    assert kept.size == values.size - (CHAUVENET_MAX_ITERATIONS + 1)
    assert kept.max() == 1000.0
