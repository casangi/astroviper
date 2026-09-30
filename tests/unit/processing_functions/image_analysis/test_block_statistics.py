"""Block wise plane statistics: the same numbers as the whole plane formulas,
without a copy of the plane."""

import tracemalloc

import numpy as np
import pytest

from astroviper.processing_functions.image_analysis import image_statistics as imgstats
from astroviper.processing_functions.imaging.deconvolution import _hogbom_peak_cube


def _reference_peak_abs_signed(plane, mask=None):
    """The formula the helper replaces."""
    if mask is None:
        absvals = np.abs(plane)
    else:
        valid = mask > 0.5
        if not np.any(valid):
            return float("nan")
        absvals = np.where(valid, np.abs(plane), np.nan)
    if np.all(np.isnan(absvals)):
        return float("nan")
    return float(plane[np.unravel_index(np.nanargmax(absvals), absvals.shape)])


@pytest.fixture
def plane():
    rng = np.random.default_rng(3)
    # more rows than one block holds, so that several blocks are scanned
    plane = rng.standard_normal((2100, 600))
    plane[1500, 17] = -9.0  # the absolute peak is negative
    plane[100, 5] = 8.5
    return plane


@pytest.fixture
def mask(plane):
    rng = np.random.default_rng(4)
    mask = rng.random(plane.shape) > 0.3
    mask[1500, 17] = False  # the absolute peak is masked
    return mask


class TestPeakAbsSigned:
    def test_matches_the_whole_plane_formula(self, plane, mask):
        assert imgstats.plane_peak_abs_signed(plane) == _reference_peak_abs_signed(
            plane
        )
        assert imgstats.plane_peak_abs_signed(plane) == -9.0
        assert imgstats.plane_peak_abs_signed(
            plane, mask
        ) == _reference_peak_abs_signed(plane, mask)

    def test_first_of_equal_peaks_wins(self):
        plane = np.zeros((5000, 3))
        plane[4000, 1] = 2.0
        plane[10, 2] = -2.0
        assert imgstats.plane_peak_abs_signed(plane) == -2.0

    def test_nan_pixels_are_ignored(self, plane):
        plane[0, 0] = np.nan
        plane[1500, 17] = np.nan
        assert imgstats.plane_peak_abs_signed(plane) == 8.5
        assert np.isnan(imgstats.plane_peak_abs_signed(np.full((4, 4), np.nan)))

    def test_fully_masked_plane_gives_nan(self, plane):
        assert np.isnan(imgstats.plane_peak_abs_signed(plane, np.zeros_like(plane)))

    def test_single_precision_and_float_masks(self, plane, mask):
        plane32 = plane.astype(np.float32)
        expected = _reference_peak_abs_signed(plane32, mask.astype(np.float32))
        assert (
            imgstats.plane_peak_abs_signed(plane32, mask.astype(np.float32)) == expected
        )


class TestAbsMaxAndSum:
    def test_abs_max(self, plane):
        assert imgstats.plane_abs_max(plane) == np.abs(plane).max()
        plane[3, 3] = np.nan
        assert np.isnan(imgstats.plane_abs_max(plane))

    def test_abs_sum(self, plane):
        assert imgstats.plane_abs_sum(plane) == pytest.approx(
            np.abs(plane).sum(), rel=1e-12
        )

    def test_cube_plane_abs_max(self):
        rng = np.random.default_rng(5)
        cube = rng.standard_normal((1, 3, 2, 300, 4000))
        np.testing.assert_array_equal(
            imgstats.cube_plane_abs_max(cube), np.abs(cube).max(axis=(-2, -1))
        )


class TestHogbomPeakCube:
    def test_matches_the_whole_cube_formula(self):
        rng = np.random.default_rng(6)
        cube = rng.standard_normal((1, 2, 2, 700, 3000))
        mask = rng.random(cube.shape) > 0.2
        clean_box = (5, 2900, -1, 650)
        search = np.abs(cube[..., :650, 5:2900])
        expected = np.where(mask[..., :650, 5:2900], search, 0.0).max(axis=(-2, -1))
        np.testing.assert_array_equal(
            _hogbom_peak_cube(cube, mask, clean_box), expected
        )
        np.testing.assert_array_equal(
            _hogbom_peak_cube(cube, None, (-1, -1, -1, -1)),
            np.abs(cube).max(axis=(-2, -1)),
        )

    def test_fully_masked_plane_gives_zero(self):
        cube = np.ones((1, 1, 1, 10, 10))
        assert (
            _hogbom_peak_cube(cube, np.zeros(cube.shape, dtype=bool), (-1, -1, -1, -1))
            == 0.0
        )


class TestNoPlaneSizedTemporaries:
    """Each helper allocates far less than the plane it scans."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda plane, mask: imgstats.plane_peak_abs_signed(plane, mask),
            lambda plane, mask: imgstats.plane_peak_abs_signed(plane),
            lambda plane, mask: imgstats.plane_abs_max(plane),
            lambda plane, mask: imgstats.plane_abs_sum(plane),
            lambda plane, mask: _hogbom_peak_cube(
                plane[None, None, None], mask[None, None, None], (-1, -1, -1, -1)
            ),
        ],
    )
    def test_peak_allocation_below_a_tenth_of_the_plane(self, call):
        plane = np.random.default_rng(7).standard_normal((3000, 3000))  # 72 MB
        mask = plane > -1.0
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            call(plane, mask)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 0.1 * plane.nbytes, f"{peak / plane.nbytes:.2f} planes allocated"
