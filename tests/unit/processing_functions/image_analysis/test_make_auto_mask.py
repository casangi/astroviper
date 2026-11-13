"""Unit tests for astroviper.processing_functions.image_analysis.make_auto_mask."""

import math

import numpy as np
import pytest
import scipy.signal
import xarray as xr

from astroviper.processing_functions.image_analysis.make_auto_mask import (
    AUTO_MASK_DEFAULTS,
    _bounding_box,
    _chauvenet_sample,
    _grow,
    _location_and_noise,
    _plane_thresholds,
    _prune,
    _smooth_and_cut,
    make_auto_mask,
)
from astroviper.processing_functions.imaging.restore import (
    _elliptical_gaussian_kernel,
)

PIXEL = 1e-6
N = 96
# Clean beam of 6 pixels: an area of 40.8 pixels, 12.2 at the default fraction.
BEAM = (6 * PIXEL, 6 * PIXEL, 0.0)


def noise_plane(seed=1, sigma=1.0, n=N):
    return np.random.default_rng(seed).normal(0.0, sigma, (n, n))


def gaussian(center, peak, fwhm=6.0, n=N):
    i, j = np.mgrid[:n, :n]
    distance = (i - center[0]) ** 2 + (j - center[1]) ** 2
    return peak * np.exp(-4 * math.log(2) * distance / fwhm**2)


def image(residual, beam=BEAM, sidelobe=0.05, primary_beam=None, mask=None):
    """Image of ``residual``, an ``(l, m)`` plane or the full five dimensions."""
    residual = np.asarray(residual, dtype=float)
    if residual.ndim == 2:
        residual = residual[None, None, None]
    planes = residual.shape[:3]
    dims = ("time", "frequency", "polarization")
    group = {
        "sky": "SKY_RESIDUAL",
        "beam_fit_params_point_spread_function": "BEAM",
        "max_sidelobe_point_spread_function": "SIDELOBE",
    }
    xds = xr.Dataset(
        {
            "SKY_RESIDUAL": ((*dims, "l", "m"), residual),
            "BEAM": (
                (*dims, "beam_param"),
                np.broadcast_to(np.asarray(beam, dtype=float), (*planes, 3)),
            ),
            "SIDELOBE": (dims, np.full(planes, sidelobe)),
        },
        coords={
            "time": np.arange(planes[0]),
            "frequency": np.arange(planes[1]),
            "polarization": np.arange(planes[2]),
            "l": np.arange(residual.shape[3]) * PIXEL,
            "m": np.arange(residual.shape[4]) * PIXEL,
        },
    )
    if primary_beam is not None:
        xds["PRIMARY_BEAM"] = (
            (*dims, "l", "m"),
            np.broadcast_to(primary_beam, residual.shape),
        )
        group["primary_beam"] = "PRIMARY_BEAM"
    if mask is not None:
        xds["MASK"] = ((*dims, "l", "m"), np.broadcast_to(mask, residual.shape))
        group["mask"] = "MASK"
    xds.attrs["data_groups"] = {"residual": group}
    return xds


def plane_mask(xds):
    return xds.MASK.values[0, 0, 0]


def test_the_defaults_are_those_of_tclean():
    assert AUTO_MASK_DEFAULTS == {
        "sidelobe_threshold": 3.0,
        "noise_threshold": 5.0,
        "low_noise_threshold": 1.5,
        "negative_threshold": 0.0,
        "smooth_factor": 1.0,
        "min_beam_fraction": 0.3,
        "cut_threshold": 0.01,
        "grow_iterations": 75,
        "grow_prune": True,
        "min_percent_change": -1.0,
        "fast_noise": True,
    }


class TestConfiguration:
    def test_a_direct_call_is_checked(self):
        with pytest.raises(ValueError, match="Unknown auto_mask_params keys"):
            make_auto_mask(image(noise_plane()), {"noise": 5.0})
        with pytest.raises(ValueError, match="primary_beam_limit"):
            make_auto_mask(image(noise_plane()), {}, primary_beam_limit=2.0)

    def test_a_checked_call_is_not(self):
        make_auto_mask(image(noise_plane()), {"noise": 5.0}, input_checked=True)

    def test_the_parameters_given_are_not_changed(self):
        params = {"noise_threshold": 4.25}
        make_auto_mask(image(noise_plane()), params)
        assert params == {"noise_threshold": 4.25}


def test_the_noise_stays_inside_the_primary_beam():
    plane = noise_plane()
    plane[:, 48:] *= 20.0
    support = np.zeros_like(plane, dtype=bool)
    support[:, :48] = True
    previous = np.zeros_like(support)
    previous[:8, :8] = True
    for fast_noise in (True, False):
        _, noise = _location_and_noise(plane, support, previous, fast_noise)
        assert abs(noise - 1.0) < 0.1


class TestStatistics:
    def test_thresholds_set_by_the_sidelobes(self):
        params = AUTO_MASK_DEFAULTS
        mask, grow, negative, noise_sets = _plane_thresholds(
            10.0, 0.1, 1.0, 0.5, params
        )
        assert (mask, grow, negative) == pytest.approx((4.0, 4.0, -2.0))
        assert not noise_sets

    def test_thresholds_set_by_the_noise(self):
        params = {**AUTO_MASK_DEFAULTS, "negative_threshold": 4.0}
        mask, grow, negative, noise_sets = _plane_thresholds(
            10.0, 0.01, 1.0, 1.0, params
        )
        assert (mask, grow, negative) == pytest.approx((6.0, 2.5, -3.0))
        assert noise_sets

    def test_chauvenet_z_score_matches_casacore(self):
        # casacore's table: 185 points reach a z-score of 3, 7893 one of 4.
        rng = np.random.default_rng(3)
        for n_points, z_score in ((185, 3.0), (7893, 4.0)):
            values = rng.normal(0.0, 1.0, n_points)
            values[0] = (z_score - 0.1) * values.std(ddof=1) + values.mean()
            assert values[0] in _chauvenet_sample(values)
            values[0] = (z_score + 0.2) * values.std(ddof=1) + values.mean()
            assert values[0] not in _chauvenet_sample(values)

    def test_chauvenet_removes_a_source(self):
        plane = noise_plane() + gaussian((40, 40), 300.0)
        sample = _chauvenet_sample(plane.ravel())
        assert sample.size < plane.size
        assert sample.max() < 6.0

    def test_fast_noise_takes_every_pixel(self):
        plane = noise_plane() + gaussian((40, 40), 300.0, fwhm=14.0)
        support = np.ones_like(plane, dtype=bool)
        previous = np.zeros_like(support)
        fast = _location_and_noise(plane, support, previous, True)
        slow = _location_and_noise(plane, support, previous, False)
        assert slow[1] == pytest.approx(1.0, abs=0.1)
        assert fast[1] > 1.1 * slow[1]

    def test_noise_outside_the_mask(self):
        plane = noise_plane() + gaussian((40, 40), 300.0, fwhm=30.0)
        support = np.ones_like(plane, dtype=bool)
        previous = gaussian((40, 40), 1.0, fwhm=30.0) > 1e-3
        location, noise = _location_and_noise(plane, support, previous, False)
        outside = plane[~previous]
        assert location == pytest.approx(np.median(outside))
        assert noise == pytest.approx(1.0, abs=0.1)

    def test_a_full_mask_falls_back_to_chauvenet(self):
        plane = noise_plane()
        support = np.ones_like(plane, dtype=bool)
        location, noise = _location_and_noise(plane, support, support, False)
        assert noise == pytest.approx(1.0, abs=0.1)


def test_prune_counts_four_connected_pixels():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True  # 4 pixels
    mask[5, 5] = mask[6, 6] = True  # 2 regions of 1 pixel, touching at a corner
    expected = np.zeros_like(mask)
    expected[1:3, 1:3] = True
    np.testing.assert_array_equal(_prune(mask, 2), expected)
    np.testing.assert_array_equal(_prune(mask, 4), expected)
    np.testing.assert_array_equal(_prune(mask, 5), np.zeros_like(mask))
    np.testing.assert_array_equal(_prune(mask, 1), mask)


def test_bounding_box():
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:4, 5] = True
    assert _bounding_box(mask) == (slice(2, 4), slice(5, 6))
    assert _bounding_box(mask, 3) == (slice(0, 7), slice(2, 9))
    assert _bounding_box(mask, 20) == (slice(0, 10), slice(0, 12))


@pytest.mark.parametrize(
    "rows, columns",
    [
        (slice(40, 44), slice(50, 60)),
        (slice(0, 5), slice(0, 5)),
        (slice(N - 3, N), slice(10, N)),
        (slice(0, N), slice(0, N)),
    ],
)
def test_smoothing_the_box_of_the_mask_is_smoothing_the_plane(rows, columns):
    mask = np.zeros((N, N), dtype=bool)
    mask[rows, columns] = True
    mask[(rows.start + rows.stop) // 2, columns.start] = False
    kernel = _elliptical_gaussian_kernel(37, 37, 6.0, 3.0, 0.5, np.float64)
    plane = scipy.signal.fftconvolve(mask.astype(np.float64), kernel, mode="same")
    np.testing.assert_array_equal(
        _smooth_and_cut(mask, kernel, 0.2), plane > 0.2 * plane.max()
    )


def test_growth_goes_as_far_as_the_iterations():
    constraint = np.zeros((N, N), dtype=bool)
    constraint[40, 10:80] = True
    mask = np.zeros_like(constraint)
    mask[40, 30:32] = True
    mask[60, 60] = True  # outside the constraint, no seed
    expected = np.zeros_like(mask)
    expected[40, 25:37] = True
    np.testing.assert_array_equal(_grow(mask, constraint, 5), expected)


def test_prune_removes_only_the_pixels_of_a_region():
    mask = np.zeros((12, 12), dtype=bool)
    mask[1, 1:4] = mask[1:4, 3] = True  # 5 pixels in rows 1-3, columns 1-3
    mask[3:12, 1] = True  # 9 pixels, the tip (3, 1) inside that box
    expected = np.zeros_like(mask)
    expected[3:12, 1] = True
    np.testing.assert_array_equal(_prune(mask, 6), expected)


class TestMaskOfOnePlane:
    def test_masks_a_source_and_no_noise(self):
        xds = image(noise_plane() + gaussian((40, 60), 50.0))
        stopped = make_auto_mask(xds, {})
        mask = plane_mask(xds)
        assert mask[40, 60]
        assert not mask[:20].any() and not mask[-20:].any()
        assert not stopped[0, 0, 0]
        assert xds.MASK.dims == xds.SKY_RESIDUAL.dims
        assert xds.attrs["data_groups"]["residual"]["mask"] == "MASK"
        assert xds.MASK.attrs["description"] == "Automatic deconvolution mask."

    def test_small_regions_are_pruned(self):
        plane = noise_plane() + gaussian((40, 60), 50.0)
        plane[10, 10] = 40.0
        xds = image(plane)
        make_auto_mask(xds, {})
        assert not plane_mask(xds)[10, 10]
        xds = image(plane)
        make_auto_mask(xds, {"min_beam_fraction": 0.0})
        assert plane_mask(xds)[10, 10]

    def test_smoothing_reaches_past_the_thresholded_pixels(self):
        plane = noise_plane() + gaussian((40, 60), 50.0)
        threshold = 5.0 * 1.4826 * np.median(np.abs(plane - np.median(plane)))
        xds = image(plane)
        make_auto_mask(xds, {})
        n_mask = np.count_nonzero(plane_mask(xds))
        assert n_mask > 3 * np.count_nonzero(plane > threshold)
        xds = image(plane)
        make_auto_mask(xds, {"cut_threshold": 0.5})
        assert np.count_nonzero(plane_mask(xds)) < n_mask

    def test_the_smoothing_kernel_follows_the_beam(self):
        beam = (12 * PIXEL, 4 * PIXEL, 0.6)
        plane = np.zeros((N, N))
        plane[48, 48] = 100.0
        xds = image(plane, beam=beam, sidelobe=0.0)
        make_auto_mask(xds, {"min_beam_fraction": 0.0, "cut_threshold": 0.2})
        kernel = _elliptical_gaussian_kernel(N, N, 12.0, 4.0, 0.6, np.float64)
        np.testing.assert_array_equal(plane_mask(xds), kernel > 0.2)

    def test_smooth_factor_scales_the_kernel(self):
        beam = (12 * PIXEL, 4 * PIXEL, 0.6)
        plane = np.zeros((N, N))
        plane[48, 48] = 100.0
        xds = image(plane, beam=beam, sidelobe=0.0)
        make_auto_mask(
            xds, {"min_beam_fraction": 0.0, "cut_threshold": 0.2, "smooth_factor": 0.5}
        )
        kernel = _elliptical_gaussian_kernel(N, N, 6.0, 2.0, 0.6, np.float64)
        np.testing.assert_array_equal(plane_mask(xds), kernel > 0.2)

    def test_nothing_above_the_threshold_gives_an_empty_mask(self):
        xds = image(noise_plane())
        make_auto_mask(xds, {"noise_threshold": 50.0})
        assert not plane_mask(xds).any()

    def test_primary_beam_limit_bounds_mask_and_peak(self):
        primary_beam = np.ones((N, N))
        primary_beam[:, 70:] = 0.1
        primary_beam[:, 90:] = 0.0
        # the source outside the limit would set a sidelobe level of 150
        plane = (
            noise_plane() + gaussian((40, 40), 40.0) + gaussian((40, 80), 1000.0, 6.0)
        )
        xds = image(plane, sidelobe=0.05, primary_beam=primary_beam)
        make_auto_mask(xds, {}, primary_beam_limit=0.2)
        mask = plane_mask(xds)
        assert mask[40, 40]
        assert not mask[:, 70:].any()
        xds = image(plane, sidelobe=0.05, primary_beam=primary_beam)
        make_auto_mask(xds, {})
        assert plane_mask(xds)[40, 80]
        assert not plane_mask(xds)[:, 90:].any()

    def test_without_pruning_emission_beyond_the_limit_smooths_in(self):
        """As in CASA, only pruning clips to the limit before the smoothing."""
        primary_beam = np.ones((N, N))
        primary_beam[:, 70:] = 0.1
        plane = noise_plane() + gaussian((40, 76), 50.0)
        xds = image(plane, primary_beam=primary_beam)
        make_auto_mask(xds, {"min_beam_fraction": 0.0}, primary_beam_limit=0.2)
        assert plane_mask(xds)[40, 69]
        assert not plane_mask(xds)[:, 70:].any()
        xds = image(plane, primary_beam=primary_beam)
        make_auto_mask(xds, {}, primary_beam_limit=0.2)
        assert not plane_mask(xds).any()

    def test_non_finite_pixels_stay_outside(self):
        plane = noise_plane() + gaussian((40, 60), 50.0)
        plane[40, 60] = np.nan
        plane[:, :5] = np.inf
        xds = image(plane)
        make_auto_mask(xds, {})
        mask = plane_mask(xds)
        assert not mask[40, 60] and not mask[:, :5].any()
        assert mask[40, 62]

    def test_a_plane_without_a_beam_keeps_its_mask(self):
        previous = gaussian((70, 20), 1.0) > 0.5
        xds = image(
            noise_plane() + gaussian((40, 60), 50.0),
            beam=(np.nan, 0.0, 0.0),
            mask=previous,
        )
        stopped = make_auto_mask(xds, {})
        np.testing.assert_array_equal(plane_mask(xds), previous)
        assert not stopped[0, 0, 0]

    def test_a_negative_peak_sets_the_sidelobe_level(self):
        plane = noise_plane() + gaussian((40, 60), 30.0) - gaussian((70, 20), 300.0)
        xds = image(plane, sidelobe=0.05)
        make_auto_mask(xds, {})
        assert not plane_mask(xds)[40, 60]

    def test_a_plane_without_finite_pixels_stays_empty(self):
        xds = image(np.full((N, N), np.nan))
        stopped = make_auto_mask(xds, {})
        assert not plane_mask(xds).any()
        assert not stopped[0, 0, 0]

    def test_negative_features_are_smoothed(self):
        plane = np.zeros((N, N))
        plane[48, 48] = -100.0
        xds = image(plane, beam=(12 * PIXEL, 4 * PIXEL, 0.6), sidelobe=0.0)
        make_auto_mask(xds, {"negative_threshold": 4.0, "cut_threshold": 0.2})
        kernel = _elliptical_gaussian_kernel(N, N, 12.0, 4.0, 0.6, np.float64)
        np.testing.assert_array_equal(plane_mask(xds), kernel > 0.2)

    def test_negative_features(self):
        plane = noise_plane() + gaussian((40, 60), 50.0) - gaussian((70, 20), 30.0)
        xds = image(plane)
        make_auto_mask(xds, {})
        assert not plane_mask(xds)[70, 20]
        xds = image(plane)
        make_auto_mask(xds, {"negative_threshold": 4.0})
        assert plane_mask(xds)[70, 20]


class TestUpdates:
    def first_and_faint(self):
        """A masked source, then its residual with a faint tail and a faint island."""
        first = noise_plane(1, 0.1) + gaussian((40, 40), 10.0)
        faint = noise_plane(2, 0.1)
        # between the growth threshold (1.5 sigma) and the mask threshold (5)
        faint[42:62, 36:44] = 0.3  # starts in the mask of the source
        faint[70:82, 70:82] = 0.3  # stands alone
        return first, faint

    def run(self, params):
        first, faint = self.first_and_faint()
        xds = image(first)
        make_auto_mask(xds, params)
        previous = plane_mask(xds).copy()
        second = image(faint, mask=previous)
        make_auto_mask(second, params)
        return previous, plane_mask(second)

    def test_a_mask_never_loses_pixels(self):
        previous, mask = self.run({})
        assert previous.any()
        assert mask[previous].all()

    def test_grows_through_faint_emission_it_touches(self):
        previous, mask = self.run({})
        assert not previous[57, 40]
        assert mask[57, 40]
        assert not mask[75, 75]

    def test_no_growth_without_a_mask(self):
        _, faint = self.first_and_faint()
        xds = image(faint)
        make_auto_mask(xds, {})
        assert not plane_mask(xds).any()

    def test_grow_iterations_limits_the_growth(self):
        _, all_of_it = self.run({})
        _, none_of_it = self.run({"grow_iterations": 0})
        _, some_of_it = self.run({"grow_iterations": 2})
        grown = [np.count_nonzero(m) for m in (none_of_it, some_of_it, all_of_it)]
        assert grown[0] < grown[1] < grown[2]

    def test_grow_prune_removes_small_growth(self):
        previous = np.zeros((N, N), dtype=bool)
        previous[40:42, 40:42] = True
        faint = noise_plane(2, 0.1)
        faint[40:42, 40:42] = 0.3  # 4 pixels, fewer than the 12.2 of the prune size
        pruned = image(faint, mask=previous)
        make_auto_mask(pruned, {})
        np.testing.assert_array_equal(plane_mask(pruned), previous)
        kept = image(faint, mask=previous)
        make_auto_mask(kept, {"grow_prune": False})
        assert np.count_nonzero(plane_mask(kept)) > 4

    def test_new_emission_and_growth_in_one_update(self):
        first, _ = self.first_and_faint()
        xds = image(first)
        make_auto_mask(xds, {})
        second = noise_plane(2, 0.1) + gaussian((75, 75), 5.0)
        second[42:62, 36:44] = 0.3
        xds = image(second, mask=plane_mask(xds).copy(), sidelobe=0.01)
        make_auto_mask(xds, {})
        assert plane_mask(xds)[75, 75]
        assert plane_mask(xds)[57, 40]

    def test_growth_starts_inside_the_constraint(self):
        previous = np.zeros((N, N), dtype=bool)
        previous[40:48, 40:48] = True
        plane = noise_plane(2, 0.1)
        plane[40:48, 40:48] = 0.0
        plane[48:60, 40:48] = 0.3
        xds = image(plane, mask=previous)
        make_auto_mask(xds, {})
        assert not plane_mask(xds)[55, 44]

    def test_a_masked_source_is_not_masked_again(self):
        plane = noise_plane() + gaussian((40, 60), 50.0)
        xds = image(plane)
        make_auto_mask(xds, {})
        first = plane_mask(xds).copy()
        make_auto_mask(xds, {"grow_iterations": 0})
        np.testing.assert_array_equal(plane_mask(xds), first)


class TestStops:
    def test_an_empty_mask_stops_when_the_noise_sets_the_threshold(self):
        xds = image(noise_plane())
        stopped = make_auto_mask(xds, {"noise_threshold": 50.0})
        assert stopped[0, 0, 0]

    def test_an_empty_mask_goes_on_when_the_sidelobes_set_the_threshold(self):
        xds = image(noise_plane(), sidelobe=1.0)
        stopped = make_auto_mask(xds, {"noise_threshold": 0.1})
        assert not plane_mask(xds).any()
        assert not stopped[0, 0, 0]

    @pytest.mark.parametrize(
        "percent, reached, expected",
        [
            (-1.0, True, False),
            (1.0, False, False),
            (1.0, True, True),
            (0.0, True, True),
        ],
    )
    def test_min_percent_change(self, percent, reached, expected):
        xds = image(noise_plane() + gaussian((40, 60), 50.0), sidelobe=0.01)
        make_auto_mask(xds, {})
        stopped = make_auto_mask(
            xds,
            {"min_percent_change": percent, "grow_iterations": 0},
            threshold_reached=np.full((1, 1, 1), reached),
        )
        assert stopped[0, 0, 0] == expected

    @pytest.mark.parametrize("percent, expected", [(1.0, False), (3.0, True)])
    def test_min_percent_change_is_a_percentage(self, percent, expected):
        plane = noise_plane() + gaussian((40, 60), 50.0)
        first = image(plane, sidelobe=0.01)
        make_auto_mask(first, {})
        previous = plane_mask(first).copy()
        rows, columns = np.nonzero(previous)
        previous[rows[:7], columns[:7]] = False  # 1.8 percent of the mask
        stopped = make_auto_mask(
            image(plane, mask=previous, sidelobe=0.01),
            {"min_percent_change": percent, "grow_iterations": 0},
            threshold_reached=np.full((1, 1, 1), True),
        )
        assert stopped[0, 0, 0] == expected

    def test_a_change_above_min_percent_change_goes_on(self):
        previous = gaussian((40, 60), 1.0) > 0.5
        xds = image(noise_plane() + gaussian((40, 60), 50.0), mask=previous, sidelobe=0)
        stopped = make_auto_mask(
            xds,
            {"min_percent_change": 1.0},
            threshold_reached=np.full((1, 1, 1), True),
        )
        assert not stopped[0, 0, 0]

    def test_a_stopped_plane_keeps_its_mask(self):
        previous = gaussian((70, 20), 1.0) > 0.5
        xds = image(noise_plane() + gaussian((40, 60), 50.0), mask=previous)
        stopped = make_auto_mask(xds, {}, planes_stopped=np.full((1, 1, 1), True))
        assert stopped[0, 0, 0]
        np.testing.assert_array_equal(plane_mask(xds), previous)

    def test_a_stopped_plane_does_not_stop_the_others(self):
        channels = np.stack([noise_plane(1) + gaussian((40, 60), 50.0)] * 2)
        stopped = np.zeros((1, 2, 1), dtype=bool)
        stopped[0, 0, 0] = True
        xds = image(channels[None, :, None])
        result = make_auto_mask(xds, {}, planes_stopped=stopped)
        assert not xds.MASK.values[0, 0, 0].any()
        assert xds.MASK.values[0, 1, 0, 40, 60]
        assert result[0, 0, 0] and not result[0, 1, 0]

    def test_the_stopped_planes_given_are_not_modified(self):
        given = np.zeros((1, 1, 1), dtype=bool)
        xds = image(noise_plane())
        stopped = make_auto_mask(xds, {"noise_threshold": 50.0}, planes_stopped=given)
        assert stopped[0, 0, 0] and not given[0, 0, 0]


class TestPruningWarning:
    @pytest.fixture
    def warnings(self, monkeypatch):
        messages = []
        monkeypatch.setattr("toolviper.utils.logger.warning", messages.append)
        return messages

    def test_the_sidelobe_level_decides_for_an_unresolved_source(self, warnings):
        plane = noise_plane(1, 0.01) + gaussian((40, 60), 50.0)
        kept = image(plane, sidelobe=0.70 / 3.0)
        make_auto_mask(kept, {})
        assert plane_mask(kept)[40, 60]
        assert not warnings
        pruned = image(plane, sidelobe=0.80 / 3.0)
        make_auto_mask(pruned, {})
        assert not plane_mask(pruned).any()
        assert len(warnings) == 1

    def test_the_message_gives_the_way_out(self, warnings):
        make_auto_mask(image(noise_plane(), sidelobe=0.27), {})
        assert "up to 0.81 in 1 of 1 planes" in warnings[0]
        assert "exp(-min_beam_fraction) = 0.74" in warnings[0]
        assert "sidelobe_threshold below 2.74" in warnings[0]
        assert "min_beam_fraction below 0.21" in warnings[0]

    def test_it_counts_the_planes(self, warnings):
        xds = image(np.stack([noise_plane()] * 3)[None, :, None])
        xds["SIDELOBE"].values[0, :, 0] = [0.1, 0.3, np.nan]
        make_auto_mask(xds, {})
        assert "in 1 of 3 planes" in warnings[0]

    def test_only_at_the_first_update(self, warnings):
        xds = image(noise_plane(), sidelobe=0.3)
        make_auto_mask(xds, {}, planes_stopped=np.zeros((1, 1, 1), dtype=bool))
        assert not warnings

    def test_without_pruning_there_is_nothing_to_warn_about(self, warnings):
        make_auto_mask(image(noise_plane(), sidelobe=0.9), {"min_beam_fraction": 0.0})
        assert not warnings


def test_the_mask_is_updated_in_place_when_it_can_be():
    previous = np.zeros((1, 1, 1, N, N), dtype=bool)
    previous[0, 0, 0, 70:75, 20:25] = True
    xds = image(noise_plane() + gaussian((40, 60), 50.0), mask=previous)
    xds["MASK"] = (xds.MASK.dims, previous)
    make_auto_mask(xds, {})
    assert np.shares_memory(xds.MASK.values, previous)
    assert previous[0, 0, 0, 40, 60]


def test_another_output_variable_leaves_the_mask_given():
    previous = np.zeros((1, 1, 1, N, N), dtype=bool)
    xds = image(noise_plane() + gaussian((40, 60), 50.0), mask=previous)
    xds["MASK"] = (xds.MASK.dims, previous)
    make_auto_mask(xds, {}, image_data_group_out_modified={"mask": "MASK_AUTOMATIC"})
    assert xds.MASK_AUTOMATIC.values[0, 0, 0, 40, 60]
    assert not xds.MASK.values.any()


def test_the_data_groups_name_the_variables():
    xds = image(noise_plane() + gaussian((40, 60), 50.0))
    xds = xds.rename({"SKY_RESIDUAL": "SKY"})
    xds.attrs["data_groups"]["residual"]["sky"] = "SKY"
    make_auto_mask(
        xds,
        {},
        image_data_group_out_name="masked",
        image_data_group_out_modified={"mask": "MASK_AUTOMATIC"},
    )
    assert xds.MASK_AUTOMATIC.values[0, 0, 0, 40, 60]
    assert xds.attrs["data_groups"]["masked"]["mask"] == "MASK_AUTOMATIC"
    assert "mask" not in xds.attrs["data_groups"]["residual"]
    assert "MASK" not in xds


def test_growth_stays_in_its_plane():
    previous = np.zeros((1, 2, 1, N, N), dtype=bool)
    previous[0, 0, 0, 40:48, 40:48] = True
    channels = np.stack([noise_plane(1, 0.1), noise_plane(2, 0.1)])
    channels[:, 40:60, 40:48] = 0.3
    xds = image(channels[None, :, None], mask=previous)
    make_auto_mask(xds, {})
    assert xds.MASK.values[0, 0, 0, 55, 44]
    assert not xds.MASK.values[0, 1, 0].any()


def test_every_plane_gets_its_own_mask():
    residual = np.stack(
        [
            np.stack(
                [
                    noise_plane(1 + f + 4 * p)
                    + gaussian((30 + 20 * f, 30 + 20 * p), 50.0)
                    for p in range(2)
                ]
            )
            for f in range(3)
        ]
    )[None]
    xds = image(residual)
    stopped = make_auto_mask(xds, {})
    assert stopped.shape == (1, 3, 2)
    for f in range(3):
        for p in range(2):
            mask = xds.MASK.values[0, f, p]
            assert mask[30 + 20 * f, 30 + 20 * p]
            for other_f, other_p in ((f, 1 - p), ((f + 1) % 3, p)):
                assert not mask[30 + 20 * other_f, 30 + 20 * other_p]


def test_single_precision_gives_the_same_mask():
    plane = noise_plane() + gaussian((40, 60), 50.0)
    double = image(plane)
    make_auto_mask(double, {"fast_noise": False})
    single = image(plane)
    single["SKY_RESIDUAL"] = single.SKY_RESIDUAL.astype(np.float32)
    make_auto_mask(single, {"fast_noise": False})
    np.testing.assert_array_equal(single.MASK.values, double.MASK.values)


def test_the_beam_may_come_in_any_dimension_order():
    plane = noise_plane() + gaussian((40, 60), 50.0)
    xds = image(plane)
    make_auto_mask(xds, {})
    transposed = image(plane)
    transposed["BEAM"] = transposed.BEAM.transpose("beam_param", "polarization", ...)
    make_auto_mask(transposed, {})
    np.testing.assert_array_equal(transposed.MASK.values, xds.MASK.values)
