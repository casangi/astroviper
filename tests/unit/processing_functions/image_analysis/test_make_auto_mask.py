"""Unit tests for astroviper.processing_functions.image_analysis.make_auto_mask."""

import math

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.make_auto_mask import (
    AUTO_MASK_DEFAULTS,
    _chauvenet_sample,
    _location_and_noise,
    _plane_thresholds,
    _prune,
    make_auto_mask,
    validate_auto_mask_params,
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


class TestValidate:
    def test_none_stays_none(self):
        assert validate_auto_mask_params(None) is None

    def test_empty_dict_gives_the_defaults(self):
        assert validate_auto_mask_params({}) == AUTO_MASK_DEFAULTS

    def test_given_values_are_kept(self):
        params = validate_auto_mask_params({"noise_threshold": 4.25})
        assert params["noise_threshold"] == 4.25
        assert params["sidelobe_threshold"] == 3.0

    def test_refuses_a_clean_mask(self):
        with pytest.raises(ValueError, match="exclude each other"):
            validate_auto_mask_params({}, clean_mask="mask.npy")

    def test_no_auto_mask_leaves_a_clean_mask_alone(self):
        assert validate_auto_mask_params(None, clean_mask="mask.npy") is None

    def test_refuses_an_unknown_key(self):
        with pytest.raises(ValueError, match="sidelobethreshold"):
            validate_auto_mask_params({"sidelobethreshold": 3.0})

    @pytest.mark.parametrize(
        "key, value, message",
        [
            ("sidelobe_threshold", -1.0, "must not be negative"),
            ("noise_threshold", "5", "finite number"),
            ("low_noise_threshold", math.nan, "finite number"),
            ("negative_threshold", True, "finite number"),
            ("min_beam_fraction", -0.1, "must not be negative"),
            ("smooth_factor", 0.0, "must be positive"),
            ("cut_threshold", 0.0, "must lie in"),
            ("cut_threshold", 1.0, "must lie in"),
            ("grow_iterations", -1, "non-negative integer"),
            ("grow_iterations", 1.5, "non-negative integer"),
            ("grow_iterations", True, "non-negative integer"),
            ("grow_prune", 1, "must be a bool"),
            ("fast_noise", "yes", "must be a bool"),
            ("min_percent_change", math.inf, "finite number"),
        ],
    )
    def test_refuses_a_bad_value(self, key, value, message):
        with pytest.raises(ValueError, match=message):
            validate_auto_mask_params({key: value})


class TestStatistics:
    def test_thresholds_set_by_the_sidelobes(self):
        params = validate_auto_mask_params({})
        mask, grow, negative, noise_sets = _plane_thresholds(
            10.0, 0.1, 1.0, 0.5, params
        )
        assert (mask, grow, negative) == pytest.approx((4.0, 4.0, -2.0))
        assert not noise_sets

    def test_thresholds_set_by_the_noise(self):
        params = validate_auto_mask_params({"negative_threshold": 4.0})
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

    def test_the_stopped_planes_given_are_not_modified(self):
        given = np.zeros((1, 1, 1), dtype=bool)
        xds = image(noise_plane())
        stopped = make_auto_mask(xds, {"noise_threshold": 50.0}, planes_stopped=given)
        assert stopped[0, 0, 0] and not given[0, 0, 0]


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
