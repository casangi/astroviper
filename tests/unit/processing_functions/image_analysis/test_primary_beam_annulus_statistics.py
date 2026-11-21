"""Unit tests for astroviper.processing_functions.image_analysis.primary_beam_annulus_statistics."""

import math

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.primary_beam_annulus_statistics import (
    ANNULUS_STATISTIC_NAMES,
    MIN_MEDIAN_N_PIXELS,
    calculate_primary_beam_annulus_statistics,
    primary_beam_annulus_limits,
    summarize_primary_beam_annulus_rms,
)
from astroviper.processing_functions.image_analysis.statistics.chauvenet import (
    chauvenet_sample,
)

DIMS = ("time", "frequency", "polarization", "l", "m")
LIMITS = (0.2, 0.3)


def make_image(sky, primary_beam, mask=None):
    """Image with a ``restored`` data group, every array broadcast to ``sky``."""
    sky = np.asarray(sky)
    n_time, n_chan, n_pol, n_l, n_m = sky.shape
    img_xds = xr.Dataset(
        {
            "SKY_RESTORED": (DIMS, sky),
            "PRIMARY_BEAM": (DIMS, np.broadcast_to(primary_beam, sky.shape).copy()),
        },
        coords={
            "time": np.arange(n_time, dtype=float),
            "frequency": 1e11 + 1e6 * np.arange(n_chan),
            "polarization": ["I", "Q", "U", "V"][:n_pol],
            "l": 1e-5 * (n_l // 2 - np.arange(n_l)),
            "m": 1e-5 * (np.arange(n_m) - n_m // 2),
        },
    )
    group = {"sky": "SKY_RESTORED", "primary_beam": "PRIMARY_BEAM"}
    if mask is not None:
        img_xds["MASK"] = (DIMS, np.broadcast_to(mask, sky.shape).copy())
        group["mask"] = "MASK"
    img_xds.attrs["data_groups"] = {"restored": group}
    return img_xds


def gaussian_beam(n_pixels, width):
    i, j = np.mgrid[:n_pixels, :n_pixels] - n_pixels // 2
    return np.exp(-(i**2 + j**2) / (2.0 * width**2))


def plane(statistics, name):
    return statistics[name].isel(time=0, frequency=0, polarization=0).item()


class TestLimits:
    def test_nominal_levels_when_the_edge_is_below_the_lower_one(self):
        img_xds = make_image(np.zeros((1, 3, 1, 48, 48)), gaussian_beam(48, 10.0))
        limits = primary_beam_annulus_limits(img_xds, LIMITS)
        assert limits == LIMITS
        assert all(type(limit) is float for limit in limits)

    def test_levels_of_the_primary_beam_when_the_edge_is_above_the_lower_one(self):
        beam = gaussian_beam(40, 30.0)
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40)), beam)
        limits = primary_beam_annulus_limits(img_xds, LIMITS)
        assert limits == (beam[20, 2], beam[20, 4])
        assert all(type(limit) is float for limit in limits)

    def test_reads_along_m_the_central_column_of_the_middle_channel(self):
        beam = np.full((2, 4, 2, 30, 50), 0.1)
        beam[0, 2, 0, 15] = np.linspace(0.5, 0.9, 50)
        img_xds = make_image(np.zeros(beam.shape), beam)
        limits = primary_beam_annulus_limits(img_xds, LIMITS)
        assert limits == (beam[0, 2, 0, 15, 2], beam[0, 2, 0, 15, 4])

    @pytest.mark.parametrize(
        ("nominal", "from_beam"), [((0.5, 0.6), True), ((0.6, 0.7), False)]
    )
    def test_the_edge_is_compared_with_the_nominal_lower_level(
        self, nominal, from_beam
    ):
        beam = np.full((40, 40), 0.1)
        beam[20] = np.linspace(0.55, 0.95, 40)
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40)), beam)
        expected = (beam[20, 2], beam[20, 4]) if from_beam else nominal
        assert primary_beam_annulus_limits(img_xds, nominal) == expected

    @pytest.mark.parametrize("edge", [0.0, np.nan, 0.2])
    def test_a_zero_nan_or_lower_level_edge_keeps_the_nominal_levels(self, edge):
        beam = gaussian_beam(40, 30.0)
        beam[20, 0] = edge
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40)), beam)
        assert primary_beam_annulus_limits(img_xds, LIMITS) == LIMITS

    def test_a_single_precision_edge_is_compared_in_double_precision(self):
        beam = gaussian_beam(40, 30.0).astype(np.float32)
        beam[20, 0] = np.float32(0.2)
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40), dtype=np.float32), beam)
        limits = primary_beam_annulus_limits(img_xds, LIMITS)
        assert limits == (float(beam[20, 2]), float(beam[20, 4]))

    def test_fewer_than_twenty_rows_give_equal_levels_from_the_primary_beam(self):
        beam = gaussian_beam(16, 30.0)
        img_xds = make_image(np.zeros((1, 1, 1, 16, 16)), beam)
        assert primary_beam_annulus_limits(img_xds, LIMITS) == (beam[8, 0], beam[8, 0])
        img_xds = make_image(np.zeros((1, 1, 1, 16, 16)), gaussian_beam(16, 4.0))
        assert primary_beam_annulus_limits(img_xds, LIMITS) == LIMITS

    @pytest.mark.parametrize(
        "limits", [(0.3, 0.2), (-0.1, 0.3), (0.2, 1.5), (0.2,), (0.2, np.nan)]
    )
    def test_malformed_nominal_levels_are_refused(self, limits):
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40)), gaussian_beam(40, 10.0))
        with pytest.raises(ValueError, match="nominal_annulus_limits"):
            primary_beam_annulus_limits(img_xds, limits)

    def test_non_numeric_nominal_levels_are_refused(self):
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40)), gaussian_beam(40, 10.0))
        with pytest.raises(TypeError, match="nominal_annulus_limits"):
            primary_beam_annulus_limits(img_xds, ("0.2", "0.3"))

    def test_input_checked_skips_the_configuration_checks(self):
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40)), gaussian_beam(40, 10.0))
        limits = primary_beam_annulus_limits(img_xds, (0.3, 0.2), input_checked=True)
        assert limits == (0.3, 0.2)

    def test_the_data_group_must_hold_the_primary_beam(self):
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40)), gaussian_beam(40, 10.0))
        with pytest.raises(ValueError, match="no data group 'residual'"):
            primary_beam_annulus_limits(
                img_xds, LIMITS, image_data_group_in_name="residual", input_checked=True
            )
        img_xds.attrs["data_groups"] = {"restored": {"sky": "SKY_RESTORED"}}
        with pytest.raises(ValueError, match=r"lacks the roles \['primary_beam'\]"):
            primary_beam_annulus_limits(img_xds, LIMITS)

    def test_the_primary_beam_must_be_on_the_image_dimensions(self):
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40)), gaussian_beam(40, 10.0))
        img_xds = img_xds.transpose("time", "frequency", "polarization", "m", "l")
        with pytest.raises(ValueError, match="PRIMARY_BEAM has the dimensions"):
            primary_beam_annulus_limits(img_xds, LIMITS)

    def test_a_dask_backed_primary_beam_is_refused(self):
        img_xds = make_image(np.zeros((1, 1, 1, 40, 40)), gaussian_beam(40, 10.0))
        with pytest.raises(TypeError, match="PRIMARY_BEAM is dask-backed"):
            primary_beam_annulus_limits(img_xds.chunk(), LIMITS)


class TestStatistics:
    def test_statistics_are_those_of_casa(self):
        # Five values the clipping keeps: the RMS includes the mean, the
        # standard deviation has n - 1 degrees of freedom, and the median
        # absolute deviation is not scaled.
        beam = np.zeros((5, 5))
        beam[0] = 0.25
        sky = np.zeros((1, 1, 1, 5, 5))
        sky[0, 0, 0, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]
        statistics = calculate_primary_beam_annulus_statistics(
            make_image(sky, beam), LIMITS
        )
        assert plane(statistics, "n_pixels") == 5
        assert plane(statistics, "rms") == pytest.approx(math.sqrt(11.0), rel=1e-15)
        assert plane(statistics, "standard_deviation") == pytest.approx(
            math.sqrt(2.5), rel=1e-15
        )
        assert plane(statistics, "median") == 3.0
        assert plane(statistics, "median_absolute_deviation") == 1.0

    def test_outliers_are_clipped(self):
        beam = gaussian_beam(48, 10.0)
        sky = np.random.default_rng(4).normal(0.0, 1.0, (48, 48))
        annulus = (beam > 0.2) & (beam < 0.3)
        sky[tuple(np.argwhere(annulus)[0])] = 1000.0
        statistics = calculate_primary_beam_annulus_statistics(
            make_image(sky[None, None, None], beam), LIMITS
        )
        kept = chauvenet_sample(sky[annulus])
        assert plane(statistics, "n_pixels") == kept.size < annulus.sum()
        assert plane(statistics, "rms") == pytest.approx(
            np.sqrt(np.mean(kept**2)), rel=1e-12
        )

    def test_levels_are_strict(self):
        levels = [0.2, 0.3, 0.25, 0.1, np.nextafter(0.2, 1.0), np.nextafter(0.3, 0.0)]
        statistics = calculate_primary_beam_annulus_statistics(
            make_image(np.ones((1, 1, 1, 1, 6)), np.array(levels)), LIMITS
        )
        assert plane(statistics, "n_pixels") == 3

    def test_a_single_precision_primary_beam_is_compared_in_double_precision(self):
        # float32(0.2) is above 0.2 and float32(0.3) above 0.3.
        beam = np.array([0.2, 0.3, 0.25], dtype=np.float32)
        statistics = calculate_primary_beam_annulus_statistics(
            make_image(np.ones((1, 1, 1, 1, 3), dtype=np.float32), beam), LIMITS
        )
        assert plane(statistics, "n_pixels") == 2

    def test_non_finite_pixels_are_left_out(self):
        sky = np.ones((1, 1, 1, 1, 4))
        sky[..., :2] = [np.nan, np.inf]
        statistics = calculate_primary_beam_annulus_statistics(
            make_image(sky, np.full(4, 0.25)), LIMITS
        )
        assert plane(statistics, "n_pixels") == 2

    def test_an_empty_annulus_has_no_pixels(self):
        statistics = calculate_primary_beam_annulus_statistics(
            make_image(np.ones((1, 1, 1, 8, 8)), np.zeros((8, 8))), LIMITS
        )
        assert plane(statistics, "n_pixels") == 0
        for name in ANNULUS_STATISTIC_NAMES[1:]:
            assert np.isnan(plane(statistics, name))
            assert np.isnan(plane(statistics, name + "_outside_mask"))

    def test_one_pixel_has_no_spread(self):
        beam = np.zeros((4, 4))
        beam[1, 2] = 0.25
        statistics = calculate_primary_beam_annulus_statistics(
            make_image(np.full((1, 1, 1, 4, 4), -2.0), beam), LIMITS
        )
        assert plane(statistics, "n_pixels") == 1
        assert plane(statistics, "rms") == 2.0
        assert plane(statistics, "standard_deviation") == 0.0
        assert plane(statistics, "median_absolute_deviation") == 0.0

    def test_two_pixels_have_one_degree_of_freedom(self):
        sky = np.zeros((1, 1, 1, 1, 4))
        sky[..., :2] = [1.0, 3.0]
        beam = np.array([0.25, 0.25, 0.0, 0.0])
        statistics = calculate_primary_beam_annulus_statistics(
            make_image(sky, beam), LIMITS
        )
        assert plane(statistics, "n_pixels") == 2
        assert plane(statistics, "standard_deviation") == math.sqrt(2.0)
        assert plane(statistics, "median_absolute_deviation") == 1.0

    def test_own_masks_and_their_union(self):
        beam = np.full((1, 6), 0.25)
        mask = np.zeros((2, 2, 1, 1, 6), dtype=bool)
        mask[0, 0, 0, 0, :2] = True
        mask[0, 1, 0, 0, 1:3] = True
        mask[1, 0, 0, 0, 5] = True
        img_xds = make_image(np.ones((2, 2, 1, 1, 6)), beam, mask)
        own = calculate_primary_beam_annulus_statistics(img_xds, LIMITS)
        union = calculate_primary_beam_annulus_statistics(
            img_xds, LIMITS, mask_union=True
        )
        assert own.n_pixels.values.ravel().tolist() == [6, 6, 6, 6]
        assert own.n_pixels_outside_mask.values.ravel().tolist() == [4, 4, 5, 6]
        assert union.n_pixels_outside_mask.values.ravel().tolist() == [2, 2, 2, 2]
        assert own.attrs["mask_source"] == union.attrs["mask_source"] == "MASK"
        assert (own.attrs["mask_union"], union.attrs["mask_union"]) == (False, True)

    def test_without_a_mask_outside_the_mask_is_the_whole_annulus(self):
        sky = np.random.default_rng(5).normal(0.0, 1.0, (1, 2, 1, 32, 32))
        img_xds = make_image(sky, gaussian_beam(32, 7.0))
        statistics = calculate_primary_beam_annulus_statistics(img_xds, LIMITS)
        for name in ANNULUS_STATISTIC_NAMES:
            np.testing.assert_array_equal(
                statistics[name], statistics[name + "_outside_mask"]
            )
        assert statistics.attrs["mask_source"] is None
        img_xds.attrs["data_groups"]["restored"]["mask"] = "MASK"
        with_absent_mask = calculate_primary_beam_annulus_statistics(img_xds, LIMITS)
        xr.testing.assert_identical(with_absent_mask, statistics)

    def test_every_plane_is_measured_on_its_own(self):
        scale = np.arange(1.0, 13.0).reshape(2, 3, 2, 1, 1)
        noise = np.random.default_rng(6).normal(0.0, 1.0, (32, 32))
        img_xds = make_image(scale * noise, gaussian_beam(32, 7.0))
        statistics = calculate_primary_beam_annulus_statistics(img_xds, LIMITS)
        rms = statistics.rms.values
        np.testing.assert_allclose(rms / rms[0, 0, 0], scale[..., 0, 0], rtol=1e-12)
        assert (statistics.n_pixels.values == statistics.n_pixels.values[0, 0, 0]).all()

    def test_layout_of_the_result(self):
        mask = np.zeros((32, 32), dtype=bool)
        img_xds = make_image(np.ones((1, 2, 2, 32, 32)), gaussian_beam(32, 7.0), mask)
        statistics = calculate_primary_beam_annulus_statistics(img_xds, (0.2, 0.3))
        names = [*ANNULUS_STATISTIC_NAMES]
        names += [name + "_outside_mask" for name in ANNULUS_STATISTIC_NAMES]
        assert list(statistics.data_vars) == names
        for name in names:
            assert statistics[name].dims == DIMS[:3]
            expected = np.int64 if name.startswith("n_pixels") else np.float64
            assert statistics[name].dtype == expected
        for dim in DIMS[:3]:
            xr.testing.assert_identical(statistics[dim], img_xds[dim])
        assert statistics.attrs == {
            "data_variable": "SKY_RESTORED",
            "mask_source": "MASK",
            "mask_union": False,
            "annulus_limits": [0.2, 0.3],
        }

    def test_a_single_precision_image_is_measured_in_double_precision(self):
        sky = np.random.default_rng(7).normal(0.0, 1.0, (1, 1, 1, 32, 32))
        sky = sky.astype(np.float32)
        beam = gaussian_beam(32, 7.0)
        single = calculate_primary_beam_annulus_statistics(
            make_image(sky, beam), LIMITS
        )
        double = calculate_primary_beam_annulus_statistics(
            make_image(sky.astype(np.float64), beam), LIMITS
        )
        xr.testing.assert_identical(single, double)

    def test_the_mask_must_be_bool(self):
        img_xds = make_image(
            np.ones((1, 1, 1, 8, 8)), np.zeros((8, 8)), np.zeros((8, 8))
        )
        with pytest.raises(TypeError, match="MASK must be bool"):
            calculate_primary_beam_annulus_statistics(img_xds, LIMITS)

    def test_dask_backed_images_are_refused(self):
        img_xds = make_image(np.ones((1, 1, 1, 8, 8)), np.zeros((8, 8))).chunk()
        with pytest.raises(TypeError, match="dask-backed"):
            calculate_primary_beam_annulus_statistics(img_xds, LIMITS)

    def test_the_data_group_must_hold_the_image_and_the_primary_beam(self):
        img_xds = make_image(np.ones((1, 1, 1, 8, 8)), np.zeros((8, 8)))
        img_xds.attrs["data_groups"] = {"restored": {"primary_beam": "PRIMARY_BEAM"}}
        with pytest.raises(ValueError, match=r"lacks the roles \['sky'\]"):
            calculate_primary_beam_annulus_statistics(img_xds, LIMITS)
        img_xds.attrs["data_groups"]["restored"]["sky"] = "SKY_MODEL"
        with pytest.raises(ValueError, match="no data variable 'SKY_MODEL'"):
            calculate_primary_beam_annulus_statistics(img_xds, LIMITS)

    @pytest.mark.parametrize(
        ("limits", "mask_union", "error"),
        [
            ((0.3, 0.2), False, ValueError),
            ((0.2, 0.3, 0.4), False, ValueError),
            (0.2, False, ValueError),
            (("a", "b"), False, TypeError),
            (LIMITS, 1, TypeError),
        ],
    )
    def test_malformed_configuration_is_refused(self, limits, mask_union, error):
        img_xds = make_image(np.ones((1, 1, 1, 8, 8)), np.zeros((8, 8)))
        with pytest.raises(error):
            calculate_primary_beam_annulus_statistics(img_xds, limits, mask_union)

    def test_input_checked_skips_only_the_configuration_checks(self):
        img_xds = make_image(np.ones((1, 1, 1, 8, 8)), np.full((8, 8), 0.25))
        statistics = calculate_primary_beam_annulus_statistics(
            img_xds, (0.3, 0.2), input_checked=True
        )
        assert plane(statistics, "n_pixels") == 0
        with pytest.raises(ValueError, match="no data group"):
            calculate_primary_beam_annulus_statistics(
                img_xds, LIMITS, image_data_group_in_name="model", input_checked=True
            )


def make_statistics(n_pixels, rms, n_pixels_outside_mask, rms_outside_mask):
    """Annulus statistics on (time, frequency, polarization) from per-channel lists."""

    def cube(values, dtype):
        values = np.asarray(values, dtype=dtype)
        return values.reshape(1, values.shape[0], -1)

    arrays = {
        "n_pixels": cube(n_pixels, np.int64),
        "rms": cube(rms, np.float64),
        "n_pixels_outside_mask": cube(n_pixels_outside_mask, np.int64),
        "rms_outside_mask": cube(rms_outside_mask, np.float64),
    }
    n_chan, n_pol = arrays["rms"].shape[1:]
    return xr.Dataset(
        {name: (DIMS[:3], values) for name, values in arrays.items()},
        coords={
            "time": [0.0],
            "frequency": 1e11 + 1e6 * np.arange(n_chan),
            "polarization": ["I", "Q", "U", "V"][:n_pol],
        },
    )


def summary_values(summary):
    return [
        summary[name].isel(time=0, polarization=0).item()
        for name in ("rms_median", "rms_min", "rms_max", "full_annulus")
    ]


class TestSummary:
    def test_the_rms_outside_the_mask_from_ten_pixels(self):
        statistics = make_statistics(
            [50, 50, 50], [9.0, 9.0, 9.0], [10, 10, 12], [1.0, 3.0, 2.0]
        )
        summary = summarize_primary_beam_annulus_rms(statistics)
        assert summary_values(summary) == [2.0, 1.0, 3.0, False]

    def test_the_whole_annulus_below_ten_pixels(self):
        statistics = make_statistics(
            [50, 50, 50], [4.0, 6.0, 5.0], [9, 30, 5], [1.0, 3.0, 2.0]
        )
        summary = summarize_primary_beam_annulus_rms(statistics)
        assert summary_values(summary) == [5.0, 4.0, 6.0, True]
        assert MIN_MEDIAN_N_PIXELS == 10

    def test_the_median_of_an_even_number_of_channels(self):
        rms = [1.0, 2.0, 3.0, 4.0]
        above = make_statistics([50] * 4, [9.0] * 4, [5, 9, 11, 40], rms)
        below = make_statistics([50] * 4, [9.0] * 4, [5, 8, 11, 40], rms)
        above_summary = summarize_primary_beam_annulus_rms(above)
        below_summary = summarize_primary_beam_annulus_rms(below)
        assert summary_values(above_summary) == [2.5, 1.0, 4.0, False]
        assert summary_values(below_summary) == [9.0, 9.0, 9.0, True]

    def test_the_whole_annulus_when_no_channel_has_pixels_outside_the_mask(self):
        statistics = make_statistics(
            [5, 6, 7], [4.0, 6.0, 5.0], [0, 0, 0], [np.nan, np.nan, np.nan]
        )
        summary = summarize_primary_beam_annulus_rms(statistics)
        assert summary_values(summary) == [5.0, 4.0, 6.0, True]

    def test_channels_without_pixels_count_as_zero(self):
        statistics = make_statistics(
            [50, 0, 50], [4.0, np.nan, 5.0], [20, 0, 20], [1.0, np.nan, 2.0]
        )
        summary = summarize_primary_beam_annulus_rms(statistics)
        assert summary_values(summary) == [1.0, 0.0, 2.0, False]

    def test_selected_channels_without_pixels_count_as_zero(self):
        statistics = make_statistics([50, 0], [4.0, np.nan], [20, 0], [1.0, np.nan])
        summary = summarize_primary_beam_annulus_rms(statistics, [1])
        assert summary_values(summary) == [0.0, 0.0, 0.0, False]

    def test_without_a_mask_nothing_is_replaced(self):
        rms = [4.0, 6.0, 5.0]
        statistics = make_statistics([5, 6, 7], rms, [5, 6, 7], rms)
        statistics.attrs["mask_source"] = None
        summary = summarize_primary_beam_annulus_rms(statistics)
        assert summary_values(summary) == [5.0, 4.0, 6.0, False]

    def test_nan_without_any_pixel(self):
        statistics = make_statistics([0, 0], [np.nan] * 2, [0, 0], [np.nan] * 2)
        median, minimum, maximum, full_annulus = summary_values(
            summarize_primary_beam_annulus_rms(statistics)
        )
        assert np.isnan([median, minimum, maximum]).all()
        assert full_annulus is True

    def test_selected_channels_count_as_often_as_they_are_given(self):
        statistics = make_statistics(
            [50, 50, 50], [9.0] * 3, [20, 20, 20], [1.0, 2.0, 3.0]
        )
        once = summarize_primary_beam_annulus_rms(statistics, [0, 2])
        twice = summarize_primary_beam_annulus_rms(statistics, np.array([0, 0, 2]))
        assert summary_values(once)[:3] == [2.0, 1.0, 3.0]
        assert summary_values(twice)[:3] == [1.0, 1.0, 3.0]

    def test_all_channels_decide_between_the_annuli(self):
        statistics = make_statistics(
            [50, 50, 50], [4.0, 6.0, 5.0], [0, 0, 50], [np.nan, np.nan, 2.0]
        )
        summary = summarize_primary_beam_annulus_rms(statistics, [2])
        assert summary_values(summary) == [5.0, 5.0, 5.0, True]

    def test_each_polarization_decides_on_its_own(self):
        statistics = make_statistics(
            [[50, 50]] * 3,
            [[4.0, 4.0]] * 3,
            [[20, 5]] * 3,
            [[1.0, 1.0]] * 3,
        )
        summary = summarize_primary_beam_annulus_rms(statistics)
        assert summary.full_annulus.dims == ("time", "polarization")
        assert summary.full_annulus.values.tolist() == [[False, True]]
        assert summary.rms_median.values.tolist() == [[1.0, 4.0]]

    def test_each_time_decides_on_its_own(self):
        rms = [1.0, 2.0, 3.0]
        statistics = xr.concat(
            [
                make_statistics([50] * 3, [9.0] * 3, [20] * 3, rms),
                make_statistics([50] * 3, [9.0] * 3, [5] * 3, rms).assign_coords(
                    time=[1.0]
                ),
            ],
            dim="time",
        )
        summary = summarize_primary_beam_annulus_rms(statistics)
        assert summary.full_annulus.values.tolist() == [[False], [True]]
        assert summary.rms_median.values.tolist() == [[2.0], [9.0]]

    def test_the_order_of_the_dimensions_does_not_matter(self):
        statistics = make_statistics(
            [[50, 50]] * 3,
            [[9.0, 9.0]] * 3,
            [[20, 5]] * 3,
            [[1.0, 2.0], [3.0, 2.0], [2.0, 2.0]],
        )
        summary = summarize_primary_beam_annulus_rms(statistics)
        for dims in (
            ("frequency", "time", "polarization"),
            ("polarization", "frequency", "time"),
        ):
            xr.testing.assert_identical(
                summarize_primary_beam_annulus_rms(statistics.transpose(*dims)),
                summary,
            )

    @pytest.mark.parametrize(
        "channel_indices",
        [[], np.array([], dtype=int), [3], [-1], [0.5], [[0, 1]]],
    )
    def test_channel_indices_must_index_channels(self, channel_indices):
        statistics = make_statistics([50] * 3, [9.0] * 3, [20] * 3, [1.0] * 3)
        with pytest.raises(ValueError, match="channel_indices"):
            summarize_primary_beam_annulus_rms(statistics, channel_indices)
