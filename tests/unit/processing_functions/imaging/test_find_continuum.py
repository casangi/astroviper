"""Tests of the cube reductions of find_continuum (processing functions)."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum import (
    CHANNEL_STATISTICS,
    MAD_TO_SIGMA,
    SPEED_OF_LIGHT_KM_S,
    _profile_value,
    channel_statistics,
    evaluate_lattice_expression,
    finalize_find_continuum_reductions,
    find_continuum_cube_reductions,
    image_statistics,
    lattice_expression_names,
    merge_find_continuum_reductions,
    radio_velocity_km_s,
)

N_CHANNELS = 12
SHAPE = (6, 5)
FREQUENCY = 230.0e9 + 1.0e6 * np.arange(N_CHANNELS)
REST_FREQUENCY = 230.5e9
VARIABLES = {"cube": "SKY", "pb": "PB"}


def _image_xds(n_time=1, n_polarization=1):
    rng = np.random.default_rng(3)
    shape = (n_time, N_CHANNELS, n_polarization, *SHAPE)
    sky = rng.normal(size=shape)
    sky[0, 2, 0, 1, 1] = np.nan
    pb = np.broadcast_to(
        np.linspace(0.1, 1.0, SHAPE[0] * SHAPE[1]).reshape(SHAPE), shape
    )
    dims = ("time", "frequency", "polarization", "l", "m")
    return xr.Dataset({"SKY": (dims, sky), "PB": (dims, pb.copy())})


def _casa_order(img_xds, variable):
    return np.transpose(img_xds[variable].values[0], (2, 3, 1, 0))


@pytest.fixture
def small_image():
    return np.linspace(0.0, 1.0, SHAPE[0] * SHAPE[1]).reshape(*SHAPE, 1, 1)


@pytest.fixture
def requests_all_kinds():
    return [
        {
            "kind": "moments",
            "image": "cube",
            "mask": '"pb">0.3',
            "channels": [1, 2, 3, 7],
            "moments": [-1, 0, 8, 10],
            "frequency": FREQUENCY.tolist(),
            "rest_frequency": REST_FREQUENCY,
        },
        {
            "kind": "profile",
            "image": "cube",
            "mask": '"cube">-0.5 && "small"<0.8',
            "channels": None,
            "function": "mean",
        },
        {
            "kind": "statistics",
            "image": "cube",
            "mask": "",
            "channels": None,
            "per_channel": True,
        },
        {
            "kind": "statistics",
            "image": "cube",
            "mask": '"pb">0.5',
            "channels": [0, 5, 6],
            "per_channel": False,
        },
    ]


class TestLatticeExpressions:
    arrays = {
        "a": np.array([1.0, 2.0, np.nan, 4.0]),
        "b": np.array([0.5, 3.0, 1.0, 4.0]),
    }

    def test_arithmetic_and_precedence(self):
        result = evaluate_lattice_expression('"a" + 2 * "b" - 1', self.arrays)
        np.testing.assert_array_equal(
            result, self.arrays["a"] + 2 * self.arrays["b"] - 1
        )
        assert evaluate_lattice_expression("-(1 + 2) * 3 / 9", {}) == -1.0
        assert evaluate_lattice_expression(".5 + 0.5e0", {}) == 1.0

    def test_masked_pixel_fails_every_comparison(self):
        np.testing.assert_array_equal(
            evaluate_lattice_expression('"a/" > 1.5', self.arrays),
            [False, True, False, True],
        )
        np.testing.assert_array_equal(
            evaluate_lattice_expression('"a" <= 1.5', self.arrays),
            [True, False, False, False],
        )

    def test_logical_operators(self):
        np.testing.assert_array_equal(
            evaluate_lattice_expression('!("a" > 1.5) && "b" < 3.5', self.arrays),
            [True, False, True, False],
        )
        np.testing.assert_array_equal(
            evaluate_lattice_expression(
                '"a" == 4 || "b" != 3 && "b" >= 1', self.arrays
            ),
            [False, False, True, True],
        )
        assert evaluate_lattice_expression("T", {}) is True
        assert evaluate_lattice_expression("False || True", {})
        assert not evaluate_lattice_expression("True && F", {})

    def test_iif_and_identifiers(self):
        np.testing.assert_array_equal(
            evaluate_lattice_expression('iif("a" >= 2, "a", -1)', self.arrays),
            [-1.0, 2.0, -1.0, 4.0],
        )
        np.testing.assert_array_equal(
            evaluate_lattice_expression(
                "IM0 / IM1", {"IM0": np.array([2.0]), "IM1": np.array([4.0])}
            ),
            [0.5],
        )

    @pytest.mark.parametrize(
        "expression, error",
        [
            ('"a" > ', ValueError),
            ("1 $ 2", ValueError),
            ('max("a", 1)', ValueError),
            ("(1 + 2", ValueError),
            ("1 2", ValueError),
            ('"c" > 1', KeyError),
        ],
    )
    def test_errors(self, expression, error):
        with pytest.raises(error):
            evaluate_lattice_expression(expression, self.arrays)

    def test_names(self):
        assert lattice_expression_names('"x/" > 1 && "y" < "x" && IM0 > 0') == [
            "x",
            "y",
        ]


class TestStatistics:
    def test_channel_statistics_match_numpy(self):
        values = np.random.default_rng(1).normal(size=101)
        stats = channel_statistics(values)
        assert set(stats) == set(CHANNEL_STATISTICS)
        assert stats["npts"] == 101
        np.testing.assert_allclose(stats["sigma"], values.std(ddof=1))
        np.testing.assert_allclose(stats["rms"], np.sqrt(np.mean(values**2)))
        np.testing.assert_allclose(
            stats["medabsdevmed"], np.median(np.abs(values - np.median(values)))
        )
        np.testing.assert_allclose(
            stats["quartile"], np.subtract(*np.percentile(values, [75, 25]))
        )

    def test_empty_and_single_value(self):
        empty = channel_statistics(np.array([]))
        assert empty["npts"] == 0 and np.isnan(empty["mean"]) and np.isnan(empty["max"])
        assert channel_statistics(np.array([2.0]))["sigma"] == 0.0

    def test_chauvenet_removes_outliers(self):
        values = np.append(np.random.default_rng(2).normal(size=1000), 1.0e3)
        assert image_statistics(values)["max"] == 1.0e3
        stats = image_statistics(values, algorithm="chauvenet", maxiter=5)
        assert stats["max"] < 10.0 and stats["npts"] <= 1000

    @pytest.mark.parametrize("algorithm, maxiter", [("chauvenet", 3), ("fit-half", -1)])
    def test_unsupported_statistics(self, algorithm, maxiter):
        with pytest.raises(ValueError):
            image_statistics(np.ones(3), algorithm=algorithm, maxiter=maxiter)

    @pytest.mark.parametrize(
        "function, expected",
        [
            ("mean", 4.0),
            ("median", 3.0),
            ("max", 10.0),
            ("min", 1.0),
            ("npts", 5.0),
            ("xmadm", MAD_TO_SIGMA),
            ("max/xmadm", 10.0 / MAD_TO_SIGMA),
        ],
    )
    def test_profile_values(self, function, expected):
        values = np.array([4.0, 1.0, 10.0, 3.0, 2.0])
        assert _profile_value(values, function) == pytest.approx(expected)

    def test_profile_value_of_no_pixels_and_unknown_function(self):
        assert _profile_value(np.array([]), "npts") == 0.0
        assert np.isnan(_profile_value(np.array([]), "mean"))
        with pytest.raises(ValueError):
            _profile_value(np.ones(2), "sum")

    def test_radio_velocity(self):
        np.testing.assert_allclose(
            radio_velocity_km_s([230.0e9, 229.0e9], 230.0e9),
            [0.0, SPEED_OF_LIGHT_KM_S / 230.0],
        )


class TestCubeReductions:
    def _reduce(self, img_xds, requests, images, chunks):
        partials = None
        for chunk in chunks:
            part = find_continuum_cube_reductions(
                img_xds.isel(frequency=chunk),
                requests,
                VARIABLES,
                images,
                np.arange(N_CHANNELS)[chunk],
            )
            partials = (
                part
                if partials is None
                else merge_find_continuum_reductions(partials, part)
            )
        return finalize_find_continuum_reductions(partials, requests)

    def test_chunks_merge_to_the_whole_cube(self, requests_all_kinds, small_image):
        img_xds = _image_xds()
        images = {"small": small_image}
        whole = self._reduce(img_xds, requests_all_kinds, images, [slice(None)])
        chunked = self._reduce(
            img_xds,
            requests_all_kinds,
            images,
            [slice(7, 12), slice(0, 4), slice(4, 7)],
        )
        for moment in (-1, 0, 8, 10):
            np.testing.assert_allclose(
                chunked[0][moment], whole[0][moment], equal_nan=True
            )
        np.testing.assert_allclose(chunked[1], whole[1], equal_nan=True)
        for key in CHANNEL_STATISTICS:
            np.testing.assert_allclose(chunked[2][key], whole[2][key], equal_nan=True)
        for key in ("npts", "sum", "sumsq", "min", "max", "mean", "sigma", "rms"):
            assert chunked[3][key] == pytest.approx(whole[3][key])

    @pytest.mark.filterwarnings(
        "ignore:All-NaN slice:RuntimeWarning",
        "ignore:Mean of empty slice:RuntimeWarning",
    )
    def test_results_match_numpy(self, requests_all_kinds, small_image):
        img_xds = _image_xds()
        cube, pb = _casa_order(img_xds, "SKY"), _casa_order(img_xds, "PB")
        moments, profile, per_channel, whole_cube = self._reduce(
            img_xds,
            requests_all_kinds,
            {"small": small_image},
            [slice(0, 5), slice(5, 12)],
        )

        selected = np.where(np.isfinite(cube) & (pb > 0.3), cube, np.nan)[
            :, :, 0, [1, 2, 3, 7]
        ]
        channel_width = SPEED_OF_LIGHT_KM_S * 1.0e6 / REST_FREQUENCY
        np.testing.assert_allclose(
            moments[8], np.nanmax(selected, axis=-1), equal_nan=True
        )
        np.testing.assert_allclose(
            moments[10], np.nanmin(selected, axis=-1), equal_nan=True
        )
        np.testing.assert_allclose(
            moments[-1], np.nanmean(selected, axis=-1), equal_nan=True
        )
        integrated = np.where(
            np.isnan(selected).all(axis=-1), np.nan, np.nansum(selected, axis=-1)
        )
        np.testing.assert_allclose(
            moments[0], integrated * channel_width, equal_nan=True
        )
        assert np.isnan(moments[8][pb[:, :, 0, 0] <= 0.3]).all()

        masked = np.where((cube > -0.5) & (small_image < 0.8), cube, np.nan)
        np.testing.assert_allclose(profile, np.nanmean(masked, axis=(0, 1, 2)))
        np.testing.assert_allclose(
            per_channel["max"], [np.nanmax(cube[..., c]) for c in range(N_CHANNELS)]
        )
        assert per_channel["npts"][2] == SHAPE[0] * SHAPE[1] - 1

        values = cube[..., [0, 5, 6]][
            (pb[..., [0, 5, 6]] > 0.5) & np.isfinite(cube[..., [0, 5, 6]])
        ]
        assert whole_cube["npts"] == values.size
        assert whole_cube["mean"] == pytest.approx(values.mean())
        assert whole_cube["sigma"] == pytest.approx(values.std(ddof=1))
        assert whole_cube["rms"] == pytest.approx(np.sqrt(np.mean(values**2)))
        assert np.isnan(whole_cube["median"]) and np.isnan(whole_cube["quartile"])

    def test_empty_whole_cube_statistics(self):
        request = {
            "kind": "statistics",
            "image": "cube",
            "mask": '"pb">2',
            "channels": None,
            "per_channel": False,
        }
        (result,) = self._reduce(
            _image_xds(), [request], {}, [slice(0, 6), slice(6, 12)]
        )
        assert (
            result["npts"] == 0
            and np.isnan(result["min"])
            and np.isnan(result["sigma"])
        )

    @pytest.mark.parametrize("n_time, n_polarization", [(2, 1), (1, 2)])
    def test_refuses_more_than_one_plane(
        self, n_time, n_polarization, requests_all_kinds
    ):
        with pytest.raises(ValueError, match="one time and one polarization"):
            find_continuum_cube_reductions(
                _image_xds(n_time, n_polarization),
                requests_all_kinds,
                VARIABLES,
                {},
                np.arange(N_CHANNELS),
            )

    def test_unknown_kind(self):
        request = {"kind": "histogram", "image": "cube", "mask": "", "channels": None}
        with pytest.raises(ValueError, match="histogram"):
            find_continuum_cube_reductions(
                _image_xds(), [request], VARIABLES, {}, np.arange(N_CHANNELS)
            )
