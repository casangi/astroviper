"""Tests of the cube reductions of find_continuum (processing functions)."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.moments import MomentsAccumulator
from astroviper.processing_functions.imaging.find_continuum import (
    CASA_MOMENTS,
    CHANNEL_STATISTICS,
    MAD_TO_SIGMA,
    PROFILE_LEVEL,
    SPEED_OF_LIGHT_KM_S,
    FindContinuumMoments,
    _profile_value,
    channel_statistics,
    evaluate_lattice_expression,
    finalize_find_continuum_reductions,
    finalize_streamed_reductions,
    find_continuum_batch_reductions,
    find_continuum_cube_reductions,
    find_continuum_first_round_requests,
    first_round_statistics,
    frequency_from_axis,
    image_statistics,
    lattice_expression_names,
    linear_channel_ranges,
    linear_frequency_axis,
    merge_channel_rows,
    merge_find_continuum_reductions,
    merge_in_tree_order,
    moment_velocity_widths,
    new_streamed_partials,
    pack_plane_masks,
    plane_statistics,
    primary_beam_masks,
    radio_velocity_km_s,
    split_lattice_conjuncts,
    tile_occupancy,
    unpack_plane_masks,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    encode_streamed_requests,
    graphviper_tree,
    make_find_continuum_image_xds,
)

N_CHANNELS = 12
SHAPE = (6, 5)
FREQUENCY = 230.0e9 + 1.0e6 * np.arange(N_CHANNELS)
FREQUENCY_AXIS = (FREQUENCY[0], FREQUENCY[1] - FREQUENCY[0], 0.0, N_CHANNELS)
REST_FREQUENCY = 230.5e9
VARIABLES = {"cube": "SKY", "pb": "PB"}
WHOLE_CUBE_SUMS = ("sum", "sumsq", "mean", "sigma", "rms")


def _image_xds(n_time=1, n_polarization=1, dtype=np.float64):
    rng = np.random.default_rng(3)
    shape = (n_time, N_CHANNELS, n_polarization, *SHAPE)
    sky = rng.normal(size=shape).astype(dtype)
    sky[0, 2, 0, 1, 1] = np.nan
    pb = np.broadcast_to(
        np.linspace(0.1, 1.0, SHAPE[0] * SHAPE[1]).reshape(SHAPE), shape
    )
    dims = ("time", "frequency", "polarization", "l", "m")
    return xr.Dataset({"SKY": (dims, sky), "PB": (dims, pb.astype(dtype))})


def _casa_order(img_xds, variable):
    return np.transpose(img_xds[variable].values[0], (2, 3, 1, 0))


def _reference_channel_statistics(values):
    """channel_statistics of 424-find-continuum-report (the reference)."""
    values = np.asarray(values, dtype=np.float64)
    npts = values.size
    if npts == 0:
        return {key: (0.0 if key == "npts" else np.nan) for key in CHANNEL_STATISTICS}
    median = np.median(values)
    q1, q3 = np.percentile(values, [25.0, 75.0])
    sumsq = float(np.dot(values, values))
    return {
        "npts": float(npts),
        "sum": float(values.sum()),
        "sumsq": sumsq,
        "mean": float(values.mean()),
        "sigma": float(values.std(ddof=1)) if npts > 1 else 0.0,
        "rms": float(np.sqrt(sumsq / npts)),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(median),
        "medabsdevmed": float(np.median(np.abs(values - median))),
        "q1": float(q1),
        "q3": float(q3),
        "quartile": float(q3 - q1),
    }


def _reference_cube_reductions(
    img_xds,
    requests,
    variables,
    images,
    channel_index,
    primary_beam=None,
    primary_beam_limit=None,
):
    """find_continuum_cube_reductions of 424-find-continuum-report: the whole
    chunk in float64 and MomentsAccumulator."""
    cubes = {
        name: np.transpose(img_xds[variable].values[0], (2, 3, 1, 0)).astype(np.float64)
        for name, variable in variables.items()
    }
    if primary_beam_limit is not None:
        beam = np.transpose(img_xds[primary_beam].values[0], (2, 3, 1, 0))
        for cube in cubes.values():
            cube[beam < primary_beam_limit] = np.nan
    arrays = {**images, **cubes}
    channel_index = np.asarray(channel_index)
    partials = []
    for request in requests:
        cube = cubes[request["image"]]
        selected = (
            np.ones(channel_index.size, dtype=bool)
            if request["channels"] is None
            else np.isin(channel_index, request["channels"])
        )
        valid = np.isfinite(cube)
        if request["mask"]:
            valid &= np.broadcast_to(
                np.asarray(
                    evaluate_lattice_expression(request["mask"], arrays), dtype=bool
                ),
                cube.shape,
            )
        positions = np.flatnonzero(selected)
        if request["kind"] == "moments":
            accumulator = MomentsAccumulator(
                [CASA_MOMENTS[moment] for moment in request["moments"]],
                radio_velocity_km_s(
                    frequency_from_axis(request["frequency_axis"]),
                    request["rest_frequency"],
                ),
                cube.shape[:2],
            )
            for p in positions:
                accumulator.add_plane(
                    int(channel_index[p]), cube[:, :, 0, p], valid[:, :, 0, p]
                )
            partials.append(accumulator)
        elif request["kind"] == "profile":
            values = [
                _profile_value(cube[..., p][valid[..., p]], request["function"])
                for p in positions
            ]
            partials.append({"channels": channel_index[positions], "values": values})
        elif request["per_channel"]:
            rows = [
                _reference_channel_statistics(cube[..., p][valid[..., p]])
                for p in positions
            ]
            partials.append({"channels": channel_index[positions], "values": rows})
        else:
            values = cube[..., selected][valid[..., selected]]
            partials.append(
                {
                    "npts": float(values.size),
                    "sum": float(values.sum()),
                    "sumsq": float(np.dot(values, values)),
                    "min": float(values.min()) if values.size else np.inf,
                    "max": float(values.max()) if values.size else -np.inf,
                }
            )
    return partials


def _reference_merge(partials, other):
    for mine, theirs in zip(partials, other, strict=True):
        if isinstance(mine, MomentsAccumulator):
            mine.merge(theirs)
        elif "channels" in mine:
            mine["channels"] = np.concatenate([mine["channels"], theirs["channels"]])
            mine["values"] = list(mine["values"]) + list(theirs["values"])
        else:
            for key in ("npts", "sum", "sumsq"):
                mine[key] += theirs[key]
            mine["min"] = min(mine["min"], theirs["min"])
            mine["max"] = max(mine["max"], theirs["max"])
    return partials


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
            "frequency_axis": FREQUENCY_AXIS,
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


class TestSplitLatticeConjuncts:
    cubes = ("cube.residual", "cube.pb")
    # findContinuum's masks (findContinuum.py 8452-8469, 8509-8518, 3733-3745,
    # 7879-7895) and other shapes, with the split each must give.
    masks = [
        ('"x.joint.mask">0 && "cube.pb">0.21', '"x.joint.mask">0', ['"cube.pb">0.21']),
        ('"x.mask">0.5 && "cube.pb/">0.21', '"x.mask">0.5', ['"cube.pb/">0.21']),
        (
            '"cube.pb">0.21 && "cube.pb"<0.3 && "x.joint.mask"<0.5',
            '"x.joint.mask"<0.5',
            ['"cube.pb">0.21', '"cube.pb"<0.3'],
        ),
        ('"cube.pb">0.21 && "cube.pb"<0.3', None, ['"cube.pb">0.21', '"cube.pb"<0.3']),
        (
            '"x.pbmom">0.210000 && "x.pbmom"<0.300000 && "x.joint.mask"==0',
            '"x.pbmom">0.210000 && "x.pbmom"<0.300000 && "x.joint.mask"==0',
            [],
        ),
        ('"cube.residual"!=0', None, ['"cube.residual"!=0']),
        ("", None, []),
        (
            '("x.a" > 1 || "x.b" < 2) && "cube.pb" > 0.21',
            '("x.a" > 1 || "x.b" < 2)',
            ['"cube.pb" > 0.21'],
        ),
        (
            'iif("x.a" > 1 && "x.b" > 1, 1, 0) > 0 && !("cube.residual" < 0)',
            'iif("x.a" > 1 && "x.b" > 1, 1, 0) > 0',
            ['!("cube.residual" < 0)'],
        ),
    ]

    def _arrays(self, rng):
        arrays = {}
        for name in ("x.joint.mask", "x.mask", "x.pbmom", "x.a", "x.b"):
            arrays[name] = rng.uniform(0.0, 1.0, (7, 6, 1, 1)).round(1)
        for name in self.cubes:
            arrays[name] = rng.uniform(-0.5, 1.0, (7, 6, 1, 4)).round(2)
        for values in arrays.values():
            values[rng.random(values.shape) < 0.1] = np.nan
        return arrays

    def _assert_parts_give_the_whole(self, expression, parts, arrays):
        two_d, cube_terms = parts
        whole = np.broadcast_to(
            np.asarray(evaluate_lattice_expression(expression, arrays), dtype=bool),
            (7, 6, 1, 4),
        )
        joined = np.ones((7, 6, 1, 4), dtype=bool)
        for part in ([two_d] if two_d else []) + cube_terms:
            joined = joined & np.asarray(
                evaluate_lattice_expression(part, arrays), dtype=bool
            )
        np.testing.assert_array_equal(joined, whole)

    @pytest.mark.parametrize("expression, two_d, cube_terms", masks)
    def test_find_continuum_masks(self, expression, two_d, cube_terms):
        parts = split_lattice_conjuncts(expression, self.cubes)
        assert parts == (two_d, cube_terms)
        if expression:
            arrays = self._arrays(np.random.default_rng(5))
            self._assert_parts_give_the_whole(expression, parts, arrays)

    def test_random_chains(self):
        rng = np.random.default_rng(6)
        names = ["x.a", "x.b", *self.cubes]
        operators = ["<", "<=", ">", ">=", "==", "!="]
        arrays = self._arrays(rng)

        def comparison():
            name, operator = rng.choice(names), rng.choice(operators)
            text = f'"{name}" {operator} {rng.choice([0, 0.3, 0.5, 0.7])}'
            return f"!({text})" if rng.random() < 0.2 else text

        for _ in range(200):
            operands = [
                f"({comparison()} || {comparison()})"
                if rng.random() < 0.2
                else comparison()
                for _ in range(rng.integers(1, 6))
            ]
            expression = " && ".join(operands)
            parts = split_lattice_conjuncts(expression, self.cubes)
            two_d = [] if parts[0] is None else parts[0].split(" && ")
            assert sorted(two_d + parts[1]) == sorted(operands)
            self._assert_parts_give_the_whole(expression, parts, arrays)
            or_chain = f"{expression} || {comparison()}"
            assert split_lattice_conjuncts(or_chain, self.cubes) is None

    @pytest.mark.parametrize("expression", ['"x.a" > 1 &&', '&& "x.a" > 1', "1 $ 2"])
    def test_malformed(self, expression):
        with pytest.raises(ValueError):
            split_lattice_conjuncts(expression, self.cubes)


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

    def test_channel_statistics_equal_the_reference(self):
        rng = np.random.default_rng(11)
        for index in range(3000):
            size = int(rng.integers(0, 4) if index % 3 == 0 else rng.integers(4, 700))
            if index % 3 == 1:
                values = rng.integers(-3, 4, size) * 0.5
            else:
                scale = 10.0 ** rng.uniform(-6, 3)
                values = (rng.normal(size=size) * scale).astype(np.float32)
                values = values.astype(np.float64)
            original = values.copy()
            expected = _reference_channel_statistics(values)
            stats = channel_statistics(values)
            np.testing.assert_array_equal(values, original)
            in_place = channel_statistics(values.copy(), overwrite_input=True)
            for result in (stats, in_place):
                for key in CHANNEL_STATISTICS:
                    if key in ("sumsq", "rms"):
                        np.testing.assert_allclose(
                            result[key], expected[key], rtol=1e-14
                        )
                    else:
                        np.testing.assert_array_equal(result[key], expected[key])

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

    def test_plane_statistics(self):
        rng = np.random.default_rng(12)
        values = rng.normal(size=(1400, 9)).astype(np.float32)
        valid = rng.random(values.shape) < 0.7
        valid[625:1250] = False
        statistics = plane_statistics(values, valid)
        selected = values[valid].astype(np.float64)
        assert statistics[0] == selected.size
        assert statistics[3] == selected.min() and statistics[4] == selected.max()
        np.testing.assert_allclose(
            statistics[1:3], [selected.sum(), np.square(selected).sum()], rtol=1e-12
        )
        total = 0.0
        for start in (0, 1250):
            block = values[start : start + 625][valid[start : start + 625]]
            total += block.astype(np.float64).sum()
        assert statistics[1] == total
        empty = plane_statistics(values, np.zeros_like(valid))
        np.testing.assert_array_equal(empty, [0.0, 0.0, 0.0, np.inf, -np.inf])

    def test_merge_channel_rows_is_independent_of_the_split(self):
        rng = np.random.default_rng(13)
        rows = np.column_stack(
            [
                rng.integers(0, 50, 40).astype(float),
                rng.normal(size=40) * 10.0 ** rng.uniform(-5, 5, 40),
                rng.uniform(0, 1e6, 40),
                rng.normal(size=40),
                rng.normal(size=40) + 3.0,
            ]
        )
        expected = merge_channel_rows(np.arange(40), rows)
        for _ in range(20):
            order = rng.permutation(40)
            assert merge_channel_rows(order, rows[order]) == expected
        assert expected["sum"] == np.cumsum(rows[:, 1])[-1]
        assert expected["min"] == rows[:, 3].min()
        assert merge_channel_rows([], np.empty((0, 5))) == {
            "npts": 0.0,
            "sum": 0.0,
            "sumsq": 0.0,
            "min": np.inf,
            "max": -np.inf,
        }


class TestFrequencyAxis:
    def test_radio_velocity(self):
        np.testing.assert_allclose(
            radio_velocity_km_s([230.0e9, 229.0e9], 230.0e9),
            [0.0, SPEED_OF_LIGHT_KM_S / 230.0],
        )

    def test_velocity_widths_equal_moments_accumulator(self):
        axis = (230.1e9, -0.48828125e6, 0.0, 37)
        velocity = radio_velocity_km_s(frequency_from_axis(axis), REST_FREQUENCY)
        accumulator = MomentsAccumulator(["integrated"], velocity, (1, 1))
        np.testing.assert_array_equal(
            moment_velocity_widths(axis, REST_FREQUENCY), accumulator.coord_widths
        )
        np.testing.assert_array_equal(
            moment_velocity_widths((230.1e9, 1.0, 0.0, 1), REST_FREQUENCY), [1.0]
        )

    @pytest.mark.parametrize(
        "frequency, ranges",
        [
            (
                np.append(1.0e9 + 1.0e6 * np.arange(5), 1.1e9 + 1e6 * np.arange(4)),
                [[0, 5], [5, 9]],
            ),
            (np.append(1.0e9 + 1.0e6 * np.arange(5), 1.5e9), [[0, 5], [5, 6]]),
            (2.0e9 - 1.0e6 * np.arange(7), [[0, 7]]),
            (
                np.append(1.0e9 + 1.0e6 * np.arange(4), 1.0045e9 + 2e6 * np.arange(3)),
                [[0, 4], [4, 7]],
            ),
            (1.0e9 + 1.0e6 * (np.arange(6) + [0, 1e-5, -1e-5, 0, 0, 0]), [[0, 6]]),
            (np.array([1.0e9]), [[0, 1]]),
            (np.array([]), []),
        ],
    )
    def test_linear_channel_ranges(self, frequency, ranges):
        assert linear_channel_ranges(frequency) == ranges

    def test_linear_frequency_axis(self):
        frequency = 230.0e9 + 0.9765625e6 * np.arange(10)
        axis = linear_frequency_axis(frequency)
        assert axis == (230.0e9, 0.9765625e6, 0.0, 10)
        assert [type(value) for value in axis] == [float, float, float, int]
        assert linear_frequency_axis(frequency[3:4]) == (frequency[3], 1.0, 0.0, 1)


def _moments_cube(rng, n_channels=24, shape=(9, 7)):
    scale = 10.0 ** rng.uniform(-4, 4, (n_channels, 1, 1))
    cube = (rng.normal(size=(n_channels, *shape)) * scale).astype(np.float32)
    cube[rng.random(cube.shape) < 0.1] = np.nan
    cube[rng.random(cube.shape) < 0.05] = -0.0
    cube[rng.random(cube.shape) < 0.05] = 0.0
    masks = rng.random(cube.shape) < 0.8
    masks[:, :2, :3] = False
    return cube, masks


class TestFindContinuumMoments:
    axis = (230.0e9, 1.0e6, 0.0, 24)

    def _reference(self, moments, cube, masks, channels):
        velocity = radio_velocity_km_s(frequency_from_axis(self.axis), REST_FREQUENCY)
        accumulator = MomentsAccumulator(
            [CASA_MOMENTS[m] for m in moments], velocity, cube.shape[1:]
        )
        for channel in channels:
            accumulator.add_plane(
                channel, cube[channel].astype(np.float64), masks[channel]
            )
        return accumulator

    def _moments(self, moments, cube, masks, channels):
        widths = moment_velocity_widths(self.axis, REST_FREQUENCY)
        accumulator = FindContinuumMoments(moments, widths, cube.shape[1:])
        for channel in channels:
            plane = cube[channel]
            accumulator.add_plane(channel, plane, masks[channel] & np.isfinite(plane))
        return accumulator

    def _assert_equal(self, moments, mine, reference):
        maps = mine.finalize()
        expected = reference.finalize()
        assert list(maps) == list(moments)
        for moment in moments:
            assert maps[moment].dtype == np.float64
            np.testing.assert_array_equal(maps[moment], expected[CASA_MOMENTS[moment]])

    @pytest.mark.parametrize("moments", [[-1], [0], [8], [10], [0, 8], [0, 8, 10]])
    def test_equals_moments_accumulator(self, moments):
        rng = np.random.default_rng(21)
        cube, masks = _moments_cube(rng)
        for channels in (range(24), sorted(rng.choice(24, 9, replace=False))):
            mine = self._moments(moments, cube, masks, channels)
            self._assert_equal(
                moments, mine, self._reference(moments, cube, masks, channels)
            )
            if 0 in moments:
                integrated = mine.state()["INTEGRATED"]
                assert not np.signbit(integrated[integrated == 0]).any()

    @pytest.mark.parametrize("moments", [[-1, 0, 8, 10], [0, 8]])
    def test_merged_parts_equal_merged_accumulators(self, moments):
        rng = np.random.default_rng(22)
        cube, masks = _moments_cube(rng)
        parts = [range(0, 7), range(7, 7), range(7, 24)]
        mine = [self._moments(moments, cube, masks, part) for part in parts]
        reference = [self._reference(moments, cube, masks, part) for part in parts]
        merged = mine[0].merge(None).merge(mine[1]).merge(mine[2])
        assert merged is mine[0]
        reference[0].merge(reference[1]).merge(reference[2])
        self._assert_equal(moments, merged, reference[0])
        late = self._moments(moments, cube, masks, parts[2])
        empty = FindContinuumMoments(moments, np.ones(24), cube.shape[1:])
        assert empty.merge(late) is empty
        expected = self._reference(moments, cube, masks, parts[2])
        self._assert_equal(moments, empty, expected)

    def test_row_blocks_and_windows_equal_whole_planes(self):
        rng = np.random.default_rng(23)
        cube, masks = _moments_cube(rng)
        moments = [-1, 0, 8, 10]
        whole = self._moments(moments, cube, masks, range(24))
        widths = moment_velocity_widths(self.axis, REST_FREQUENCY)
        blocks = FindContinuumMoments(moments, widths, cube.shape[1:])
        scratch = np.empty(64)
        for channel in range(24):
            valid = masks[channel] & np.isfinite(cube[channel])
            for rows in (slice(0, 4), slice(4, 9)):
                for columns in (slice(0, 3), slice(3, 7)):
                    blocks.add_plane(
                        channel,
                        cube[channel][rows, columns],
                        valid[rows, columns],
                        rows=(rows, columns),
                        scratch=scratch,
                    )
        for moment, values in whole.finalize().items():
            np.testing.assert_array_equal(blocks.finalize()[moment], values)

    @pytest.mark.parametrize(
        "moments, bytes_per_pixel",
        [([0, 8], 12), ([0], 10), ([-1], 10), ([8], 4), ([10], 4), ([0, 8, 10], 16)],
    )
    def test_state_and_slim_maps(self, moments, bytes_per_pixel):
        rng = np.random.default_rng(24)
        cube, masks = _moments_cube(rng)
        mine = self._moments(moments, cube, masks, range(24))
        assert mine.nbytes == bytes_per_pixel * cube[0].size
        restored = FindContinuumMoments.from_state(
            moments, np.ones(1), cube.shape[1:], mine.state()
        )
        for moment, values in mine.finalize().items():
            np.testing.assert_array_equal(restored.finalize()[moment], values)
        unallocated = FindContinuumMoments.from_state(moments, [], (3, 2), None)
        assert unallocated.state() is None and unallocated.nbytes == 0
        for values in unallocated.finalize().values():
            assert values.shape == (3, 2) and np.isnan(values).all()

    def test_count_widens_from_65536_channels(self):
        plane, valid = np.ones((2, 2), np.float32), np.ones((2, 2), bool)
        for n_channels, dtype in ((65535, np.uint16), (65536, np.uint32)):
            moments = FindContinuumMoments([0], np.ones(n_channels), (2, 2))
            moments.add_plane(0, plane, valid)
            assert moments.state()["COUNT"].dtype == dtype

    def test_unsupported_moment(self):
        with pytest.raises(ValueError, match="not supported"):
            FindContinuumMoments([0, 1], np.ones(3), (2, 2))


@pytest.mark.parametrize("n_leaves", range(1, 34))
def test_merge_in_tree_order_equals_the_graphviper_tree(n_leaves):
    rng = np.random.default_rng(100 + n_leaves)
    n_channels = 3 * n_leaves
    scale = 10.0 ** rng.uniform(-6, 6, (n_channels, 1, 1))
    cube = (rng.normal(size=(n_channels, 5, 4)) * scale).astype(np.float32)
    masks = rng.random(cube.shape) < 0.9
    velocity = rng.uniform(-50.0, 50.0, n_channels).cumsum()
    moments = [-1, 0, 8, 10]
    missing = rng.random(n_leaves) < 0.3
    references, leaves = [], []
    for leaf in range(n_leaves):
        accumulator = MomentsAccumulator(
            [CASA_MOMENTS[m] for m in moments], velocity, (5, 4)
        )
        mine = None
        if not missing[leaf]:
            mine = FindContinuumMoments(moments, np.abs(np.gradient(velocity)), (5, 4))
            for channel in range(3 * leaf, 3 * leaf + 3):
                plane = cube[channel]
                accumulator.add_plane(channel, plane.astype(np.float64), masks[channel])
                mine.add_plane(channel, plane, masks[channel])
        references.append(accumulator)
        leaves.append(mine)
    expected = graphviper_tree(references, lambda left, right: left.merge(right))
    merged = merge_in_tree_order(iter(leaves))
    if missing.all():
        assert merged is None
        return
    maps = merged.finalize()
    for moment in moments:
        np.testing.assert_array_equal(
            maps[moment], expected.finalize()[CASA_MOMENTS[moment]]
        )


def test_merge_in_tree_order_tells_trees_apart():
    """A left fold of the same leaves differs from the tree in moment 0, so the
    tree test above checks the order of the merges."""
    rng = np.random.default_rng(7)
    differs = 0
    for n_leaves in range(3, 34):
        states = []
        for _ in range(n_leaves):
            leaf = FindContinuumMoments([0], np.ones(1), (8, 8))
            plane = rng.normal(size=(8, 8)) * 10.0 ** rng.uniform(-6, 6)
            leaf.add_plane(0, plane.astype(np.float32), np.ones((8, 8), dtype=bool))
            states.append(leaf.state())
        tree = merge_in_tree_order(
            FindContinuumMoments.from_state([0], [], (8, 8), dict(state))
            for state in states
        )
        fold = FindContinuumMoments([0], [], (8, 8))
        for state in states:
            copied = {key: value.copy() for key, value in state.items()}
            fold.merge(FindContinuumMoments.from_state([0], [], (8, 8), copied))
        differs += not np.array_equal(tree.finalize()[0], fold.finalize()[0])
    assert merge_in_tree_order([]) is None
    assert differs > 0


class TestPrimaryBeamMasks:
    @staticmethod
    def _port_masks(beam, limit):
        """The port's blanking (a Python float limit) and its evaluation of
        "pb">0.21 on the float64 blanked cube."""
        blanked = beam.astype(np.float64)
        inside = np.ones(beam.shape, dtype=bool)
        if limit is not None:
            blanked[beam < limit] = np.nan
            inside = ~(beam < limit)
        profile = evaluate_lattice_expression('"pb">0.21', {"pb": blanked})
        return inside, profile

    @pytest.mark.parametrize("limit", [0.2, 0.21, 0.25, 0.1999999, None])
    def test_every_float32_beam_value(self, limit):
        first, last = np.float32(0.15).view(np.int32), np.float32(0.30).view(np.int32)
        beam = np.arange(first, last + 1, dtype=np.int32).view(np.float32)
        beam = np.append(beam, np.float32([np.nan, 0.0, 1.0])).reshape(1, 1, 1, 1, -1)
        expected_inside, expected_profile = self._port_masks(beam, limit)
        for given in (limit, None if limit is None else np.float64(limit)):
            inside, profile = primary_beam_masks(beam, given)
            np.testing.assert_array_equal(inside, expected_inside)
            np.testing.assert_array_equal(profile, expected_profile)
        assert inside[..., -3] and not profile[..., -3]

    def test_float64_beam(self):
        levels = [0.2, 0.21, 0.25]
        near = [np.nextafter(level, side) for level in levels for side in (0.0, 1.0)]
        beam = np.array(near + levels + [np.nan]).reshape(1, 1, 1, 1, -1)
        for limit in (0.2, 0.25, None):
            inside, profile = primary_beam_masks(beam, limit)
            expected_inside, expected_profile = self._port_masks(beam, limit)
            np.testing.assert_array_equal(inside, expected_inside)
            np.testing.assert_array_equal(profile, expected_profile)
        assert PROFILE_LEVEL == 0.21


class TestPlaneMasks:
    @pytest.mark.parametrize("shape", [(1, 3, 1, 37, 21), (1, 1, 1, 1300, 13)])
    def test_pack_round_trip(self, shape):
        mask = np.random.default_rng(31).random(shape) < 0.3
        packed = pack_plane_masks(mask)
        assert packed.dtype == np.uint8
        assert packed.shape == shape[:-1] + (-(-shape[-1] // 8),)
        np.testing.assert_array_equal(unpack_plane_masks(packed, shape[-1]), mask)
        out = np.ones(shape, dtype=bool)
        assert unpack_plane_masks(packed, shape[-1], out=out) is out
        np.testing.assert_array_equal(out, mask)

    def test_tile_occupancy_with_ragged_tiles(self):
        mask = np.random.default_rng(32).random((1, 2, 1, 37, 21)) < 0.004
        tiles = tile_occupancy(mask, (8, 5))
        assert tiles.shape == (1, 2, 1, 5, 5)
        for i in range(5):
            for j in range(5):
                np.testing.assert_array_equal(
                    tiles[..., i, j],
                    mask[..., 8 * i : 8 * i + 8, 5 * j : 5 * j + 5].any(axis=(-2, -1)),
                )
        assert tile_occupancy(mask, (37, 21)).shape == (1, 2, 1, 1, 1)


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

    @pytest.mark.parametrize("limit", [None, 0.35])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_results_equal_the_published_port(
        self, requests_all_kinds, small_image, limit, dtype
    ):
        """Bit for bit the reductions of 424-find-continuum-report."""
        img_xds = _image_xds(dtype=dtype)
        images = {"small": small_image}
        for chunks in ([slice(None)], [slice(0, 5), slice(5, 9), slice(9, 12)]):
            partials = reference = None
            for chunk in chunks:
                arguments = (
                    img_xds.isel(frequency=chunk),
                    requests_all_kinds,
                    VARIABLES,
                    images,
                    np.arange(N_CHANNELS)[chunk],
                    "PB",
                    limit,
                )
                part = find_continuum_cube_reductions(*arguments)
                expected = _reference_cube_reductions(*arguments)
                if partials is None:
                    partials, reference = part, expected
                else:
                    merge_find_continuum_reductions(partials, part)
                    _reference_merge(reference, expected)
            results = finalize_find_continuum_reductions(partials, requests_all_kinds)
            moments, profile, per_channel, whole_cube = results
            maps = reference[0].finalize()
            for moment in (-1, 0, 8, 10):
                np.testing.assert_array_equal(
                    moments[moment], maps[CASA_MOMENTS[moment]]
                )
            order = np.argsort(reference[1]["channels"], kind="stable")
            np.testing.assert_array_equal(
                profile, np.asarray(reference[1]["values"])[order]
            )
            order = np.argsort(reference[2]["channels"], kind="stable")
            for key in CHANNEL_STATISTICS:
                values = [reference[2]["values"][i][key] for i in order]
                np.testing.assert_array_equal(per_channel[key], values)
            for key in ("npts", "sum", "sumsq", "min", "max"):
                assert partials[3][key] == reference[3][key]
            assert whole_cube["npts"] == reference[3]["npts"]

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

    def test_chunk_without_selected_channels(self, requests_all_kinds):
        requests = [requests_all_kinds[0], requests_all_kinds[3]]
        partials = find_continuum_cube_reductions(
            _image_xds().isel(frequency=slice(8, 12)),
            requests,
            VARIABLES,
            {},
            np.arange(8, 12),
        )
        assert partials[0] is None
        assert partials[1] == dict(npts=0, sum=0, sumsq=0, min=np.inf, max=-np.inf)
        with pytest.raises(ValueError, match="No channel"):
            finalize_find_continuum_reductions(partials, requests)

    def test_primary_beam_limit_masks_every_cube(self):
        img_xds = _image_xds()
        cube, pb = _casa_order(img_xds, "SKY"), _casa_order(img_xds, "PB")
        requests = [
            {
                "kind": "statistics",
                "image": name,
                "mask": "",
                "channels": None,
                "per_channel": False,
            }
            for name in ("cube", "pb")
        ]
        cube_statistics, pb_statistics = finalize_find_continuum_reductions(
            find_continuum_cube_reductions(
                img_xds, requests, VARIABLES, {}, np.arange(N_CHANNELS), "PB", 0.5
            ),
            requests,
        )
        inside = pb >= 0.5
        assert cube_statistics["npts"] == (inside & np.isfinite(cube)).sum()
        assert pb_statistics["min"] == pytest.approx(pb[inside].min())

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
        with pytest.raises(ValueError, match="histogram"):
            new_streamed_partials([request])


PORT_VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
N_BATCH_CHANNELS = 96


@pytest.fixture(scope="module")
def batch_cube():
    lines = (
        dict(
            channel=30,
            fwhm_channels=6,
            peak_jy=0.04,
            offset_pixels=(4, -4),
            size_fwhm_pixels=6,
        ),
        dict(
            channel=70,
            fwhm_channels=3,
            peak_jy=0.02,
            offset_pixels=(-6, 8),
            size_fwhm_pixels=4,
        ),
    )
    img_xds = make_find_continuum_image_xds(
        n_pixels=64,
        n_channels=N_BATCH_CHANNELS,
        primary_beam_fwhm_arcsec=10.0,
        lines=lines,
    )[list(PORT_VARIABLES.values())]
    sky, pb = img_xds["SKY_RESIDUAL"].values, img_xds["PRIMARY_BEAM"].values
    sky[0, 5, 0, 30:33, 30:33] = 0.0
    sky[0, 6, 0, 31, 31] = -0.0
    sky[0, 7, 0, 20, 20] = np.nan
    pb[0, 9, 0, 30, 31] = np.nan
    l_index, m_index = np.indices((64, 64))
    joint = (((l_index - 36) ** 2 + (m_index - 28) ** 2) < 49).astype(float)
    pbmom = pb[0, :, 0].mean(axis=0).astype(np.float64)
    pbmom[0, 0] = np.nan
    images = {"joint": joint[:, :, None, None], "pbmom": pbmom[:, :, None, None]}
    rng = np.random.default_rng(41)
    selection = sorted(rng.choice(N_BATCH_CHANNELS, 60, replace=False).tolist())
    moments = dict(
        kind="moments",
        frequency_axis=linear_frequency_axis(img_xds.frequency.values),
        rest_frequency=float(img_xds.frequency.attrs["rest_frequency"]["data"]),
    )
    statistics = dict(kind="statistics", image="sky", mask="", per_channel=False)
    requests = [
        dict(moments, image="primary_beam", mask="", channels=[12, 24, 48, 72, 84])
        | {"moments": [-1]},
        dict(moments, image="sky", mask='"sky"!=0', channels=None, moments=[0, 8]),
        dict(statistics, channels=None),
        dict(statistics, image="primary_beam", channels=None),
        dict(
            kind="profile",
            image="sky",
            mask='"joint">0 && "primary_beam">0.21',
            channels=None,
            function="mean",
        ),
        dict(
            kind="profile",
            image="sky",
            mask='"primary_beam">0.25 && "primary_beam"<0.5 && "joint"<0.5',
            channels=None,
            function="xmadm",
        ),
        dict(
            statistics,
            mask='"pbmom">0.3 && "pbmom"<0.6 && "joint"==0',
            channels=selection,
            per_channel=True,
        ),
        dict(moments, image="sky", mask="", channels=selection, moments=[0, 8]),
        dict(statistics, channels=selection),
        dict(
            moments,
            image="sky",
            mask='"sky"<0 && "joint"<0.5',
            channels=selection[::2],
            moments=[-1, 0, 10],
        ),
        dict(
            kind="profile",
            image="sky",
            mask='"sky">0.001 || "joint">0',
            channels=None,
            function="max",
        ),
        dict(
            statistics,
            image="primary_beam",
            mask='"sky"!=0 && "joint">0',
            channels=selection[:20],
            per_channel=True,
        ),
    ]
    return img_xds, images, requests


def _chunks(n_chunks):
    edges = np.linspace(0, N_BATCH_CHANNELS, n_chunks + 1).astype(int)
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def _port(img_xds, requests, images, n_chunks, limit=0.2):
    partials = [
        find_continuum_cube_reductions(
            img_xds.isel(frequency=chunk),
            requests,
            PORT_VARIABLES,
            images,
            np.arange(N_BATCH_CHANNELS)[chunk],
            "PRIMARY_BEAM",
            limit,
        )
        for chunk in _chunks(n_chunks)
    ]
    merged = graphviper_tree(partials, merge_find_continuum_reductions)
    return finalize_find_continuum_reductions(merged, requests)


def _concatenate(rows):
    merged = {"channels": np.concatenate([row["channels"] for row in rows])}
    if "rows" in rows[0]:
        merged["rows"] = np.concatenate([row["rows"] for row in rows])
    elif isinstance(rows[0]["values"], dict):
        merged["values"] = {
            key: np.concatenate([row["values"][key] for row in rows])
            for key in rows[0]["values"]
        }
    else:
        merged["values"] = np.concatenate([row["values"] for row in rows])
    return merged


def _skip_tiles(sky, positions, encoded, two_d_masks, inside, profile, offset):
    """Fill with NaN the 16-pixel tiles no request can use, as the reader fills
    the inner chunks it skips."""
    sky = sky.copy()
    for index, position in enumerate(positions):
        needed = np.zeros(sky.shape[-2:], dtype=bool)
        for request in encoded:
            channels = request["channels"]
            if channels is not None and position + offset not in channels:
                continue
            footprint = inside[0, position, 0].copy()
            if request["two_d_mask"] is not None:
                footprint &= two_d_masks[request["two_d_mask"]]
            if ("profile",) in request["cube_terms"]:
                footprint &= profile[0, position, 0]
            needed |= footprint
        tiles = tile_occupancy(needed[None, None, None], (16, 16))[0, 0, 0]
        read = np.repeat(np.repeat(tiles, 16, axis=0), 16, axis=1)
        sky[0, index, 0][~read] = np.nan
    return sky


def _ultra(img_xds, requests, images, n_chunks, batch, limit=0.2, offset=0, skip=False):
    """find_continuum_ultra's frequency tasks and merges, in process."""
    sky, pb = img_xds["SKY_RESIDUAL"].values, img_xds["PRIMARY_BEAM"].values
    encoded, two_d_masks = encode_streamed_requests(
        requests, images, sky.shape[-2:], offset
    )
    inside, profile = primary_beam_masks(pb, limit)
    tasks, buffers = [], []
    for chunk in _chunks(n_chunks):
        partials, task_buffers = new_streamed_partials(encoded), {}
        positions = np.arange(N_BATCH_CHANNELS)[chunk]
        for start in range(0, positions.size, batch):
            planes = positions[start : start + batch]
            planes_sky = sky[:, planes]
            if skip:
                planes_sky = _skip_tiles(
                    planes_sky, planes, encoded, two_d_masks, inside, profile, offset
                )
            find_continuum_batch_reductions(
                planes_sky,
                planes + offset,
                encoded,
                partials,
                inside=None if limit is None else inside[:, planes],
                profile=profile[:, planes],
                two_d_masks=two_d_masks,
                primary_beam=pb[:, planes],
                images=images,
                buffers=task_buffers,
            )
        tasks.append(partials)
        buffers.append(task_buffers)
    merged = [
        merge_in_tree_order(task[k] for task in tasks)
        if request["kind"] == "moments"
        else _concatenate([task[k] for task in tasks])
        for k, request in enumerate(encoded)
    ]
    return finalize_streamed_reductions(merged, encoded), buffers, tasks


def _assert_results_equal(results, expected, requests, exact=False):
    """Moments and profiles bitwise; per-channel statistics bitwise but for
    sumsq and rms (rtol 1e-14); whole-cube sums to rel 1e-12 (bitwise with
    ``exact``)."""
    for result, reference, request in zip(results, expected, requests, strict=True):
        if request["kind"] == "moments":
            assert list(result) == list(request["moments"])
            for moment in request["moments"]:
                np.testing.assert_array_equal(result[moment], reference[moment])
        elif request["kind"] == "profile":
            np.testing.assert_array_equal(result, reference)
        else:
            approximate = (
                ("sumsq", "rms") if request["per_channel"] else WHOLE_CUBE_SUMS
            )
            rtol = 1e-14 if request["per_channel"] else 1e-12
            assert set(result) == set(CHANNEL_STATISTICS)
            for key in CHANNEL_STATISTICS:
                if key in approximate and not exact:
                    np.testing.assert_allclose(result[key], reference[key], rtol=rtol)
                else:
                    np.testing.assert_array_equal(result[key], reference[key])


class TestBatchReductions:
    @pytest.mark.parametrize("n_chunks, batch", [(3, 1), (3, 2), (3, 5), (1, 2)])
    def test_equals_the_chunk_reductions(self, batch_cube, n_chunks, batch):
        img_xds, images, requests = batch_cube
        results, buffers, _ = _ultra(img_xds, requests, images, n_chunks, batch)
        expected = _port(img_xds, requests, images, n_chunks)
        _assert_results_equal(results, expected, requests)
        for key in ("T_mask", "T_moments", "T_statistics", "T_profile"):
            assert buffers[0][key] > 0

    def test_without_a_primary_beam_limit(self, batch_cube):
        img_xds, images, requests = batch_cube
        results, _, _ = _ultra(img_xds, requests, images, 3, 4, limit=None)
        expected = _port(img_xds, requests, images, 3, limit=None)
        _assert_results_equal(results, expected, requests)

    def test_channel_offset(self, batch_cube):
        img_xds, images, requests = batch_cube
        results, _, _ = _ultra(img_xds, requests, images, 3, 4)
        shifted, _, _ = _ultra(img_xds, requests, images, 3, 4, offset=100)
        _assert_results_equal(shifted, results, requests, exact=True)

    def test_skipped_tiles_and_bounding_boxes_change_nothing(self, batch_cube):
        img_xds, images, requests = batch_cube
        moments = dict(requests[1], mask='"joint">0 && "sky"!=0')
        compact = [requests[4], requests[11], moments | {"channels": list(range(40))}]
        results, buffers, _ = _ultra(img_xds, compact, images, 3, 4)
        assert set(buffers[0]["bounds"].values()) == {(30, 43, 22, 35)}
        skipped, _, _ = _ultra(img_xds, compact, images, 3, 4, skip=True)
        _assert_results_equal(skipped, results, compact, exact=True)
        _assert_results_equal(results, _port(img_xds, compact, images, 3), compact)
        everything, _, _ = _ultra(img_xds, requests, images, 3, 4, skip=True)
        expected = _port(img_xds, requests, images, 3)
        _assert_results_equal(everything, expected, requests)

    def test_first_round_rows_equal_the_streamed_rows(self, batch_cube):
        """The imager's rows (first_round_statistics) and P1's are bitwise equal."""
        img_xds, images, requests = batch_cube
        sky, pb = img_xds["SKY_RESIDUAL"].values, img_xds["PRIMARY_BEAM"].values
        rows = first_round_statistics(sky, pb, primary_beam_masks(pb, 0.2)[0])
        assert rows.shape == (N_BATCH_CHANNELS, 2, 5)
        _, _, tasks = _ultra(img_xds, requests[2:4], images, 3, 5)
        port = _port(img_xds, requests[2:4], images, 1)
        for index, expected in enumerate(port):
            streamed = _concatenate([task[index] for task in tasks])
            np.testing.assert_array_equal(
                streamed["channels"], np.arange(N_BATCH_CHANNELS)
            )
            np.testing.assert_array_equal(streamed["rows"], rows[:, index])
            sums = merge_channel_rows(np.arange(N_BATCH_CHANNELS), rows[:, index])
            for key in ("npts", "min", "max"):
                assert sums[key] == expected[key]
            for key in ("sum", "sumsq"):
                assert sums[key] == pytest.approx(expected[key], rel=1e-12)

    def test_moments_of_unselected_requests_stay_none(self, batch_cube):
        img_xds, images, requests = batch_cube
        _, _, tasks = _ultra(img_xds, requests[:1], images, 8, 4)
        allocated = [task[0] is not None for task in tasks]
        assert allocated == [False, True, True, False, True, False, True, True]
        assert finalize_streamed_reductions([None], requests[:1]) == [None]

    def test_empty_two_dimensional_masks(self, batch_cube):
        img_xds, images, requests = batch_cube
        empty = [
            dict(request, mask=f'"joint">5 && {request["mask"] or "T"}')
            for request in (requests[1], requests[4], requests[6], requests[8])
        ]
        results, buffers, _ = _ultra(img_xds, empty, images, 3, 4)
        assert set(buffers[0]["bounds"].values()) == {None}
        _assert_results_equal(results, _port(img_xds, empty, images, 3), empty)
        assert np.isnan(results[0][0]).all() and np.isnan(results[1]).all()
        assert (results[2]["npts"] == 0).all() and results[3]["npts"] == 0

    def test_refusals(self, batch_cube):
        sky = batch_cube[0]["SKY_RESIDUAL"].values
        with pytest.raises(ValueError, match="one time and one polarization"):
            find_continuum_batch_reductions(
                np.concatenate([sky[:, :2]] * 2), [0, 1], [], []
            )
        with pytest.raises(ValueError, match="one channel per plane"):
            find_continuum_batch_reductions(sky[:, :2], [0], [], [])


@pytest.mark.parametrize(
    "n_channels, beam_channels", [(480, [60, 120, 240, 360, 420]), (10, None)]
)
def test_first_round_requests(n_channels, beam_channels):
    axis = (230.0e9, 1.0e6, 0.0, n_channels)
    requests = find_continuum_first_round_requests(axis, 230.5e9, "sky", "beam")
    mean_beam, moments, cube_statistics, beam_statistics = requests
    assert mean_beam["image"] == "beam" and mean_beam["moments"] == [-1]
    assert mean_beam["channels"] == beam_channels
    assert moments["image"] == "sky" and moments["mask"] == '"sky"!=0'
    assert moments["moments"] == [0, 8] and moments["channels"] is None
    assert moments["frequency_axis"] == axis
    assert moments["rest_frequency"] == 230.5e9
    assert cube_statistics == dict(
        kind="statistics", image="sky", mask="", channels=None, per_channel=False
    )
    assert beam_statistics["image"] == "beam"
