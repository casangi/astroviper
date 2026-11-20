"""Tests of the uv continuum subtraction processing functions.

The fits are compared with a weighted least-squares fit of every spectrum in
CASA's basis, and the per-time continuum channels of a TOPO measurement set
with a line at a fixed LSRK frequency on the Antennae test data
(``Antennae_fld1_topo.ps.zarr``: 70 times over 15 days, 166 channels).
"""

import copy

import numpy as np
import pytest
import xarray as xr
from toolviper.utils.data import download

from astroviper.processing_functions.visibility_manipulation import (
    continuum_channel_mask,
    continuum_fit_from_find_continuum,
    subtract_polynomial_continuum,
    uv_continuum_subtraction,
)
from astroviper.processing_functions.visibility_manipulation.ms_spectral_frame_conversion import (
    doppler_factor,
)
from astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction import (
    check_continuum_fit,
    check_output_data_group,
    measurement_set_doppler_factors,
    weighted_spectrum_sums,
)

TOPO_PS = "Antennae_fld1_topo.ps.zarr"
FREQUENCY = 230e9 + 1e6 * np.arange(48)
LINE_CHANNELS = slice(20, 30)


def _basis(frequency, n_terms):
    """CASA's powers of x = (f - f_mid) / (f_min - f_mid)."""
    middle = 0.5 * (frequency.min() + frequency.max())
    x = (frequency - middle) / (frequency.min() - middle)
    return x[:, np.newaxis] ** np.arange(n_terms)


def _reference(visibility, flag, weight, frequency, continuum, fit_order):
    """Weighted least squares of every spectrum, one at a time."""
    subtracted = visibility.astype(np.complex128)
    flag_out = flag.copy()
    n_time, n_baseline, _, n_polarization = visibility.shape
    chi_squared = np.full((n_time, n_baseline, n_polarization, 2), np.nan)
    for t in range(n_time):
        for b in range(n_baseline):
            for p in range(n_polarization):
                valid = (
                    continuum[min(t, len(continuum) - 1)]
                    & ~flag[t, b, :, p]
                    & (weight[t, b, :, p] > 0)
                )
                if not valid.any():
                    flag_out[t, b, :, p] = True
                    continue
                n_terms = min(fit_order, valid.sum() - 1) + 1
                design = _basis(frequency, n_terms)
                root = np.sqrt(weight[t, b, valid, p])
                coefficients = np.linalg.lstsq(
                    design[valid] * root[:, np.newaxis],
                    visibility[t, b, valid, p] * root,
                    rcond=None,
                )[0]
                subtracted[t, b, :, p] = visibility[t, b, :, p] - design @ coefficients
                residual = subtracted[t, b, valid, p]
                chi_squared[t, b, p] = [
                    np.sum(weight[t, b, valid, p] * residual.real**2),
                    np.sum(weight[t, b, valid, p] * residual.imag**2),
                ]
    return subtracted, flag_out, chi_squared


@pytest.fixture(scope="module")
def noisy_spectra():
    rng = np.random.default_rng(470)
    shape = (5, 7, FREQUENCY.size, 2)
    coefficients = rng.normal(size=(5, 7, 2, 3)) + 1j * rng.normal(size=(5, 7, 2, 3))
    continuum = np.einsum("tbpk,fk->tbfp", coefficients, _basis(FREQUENCY, 3))
    noise = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    visibility = continuum + 0.1 * noise
    visibility[:, :, LINE_CHANNELS, :] += 3.0
    flag = rng.uniform(size=shape) < 0.2
    flag[0, 0, :, 0] = True  # all flagged
    flag[1, 1, :, 1] = True
    flag[1, 1, 25, 1] = False  # a line channel only: no continuum
    flag[2, 2, :, 0] = True
    flag[2, 2, 5, 0] = False  # one continuum channel: order 0
    weight = rng.uniform(0.5, 2.0, size=shape)
    weight[3, 3, :10, 1] = 0.0
    continuum_channels = np.ones((1, FREQUENCY.size), dtype=bool)
    continuum_channels[0, 18:32] = False
    return visibility, flag, weight, continuum_channels


@pytest.mark.parametrize("fit_order", [0, 1, 2])
def test_fit_equals_weighted_least_squares_of_every_spectrum(noisy_spectra, fit_order):
    visibility, flag, weight, continuum_channels = noisy_spectra
    subtracted, flag_out, statistics = subtract_polynomial_continuum(
        visibility, flag, weight, FREQUENCY, continuum_channels, fit_order
    )
    expected, expected_flag, chi_squared = _reference(
        visibility, flag, weight, FREQUENCY, continuum_channels, fit_order
    )
    np.testing.assert_allclose(subtracted, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(flag_out, expected_flag)
    # The spectrum without a continuum channel is kept and flagged.
    np.testing.assert_array_equal(subtracted[1, 1, :, 1], visibility[1, 1, :, 1])
    assert flag_out[1, 1, :, 1].all()
    np.testing.assert_array_equal(statistics["n_spectra"], [35, 35])
    np.testing.assert_array_equal(statistics["n_without_continuum"], [1, 1])
    np.testing.assert_array_equal(statistics["n_flagged_without_continuum"], [0, 1])
    np.testing.assert_array_equal(statistics["n_fitted"], [34, 34])
    np.testing.assert_array_equal(
        statistics["n_order_reduced"], [1, 0] if fit_order else [0, 0]
    )
    for index, part in enumerate(("real", "imag")):
        values = np.moveaxis(chi_squared[..., index], 2, 0).reshape(2, -1)
        np.testing.assert_allclose(
            statistics[f"chi_squared_sum_{part}"],
            np.nansum(values, axis=1),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            statistics[f"chi_squared_min_{part}"],
            np.nanmin(values, axis=1),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            statistics[f"chi_squared_max_{part}"],
            np.nanmax(values, axis=1),
            atol=1e-12,
        )


def test_polynomial_continuum_is_removed_exactly_and_the_line_kept():
    rng = np.random.default_rng(1)
    coefficients = rng.normal(size=(3, 4, 2, 2)) + 1j * rng.normal(size=(3, 4, 2, 2))
    continuum = np.einsum("tbpk,fk->tbfp", coefficients, _basis(FREQUENCY, 2))
    line = np.zeros(FREQUENCY.size)
    line[LINE_CHANNELS] = np.hanning(10) + 0.5
    visibility = (continuum + line[np.newaxis, np.newaxis, :, np.newaxis]).astype(
        np.complex64
    )
    continuum_channels = np.ones((1, FREQUENCY.size), dtype=bool)
    continuum_channels[0, LINE_CHANNELS] = False
    subtracted, flag_out, _ = subtract_polynomial_continuum(
        visibility,
        np.zeros(visibility.shape, dtype=bool),
        np.ones(visibility.shape, dtype=np.float32),
        FREQUENCY,
        continuum_channels,
        1,
    )
    assert subtracted.dtype == np.complex64
    np.testing.assert_allclose(
        subtracted, np.broadcast_to(line[:, np.newaxis], subtracted.shape), atol=2e-6
    )
    assert not flag_out.any()


def test_continuum_channels_may_change_with_time():
    visibility = np.ones((2, 1, FREQUENCY.size, 1), dtype=np.complex128)
    visibility[1, 0, :10, 0] = 5.0  # a line at the first channels, second time
    continuum_channels = np.ones((2, FREQUENCY.size), dtype=bool)
    continuum_channels[1, :10] = False
    subtracted, _, _ = subtract_polynomial_continuum(
        visibility,
        np.zeros(visibility.shape, dtype=bool),
        np.ones(visibility.shape),
        FREQUENCY,
        continuum_channels,
        0,
    )
    np.testing.assert_allclose(subtracted[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(subtracted[1, 0, :10, 0], 4.0)
    np.testing.assert_allclose(subtracted[1, 0, 10:, 0], 0.0, atol=1e-14)


def test_non_finite_samples_are_not_fitted(noisy_spectra):
    visibility, flag, weight, continuum_channels = noisy_spectra
    visibility = visibility.copy()
    weight = weight.copy()
    visibility[4, 4, 3, 0] = np.nan
    weight[4, 5, 7, 1] = np.nan
    subtracted, flag_out, _ = subtract_polynomial_continuum(
        visibility, flag, weight, FREQUENCY, continuum_channels, 1
    )
    reference_flag = flag.copy()
    reference_flag[4, 4, 3, 0] = reference_flag[4, 5, 7, 1] = True
    expected, _, _ = _reference(
        np.nan_to_num(visibility),
        reference_flag,
        np.nan_to_num(weight),
        FREQUENCY,
        continuum_channels,
        1,
    )
    assert np.isnan(subtracted[4, 4, 3, 0])
    finite = np.isfinite(visibility)
    np.testing.assert_allclose(subtracted[finite], expected[finite], atol=1e-12)
    assert not flag_out[4, 4, 3, 0]


def test_concurrent_threads_get_the_results_of_one_thread():
    """Node tasks of a threaded scheduler run the fit in threads of one process;
    an OpenBLAS built with OpenMP gives wrong products when called from several
    threads at once."""
    import threading

    rng = np.random.default_rng(470)
    frequency = 345e9 + 1.8e5 * np.arange(166)
    continuum_channels = np.ones((1, 166), dtype=bool)
    continuum_channels[0, 60:90] = False
    blocks = []
    for _ in range(3):
        shape = (24, 86, 166, 2)
        visibility = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        blocks.append(
            (
                visibility.astype(np.complex64),
                rng.uniform(size=shape) < 0.05,
                rng.uniform(0.5, 2.0, shape).astype(np.float32),
            )
        )

    def fit(visibility, flag, weight):
        return subtract_polynomial_continuum(
            visibility, flag, weight, frequency, continuum_channels, 1
        )[0]

    expected = [fit(*block) for block in blocks]
    wrong = []

    def work(index):
        for _ in range(10):
            if not np.array_equal(fit(*blocks[index]), expected[index]):
                wrong.append(index)
                return

    for _ in range(3):
        threads = [threading.Thread(target=work, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert wrong == []


def test_infinite_weights_are_not_valid(noisy_spectra):
    visibility, flag, weight, continuum_channels = noisy_spectra
    infinite = weight.copy()
    infinite[3, 4, 5, 0] = np.inf
    zero = weight.copy()
    zero[3, 4, 5, 0] = 0.0
    with_infinite = subtract_polynomial_continuum(
        visibility, flag, infinite, FREQUENCY, continuum_channels, 1
    )
    with_zero = subtract_polynomial_continuum(
        visibility, flag, zero, FREQUENCY, continuum_channels, 1
    )
    np.testing.assert_array_equal(with_infinite[0], with_zero[0])
    assert np.isfinite(with_infinite[2]["chi_squared_sum_real"]).all()


@pytest.mark.parametrize(
    "fit_order, continuum",
    [(4, slice(0, 58)), (5, slice(940, 980)), (6, slice(0, 192))],
)
def test_high_orders_on_narrow_continuum_are_least_squares(fit_order, continuum):
    """Narrow continuum coverage and high orders make the normal equations of
    the whole band ill-conditioned; the fit stays the weighted least-squares
    fit of the continuum samples."""
    rng = np.random.default_rng(fit_order)
    frequency = 100e9 + 1e5 * np.arange(1920)
    shape = (2, 3, frequency.size, 1)
    visibility = 1.0 + 0.1 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    flag = np.zeros(shape, dtype=bool)
    weight = rng.uniform(0.5, 2.0, size=shape)
    continuum_channels = np.zeros((1, frequency.size), dtype=bool)
    continuum_channels[0, continuum] = True
    subtracted, _, statistics = subtract_polynomial_continuum(
        visibility, flag, weight, frequency, continuum_channels, fit_order
    )
    expected, _, chi_squared = _reference(
        visibility, flag, weight, frequency, continuum_channels, fit_order
    )
    # The reference fits CASA's powers of x over the band (condition number up
    # to 2e9 here); the normal equations of that basis were off by up to 8.
    np.testing.assert_allclose(
        subtracted[:, :, continuum], expected[:, :, continuum], atol=1e-7
    )
    np.testing.assert_allclose(
        statistics["chi_squared_sum_real"],
        np.nansum(chi_squared[..., 0]),
        rtol=1e-9,
    )


@pytest.mark.parametrize("fit_order", [4, 5])
def test_spectra_with_clustered_valid_samples_are_fitted_by_least_squares(
    fit_order, monkeypatch
):
    """Continuum at both 10 % edges of the band, the upper edge flagged on half
    the spectra: their valid samples cluster at one edge, their normal
    equations are ill-conditioned and the fallback fits them."""
    import importlib

    module = importlib.import_module(
        "astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction"
    )
    calls = []
    fallback = module._weighted_least_squares

    def counting(*args):
        calls.append(len(args[1]))
        return fallback(*args)

    monkeypatch.setattr(module, "_weighted_least_squares", counting)
    rng = np.random.default_rng(fit_order)
    frequency = 100e9 + 1e5 * np.arange(1920)
    shape = (2, 4, frequency.size, 1)
    visibility = 1.0 + 0.1 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    flag = np.zeros(shape, dtype=bool)
    flag[:, :2, 1728:] = True
    weight = rng.uniform(0.5, 2.0, size=shape)
    continuum_channels = np.zeros((1, frequency.size), dtype=bool)
    continuum_channels[0, :192] = continuum_channels[0, 1728:] = True
    subtracted, _, _ = subtract_polynomial_continuum(
        visibility, flag, weight, frequency, continuum_channels, fit_order
    )
    expected, _, _ = _reference(
        visibility, flag, weight, frequency, continuum_channels, fit_order
    )
    assert sum(calls) == 4  # the 4 spectra flagged at the upper edge
    clustered = np.s_[:, :2, :192]
    np.testing.assert_allclose(subtracted[clustered], expected[clustered], atol=1e-7)
    spread = np.s_[:, 2:, continuum_channels[0]]
    np.testing.assert_allclose(subtracted[spread], expected[spread], atol=1e-7)


def test_a_high_order_far_from_its_channels_never_leaves_unflagged_overflows():
    frequency = 100e9 + 1e5 * np.arange(1920)
    shape = (1, 2, frequency.size, 1)
    rng = np.random.default_rng(16)
    visibility = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64
    )
    continuum_channels = np.zeros((1, frequency.size), dtype=bool)
    continuum_channels[0, :90] = True
    for fit_order in (16, 89):
        subtracted, flag_out, _ = subtract_polynomial_continuum(
            visibility,
            np.zeros(shape, dtype=bool),
            np.ones(shape, dtype=np.float32),
            frequency,
            continuum_channels,
            fit_order,
        )
        assert np.isfinite(subtracted[~flag_out]).all()
        assert np.isfinite(subtracted[:, :, :90]).all()


def test_a_fit_order_no_spectrum_can_use_costs_nothing(noisy_spectra):
    visibility, flag, weight, continuum_channels = noisy_spectra
    n_valid_max = int(continuum_channels.sum())
    usable = subtract_polynomial_continuum(
        visibility, flag, weight, FREQUENCY, continuum_channels, n_valid_max - 1
    )
    huge = subtract_polynomial_continuum(
        visibility, flag, weight, FREQUENCY, continuum_channels, 10**6
    )
    np.testing.assert_array_equal(usable[0], huge[0])
    assert huge[2]["n_order_reduced"].sum() == huge[2]["n_fitted"].sum()


def test_weighted_spectrum_sums_skip_invalid_samples():
    visibility = np.ones((2, 1, 2, 1), dtype=np.complex128)
    flag = np.zeros(visibility.shape, dtype=bool)
    weight = np.ones(visibility.shape)
    visibility[0, 0, 0, 0] = np.nan  # flagged NaN
    flag[0, 0, 0, 0] = True
    visibility[1, 0, 0, 0] = np.nan  # unflagged NaN
    weight[0, 0, 1, 0] = np.inf
    weight[1, 0, 1, 0] = np.nan
    weighted, total = weighted_spectrum_sums(visibility, flag, weight)
    np.testing.assert_array_equal(weighted, [[0.0], [0.0]])
    np.testing.assert_array_equal(total, [[0.0], [0.0]])


@pytest.mark.parametrize(
    "out_name, modified, dims, overwrite, message",
    [
        (
            "base",
            {"correlated_data": "V_OUT", "flag": "F_OUT"},
            {},
            True,
            "is the input",
        ),
        ("model", {"correlated_data": "V_OUT", "flag": "F_OUT"}, {}, False, "exists"),
        ("line", {"correlated_data": "V_OUT", "flag": "WEIGHT"}, {}, True, "'base'"),
        (
            "line",
            {"correlated_data": "VISIBILITY_MODEL", "flag": "F"},
            {},
            True,
            "'model'",
        ),
        (
            "line",
            {"correlated_data": "V_OUT", "flag": "F_OUT"},
            {"V_OUT": ("time", "baseline_id", "frequency", "polarization")},
            False,
            "set overwrite=True",
        ),
        (
            "line",
            {"correlated_data": "V_OUT", "flag": "F_OUT"},
            {"F_OUT": ("time", "baseline_id")},
            True,
            "not those of the visibilities",
        ),
    ],
)
def test_output_data_group_refusals(out_name, modified, dims, overwrite, message):
    data_groups = {
        "base": {"correlated_data": "VISIBILITY", "flag": "FLAG", "weight": "WEIGHT"},
        "model": {"correlated_data": "VISIBILITY_MODEL", "date": "2026"},
    }
    with pytest.raises(ValueError, match=message):
        check_output_data_group(
            data_groups, dims, "base", out_name, modified, overwrite
        )


def test_an_earlier_output_group_is_replaced_with_overwrite():
    data_groups = {
        "base": {"correlated_data": "VISIBILITY", "flag": "FLAG", "weight": "WEIGHT"},
        "line": {"correlated_data": "V_OUT", "flag": "F_OUT", "weight": "WEIGHT"},
    }
    visibility_dims = ("time", "baseline_id", "frequency", "polarization")
    check_output_data_group(
        data_groups,
        {"V_OUT": visibility_dims, "F_OUT": visibility_dims},
        "base",
        "line",
        {"correlated_data": "V_OUT", "flag": "F_OUT"},
        True,
    )


def test_singular_normal_equations_are_solved_by_least_squares():
    from astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction import (
        _solve_normal_equations,
    )

    normal = np.array([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 1.0]]])
    right_hand_side = np.array([[[2.0, 1.0], [2.0, 1.0]], [[2.0, 4.0], [1.0, 2.0]]])
    solution = _solve_normal_equations(normal, right_hand_side)
    np.testing.assert_allclose(normal @ solution, right_hand_side)
    np.testing.assert_allclose(solution[0], [[1.0, 0.5], [1.0, 0.5]])


def test_single_channel_spectrum():
    subtracted, flag_out, statistics = subtract_polynomial_continuum(
        np.full((1, 1, 1, 1), 2.0 + 1.0j),
        np.zeros((1, 1, 1, 1), dtype=bool),
        np.ones((1, 1, 1, 1)),
        FREQUENCY[:1],
        np.ones((1, 1), dtype=bool),
        1,
    )
    np.testing.assert_allclose(subtracted, 0.0)
    assert statistics["n_order_reduced"][0] == 1


def test_continuum_channel_mask():
    frequency = np.arange(10.0)
    ranges = [[6.0, 7.0], [0.5, 2.0], [1.5, 3.0]]  # unsorted, overlapping
    mask = continuum_channel_mask(frequency, ranges)
    assert mask.shape == (1, 10)
    np.testing.assert_array_equal(
        np.flatnonzero(mask[0]), [1, 2, 3, 6, 7]
    )  # edges included
    shifted = continuum_channel_mask(frequency, ranges, np.array([1.0, 2.0]))
    assert shifted.shape == (2, 10)
    np.testing.assert_array_equal(shifted[0], mask[0])
    # At the second time the channel frequencies double: 2 f in a range.
    np.testing.assert_array_equal(np.flatnonzero(shifted[1]), [1, 3])
    nested = continuum_channel_mask(frequency, [[0.0, 9.0], [2.0, 3.0]])
    assert nested.all()


def test_continuum_fit_from_find_continuum():
    result = {
        "continuum_frequency_ranges": [(1.0e9, 1.1e9), (1.3e9, 1.4e9)],
        "frequency_frame": "LSRK",
        "low_bandwidth": False,
        "low_spread": False,
    }
    assert continuum_fit_from_find_continuum(result) == {
        "continuum_frequency_ranges": [[1.0e9, 1.1e9], [1.3e9, 1.4e9]],
        "continuum_frequency_frame": "lsrk",
        "fit_order": 1,
    }
    for flag in ("low_bandwidth", "low_spread"):
        fit = continuum_fit_from_find_continuum({**result, flag: True})
        assert fit["fit_order"] == 0
    fit = continuum_fit_from_find_continuum({**result, "frequency_frame": "BARY"})
    assert fit["continuum_frequency_frame"] == "BARY"
    fit = continuum_fit_from_find_continuum({**result, "frequency_frame": "GEO"})
    assert fit["continuum_frequency_frame"] == "gcrs"


@pytest.mark.parametrize(
    "ranges, frame, order, modified, message",
    [
        ([], "lsrk", 1, None, "non-empty"),
        ([1.0, 2.0], "lsrk", 1, None, "non-empty"),
        ([[1.0, 2.0, 3.0]], "lsrk", 1, None, "non-empty"),
        ([["a", "b"]], "lsrk", 1, None, "pairs of frequencies"),
        ([[2.0, 1.0]], "lsrk", 1, None, "low <= high"),
        ([[1.0, np.inf]], "lsrk", 1, None, "finite"),
        ([[1.0, 2.0]], None, 1, None, "frame must be a string"),
        ([[1.0, 2.0]], "lsrk", True, None, "integer"),
        ([[1.0, 2.0]], "lsrk", 1.0, None, "integer"),
        ([[1.0, 2.0]], "lsrk", -1, None, "at least 0"),
        (
            [[1.0, 2.0]],
            "lsrk",
            1,
            {"correlated_data": "VISIBILITY_LINE"},
            "exactly the roles",
        ),
        (
            [[1.0, 2.0]],
            "lsrk",
            1,
            {"correlated_data": "VISIBILITY_LINE", "flag": "VISIBILITY_LINE"},
            "a different variable for each role",
        ),
    ],
)
def test_configuration_refusals(ranges, frame, order, modified, message):
    with pytest.raises(ValueError, match=message):
        check_continuum_fit(ranges, frame, order, modified)


def test_weighted_spectrum_sums():
    visibility = np.arange(8.0).reshape(2, 1, 2, 2) + 1j
    flag = np.zeros(visibility.shape, dtype=bool)
    flag[0, 0, 0, 0] = True
    weight = np.full(visibility.shape, 2.0)
    weighted, total = weighted_spectrum_sums(visibility, flag, weight)
    np.testing.assert_allclose(total, [[2.0, 4.0], [4.0, 4.0]])
    np.testing.assert_allclose(weighted[0, 0], 2.0 * (4.0 + 1j))
    np.testing.assert_allclose(weighted[1, 1], 2.0 * (3.0 + 7.0 + 2j))


def _measurement_set(visibility, observer="lsrk", data_groups=None, dims=None):
    """A minimal in-memory MSv4 with the base data group."""
    dims = dims or ("time", "baseline_id", "frequency", "polarization")
    sizes = dict(zip(dims, visibility.shape, strict=True))
    dataset = xr.Dataset(
        {
            "VISIBILITY": (dims, visibility, {"type": "quantity", "units": "Jy"}),
            "FLAG": (dims, np.zeros(visibility.shape, dtype=bool)),
            "WEIGHT": (dims, np.ones(visibility.shape, dtype=np.float32)),
        },
        coords={
            "time": np.arange(sizes["time"], dtype=float),
            "baseline_id": np.arange(sizes["baseline_id"]),
            "frequency": (
                "frequency",
                FREQUENCY[: sizes["frequency"]],
                {"units": "Hz", "observer": observer, "type": "spectral_coord"},
            ),
            "polarization": ["XX", "YY"][: sizes["polarization"]],
        },
        attrs={
            "data_groups": data_groups
            or {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "description": "test",
                }
            }
        },
    )
    return xr.DataTree(dataset)


def _line_measurement_set():
    visibility = np.ones((3, 2, FREQUENCY.size, 2), dtype=np.complex64)
    visibility[:, :, LINE_CHANNELS, :] += 2.0
    return _measurement_set(visibility)


def test_uv_continuum_subtraction_adds_the_output_data_group():
    ms_xdt = _line_measurement_set()
    continuum = [[FREQUENCY[0], FREQUENCY[19]], [FREQUENCY[30], FREQUENCY[-1]]]
    result = uv_continuum_subtraction(ms_xdt, continuum, "lsrk", fit_order=1)
    group = ms_xdt.attrs["data_groups"]["continuum_subtracted"]
    assert group["correlated_data"] == "VISIBILITY_CONTINUUM_SUBTRACTED"
    assert group["flag"] == "FLAG_CONTINUUM_SUBTRACTED"
    assert group["weight"] == "WEIGHT"
    assert group["description"].startswith("test; Continuum subtracted from base")
    output = ms_xdt["VISIBILITY_CONTINUUM_SUBTRACTED"]
    assert output.dtype == np.complex64
    assert output.attrs["units"] == "Jy"
    expected = np.zeros(FREQUENCY.size)
    expected[LINE_CHANNELS] = 2.0
    np.testing.assert_allclose(
        output.isel(time=0, baseline_id=0, polarization=1), expected, atol=1e-6
    )
    assert not ms_xdt["FLAG_CONTINUUM_SUBTRACTED"].values.any()
    statistics = result["fit_statistics"]
    assert statistics["polarization"] == ["XX", "YY"]
    np.testing.assert_array_equal(statistics["n_fitted"], [6, 6])
    assert (
        result["n_continuum_channels_min"] == result["n_continuum_channels_max"] == 38
    )
    assert result["continuum_channel_fraction"] == pytest.approx(38 / 48)
    assert result["T_fit_and_subtract"] >= 0.0

    with pytest.raises(ValueError, match="exists in the measurement set"):
        uv_continuum_subtraction(ms_xdt, continuum, "lsrk", fit_order=0)
    uv_continuum_subtraction(ms_xdt, continuum, "lsrk", fit_order=0, overwrite=True)
    renamed = {"correlated_data": "VISIBILITY_LINE", "flag": "FLAG_LINE"}
    uv_continuum_subtraction(
        ms_xdt,
        continuum,
        "lsrk",
        ms_data_group_out_name="line",
        ms_data_group_out_modified=renamed,
        input_checked=True,
    )
    assert ms_xdt.attrs["data_groups"]["line"]["correlated_data"] == "VISIBILITY_LINE"


def test_uv_continuum_subtraction_refusals():
    continuum = [[FREQUENCY[0], FREQUENCY[10]]]
    ms_xdt = _line_measurement_set()
    with pytest.raises(ValueError, match="not in the measurement set"):
        uv_continuum_subtraction(ms_xdt, continuum, "lsrk", ms_data_group_in_name="x")
    with pytest.raises(ValueError, match="is a variable of the data group 'base'"):
        uv_continuum_subtraction(
            ms_xdt,
            continuum,
            "lsrk",
            ms_data_group_out_modified={"correlated_data": "VISIBILITY", "flag": "F2"},
        )
    with pytest.raises(ValueError, match="cannot be converted"):
        uv_continuum_subtraction(ms_xdt, continuum, "TOPO")
    with pytest.raises(ValueError, match="one value per time"):
        uv_continuum_subtraction(ms_xdt, continuum, "lsrk", doppler_factors=np.ones(2))
    no_weight = _measurement_set(
        np.ones((1, 1, 4, 1), dtype=np.complex64),
        data_groups={"base": {"correlated_data": "VISIBILITY", "flag": "FLAG"}},
    )
    with pytest.raises(ValueError, match="no 'weight' role"):
        uv_continuum_subtraction(no_weight, continuum, "lsrk")
    transposed = _measurement_set(
        np.ones((1, 1, 1, 4), dtype=np.complex64),
        dims=("time", "baseline_id", "polarization", "frequency"),
    )
    with pytest.raises(ValueError, match="dimensions"):
        uv_continuum_subtraction(transposed, continuum, "lsrk")
    with pytest.raises(ValueError, match="not those of the visibilities"):
        uv_continuum_subtraction(
            _line_measurement_set(),
            continuum,
            "lsrk",
            ms_data_group_out_modified={"correlated_data": "time", "flag": "F2"},
            overwrite=True,
        )
    in_ghz = _line_measurement_set()
    in_ghz.frequency.attrs["units"] = "GHz"
    with pytest.raises(ValueError, match="in Hz"):
        uv_continuum_subtraction(in_ghz, continuum, "lsrk")
    lsrk = _line_measurement_set()
    with pytest.raises(ValueError, match="cannot be converted"):
        measurement_set_doppler_factors(lsrk, "BARY")
    assert measurement_set_doppler_factors(lsrk, "lsrk") is None


@pytest.fixture(scope="module")
def topo_ms(tmp_path_factory):
    folder = tmp_path_factory.mktemp("uv_continuum_subtraction")
    download(TOPO_PS, folder=str(folder))
    ps = xr.open_datatree(folder / TOPO_PS)
    return ps[next(iter(ps.children))].load()


def test_continuum_channels_of_a_topo_measurement_set_follow_the_doppler_shift(
    topo_ms,
):
    """A line at a fixed LSRK frequency drifts over the TOPO channels; with LSRK
    ranges that exclude it the continuum is removed exactly at every time."""
    frequency = topo_ms.frequency.values
    doppler = doppler_factor(topo_ms, "lsrk")
    lsrk = doppler[:, np.newaxis] * frequency[np.newaxis, :]
    width = abs(frequency[1] - frequency[0])
    line_frequency = np.median(lsrk[:, 80])
    line = np.exp(-0.5 * ((lsrk - line_frequency) / (1.5 * width)) ** 2)
    line[np.abs(lsrk - line_frequency) > 6 * width] = 0.0
    rng = np.random.default_rng(3)
    shape = topo_ms.VISIBILITY.shape
    coefficients = rng.normal(size=shape[:2] + (shape[3], 2)) + 1j * rng.normal(
        size=shape[:2] + (shape[3], 2)
    )
    continuum = np.einsum("tbpk,fk->tbfp", coefficients, _basis(frequency, 2))
    visibility = continuum + line[:, np.newaxis, :, np.newaxis]
    ms_xdt = topo_ms.copy()
    dataset = ms_xdt.to_dataset()
    dataset["VISIBILITY"] = (dataset.VISIBILITY.dims, visibility)
    dataset["FLAG"] = (dataset.FLAG.dims, np.zeros(shape, dtype=bool))
    # The processing function adds its group to the data_groups dict in place.
    dataset.attrs = copy.deepcopy(topo_ms.attrs)
    ms_xdt.dataset = dataset

    excluded = [line_frequency - 7 * width, line_frequency + 7 * width]
    band = [lsrk.min() - width, lsrk.max() + width]
    ranges = [[band[0], excluded[0]], [excluded[1], band[1]]]
    np.testing.assert_array_equal(
        measurement_set_doppler_factors(ms_xdt, "lsrk"), doppler
    )
    result = uv_continuum_subtraction(ms_xdt, ranges, "lsrk", fit_order=1)
    # The missing baselines of the data have NaN weights, unflagged: they have
    # no valid sample and are flagged.
    missing = np.isnan(topo_ms.WEIGHT.values).all(axis=2)
    assert missing.any()
    assert ms_xdt.FLAG_CONTINUUM_SUBTRACTED.values.all(axis=2)[missing].all()
    assert not ms_xdt.FLAG_CONTINUUM_SUBTRACTED.values.any(axis=2)[~missing].any()
    expected = np.broadcast_to(line[:, np.newaxis, :, np.newaxis], shape)
    np.testing.assert_allclose(
        ms_xdt.VISIBILITY_CONTINUUM_SUBTRACTED.values.transpose(0, 1, 3, 2)[~missing],
        expected.transpose(0, 1, 3, 2)[~missing],
        atol=1e-9,
    )
    # The 14 excluded channels move over the TOPO channels in the 15 days.
    assert result["n_continuum_channels_min"] == 152
    mask = continuum_channel_mask(frequency, ranges, doppler)
    assert (~mask).any(axis=0).sum() - (~mask).all(axis=0).sum() > 10

    # The same ranges taken as TOPO frequencies miss the drift.
    static = topo_ms.copy()
    static_dataset = dataset.assign_coords(
        frequency=dataset.frequency.assign_attrs(observer="lsrk")
    )
    static_dataset.attrs = copy.deepcopy(topo_ms.attrs)
    static.dataset = static_dataset
    uv_continuum_subtraction(static, ranges, "lsrk", fit_order=1)
    error = np.abs(
        static.VISIBILITY_CONTINUUM_SUBTRACTED.values.transpose(0, 1, 3, 2)[~missing]
        - expected.transpose(0, 1, 3, 2)[~missing]
    ).max()
    assert error > 1e-3
