"""Tests of find_continuum_ultra's spectrum steps against the vendored
findContinuum and the simple port's mask expressions."""

import contextlib
import io
import random
import re
import warnings

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum import (
    evaluate_lattice_expression,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.spectrum import (
    computed_round_spectrum,
    extra_mask_regions,
    extra_mask_spectrum,
    initial_peak_over_mad,
    line_full_ranges,
    normalize_spectrum_by_mad,
    quantize_extra_mask_spectrum,
    remove_initial_quadratic,
    replace_line_full_ranges_with_noise,
    robust_continuum_noise,
    smooth_spectrum,
    spectrum_regions,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    AnnulusLevels,
)
from tests.unit.processing_functions.imaging.find_continuum_ultra.vendored_oracle import (
    findContinuum,
    quiet,
    random_ranges,
    random_spectrum,
    vendored_block,
)

# fC 8515-8565: the normalization of computeStatisticalSpectrumFromMask.
_NORMALIZATION = vendored_block(
    8515,
    "originalMAD = MAD(avgIntensity)",
    8565,
    "normalized = True",
    ("avgIntensity", "xmadm", "projectCode", "normalized"),
)
# fC 8283-8295: the peak over MAD, the cap of n_bin and the smoothing of a
# computed round.
_PEAK_CAP_AND_SMOOTHING = vendored_block(
    8283,
    "initialPeakOverMad = (np.nanmax(intensity) - np.nanmedian(intensity)) / "
    "MAD(intensity)",
    8295,
    "intensity = smooth(intensity, nbin, window)  # original heuristic",
    ("intensity", "nbin", "maxBaseline", "window"),
)
# The annulus levels of line_set's primary-beam cube, which "%g" changes.
CUBE_LEVELS = AnnulusLevels(0.70217347, 0.87508026)
MEAN_PRIMARY_BEAM_LEVELS = AnnulusLevels(0.31234567, 0.45678912)
# A level that "%f" changes (imaging's onlyExtraMask level).
EXTRA_LEVEL = 0.0100961309


def _same(value, expected):
    """Equal bit for bit, NaN equal to NaN."""
    value = np.asarray(value, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    valued = ~np.isnan(value)
    return np.array_equal(value, expected, equal_nan=True) and np.array_equal(
        np.signbit(value[valued]), np.signbit(expected[valued])
    )


def _selection(ranges):
    """findContinuum's selection string of ``[start, stop)`` ranges."""
    return ";".join(f"{start}~{stop - 1}" for start, stop in ranges)


def _ranges(selection):
    """``[start, stop)`` ranges of a findContinuum selection string."""
    return [
        [int(first), int(last) + 1]
        for first, last in (part.split("~") for part in selection.split(";"))
    ]


def _findcontinuum(function, *args):
    """A vendored function's result and what it printed, with the process-wide
    random state that it may reseed restored."""
    state = random.getstate()
    log = io.StringIO()
    try:
        with quiet(), contextlib.redirect_stdout(log), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = function(*args)
    finally:
        random.setstate(state)
    return result, log.getvalue()


def _increasing_ranges(rng, n_channels, n_ranges):
    """Up to ``n_ranges`` ranges with a channel between consecutive ones,
    starting and ending inside the spectrum, or at its first or last channel
    when there are several."""
    n_ranges = min(n_ranges, (n_channels - 1) // 2)
    edges = np.sort(rng.choice(np.arange(1, n_channels), 2 * n_ranges, replace=False))
    ranges = [[int(start), int(stop)] for start, stop in edges.reshape(-1, 2)]
    if len(ranges) > 1 and rng.random() < 0.3:
        ranges[0][0] = 0
    if len(ranges) > 1 and rng.random() < 0.3:
        ranges[-1][1] = n_channels
    return ranges


def _spectra(rng, n_channels):
    """A mean spectrum and a scaled-MAD spectrum that share an atmospheric
    line of random depth, NaN in one channel in a tenth of the cases (an empty
    channel of the port)."""
    channels = np.arange(n_channels)
    noise = 10.0 ** rng.uniform(-4, -1)
    width = rng.uniform(0.5, max(1.0, n_channels / 6))
    atmosphere = np.exp(-0.5 * ((channels - rng.uniform(0, n_channels)) / width) ** 2)
    depth = float(rng.choice([0.0, 0.5, 5.0, 50.0]))
    scaled_mad = (
        noise * (1 + depth * atmosphere) * (1 + 0.05 * rng.standard_normal(n_channels))
    )
    mean = random_spectrum(rng, n_channels)
    mean += rng.uniform(0, 3) * depth * noise * atmosphere
    if rng.random() < 0.1:
        channel = rng.integers(0, n_channels)
        mean[channel] = scaled_mad[channel] = np.nan
    return mean, scaled_mad


def test_normalize_spectrum_by_mad_equals_findcontinuum():
    rng = np.random.default_rng(11)
    outcomes = set()
    for _ in range(400):
        mean, scaled_mad = _spectra(rng, int(rng.choice([8, 40, 120, 480, 1920])))
        inputs = mean.copy(), scaled_mad.copy()
        with quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = _NORMALIZATION(mean.copy(), scaled_mad.copy(), "", False)
            spectrum, normalized = normalize_spectrum_by_mad(mean, scaled_mad)
        assert normalized == expected["normalized"]
        assert _same(spectrum, expected["avgIntensity"])
        assert _same(mean, inputs[0]) and _same(scaled_mad, inputs[1])
        outcomes.add((normalized, bool(expected["mymin"] < 0.05 * expected["mymax"])))
    # accepted, rejected, and accepted with the factor's floor
    assert outcomes >= {(True, False), (False, False), (True, True)}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_remove_initial_quadratic_equals_findcontinuum():
    rng = np.random.default_rng(12)
    cases = [
        random_spectrum(rng, int(rng.choice([8, 64, 241, 480, 1920])))
        for _ in range(150)
    ]
    for n_channels in (40, 480, 3840):
        offsets = (np.arange(n_channels) - rng.uniform(0, n_channels)) / n_channels
        noise = 1e-3 * rng.standard_normal(n_channels)
        cases += [1.0 + noise + 0.5 * sign * offsets**2 for sign in (1.0, -1.0)]
        # More than half of the values equal: both scaled MADs are 0.
        cases.append(
            np.r_[np.full(n_channels - n_channels // 3, 0.25), noise[: n_channels // 3]]
        )
    outcomes = set()
    for spectrum in cases:
        expected, _ = _findcontinuum(
            findContinuum.removeInitialQuadraticIfNeeded, spectrum.copy()
        )
        result = remove_initial_quadratic(spectrum)
        assert result[1] == expected[1]
        assert _same(result[0], expected[0]) and _same(result[2], expected[2])
        outcomes.add((result[1], result[2] == 1.0))
    assert outcomes >= {(True, False), (False, False), (False, True)}


def test_initial_peak_over_mad_and_smoothing_equal_findcontinuum():
    rng = np.random.default_rng(13)
    n_bins = set()
    for _ in range(400):
        n_channels = int(rng.choice([8, 9, 16, 64, 120, 480, 3840]))
        spectrum = random_spectrum(rng, n_channels)
        if rng.random() < 0.1:
            spectrum[rng.integers(0, n_channels)] = np.nan
        n_bin = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 8]))
        max_baseline = float(rng.choice([25.0, 60.0, 60.000001, 400.0, 16000.0]))
        with quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = _PEAK_CAP_AND_SMOOTHING(
                spectrum.copy(), n_bin, max_baseline, "flat"
            )
            peak_over_mad = initial_peak_over_mad(spectrum)
            smoothed, capped = smooth_spectrum(
                spectrum, n_bin, max_baseline, peak_over_mad
            )
        assert _same(peak_over_mad, expected["initialPeakOverMad"])
        assert capped == expected["nbin"]
        assert _same(smoothed, expected["intensity"])
        n_bins.add((n_bin, capped, max_baseline <= 60))
    # uncapped, capped at 2, and capped at 3 for the ACA (also from 2)
    assert {(5, 5, False), (5, 2, False), (5, 3, True), (2, 3, True)} <= n_bins


def test_smoothing_wider_than_the_spectrum_raises_as_findcontinuum():
    spectrum = np.random.default_rng(14).standard_normal(8)
    assert initial_peak_over_mad(spectrum) <= 10
    with pytest.raises(ValueError):
        _findcontinuum(findContinuum.smooth, spectrum.copy(), 9, "flat")
    with pytest.raises(ValueError):
        smooth_spectrum(spectrum, 9, 400.0, initial_peak_over_mad(spectrum))


def _findcontinuum_computed_spectrum(
    mean, scaled_mad, bad_atmosphere, n_bin, max_baseline, after_fallback
):
    """fC 8255-8295 on the spectra of a pass: the normalization, the MAD-zero
    test (``None`` when it asks for the fallback), the quadratic and fC
    8283-8295."""
    intensity, normalized = mean.copy(), False
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if bad_atmosphere:
            locals_ = _NORMALIZATION(mean.copy(), scaled_mad.copy(), "", False)
            intensity, normalized = locals_["avgIntensity"], locals_["normalized"]
        if after_fallback:
            removed, ratio = False, 1.0
        elif findContinuum.MAD(intensity) == 0.0:
            return None
        else:
            intensity, removed, ratio = findContinuum.removeInitialQuadraticIfNeeded(
                intensity
            )
        locals_ = _PEAK_CAP_AND_SMOOTHING(intensity, n_bin, max_baseline, "flat")
    return (
        locals_["intensity"],
        normalized,
        removed,
        ratio,
        locals_["initialPeakOverMad"],
        locals_["nbin"],
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_computed_round_spectrum_equals_findcontinuum():
    rng = np.random.default_rng(15)
    for index in range(240):
        mean, scaled_mad = _spectra(rng, int(rng.choice([8, 40, 120, 480, 1920])))
        bad_atmosphere, after_fallback = index % 2 == 1, index % 3 == 0
        n_bin = int(rng.choice([1, 1, 2, 3, 5]))
        max_baseline = float(rng.choice([40.0, 400.0]))
        expected = _findcontinuum_computed_spectrum(
            mean, scaled_mad, bad_atmosphere, n_bin, max_baseline, after_fallback
        )
        result = computed_round_spectrum(
            mean,
            scaled_mad if bad_atmosphere else None,
            bad_atmosphere,
            n_bin,
            max_baseline,
            after_fallback,
        )
        assert result.needs_primary_beam_based_mask == (expected is None)
        if expected is None:
            continue
        assert result.mean_spectrum is mean
        assert result.normalized == expected[1] and result.n_bin == expected[5]
        assert result.initial_quadratic_removed == expected[2]
        assert _same(result.spectrum, expected[0])
        assert _same(result.quadratic_ratio, expected[3])
        assert _same(result.initial_peak_over_mad, expected[4])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_spectrum_of_zero_scaled_mad_asks_for_the_primary_beam_based_mask():
    """More than half of the channels empty in CASA (0.0) give a scaled MAD of
    0, and the round's spectrum comes from the primary-beam-based mask (fC
    8258-8277), whose spectrum is not tested again."""
    mean = np.r_[np.zeros(30), np.linspace(1.0, 2.0, 10)]
    assert findContinuum.MAD(mean) == 0.0
    result = computed_round_spectrum(mean, None, False, 3, 400.0)
    assert result.needs_primary_beam_based_mask and result.n_bin == 3
    expected = _findcontinuum_computed_spectrum(mean, None, False, 3, 400.0, True)
    fallback = computed_round_spectrum(mean, None, False, 3, 400.0, True)
    assert not fallback.needs_primary_beam_based_mask
    assert not fallback.initial_quadratic_removed and fallback.quadratic_ratio == 1.0
    assert _same(fallback.spectrum, expected[0]) and fallback.n_bin == expected[5]
    assert _same(fallback.initial_peak_over_mad, expected[4])


def test_robust_continuum_noise_equals_findcontinuum():
    rng = np.random.default_rng(16)
    python_rng = random.Random(16)
    cases = []
    for _ in range(300):
        n_channels = int(rng.choice([8, 40, 120, 480, 960, 1920]))
        ranges = (
            random_ranges(python_rng, n_channels)
            if rng.random() < 0.5
            else _increasing_ranges(rng, n_channels, int(rng.integers(1, 6)))
        )
        cases.append((random_spectrum(rng, n_channels), ranges))
    # Lines left in 1300 continuum channels: the lowest 19 % lower the noise.
    spectrum = 1e-3 * rng.standard_normal(1920) + 0.01 * np.arange(1920) / 1920
    spectrum[rng.choice(np.arange(100, 700), 40, replace=False)] += 0.02
    cases.append((spectrum, [[100, 700], [800, 1500]]))
    # A widely spread lowest 19 % does not.
    spectrum = 1e-5 * rng.standard_normal(1920)
    low = rng.choice(np.arange(100, 1500), 400, replace=False)
    spectrum[low] = -1e-3 * rng.uniform(1, 10, low.size)
    spectrum[50 + 1100] += 1.0
    cases.append((spectrum, [[100, 1500]]))
    cases.append((1e-3 * rng.standard_normal(480), [[3, 6], [20, 24]]))
    branches = set()
    for spectrum, ranges in cases:
        inputs = spectrum.copy()
        channels = findContinuum.convertSelectionIntoChannelList(_selection(ranges))
        (scaled_mad, median), log = _findcontinuum(
            findContinuum.robustMADofContinuumRanges, channels, spectrum.copy()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = robust_continuum_noise(spectrum, ranges)
        assert _same(result[0], median) and _same(result[1], scaled_mad)
        assert type(result[0]) is float and type(result[1]) is float
        assert _same(spectrum, inputs)
        for message, branch in (
            ("too few channels", "few"),
            ("slope=", "runs"),
            ("significant signal", "lowest"),
            ("the MAD is lower", "lowered"),
        ):
            if message in log:
                branches.add(branch)
        if "significant signal" in log and "the MAD is lower" not in log:
            branches.add("kept")
    assert branches == {"few", "runs", "lowest", "lowered", "kept"}


def test_line_full_ranges_equal_findcontinuum_and_raise_where_it_fails():
    rng = np.random.default_rng(17)
    python_rng = random.Random(17)
    cases = []
    for _ in range(200):
        n_channels = int(rng.choice([8, 40, 480, 3840]))
        cases.append((random_ranges(python_rng, n_channels), n_channels))
        cases.append(
            (_increasing_ranges(rng, n_channels, int(rng.integers(1, 6))), n_channels)
        )
    cases += [
        ([], 100),
        ([[0, 100]], 100),
        ([[10, 100]], 100),
        ([[0, 10]], 100),
        ([[0, 10], [20, 100]], 100),
        ([[5, 10], [10, 20]], 100),
        ([[5, 12], [10, 20]], 100),
        ([[30, 40], [5, 10]], 100),
        ([[30, 40], [0, 10]], 100),
    ]
    failures = 0
    for ranges, n_channels in cases:
        try:
            line_selection, _ = _findcontinuum(
                findContinuum.invertChannelRanges, _selection(ranges), n_channels
            )
            _findcontinuum(
                findContinuum.replaceLineFullRangesWithNoise,
                np.zeros(n_channels),
                line_selection,
                0.0,
                1.0,
            )
        except ValueError:
            failures += 1
            with pytest.raises(ValueError):
                line_full_ranges(ranges, n_channels)
            continue
        assert line_full_ranges(ranges, n_channels) == _ranges(line_selection)
    assert failures >= 7


def _strong_lines(rng, spectrum, line_ranges):
    """A copy of the spectrum with a line of random strength in most line-full
    ranges."""
    spectrum = spectrum.copy()
    scale = np.nanstd(spectrum) + 1e-6
    for start, stop in line_ranges:
        if rng.random() < 0.7:
            spectrum[int(rng.integers(start, stop))] += rng.uniform(0, 60) * scale
    return spectrum


def _replaced_ranges(log):
    """The ranges that findContinuum's noise replacement says it replaced."""
    return [
        [int(start), int(last) + 1]
        for start, last in re.findall(r"Replacing channel range (\d+):(\d+)", log)
    ]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_noise_replacement_equals_findcontinuum():
    rng = np.random.default_rng(18)
    replaced = 0
    for _ in range(200):
        n_channels = int(rng.choice([40, 120, 480, 1920]))
        ranges = _increasing_ranges(rng, n_channels, int(rng.integers(1, 6)))
        line_ranges = line_full_ranges(ranges, n_channels)
        spectrum = _strong_lines(rng, random_spectrum(rng, n_channels), line_ranges)
        inputs = spectrum.copy()
        median, scaled_mad = robust_continuum_noise(spectrum, ranges)
        expected, log = _findcontinuum(
            findContinuum.replaceLineFullRangesWithNoise,
            spectrum.copy(),
            _selection(line_ranges),
            median,
            scaled_mad,
        )
        result = replace_line_full_ranges_with_noise(
            spectrum, line_ranges, median, scaled_mad
        )
        assert _same(result, expected) and _same(spectrum, inputs)
        if _replaced_ranges(log) and scaled_mad > 0:
            replaced += 1
            assert not _same(
                replace_line_full_ranges_with_noise(
                    spectrum, line_ranges, median, scaled_mad, seed=7
                ),
                result,
            )
    assert replaced > 50


def test_quantized_spectrum_equals_findcontinuums_spectrum_file(tmp_path):
    """Values and end frequencies after findContinuum's spectrum file
    (writeMeanSpectrum, readPreviousMeanSpectrum)."""
    rng = np.random.default_rng(19)
    for index in range(12):
        n_channels = int(rng.choice([8, 75, 480, 3840]))
        spectrum = random_spectrum(rng, n_channels)
        spectrum[:5] = [np.nan, -1e-12, 5e-11, 1.00000000005, -0.0]
        width = float(rng.choice([-1.0, 1.0])) * rng.choice([7.8125e6, 976562.5, 488.3])
        first = rng.uniform(86e9, 650e9)
        last = first + (n_channels - 1) * width + rng.uniform(-1e-3, 1e-3)
        path = str(tmp_path / f"spectrum_{index}")
        _findcontinuum(
            findContinuum.writeMeanSpectrum,
            path,
            np.linspace(first, last, n_channels),
            spectrum,
            spectrum,
            0.0123,
            n_channels,
            0,
        )
        expected, _ = _findcontinuum(findContinuum.readPreviousMeanSpectrum, path)
        values, first_frequency, last_frequency = quantize_extra_mask_spectrum(
            spectrum, first, last
        )
        assert _same(values, expected[1]) and values.dtype == np.float64
        assert _same(first_frequency, expected[6]) and _same(
            last_frequency, expected[7]
        )


def _findcontinuum_extra_mask_spectrum(
    mean, scaled_mad, bad_atmosphere, ranges, first, last, path
):
    """fC 2041-2055 with the spectrum file read back as the next round reads
    it (fC 5198)."""
    intensity = mean.copy()
    if bad_atmosphere:
        with quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            intensity = _NORMALIZATION(mean.copy(), scaled_mad.copy(), "", False)[
                "avgIntensity"
            ]
    n_channels = intensity.size
    line_selection, _ = _findcontinuum(
        findContinuum.invertChannelRanges, _selection(ranges), n_channels
    )
    channels = findContinuum.convertSelectionIntoChannelList(_selection(ranges))
    (noise, median), _ = _findcontinuum(
        findContinuum.robustMADofContinuumRanges, channels, intensity
    )
    intensity, log = _findcontinuum(
        findContinuum.replaceLineFullRangesWithNoise,
        intensity,
        line_selection,
        median,
        noise,
    )
    frequency = np.linspace(first, last, n_channels)
    _findcontinuum(
        findContinuum.writeMeanSpectrum,
        path,
        frequency,
        intensity,
        intensity,
        0.0123,
        n_channels,
        0,
        0,
        "mom0mom8jointMask",
    )
    read, _ = _findcontinuum(findContinuum.readPreviousMeanSpectrum, path)
    return read[1], read[6], read[7], median, noise, _replaced_ranges(log)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_extra_mask_spectrum_equals_findcontinuum(tmp_path):
    rng = np.random.default_rng(20)
    replaced = 0
    for index in range(80):
        n_channels = int(rng.choice([40, 120, 480, 1920]))
        ranges = _increasing_ranges(rng, n_channels, int(rng.integers(1, 5)))
        mean, scaled_mad = _spectra(rng, n_channels)
        mean = _strong_lines(rng, mean, line_full_ranges(ranges, n_channels))
        bad_atmosphere = index % 2 == 1
        first = rng.uniform(86e9, 650e9)
        last = first + (n_channels - 1) * 976562.5
        expected = _findcontinuum_extra_mask_spectrum(
            mean,
            scaled_mad,
            bad_atmosphere,
            ranges,
            first,
            last,
            str(tmp_path / f"spectrum_{index}"),
        )
        result = extra_mask_spectrum(
            mean,
            scaled_mad if bad_atmosphere else None,
            bad_atmosphere,
            ranges,
            first,
            last,
        )
        assert _same(result.spectrum, expected[0])
        assert _same(result.first_frequency, expected[1])
        assert _same(result.last_frequency, expected[2])
        assert _same(result.median, expected[3])
        assert _same(result.scaled_mad, expected[4])
        assert result.replaced_ranges == expected[5]
        replaced += bool(result.replaced_ranges)
    assert replaced > 20
    with pytest.raises(ValueError):
        extra_mask_spectrum(mean, None, False, [[0, mean.size]], first, last)


def _image(values):
    """An ``(l, m)`` array as an image-shaped DataArray of one time, frequency
    and polarization."""
    n_l, n_m = values.shape
    return xr.DataArray(
        values[None, None, None],
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={
            "time": [0.0],
            "frequency": [230.0e9],
            "polarization": ["I"],
            "l": np.arange(n_l) * -1e-6,
            "m": np.arange(n_m) * 1e-6,
        },
    )


def _between(low, high):
    """A float32 value strictly between two float64 values."""
    value = np.float32((low + high) / 2)
    assert low < np.float64(value) < high
    return value


def _cube_and_maps(rng):
    """A float32 primary-beam cube ``(l, m, channel)`` with values between the
    "%g" and the exact annulus levels, NaN below the 0.2 limit as the port
    holds it, and the mean primary beam."""
    n_l, n_m, n_channels = 20, 23, 5
    l, m = np.meshgrid(np.arange(n_l) - 9.5, np.arange(n_m) - 11.0, indexing="ij")
    widths = 7.0 + 0.4 * np.arange(n_channels)
    cube = np.exp(-(l[..., None] ** 2 + m[..., None] ** 2) / (2 * widths**2))
    cube = cube.astype(np.float32).reshape(-1)
    boundary = [
        _between(0.702173, CUBE_LEVELS.lower),
        _between(0.87508, CUBE_LEVELS.higher),
    ]
    cube[rng.choice(cube.size, 120, replace=False)] = np.repeat(boundary, 60)
    cube = cube.reshape(n_l, n_m, n_channels).astype(np.float64)
    primary_beam = np.where(cube < 0.2, np.nan, cube)
    mean_primary_beam = cube.mean(axis=2)
    return primary_beam, np.where(mean_primary_beam < 0.2, np.nan, mean_primary_beam)


def _pixels(region, primary_beam, use_spectrum_mask, levels):
    """The pixels of every channel that a region and its per-channel
    primary-beam test select (the per-plane rule of the spectrum pass)."""
    selected = region.values[0, 0, 0][..., None] & np.ones(primary_beam.shape, bool)
    if use_spectrum_mask:
        selected &= primary_beam > 0.21
    if levels is not None:
        selected &= (primary_beam > levels.lower) & (primary_beam < levels.higher)
    return selected


def test_spectrum_regions_select_the_ports_pixels():
    """fC 8255 in the port: '"J">0 && "pb">0.21' for the mean and
    '"pb">%g && "pb"<%g && "J"<0.5' for the scaled MAD, the joint mask NaN
    outside its footprint."""
    rng = np.random.default_rng(21)
    primary_beam, mean_primary_beam = _cube_and_maps(rng)
    shape = mean_primary_beam.shape
    footprint = np.isfinite(mean_primary_beam) & (rng.random(shape) < 0.9)
    mask = rng.random(shape) < 0.4
    joint_mask = xr.Dataset(
        {"JOINT_MASK": _image(mask), "JOINT_MASK_FOOTPRINT": _image(footprint)},
        attrs={"kind": "phase_2", "primary_beam_based": False},
    )
    arrays = {
        "J": np.where(footprint, mask.astype(np.float64), np.nan)[..., None],
        "pb": primary_beam,
    }
    regions = spectrum_regions(joint_mask, CUBE_LEVELS, True)
    expected = evaluate_lattice_expression('"J">0 && "pb">0.21', arrays)
    assert regions.use_spectrum_mask and regions.mean_region.name == "MEAN_REGION"
    assert np.array_equal(
        _pixels(regions.mean_region, primary_beam, True, None), expected
    )
    expression = '"pb">%g && "pb"<%g && "J"<0.5'
    expected = evaluate_lattice_expression(expression % CUBE_LEVELS, arrays)
    levels = regions.scaled_mad_primary_beam_levels
    assert levels == AnnulusLevels(0.702173, 0.87508)
    selected = _pixels(regions.scaled_mad_region, primary_beam, False, levels)
    assert np.array_equal(selected, expected)
    assert not np.array_equal(
        _pixels(regions.scaled_mad_region, primary_beam, False, CUBE_LEVELS), expected
    )
    assert regions.mean_region.dims == ("time", "frequency", "polarization", "l", "m")
    assert regions.scaled_mad_region.dtype == bool
    assert spectrum_regions(joint_mask, CUBE_LEVELS, False)[2:] == (None, None)


def test_extra_mask_regions_select_the_ports_pixels():
    """fC 2041, 2441 and 2513 in the port, the moment difference NaN where it
    is invalid."""
    rng = np.random.default_rng(22)
    primary_beam, mean_primary_beam = _cube_and_maps(rng)
    shape = mean_primary_beam.shape
    moment_difference = EXTRA_LEVEL * rng.uniform(0.5, 1.5, shape)
    moment_difference.reshape(-1)[rng.choice(moment_difference.size, 40)] = 0.01009612
    invalid = ~(mean_primary_beam > 0.23) | (rng.random(shape) < 0.05)
    moment_difference[invalid] = np.nan
    assert np.any((moment_difference > 0.010096) != (moment_difference > EXTRA_LEVEL))
    amended = rng.random(shape) < 0.3
    arrays = {
        "momDiff": moment_difference[..., None],
        "pbmom": mean_primary_beam[..., None],
        "pb": primary_beam,
        "U": amended.astype(np.float64)[..., None],
    }
    region = _image(evaluate_lattice_expression('"pbmom">0.23', arrays)[..., 0])
    normalization = '"pbmom">%g && "pbmom"<%g'
    normalization_region = _image(
        evaluate_lattice_expression(normalization % MEAN_PRIMARY_BEAM_LEVELS, arrays)[
            ..., 0
        ]
    )
    amended_joint_mask = xr.Dataset(
        {
            "JOINT_MASK": _image(amended),
            "JOINT_MASK_FOOTPRINT": _image(np.ones(shape, bool)),
        },
        attrs={"kind": "amended", "primary_beam_based": False},
    )
    for form in ("only_extra_mask", "with_intersection", "without_intersection"):
        regions = extra_mask_regions(
            form,
            _image(moment_difference),
            EXTRA_LEVEL,
            region,
            normalization_region,
            CUBE_LEVELS,
            True,
            amended_joint_mask if form != "only_extra_mask" else None,
        )
        if form == "only_extra_mask":
            mean = '"momDiff">%f && "pbmom">0.23'
            scaled_mad = normalization % MEAN_PRIMARY_BEAM_LEVELS
        else:
            mean = '"momDiff">%f && "pb">0.21'
            scaled_mad = '"pb">%g && "pb"<%g && "U"<0.5' % CUBE_LEVELS  # noqa: UP031
        expected = evaluate_lattice_expression(mean % EXTRA_LEVEL, arrays)
        selected = _pixels(
            regions.mean_region, primary_beam, regions.use_spectrum_mask, None
        )
        assert np.array_equal(selected, expected & np.ones(selected.shape, bool))
        assert regions.use_spectrum_mask == (form != "only_extra_mask")
        expected = evaluate_lattice_expression(scaled_mad, arrays)
        selected = _pixels(
            regions.scaled_mad_region,
            primary_beam,
            False,
            regions.scaled_mad_primary_beam_levels,
        )
        assert np.array_equal(selected, expected & np.ones(selected.shape, bool))
        unnormalized = extra_mask_regions(
            form,
            _image(moment_difference),
            EXTRA_LEVEL,
            region,
            normalization_region,
            CUBE_LEVELS,
            False,
            None,
        )
        assert unnormalized[1:] == (regions.use_spectrum_mask, None, None)
        assert unnormalized.mean_region.equals(regions.mean_region)


def test_regions_refuse_other_forms_and_more_than_one_plane():
    values = np.zeros((4, 5), bool)
    joint_mask = xr.Dataset(
        {"JOINT_MASK": _image(values), "JOINT_MASK_FOOTPRINT": _image(~values)}
    )
    moment_difference = _image(np.zeros((4, 5)))
    arguments = (_image(values), _image(values), CUBE_LEVELS, False, joint_mask)
    with pytest.raises(ValueError, match="one time and one polarization"):
        spectrum_regions(joint_mask.isel(time=[0, 0]), CUBE_LEVELS, False)
    with pytest.raises(ValueError, match="one time and one polarization"):
        extra_mask_regions(
            "with_intersection",
            moment_difference.isel(polarization=[0, 0]),
            0.1,
            *arguments,
        )
    with pytest.raises(ValueError, match="form"):
        extra_mask_regions("extra_mask", moment_difference, 0.1, *arguments)
