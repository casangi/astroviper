"""The spectrum steps of find_continuum_ultra.

The regions of a round's mean and scaled-MAD spectra, the normalization of the
mean spectrum by the scaled-MAD spectrum, the initial quadratic, the smoothing,
and the spectrum of an extra mask with its line-full ranges replaced by noise,
each computed as findContinuum computes it on the pipeline path. Regions are
image-shaped boolean ``xarray.DataArray`` objects of one time and one
polarization, spectra one-dimensional float64 arrays, channel ranges
``[start, stop)`` pairs. ``fC`` line numbers are those of the pipeline's
``findContinuum.py`` (pipeline 2026.3.0.7).
"""

import itertools
import random

import numpy as np
import scipy.optimize
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    channels_of_ranges,
    contiguous_runs,
    line_fit,
    round_to_decimals,
    round_to_significant_digits,
    scaled_median_absolute_deviation,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
    ACA_MAX_BASELINE,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    AnnulusLevels,
    ComputedSpectrum,
    ExtraMaskSpectrum,
    SpectrumRegions,
)

# The extra-mask spectrum after round "original" (fC 2041), and after round
# "amended_mask" with (fC 2441) or without (fC 2513) the intersection of the
# selections.
EXTRA_MASK_FORMS = ("only_extra_mask", "with_intersection", "without_intersection")
NORMALIZATION_THRESHOLD = 3.2  # fC 8551
NORMALIZATION_FACTOR_FLOOR = 0.05  # fC 8525-8526
QUADRATIC_IMPROVEMENT_THRESHOLD = 1.6  # fC 4869, 7728
N_BIN_THRESHOLD = 2  # fC 359
N_BIN_CAP_PEAK_OVER_MAD = 10  # fC 8286
N_BIN_CAP = 2  # fC 8292
N_BIN_CAP_ACA = 3  # fC 8290
NOISE_REPLACEMENT_SNR = 7  # fC 3620
NOISE_REPLACEMENT_SEED = 2001  # fC 3615


def _check_one_plane(values: xr.DataArray | xr.Dataset) -> None:
    """Refuse maps and regions of more than one time or polarization."""
    if values.sizes["time"] != 1 or values.sizes["polarization"] != 1:
        raise ValueError(
            "find_continuum_ultra's maps and regions have one time and one "
            f"polarization, not {values.sizes['time']} and "
            f"{values.sizes['polarization']}"
        )


def _outside_joint_mask(joint_mask: xr.Dataset) -> xr.DataArray:
    """Pixels of the joint mask's footprint outside the mask (fC 8505); the
    port's comparison fails outside the footprint."""
    return (joint_mask["JOINT_MASK_FOOTPRINT"] & ~joint_mask["JOINT_MASK"]).rename(
        "SCALED_MAD_REGION"
    )


def _significant_levels(levels: AnnulusLevels) -> AnnulusLevels:
    """Annulus levels written with six significant digits (fC 8503-8505)."""
    return AnnulusLevels(
        round_to_significant_digits(levels.lower, 6),
        round_to_significant_digits(levels.higher, 6),
    )


def spectrum_regions(
    joint_mask: xr.Dataset, cube_annulus_levels: AnnulusLevels, bad_atmosphere: bool
) -> SpectrumRegions:
    """The regions of a computed round's mean and scaled-MAD spectra
    (fC 8442-8459, 8497-8508).

    The mean spectrum averages the valid sky pixels inside the joint mask
    where the channel's primary beam is above 0.21. With a bad atmosphere the
    scaled-MAD spectrum that normalizes it takes the valid sky pixels of the
    joint mask's footprint outside the mask where the channel's primary beam
    lies between the cube's annulus levels written with six significant
    digits. Pixels outside the footprint are in neither region.

    Parameters
    ----------
    joint_mask : xarray.Dataset
        Boolean ``JOINT_MASK`` and ``JOINT_MASK_FOOTPRINT`` of one time and
        one polarization.
    cube_annulus_levels : AnnulusLevels
        The annulus levels of the primary-beam cube.
    bad_atmosphere : bool
        Whether the mean spectrum is normalized by the scaled-MAD spectrum.

    Returns
    -------
    SpectrumRegions
        ``MEAN_REGION`` with the per-channel primary-beam test; with a bad
        atmosphere also ``SCALED_MAD_REGION`` and its primary-beam levels.

    Raises
    ------
    ValueError
        If the joint mask has more than one time or polarization.
    """
    _check_one_plane(joint_mask)
    mean_region = (
        joint_mask["JOINT_MASK"] & joint_mask["JOINT_MASK_FOOTPRINT"]
    ).rename("MEAN_REGION")
    if not bad_atmosphere:
        return SpectrumRegions(mean_region, True, None, None)
    return SpectrumRegions(
        mean_region,
        True,
        _outside_joint_mask(joint_mask),
        _significant_levels(cube_annulus_levels),
    )


def normalize_spectrum_by_mad(
    mean_spectrum: np.ndarray, scaled_mad_spectrum: np.ndarray
) -> tuple[np.ndarray, bool]:
    """The mean spectrum normalized by the scaled-MAD spectrum, if that
    spectrum's features are strong enough (fC 8515-8567).

    The mean spectrum, shifted to be non-negative, is divided by one plus the
    scaled-MAD spectrum's rise above its minimum, scaled to the mean
    spectrum's peak over median (and floored at ``NORMALIZATION_FACTOR_FLOOR``
    of its maximum), shifted back, and rescaled to the original median and
    scaled MAD.

    Parameters
    ----------
    mean_spectrum : numpy.ndarray
        float64 ``(n_channels,)``.
    scaled_mad_spectrum : numpy.ndarray
        float64 ``(n_channels,)``: the scaled median absolute deviation of
        every channel [findContinuum ``xmadm``].

    Returns
    -------
    spectrum : numpy.ndarray
        The normalized spectrum, or ``mean_spectrum``.
    normalized : bool
        Whether the peak of the scaled-MAD spectrum over the larger scaled MAD
        of the two spectra exceeds that of the mean spectrum divided by
        ``NORMALIZATION_THRESHOLD``.
    """
    original_mad = scaled_median_absolute_deviation(mean_spectrum)
    original_median = np.median(mean_spectrum)
    offset = np.max([0, -np.min(mean_spectrum)])
    shifted = mean_spectrum + offset
    peak_over_median = np.max(shifted) / np.median(shifted) - 1
    rise = scaled_mad_spectrum - np.min(scaled_mad_spectrum)
    factor = 1 + rise * peak_over_median / np.max(rise)
    lowest, highest = np.min(factor), np.max(factor)
    if lowest < NORMALIZATION_FACTOR_FLOOR * highest:
        factor = factor + (NORMALIZATION_FACTOR_FLOOR * highest - lowest)
    normalized = shifted / factor - offset
    normalized = (normalized - np.median(normalized)) * original_mad / (
        scaled_median_absolute_deviation(normalized)
    ) + original_median
    highest_mad = np.max(
        [original_mad, scaled_median_absolute_deviation(scaled_mad_spectrum)]
    )
    if (
        np.max(scaled_mad_spectrum) / highest_mad
        > np.max(mean_spectrum) / highest_mad / NORMALIZATION_THRESHOLD
    ):
        return normalized, True
    return mean_spectrum, False


def _quadratic_residuals(
    parameters: np.ndarray, x: np.ndarray, y: np.ndarray, error: np.ndarray
) -> np.ndarray:
    """findContinuum's weighted residuals of a quadratic about a free centre
    (fC 9411-9414)."""
    return (
        y
        - (
            parameters[2]
            + parameters[1] * (x - parameters[3])
            + parameters[0] * (x - parameters[3]) ** 2
        )
    ) / (error**2)


def remove_initial_quadratic(spectrum: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """The spectrum without its initial quadratic, if removing it divides the
    scaled MAD by more than ``QUADRATIC_IMPROVEMENT_THRESHOLD``
    [findContinuum ``removeInitialQuadraticIfNeeded``] (fC 4601-4629).

    ``scipy.optimize.leastsq`` fits findContinuum's four parameters
    (curvature, slope, value and centre, from ``[0, 0, mean, n_channels //
    2]``) with weights ``(spectrum / scaled MAD)**2`` (fC 9393-9418); the
    centre is redundant, so only this fit gives findContinuum's ratio.

    Parameters
    ----------
    spectrum : numpy.ndarray
        float64 ``(n_channels,)``.

    Returns
    -------
    spectrum : numpy.ndarray
        The spectrum minus the quadratic plus its NaN-skipping mean, or the
        input.
    removed : bool
    quadratic_ratio : float
        The scaled MAD before over the one after; 1.0 if the one after is 0.
    """
    n_channels = spectrum.size
    prior_mad = scaled_median_absolute_deviation(spectrum)
    channels = np.arange(n_channels)
    curvature, slope, value, centre = scipy.optimize.leastsq(
        _quadratic_residuals,
        [0, 0, np.mean(spectrum), channels[int(n_channels / 2)]],
        args=(channels, spectrum, np.ones(n_channels) * prior_mad / spectrum),
        full_output=1,
    )[0]
    offsets = channels - centre
    trial = (
        spectrum
        + np.nanmean(spectrum)
        - (offsets**2 * curvature + offsets * slope + value)
    )
    post_mad = scaled_median_absolute_deviation(trial)
    if post_mad == 0.0:
        return spectrum, False, 1.0
    ratio = prior_mad / post_mad
    if ratio > QUADRATIC_IMPROVEMENT_THRESHOLD:
        return trial, True, ratio
    return spectrum, False, ratio


def initial_peak_over_mad(spectrum: np.ndarray) -> float:
    """The peak of a spectrum above its median over its scaled MAD, NaN values
    skipped (fC 8283).

    Parameters
    ----------
    spectrum : numpy.ndarray
        float64 ``(n_channels,)``.

    Returns
    -------
    float
        ``(nanmax - nanmedian) / scaled MAD``.
    """
    return float(
        (np.nanmax(spectrum) - np.nanmedian(spectrum))
        / scaled_median_absolute_deviation(spectrum)
    )


def smooth_spectrum(
    spectrum: np.ndarray, n_bin: int, max_baseline: float, initial_peak_over_mad: float
) -> tuple[np.ndarray, int]:
    """The spectrum averaged over ``n_bin`` channels, ``n_bin`` capped when the
    spectrum has strong lines (fC 8284-8295).

    From ``N_BIN_THRESHOLD`` up, an initial peak over MAD above
    ``N_BIN_CAP_PEAK_OVER_MAD`` caps ``n_bin`` at ``N_BIN_CAP_ACA`` for a
    longest baseline up to ``ACA_MAX_BASELINE``, else at ``N_BIN_CAP``. A
    window of three or more channels averages the spectrum reflected at its
    ends without the end channels; the window of output channel ``k`` is
    centred on ``k - 1`` (odd windows) or ``k - 1.5`` (even windows)
    [findContinuum ``smooth``] (fC 3254-3317).

    Parameters
    ----------
    spectrum : numpy.ndarray
        float64 ``(n_channels,)``.
    n_bin : int
        Channels per average: the settings' ``n_bin``, or the capped one of
        the previous computed round.
    max_baseline : float
        Longest baseline (m).
    initial_peak_over_mad : float
        From :func:`initial_peak_over_mad`.

    Returns
    -------
    spectrum : numpy.ndarray
    n_bin : int
        ``n_bin`` after the cap.

    Raises
    ------
    ValueError
        If ``n_bin`` is at least ``N_BIN_THRESHOLD`` and larger than the
        number of channels (fC 3292-3293).
    """
    if n_bin < N_BIN_THRESHOLD:
        return spectrum, n_bin
    if initial_peak_over_mad > N_BIN_CAP_PEAK_OVER_MAD:
        n_bin = N_BIN_CAP_ACA if max_baseline <= ACA_MAX_BASELINE else N_BIN_CAP
    if spectrum.size < n_bin:
        raise ValueError(
            f"n_bin {n_bin} exceeds the {spectrum.size} channels of the spectrum"
        )
    if n_bin < 3:  # fC 3295-3298
        return spectrum, n_bin
    reflected = np.r_[spectrum[n_bin:0:-1], spectrum, spectrum[-2:-n_bin:-1]]
    window = np.ones(n_bin)
    smoothed = np.convolve(window / window.sum(), reflected, mode="same")
    return smoothed[n_bin - 1 : 1 - n_bin], n_bin


def computed_round_spectrum(
    mean_spectrum: np.ndarray,
    scaled_mad_spectrum: np.ndarray | None,
    bad_atmosphere: bool,
    n_bin: int,
    max_baseline: float,
    after_mad_zero_fallback: bool = False,
) -> ComputedSpectrum:
    """The spectrum of a computed round from its mean and scaled-MAD spectra
    (fC 8255-8300).

    The normalization with a bad atmosphere; then, unless after the MAD-zero
    fallback, the test for a scaled MAD of 0, which asks for the
    primary-beam-based joint mask (fC 8258), and the initial quadratic; the
    initial peak over MAD; the cap of ``n_bin`` and the smoothing. After the
    fallback the quadratic is not tested (ratio 1.0, fC 8276-8277).

    Parameters
    ----------
    mean_spectrum : numpy.ndarray
        float64 ``(n_channels,)``, NaN for a channel without a pixel.
    scaled_mad_spectrum : numpy.ndarray or None
        float64 ``(n_channels,)``; read only with a bad atmosphere.
    bad_atmosphere : bool
        Whether the mean spectrum is normalized by the scaled-MAD spectrum.
    n_bin : int
        Channels per average: the settings' ``n_bin``, or the capped one of
        the previous computed round.
    max_baseline : float
        Longest baseline (m).
    after_mad_zero_fallback : bool, default False
        The spectra are those of the primary-beam-based joint mask that a
        scaled MAD of 0 asked for.

    Returns
    -------
    ComputedSpectrum
        With ``needs_primary_beam_based_mask`` the spectrum is the
        (normalized) mean spectrum, the quadratic is not tested, the peak over
        MAD is NaN and ``n_bin`` is not capped.

    Raises
    ------
    ValueError
        From :func:`smooth_spectrum`.
    """
    spectrum, normalized = (
        normalize_spectrum_by_mad(mean_spectrum, scaled_mad_spectrum)
        if bad_atmosphere
        else (mean_spectrum, False)
    )
    if after_mad_zero_fallback:
        removed, ratio = False, 1.0
    elif scaled_median_absolute_deviation(spectrum) == 0.0:
        return ComputedSpectrum(
            spectrum, mean_spectrum, normalized, False, 1.0, np.nan, n_bin, True
        )
    else:
        spectrum, removed, ratio = remove_initial_quadratic(spectrum)
    peak_over_mad = initial_peak_over_mad(spectrum)
    spectrum, n_bin = smooth_spectrum(spectrum, n_bin, max_baseline, peak_over_mad)
    return ComputedSpectrum(
        spectrum, mean_spectrum, normalized, removed, ratio, peak_over_mad, n_bin, False
    )


def extra_mask_regions(
    form: str,
    moment_difference: xr.DataArray,
    extra_level: float,
    moment_difference_region: xr.DataArray,
    only_extra_mask_normalization_region: xr.DataArray,
    cube_annulus_levels: AnnulusLevels,
    bad_atmosphere: bool,
    amended_joint_mask: xr.Dataset | None,
) -> SpectrumRegions:
    """The regions of an extra-mask spectrum (fC 2041, 2441, 2513; 8498-8508).

    The mean spectrum averages the valid sky pixels where the valid moment
    difference is above the extra level written with six decimals: with
    ``"only_extra_mask"`` inside the moment-difference region and with no
    per-channel primary-beam test, in the two other forms where the channel's
    primary beam is above 0.21. With a bad atmosphere the scaled-MAD spectrum
    takes, with ``"only_extra_mask"``, the pixels of the mean primary beam's
    annulus, with no joint-mask term (fC 8505); in the two other forms the
    pixels of the amended joint mask's footprint outside the mask where the
    channel's primary beam lies between the cube's annulus levels written with
    six significant digits.

    Parameters
    ----------
    form : str
        ``"only_extra_mask"`` (after round ``original``),
        ``"with_intersection"`` or ``"without_intersection"`` (after round
        ``amended_mask``).
    moment_difference : xarray.DataArray
        float64, NaN where invalid, of one time and one polarization: of the
        deciding round's selection, or of the intersection.
    extra_level : float
        The level of the extra-mask decision.
    moment_difference_region : xarray.DataArray
        bool: the valid mean primary beam above 0.23.
    only_extra_mask_normalization_region : xarray.DataArray
        bool: the valid mean primary beam between its own annulus levels
        written with six significant digits.
    cube_annulus_levels : AnnulusLevels
        The annulus levels of the primary-beam cube.
    bad_atmosphere : bool
        Whether the mean spectrum is normalized by the scaled-MAD spectrum.
    amended_joint_mask : xarray.Dataset or None
        The joint mask of round ``amended_mask`` (primary-beam based after a
        MAD-zero fallback in that round, fC 8260-8271), read by the two forms
        after it with a bad atmosphere.

    Returns
    -------
    SpectrumRegions

    Raises
    ------
    ValueError
        If ``form`` is not one of ``EXTRA_MASK_FORMS``, or the moment
        difference has more than one time or polarization.
    """
    if form not in EXTRA_MASK_FORMS:
        raise ValueError(f"form is one of {EXTRA_MASK_FORMS}, not {form!r}")
    _check_one_plane(moment_difference)
    above = moment_difference > round_to_decimals(extra_level, 6)
    if form == "only_extra_mask":
        return SpectrumRegions(
            (above & moment_difference_region).rename("MEAN_REGION"),
            False,
            (
                only_extra_mask_normalization_region.rename("SCALED_MAD_REGION")
                if bad_atmosphere
                else None
            ),
            None,
        )
    if not bad_atmosphere:
        return SpectrumRegions(above.rename("MEAN_REGION"), True, None, None)
    return SpectrumRegions(
        above.rename("MEAN_REGION"),
        True,
        _outside_joint_mask(amended_joint_mask),
        _significant_levels(cube_annulus_levels),
    )


def line_full_ranges(continuum_ranges: list, n_channels: int) -> list[list[int]]:
    """The channel ranges between and around continuum ranges
    [findContinuum ``invertChannelRanges``] (fC 6271-6314).

    Parameters
    ----------
    continuum_ranges : list
        ``[start, stop)`` pairs.
    n_channels : int
        Channels of the spectrum.

    Returns
    -------
    list of list of int
        ``[0, start)`` of the first range if it starts after channel 0, the
        gaps between consecutive ranges, and ``[stop, n_channels)`` of the
        last range if it ends before the last channel, in this order.

    Raises
    ------
    ValueError
        Where findContinuum fails: no continuum range (fC 6295), a single
        range that ends at the last channel (no line-full range, or one
        followed by an empty one, fC 3617), and consecutive ranges that do not
        increase with a channel between them (an empty line-full range,
        fC 3619).
    """
    if not continuum_ranges:
        raise ValueError("No continuum range to take line-full ranges from")
    if len(continuum_ranges) == 1 and continuum_ranges[0][1] >= n_channels:
        raise ValueError(
            "A single continuum range that ends at the last channel leaves no "
            "line-full range to read"
        )
    gaps = [
        [stop, start] for (_, stop), (start, _) in itertools.pairwise(continuum_ranges)
    ]
    if any(start >= stop for start, stop in gaps):
        raise ValueError(
            "Continuum ranges must increase with a channel between consecutive "
            f"ranges: {continuum_ranges}"
        )
    first, last = continuum_ranges[0][0], continuum_ranges[-1][1]
    return (
        ([[0, first]] if first > 0 else [])
        + gaps
        + ([[last, n_channels]] if last < n_channels else [])
    )


def robust_continuum_noise(
    spectrum: np.ndarray, continuum_ranges: list
) -> tuple[float, float]:
    """The median and scaled MAD of a spectrum's continuum channels, robust to
    a slope and to lines left in them [findContinuum
    ``robustMADofContinuumRanges``] (fC 3319-3385).

    With ten or more continuum channels: their median; with several runs of
    contiguous channels, a line fitted to the runs' medians subtracted run by
    run; their scaled MAD; then, if the lowest 19 % are more than 35 channels
    and the peak is more than 4 scaled MADs above the median, each averaged
    with that of the lowest 19 % when their scaled MAD is lower. As in
    findContinuum the values tested are shifted by the median also with a
    single run, where no line was subtracted. With fewer continuum channels,
    the scaled MAD of the whole spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        float64 ``(n_channels,)``.
    continuum_ranges : list
        ``[start, stop)`` pairs.

    Returns
    -------
    median : float
    scaled_mad : float
    """
    values = np.array(spectrum, dtype=np.float64)
    channels = channels_of_ranges(continuum_ranges)
    continuum = values[channels]
    median = np.median(continuum)
    if channels.size < 10:  # fC 3330
        return float(median), scaled_median_absolute_deviation(values)
    runs = contiguous_runs(channels)
    if len(runs) > 1:
        run_medians = np.array([np.median(values[run]) for run in runs])
        slope, intercept = line_fit(
            np.arange(len(runs)), run_medians, run_medians * 0.01
        )
        for index, run in enumerate(runs):
            values[run] -= intercept + slope * index
        continuum = values[channels]
    scaled_mad = scaled_median_absolute_deviation(continuum)
    continuum += median
    snr = (np.max(continuum) - median) / scaled_mad
    n_lowest = int(np.ceil(0.19 * channels.size))  # fC 3363-3364
    if n_lowest > 35 and snr > 4:  # fC 3367
        lowest = np.sort(continuum)[:n_lowest]
        lowest_mad = scaled_median_absolute_deviation(lowest)
        if lowest_mad < scaled_mad:
            scaled_mad = np.mean([scaled_mad, lowest_mad])
            median = np.mean([median, np.median(lowest)])
    return float(median), float(scaled_mad)


def _polar_normal(generator: random.Random) -> float:
    """One Gaussian deviate by Marsaglia's polar method, written as
    findContinuum writes it [findContinuum ``pickRandomError``]
    (fC 4726-4743)."""
    w = 1.0
    while w >= 1.0:
        x1 = 2.0 * generator.random() - 1.0
        x2 = 2.0 * generator.random() - 1.0
        w = x1 * x1 + x2 * x2
    w = np.sqrt((-2.0 * np.log(w)) / w)
    return x1 * w


def _ranges_above_noise(
    spectrum: np.ndarray, line_ranges: list, median: float, scaled_mad: float
) -> list[list[int]]:
    """The line-full ranges that peak more than ``NOISE_REPLACEMENT_SNR``
    scaled MADs above the median (fC 3619-3620)."""
    return [
        [start, stop]
        for start, stop in line_ranges
        if (np.max(spectrum[start:stop]) - median) / scaled_mad > NOISE_REPLACEMENT_SNR
    ]


def replace_line_full_ranges_with_noise(
    spectrum: np.ndarray,
    line_ranges: list,
    median: float,
    scaled_mad: float,
    seed: int = NOISE_REPLACEMENT_SEED,
) -> np.ndarray:
    """The spectrum with every line-full range that peaks more than
    ``NOISE_REPLACEMENT_SNR`` scaled MADs above the median replaced by
    Gaussian noise of that median and scaled MAD [findContinuum
    ``replaceLineFullRangesWithNoise``] (fC 3605-3628).

    The noise is drawn from ``random.Random(seed)`` by Marsaglia's polar
    method as findContinuum draws it (fC 4726-4743), channel by channel in
    range order; a range not replaced draws nothing.

    Parameters
    ----------
    spectrum : numpy.ndarray
        float64 ``(n_channels,)``.
    line_ranges : list
        ``[start, stop)`` pairs, from :func:`line_full_ranges`.
    median : float
        Median of the noise.
    scaled_mad : float
        Scaled MAD of the noise.
    seed : int, default NOISE_REPLACEMENT_SEED
        Seed of the generator.

    Returns
    -------
    numpy.ndarray
        The new float64 spectrum.

    Raises
    ------
    ValueError
        If a line-full range is empty.
    """
    generator = random.Random(seed)
    replaced = np.array(spectrum, dtype=np.float64)
    for start, stop in _ranges_above_noise(spectrum, line_ranges, median, scaled_mad):
        noise = np.array([_polar_normal(generator) for _ in range(start, stop)])
        replaced[start:stop] = median + noise * scaled_mad
    return replaced


def quantize_extra_mask_spectrum(
    spectrum: np.ndarray, first_frequency: float, last_frequency: float
) -> tuple[np.ndarray, float, float]:
    """An extra-mask spectrum and its end frequencies as findContinuum writes
    them to its spectrum file and reads them back in the next round: every
    value with ten decimals, the frequencies with one (fC 6511-6514,
    11794-11841).

    Parameters
    ----------
    spectrum : numpy.ndarray
        float64 ``(n_channels,)``.
    first_frequency, last_frequency : float
        Frequencies of the first and last channel (Hz).

    Returns
    -------
    spectrum : numpy.ndarray
        float64 ``(n_channels,)``.
    first_frequency, last_frequency : float
    """
    return (
        np.array(
            [round_to_decimals(value, 10) for value in spectrum], dtype=np.float64
        ),
        round_to_decimals(first_frequency, 1),
        round_to_decimals(last_frequency, 1),
    )


def extra_mask_spectrum(
    mean_spectrum: np.ndarray,
    scaled_mad_spectrum: np.ndarray | None,
    bad_atmosphere: bool,
    continuum_ranges: list,
    first_frequency: float,
    last_frequency: float,
) -> ExtraMaskSpectrum:
    """The spectrum of the rounds after an extra mask (fC 2040-2055,
    2439-2458, 2510-2530).

    The mean spectrum, normalized with a bad atmosphere; the robust noise of
    its continuum channels; its line-full ranges replaced by noise; the values
    and end frequencies as findContinuum's spectrum file holds them. No
    MAD-zero test, quadratic or smoothing.

    Parameters
    ----------
    mean_spectrum : numpy.ndarray
        float64 ``(n_channels,)`` of the extra-mask regions.
    scaled_mad_spectrum : numpy.ndarray or None
        float64 ``(n_channels,)``; read only with a bad atmosphere.
    bad_atmosphere : bool
        Whether the mean spectrum is normalized by the scaled-MAD spectrum.
    continuum_ranges : list
        ``[start, stop)`` pairs of the round that decided the extra mask, or
        of the intersection.
    first_frequency, last_frequency : float
        Frequencies of the window's first and last channel (Hz).

    Returns
    -------
    ExtraMaskSpectrum

    Raises
    ------
    ValueError
        From :func:`line_full_ranges`.
    """
    spectrum = (
        normalize_spectrum_by_mad(mean_spectrum, scaled_mad_spectrum)[0]
        if bad_atmosphere
        else mean_spectrum
    )
    line_ranges = line_full_ranges(continuum_ranges, spectrum.size)
    median, scaled_mad = robust_continuum_noise(spectrum, continuum_ranges)
    quantized, first, last = quantize_extra_mask_spectrum(
        replace_line_full_ranges_with_noise(spectrum, line_ranges, median, scaled_mad),
        first_frequency,
        last_frequency,
    )
    return ExtraMaskSpectrum(
        quantized,
        first,
        last,
        median,
        scaled_mad,
        _ranges_above_noise(spectrum, line_ranges, median, scaled_mad),
    )
