"""findContinuum's selection of continuum channels on a spectrum.

The baseline, thresholds and candidate continuum channels of one call, and the
selection of a round: its threshold adjustments and second call, the pruning
of the ranges by the baseline channels and the range added to a single narrow
range. Every value is computed as findContinuum computes it, including the
order of the arithmetic, the ties of its sorts and the order of its ranges.
Channel ranges are ``[start, stop)`` pairs relative to the spectral window, in
findContinuum's list order. ``fC`` line numbers are those of the pipeline's
``findContinuum.py`` (pipeline 2026.3.0.7).
"""

import numpy as np

from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    contiguous_runs,
    line_fit,
    range_count,
    ranges_of_channels,
    scaled_median_absolute_deviation,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
    is_time_division_mode,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    ChannelSelection,
    ChannelThreshold,
)

MAX_TRIM = 20  # fC 360
MAX_TRIM_FRACTION = 1.0  # fC 456
NEGATIVE_THRESHOLD_FACTOR = 1.15  # fC 468
DROPPED_BASELINE_PERCENT = 2.0  # fC 470
MAD_RATIO_LOWER_LIMIT = 1.15  # fC 471
MAD_RATIO_UPPER_LIMIT = 1.5  # fC 471
MIN_PEAK_OVER_MAD_FOR_THRESHOLD_ADJUSTMENT = 25  # fC 475
MAX_MAD_RATIO_FOR_THRESHOLD_ADJUSTMENT = 1.18  # fC 489-490
PEAK_OVER_MAD_CRITERION = 65  # fC 492
MIN_RANGES_FOR_THRESHOLD_ADJUSTMENT = 7  # fC 493
MAX_RANGES = 15  # fC 493
MOMENT_DIFFERENCE_APPROACH_LEVEL = 15.0  # fC 493
THRESHOLD_FACTOR_WITH_MAXIMUM_MASK = 0.8  # fC 352
ALL_CONTINUUM_FRACTION = 0.925  # fC 354
ALL_CONTINUUM_FRACTION_FEW_CHANNELS = 0.91  # below 75 channels, fC 355
SIGNAL_RATIO_TIER_2 = 0.94  # fC 4874
MAX_RANGES_FOR_MAX_TRIM_RESTORE = 3  # fC 6521
HIGH_BASELINE_MAD_FACTOR = 1.5  # fC 6522
LINE_SNR_THRESHOLD = 20  # fC 6523
SPEED_OF_LIGHT = 2.99792458e8  # m/s, fC 5314
# The rounds of an extra-mask spectrum (fC 6693, 6756).
EXTRA_MASK_SPECTRUM_ROUNDS = (
    "only_extra_mask",
    "extra_mask",
    "auto_lower",
    "auto_lower_2",
)


def select_continuum_channels(
    spectrum: np.ndarray,
    n_baseline_channels: int,
    channel_width: float,
    first_frequency: float,
    last_frequency: float,
    threshold_sigma: float | str,
    threshold_sigma_mode: str,
    signal_ratio_tier_1: float,
    round_kind: str,
    moment_difference_snr: float,
    n_pixels_maximum_mask: int,
    integrated_peak: float,
    maximum_peak: float,
    single_continuum: bool,
    nonzero_mad_positions: np.ndarray | None,
    rounds_enabled: bool,
) -> ChannelSelection:
    """The continuum channel ranges of a round's spectrum [findContinuum
    ``runFindContinuum``] (fC 5227-5862).

    One or two channel thresholds (:func:`continuum_channel_threshold`) with
    the threshold sigma adjusted between them, the ranges pruned to the
    baseline channels they hold, and a second range added to a single range
    of less than 5 % of the channels.

    Parameters
    ----------
    spectrum : numpy.ndarray
        The round's spectrum, float64 ``(n_channels,)`` without NaN.
    n_baseline_channels : int
        Channels of the baseline [findContinuum ``nBaselineChannels``].
    channel_width : float
        Channel width of the cube in Hz, also in a round of an extra-mask
        spectrum.
    first_frequency, last_frequency : float
        Frequencies of the first and last channel in Hz: in a round of an
        extra-mask spectrum the quantized ones.
    threshold_sigma : float or str
        ``"auto"``, or the multiplier of the baseline's scaled median absolute
        deviation [findContinuum ``sigmaFindContinuum``].
    threshold_sigma_mode : str
        ``"auto"`` or ``"auto_lower"`` [findContinuum
        ``sigmaFindContinuumMode``].
    signal_ratio_tier_1 : float
        Signal ratio above which the threshold is desensitized
        [findContinuum ``signalRatioTier1``].
    round_kind : str
        ``"original"``, ``"single"``, ``"amended_mask"``,
        ``"only_extra_mask"``, ``"extra_mask"``, ``"auto_lower"`` or
        ``"auto_lower_2"``.
    moment_difference_snr : float
        Signal-to-noise ratio of the last moment difference, -1 before the
        first.
    n_pixels_maximum_mask : int
        Pixels of the first round's maximum mask, 0 in a round of an
        extra-mask spectrum.
    integrated_peak, maximum_peak : float
        Peaks of the first round's integrated and maximum maps, read only
        with at least 4 pixels in the maximum mask.
    single_continuum : bool
        The spectral window is a single-continuum setup: the threshold sigma
        of a time-division-mode spectrum is then 9.0 instead of 7.2.
    nonzero_mad_positions : numpy.ndarray or None
        Positions of the candidate baseline values that are not zero, as an
        earlier computed round returned them.
    rounds_enabled : bool
        The rounds after the first are enabled [findContinuum ``amendMask``].

    Returns
    -------
    ChannelSelection
        The ranges, the ranges before the pruning, the final threshold sigma,
        the values of the last channel threshold and the spectrum's peak over
        its scaled median absolute deviation. ``figure_values`` holds
        ``peak_minus_median_over_mad``, ``signal_ratio``, ``mad_ratio``,
        ``use_middle_baseline``, ``correction_factor``, ``trim``,
        ``max_trim``, ``narrow`` and ``added_range`` (None, or the range added
        to a single narrow range).

    Raises
    ------
    ValueError
        Where findContinuum fails or finds no result: a channel threshold
        fails, no channel lies below a corrected median (fC 6472), a gap
        between the baseline channels of a time-division-mode spectrum holds
        no channel (fC 5619), no range is left without rounds (fC 5683) or no
        range is left at all (fC 5942).
    """
    n_channels = spectrum.size
    narrow = int(np.ceil(np.log10(n_channels)))
    time_division_threshold_sigma = 9.0 if single_continuum else 7.2  # fC 4661-4668
    spectrum_mad = scaled_median_absolute_deviation(spectrum)
    peak_over_mad = np.max(spectrum) / spectrum_mad
    peak_minus_median_over_mad = (
        np.max(spectrum[1:-1]) - np.median(spectrum)
    ) / spectrum_mad
    time_division_mode = is_time_division_mode(channel_width, n_channels)
    if threshold_sigma == "auto":  # fC 5244-5273
        if time_division_mode:
            threshold_sigma = time_division_threshold_sigma
        elif n_channels < 750:
            threshold_sigma = 4.2 if peak_over_mad > 6 else 4.5
        else:
            threshold_sigma = 2.5 if peak_over_mad > 6 else 3.2

    def channel_threshold(threshold_sigma, trim, max_trim, nonzero_mad_positions):
        threshold = continuum_channel_threshold(
            spectrum,
            n_baseline_channels,
            threshold_sigma,
            trim,
            max_trim,
            narrow,
            peak_over_mad,
            signal_ratio_tier_1,
            threshold_sigma_mode,
            round_kind,
            nonzero_mad_positions,
        )
        if not np.any(spectrum < threshold.corrected_median):
            raise ValueError(
                "No channel of the spectrum lies below the corrected median "
                f"{threshold.corrected_median}."
            )
        return threshold

    threshold = channel_threshold(
        threshold_sigma, "auto", MAX_TRIM, nonzero_mad_positions
    )
    n_ranges = threshold.group_count
    spectral_window_bandwidth = n_channels * channel_width
    spectral_window_bandwidth_kms = (
        0.001
        * SPEED_OF_LIGHT
        * spectral_window_bandwidth
        / np.mean([first_frequency, last_frequency])
    )
    max_trim_scaled = (
        n_ranges <= 2 and not time_division_mode and threshold.trim == "auto"
    )  # fC 5321-5326
    max_trim = MAX_TRIM * n_channels / 128 if max_trim_scaled else MAX_TRIM
    adjusted = False
    if (
        threshold.single_channel_peaks_above_threshold
        == threshold.groups_above_threshold
        and threshold.groups_above_threshold > 1
    ):  # fC 5329-5361
        if (
            threshold_sigma < time_division_threshold_sigma
            and threshold_sigma_mode == "auto"
        ):
            threshold_sigma *= 1.5
            threshold = channel_threshold(
                threshold_sigma,
                threshold.trim,
                max_trim,
                threshold.nonzero_mad_positions,
            )
    else:  # fC 5430-5533
        if not (
            round_kind == "auto_lower"
            and moment_difference_snr < MOMENT_DIFFERENCE_APPROACH_LEVEL
        ):
            if threshold.mad_ratio is None:
                mad_ratio_check = peak_over_mad > 40
            else:
                mad_ratio_check = (
                    threshold.mad_ratio < MAX_MAD_RATIO_FOR_THRESHOLD_ADJUSTMENT
                    or peak_over_mad > PEAK_OVER_MAD_CRITERION
                )
            if (
                (
                    (
                        n_ranges >= MIN_RANGES_FOR_THRESHOLD_ADJUSTMENT
                        and not time_division_mode
                    )
                    or (n_ranges >= 3 and time_division_mode)
                )
                and threshold_sigma_mode in ("auto", "auto_lower")
                and peak_over_mad > MIN_PEAK_OVER_MAD_FOR_THRESHOLD_ADJUSTMENT
                and mad_ratio_check
            ):
                scaling = 5
                if round_kind in ("auto_lower", "auto_lower_2") or (
                    round_kind in ("only_extra_mask", "extra_mask") and n_ranges > 32
                ):
                    scaling = 6
                if n_ranges < MAX_RANGES:
                    factor = np.max(
                        [
                            scaling / 7.0,
                            (1 - n_ranges / float(MAX_RANGES))
                            ** (spectral_window_bandwidth / 1.875e9),
                        ]
                    )
                else:
                    factor = scaling / 7.0
                threshold_sigma *= factor
                adjusted = True
            elif n_pixels_maximum_mask >= 4 and all_continuum_selected(
                threshold.continuum_ranges, n_channels
            ):
                normalized_peak = integrated_peak / spectral_window_bandwidth_kms
                if (
                    maximum_peak
                    > (1.5 if time_division_mode else 1.4) * normalized_peak
                ):
                    threshold_sigma *= THRESHOLD_FACTOR_WITH_MAXIMUM_MASK
                    adjusted = (
                        maximum_peak > 3.5 * normalized_peak and peak_over_mad > 5.5
                    )
            elif n_ranges == 3 and time_division_mode:
                adjusted = True
        if (
            max_trim_scaled
            or (
                all_continuum_selected(threshold.continuum_ranges, n_channels)
                and n_pixels_maximum_mask > 0
            )
            or (
                n_ranges > MIN_RANGES_FOR_THRESHOLD_ADJUSTMENT
                and not time_division_mode
            )
        ):
            threshold = channel_threshold(
                threshold_sigma,
                threshold.trim,
                max_trim,
                threshold.nonzero_mad_positions,
            )

    ranges, n_ranges = threshold.continuum_ranges, threshold.group_count
    n_selected = sum(stop - start for start, stop in ranges)
    if (
        adjusted
        or (
            (peak_over_mad > 6 or n_selected > 4 * n_channels / 5)
            and not time_division_mode
        )
    ) and n_selected > 0:  # fC 5539-5540
        pruned = _ranges_pruned_to_baseline_channels(
            spectrum,
            ranges,
            threshold.baseline_channels,
            n_ranges,
            narrow,
            peak_over_mad,
            first_frequency,
            spectral_window_bandwidth,
            time_division_mode,
        )
        if pruned:
            ranges, n_ranges = pruned, len(pruned)

    added_range = None
    if n_ranges == 1:  # fC 5679-5712
        if ranges:
            [[start, stop]] = ranges
            n_selected, range_center = stop - start, np.mean([start, stop - 1])
        elif rounds_enabled:
            n_selected, range_center = 0, -1.0
        else:
            raise ValueError("The spectrum has no continuum channel range.")
        if n_selected / n_channels < 5.0 / 100:
            run = widest_run_in_other_half(
                threshold.baseline_channels, range_center < n_channels / 2, n_channels
            )
            if run is not None:
                added_range = [int(run[0]), int(run[-1]) + 1]
                if ranges and run[0] <= ranges[0][1] - 1:
                    ranges = [added_range, *ranges]
                else:
                    ranges = [*ranges, added_range]
    if not ranges:
        raise ValueError(
            "The spectrum has no continuum channel range and no baseline channels "
            "in the other half of the spectral window."
        )
    return ChannelSelection(
        continuum_ranges=ranges,
        ranges_before_blue_pruning=threshold.continuum_ranges,
        final_threshold_sigma=threshold_sigma,
        positive_threshold=threshold.positive_threshold,
        negative_threshold=threshold.negative_threshold,
        corrected_median=threshold.corrected_median,
        median=threshold.median,
        mad=threshold.mad,
        baseline_channels=threshold.baseline_channels,
        baseline_values=threshold.baseline_values,
        nonzero_mad_positions=threshold.nonzero_mad_positions,
        peak_over_mad=peak_over_mad,
        figure_values={
            "peak_minus_median_over_mad": peak_minus_median_over_mad,
            "signal_ratio": threshold.signal_ratio,
            "mad_ratio": threshold.mad_ratio,
            "use_middle_baseline": threshold.use_middle_baseline,
            "correction_factor": threshold.correction_factor,
            "trim": threshold.trim,
            "max_trim": max_trim,
            "narrow": narrow,
            "added_range": added_range,
        },
    )


def continuum_channel_threshold(
    spectrum: np.ndarray,
    n_baseline_channels: int,
    threshold_sigma: float,
    trim: int | str,
    max_trim: float,
    narrow: int,
    peak_over_mad: float,
    signal_ratio_tier_1: float,
    threshold_sigma_mode: str,
    round_kind: str,
    nonzero_mad_positions: np.ndarray | None,
) -> ChannelThreshold:
    """The baseline, the thresholds and the candidate continuum channels of a
    spectrum [findContinuum ``findContinuumChannels``] (fC 6518-7079).

    The baseline is the lowest, highest or middle values of the spectrum
    without its constant ends; its median, corrected for the selection of the
    values, and its scaled median absolute deviation give the thresholds. The
    runs of channels between them lose their channels at the spectrum's
    minimum next to the ends, runs without variation, their trimmed ends, runs
    narrower than ``narrow`` and narrow inner runs.

    Parameters
    ----------
    spectrum : numpy.ndarray
        The spectrum, float64 ``(n_channels,)`` without NaN.
    n_baseline_channels : int
        Channels of the baseline [findContinuum ``nBaselineChannels``].
    threshold_sigma : float
        Multiplier of the baseline's scaled median absolute deviation
        [findContinuum ``sigmaFindContinuum``].
    trim : int or str
        ``"auto"``, or the channels trimmed from both ends of a run (a
        fraction of the run when below 1).
    max_trim : float
        The most channels the automatic trim takes from an end.
    narrow : int
        Fewest channels of a run among several.
    peak_over_mad : float
        The spectrum's peak over its scaled median absolute deviation.
    signal_ratio_tier_1 : float
        Signal ratio above which the threshold is desensitized
        [findContinuum ``signalRatioTier1``].
    threshold_sigma_mode : str
        ``"auto"`` or ``"auto_lower"`` [findContinuum
        ``sigmaFindContinuumMode``].
    round_kind : str
        The round; in the rounds of an extra-mask spectrum the baseline is the
        middle values.
    nonzero_mad_positions : numpy.ndarray or None
        Positions of the candidate baseline values that may be baseline
        values; None for all.

    Returns
    -------
    ChannelThreshold
        ``continuum_ranges`` and their ``group_count`` (1 for no range), the
        thresholds, the baseline's median, corrected median and scaled median
        absolute deviation, the trim for a next call and the baseline's
        channels and values. ``nonzero_mad_positions`` is the input, or the
        positions of the non-zero candidate values when the lowest values have
        no deviation.

    Raises
    ------
    ValueError
        If the signal ratio is 0 (fC 6874), or if no channel lies strictly
        between the thresholds or only channels at the spectrum's minimum next
        to an end do (fC 6941-6973).
    """
    n_channels = spectrum.size
    extra_mask_spectrum = round_kind in EXTRA_MASK_SPECTRUM_ROUNDS
    percentile = 100.0 * n_baseline_channels / n_channels
    correction_factor = (
        (n_channels / 128.0) ** 0.08 * 2.8 * (percentile / 10.0) ** -0.25
    )
    sigma_effective = threshold_sigma * correction_factor

    candidate_channels = np.arange(n_channels)
    candidate_values = spectrum
    if n_channels > 10:  # fC 6650-6672
        edge_runs = contiguous_runs(
            np.flatnonzero((spectrum == spectrum[0]) | (spectrum == spectrum[-1]))
        )
        inner = np.arange(edge_runs[0][-1] + 1, edge_runs[-1][0])
        if inner.size:
            candidate_channels, candidate_values = inner, spectrum[inner]
    order = np.argsort(candidate_values)
    if nonzero_mad_positions is not None:
        order = order[np.isin(order, nonzero_mad_positions)]
    low, low_by_position, high, middle = _baseline_values(
        spectrum, candidate_values, order, n_baseline_channels
    )
    mad_low, mad_high, mad_middle = (
        scaled_median_absolute_deviation(values) for values in (low, high, middle)
    )
    baseline = "low"
    if not extra_mask_spectrum:  # fC 6693-6724
        which = np.argmin(
            [
                mad_low,
                HIGH_BASELINE_MAD_FACTOR * mad_high,
                HIGH_BASELINE_MAD_FACTOR * mad_middle,
            ]
        )
        if which == 0 and mad_low <= 0.0:
            nonzero_mad_positions = np.flatnonzero(candidate_values != 0.0)
            order = order[np.isin(order, nonzero_mad_positions)]
            low, low_by_position, high, middle = _baseline_values(
                spectrum, candidate_values, order, n_baseline_channels
            )
            mad_low, mad_high, mad_middle = (
                scaled_median_absolute_deviation(values)
                for values in (low, high, middle)
            )
            which = np.argmin(
                [
                    mad_low,
                    HIGH_BASELINE_MAD_FACTOR * mad_high,
                    HIGH_BASELINE_MAD_FACTOR * mad_middle,
                ]
            )
        highest = order[-n_baseline_channels:]
        no_low_inside_high = (
            np.intersect1d(
                np.arange(np.min(highest), np.max(highest)), order[:n_baseline_channels]
            ).size
            == 0
        )
        if which == 0 or no_low_inside_high:
            baseline = "low"
        elif which == 1:
            baseline = "high"
        else:
            baseline = "middle" if mad_middle > 0 else "low"
    middle_channels = candidate_channels[order][
        n_baseline_channels:-n_baseline_channels
    ]
    if baseline == "high":
        values = high[::-1]
        baseline_channels = candidate_channels[order][-n_baseline_channels:][::-1]
        baseline_mad = mad_high
    elif baseline == "middle":
        values, baseline_channels, baseline_mad = middle, middle_channels, mad_middle
    else:
        values = low
        baseline_channels = candidate_channels[order][:n_baseline_channels]
        baseline_mad = mad_low
    use_low_baseline, use_middle_baseline = baseline == "low", baseline == "middle"
    if extra_mask_spectrum and not np.isnan(mad_middle):  # fC 6756-6768
        values, baseline_channels, baseline_mad = middle, middle_channels, mad_middle
        use_low_baseline, use_middle_baseline = False, True

    mad_ratio = None
    n_dropped = int(int(len(order) * DROPPED_BASELINE_PERCENT) / 100)
    if not use_middle_baseline and n_dropped > 0:  # fC 6770-6794
        dropped = order[n_dropped : n_baseline_channels + n_dropped]
        dropped_mad = scaled_median_absolute_deviation(candidate_values[dropped])
        if not dropped_mad <= 0:  # also when NaN
            mad_ratio = baseline_mad / dropped_mad
            if MAD_RATIO_LOWER_LIMIT < mad_ratio < MAD_RATIO_UPPER_LIMIT:
                values = candidate_values[dropped]
                baseline_channels = candidate_channels[dropped]
                low_by_position = spectrum[baseline_channels]
    mad = scaled_median_absolute_deviation(values)
    mad_original = scaled_median_absolute_deviation(low_by_position)
    if mad < 1e-17 or mad_original < 1e-17:  # fC 6804-6821
        if n_channels > 10:
            reduced = spectrum[(spectrum != spectrum[0]) & (spectrum != spectrum[-1])]
            reduced = reduced[reduced != 0.0]
        else:
            reduced = spectrum[spectrum != values[0]]
        reduced_order = np.argsort(reduced)[:n_baseline_channels]
        # Positions in the reduced spectrum, not channels (fC 6816).
        values, baseline_channels = reduced[reduced_order], reduced_order
        mad = scaled_median_absolute_deviation(values)
    median = np.nanmedian(values)

    deviation = np.abs(spectrum - median)
    signal_ratio = (
        1.0 - 1.0 * np.count_nonzero(deviation > sigma_effective * mad * 2) / n_channels
    ) ** 2
    if signal_ratio >= 0.99 or (signal_ratio > 0.98 and 15 > peak_over_mad > 10):
        signal_ratio = (
            1.0
            - 1.0
            * np.count_nonzero(deviation > sigma_effective * mad * 1.35)
            / n_channels
        ) ** 2
    spectrum_median = np.median(spectrum)
    spectral_difference = 100 * np.median(np.abs(np.diff(spectrum))) / spectrum_median
    spectral_difference_2 = (
        100 * np.median(np.abs(np.diff(spectrum, n=2))) / spectrum_median
    )
    if signal_ratio == 0:
        raise ValueError(
            "Every channel of the spectrum deviates from the baseline median "
            f"{median} by more than twice the threshold."
        )
    line_strength_factor = 1.0 / signal_ratio
    if (
        spectral_difference_2 < 0.65
        and n_channels > 192
        and signal_ratio < SIGNAL_RATIO_TIER_2
    ):  # fC 6875-6904
        signal_ratio = 0.0
        x = np.arange(n_channels)
        slope, _ = line_fit(x, spectrum, scaled_median_absolute_deviation(spectrum))
        flattened = spectrum - x * slope
        line_snr = (
            np.max(flattened[1:-1]) - np.median(flattened)
        ) / scaled_median_absolute_deviation(flattened)
        if line_snr > LINE_SNR_THRESHOLD:
            mad *= 0.33
            if trim == "auto":
                trim = 6
    if use_middle_baseline:
        corrected_median = median
    else:
        median_correction = 6.3 * (5.0 / percentile) ** 0.5
        if use_low_baseline:
            corrected_median = median + median_correction * mad * (
                1.0 / line_strength_factor
            )
        else:
            corrected_median = median - median_correction * mad * (
                1.0 / line_strength_factor
            )
        if corrected_median < np.min(spectrum):
            corrected_median = np.median(spectrum)
    peak_feature_sigma = (np.max(spectrum) - corrected_median) / mad
    if (
        signal_ratio > signal_ratio_tier_1
        or (signal_ratio > SIGNAL_RATIO_TIER_2 and peak_over_mad < 5)
    ) and threshold_sigma_mode in ("auto", "auto_lower"):  # fC 6923-6930
        max_peak_feature_sigma = (
            np.max(spectrum) - np.min(spectrum)
        ) / scaled_median_absolute_deviation(spectrum)
        if peak_feature_sigma > max_peak_feature_sigma:
            mad *= peak_feature_sigma / max_peak_feature_sigma
            peak_feature_sigma = max_peak_feature_sigma
    positive_threshold = sigma_effective * mad + corrected_median
    negative_threshold = (
        -NEGATIVE_THRESHOLD_FACTOR * sigma_effective * mad + corrected_median
    )

    candidates = np.flatnonzero(
        (spectrum < positive_threshold) & (spectrum > negative_threshold)
    ).tolist()
    if not candidates:
        raise ValueError(
            f"No channel of the spectrum lies between the thresholds {negative_threshold} "
            f"and {positive_threshold}."
        )
    minimum = np.min(spectrum)
    tolerance = abs(1e-10 * minimum)
    if abs(spectrum[candidates[0]] - minimum) < tolerance:  # fC 6946-6959
        del candidates[0]
        _check_candidates_left(candidates, minimum)
        for channel in range(candidates[0], candidates[-1]):
            if abs(spectrum[channel] - minimum) > tolerance:
                break
            if channel in candidates:
                candidates.remove(channel)
    if abs(spectrum[candidates[-1]] - minimum) < tolerance:  # fC 6961-6974
        del candidates[-1]
        _check_candidates_left(candidates, minimum)
        for channel in range(candidates[-1], candidates[0] - 1, -1):
            if abs(spectrum[channel] - minimum) > tolerance:
                break
            if channel in candidates:
                candidates.remove(channel)

    peak_runs = contiguous_runs(np.flatnonzero(spectrum > positive_threshold))
    if (
        0 < signal_ratio < 0.925
        and spectral_difference_2 < 1.3
        and n_channels > 1000
        and trim == "auto"
        and max((run.size for run in peak_runs), default=0) < n_channels / 8
    ):  # fC 6981-6990
        trim = 13
    groups_above_threshold = len(peak_runs)
    single_channel_peaks_above_threshold = groups_above_threshold - sum(
        run.size for run in peak_runs if run.size >= 2
    )

    channels = np.array(candidates, dtype=np.int64)
    if channels.size:  # fC 7003-7058
        channels = _joined(
            run for run in contiguous_runs(channels) if np.std(spectrum[run]) > 1e-17
        )
    if channels.size:
        untrimmed = channels
        channels = trim_runs(n_channels, channels, trim, max_trim)
        if not channels.size:
            max_trim, trim = 0.1, 2
            channels = trim_runs(n_channels, untrimmed, trim, max_trim)
            if not channels.size:
                channels = untrimmed
        n_ranges = len(contiguous_runs(channels))
        if (
            n_ranges > MAX_RANGES_FOR_MAX_TRIM_RESTORE
            and trim == "auto"
            and max_trim > MAX_TRIM
        ):
            max_trim = MAX_TRIM
            trimmed = trim_runs(n_channels, channels, trim, max_trim)
            if trimmed.size:
                channels = trimmed
                n_ranges = len(contiguous_runs(channels))
        if n_ranges > 1:
            wide = _joined(
                run for run in contiguous_runs(channels) if run.size >= narrow
            )
            if wide.size:
                channels = wide
    channels, ranges_dropped = reject_narrow_inner_runs(channels)
    continuum_ranges = ranges_of_channels(channels)
    return ChannelThreshold(
        continuum_ranges=continuum_ranges,
        group_count=range_count(continuum_ranges),
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        median=median,
        corrected_median=corrected_median,
        mad=mad,
        correction_factor=correction_factor,
        signal_ratio=signal_ratio,
        line_strength_factor=line_strength_factor,
        groups_above_threshold=groups_above_threshold,
        single_channel_peaks_above_threshold=single_channel_peaks_above_threshold,
        spectral_difference=spectral_difference,
        spectral_difference_2=spectral_difference_2,
        trim=trim,
        use_low_baseline=use_low_baseline,
        narrow=narrow,
        baseline_channels=baseline_channels,
        baseline_values=values,
        mad_ratio=mad_ratio,
        use_middle_baseline=use_middle_baseline,
        ranges_dropped=ranges_dropped,
        nonzero_mad_positions=nonzero_mad_positions,
    )


def trim_runs(
    n_channels: int, channels: np.ndarray, trim: int | str, max_trim: float
) -> np.ndarray:
    """The channels of every run with its ends trimmed [findContinuum
    ``splitListIntoContiguousListsAndTrim``, ``pickAutoTrimChannels``]
    (fC 7146-7208, 7240-7255).

    The automatic trim takes ``max_trim`` channels from each end of a run when
    a tenth of the channels exceeds it, the median run is longer than 1.5
    times it and the spectral window has more than 256 channels; otherwise a
    tenth of the run, rounded up, at most the whole run. The outer ends of the
    first and last runs lose at most 2 channels (3 from 75 channels on). A run
    too short for its trim is dropped, except that a single run keeps its
    first channel.

    Parameters
    ----------
    n_channels : int
        Channels of the spectral window.
    channels : numpy.ndarray
        Increasing integer channels.
    trim : int or str
        ``"auto"``, or the channels trimmed from both ends of a run (a
        fraction of the run when below 1); 0 or less keeps every channel.
    max_trim : float
        The most channels the automatic trim takes from an end.

    Returns
    -------
    numpy.ndarray
        The channels kept.
    """
    if not isinstance(trim, str) and trim <= 0:
        return np.asarray(channels)
    runs = contiguous_runs(channels)
    amount = trim
    if trim == "auto":
        amount = 0.1
        if (
            len(channels) * amount > max_trim
            and np.median([run.size for run in runs]) > max_trim * 1.5
            and n_channels > 256
        ):
            amount = max_trim
    edge_limit = 2 if n_channels < 75 else 3
    kept = []
    for index, run in enumerate(runs):
        n_trim = int(np.ceil(run.size * amount)) if amount < 1 else amount
        if trim == "auto" and 1.0 * n_trim / run.size > MAX_TRIM_FRACTION:
            n_trim = int(np.floor(MAX_TRIM_FRACTION * run.size))
        if run.size < 1 + n_trim * 2:
            if len(runs) == 1:
                kept.append(run[:1])
        elif index == 0 and n_trim > edge_limit:
            kept.append(
                run[edge_limit:-edge_limit]
                if len(runs) == 1
                else run[edge_limit:-n_trim]
            )
        elif index == len(runs) - 1 and n_trim > edge_limit:
            kept.append(run[n_trim:-edge_limit])
        else:
            kept.append(run[n_trim:-n_trim])
    return _joined(kept)


def reject_narrow_inner_runs(channels: np.ndarray) -> tuple[np.ndarray, int]:
    """The channels without the inner runs narrower than both edge runs
    [findContinuum ``rejectNarrowInnerWindowsChannels``] (fC 7062-7066,
    7081-7120).

    Only 3 to 15 runs are examined; from 8 runs on an inner run is kept when
    it holds a fifth of the narrower edge run.

    Parameters
    ----------
    channels : numpy.ndarray
        Increasing integer channels.

    Returns
    -------
    channels : numpy.ndarray
        The channels kept.
    n_dropped : int
        Runs dropped.
    """
    runs = contiguous_runs(channels)
    if not 2 < len(runs) < 16:
        return channels, 0
    width = np.min([runs[0].size, runs[-1].size]) * (1 if len(runs) < 8 else 0.2)
    kept = [runs[0], *(run for run in runs[1:] if run.size >= width)]
    return _joined(kept), len(runs) - len(kept)


def widest_run_in_other_half(
    channels: np.ndarray, lower_half: bool, n_channels: int
) -> np.ndarray | None:
    """The longest run of consecutive channels that starts in the other half
    of the spectral window [findContinuum
    ``findWidestContiguousListInOtherHalf``] (fC 6361-6382).

    Parameters
    ----------
    channels : numpy.ndarray
        Integer channels, in any order.
    lower_half : bool
        The range to complement lies in the lower half: the run must start at
        or above ``n_channels / 2``, otherwise below it.
    n_channels : int
        Channels of the spectral window.

    Returns
    -------
    numpy.ndarray or None
        The run, the first of equally long ones; None if no run starts in the
        other half.
    """
    widest = None
    for run in contiguous_runs(np.sort(channels)):
        in_other_half = (
            run[0] >= n_channels / 2 if lower_half else run[0] < n_channels / 2
        )
        if in_other_half and run.size > (0 if widest is None else widest.size):
            widest = run
    return widest


def all_continuum_selected(ranges: list, n_channels: int) -> bool:
    """Whether a single range covers at least 92.5 % of the channels (91 %
    below 75 channels) [findContinuum ``allContinuumSelected``]
    (fC 3984-4015).

    Parameters
    ----------
    ranges : list
        ``[start, stop)`` pairs.
    n_channels : int
        Channels of the spectral window.

    Returns
    -------
    bool
        False for no range or several.
    """
    fraction = (
        ALL_CONTINUUM_FRACTION_FEW_CHANNELS
        if n_channels < 75
        else ALL_CONTINUUM_FRACTION
    )
    return (
        len(ranges) == 1
        and (ranges[0][1] - ranges[0][0]) * 1.0 / n_channels >= fraction
    )


def _baseline_values(spectrum, candidate_values, order, n_baseline_channels):
    """The lowest candidate values, the spectrum at their positions, and the
    highest and middle candidate values (fC 6683-6691)."""
    lowest = order[:n_baseline_channels]
    return (
        candidate_values[lowest],
        spectrum[lowest],
        candidate_values[order[-n_baseline_channels:]],
        candidate_values[order[n_baseline_channels:-n_baseline_channels]],
    )


def _check_candidates_left(candidates, minimum):
    if not candidates:
        raise ValueError(
            "The only channel between the thresholds holds the minimum "
            f"{minimum} of the spectrum."
        )


def _joined(runs):
    runs = list(runs)
    return np.concatenate(runs) if runs else np.array([], dtype=np.int64)


def _ranges_pruned_to_baseline_channels(
    spectrum,
    ranges,
    baseline_channels,
    n_ranges,
    narrow,
    peak_over_mad,
    first_frequency,
    spectral_window_bandwidth,
    time_division_mode,
):
    """Each range cut to the span of its baseline channels and split at wide
    gaps between them, dropping too narrow pieces (fC 5541-5666)."""
    n_channels = spectrum.size
    gap_max = int(n_channels / 4) if time_division_mode else int(n_channels / 3)
    pruned = []
    for start, stop in ranges:
        if n_ranges == 1:  # fC 5567-5591
            gap_min = int(
                30
                * (first_frequency / 300e9)
                * (spectral_window_bandwidth / 58e6)
                * (n_channels / 480.0)
            )
            if peak_over_mad <= 6:
                if gap_min < 30:
                    gap_min = 30
                elif gap_min > 45:
                    gap_min = 45
            else:
                gap_min = np.max([16, gap_min])
            if gap_min > gap_max:
                gap_min = 30
                if gap_min > gap_max:
                    gap_min = gap_max - 10
        else:
            gap_min = int(np.max([(stop - start) / 2, narrow]))
        range_channels = np.arange(start, stop)
        blue = range_channels[np.isin(range_channels, baseline_channels)].tolist()
        # The step from the first to the second baseline channel is never a gap,
        # and one baseline channel leaves last_blue at -1 (fC 5598-5616).
        first_blue = last_blue = -1
        gaps = []
        for channel in blue:
            if first_blue == -1:
                first_blue = channel
                continue
            if last_blue > -1 and gap_min < channel - last_blue < gap_max:
                gaps.append([last_blue + 1, channel - 1])
            last_blue = channel
        if len(gaps) == 1 and time_division_mode:  # fC 5617-5621
            gap_values = np.abs(spectrum[gaps[0][0] : gaps[0][1]])
            if not gap_values.size:
                raise ValueError(
                    f"The gap at channels {gaps[0][0]} to {gaps[0][1]} between the "
                    "baseline channels of a time-division-mode spectrum holds no "
                    "channel to compare with the spectrum's peak."
                )
            if np.max(np.abs(spectrum)) > np.max(gap_values):
                gaps = []
        if not gaps:
            if blue and last_blue - first_blue >= narrow - 1:
                pruned.append([first_blue, last_blue + 1])
            continue
        piece_start = first_blue
        for gap_first, gap_last in gaps:
            if gap_first - piece_start >= narrow:
                pruned.append([piece_start, gap_first])
            piece_start = gap_last + 1
        if last_blue - gaps[-1][1] >= narrow:
            pruned.append([gaps[-1][1] + 1, last_blue + 1])
    return pruned
