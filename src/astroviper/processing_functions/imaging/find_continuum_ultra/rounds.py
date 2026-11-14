"""findContinuum's rounds: the decisions that end a round with a result or
choose the next round, the channel ranges whose maps a round needs, the change
and path codes, and the steps after the rounds (the reversion and the
low-spread fix).

Every function takes the values it reads and returns the values it decides;
the round transitions return everything the next round needs as a
``RoundTransition``. Decisions are ``"not_evaluated"``, ``"no"``,
``"yes_moment"``, ``"yes_cube"`` and ``"no_improvement"`` [findContinuum
``False``, ``'No'``, ``'YesMom'``, ``'YesCube'``, ``'NoImprovement'``]. Channel
ranges are ``[start, stop)`` pairs relative to the spectral window, in
findContinuum's list order. Maps and regions are image-shaped (``time,
frequency, polarization, l, m``, frequency length 1) with one time and one
polarization. ``fC`` line numbers are those of the pipeline's
``findContinuum.py`` (pipeline 2026.3.0.7).
"""

import numpy as np
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum_ultra.first_round_maps import (
    count_in_primary_beam,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    channels_of_ranges,
    contiguous_runs,
    ranges_of_channels,
    round_to_decimals,
    scaled_median_absolute_deviation,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.products import (
    SMALL_SPREAD_FRACTION,
    selection_bandwidth,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.selection_maps import (
    CUBE_LEVEL,
    MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE,
    MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE_ONLY_EXTRA_MASK,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    BandwidthWarnings,
    CubeNoiseStatistics,
    Decision,
    ExtraMaskRequest,
    FinalStep,
    MapsRanges,
    MomentDifferenceCounts,
    RoundTransition,
    SelectionStatistics,
)

COMPUTED_ROUND_KINDS = ("original", "single", "amended_mask")
EXTRA_ROUND_KINDS = ("only_extra_mask", "extra_mask", "auto_lower", "auto_lower_2")
YES_DECISIONS = ("yes_moment", "yes_cube")

PROCEEDING_LEVEL = 8.0  # fC 492
APPROACH_LEVEL = 15.0  # fC 493
DIFFERENCE_LEVEL = 8.5  # fC 1373, times sqrt(2) (2222)
DIFFERENCE_BAD_ATMOSPHERE_LEVEL = 12  # fC 1374, times sqrt(2) (2220)
CUBE_SIGMA_THRESHOLD = 7.5  # fC 487
MINIMUM_PIXELS_TO_AMEND = 9  # fC 3833, 3844, 3849
MINIMUM_PIXELS_FOR_ONLY_EXTRA_MASK = 7  # fC 488
TOO_MANY_PIXELS = 850  # fC 498
PIXEL_RATIO_LIMIT = 2.2  # fC 3811
CUBE_PIXEL_RATIO_LIMIT = 2.0  # fC 3857
NEGATIVE_FRACTION_LIMIT = 0.1  # fC 3816, 3822
BAD_ATMOSPHERE_SIGMA = 8.0  # fC 3835, 3885, 3949
MOMENT_SIGMA_FLOOR = 5.5  # fC 3846, 3899, 3962
CUBE_SIGMA_FLOOR = 5.0  # fC 3859
DISABLE_CUBE_NOISE = 1e9  # fC 222
PIXEL_RATIO_EXCEEDED = -1.0  # fC 357
PIXELS_ABOVE_THRESHOLD_EXCEEDED = -2.0  # fC 358
SIGNAL_RATIO_TIER_1 = 0.967  # fC 476
SIGNAL_RATIO_TIER_1_EXTRA_MASK = 1.0  # fC 2061, 2468, 2536, 2701, 2731
AMENDED_GROWTH_LIMIT = 2  # fC 2115
CHANGE_THRESHOLD = 0.01  # fC 484
REVERSION_CODES = ("HHHH", "HHHS")  # fC 2836
WIDENING_MINIMUM_CHANNELS = 500  # fC 2603


def amend_mask_decision(
    bad_atmosphere: bool,
    moment_difference_median: float,
    moment_difference_snr_outside: float,
    moment_difference_snr_cube: float,
    n_above: int,
    n_above_2: int,
    n_above_bad_atmosphere: int,
    n_above_cube: int,
    n_above_cube_2: int,
    moment_difference_scaled_mad: float,
    ten_event_sigma: float,
    cube_scaled_mad_outside: float,
    negative_fraction: float,
) -> Decision:
    """Whether round ``original`` amends the joint mask [findContinuum
    ``amendMaskYesOrNo``] (fC 3790-3862, called at 1952-1956).

    Parameters
    ----------
    bad_atmosphere : bool
        Whether the atmosphere varies across the window.
    moment_difference_median, moment_difference_scaled_mad : float
        Median and scaled MAD of the moment difference.
    moment_difference_snr_outside : float
        Its signal-to-noise ratio with the peak outside the joint mask.
    moment_difference_snr_cube : float
        Its peak outside the joint mask above its median, in cube scaled MADs.
    n_above, n_above_2 : int
        Its pixels outside the joint mask above the moment-difference level and
        0.5 above it; the ratio of the two decides also with a bad atmosphere.
    n_above_bad_atmosphere : int
        Its pixels outside the joint mask above level 11.5.
    n_above_cube, n_above_cube_2 : int
        Its pixels outside the joint mask above 7.5 and 7.25 cube scaled MADs.
    ten_event_sigma : float
        The sigma of ten events among the valid voxels.
    cube_scaled_mad_outside : float
        The cube's scaled MAD outside the joint mask.
    negative_fraction : float
        The fraction of negative pixels of the maximum map outside the joint
        mask.

    Returns
    -------
    Decision
        ``"yes_moment"`` or ``"yes_cube"`` with the level that amends the mask,
        or ``"no"`` with level 0.0; ``sigma_used`` -1.0 for a pixel ratio
        above 2.2, -2.0 for too many pixels with many negative ones, 0.0 when
        no threshold is met.
    """
    if n_above_2 > 0 and float(n_above) / n_above_2 > PIXEL_RATIO_LIMIT:
        return Decision("no", 0.0, PIXEL_RATIO_EXCEEDED)
    n_above_atmosphere = n_above_bad_atmosphere if bad_atmosphere else n_above
    if (
        negative_fraction > NEGATIVE_FRACTION_LIMIT
        and n_above_atmosphere > TOO_MANY_PIXELS
    ):
        return Decision("no", 0.0, PIXELS_ABOVE_THRESHOLD_EXCEEDED)
    if bad_atmosphere:
        if (
            moment_difference_snr_outside >= MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE
            and n_above_bad_atmosphere >= MINIMUM_PIXELS_TO_AMEND
        ):
            return _yes(
                "yes_moment",
                moment_difference_median,
                BAD_ATMOSPHERE_SIGMA,
                moment_difference_scaled_mad,
            )
    elif (
        moment_difference_snr_outside >= PROCEEDING_LEVEL
        and n_above >= MINIMUM_PIXELS_TO_AMEND
    ):
        return _yes(
            "yes_moment",
            moment_difference_median,
            np.max([MOMENT_SIGMA_FLOOR, ten_event_sigma]),
            moment_difference_scaled_mad,
        )
    elif (
        moment_difference_snr_cube >= CUBE_LEVEL
        and n_above_cube >= MINIMUM_PIXELS_TO_AMEND
        and float(n_above_cube_2) / n_above_cube < CUBE_PIXEL_RATIO_LIMIT
    ):
        return _yes(
            "yes_cube",
            moment_difference_median,
            np.max([CUBE_SIGMA_FLOOR, ten_event_sigma]),
            cube_scaled_mad_outside,
        )
    return Decision("no", 0.0, 0.0)


def only_extra_mask_decision(
    bad_atmosphere: bool,
    moment_difference_median: float,
    moment_difference_snr: float,
    moment_difference_scaled_mad: float,
    n_above_anywhere: int,
    n_above_anywhere_bad_atmosphere: int,
    cube_peak: float,
    first_channel_median_outside: float,
    cube_scaled_mad_outside: float,
    ten_event_sigma: float,
) -> Decision:
    """Whether round ``original`` is followed by round ``only_extra_mask``
    [findContinuum ``onlyExtraMaskYesOrNo``] (fC 3917-3982, called at
    1979-1984).

    Parameters
    ----------
    bad_atmosphere : bool
        Whether the atmosphere varies across the window.
    moment_difference_median, moment_difference_scaled_mad : float
        Median and scaled MAD of the moment difference.
    moment_difference_snr : float
        Its signal-to-noise ratio with the peak anywhere.
    n_above_anywhere, n_above_anywhere_bad_atmosphere : int
        Its pixels anywhere above the moment-difference level and above level
        11.0.
    cube_peak : float
        The maximum of the selection's maximum map.
    first_channel_median_outside : float
        The median outside the joint mask of the lowest selected channel.
    cube_scaled_mad_outside : float
        The cube's scaled MAD outside the joint mask.
    ten_event_sigma : float
        The sigma of ten events among the valid voxels.

    Returns
    -------
    Decision
        ``"yes_moment"`` with the level of the extra mask, or ``"no"`` with
        level 0.0.
    """
    cube_contrast = cube_peak - first_channel_median_outside
    if bad_atmosphere:
        if (
            moment_difference_snr
            >= MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE_ONLY_EXTRA_MASK
            and n_above_anywhere_bad_atmosphere >= MINIMUM_PIXELS_FOR_ONLY_EXTRA_MASK
            and (
                cube_contrast > CUBE_SIGMA_THRESHOLD * cube_scaled_mad_outside
                or cube_scaled_mad_outside == DISABLE_CUBE_NOISE
            )
        ):
            return _yes(
                "yes_moment",
                moment_difference_median,
                BAD_ATMOSPHERE_SIGMA,
                moment_difference_scaled_mad,
            )
        return Decision("no", 0.0, BAD_ATMOSPHERE_SIGMA)
    sigma = np.max([MOMENT_SIGMA_FLOOR, ten_event_sigma])
    if (
        moment_difference_snr >= PROCEEDING_LEVEL
        and n_above_anywhere >= MINIMUM_PIXELS_FOR_ONLY_EXTRA_MASK
        and cube_contrast > CUBE_SIGMA_THRESHOLD * cube_scaled_mad_outside
    ):
        return _yes(
            "yes_moment", moment_difference_median, sigma, moment_difference_scaled_mad
        )
    return Decision("no", 0.0, float(sigma))


def extra_mask_decision(
    bad_atmosphere: bool,
    moment_difference_median: float,
    moment_difference_snr: float,
    moment_difference_scaled_mad: float,
    ten_event_sigma: float,
) -> Decision:
    """Whether a round needs an extra mask, or a lower threshold sigma in the
    auto-lower decisions [findContinuum ``extraMaskYesOrNo``] (fC 3864-3915,
    called at 2141, 2352, 2691, 2721).

    Parameters
    ----------
    bad_atmosphere : bool
        Whether the atmosphere varies across the window.
    moment_difference_median, moment_difference_scaled_mad : float
        Median and scaled MAD of the round's moment difference.
    moment_difference_snr : float
        Its signal-to-noise ratio with the peak anywhere.
    ten_event_sigma : float
        The sigma of ten events among the valid voxels.

    Returns
    -------
    Decision
        ``"yes_moment"`` with the level of the extra mask, or ``"no"`` with
        level 0.0.
    """
    if bad_atmosphere:
        if moment_difference_snr >= MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE:
            return _yes(
                "yes_moment",
                moment_difference_median,
                BAD_ATMOSPHERE_SIGMA,
                moment_difference_scaled_mad,
            )
        return Decision("no", 0.0, BAD_ATMOSPHERE_SIGMA)
    sigma = np.max([MOMENT_SIGMA_FLOOR, ten_event_sigma])
    if moment_difference_snr >= PROCEEDING_LEVEL:
        return _yes(
            "yes_moment", moment_difference_median, sigma, moment_difference_scaled_mad
        )
    return Decision("no", 0.0, float(sigma))


def amended_joint_mask(
    joint_mask: xr.Dataset, maximum: xr.DataArray, amend_level: float
) -> xr.Dataset:
    """The joint mask grown by the pixels of a maximum map above the amend
    level (fC 2096-2099).

    Parameters
    ----------
    joint_mask : xarray.Dataset
        ``JOINT_MASK`` and ``JOINT_MASK_FOOTPRINT`` of round ``original``.
    maximum : xarray.DataArray
        The maximum map of round ``original``'s selection, NaN where invalid.
    amend_level : float
        The level of the amend decision, compared as findContinuum writes it,
        with 6 decimals.

    Returns
    -------
    xarray.Dataset
        ``JOINT_MASK`` where the joint mask holds or the maximum exceeds the
        level, with a footprint of every pixel; attributes ``kind``
        ``"amended"`` and ``primary_beam_based`` False.

    Raises
    ------
    ValueError
        If a map has more than one time or polarization.
    """
    _check_one_plane(joint_mask, maximum)
    mask = joint_mask["JOINT_MASK"]
    above = np.asarray(maximum.values, dtype=np.float64) > round_to_decimals(
        amend_level, 6
    )
    return xr.Dataset(
        {
            "JOINT_MASK": mask.copy(
                data=(mask.values & joint_mask["JOINT_MASK_FOOTPRINT"].values) | above
            ),
            "JOINT_MASK_FOOTPRINT": mask.copy(data=np.ones(mask.shape, dtype=bool)),
        },
        attrs={"kind": "amended", "primary_beam_based": False},
    )


def round_maps_ranges(
    round_kind: str,
    continuum_ranges: list,
    selection_reference: list | None,
    n_channels: int,
) -> MapsRanges:
    """The channel ranges whose maps a round needs (fC 2545-2614).

    A computed round needs the maps of its own ranges. Every other round needs
    those of the intersection of its channels with the reference's: one
    channel becomes three above 500 channels (two at a window edge), unless it
    is the whole reference. An empty intersection ends the rounds, with the
    statistics of the round's own ranges.

    Parameters
    ----------
    round_kind : str
        ``"original"``, ``"single"``, ``"amended_mask"``,
        ``"only_extra_mask"``, ``"extra_mask"``, ``"auto_lower"`` or
        ``"auto_lower_2"``.
    continuum_ranges : list
        The round's ranges.
    selection_reference : list or None
        The ranges the round intersects with; None in computed rounds.
    n_channels : int
        Channels of the window.

    Returns
    -------
    MapsRanges
        The ranges; whether the intersection is empty, whether it kept every
        channel of the reference, in its order, and whether its one channel was
        widened.

    Raises
    ------
    ValueError
        For an unknown round kind.
    """
    if round_kind in COMPUTED_ROUND_KINDS:
        return MapsRanges(continuum_ranges, False, False, False)
    if round_kind not in EXTRA_ROUND_KINDS:
        raise ValueError(f"Unknown round kind {round_kind!r}")
    reference_channels = channels_of_ranges(selection_reference)
    common = np.intersect1d(reference_channels, channels_of_ranges(continuum_ranges))
    if common.size == 0:
        return MapsRanges(continuum_ranges, True, False, False)
    if np.array_equal(common, reference_channels):
        return MapsRanges(ranges_of_channels(common), False, True, False)
    if common.size == 1 and n_channels > WIDENING_MINIMUM_CHANNELS:
        channel = int(common[0])
        widened = [[max(1, channel - 1), min(channel + 1, n_channels - 2) + 1]]
        return MapsRanges(widened, False, False, True)
    return MapsRanges(ranges_of_channels(common), False, False, False)


def change_code(
    values: tuple, references: tuple, threshold: float = CHANGE_THRESHOLD
) -> str:
    """The letters of the change of four statistics of a moment difference
    from their references [findContinuum ``compute4LetterCode``]
    (fC 3630-3687).

    Parameters
    ----------
    values, references : tuple
        Peak anywhere, peak outside the joint mask, sum and scaled MAD.
    threshold : float, default 0.01
        Relative change above which a statistic is higher or lower; twice as
        much for the scaled MAD.

    Returns
    -------
    str
        Per statistic ``"H"`` (higher), ``"L"`` (lower) or ``"S"`` (same). A
        zero reference gives an infinite or NaN change, as with findContinuum's
        NumPy values.
    """
    letters = ""
    with np.errstate(divide="ignore", invalid="ignore"):
        for index, (value, reference) in enumerate(
            zip(values, references, strict=True)
        ):
            change = (np.float64(value) - reference) / np.float64(reference)
            limit = 2 * threshold if index == 3 else threshold
            letters += "H" if change > limit else "L" if change < -limit else "S"
    return letters


def after_original_round(
    bad_atmosphere: bool,
    statistics: SelectionStatistics,
    cube_noise: CubeNoiseStatistics,
    counts: MomentDifferenceCounts,
    negative_fraction: float,
    low_bandwidth: bool,
    continuum_ranges: list,
    ranges_before_blue_pruning: list,
    aggregate_bandwidth: float,
    final_threshold_sigma: float,
    threshold_sigma: float | str,
    threshold_sigma_mode: str,
    joint_mask: xr.Dataset,
    maximum: xr.DataArray,
    moment_difference: xr.DataArray,
    primary_beam_above_spectrum_level: xr.DataArray,
) -> RoundTransition:
    """The decisions after round ``original`` (fC 1934-2126).

    Without an amended mask, the only-extra-mask decision (not for too many
    pixels); a low bandwidth makes it ``"not_evaluated"``. Then result 0 (change
    code ``SSSS``, no references), or round ``only_extra_mask`` with an extra
    mask of the moment difference. With an amended mask, round
    ``amended_mask``; if the mask grew less than twice inside the primary beam,
    with this round's final threshold sigma in mode ``"auto_lower"``.

    Parameters
    ----------
    bad_atmosphere : bool
        Whether the atmosphere varies across the window.
    statistics : SelectionStatistics
        The statistics of the round's maps.
    cube_noise : CubeNoiseStatistics
        The cube's noise outside the joint mask.
    counts : MomentDifferenceCounts
        The moment-difference pixels above the decisions' levels.
    negative_fraction : float
        The fraction of negative pixels of the maximum map outside the joint
        mask.
    low_bandwidth : bool
        Whether the round's ranges have the low-bandwidth warning.
    continuum_ranges, ranges_before_blue_pruning : list
        The round's ranges and its ranges before blue pruning.
    aggregate_bandwidth : float
        Their aggregate bandwidth in Hz.
    final_threshold_sigma : float
        The threshold sigma the round's selection ended with.
    threshold_sigma : float or str
        The round's threshold sigma.
    threshold_sigma_mode : str
        Its mode, ``"auto"`` or ``"auto_lower"``.
    joint_mask : xarray.Dataset
        The joint mask of round ``original``.
    maximum, moment_difference : xarray.DataArray
        The round's maximum map and moment difference.
    primary_beam_above_spectrum_level : xarray.DataArray
        Where the mean primary beam is valid and above 0.21.

    Returns
    -------
    RoundTransition
        Decisions ``amend_mask`` and ``only_extra_mask``; the round record adds
        ``n_masked_before`` and ``n_masked_after`` when the mask is amended.

    Raises
    ------
    ValueError
        If a map has more than one time or polarization.
    """
    _check_one_plane(
        joint_mask, maximum, moment_difference, primary_beam_above_spectrum_level
    )
    values = _change_values(statistics)
    amend = amend_mask_decision(
        bad_atmosphere,
        statistics.moment_difference_median,
        statistics.moment_difference_snr_outside,
        cube_noise.moment_difference_snr_cube,
        counts.n_above,
        counts.n_above_2,
        counts.n_above_bad_atmosphere,
        counts.n_above_cube,
        counts.n_above_cube_2,
        statistics.moment_difference_scaled_mad,
        cube_noise.ten_event_sigma,
        cube_noise.cube_scaled_mad_outside,
        negative_fraction,
    )
    unchanged = _transition(
        threshold_sigma=threshold_sigma,
        threshold_sigma_mode=threshold_sigma_mode,
        signal_ratio_tier_1=SIGNAL_RATIO_TIER_1,
        selection_reference=None,
        change_code_references=None,
        continuum_ranges=continuum_ranges,
        ranges_before_blue_pruning=ranges_before_blue_pruning,
        aggregate_bandwidth=aggregate_bandwidth,
        decisions=_decision_entries("amend_mask", amend),
    )
    if amend.kind in YES_DECISIONS:
        amended = amended_joint_mask(joint_mask, maximum, amend.level)
        n_masked_before = count_in_primary_beam(
            joint_mask["JOINT_MASK"],
            joint_mask["JOINT_MASK_FOOTPRINT"],
            primary_beam_above_spectrum_level,
        )
        n_masked_after = count_in_primary_beam(
            amended["JOINT_MASK"],
            amended["JOINT_MASK_FOOTPRINT"],
            primary_beam_above_spectrum_level,
        )
        lower = (
            n_masked_before > 0
            and float(n_masked_after) / n_masked_before < AMENDED_GROWTH_LIMIT
        )
        return _with_round_record(
            unchanged._replace(
                next_round_kind="amended_mask",
                threshold_sigma=final_threshold_sigma if lower else threshold_sigma,
                threshold_sigma_mode="auto_lower" if lower else threshold_sigma_mode,
                selection_reference=continuum_ranges,
                change_code_references=values,
                amended_joint_mask=amended,
            ),
            statistics,
            n_masked_before=n_masked_before,
            n_masked_after=n_masked_after,
        )
    only_extra = Decision("not_evaluated", 0.0, 0.0)
    if amend.sigma_used >= PIXEL_RATIO_EXCEEDED:
        only_extra = only_extra_mask_decision(
            bad_atmosphere,
            statistics.moment_difference_median,
            statistics.moment_difference_snr,
            statistics.moment_difference_scaled_mad,
            counts.n_above_anywhere,
            counts.n_above_anywhere_bad_atmosphere,
            statistics.cube_peak,
            cube_noise.first_channel_median_outside,
            cube_noise.cube_scaled_mad_outside,
            cube_noise.ten_event_sigma,
        )
    if low_bandwidth:
        only_extra = only_extra._replace(kind="not_evaluated")
    decisions = {
        **unchanged.decisions,
        **_decision_entries("only_extra_mask", only_extra),
    }
    if only_extra.kind in YES_DECISIONS:
        return _with_round_record(
            unchanged._replace(
                next_round_kind="only_extra_mask",
                signal_ratio_tier_1=SIGNAL_RATIO_TIER_1_EXTRA_MASK,
                selection_reference=continuum_ranges,
                change_code_references=values,
                extra_mask_request=ExtraMaskRequest(
                    "only_extra_mask",
                    only_extra.level,
                    moment_difference,
                    continuum_ranges,
                ),
                decisions=decisions,
            ),
            statistics,
        )
    return _with_round_record(
        unchanged._replace(
            result="result_0",
            change_code=change_code(values, values),
            decisions=decisions,
        ),
        statistics,
    )


def after_amended_round(
    bad_atmosphere: bool,
    statistics: SelectionStatistics,
    ten_event_sigma: float,
    low_bandwidth: bool,
    difference_snr: float,
    continuum_ranges: list,
    ranges_before_blue_pruning: list,
    aggregate_bandwidth: float,
    selection_reference: list,
    change_code_references: tuple,
    moment_difference: xr.DataArray,
    n_channels: int,
    threshold_sigma: float | str,
    threshold_sigma_mode: str,
) -> RoundTransition:
    """The decisions after round ``amended_mask`` (fC 2128-2223, 2471-2542).

    The extra-mask decision; a low bandwidth makes it ``"not_evaluated"``,
    which continues as a yes would with the decision's level (fC 2161). Only
    ``"no"`` ends with result 1. Then, if the difference of this round's moment
    difference from round ``original``'s is significant, the intersection of
    this round's channels with round ``original``'s: empty, result 2 with round
    ``original``'s ranges (the bandwidth and the ranges before blue pruning
    stay this round's); otherwise a request for the intersection's maps, which
    :func:`after_amended_intersection` decides on. Without a significant
    difference, round ``extra_mask`` with an extra mask of this round's moment
    difference.

    Parameters
    ----------
    bad_atmosphere : bool
        Whether the atmosphere varies across the window.
    statistics : SelectionStatistics
        The statistics of the round's maps.
    ten_event_sigma : float
        The sigma of ten events among the valid voxels.
    low_bandwidth : bool
        Whether the round's ranges have the low-bandwidth warning.
    difference_snr : float
        The signal-to-noise ratio of the difference of the moment differences,
        significant above ``8.5 * sqrt(2)``, ``12 * sqrt(2)`` with a bad
        atmosphere (fC 2219-2223).
    continuum_ranges, ranges_before_blue_pruning : list
        The round's ranges and its ranges before blue pruning.
    aggregate_bandwidth : float
        Their aggregate bandwidth in Hz.
    selection_reference : list
        The ranges of round ``original``.
    change_code_references : tuple
        The references of the change code.
    moment_difference : xarray.DataArray
        The round's moment difference.
    n_channels : int
        Channels of the window; findContinuum's intersection here widens no
        channel (fC 2227-2272), so the value is not read.
    threshold_sigma : float or str
        The round's threshold sigma.
    threshold_sigma_mode : str
        Its mode.

    Returns
    -------
    RoundTransition
        Decision ``extra_mask``, and ``intersection_of_selections`` with the
        maps request; the round record adds ``difference_snr`` when the
        difference was tested.

    Raises
    ------
    ValueError
        If the moment difference has more than one time or polarization.
    """
    _check_one_plane(moment_difference)
    extra = extra_mask_decision(
        bad_atmosphere,
        statistics.moment_difference_median,
        statistics.moment_difference_snr,
        statistics.moment_difference_scaled_mad,
        ten_event_sigma,
    )
    if low_bandwidth:
        extra = extra._replace(kind="not_evaluated")
    unchanged = _transition(
        threshold_sigma=threshold_sigma,
        threshold_sigma_mode=threshold_sigma_mode,
        signal_ratio_tier_1=SIGNAL_RATIO_TIER_1,
        selection_reference=selection_reference,
        change_code_references=change_code_references,
        continuum_ranges=continuum_ranges,
        ranges_before_blue_pruning=ranges_before_blue_pruning,
        aggregate_bandwidth=aggregate_bandwidth,
        decisions=_decision_entries("extra_mask", extra),
    )
    if extra.kind == "no":
        return _with_round_record(
            unchanged._replace(
                result="result_1",
                change_code=change_code(
                    _change_values(statistics), change_code_references
                ),
            ),
            statistics,
        )
    difference_level = (
        DIFFERENCE_BAD_ATMOSPHERE_LEVEL if bad_atmosphere else DIFFERENCE_LEVEL
    ) * np.sqrt(2)
    if difference_snr > difference_level:
        reference_channels = channels_of_ranges(selection_reference)
        common = np.intersect1d(
            reference_channels, channels_of_ranges(continuum_ranges)
        )
        if common.size == 0:
            return _with_round_record(
                unchanged._replace(
                    result="result_2",
                    continuum_ranges=ranges_of_channels(reference_channels),
                    change_code=change_code(
                        _change_values(statistics), change_code_references
                    ),
                ),
                statistics,
                difference_snr=difference_snr,
            )
        intersection_ranges = ranges_of_channels(common)
        return _with_round_record(
            unchanged._replace(
                maps_request=intersection_ranges,
                continuum_ranges=intersection_ranges,
                ranges_before_blue_pruning=intersection_ranges,
                intersection_of_selections=True,
                decisions={**unchanged.decisions, "intersection_of_selections": True},
            ),
            statistics,
            difference_snr=difference_snr,
        )
    return _with_round_record(
        unchanged._replace(
            next_round_kind="extra_mask",
            signal_ratio_tier_1=SIGNAL_RATIO_TIER_1_EXTRA_MASK,
            selection_reference=continuum_ranges,
            extra_mask_request=ExtraMaskRequest(
                "without_intersection", extra.level, moment_difference, continuum_ranges
            ),
        ),
        statistics,
        difference_snr=difference_snr,
    )


def after_amended_intersection(
    bad_atmosphere: bool,
    statistics: SelectionStatistics,
    ten_event_sigma: float,
    intersection_ranges: list,
    channel_width: float,
    change_code_references: tuple,
    moment_difference: xr.DataArray,
    threshold_sigma: float | str,
    threshold_sigma_mode: str,
) -> RoundTransition:
    """The second extra-mask decision of round ``amended_mask``, on the maps
    of the intersection of its channels with round ``original``'s
    (fC 2266-2470).

    ``"no"`` ends with result 3 and the intersection's bandwidth; otherwise
    round ``extra_mask`` with an extra mask of the intersection's moment
    difference. No bandwidth warning stops this decision.

    Parameters
    ----------
    bad_atmosphere : bool
        Whether the atmosphere varies across the window.
    statistics : SelectionStatistics
        The statistics of the intersection's maps.
    ten_event_sigma : float
        The sigma of ten events among the valid voxels.
    intersection_ranges : list
        The ranges of the intersection.
    channel_width : float
        Channel width in Hz.
    change_code_references : tuple
        The references of the change code.
    moment_difference : xarray.DataArray
        The intersection's moment difference.
    threshold_sigma : float or str
        The round's threshold sigma.
    threshold_sigma_mode : str
        Its mode.

    Returns
    -------
    RoundTransition
        Decision ``extra_mask_2`` and ``intersection_of_selections``. The ranges
        and the ranges before blue pruning are the intersection's; the
        selection reference, read only by a next round, is None at result 3.

    Raises
    ------
    ValueError
        If the moment difference has more than one time or polarization.
    """
    _check_one_plane(moment_difference)
    extra_2 = extra_mask_decision(
        bad_atmosphere,
        statistics.moment_difference_median,
        statistics.moment_difference_snr,
        statistics.moment_difference_scaled_mad,
        ten_event_sigma,
    )
    unchanged = _transition(
        threshold_sigma=threshold_sigma,
        threshold_sigma_mode=threshold_sigma_mode,
        signal_ratio_tier_1=SIGNAL_RATIO_TIER_1,
        selection_reference=None,
        change_code_references=change_code_references,
        continuum_ranges=intersection_ranges,
        ranges_before_blue_pruning=intersection_ranges,
        aggregate_bandwidth=selection_bandwidth(intersection_ranges, channel_width),
        decisions={
            **_decision_entries("extra_mask_2", extra_2),
            "intersection_of_selections": True,
        },
    )._replace(intersection_of_selections=True)
    if extra_2.kind == "no":
        return _with_round_record(
            unchanged._replace(
                result="result_3",
                change_code=change_code(
                    _change_values(statistics), change_code_references
                ),
            ),
            statistics,
        )
    return _with_round_record(
        unchanged._replace(
            next_round_kind="extra_mask",
            signal_ratio_tier_1=SIGNAL_RATIO_TIER_1_EXTRA_MASK,
            selection_reference=intersection_ranges,
            extra_mask_request=ExtraMaskRequest(
                "with_intersection",
                extra_2.level,
                moment_difference,
                intersection_ranges,
            ),
        ),
        statistics,
    )


def after_extra_round(
    round_kind: str,
    maps_ranges: MapsRanges,
    statistics: SelectionStatistics,
    final_threshold_sigma: float,
    warnings: BandwidthWarnings,
    bad_atmosphere: bool,
    n_rounds_after_original: int,
    ten_event_sigma: float,
    change_code_references: tuple,
    continuum_ranges: list,
    ranges_before_blue_pruning: list,
    aggregate_bandwidth: float,
    selection_reference: list,
    channel_width: float,
    threshold_sigma_mode: str,
) -> RoundTransition:
    """The decisions after a round ``only_extra_mask``, ``extra_mask``,
    ``auto_lower`` or ``auto_lower_2`` (fC 2545-2816).

    An empty intersection ends with result 6: the reference's ranges, the
    round's own bandwidth and ranges before blue pruning, and the change code
    of the own maps. Otherwise the intersection's ranges, ``"no_improvement"``
    for the round's decision when they kept the reference (not in
    ``only_extra_mask``), then the auto-lower decisions: round ``auto_lower``
    after ``extra_mask`` (more than two rounds) or ``only_extra_mask`` (more
    than one) without warnings, with the final threshold sigma times 6/7 (5/7
    from a signal-to-noise ratio of 15); round ``auto_lower_2`` after
    ``auto_lower`` (more than three rounds), times 6/7; ``auto_lower_2``
    decided ``"yes_moment"`` otherwise, outside ``auto_lower`` (fC 2717-2719).
    Else result 7 with the intersection's bandwidth.

    Parameters
    ----------
    round_kind : str
        The round's kind.
    maps_ranges : MapsRanges
        The ranges of the round's maps (:func:`round_maps_ranges`).
    statistics : SelectionStatistics
        The statistics of those maps.
    final_threshold_sigma : float
        The threshold sigma the round's selection ended with.
    warnings : BandwidthWarnings
        The warnings of ``maps_ranges.ranges``.
    bad_atmosphere : bool
        Whether the atmosphere varies across the window.
    n_rounds_after_original : int
        Rounds allowed after round ``original``, at most 4 as findContinuum's
        settings make it.
    ten_event_sigma : float
        The sigma of ten events among the valid voxels.
    change_code_references : tuple
        The references of the change code.
    continuum_ranges : list
        The round's own ranges; ``maps_ranges`` already holds what is read of
        them.
    ranges_before_blue_pruning : list
        The round's own ranges before blue pruning.
    aggregate_bandwidth : float
        The round's own aggregate bandwidth in Hz.
    selection_reference : list
        The ranges the round intersected with.
    channel_width : float
        Channel width in Hz.
    threshold_sigma_mode : str
        The round's threshold-sigma mode.

    Returns
    -------
    RoundTransition
        Decisions among ``extra_mask``, ``auto_lower`` and ``auto_lower_2``;
        the round record adds the three flags of ``maps_ranges``. The threshold
        sigma, read only by a next round, is the final one at a result.

    Raises
    ------
    ValueError
        For a round kind that is not one of these four.
    """
    if round_kind not in EXTRA_ROUND_KINDS:
        raise ValueError(f"Round kind {round_kind!r} is not an extra-mask round")
    flags = {
        "empty_intersection": maps_ranges.empty_intersection,
        "no_improvement": maps_ranges.no_improvement,
        "widened": maps_ranges.widened,
    }
    code = change_code(_change_values(statistics), change_code_references)
    unchanged = _transition(
        threshold_sigma=final_threshold_sigma,
        threshold_sigma_mode=threshold_sigma_mode,
        signal_ratio_tier_1=SIGNAL_RATIO_TIER_1_EXTRA_MASK,
        selection_reference=selection_reference,
        change_code_references=change_code_references,
        continuum_ranges=maps_ranges.ranges,
        ranges_before_blue_pruning=maps_ranges.ranges,
        aggregate_bandwidth=aggregate_bandwidth,
        decisions={},
    )
    if maps_ranges.empty_intersection:
        return _with_round_record(
            unchanged._replace(
                result="result_6",
                continuum_ranges=ranges_of_channels(
                    channels_of_ranges(selection_reference)
                ),
                ranges_before_blue_pruning=ranges_before_blue_pruning,
                change_code=code,
            ),
            statistics,
            **flags,
        )
    decisions = {}
    if maps_ranges.no_improvement and round_kind != "only_extra_mask":
        decisions.update(
            _decision_entries(round_kind, Decision("no_improvement", 0.0, 0.0))
        )
    if (round_kind == "extra_mask" and n_rounds_after_original > 2) or (
        round_kind == "only_extra_mask" and n_rounds_after_original > 1
    ):
        if not (warnings.low_bandwidth or warnings.low_spread):
            auto_lower = extra_mask_decision(
                bad_atmosphere,
                statistics.moment_difference_median,
                statistics.moment_difference_snr,
                statistics.moment_difference_scaled_mad,
                ten_event_sigma,
            )
            decisions.update(_decision_entries("auto_lower", auto_lower))
            if auto_lower.kind in YES_DECISIONS:
                if statistics.moment_difference_snr < APPROACH_LEVEL:
                    lowered = final_threshold_sigma * 6 / 7.0
                else:
                    lowered = final_threshold_sigma * 5 / 7.0
                return _lowered(
                    unchanged, "auto_lower", lowered, decisions, statistics, flags
                )
    elif round_kind != "auto_lower":
        decisions.update(
            _decision_entries("auto_lower_2", Decision("yes_moment", 0.0, 0.0))
        )
    else:
        auto_lower_2 = extra_mask_decision(
            bad_atmosphere,
            statistics.moment_difference_median,
            statistics.moment_difference_snr,
            statistics.moment_difference_scaled_mad,
            ten_event_sigma,
        )
        decisions.update(_decision_entries("auto_lower_2", auto_lower_2))
        if auto_lower_2.kind in YES_DECISIONS and n_rounds_after_original > 3:
            return _lowered(
                unchanged,
                "auto_lower_2",
                final_threshold_sigma * 6 / 7.0,
                decisions,
                statistics,
                flags,
            )
    return _with_round_record(
        unchanged._replace(
            result="result_7",
            aggregate_bandwidth=selection_bandwidth(maps_ranges.ranges, channel_width),
            change_code=code,
            decisions=decisions,
        ),
        statistics,
        **flags,
    )


def after_rounds(
    change_code: str | None,
    moment_difference_snr: float,
    moment_difference_level: float,
    warnings: BandwidthWarnings,
) -> FinalStep:
    """Whether the rounds' ranges revert to round ``original``'s, or else get
    the low-spread fix (fC 2834-2853).

    Parameters
    ----------
    change_code : str or None
        The change code of the result.
    moment_difference_snr : float
        The signal-to-noise ratio of the last moment difference.
    moment_difference_level : float
        The moment-difference level, as round ``original`` left it.
    warnings : BandwidthWarnings
        The warnings of the rounds' ranges.

    Returns
    -------
    FinalStep
        Revert for a change code ``HHHH`` or ``HHHS`` with a signal-to-noise
        ratio above the level; otherwise the fix runs for a low spread.
    """
    revert = bool(
        change_code in REVERSION_CODES
        and moment_difference_snr > moment_difference_level
    )
    return FinalStep(revert, not revert and warnings.low_spread)


def low_spread_ranges(
    continuum_ranges: list,
    n_channels: int,
    baseline_channels: np.ndarray,
    spectrum: np.ndarray,
    positive_threshold: float,
    negative_threshold: float,
) -> list | None:
    """The ranges with the channels the low-spread fix adds on the emptier
    side of the window (fC 2856-2919, 6384-6445).

    The candidate is the widest run of baseline channels on that side, scored
    by the spread it gives; without one, the channels on that side within a
    scaled MAD of the baseline values' range. The candidate is dropped when its
    median lies more than a scaled MAD beyond a threshold; a dropped run is
    replaced by the longest run of channels strictly between
    ``negative_threshold - s * mad`` and ``positive_threshold + s * mad``, ``s``
    the median's distance below the negative threshold in scaled MADs
    (fC 2890-2899). Each run of the candidate goes after the ranges if it lies
    above them, else before them, so that several runs below end in decreasing
    order.

    Parameters
    ----------
    continuum_ranges : list
        The ranges, in list order.
    n_channels : int
        Channels of the window.
    baseline_channels : numpy.ndarray
        The baseline channels of the last computed round.
    spectrum : numpy.ndarray
        Its spectrum.
    positive_threshold, negative_threshold : float
        Its thresholds.

    Returns
    -------
    list or None
        The new ranges, or None when no channel is added.
    """
    channels = channels_of_ranges(continuum_ranges)
    first, last = channels[0], channels[-1]
    if first > n_channels - last:
        window = (
            0,
            int(
                np.min([last - n_channels * SMALL_SPREAD_FRACTION - 1, n_channels // 2])
            ),
        )
    else:
        window = (int(first + n_channels * SMALL_SPREAD_FRACTION + 1), int(n_channels))
    baseline_values = spectrum[baseline_channels]
    scaled_mad = scaled_median_absolute_deviation(baseline_values)
    candidate = _widest_baseline_run(baseline_channels, window, first, last)
    scanned = candidate is None
    if scanned:
        candidate = _comparable_channels(
            spectrum,
            window,
            np.min(baseline_values),
            np.max(baseline_values),
            1,
            scaled_mad,
        )
        if candidate.size == 0:
            return None
    with np.errstate(divide="ignore", invalid="ignore"):
        median = np.median(spectrum[candidate])
        negative_sigma = (negative_threshold - median) / scaled_mad
        positive_sigma = (median - positive_threshold) / scaled_mad
    if negative_sigma > 1 or positive_sigma > 1:
        if scanned:
            return None
        retry = _comparable_channels(
            spectrum,
            window,
            negative_threshold,
            positive_threshold,
            negative_sigma,
            scaled_mad,
        )
        if retry.size == 0:
            return None
        runs = contiguous_runs(retry)
        candidate = runs[int(np.argmax([run.size for run in runs]))]
    ranges = [list(pair) for pair in continuum_ranges]
    for run in contiguous_runs(candidate):
        added = [int(run[0]), int(run[-1]) + 1]
        if run[0] > last:
            ranges.append(added)
        else:
            ranges.insert(0, added)
    return ranges


def after_low_spread_fix(
    low_spread_change_code: str | None,
    moment_difference_snr: float,
    moment_difference_level: float,
    warnings: BandwidthWarnings,
) -> bool:
    """Whether the ranges revert to round ``original``'s after the low-spread
    fix (fC 3040-3053).

    Parameters
    ----------
    low_spread_change_code : str or None
        The change code of the fixed ranges' maps (threshold 0.05), or None
        when the fix added no channel.
    moment_difference_snr : float
        The signal-to-noise ratio of the last moment difference.
    moment_difference_level : float
        The moment-difference level, as round ``original`` left it.
    warnings : BandwidthWarnings
        The warnings of the fixed ranges, or of the unchanged ones.

    Returns
    -------
    bool
        True for a change code ``HHHH`` or ``HHHS`` with a signal-to-noise ratio
        above the level, or for both warnings.
    """
    return bool(
        (
            low_spread_change_code in REVERSION_CODES
            and moment_difference_snr > moment_difference_level
        )
        or (warnings.low_bandwidth and warnings.low_spread)
    )


def path_code(decisions: list[dict], rounds_enabled: bool) -> str:
    """The code of the decisions of the rounds [findContinuum's decision code
    in ``compute4LetterCodeAndUpdateLegend``] (fC 3424-3482, 3083-3084).

    Parameters
    ----------
    decisions : list of dict
        The ``decisions`` of the round transitions in order, with
        ``{"added_selection": True}`` after the low-spread fix added channels.
        Later entries replace earlier ones, as findContinuum overwrites its
        decisions.
    rounds_enabled : bool
        Whether the rounds ran, with 4 rounds allowed after round
        ``original``.

    Returns
    -------
    str
        For instance ``"S"``, ``"PR"``, ``"P#"``, ``"ED"``, ``"AD"``, ``"ADIE"``
        or ``"EDxY+"``; empty without rounds.
    """
    if not rounds_enabled:
        return ""
    merged = {}
    for entries in decisions:
        merged.update(entries)
    amend = merged.get("amend_mask_decision", "not_evaluated")
    extra = merged.get(
        "extra_mask_decision",
        merged.get("only_extra_mask_decision", "not_evaluated"),
    )
    extra_2 = merged.get("extra_mask_2_decision", "not_evaluated")
    intersection = merged.get("intersection_of_selections", False)
    if amend in ("not_evaluated", "no"):
        if extra in ("not_evaluated", "no"):
            sigma_used = merged.get("amend_mask_sigma_used")
            code = (
                "PR"
                if sigma_used == PIXEL_RATIO_EXCEEDED
                else "P#"
                if sigma_used == PIXELS_ABOVE_THRESHOLD_EXCEEDED
                else "S"
            )
        else:
            code = {"no_improvement": "Se", "yes_moment": "ED", "yes_cube": "EC"}.get(
                extra, ""
            )
    else:
        code = "A" + {"yes_moment": "D", "yes_cube": "C"}.get(amend, "")
        if (extra in YES_DECISIONS or extra_2 in YES_DECISIONS) and intersection:
            code += "IE" if extra_2 in YES_DECISIONS else "I"
        elif extra in YES_DECISIONS or extra_2 in YES_DECISIONS:
            code += "E"
        elif intersection:
            code += "I"
    code += _letter(merged.get("auto_lower_decision", "not_evaluated"), "X")
    code += _letter(merged.get("auto_lower_2_decision", "not_evaluated"), "Y")
    return code + ("+" if merged.get("added_selection", False) else "")


def _yes(kind, median, sigma, scaled_mad):
    """A yes decision at level ``median + sigma * scaled_mad``."""
    return Decision(kind, float(median + sigma * scaled_mad), float(sigma))


def _change_values(statistics):
    """The statistics of the change code: peak anywhere, peak outside the joint
    mask, sum and scaled MAD of the moment difference."""
    return (
        statistics.moment_difference_peak,
        statistics.moment_difference_peak_outside,
        statistics.moment_difference_sum,
        statistics.moment_difference_scaled_mad,
    )


def _decision_entries(name, decision):
    """The ``decisions`` entries of one decision."""
    return {
        f"{name}_decision": decision.kind,
        f"{name}_level": decision.level,
        f"{name}_sigma_used": decision.sigma_used,
    }


def _transition(
    *,
    threshold_sigma,
    threshold_sigma_mode,
    signal_ratio_tier_1,
    selection_reference,
    change_code_references,
    continuum_ranges,
    ranges_before_blue_pruning,
    aggregate_bandwidth,
    decisions,
):
    """A transition to no result and no next round, with the given state."""
    return RoundTransition(
        result=None,
        next_round_kind=None,
        threshold_sigma=threshold_sigma,
        threshold_sigma_mode=threshold_sigma_mode,
        signal_ratio_tier_1=signal_ratio_tier_1,
        selection_reference=selection_reference,
        change_code_references=change_code_references,
        amended_joint_mask=None,
        maps_request=None,
        extra_mask_request=None,
        continuum_ranges=continuum_ranges,
        ranges_before_blue_pruning=ranges_before_blue_pruning,
        aggregate_bandwidth=aggregate_bandwidth,
        change_code=None,
        intersection_of_selections=False,
        decisions=decisions,
        round_record={},
    )


def _lowered(unchanged, round_kind, threshold_sigma, decisions, statistics, flags):
    """The transition to an auto-lower round, which intersects with the
    current ranges (fC 2697-2711, 2727-2736)."""
    return _with_round_record(
        unchanged._replace(
            next_round_kind=round_kind,
            threshold_sigma=threshold_sigma,
            threshold_sigma_mode="auto_lower",
            selection_reference=unchanged.continuum_ranges,
            decisions=decisions,
        ),
        statistics,
        **flags,
    )


def _with_round_record(transition, statistics, **diagnostics):
    """``transition`` with its row of the rounds' diagnostics."""
    return transition._replace(
        round_record={
            "result": transition.result,
            "next_round_kind": transition.next_round_kind,
            **transition.decisions,
            **statistics._asdict(),
            **diagnostics,
        }
    )


def _letter(kind, letter):
    """The path-code letter of an auto-lower decision (fC 3467-3480)."""
    if kind in YES_DECISIONS:
        return letter
    if kind == "no_improvement":
        return letter.lower()
    return ""


def _check_one_plane(*maps):
    """Raise unless every map has one time and one polarization."""
    for values in maps:
        if values.sizes.get("time") != 1 or values.sizes.get("polarization") != 1:
            raise ValueError(
                "find_continuum_ultra's maps have one time and one polarization"
            )


def _widest_baseline_run(baseline_channels, window, first, last):
    """The first run of baseline channels in ``[window[0], window[1])`` with the
    highest positive score, run length times the spread it would give over the
    ranges' spread [findContinuum ``findWidestContiguousListInChannelRange``]
    (fC 6418-6445); None if none."""
    spread = last - first + 1
    widest, widest_score = None, 0
    for run in contiguous_runs(np.sort(baseline_channels)):
        if run[0] >= window[0] and run[-1] < window[1]:
            score = run.size * (
                np.max([run[-1] - first + 1, last - run[0] + 1]) / spread
            )
            if score > widest_score:
                widest, widest_score = run, score
    return widest


def _comparable_channels(spectrum, window, lower, upper, within, scaled_mad):
    """The channels in ``[window[0], window[1])`` whose values lie strictly
    between ``lower - within * scaled_mad`` and ``upper + within * scaled_mad``
    [findContinuum ``findChannelsInChannelRangeWithComparableIntensity``]
    (fC 6384-6416)."""
    with np.errstate(invalid="ignore"):
        low = lower - within * scaled_mad
        high = upper + within * scaled_mad
    channels = np.arange(window[0], window[1])
    values = spectrum[channels]
    return channels[(low < values) & (values < high)]
