"""The values that the steps of find_continuum_ultra return, as named tuples
without methods.

Maps and regions are image-shaped (``time, frequency, polarization, l, m``,
frequency length 1); spectra are one-dimensional float64 arrays; channel ranges
are ``[start, stop)`` pairs relative to the spectral window, in findContinuum's
list order. ``fC`` line numbers are those of the pipeline's ``findContinuum.py``
(pipeline 2026.3.0.7).
"""

from typing import NamedTuple

import numpy as np
import xarray as xr


class FindContinuumSettings(NamedTuple):
    """findContinuum's settings of a spectral window, from its metadata."""

    n_channels: int
    first_frequency: float  # Hz, f[0] (fC 7460-7464)
    last_frequency: float  # Hz, linear extrapolation, not f[-1] (fC 7461-7464)
    channel_increment: float  # Hz, signed (fC 7460)
    channel_width: float  # Hz, abs(channel_increment) (fC 951)
    mean_frequency: float  # np.mean([first_frequency, last_frequency]) (fC 959)
    rest_frequency: float  # Hz
    # (n_channels,) km/s: abs(numpy.gradient(radio velocity))
    moment_channel_weights: np.ndarray
    n_l: int
    n_m: int
    cell_l_arcsec: float  # signed (fC 7410-7417)
    cell_m_arcsec: float
    beam_major_arcsec: float  # median over channels (fC 7398-7399)
    beam_minor_arcsec: float
    max_baseline: float  # m (fC 7430-7433)
    is_time_division_mode: bool  # ALMA rule, for the spectrum and selection steps
    # Telescope rule, for the number of rounds only (fC 1071).
    is_time_division_mode_for_telescope: bool
    n_bin: int  # fC 989-1068
    n_rounds_after_original: int  # fC 1070-1080
    rounds_enabled: bool  # fC 1302-1309
    transmission_variation: float  # fC 4479-4480
    sky_temperature_std: float  # fC 4487
    bad_atmosphere: bool  # also the normalization switch (fC 1244-1254)
    moment_difference_level: float  # 8.0 (+3.5) (fC 1364-1370)
    n_baseline_channels: int  # fC 5092-5097
    single_continuum: bool
    primary_beam_mean_channels: tuple  # of the mean primary beam (fC 7793-7794)


class FirstRoundRows(NamedTuple):
    """Per-channel rows of the first round, float64 NaN for an empty channel."""

    sky_valid_count: np.ndarray
    sky_minimum: np.ndarray
    sky_maximum: np.ndarray
    primary_beam_valid_count: np.ndarray
    primary_beam_minimum: np.ndarray


class BeamGeometry(NamedTuple):
    """The beam, the pixel cells and the longest baseline of a spectral window."""

    beam_major_arcsec: float
    beam_minor_arcsec: float
    cell_l_arcsec: float
    cell_m_arcsec: float
    max_baseline: float


class EdgeSamples(NamedTuple):
    """The edges of the middle primary-beam plane (fC 8341-8368, 8389-8402)."""

    corners_empty_edges_valid: bool
    edge_maximum: float  # np.nanmax of the four edge-line maxima


class AnnulusLevels(NamedTuple):
    """The primary-beam levels of an annulus."""

    lower: float  # np.max([0.21, minimum]) (fC 8411)
    higher: float


class FirstRoundStatistics(NamedTuple):
    """The statistics of the first round's joint mask."""

    integrated_peak: float
    integrated_snr: float
    maximum_peak: float
    maximum_snr: float
    # Peaks over the unclipped MADs (fC 7889-7899).
    integrated_peak_over_mad: float
    maximum_peak_over_mad: float
    # fC 7939; the amended round divides by it (fC 8214).
    integrated_chauvenet_mad: float
    n_pixels_maximum_mask: int  # fC 8026
    n_pixels_in_mask: int  # fC 8186
    regions_pruned: int
    prune_size: int
    thresholds: tuple  # the formatted levels of phases 1 and 2 (diagnostics)
    needs_primary_beam_based_mask: bool  # fC 8187


class SpectrumRegions(NamedTuple):
    """The two-dimensional regions and per-plane tests of one spectrum pass."""

    mean_region: xr.DataArray  # bool
    use_spectrum_mask: bool  # primary beam > 0.21 in each plane (fC 8453)
    scaled_mad_region: xr.DataArray | None
    # Already round_to_significant_digits(level, 6) (fC 8503-8505).
    scaled_mad_primary_beam_levels: AnnulusLevels | None


class ComputedSpectrum(NamedTuple):
    """The spectrum of a computed round."""

    spectrum: np.ndarray  # after normalization, quadratic and smoothing
    mean_spectrum: np.ndarray  # the pass's mean profile (diagnostic)
    normalized: bool
    initial_quadratic_removed: bool
    quadratic_ratio: float
    initial_peak_over_mad: float
    n_bin: int  # capped (fC 8284-8292)
    needs_primary_beam_based_mask: bool  # scaled MAD of the spectrum == 0 (fC 8258)


class ExtraMaskSpectrum(NamedTuple):
    """The spectrum of an extra-mask round, rounded as in findContinuum's
    spectrum file."""

    spectrum: np.ndarray  # float("%.10f" % v) per value (fC 6511-6514)
    first_frequency: float  # float("%.1f" % f)
    last_frequency: float
    median: float
    scaled_mad: float
    replaced_ranges: list


class ChannelThreshold(NamedTuple):
    """The results of one channel threshold [findContinuum
    ``findContinuumChannels``] (fC 7071-7079)."""

    continuum_ranges: list
    group_count: int
    positive_threshold: float
    negative_threshold: float
    median: float
    corrected_median: float
    mad: float
    correction_factor: float
    signal_ratio: float
    line_strength_factor: float
    groups_above_threshold: int
    single_channel_peaks_above_threshold: int
    spectral_difference: float
    spectral_difference_2: float
    trim: int | str
    use_low_baseline: bool
    narrow: int
    baseline_channels: np.ndarray
    baseline_values: np.ndarray
    mad_ratio: float | None
    use_middle_baseline: bool
    ranges_dropped: int
    nonzero_mad_positions: np.ndarray | None


class ChannelSelection(NamedTuple):
    """The continuum channels of a round and the values that selected them."""

    continuum_ranges: list
    ranges_before_blue_pruning: list
    final_threshold_sigma: float
    positive_threshold: float
    negative_threshold: float
    corrected_median: float
    median: float
    mad: float
    baseline_channels: np.ndarray
    baseline_values: np.ndarray
    nonzero_mad_positions: np.ndarray | None
    peak_over_mad: float
    figure_values: dict  # the legend values of findContinuum's figure (diagnostics)


class MapStatistics(NamedTuple):
    """The statistics of the valid pixels of a map in a region."""

    count: int
    sum: float
    maximum: float
    minimum: float
    median: float
    median_absolute_deviation: float
    standard_deviation: float  # ddof 1; 0.0 for one value


class ImageSnr(NamedTuple):
    """The signal-to-noise ratio of a map [findContinuum ``imageSNR``]."""

    snr: float
    peak: float
    median: float
    scaled_mad: float


class SelectionStatistics(NamedTuple):
    """The statistics of the maximum and moment-difference maps of a selection."""

    maximum_snr: float
    maximum_peak: float
    maximum_median: float
    maximum_scaled_mad: float
    maximum_snr_outside: float
    maximum_peak_outside: float
    maximum_sum: float
    moment_difference_snr: float
    moment_difference_peak: float
    moment_difference_median: float
    moment_difference_scaled_mad: float
    moment_difference_snr_outside: float
    moment_difference_peak_outside: float
    moment_difference_sum: float
    cube_peak: float  # maximum of the selection's maximum map (fC 1971-1972)


class CubeNoiseStatistics(NamedTuple):
    """The noise of the cube outside the joint mask."""

    cube_scaled_mad_outside: float
    first_channel_median_outside: float
    ten_event_sigma: float
    moment_difference_snr_cube: float


class MomentDifferenceCounts(NamedTuple):
    """The pixels of the moment difference above the levels of the decisions."""

    n_above: int
    n_above_2: int
    n_above_bad_atmosphere: int
    n_above_2_bad_atmosphere: int
    n_above_cube: int
    n_above_cube_2: int
    n_above_anywhere: int  # no region (fC 1975)
    n_above_anywhere_bad_atmosphere: int  # level 11.0 (fC 1976)


class Decision(NamedTuple):
    """A decision of a round."""

    kind: str  # "not_evaluated", "no", "yes_moment", "yes_cube", "no_improvement"
    level: float
    sigma_used: float


class MapsRanges(NamedTuple):
    """The channel ranges whose maps a round needs (fC 2545-2614)."""

    ranges: list
    empty_intersection: bool
    no_improvement: bool
    widened: bool


class ExtraMaskRequest(NamedTuple):
    """The spectrum pass of an extra mask."""

    # "only_extra_mask" (fC 2041), "with_intersection" (fC 2441) or
    # "without_intersection" (fC 2513).
    form: str
    level: float
    moment_difference: xr.DataArray
    continuum_ranges: list


class RoundTransition(NamedTuple):
    """What a round decided: a result or the next round, and the values for
    it."""

    result: str | None  # "result_0" ... "result_7"
    next_round_kind: str | None
    threshold_sigma: float | str
    threshold_sigma_mode: str
    signal_ratio_tier_1: float
    selection_reference: list | None
    change_code_references: tuple | None  # (peak, peak outside, sum, scaled MAD)
    amended_joint_mask: xr.Dataset | None
    # The ranges of the intersection whose maps the next step needs.
    maps_request: list | None
    extra_mask_request: ExtraMaskRequest | None
    continuum_ranges: list
    ranges_before_blue_pruning: list
    aggregate_bandwidth: float  # Hz
    change_code: str | None
    intersection_of_selections: bool
    decisions: dict  # values, levels, sigma used (diagnostics)
    round_record: dict  # one row of diagnostics["rounds"]


class BandwidthWarnings(NamedTuple):
    """The low-bandwidth and low-spread warnings of a selection."""

    low_bandwidth: bool
    low_spread: bool
    bandwidth_fraction: float
    spread_fraction: float


class FinalStep(NamedTuple):
    """The steps after the rounds."""

    revert: bool
    run_low_spread_fix: bool
