"""findContinuum's settings of a spectral window, from its metadata.

The spectral axis, the pixel cells, the beam and the longest baseline, the
time-division mode, the channels to average, the number of rounds, the
variation of the atmosphere, and the levels and counts the later steps start
from, each computed as findContinuum computes it before its first round.
``fC`` line numbers are those of the pipeline's ``findContinuum.py`` (pipeline
2026.3.0.7).
"""

import numpy as np
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    line_fit,
    round_half_up,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    FindContinuumSettings,
)
from astroviper.utils.param_docs import shares_param_docs

ARCSEC_PER_RAD = 206264.80624709636  # fC 7368
SPEED_OF_LIGHT = 2.99792458e8  # m/s (fC 5314, 7369)
# km/s, of the radio velocity whose channel widths weight moment 0, and of the
# channel width in km/s (fC 1757).
SPEED_OF_LIGHT_KM_S = 299792.458
BASELINE_CHANNEL_FRACTION = 0.19  # nBaselineChannels (fC 450)
MIN_BINS = 13  # fewest bins of n_bin channels [findContinuum minRegions] (fC 990)
TIME_DIVISION_MODE_MAX_BASELINE = 400  # m (fC 1072)
ROUNDS_AFTER_ORIGINAL = 4  # amendMaskIterations (fC 1076, 1079)
TRANSMISSION_VARIATION_THRESHOLD = 0.40  # skyTransmissionThreshold (fC 464)
SKY_TEMPERATURE_STD_THRESHOLD = 5.0  # K, skyTempThreshold (fC 463)
ACA_MAX_BASELINE = 60  # m (fC 223)
MOMENT_DIFFERENCE_LEVEL = 8.0  # momDiffLevel (fC 491)


def linear_spectral_axis(frequency: np.ndarray) -> tuple[int, float, float, float]:
    """The linear spectral axis of a spectral window as findContinuum sees it
    [findContinuum ``numberOfChannelsInCube``] (fC 942-951, 7436-7477).

    Parameters
    ----------
    frequency : numpy.ndarray
        Frequencies of the window's channels (Hz), one linear run.

    Returns
    -------
    n_channels : int
    first_frequency : float
        ``frequency[0]`` (Hz).
    last_frequency : float
        ``first_frequency + (n_channels - 1) * channel_increment``, the linear
        axis at the last channel rather than ``frequency[-1]`` (Hz).
    channel_increment : float
        ``frequency[1] - frequency[0]``, signed (Hz); 1.0 for a window of one
        channel.
    """
    frequency = np.asarray(frequency)
    n_channels = int(frequency.size)
    first_frequency = float(frequency[0])
    channel_increment = float(frequency[1] - frequency[0]) if n_channels > 1 else 1.0
    return (
        n_channels,
        first_frequency,
        first_frequency + (n_channels - 1) * channel_increment,
        channel_increment,
    )


def moment_channel_weights(
    first_frequency: float,
    channel_increment: float,
    n_channels: int,
    rest_frequency: float,
) -> np.ndarray:
    """The weight of every channel in moment 0: its width in radio velocity.

    Parameters
    ----------
    first_frequency : float
        Frequency of the window's first channel (Hz).
    channel_increment : float
        Signed channel increment of the window's linear axis (Hz).
    n_channels : int
        Channels of the window.
    rest_frequency : float
        Rest frequency of the radio velocity (Hz).

    Returns
    -------
    numpy.ndarray
        float64 ``(n_channels,)``: ``abs(numpy.gradient(v))`` of the radio
        velocity ``v = 299792.458 (1 - f / rest_frequency)`` (km/s) of every
        channel of the linear axis; one for a window of one channel.
    """
    frequency = first_frequency + channel_increment * np.arange(n_channels)
    velocity = SPEED_OF_LIGHT_KM_S * (1.0 - frequency / rest_frequency)
    return np.abs(np.gradient(velocity)) if n_channels > 1 else np.ones(1)


def pixel_cells_arcsec(l: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    """The pixel cells of an image (fC 7368, 7410-7417).

    Parameters
    ----------
    l, m : numpy.ndarray
        Image coordinates (rad), at least two pixels each.

    Returns
    -------
    cell_l_arcsec, cell_m_arcsec : float
        ``(l[1] - l[0]) * 206264.80624709636`` and the same along ``m``
        (arcsec, signed).
    """
    return (
        float(l[1] - l[0]) * ARCSEC_PER_RAD,
        float(m[1] - m[0]) * ARCSEC_PER_RAD,
    )


def max_baseline_from_beam_fits(
    beam_fit_params_point_spread_function: xr.DataArray, first_frequency: float
) -> tuple[float, float, float]:
    """The beam of the point spread function and the longest baseline it
    implies (fC 980-986, 7375-7433).

    Parameters
    ----------
    beam_fit_params_point_spread_function : xarray.DataArray
        Gaussian fits of the point spread function, dims ``(time, frequency,
        polarization, beam_params_label)``, ``[major, minor, position angle]``
        with the axes as FWHM in rad; one time, Stokes I among the
        polarizations.
    first_frequency : float
        Frequency of the window's first channel (Hz), at which findContinuum
        reads the beam.

    Returns
    -------
    max_baseline : float
        ``2.99792458e8 / ((first_frequency (major minor)**0.5) /
        206264.80624709636)`` (m) in findContinuum's order of operations; 0.0
        unless the major axis is positive.
    beam_major_arcsec, beam_minor_arcsec : float
        Medians over the channels of the Stokes I axes, converted to arcsec
        after widening to float64.

    Raises
    ------
    ValueError
        Unless the beam fits have one time and Stokes I.
    """
    beams = beam_fit_params_point_spread_function
    if beams.sizes["time"] != 1 or "I" not in beams.polarization.values:
        raise ValueError(
            "findContinuum reads the Stokes I beams of one time; the beam fits "
            f"have {beams.sizes['time']} times and the polarizations "
            f"{beams.polarization.values.tolist()}"
        )
    values = np.asarray(
        beams.isel(time=0)
        .sel(polarization="I")
        .transpose("frequency", "beam_params_label")
        .values,
        dtype=np.float64,
    )
    beam_major = np.median(np.degrees(values[:, 0]) * 3600.0)
    beam_minor = np.median(np.degrees(values[:, 1]) * 3600.0)
    frequency_ghz = first_frequency * 1e-9
    # The power of a NumPy scalar is C pow, as findContinuum's, where an array
    # takes sqrt; its division gives inf, not an error, for a zero minor axis.
    max_baseline = (
        SPEED_OF_LIGHT
        / ((1e9 * frequency_ghz * (beam_major * beam_minor) ** 0.5) / ARCSEC_PER_RAD)
        if beam_major > 0
        else 0.0
    )
    return float(max_baseline), float(beam_major), float(beam_minor)


@shares_param_docs
def is_time_division_mode(
    channel_width: float, n_channels: int, telescope_name: str = "ALMA"
) -> bool:
    """Whether a spectral window is in time-division mode [findContinuum
    ``tdmSpectrum``] (fC 4441-4463).

    ALMA's rule: channels of at least 7.5 MHz in more than 240 channels, or of
    at least 15 MHz. The VLA's rule: channels of at least 0.97 MHz over at
    least 62 MHz.

    Parameters
    ----------
    channel_width : float
        Channel width (Hz).
    n_channels : int
        Channels of the window.
    telescope_name : str, default "ALMA"
        Telescope whose rule applies: the VLA's for a name containing
        ``"VLA"``, ALMA's otherwise.

    Returns
    -------
    bool
    """
    if "VLA" in telescope_name:
        return bool(channel_width >= 0.97e6 and channel_width * n_channels >= 62e6)
    return bool(
        (channel_width >= 15e6 / 2.0 and n_channels > 240) or channel_width >= 15e6
    )


@shares_param_docs
def continuum_n_bin(
    n_channels: int,
    channel_width: float,
    n_bin: int,
    spectral_dynamic_range_bandwidth: float | None,
) -> int:
    """The channels to average in the mean spectrum [findContinuum ``nbin``]
    (fC 989-1068).

    With a spectral dynamic range bandwidth, the channels it spans, rounded
    half to even and at most ``n_channels // 13``, whatever ``n_bin`` is (fC
    992-998). Without one, ``n_channels // 13`` when fewer than 13 bins of
    ``n_bin`` channels fit in the window (fC 1009, 1062-1064), else ``n_bin``.

    Parameters
    ----------
    n_channels : int
        Channels of the window.
    channel_width : float
        Channel width (Hz).
    n_bin : int
        Channels to average in the mean spectrum, as the caller resolved
        them [findContinuum ``nbin``].
    spectral_dynamic_range_bandwidth : float or None
        The requested spectral dynamic range bandwidth (Hz) [findContinuum
        ``spectralDynamicRangeBandWidth``]; ``None``: none.

    Returns
    -------
    int
        Channels to average, 0 or 1 for no averaging.
    """
    if spectral_dynamic_range_bandwidth is not None:
        # Parsed from its text and taken through GHz, as findContinuum parses
        # a frequency argument; a value up to 1000 counts as GHz (fC 9300-9343).
        bandwidth = float(str(spectral_dynamic_range_bandwidth))
        if bandwidth > 1000:
            bandwidth *= 1e-9
        return int(min(n_channels // MIN_BINS, round(bandwidth * 1e9 / channel_width)))
    if n_channels / n_bin < MIN_BINS:
        return n_channels // MIN_BINS
    return int(n_bin)


def rounds_after_original(
    is_time_division_mode_for_telescope: bool, max_baseline: float
) -> int:
    """The rounds that may follow the original round [findContinuum
    ``amendMaskIterations``] (fC 1070-1080).

    Parameters
    ----------
    is_time_division_mode_for_telescope : bool
        Whether the window is in time-division mode by its telescope's rule.
    max_baseline : float
        Longest baseline (m).

    Returns
    -------
    int
        0 for a window in time-division mode with baselines longer than
        400 m, else 4.
    """
    if (
        is_time_division_mode_for_telescope
        and max_baseline > TIME_DIVISION_MODE_MAX_BASELINE
    ):
        return 0
    return ROUNDS_AFTER_ORIGINAL


@shares_param_docs
def atmosphere_variation(
    first_frequency: float,
    last_frequency: float,
    n_channels: int,
    atmospheric_transmission: np.ndarray,
    atmospheric_sky_temperature: np.ndarray,
) -> tuple[float, float]:
    """The variation of the atmosphere across a spectral window [findContinuum
    ``atmosphereVariation``] (fC 4465-4488).

    Each spectrum loses the line of findContinuum's fit over the linear axis
    in GHz (fC 8794, 9366-9391) and keeps the mean of its original values.

    Parameters
    ----------
    first_frequency, last_frequency : float
        Frequencies of the first and last channels of the window's linear axis
        (Hz).
    n_channels : int
        Channels of the window.
    atmospheric_transmission : numpy.ndarray
        Atmospheric transmission (0 to 1) at every channel of the spectral
        window.
    atmospheric_sky_temperature : numpy.ndarray
        Sky brightness temperature (K) at every channel of the spectral
        window.

    Returns
    -------
    transmission_variation : float
        ``(max - min) / mean`` of the detrended transmission.
    sky_temperature_std : float
        Standard deviation (``ddof=0``) of the detrended sky temperature (K).
    """
    frequency_ghz = np.linspace(
        first_frequency * 1e-9, last_frequency * 1e-9, n_channels
    )

    def detrended(values):
        values = np.asarray(values, dtype=np.float64)
        slope, intercept = line_fit(frequency_ghz, values, values * 0.001)
        return values - (frequency_ghz * slope + intercept) + np.mean(values)

    transmission = detrended(atmospheric_transmission)
    return (
        float((np.max(transmission) - np.min(transmission)) / np.mean(transmission)),
        float(np.std(detrended(atmospheric_sky_temperature))),
    )


def is_bad_atmosphere(
    transmission_variation: float, sky_temperature_std: float
) -> bool:
    """Whether the atmosphere varies enough across the window for its spectrum
    to be normalized by the noise [findContinuum ``badAtmosphere``,
    ``normalizeByMAD``] (fC 1244-1254).

    Parameters
    ----------
    transmission_variation : float
        See :func:`atmosphere_variation`.
    sky_temperature_std : float
        See :func:`atmosphere_variation` (K).

    Returns
    -------
    bool
        ``transmission_variation > 0.40 or sky_temperature_std > 5.0``.
    """
    return bool(
        transmission_variation > TRANSMISSION_VARIATION_THRESHOLD
        or sky_temperature_std > SKY_TEMPERATURE_STD_THRESHOLD
    )


@shares_param_docs
def initial_moment_difference_level(
    rounds_enabled: bool, max_baseline: float, single_continuum: bool
) -> float:
    """The level of the moment difference at the start of the rounds
    [findContinuum ``momDiffLevel``] (fC 1364-1370).

    Parameters
    ----------
    rounds_enabled : bool
        Whether rounds follow the original round.
    max_baseline : float
        Longest baseline (m).
    single_continuum : bool
        The spectral window is a single-continuum setup [findContinuum
        ``singleContinuum``].

    Returns
    -------
    float
        8.0, raised by 3.5 for a single-continuum window with rounds and
        baselines up to 60 m (the 7 m array).
    """
    if rounds_enabled and max_baseline <= ACA_MAX_BASELINE and single_continuum:
        return MOMENT_DIFFERENCE_LEVEL + 3.5
    return MOMENT_DIFFERENCE_LEVEL


def n_baseline_channels(n_channels: int) -> int:
    """The channels of the baseline of a channel selection [findContinuum
    ``nBaselineChannels``] (fC 5092-5097).

    Parameters
    ----------
    n_channels : int
        Channels of the window.

    Returns
    -------
    int
        ``0.19 n_channels`` rounded half up.
    """
    return int(round_half_up(BASELINE_CHANNEL_FRACTION * n_channels))


def primary_beam_mean_channels(n_channels: int) -> tuple[int, ...]:
    """The channels whose primary beams are averaged (fC 7793-7796).

    Parameters
    ----------
    n_channels : int
        Channels of the window.

    Returns
    -------
    tuple of int
        Every channel of a window of fewer than 16 channels, else ``n // 8``,
        ``n // 4``, ``n // 2``, ``3 n // 4`` and ``7 n // 8``.
    """
    if n_channels < 16:
        return tuple(range(n_channels))
    return (
        n_channels // 8,
        n_channels // 4,
        n_channels // 2,
        3 * n_channels // 4,
        7 * n_channels // 8,
    )


@shares_param_docs
def find_continuum_settings(
    frequency: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    rest_frequency: float,
    beam_fit_params_point_spread_function: xr.DataArray,
    atmospheric_transmission: np.ndarray,
    atmospheric_sky_temperature: np.ndarray,
    telescope_name: str,
    single_continuum: bool,
    n_bin: int,
    spectral_dynamic_range_bandwidth: float | None,
) -> FindContinuumSettings:
    """findContinuum's settings of a spectral window, from its metadata.

    Parameters
    ----------
    frequency : numpy.ndarray
        Frequencies of the window's channels (Hz), one linear run.
    l, m : numpy.ndarray
        Image coordinates (rad), at least two pixels each.
    rest_frequency : float
        Rest frequency of the cube (Hz), of the radio velocity that weights
        moment 0.
    beam_fit_params_point_spread_function : xarray.DataArray
        Gaussian fits of the point spread function in the window's channels,
        dims ``(time, frequency, polarization, beam_params_label)``, ``[major,
        minor, position angle]`` with the axes as FWHM in rad; one time, Stokes
        I among the polarizations.
    atmospheric_transmission : numpy.ndarray
        Atmospheric transmission (0 to 1) at every channel of the spectral
        window.
    atmospheric_sky_temperature : numpy.ndarray
        Sky brightness temperature (K) at every channel of the spectral
        window.
    telescope_name : str
        Telescope, ``"ALMA"`` or ``"VLA"``, whose time-division-mode rule sets
        the number of rounds; every other step takes ALMA's rule, as
        findContinuum does.
    single_continuum : bool
        The spectral window is a single-continuum setup [findContinuum
        ``singleContinuum``].
    n_bin : int
        Channels to average in the mean spectrum, as the caller resolved
        them [findContinuum ``nbin``].
    spectral_dynamic_range_bandwidth : float or None
        The requested spectral dynamic range bandwidth (Hz) [findContinuum
        ``spectralDynamicRangeBandWidth``]; ``None``: none.

    Returns
    -------
    FindContinuumSettings

    Raises
    ------
    ValueError
        Unless the beam fits have one time and Stokes I.
    """
    n_channels, first_frequency, last_frequency, channel_increment = (
        linear_spectral_axis(frequency)
    )
    channel_width = abs(channel_increment)
    cell_l_arcsec, cell_m_arcsec = pixel_cells_arcsec(l, m)
    max_baseline, beam_major_arcsec, beam_minor_arcsec = max_baseline_from_beam_fits(
        beam_fit_params_point_spread_function, first_frequency
    )
    is_time_division_mode_for_telescope = is_time_division_mode(
        channel_width, n_channels, telescope_name
    )
    n_rounds_after_original = rounds_after_original(
        is_time_division_mode_for_telescope, max_baseline
    )
    rounds_enabled = n_rounds_after_original != 0
    transmission_variation, sky_temperature_std = atmosphere_variation(
        first_frequency,
        last_frequency,
        n_channels,
        atmospheric_transmission,
        atmospheric_sky_temperature,
    )
    return FindContinuumSettings(
        n_channels=n_channels,
        first_frequency=first_frequency,
        last_frequency=last_frequency,
        channel_increment=channel_increment,
        channel_width=channel_width,
        mean_frequency=float(np.mean([first_frequency, last_frequency])),
        rest_frequency=float(rest_frequency),
        moment_channel_weights=moment_channel_weights(
            first_frequency, channel_increment, n_channels, rest_frequency
        ),
        n_l=len(l),
        n_m=len(m),
        cell_l_arcsec=cell_l_arcsec,
        cell_m_arcsec=cell_m_arcsec,
        beam_major_arcsec=beam_major_arcsec,
        beam_minor_arcsec=beam_minor_arcsec,
        max_baseline=max_baseline,
        is_time_division_mode=is_time_division_mode(channel_width, n_channels),
        is_time_division_mode_for_telescope=is_time_division_mode_for_telescope,
        n_bin=continuum_n_bin(
            n_channels, channel_width, n_bin, spectral_dynamic_range_bandwidth
        ),
        n_rounds_after_original=n_rounds_after_original,
        rounds_enabled=rounds_enabled,
        transmission_variation=transmission_variation,
        sky_temperature_std=sky_temperature_std,
        bad_atmosphere=is_bad_atmosphere(transmission_variation, sky_temperature_std),
        moment_difference_level=initial_moment_difference_level(
            rounds_enabled, max_baseline, single_continuum
        ),
        n_baseline_channels=n_baseline_channels(n_channels),
        single_continuum=single_continuum,
        primary_beam_mean_channels=primary_beam_mean_channels(n_channels),
    )
