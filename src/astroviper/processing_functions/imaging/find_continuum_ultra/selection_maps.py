"""findContinuum's maps of a channel selection and their statistics.

The regions of round ``original``, the maximum and moment-difference maps of a
channel selection, their signal-to-noise ratios, the noise of the cube outside
the joint mask, and the pixel counts and fractions that the decisions after
round ``original`` and the difference test of round ``amended_mask`` read. Maps
and regions are image-shaped (``time, frequency, polarization, l, m``) with one
time, frequency and polarization; maps are float64, NaN where invalid, regions
bool. Statistics stay NumPy float64 scalars, as findContinuum's are, so that a
zero divisor gives ``inf`` or NaN as it does in findContinuum. ``fC`` line
numbers are those of the pipeline's ``findContinuum.py`` (pipeline 2026.3.0.7).
"""

import numpy as np
import scipy.stats
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    MAD_DIVISOR,
    one_event_sigma,
    round_to_decimals,
    round_to_significant_digits,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    AnnulusLevels,
    CubeNoiseStatistics,
    ImageSnr,
    MapStatistics,
    MomentDifferenceCounts,
    SelectionStatistics,
)

SPEED_OF_LIGHT_KM_S = 299792.458  # fC 1757
# The mean primary beam above which the moment difference is valid (fC 1770;
# the log lines fC 2323, 2662 and 2992 say 0.3).
MOMENT_DIFFERENCE_PRIMARY_BEAM_LEVEL = 0.23
# The mean primary beam above which findContinuum counts pixels (fC 7672).
PIXEL_COUNT_PRIMARY_BEAM_LEVEL = 0.21
# The moment-difference level drops by LOWERING_STEP when the moment
# difference's signal-to-noise ratio is below LOWERING_RATIO times that of the
# maximum map outside the joint mask (fC 1915-1920).
LOWERING_RATIO = 0.3
LOWERING_STEP = 0.5
# The second count of a level counts above the level plus LEVEL_STEP
# (fC 1923, 1925).
LEVEL_STEP = 0.5
MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE = 11.5  # fC 1371
MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE_ONLY_EXTRA_MASK = 11.0  # fC 1372
CUBE_LEVEL = 7.5  # fC 1375
CUBE_LEVEL_2 = 7.25  # fC 1376


def _check_planes(*images):
    """Raise unless every map, region or coordinate set given has one time and
    one polarization."""
    for image in images:
        if image is not None and (
            image.sizes.get("time") != 1 or image.sizes.get("polarization") != 1
        ):
            raise ValueError(
                "findContinuum's maps have one time and one polarization (Stokes "
                f"I), not the sizes {dict(image.sizes)}"
            )


def _between(values, lower, higher):
    """Where ``values`` is strictly between ``lower`` and ``higher`` (False
    where NaN)."""
    return (values > lower) & (values < higher)


def round_zero_regions(
    joint_mask: xr.Dataset,
    primary_beam_mean: xr.DataArray,
    cube_annulus_levels: AnnulusLevels,
    mean_annulus_levels: AnnulusLevels,
) -> xr.Dataset:
    """The regions of findContinuum's statistics, made in round ``original``
    from its joint mask (fC 1709-1723, 3721-3728, 1770, 8186, 8043, 8498-8505).

    Parameters
    ----------
    joint_mask : xarray.Dataset
        ``JOINT_MASK`` and ``JOINT_MASK_FOOTPRINT`` (bool) of round
        ``original``, the primary-beam-based mask after a fallback.
    primary_beam_mean : xarray.DataArray
        ``PRIMARY_BEAM_MEAN``, NaN where invalid.
    cube_annulus_levels : AnnulusLevels
        Annulus levels of the primary-beam cube (fC 1717).
    mean_annulus_levels : AnnulusLevels
        Annulus levels of ``primary_beam_mean`` (fC 3721).

    Returns
    -------
    xarray.Dataset
        Bool regions on the coordinates of ``primary_beam_mean``, with ``P``
        the mean primary beam and levels written with ``%f`` unless said:
        ``OUTSIDE_JOINT_MASK`` (in the footprint, not in the mask, fC 1715),
        ``NOISE_ANNULUS`` (outside the joint mask, ``P`` strictly between the
        cube's levels, fC 1721), ``CUBE_NOISE_ANNULUS`` (outside the joint
        mask, ``P`` strictly between its own levels, fC 3726),
        ``MOMENT_DIFFERENCE_REGION`` (``P > 0.23``, fC 1770),
        ``PRIMARY_BEAM_ABOVE_SPECTRUM_LEVEL`` (``P > 0.21``, fC 8186),
        ``ONLY_EXTRA_MASK_NORMALIZATION_REGION`` (``P`` strictly between its
        own levels written with ``%g``, fC 8505) and ``PRIMARY_BEAM_FOOTPRINT``
        (``P > 0``, fC 8043).

    Raises
    ------
    ValueError
        Unless the maps have one time and one polarization.
    """
    _check_planes(joint_mask, primary_beam_mean)
    mean = primary_beam_mean.values
    outside = (
        joint_mask["JOINT_MASK_FOOTPRINT"].values & ~joint_mask["JOINT_MASK"].values
    )
    noise_levels = [round_to_decimals(level, 6) for level in cube_annulus_levels]
    cube_noise_levels = [round_to_decimals(level, 6) for level in mean_annulus_levels]
    normalization_levels = [
        round_to_significant_digits(level, 6) for level in mean_annulus_levels
    ]
    regions = {
        "OUTSIDE_JOINT_MASK": outside,
        "NOISE_ANNULUS": outside & _between(mean, *noise_levels),
        "CUBE_NOISE_ANNULUS": _between(mean, *cube_noise_levels) & outside,
        "MOMENT_DIFFERENCE_REGION": mean > MOMENT_DIFFERENCE_PRIMARY_BEAM_LEVEL,
        "PRIMARY_BEAM_ABOVE_SPECTRUM_LEVEL": mean > PIXEL_COUNT_PRIMARY_BEAM_LEVEL,
        "ONLY_EXTRA_MASK_NORMALIZATION_REGION": _between(mean, *normalization_levels),
        "PRIMARY_BEAM_FOOTPRINT": mean > 0,
    }
    return xr.Dataset(
        {name: (primary_beam_mean.dims, region) for name, region in regions.items()},
        coords=primary_beam_mean.coords,
    )


def selection_maps(
    integral: np.ndarray,
    maximum: np.ndarray,
    n_selected_channels: int,
    channel_width: float,
    mean_frequency: float,
    moment_difference_region: xr.DataArray,
    map_coords: xr.Dataset,
) -> xr.Dataset:
    """The maximum and moment-difference maps of a channel selection
    (fC 1757-1773).

    The moment difference is the maximum less the integral times
    ``1.0/channel_width_km_s/n_selected_channels`` written with ``%f``, where
    ``channel_width_km_s = 299792.458*channel_width/mean_frequency``, valid in
    the moment-difference region. It is computed in the integral's buffer.

    Parameters
    ----------
    integral : numpy.ndarray
        float64 moment 0 of the selected channels, NaN where no channel
        contributed, of the shape of ``moment_difference_region`` or of its
        ``(l, m)``. Overwritten with the moment difference.
    maximum : numpy.ndarray
        Moment 8 of the selected channels, of the same shape, NaN where no
        channel contributed.
    n_selected_channels : int
        The channels of the selection's ranges, a channel counted once per
        range that holds it (fC 1758).
    channel_width : float
        Channel width (Hz).
    mean_frequency : float
        Mean of the window's first and last frequency (Hz).
    moment_difference_region : xarray.DataArray
        ``MOMENT_DIFFERENCE_REGION`` of :func:`round_zero_regions`.
    map_coords : xarray.Dataset
        Coordinates of the maps.

    Returns
    -------
    xarray.Dataset
        ``SKY_MOMENT_MAXIMUM_SELECTION`` (float64) and ``MOMENT_DIFFERENCE``
        (the integral's buffer), NaN where invalid.

    Raises
    ------
    ValueError
        Unless the maps have one time and one polarization.
    """
    _check_planes(moment_difference_region, map_coords)
    shape, dims = moment_difference_region.shape, moment_difference_region.dims
    maximum = np.asarray(maximum, dtype=np.float64).reshape(shape)
    moment_difference = integral.reshape(shape)
    channel_width_km_s = SPEED_OF_LIGHT_KM_S * channel_width / mean_frequency
    moment_difference *= round_to_decimals(
        1.0 / channel_width_km_s / n_selected_channels, 6
    )
    np.subtract(maximum, moment_difference, out=moment_difference)
    moment_difference[~moment_difference_region.values] = np.nan
    return xr.Dataset(
        {
            "SKY_MOMENT_MAXIMUM_SELECTION": (dims, maximum),
            "MOMENT_DIFFERENCE": (dims, moment_difference),
        },
        coords=map_coords.coords,
    )


def map_statistics(values: np.ndarray, region: np.ndarray) -> MapStatistics | None:
    """Statistics of the valid values of a map in a region, as findContinuum
    takes them of an image [CASA ``imstat``].

    Parameters
    ----------
    values : numpy.ndarray
        Map, NaN where invalid.
    region : numpy.ndarray
        Bool region of the shape of ``values``.

    Returns
    -------
    MapStatistics or None
        Of the valid values in the region, gathered in C order: the sum is
        NumPy's pairwise sum of the gather, medians of an even count the mean
        of the two middle values, the standard deviation has ``ddof=1`` (0.0
        for one value). None if no valid value is in the region.
    """
    sample = values[region & np.isfinite(values)].astype(np.float64, copy=False)
    if sample.size == 0:
        return None
    count, total = sample.size, sample.sum()
    maximum, minimum = sample.max(), sample.min()
    standard_deviation = sample.std(ddof=1) if count > 1 else np.float64(0.0)
    median = np.median(sample, overwrite_input=True)
    np.abs(np.subtract(sample, median, out=sample), out=sample)
    return MapStatistics(
        count,
        total,
        maximum,
        minimum,
        median,
        np.median(sample, overwrite_input=True),
        standard_deviation,
    )


def _valid_values(values):
    """The valid values of a map, gathered in C order, in float64."""
    return values[np.isfinite(values)].astype(np.float64, copy=False)


def _region_statistics(values, outside_joint_mask, noise_annulus):
    """The statistics of a map outside the joint mask and in the noise annulus."""
    statistics = (
        map_statistics(values, outside_joint_mask),
        map_statistics(values, noise_annulus),
    )
    if None in statistics:
        raise ValueError(
            "The map has no valid pixel outside the joint mask or none in the "
            "noise annulus; findContinuum's signal-to-noise ratio needs both"
        )
    return statistics


def _snr(peak, outside, annulus):
    """findContinuum's signal-to-noise ratio of a peak from the statistics
    outside the joint mask and in the noise annulus (fC 9661-9689)."""
    mad = outside.median_absolute_deviation
    if annulus.median_absolute_deviation > 0 and mad > 0:
        mad = min(annulus.median_absolute_deviation, mad)
    elif annulus.median_absolute_deviation > 0:
        mad = annulus.median_absolute_deviation
    if mad > 0:
        snr = (peak - annulus.median) / (mad / MAD_DIVISOR)
    else:
        snr = (peak - annulus.median) / outside.standard_deviation
    return ImageSnr(snr, peak, annulus.median, mad / MAD_DIVISOR)


def image_snr(
    values: xr.DataArray, outside_joint_mask: xr.DataArray, noise_annulus: xr.DataArray
) -> ImageSnr:
    """The signal-to-noise ratio of a map's peak [findContinuum ``imageSNR``]
    (fC 9631-9691).

    The peak is the largest valid value of the map, the median that of the
    noise annulus, the median absolute deviation the smaller of the positive
    ones outside the joint mask and in the noise annulus, or the one outside
    the joint mask if neither is positive; a zero deviation makes the standard
    deviation outside the joint mask the divisor.

    Parameters
    ----------
    values : xarray.DataArray
        Map, NaN where invalid.
    outside_joint_mask : xarray.DataArray
        ``OUTSIDE_JOINT_MASK`` of :func:`round_zero_regions`.
    noise_annulus : xarray.DataArray
        ``NOISE_ANNULUS`` of :func:`round_zero_regions`.

    Returns
    -------
    ImageSnr
        ``(peak - median)/(mad/0.6745)``, the peak, the median and
        ``mad/0.6745``.

    Raises
    ------
    ValueError
        If no valid pixel of the map is outside the joint mask or in the noise
        annulus (findContinuum raises), or unless the maps have one time and
        one polarization.
    """
    _check_planes(values, outside_joint_mask, noise_annulus)
    outside, annulus = _region_statistics(
        values.values, outside_joint_mask.values, noise_annulus.values
    )
    return _snr(_valid_values(values.values).max(), outside, annulus)


def image_snr_outside(
    values: xr.DataArray,
    outside_joint_mask: xr.DataArray,
    noise_annulus: xr.DataArray,
    peak_outside: bool = True,
) -> tuple[float, float]:
    """The signal-to-noise ratio of a map's peak outside the joint mask
    [findContinuum ``imageSNRAnnulus``] (fC 9622-9629).

    The median and scaled median absolute deviation are those of
    :func:`image_snr`, without its fallback: a zero deviation divides by zero.

    Parameters
    ----------
    values : xarray.DataArray
        Map, NaN where invalid.
    outside_joint_mask : xarray.DataArray
        ``OUTSIDE_JOINT_MASK`` of :func:`round_zero_regions`.
    noise_annulus : xarray.DataArray
        ``NOISE_ANNULUS`` of :func:`round_zero_regions`.
    peak_outside : bool, default True
        The peak outside the joint mask; False: the largest valid value.

    Returns
    -------
    snr : float
        ``(peak - median)/scaled_mad``.
    peak : float

    Raises
    ------
    ValueError
        As :func:`image_snr`.
    """
    _check_planes(values, outside_joint_mask, noise_annulus)
    outside, annulus = _region_statistics(
        values.values, outside_joint_mask.values, noise_annulus.values
    )
    peak = outside.maximum if peak_outside else _valid_values(values.values).max()
    snr = _snr(peak, outside, annulus)
    return (peak - snr.median) / snr.scaled_mad, peak


def selection_statistics(maps: xr.Dataset, regions: xr.Dataset) -> SelectionStatistics:
    """The statistics of a selection's maximum and moment-difference maps
    (fC 1785-1801, 1850-1857, 1971-1972).

    Each map's statistics outside the joint mask and in the noise annulus are
    taken once for both of its signal-to-noise ratios; findContinuum's repeated
    calls (fC 1900, 1907, 1914) give the same values.

    Parameters
    ----------
    maps : xarray.Dataset
        ``SKY_MOMENT_MAXIMUM_SELECTION`` and ``MOMENT_DIFFERENCE`` of
        :func:`selection_maps`.
    regions : xarray.Dataset
        ``OUTSIDE_JOINT_MASK`` and ``NOISE_ANNULUS`` of
        :func:`round_zero_regions`.

    Returns
    -------
    SelectionStatistics
        Of each map :func:`image_snr`, :func:`image_snr_outside` and the sum of
        its valid values (NumPy's pairwise sum of their C-order gather);
        ``cube_peak``, the maximum map's peak, is the largest valid residual
        value of the selected channels.

    Raises
    ------
    ValueError
        As :func:`image_snr`.
    """
    _check_planes(maps, regions)
    outside_joint_mask = regions["OUTSIDE_JOINT_MASK"].values
    noise_annulus = regions["NOISE_ANNULUS"].values
    values = []
    for name in ("SKY_MOMENT_MAXIMUM_SELECTION", "MOMENT_DIFFERENCE"):
        image = maps[name].values
        outside, annulus = _region_statistics(image, outside_joint_mask, noise_annulus)
        valid = _valid_values(image)
        snr = _snr(valid.max(), outside, annulus)
        values += [
            *snr,
            (outside.maximum - snr.median) / snr.scaled_mad,
            outside.maximum,
            valid.sum(),
        ]
    return SelectionStatistics(*values, cube_peak=values[1])


def original_round_cube_statistics(
    noise_medians: np.ndarray,
    noise_mads: np.ndarray,
    selected_channels: np.ndarray,
    n_valid_voxels: int,
    moment_difference_peak_outside: float,
    moment_difference_median: float,
) -> CubeNoiseStatistics:
    """The noise of the cube outside the joint mask in round ``original``
    [findContinuum ``cubeNoiseLevel``] (fC 1877-1912, 3732-3743).

    Parameters
    ----------
    noise_medians : numpy.ndarray
        One-dimensional: the median of the valid residual values of each
        selected channel in ``CUBE_NOISE_ANNULUS``, NaN for a channel without
        one.
    noise_mads : numpy.ndarray
        The unscaled median absolute deviations of the same values.
    selected_channels : numpy.ndarray
        The channel of each value.
    n_valid_voxels : int
        Valid voxels of the window's residual.
    moment_difference_peak_outside : float
        Peak of the moment difference outside the joint mask.
    moment_difference_median : float
        Median of the moment difference in the noise annulus.

    Returns
    -------
    CubeNoiseStatistics
        ``scipy.stats.scoreatpercentile(noise_mads, 50)/0.6745``, the median
        of the lowest selected channel, the ten-event sigma of the valid voxels
        (fC 1893) and the moment difference's signal-to-noise ratio over the
        cube's scaled deviation (fC 1912).
    """
    cube_scaled_mad_outside = (
        scipy.stats.scoreatpercentile(noise_mads, 50) / MAD_DIVISOR
    )
    return CubeNoiseStatistics(
        cube_scaled_mad_outside,
        noise_medians[np.argmin(selected_channels)],
        one_event_sigma(n_valid_voxels, 10),
        (moment_difference_peak_outside - moment_difference_median)
        / cube_scaled_mad_outside,
    )


def lowered_moment_difference_level(
    moment_difference_level: float,
    moment_difference_snr: float,
    maximum_snr_outside: float,
) -> float:
    """The moment-difference level after round ``original`` (fC 1915-1920).

    Parameters
    ----------
    moment_difference_level : float
        The level before.
    moment_difference_snr : float
        Signal-to-noise ratio of the moment difference's peak.
    maximum_snr_outside : float
        Signal-to-noise ratio of the maximum map's peak outside the joint mask.

    Returns
    -------
    float
        The level less 0.5 if ``maximum_snr_outside`` is positive and the ratio
        of the two signal-to-noise ratios is below 0.3, else the level.
    """
    if (
        maximum_snr_outside > 0
        and moment_difference_snr / maximum_snr_outside < LOWERING_RATIO
    ):
        return moment_difference_level - LOWERING_STEP
    return moment_difference_level


def count_above_level(
    values: xr.DataArray,
    median: float,
    level: float,
    scaled_mad: float,
    region: xr.DataArray | None,
) -> int:
    """The valid pixels of a map above ``median + level*scaled_mad`` written
    with ``%f``, in a region [findContinuum ``computeNpixMom8Median``,
    ``computeNpixMom8MedianBadAtm``, ``computeNpixCubeMedian``]
    (fC 3545-3603).

    Parameters
    ----------
    values : xarray.DataArray
        Map, NaN where invalid.
    median, level, scaled_mad : float
        Of the threshold.
    region : xarray.DataArray or None
        Bool region; None for the whole map.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        Unless the maps have one time and one polarization.
    """
    _check_planes(values, region)
    above = values.values > round_to_decimals(median + level * scaled_mad, 6)
    if region is not None:
        above &= region.values
    return int(np.count_nonzero(above))


def moment_difference_counts(
    moment_difference: xr.DataArray,
    moment_difference_median: float,
    moment_difference_scaled_mad: float,
    cube_scaled_mad_outside: float,
    moment_difference_level: float,
    outside_joint_mask: xr.DataArray,
) -> MomentDifferenceCounts:
    """The pixels of the moment difference above the levels of round
    ``original``'s decisions (fC 1922-1933, 1975-1976).

    Parameters
    ----------
    moment_difference : xarray.DataArray
        ``MOMENT_DIFFERENCE`` of round ``original``.
    moment_difference_median : float
        Its median in the noise annulus.
    moment_difference_scaled_mad : float
        Its scaled median absolute deviation.
    cube_scaled_mad_outside : float
        Scaled median absolute deviation of the cube outside the joint mask.
    moment_difference_level : float
        The level after :func:`lowered_moment_difference_level`.
    outside_joint_mask : xarray.DataArray
        ``OUTSIDE_JOINT_MASK`` of :func:`round_zero_regions`.

    Returns
    -------
    MomentDifferenceCounts
        :func:`count_above_level` outside the joint mask at the level, the
        level plus 0.5, 11.5 and 12.0 times the scaled deviation and at 7.5 and
        7.25 times the cube's, and on the whole map at the level and at 11.0.

    Raises
    ------
    ValueError
        Unless the maps have one time and one polarization.
    """

    def count(level, scaled_mad, region):
        return count_above_level(
            moment_difference, moment_difference_median, level, scaled_mad, region
        )

    mad, outside = moment_difference_scaled_mad, outside_joint_mask
    return MomentDifferenceCounts(
        n_above=count(moment_difference_level, mad, outside),
        n_above_2=count(moment_difference_level + LEVEL_STEP, mad, outside),
        n_above_bad_atmosphere=count(
            MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE, mad, outside
        ),
        n_above_2_bad_atmosphere=count(
            MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE + LEVEL_STEP, mad, outside
        ),
        n_above_cube=count(CUBE_LEVEL, cube_scaled_mad_outside, outside),
        n_above_cube_2=count(CUBE_LEVEL_2, cube_scaled_mad_outside, outside),
        n_above_anywhere=count(moment_difference_level, mad, None),
        n_above_anywhere_bad_atmosphere=count(
            MOMENT_DIFFERENCE_LEVEL_BAD_ATMOSPHERE_ONLY_EXTRA_MASK, mad, None
        ),
    )


def negative_pixel_fraction(
    maximum: xr.DataArray, outside_joint_mask: xr.DataArray
) -> float:
    """The fraction of a maximum map's valid pixels that are negative and
    outside the joint mask (fC 1939-1947).

    Parameters
    ----------
    maximum : xarray.DataArray
        ``SKY_MOMENT_MAXIMUM_SELECTION`` of round ``original``.
    outside_joint_mask : xarray.DataArray
        ``OUTSIDE_JOINT_MASK`` of :func:`round_zero_regions`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the map has no valid pixel (findContinuum raises), or unless the
        maps have one time and one polarization.
    """
    _check_planes(maximum, outside_joint_mask)
    values = maximum.values
    n_valid = np.count_nonzero(np.isfinite(values))
    if n_valid == 0:
        raise ValueError("The maximum map has no valid pixel")
    return np.count_nonzero(outside_joint_mask.values & (values < 0)) / n_valid


def difference_map_snr(
    moment_difference: xr.DataArray,
    reference_moment_difference: xr.DataArray,
    outside_joint_mask: xr.DataArray,
    noise_annulus: xr.DataArray,
) -> float:
    """The signal-to-noise ratio of the peak of the difference of two
    moment-difference maps (fC 2200-2207).

    Parameters
    ----------
    moment_difference : xarray.DataArray
        ``MOMENT_DIFFERENCE`` of round ``amended_mask``.
    reference_moment_difference : xarray.DataArray
        ``MOMENT_DIFFERENCE`` of round ``original``.
    outside_joint_mask : xarray.DataArray
        ``OUTSIDE_JOINT_MASK`` of :func:`round_zero_regions`.
    noise_annulus : xarray.DataArray
        ``NOISE_ANNULUS`` of :func:`round_zero_regions`.

    Returns
    -------
    float
        :func:`image_snr_outside` of ``moment_difference -
        reference_moment_difference`` (valid where both are) with the peak
        anywhere.

    Raises
    ------
    ValueError
        As :func:`image_snr`.
    """
    _check_planes(moment_difference, reference_moment_difference)
    difference = moment_difference.copy(
        data=moment_difference.values - reference_moment_difference.values
    )
    return image_snr_outside(
        difference, outside_joint_mask, noise_annulus, peak_outside=False
    )[0]
