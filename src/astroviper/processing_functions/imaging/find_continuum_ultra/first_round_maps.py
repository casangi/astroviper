"""The first round of find_continuum_ultra on two-dimensional maps.

The mean primary beam, the edge samples and annulus levels of a primary-beam
image, and the joint mask of emission that the first round makes from the
integrated and maximum moments of every channel [findContinuum
``meanSpectrumFromMom0Mom8JointMask`` without a given joint mask]. Maps and
regions are image-shaped (``time, frequency, polarization, l, m`` with one
time, frequency and polarization), maps float64 NaN where invalid, regions
bool. A joint mask holds ``JOINT_MASK`` and ``JOINT_MASK_FOOTPRINT``: outside
the footprint a pixel is neither in the mask nor outside it, and
``JOINT_MASK`` is False there. ``fC`` line numbers are those of the pipeline's
``findContinuum.py`` (pipeline 2026.3.0.7).
"""

import numpy as np
import scipy.ndimage
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
    SPECTRUM_PRIMARY_BEAM_LEVEL,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    chauvenet_statistics,
    median_absolute_deviation,
    one_event_sigma,
    round_to_decimals,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
    ACA_MAX_BASELINE,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    AnnulusLevels,
    BeamGeometry,
    EdgeSamples,
    FirstRoundStatistics,
)

MAP_DIMS = ("time", "frequency", "polarization", "l", "m")
# The edge lines of a plane: l = 0, l = n_l - 1 (along m), m = 0, m = n_m - 1.
EDGE_LINES = ("first_l", "last_l", "first_m", "last_m")
INTEGRATED_MIN_SNR = 5  # fC 436, 477
RAISED_INTEGRATED_MIN_SNR = 8  # fC 7897
INTEGRATED_PEAK_OVER_MAD_LIMIT = 90  # fC 7896
MAXIMUM_MIN_SNR = 4  # fC 435, 477
PHASE_2_SNR = 23  # fC 477
MIN_PIXELS_IN_JOINT_MASK = 3  # fC 476
MAD_TO_SIGMA = 1.4826  # of the first round's thresholds (fC 7942, 7986, 8076, 8115)
MIN_BEAM_FRACTION = 0.3  # fC 447
MIN_PRUNE_SIZE = 4  # fC 12023, 12047


def add_primary_beam_planes(
    primary_beam_sum: np.ndarray,
    valid_count: np.ndarray,
    primary_beam: np.ndarray,
    limit_mask: np.ndarray,
) -> None:
    """Add primary-beam planes to the sum and count of the mean primary beam,
    in place (fC 7790-7798).

    A pixel of a plane is added where the primary beam is finite and inside
    the primary-beam limit. The float64 sum of at most 15 float32 values in
    [0.2, 1] is exact, so the planes may come in any order.

    Parameters
    ----------
    primary_beam_sum : numpy.ndarray
        float64 ``(time, 1, polarization, l, m)``, updated in place.
    valid_count : numpy.ndarray
        Integer, same shape: the planes added at each pixel, updated in place.
    primary_beam : numpy.ndarray
        ``(time, k, polarization, l, m)``, k planes of the primary beam.
    limit_mask : numpy.ndarray
        Boolean, same shape as ``primary_beam``: the pixels not below the
        primary-beam limit.
    """
    for index in range(primary_beam.shape[1]):
        plane = primary_beam[:, index : index + 1]
        valid = np.isfinite(plane) & limit_mask[:, index : index + 1]
        np.add(primary_beam_sum, plane, out=primary_beam_sum, where=valid)
        valid_count += valid


def primary_beam_mean(
    primary_beam_sum: np.ndarray, valid_count: np.ndarray, map_coords: xr.Dataset
) -> xr.DataArray:
    """The mean primary beam of the summed planes [findContinuum ``pbmom``]
    (fC 7790-7798).

    Parameters
    ----------
    primary_beam_sum, valid_count : numpy.ndarray
        ``(time, 1, polarization, l, m)`` sum and count of
        :func:`add_primary_beam_planes`.
    map_coords : xarray.Dataset
        Coordinates of the maps (frequency of length 1).

    Returns
    -------
    xarray.DataArray
        ``PRIMARY_BEAM_MEAN``, float64, NaN where no plane was added.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = primary_beam_sum / valid_count
    mean[valid_count == 0] = np.nan
    return map_coords.assign(PRIMARY_BEAM_MEAN=(MAP_DIMS, mean))["PRIMARY_BEAM_MEAN"]


def plane_edge_samples(edge_lines: dict, edge_lines_valid: dict) -> EdgeSamples:
    """The corners, edge midpoints and edge maxima of a primary-beam plane
    [findContinuum ``cornersEmptyEdgesNotEmpty``] (fC 8341-8368, 8389-8402).

    Parameters
    ----------
    edge_lines : dict
        ``"first_l"`` and ``"last_l"`` (the lines l = 0 and l = n_l - 1, along
        m), ``"first_m"`` and ``"last_m"`` (the lines m = 0 and m = n_m - 1,
        along l) of the plane, any shape that ravels to the line.
    edge_lines_valid : dict
        The same keys: where each line is inside the primary-beam limit; a
        pixel is valid there if also finite.

    Returns
    -------
    EdgeSamples
        Whether no corner pixel and every edge midpoint is valid, and the
        largest valid value of the four lines, NaN if a line has no valid
        pixel.
    """
    lines = {key: np.ravel(edge_lines[key]) for key in EDGE_LINES}
    valid = {
        key: np.ravel(edge_lines_valid[key]) & np.isfinite(lines[key])
        for key in EDGE_LINES
    }
    n_l, n_m = lines["first_m"].size, lines["first_l"].size
    corners = (
        valid["first_l"][0],
        valid["first_l"][-1],
        valid["last_l"][0],
        valid["last_l"][-1],
    )
    midpoints = (
        valid["first_l"][n_m // 2],
        valid["last_l"][n_m // 2],
        valid["first_m"][n_l // 2],
        valid["last_m"][n_l // 2],
    )
    maxima = [
        float(lines[key][valid[key]].max()) for key in EDGE_LINES if valid[key].any()
    ]
    return EdgeSamples(
        corners_empty_edges_valid=not any(corners) and all(midpoints),
        edge_maximum=max(maxima) if len(maxima) == len(EDGE_LINES) else np.nan,
    )


def map_edge_samples(values: xr.DataArray) -> EdgeSamples:
    """:func:`plane_edge_samples` of a map, valid where finite.

    Parameters
    ----------
    values : xarray.DataArray
        A map, such as ``PRIMARY_BEAM_MEAN`` (its single plane is the middle
        plane of fC 8387).

    Returns
    -------
    EdgeSamples

    Raises
    ------
    ValueError
        Unless ``values`` has one time, frequency and polarization.
    """
    plane = _plane(values)
    lines = {
        "first_l": plane[0],
        "last_l": plane[-1],
        "first_m": plane[:, 0],
        "last_m": plane[:, -1],
    }
    return plane_edge_samples(
        lines, {key: np.isfinite(line) for key, line in lines.items()}
    )


def annulus_levels(minimum: float, edge_samples: EdgeSamples) -> AnnulusLevels:
    """The primary-beam levels of the outer annulus of a primary-beam image
    [findContinuum ``findOuterAnnulusForPBCube``] (fC 8370-8411).

    The higher level is the response of a Gaussian beam of FWHM 1 at the
    radius of a level divided by 1.15 (fC 9596-9620): of the largest edge
    value when the minimum exceeds 0.22 or the corners are empty and the edge
    midpoints valid (fC 8388), else of the minimum.

    Parameters
    ----------
    minimum : float
        Minimum of the image's valid values.
    edge_samples : EdgeSamples
        Of the image's middle plane (:func:`plane_edge_samples`).

    Returns
    -------
    AnnulusLevels
        ``max(0.21, minimum)`` and the higher level.

    Raises
    ------
    ValueError
        If the edge maximum is needed and an edge line has no valid pixel, or
        the level is not in (0, 1] (findContinuum raises at fC 8398 and after
        9603-9605).
    """
    minimum = float(minimum)
    if minimum > 0.22 or edge_samples.corners_empty_edges_valid:
        level = edge_samples.edge_maximum
        if np.isnan(level):
            raise ValueError(
                "An edge line of the middle primary-beam plane has no valid pixel"
            )
    else:
        level = minimum
    if level <= 0 or level > 1:
        raise ValueError(
            f"The annulus level {level} of the primary beam is not in (0, 1]"
        )
    radius = (1.0 / 2.3548) * (-2 * np.log(level)) ** 0.5 / 1.15
    sigma = 1 / 2.3548
    higher = np.exp(-((radius / sigma) ** 2) * 0.5)
    return AnnulusLevels(
        lower=float(np.max([SPECTRUM_PRIMARY_BEAM_LEVEL, minimum])),
        higher=float(higher),
    )


def first_round_joint_mask(
    sky_moment_integrated: xr.DataArray,
    sky_moment_maximum: xr.DataArray,
    primary_beam_mean: xr.DataArray,
    n_valid_voxels: int,
    beam_geometry: BeamGeometry,
) -> tuple[xr.Dataset, FirstRoundStatistics]:
    """The joint mask of the first round and its statistics
    [findContinuum ``meanSpectrumFromMom0Mom8JointMask``] (fC 7889-8186).

    The integrated and maximum moments are thresholded at a Chauvenet-clipped
    median plus a sigma times 1.4826 times the clipped median absolute
    deviation, the levels written with 9 decimals; the union of the two masks
    is pruned of small regions. With a peak signal to noise above 23, phase 2
    thresholds again with the statistics outside that mask, one-sided, the
    levels written with 6 decimals. The footprint is where the mean primary
    beam is valid and positive.

    Parameters
    ----------
    sky_moment_integrated, sky_moment_maximum : xarray.DataArray
        ``SKY_MOMENT_INTEGRATED`` and ``SKY_MOMENT_MAXIMUM`` of every channel
        (exact zeros left out), NaN where no channel contributed.
    primary_beam_mean : xarray.DataArray
        ``PRIMARY_BEAM_MEAN``.
    n_valid_voxels : int
        Valid voxels of the window's sky, zeros included.
    beam_geometry : BeamGeometry
        Of the window: the size of the regions pruned and the ACA retry.

    Returns
    -------
    joint_mask : xarray.Dataset
        ``JOINT_MASK`` and ``JOINT_MASK_FOOTPRINT``; attributes ``kind``
        (``"phase_1"`` or ``"phase_2"``) and ``primary_beam_based`` (False).
    statistics : FirstRoundStatistics
        ``needs_primary_beam_based_mask`` says that the joint mask is to be
        replaced by :func:`primary_beam_based_joint_mask` (fC 8187).

    Raises
    ------
    ValueError
        Unless the maps have one time, frequency and polarization; if a moment
        map has no valid pixel, the mean primary beam no pixel in the
        footprint, a phase-2 region outside the joint mask no valid pixel, or
        a threshold cannot rise above a map's minimum (findContinuum raises at
        fC 7892, 8046, 8075, 8108, 8167 and 8169, and loops forever at
        7955-7959 and 8013-8017).
    """
    integrated = _plane(sky_moment_integrated).astype(np.float64, copy=False)
    maximum = _plane(sky_moment_maximum).astype(np.float64, copy=False)
    beam = _plane(primary_beam_mean).astype(np.float64, copy=False)
    integrated_valid, maximum_valid = np.isfinite(integrated), np.isfinite(maximum)
    if not (integrated_valid.any() and maximum_valid.any()):
        raise ValueError("A moment map of the first round has no valid pixel")
    footprint = np.isfinite(beam) & (beam > 0)
    beam_above = footprint & (beam > SPECTRUM_PRIMARY_BEAM_LEVEL)
    integrated_values = integrated[integrated_valid]
    maximum_values = maximum[maximum_valid]
    integrated_peak, maximum_peak = integrated_values.max(), maximum_values.max()
    integrated_peak_over_mad = _ratio(
        integrated_peak, median_absolute_deviation(integrated_values)
    )
    maximum_peak_over_mad = _ratio(
        maximum_peak, median_absolute_deviation(maximum_values)
    )
    integrated_min_snr = (
        RAISED_INTEGRATED_MIN_SNR
        if integrated_peak_over_mad > INTEGRATED_PEAK_OVER_MAD_LIMIT
        else INTEGRATED_MIN_SNR
    )

    integrated_median, integrated_mad, _ = chauvenet_statistics(integrated_values)
    integrated_snr = _ratio(integrated_peak, integrated_mad)
    sigma = (
        np.max([INTEGRATED_MIN_SNR, one_event_sigma(n_valid_voxels, 1)])
        if integrated_min_snr == INTEGRATED_MIN_SNR
        else integrated_min_snr
    )
    threshold = _raised_threshold(
        sigma,
        integrated_mad * MAD_TO_SIGMA,
        integrated_median,
        integrated_values.min(),
        integrated_peak,
    )
    integrated_level = round_to_decimals(threshold, 9)
    integrated_mask = (integrated > integrated_level) | (integrated < -integrated_level)

    maximum_median, maximum_mad, maximum_deviation = chauvenet_statistics(
        maximum_values
    )
    scaled_mad = maximum_mad * MAD_TO_SIGMA
    if scaled_mad <= 0:
        scaled_mad = maximum_deviation
        maximum_mad = scaled_mad / MAD_TO_SIGMA
    maximum_snr = _ratio(maximum_peak, maximum_mad)
    threshold = _raised_threshold(
        np.max([MAXIMUM_MIN_SNR, one_event_sigma(n_valid_voxels, 1)]),
        scaled_mad,
        maximum_median,
        maximum_values.min(),
        maximum_peak,
    )
    maximum_level = round_to_decimals(threshold, 9)
    maximum_mask = (maximum > maximum_level) | (maximum < -maximum_level)
    n_pixels_maximum_mask = int(np.count_nonzero(maximum_mask & beam_above))

    geometry = (
        beam_geometry.beam_major_arcsec,
        beam_geometry.beam_minor_arcsec,
        beam_geometry.cell_l_arcsec,
        beam_geometry.cell_m_arcsec,
    )
    size = prune_size(*geometry)
    # fC 12047 takes (pixels per beam * 0.3) * 0.5: the same value, as halving
    # is exact.
    aca_size = prune_size(*geometry, MIN_BEAM_FRACTION * 0.5)
    joint, regions_pruned = _pruned(
        integrated_mask | maximum_mask, size, aca_size, beam_geometry.max_baseline
    )
    if not footprint.any():
        raise ValueError("The mean primary beam has no valid positive pixel")
    pixels_in_mask = bool((joint & footprint).any())
    kind, thresholds = "phase_1", (integrated_level, maximum_level)

    if np.max([integrated_snr, maximum_snr]) > PHASE_2_SNR and pixels_in_mask:
        outside = footprint & ~joint
        median, mad, _ = chauvenet_statistics(
            integrated[_outside(integrated_valid, outside)]
        )
        sigma = (
            np.max([INTEGRATED_MIN_SNR - 1, one_event_sigma(n_valid_voxels, 0.5)])
            if integrated_min_snr == INTEGRATED_MIN_SNR
            else integrated_min_snr - 1
        )
        integrated_level_2 = round_to_decimals(sigma * (mad * MAD_TO_SIGMA) + median, 6)
        median, mad, _ = chauvenet_statistics(maximum[_outside(maximum_valid, outside)])
        sigma = np.max([MAXIMUM_MIN_SNR, one_event_sigma(n_valid_voxels, 0.5)])
        maximum_level_2 = round_to_decimals(sigma * (mad * MAD_TO_SIGMA) + median, 6)
        integrated_mask_2 = integrated > integrated_level_2
        maximum_mask_2 = maximum > maximum_level_2
        # A maximum mask holding every pixel is left out of the union (fC 8131-8144).
        union = (
            integrated_mask_2
            if maximum_mask_2.all()
            else integrated_mask_2 | maximum_mask_2
        )
        joint, regions_pruned = _pruned(
            union, size, aca_size, beam_geometry.max_baseline
        )
        pixels_in_mask = bool((joint & footprint).any())
        # fC 8165-8169 take statistics outside the new joint mask, raising without
        # a pixel there.
        outside = footprint & ~joint
        _outside(integrated_valid, outside)
        _outside(maximum_valid, outside)
        kind = "phase_2"
        thresholds += (integrated_level_2, maximum_level_2)

    n_pixels_in_mask = int(np.count_nonzero(joint & beam_above))
    statistics = FirstRoundStatistics(
        integrated_peak=float(integrated_peak),
        integrated_snr=integrated_snr,
        maximum_peak=float(maximum_peak),
        maximum_snr=maximum_snr,
        integrated_peak_over_mad=integrated_peak_over_mad,
        maximum_peak_over_mad=maximum_peak_over_mad,
        integrated_chauvenet_mad=integrated_mad,
        n_pixels_maximum_mask=n_pixels_maximum_mask,
        n_pixels_in_mask=n_pixels_in_mask,
        regions_pruned=regions_pruned,
        prune_size=size,
        thresholds=thresholds,
        needs_primary_beam_based_mask=not pixels_in_mask
        or n_pixels_in_mask < MIN_PIXELS_IN_JOINT_MASK,
    )
    joint_mask = _joint_mask(
        joint, footprint, sky_moment_integrated.coords, kind, primary_beam_based=False
    )
    return joint_mask, statistics


def primary_beam_based_joint_mask(
    primary_beam_plane: xr.DataArray,
    footprint: xr.DataArray,
    higher_level: float,
    kind: str,
) -> xr.Dataset:
    """The joint mask of the primary beam above the annulus, used when a joint
    mask holds too few pixels (fC 8197-8203, 8264-8271).

    Parameters
    ----------
    primary_beam_plane : xarray.DataArray
        The primary beam of the window's channel 1, NaN where invalid.
    footprint : xarray.DataArray
        Where the mean primary beam is valid and positive (fC 8203, 8271).
    higher_level : float
        The higher annulus level of the primary-beam cube, written with 6
        decimals (fC 8200, 8267).
    kind : str
        The kind of the joint mask replaced, kept.

    Returns
    -------
    xarray.Dataset
        ``JOINT_MASK`` (valid primary beam above the level, in the footprint)
        and ``JOINT_MASK_FOOTPRINT``; attributes ``kind`` and
        ``primary_beam_based`` (True).

    Raises
    ------
    ValueError
        Unless the maps have one time, frequency and polarization.
    """
    plane = _plane(primary_beam_plane).astype(np.float64, copy=False)
    joint = np.isfinite(plane) & (plane > round_to_decimals(higher_level, 6))
    return _joint_mask(
        joint, _plane(footprint), footprint.coords, kind, primary_beam_based=True
    )


def prune_size(
    beam_major_arcsec: float,
    beam_minor_arcsec: float,
    cell_l_arcsec: float,
    cell_m_arcsec: float,
    fraction: float = MIN_BEAM_FRACTION,
) -> int:
    """The size of the smallest region a pruning keeps (fC 12021-12023).

    Parameters
    ----------
    beam_major_arcsec, beam_minor_arcsec : float
        Axes of the beam (arcsec).
    cell_l_arcsec, cell_m_arcsec : float
        Pixel increments (arcsec, signed).
    fraction : float, default 0.3
        Of the pixels in the ellipse of the beam's axes.

    Returns
    -------
    int
        At least 4.
    """
    pixels_per_beam = (
        beam_major_arcsec
        * beam_minor_arcsec
        * np.pi
        / 4.0
        / np.abs(cell_l_arcsec * cell_m_arcsec)
    )
    return max(MIN_PRUNE_SIZE, int(pixels_per_beam * fraction))


def prune_small_regions(mask: np.ndarray, prune_size: int) -> tuple[np.ndarray, int]:
    """``mask`` without its 4-connected regions of fewer than ``prune_size``
    pixels (fC 12033-12037).

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean ``(..., l, m)``; regions connect along ``l`` and ``m`` only.
    prune_size : int
        Pixels of the smallest region kept.

    Returns
    -------
    pruned : numpy.ndarray
        Boolean, shaped as ``mask``.
    n_pruned : int
        Regions removed.
    """
    structure = np.zeros((3,) * mask.ndim, dtype=bool)
    structure[(1,) * (mask.ndim - 2)] = scipy.ndimage.generate_binary_structure(2, 1)
    labels, n_regions = scipy.ndimage.label(mask, structure)
    keep = np.bincount(labels[mask], minlength=n_regions + 1) >= prune_size
    keep[0] = False
    return keep[labels], n_regions - int(np.count_nonzero(keep))


def amended_round_statistics(
    first_round_statistics: FirstRoundStatistics,
) -> FirstRoundStatistics:
    """The statistics of round ``amended_mask``, which makes no map and no
    mask (fC 8205-8246).

    Parameters
    ----------
    first_round_statistics : FirstRoundStatistics
        Of :func:`first_round_joint_mask`.

    Returns
    -------
    FirstRoundStatistics
        The maximum SNR divided by the integrated moment's Chauvenet median
        absolute deviation (fC 8214), no region pruned, no threshold, no
        primary-beam-based mask (fC 7901, 7920); the other values of the first
        round.
    """
    return first_round_statistics._replace(
        maximum_snr=_ratio(
            first_round_statistics.maximum_peak,
            first_round_statistics.integrated_chauvenet_mad,
        ),
        regions_pruned=0,
        thresholds=(),
        needs_primary_beam_based_mask=False,
    )


def count_in_primary_beam(
    mask: xr.DataArray,
    mask_valid: xr.DataArray,
    primary_beam_above_spectrum_level: xr.DataArray,
) -> int:
    """The pixels of a mask where the mean primary beam exceeds 0.21
    [findContinuum ``countPixelsAboveZero``] (fC 8186, 8257, 2108-2109).

    Parameters
    ----------
    mask, mask_valid : xarray.DataArray
        Boolean region and where it is valid (its footprint).
    primary_beam_above_spectrum_level : xarray.DataArray
        Boolean region ``PRIMARY_BEAM_ABOVE_SPECTRUM_LEVEL``.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        Unless the regions have one time, frequency and polarization.
    """
    return int(
        np.count_nonzero(
            _plane(mask)
            & _plane(mask_valid)
            & _plane(primary_beam_above_spectrum_level)
        )
    )


def _plane(values: xr.DataArray) -> np.ndarray:
    """The ``(l, m)`` values of a map or region."""
    array = values.transpose(*MAP_DIMS).values
    if array.shape[:3] != (1, 1, 1):
        raise ValueError(
            "find_continuum_ultra's maps have one time, frequency and polarization, "
            f"not {dict(values.sizes)}"
        )
    return array[0, 0, 0]


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` as NumPy divides: inf or NaN for zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(numerator) / denominator)


def _raised_threshold(
    sigma: float, scaled_mad: float, median: float, minimum: float, peak: float
) -> float:
    """``sigma * scaled_mad + median``, raised a sigma at a time while the
    minimum is not below it (fC 7951-7959, 8010-8017)."""
    threshold = sigma * scaled_mad + median
    while minimum >= threshold and threshold < peak:
        if scaled_mad == 0:
            raise ValueError(
                "The first-round threshold cannot rise above the minimum of a "
                "moment map whose median absolute deviation is 0"
            )
        sigma += 1
        threshold = sigma * scaled_mad + median
    return threshold


def _pruned(
    mask: np.ndarray, size: int, aca_size: int, max_baseline: float
) -> tuple[np.ndarray, int]:
    """The pruned joint mask and the regions pruned; with nothing left of an
    ACA mask, pruned again with the smaller size (fC 12033-12060)."""
    pruned, n_pruned = prune_small_regions(mask, size)
    if (
        n_pruned > 0
        and not pruned.any()
        and max_baseline <= ACA_MAX_BASELINE
        and aca_size != size
    ):
        pruned, n_pruned = prune_small_regions(mask, aca_size)
    return pruned, n_pruned


def _outside(valid: np.ndarray, outside: np.ndarray) -> np.ndarray:
    """The valid pixels of a moment map in the footprint outside the joint
    mask (fC 8071-8073, 8105-8107, 8165-8169)."""
    selected = valid & outside
    if not selected.any():
        raise ValueError(
            "No valid pixel of a moment map lies in the footprint outside the "
            "joint mask"
        )
    return selected


def _joint_mask(
    joint: np.ndarray,
    footprint: np.ndarray,
    coords: xr.Coordinates,
    kind: str,
    primary_beam_based: bool,
) -> xr.Dataset:
    return xr.Dataset(
        {
            "JOINT_MASK": (MAP_DIMS, (joint & footprint)[None, None, None]),
            "JOINT_MASK_FOOTPRINT": (MAP_DIMS, footprint[None, None, None]),
        },
        coords=coords,
        attrs={"kind": kind, "primary_beam_based": primary_beam_based},
    )
