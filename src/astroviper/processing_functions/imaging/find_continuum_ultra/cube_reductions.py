"""Reductions of batches of planes of the residual and primary-beam cubes.

The node tasks of find_continuum_ultra call these functions on batches of
planes ``(time, B, polarization, l, m)`` read from the image store: the
primary-beam masks and the valid voxels, the first round's per-channel rows,
moments 0 and 8 as partials that merge in GraphVIPER's binary-tree order, and
per-channel means, scaled median absolute deviations, medians and median
absolute deviations over a two-dimensional region. Every function processes
each time and polarization, a block of rows at a time, and gathers values in C
order into float64 before a mean or a median, the arithmetic of the simple port
(``find_continuum``). ``fC`` line numbers are those of the pipeline's
``findContinuum.py`` (pipeline 2026.3.0.7).
"""

from collections.abc import Iterable

import numpy as np

from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    AnnulusLevels,
)
from astroviper.utils.param_docs import shares_param_docs

# The primary-beam level of findContinuum's spectrum mask, "pb">0.21 (fC 8453),
# pixel counts (fC 7672) and annulus (fC 8411), which its comments keep equal.
SPECTRUM_PRIMARY_BEAM_LEVEL = 0.21
# casacore's factor from the median absolute deviation to a Gaussian sigma, of
# the scaled-MAD profile.
SCALED_MAD_PROFILE_FACTOR = 1.482602218505602
# Rows of the blocks that planes are reduced in.
_BLOCK_ROWS = 625


@shares_param_docs
def primary_beam_masks(
    primary_beam: np.ndarray,
    primary_beam_limit: float,
    spectrum_level: float = SPECTRUM_PRIMARY_BEAM_LEVEL,
) -> tuple[np.ndarray, np.ndarray]:
    """The primary-beam limit mask and findContinuum's spectrum mask of planes.

    The limit mask is ``~(primary_beam < primary_beam_limit)`` compared in the
    primary beam's dtype, the pixel mask of the residual and primary-beam
    cubes (a NaN primary beam is inside). The spectrum mask is the limit mask
    and ``primary_beam > spectrum_level`` compared in float64 (fC 8453).

    Parameters
    ----------
    primary_beam : numpy.ndarray
        ``(time, frequency, polarization, l, m)``.
    primary_beam_limit : float
        Primary-beam level below which a voxel of the residual and of the
        primary beam is invalid [tclean ``pblimit``].
    spectrum_level : float, default SPECTRUM_PRIMARY_BEAM_LEVEL
        Level of the spectrum mask.

    Returns
    -------
    limit_mask, spectrum_mask : numpy.ndarray
        Boolean, shaped as ``primary_beam``.
    """
    limit_mask = np.less(primary_beam, float(primary_beam_limit))
    np.logical_not(limit_mask, out=limit_mask)
    spectrum_mask = np.greater(primary_beam, np.float64(spectrum_level))
    spectrum_mask &= limit_mask
    return limit_mask, spectrum_mask


def valid_voxels(sky: np.ndarray, limit_mask: np.ndarray) -> np.ndarray:
    """The voxels findContinuum's reductions use: finite and inside the
    primary-beam limit.

    Parameters
    ----------
    sky : numpy.ndarray
        ``(time, frequency, polarization, l, m)``.
    limit_mask : numpy.ndarray
        Boolean primary-beam limit mask (:func:`primary_beam_masks`), shaped as
        ``sky``.

    Returns
    -------
    numpy.ndarray
        Boolean ``isfinite(sky) & limit_mask``.
    """
    valid = np.isfinite(sky)
    valid &= limit_mask
    return valid


def first_round_channel_rows(
    sky: np.ndarray, primary_beam: np.ndarray, limit_mask: np.ndarray
) -> dict:
    """The first round's rows of every plane: the count, minimum and maximum of
    the valid residual voxels and the count and minimum of the valid
    primary-beam voxels (fC 7609-7644, 8383; heur:89-92).

    Parameters
    ----------
    sky, primary_beam : numpy.ndarray
        ``(time, frequency, polarization, l, m)`` planes of the residual and of
        the primary beam.
    limit_mask : numpy.ndarray
        Boolean primary-beam limit mask, shaped as ``sky``.

    Returns
    -------
    dict
        ``sky_valid_count``, ``primary_beam_valid_count`` (int64),
        ``sky_minimum``, ``sky_maximum`` and ``primary_beam_minimum``
        (float64, NaN for a plane without a valid voxel), each ``(time,
        frequency, polarization)``.

    Raises
    ------
    ValueError
        If the arrays are not planes of one shape.
    """
    _check_planes(sky, primary_beam=primary_beam, limit_mask=limit_mask)
    plane_shape = sky.shape[:3]
    rows = {
        "sky_valid_count": np.zeros(plane_shape, dtype=np.int64),
        "sky_minimum": np.full(plane_shape, np.nan),
        "sky_maximum": np.full(plane_shape, np.nan),
        "primary_beam_valid_count": np.zeros(plane_shape, dtype=np.int64),
        "primary_beam_minimum": np.full(plane_shape, np.nan),
    }
    for index in np.ndindex(plane_shape):
        sky_count = beam_count = 0
        sky_minimum = beam_minimum = np.inf
        sky_maximum = -np.inf
        for rows_block in _row_blocks(0, sky.shape[3]):
            where = (*index, rows_block)
            values = sky[where]
            valid = np.isfinite(values)
            valid &= limit_mask[where]
            sky_count += np.count_nonzero(valid)
            sky_minimum = np.minimum(
                sky_minimum, values.min(where=valid, initial=np.inf)
            )
            sky_maximum = np.maximum(
                sky_maximum, values.max(where=valid, initial=-np.inf)
            )
            values = primary_beam[where]
            np.isfinite(values, out=valid)
            valid &= limit_mask[where]
            beam_count += np.count_nonzero(valid)
            beam_minimum = np.minimum(
                beam_minimum, values.min(where=valid, initial=np.inf)
            )
        rows["sky_valid_count"][index] = sky_count
        rows["primary_beam_valid_count"][index] = beam_count
        if sky_count:
            rows["sky_minimum"][index] = sky_minimum
            rows["sky_maximum"][index] = sky_maximum
        if beam_count:
            rows["primary_beam_minimum"][index] = beam_minimum
    return rows


def new_moment_partials(map_shape: tuple, value_dtype: np.dtype) -> dict:
    """Moment partials without a channel.

    Parameters
    ----------
    map_shape : tuple of int
        ``(time, 1, polarization, l, m)``.
    value_dtype : numpy.dtype
        dtype of the residual cube.

    Returns
    -------
    dict
        ``integrated``: float64 zeros (moment 0); ``maximum``: ``-inf`` in
        ``value_dtype`` (moment 8), the mark of a pixel without a contributing
        channel (a valid value is finite).
    """
    return {
        "integrated": np.zeros(map_shape),
        "maximum": np.full(map_shape, -np.inf, dtype=value_dtype),
    }


def accumulate_moments(
    moment_partials: dict,
    sky: np.ndarray,
    valid: np.ndarray,
    channel_weights: np.ndarray,
) -> None:
    """Add planes to moment partials, in place, channel after channel.

    At every voxel of ``valid``, ``integrated += float64(sky) * weight`` and
    ``maximum = max(maximum, sky)`` (moments 0 and 8 of fC 1741, and of fC
    7867-7876 with ``valid & (sky != 0)``).

    Parameters
    ----------
    moment_partials : dict
        Partials of :func:`new_moment_partials`, ``(time, 1, polarization, l,
        m)``.
    sky : numpy.ndarray
        ``(time, B, polarization, l, m)`` planes, in increasing channel order.
    valid : numpy.ndarray
        Boolean voxels to add, shaped as ``sky``.
    channel_weights : numpy.ndarray
        float64 ``(B,)`` moment-0 weight of every plane's channel (km/s).

    Raises
    ------
    ValueError
        If the arrays do not have matching shapes.
    """
    _check_planes(sky, valid=valid)
    n_time, n_planes, n_polarization, n_l, n_m = sky.shape
    if len(channel_weights) != n_planes:
        raise ValueError(
            f"{len(channel_weights)} channel weights for {n_planes} planes"
        )
    integrated, maximum = moment_partials["integrated"], moment_partials["maximum"]
    map_shape = (n_time, 1, n_polarization, n_l, n_m)
    if integrated.shape != map_shape or maximum.shape != map_shape:
        raise ValueError(
            f"Moment partials of shape {integrated.shape} and {maximum.shape} "
            f"for planes of shape {sky.shape}"
        )
    product = np.empty((min(_BLOCK_ROWS, n_l), n_m))
    for time, plane, polarization in np.ndindex(n_time, n_planes, n_polarization):
        weight = channel_weights[plane]
        for rows in _row_blocks(0, n_l):
            values = sky[time, plane, polarization, rows]
            selected = valid[time, plane, polarization, rows]
            weighted = product[: values.shape[0]]
            np.multiply(values, weight, out=weighted, dtype=np.float64)
            target = integrated[time, 0, polarization, rows]
            np.add(target, weighted, out=target, where=selected)
            target = maximum[time, 0, polarization, rows]
            np.maximum(target, values, out=target, where=selected)


def merge_moment_partials(left: dict | None, right: dict | None) -> dict | None:
    """Merge the moment partials of disjoint channels: ``left += right`` and
    the maximum of both, in ``left``.

    Parameters
    ----------
    left, right : dict or None
        Moment partials; ``None`` (no channel) is the identity.

    Returns
    -------
    dict or None
        ``left`` holding the merged partials, or the other one if either is
        ``None``.
    """
    if left is None:
        return right
    if right is None:
        return left
    np.add(left["integrated"], right["integrated"], out=left["integrated"])
    np.maximum(left["maximum"], right["maximum"], out=left["maximum"])
    return left


def merge_moment_partials_in_tree_order(partials: Iterable) -> dict | None:
    """Merge moment partials as GraphVIPER's binary tree reduce does.

    The tree pairs neighbours level by level and carries an odd last node up.
    The partials are merged as they arrive, holding at most
    ``ceil(log2(n)) + 1`` of them, with the tree's pairs and order
    (``left += right``), so the sums equal the tree's bit for bit.

    Parameters
    ----------
    partials : iterable of dict or None
        Moment partials per slot, in slot order; ``None`` for a slot without
        channels.

    Returns
    -------
    dict or None
        The merged partials; ``None`` without a partial that is not ``None``.
    """
    # Complete subtrees as [number of leaves, merged partials].
    stack = []

    def merge_top():
        size, right = stack.pop()
        stack[-1] = [stack[-1][0] + size, merge_moment_partials(stack[-1][1], right)]

    for partial in partials:
        stack.append([1, partial])
        while len(stack) > 1 and stack[-1][0] == stack[-2][0]:
            merge_top()
    while len(stack) > 1:
        merge_top()
    return stack[0][1] if stack else None


def finalize_moments(
    moment_partials: dict | None, map_shape: tuple, value_dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """Moments 0 and 8 of merged moment partials.

    Parameters
    ----------
    moment_partials : dict or None
        Merged partials; ``None``: no channel.
    map_shape : tuple of int
        ``(time, 1, polarization, l, m)``.
    value_dtype : numpy.dtype
        dtype of the residual cube.

    Returns
    -------
    integral : numpy.ndarray
        float64 moment 0, NaN where no channel contributed.
    maximum : numpy.ndarray
        Moment 8 in ``value_dtype``, NaN where no channel contributed.
    """
    if moment_partials is None:
        return np.full(map_shape, np.nan), np.full(map_shape, np.nan, value_dtype)
    empty = moment_partials["maximum"] == -np.inf
    integral = np.where(empty, np.nan, moment_partials["integrated"])
    maximum = moment_partials["maximum"].astype(value_dtype)
    maximum[empty] = np.nan
    return integral, maximum


def masked_channel_means(
    sky: np.ndarray,
    valid: np.ndarray,
    region: np.ndarray,
    spectrum_mask: np.ndarray | None = None,
) -> np.ndarray:
    """The mean of the valid voxels of every plane in a region (fC 8442-8459,
    and the extra-mask spectra of fC 2041, 2441, 2513).

    Parameters
    ----------
    sky : numpy.ndarray
        ``(time, B, polarization, l, m)`` planes.
    valid : numpy.ndarray
        Boolean valid voxels (:func:`valid_voxels`), shaped as ``sky``.
    region : numpy.ndarray
        Boolean ``(l, m)`` region, the same in every plane.
    spectrum_mask : numpy.ndarray, optional
        Boolean spectrum mask (:func:`primary_beam_masks`), shaped as ``sky``,
        that the voxels must also be in.

    Returns
    -------
    numpy.ndarray
        float64 ``(time, B, polarization)``; NaN for a plane without a voxel.

    Raises
    ------
    ValueError
        If the arrays do not have matching shapes.
    """
    _check_planes(sky, region, valid=valid, spectrum_mask=spectrum_mask)
    means = np.full(sky.shape[:3], np.nan)
    for index, values in _gathered_planes(sky, valid, region, spectrum_mask):
        if values.size:
            means[index] = values.mean()
    return means


def masked_channel_scaled_mads(
    sky: np.ndarray,
    valid: np.ndarray,
    region: np.ndarray,
    primary_beam: np.ndarray | None = None,
    primary_beam_levels: AnnulusLevels | None = None,
) -> np.ndarray:
    """The scaled median absolute deviation of the valid voxels of every plane
    in a region, ``1.482602218505602 * median(|v - median(v)|)``, deviations in
    float64 (fC 8502-8508).

    Parameters
    ----------
    sky : numpy.ndarray
        ``(time, B, polarization, l, m)`` planes.
    valid : numpy.ndarray
        Boolean valid voxels (:func:`valid_voxels`), shaped as ``sky``.
    region : numpy.ndarray
        Boolean ``(l, m)`` region, the same in every plane.
    primary_beam : numpy.ndarray, optional
        Primary-beam planes, shaped as ``sky``; needed with
        ``primary_beam_levels``.
    primary_beam_levels : AnnulusLevels, optional
        ``(lower, higher)``, already rounded as findContinuum writes them,
        between which the primary beam of a voxel must be, ``lower <
        float64(primary_beam) < higher`` (a NaN primary beam is not).

    Returns
    -------
    numpy.ndarray
        float64 ``(time, B, polarization)``; NaN for a plane without a voxel.

    Raises
    ------
    ValueError
        If the arrays do not have matching shapes, or only one of
        ``primary_beam`` and ``primary_beam_levels`` is given.
    """
    _check_planes(sky, region, valid=valid, primary_beam=primary_beam)
    if (primary_beam is None) != (primary_beam_levels is None):
        raise ValueError("primary_beam and primary_beam_levels go together")
    mads = np.full(sky.shape[:3], np.nan)
    for index, values in _gathered_planes(
        sky, valid, region, primary_beam=primary_beam, levels=primary_beam_levels
    ):
        if values.size:
            mads[index] = SCALED_MAD_PROFILE_FACTOR * _median_and_deviation(values)[1]
    return mads


def masked_channel_medians_and_mads(
    sky: np.ndarray, valid: np.ndarray, region: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The median and the unscaled median absolute deviation of the valid
    voxels of every plane in a region (fC 3732).

    Parameters
    ----------
    sky : numpy.ndarray
        ``(time, B, polarization, l, m)`` planes.
    valid : numpy.ndarray
        Boolean valid voxels (:func:`valid_voxels`), shaped as ``sky``.
    region : numpy.ndarray
        Boolean ``(l, m)`` region, the same in every plane.

    Returns
    -------
    medians, mads : numpy.ndarray
        float64 ``(time, B, polarization)``, as ``numpy.median`` gives them;
        NaN for a plane without a voxel.

    Raises
    ------
    ValueError
        If the arrays do not have matching shapes.
    """
    _check_planes(sky, region, valid=valid)
    medians = np.full(sky.shape[:3], np.nan)
    mads = np.full(sky.shape[:3], np.nan)
    for index, values in _gathered_planes(sky, valid, region):
        if values.size:
            medians[index], mads[index] = _median_and_deviation(values)
    return medians, mads


def tile_occupancy(mask: np.ndarray, tile_shape: tuple) -> np.ndarray:
    """Whether each tile of each plane holds a ``True`` pixel.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean ``(time, frequency, polarization, l, m)``.
    tile_shape : tuple of int
        ``(l, m)`` size of the tiles from the first pixel; edge tiles may be
        smaller.

    Returns
    -------
    numpy.ndarray
        Boolean ``(time, frequency, polarization, n_tiles_l, n_tiles_m)``.
    """
    rows = np.logical_or.reduceat(
        mask, np.arange(0, mask.shape[-2], tile_shape[0]), axis=-2
    )
    return np.logical_or.reduceat(
        rows, np.arange(0, mask.shape[-1], tile_shape[1]), axis=-1
    )


def pack_plane_masks(mask: np.ndarray) -> np.ndarray:
    """Boolean masks packed eight pixels per byte along ``m``.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean ``(..., l, m)``.

    Returns
    -------
    numpy.ndarray
        uint8 ``(..., l, ceil(m / 8))`` (``numpy.packbits``).
    """
    return np.packbits(mask, axis=-1)


def unpack_plane_masks(
    packed: np.ndarray, n_m: int, out: np.ndarray | None = None
) -> np.ndarray:
    """Masks of :func:`pack_plane_masks`.

    Parameters
    ----------
    packed : numpy.ndarray
        uint8 ``(..., l, ceil(n_m / 8))``.
    n_m : int
        Pixels along ``m``.
    out : numpy.ndarray, optional
        Boolean ``(..., l, n_m)`` to unpack into, a block of rows at a time.

    Returns
    -------
    numpy.ndarray
        Boolean ``(..., l, n_m)``.
    """
    if out is None:
        return np.unpackbits(packed, axis=-1, count=n_m).view(bool)
    for rows in _row_blocks(0, packed.shape[-2]):
        out[..., rows, :] = np.unpackbits(
            packed[..., rows, :], axis=-1, count=n_m
        ).view(bool)
    return out


def _check_planes(sky, region=None, **planes):
    if sky.ndim != 5:
        raise ValueError(
            "Planes have dimensions (time, frequency, polarization, l, m), not "
            f"shape {sky.shape}"
        )
    for name, array in planes.items():
        if array is not None and array.shape != sky.shape:
            raise ValueError(
                f"{name} has shape {array.shape}, not the planes' {sky.shape}"
            )
    if region is not None and region.shape != sky.shape[-2:]:
        raise ValueError(
            f"region has shape {region.shape}, not the planes' {sky.shape[-2:]}"
        )


def _row_blocks(start, stop):
    for first in range(start, stop, _BLOCK_ROWS):
        yield slice(first, min(first + _BLOCK_ROWS, stop))


def _gathered_planes(
    sky, valid, region, spectrum_mask=None, primary_beam=None, levels=None
):
    """Per plane index ``(time, frequency, polarization)``, the float64 values
    of ``sky`` at the voxels that are valid, in ``region``, in
    ``spectrum_mask`` and between the primary-beam ``levels``, in C order;
    nothing for an empty region."""
    region_rows = np.flatnonzero(region.any(axis=1))
    if region_rows.size == 0:
        return
    region_columns = np.flatnonzero(region.any(axis=0))
    columns = slice(region_columns[0], region_columns[-1] + 1)
    lower, higher = (None, None) if levels is None else levels
    for index in np.ndindex(sky.shape[:3]):
        blocks = []
        for rows in _row_blocks(region_rows[0], region_rows[-1] + 1):
            where = (*index, rows, columns)
            selected = valid[where] & region[rows, columns]
            if spectrum_mask is not None:
                selected &= spectrum_mask[where]
            if levels is not None:
                beam = primary_beam[where].astype(np.float64)
                selected &= beam > lower
                selected &= beam < higher
            blocks.append(sky[where][selected])
        yield index, np.concatenate(blocks, dtype=np.float64)


def _median_and_deviation(values):
    """``numpy.median`` of float64 values and of their absolute deviations from
    it, in place."""
    median = np.median(values, overwrite_input=True)
    np.subtract(values, median, out=values)
    np.abs(values, out=values)
    return median, np.median(values, overwrite_input=True)
