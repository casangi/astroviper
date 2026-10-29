"""
Methods and utilities for computing statistics on image data, cubes and single
plane images.
"""

import numpy as np


def get_image_masksum(image_xds, data_group_name):
    """
    Compute the per-plane sum of the mask in an image dataset.

    Parameters:
    -----------
    image_xds: xarray.Dataset
        The image dataset with dimensions (time, frequency, polarization, l, m).
    data_group_name: str
        Name of the entry in ``image_xds.attrs["data_groups"]`` whose
        ``"sky"`` key gives the data variable to size against and whose
        optional ``"mask"`` key gives the mask data variable.

    Returns:
    --------
    mask_sum: numpy.ndarray
        Sum of valid (unmasked) pixels per ``(time, frequency, polarization)``
        plane.
    """

    data_group = image_xds.attrs["data_groups"][data_group_name]
    sky_name = data_group["sky"]
    spatial_dims = image_xds[sky_name].dims[-2:]
    mask_name = data_group.get("mask", None)

    if mask_name is not None and mask_name in image_xds:
        mask_sum = image_xds[mask_name].sum(dim=spatial_dims).values
    else:
        # No mask present = all pixels valid, on every plane
        n_pixels = int(np.prod(image_xds[sky_name].shape[-2:]))
        plane_shape = image_xds[sky_name].shape[:-2]
        mask_sum = np.full(plane_shape, n_pixels, dtype=int)

    return mask_sum


def image_peak_residual(
    image_xds, data_group_name, per_plane_stats=False, use_mask=True, dv="SKY_RESIDUAL"
):
    """
    Compute the peak residual of an image, optionally per plane.

    Parameters:
    -----------
    image_xds: xarray.Dataset
        The image dataset with dimensions (time, frequency, polarization, l, m).
    data_group_name: str
        Name of the entry in ``image_xds.attrs["data_groups"]`` whose
        optional ``"mask"`` key gives the mask data variable, used when
        ``use_mask`` is True.
    per_plane_stats: bool
        If True, compute peak residual for each (time, frequency, polarization) plane.
        If False, compute peak residual for the entire image across all planes.
    use_mask: bool
        If True, consider only unmasked pixels in the computation.
    dv: str
        The data variable in the xarray.Dataset to compute the peak residual from.
        Default is 'SKY'.

    Returns:
    --------
    peak_residual: float
        The peak residual value of the image.
    """

    # Get location of the peak absolute value
    # Use that to index into the original image to get the signed value

    # Apply the mask if requested
    if use_mask:
        mask_name = image_xds.attrs["data_groups"][data_group_name].get("mask", None)
        if mask_name is not None and mask_name in image_xds:
            mask_xds = image_xds[mask_name]
            image_xds = image_xds.where(mask_xds)

    if per_plane_stats:
        # Compute peak residual for each (time, frequency, polarization) plane
        peak_residual = image_xds[dv].reduce(
            np.vectorize(
                lambda arr: arr[np.unravel_index(np.abs(arr).argmax(), arr.shape)]
            ),
            dim=["y", "x"],
        )
    else:
        # Compute peak residual for the entire image across all planes
        peak_res_idx = np.unravel_index(
            np.abs(image_xds[dv].values).argmax(), image_xds[dv].shape
        )
        peak_residual = image_xds[dv].values[peak_res_idx]

    return peak_residual


# Largest number of (spatial bin, flux bin) cells of the entropy histogram.
MAX_ENTROPY_CELLS = 1 << 24


def image_residual_entropy(
    image_xds, data_group_name, spatial_bins=7, flux_bins=10, max_snr=6.0, mask=None
):
    """
    Entropy of the residual image, per plane.

    The entropy of Homan, Roth and Pushkarev (2024, AJ 167, 11) measures how
    much a residual image looks like noise everywhere. The image is divided
    into ``spatial_bins x spatial_bins`` spatial bins and the pixel values, in
    units of the RMS of the plane, into flux bins ``1 / flux_bins`` wide. With
    ``q`` the fraction of the pixels of the plane that fall into one
    (spatial bin, flux bin) cell, the entropy is ``-sum(q * ln(q))`` over all
    cells that hold a pixel. It rises while CLEAN removes the emission and
    falls again once CLEAN starts to fit the noise inside the clean mask,
    which makes the residual there differ from its surroundings.

    The entropy is taken over the whole plane, as in the paper. It is only
    worked out for planes whose peak residual is at most ``max_snr`` times
    their RMS: a stronger peak means the image is dominated by the pattern of
    the point spread function, where the entropy is not a measure of noise.

    Parameters
    ----------
    image_xds : xarray.Dataset
        The image dataset with dimensions (time, frequency, polarization, l, m).
    data_group_name : str
        Name of the entry in ``image_xds.attrs["data_groups"]`` whose ``"sky"``
        key gives the residual image and whose optional ``"mask"`` key gives
        the clean mask.
    spatial_bins : int, optional
        Number of spatial bins along each of the two image axes. Default 7.
    flux_bins : int, optional
        Number of flux bins per unit of RMS. Default 10.
    max_snr : float, optional
        Largest ratio of peak to RMS for which the entropy is worked out.
        Default 6.
    mask : numpy.ndarray, optional
        Clean mask of the shape of the residual, used instead of the mask of
        the data group (for example the mask the first model update will
        make, before it exists). Default: the mask of the data group.

    Returns
    -------
    entropy : numpy.ndarray
        Entropy per ``(time, frequency, polarization)`` plane. Not a number
        for a plane whose ratio of peak to RMS is above ``max_snr``, and for a
        plane without any finite pixel or with an RMS of zero.
    snr : numpy.ndarray
        Ratio of the peak of ``|residual|`` inside the clean mask (the whole
        plane if there is no mask) to the RMS of the whole plane, per plane.
        Not a number for a plane without any finite pixel or with an RMS of
        zero.

    Notes
    -----
    Pixels that are not finite are left out. Pixel values beyond
    ``2 * max_snr`` times the RMS, which can only lie outside the clean mask,
    are counted in the outermost flux bin, so that the number of flux bins
    stays small. The histogram never has more than ``MAX_ENTROPY_CELLS``
    cells: with a very large ``max_snr`` or very many bins the outermost flux
    bin moves inward. The histogram is filled in blocks of rows: no array of
    the size of the image is allocated.
    """
    data_group = image_xds.attrs["data_groups"][data_group_name]
    residual = image_xds[data_group["sky"]].values
    if mask is None:
        mask_name = data_group.get("mask", None)
        if mask_name is not None and mask_name in image_xds:
            mask = image_xds[mask_name].values
    plane_shape = residual.shape[:-2]
    n_l, n_m = residual.shape[-2:]
    entropy = np.full(plane_shape, np.nan, dtype=float)
    snr = np.full(plane_shape, np.nan, dtype=float)

    l_bin = (np.arange(n_l) * spatial_bins) // n_l
    m_bin = (np.arange(n_m) * spatial_bins) // n_m
    # flux bins -edge ... edge - 1; values beyond go into the outermost bins
    edge = int(
        min(
            np.ceil(2.0 * max_snr * flux_bins),
            max(1, MAX_ENTROPY_CELLS // (2 * spatial_bins * spatial_bins)),
        )
    )
    n_flux = 2 * edge
    rows_per_block = max(1, (1 << 20) // n_m)

    for index in np.ndindex(plane_shape):
        plane = residual[index]
        sum_squares = 0.0
        peak = 0.0
        n_finite = 0
        for start in range(0, n_l, rows_per_block):
            rows = plane[start : start + rows_per_block]
            finite = np.isfinite(rows)
            values = rows[finite].astype(float)
            n_finite += values.size
            sum_squares += float(np.dot(values, values))
            if mask is not None:
                searched = mask[index][start : start + rows_per_block][finite]
                values = values[searched.astype(bool)]
            if values.size:
                peak = max(peak, float(values.max()), -float(values.min()))
        if n_finite == 0 or sum_squares <= 0.0:
            continue
        rms = np.sqrt(sum_squares / n_finite)
        snr[index] = peak / rms
        if snr[index] > max_snr:
            continue
        counts = np.zeros(spatial_bins * spatial_bins * n_flux, dtype=np.int64)
        for start in range(0, n_l, rows_per_block):
            rows = plane[start : start + rows_per_block]
            finite = np.isfinite(rows)
            flux = np.floor(rows[finite] * (flux_bins / rms)).astype(np.int64)
            np.clip(flux, -edge, edge - 1, out=flux)
            cell = (
                l_bin[start : start + rows_per_block, None] * spatial_bins
                + m_bin[None, :]
            )[finite]
            counts += np.bincount(cell * n_flux + flux + edge, minlength=counts.size)
        fraction = counts[counts > 0] / n_finite
        entropy[index] = float(-(fraction * np.log(fraction)).sum())

    return entropy, snr


# ----------------------------------------------------------------------------
# Plane statistics without full size temporaries
# ----------------------------------------------------------------------------
#
# ``np.abs(plane)`` or ``np.where(mask, plane, nan)`` allocates a copy of the
# whole plane, and a cube wide ``np.abs(cube)`` a copy of the whole cube. Inside
# the imaging cycle such copies are made for every plane of every residual
# update and model update, so the helpers below scan the plane in blocks of
# rows and never hold more than one block of temporaries.

SCAN_BLOCK_ELEMENTS = 1 << 18  # about 2 MB of float64 per block


def _rows_per_block(plane):
    """Rows of ``plane`` that make a block of about ``SCAN_BLOCK_ELEMENTS``."""
    return max(1, SCAN_BLOCK_ELEMENTS // max(1, plane.shape[-1]))


def plane_peak_abs_signed(plane, mask=None):
    """
    Signed value of a plane at its largest absolute value, without a copy of
    the plane.

    Pixels that are not a number are ignored, as are pixels outside the mask.
    Among equal absolute values the first in row major order is taken, so the
    result is the same as ``plane[np.nanargmax(np.abs(plane))]`` (with the
    mask applied) but only a block of rows is ever copied.

    Parameters
    ----------
    plane : numpy.ndarray
        2-D image plane.
    mask : numpy.ndarray, optional
        Array of the same shape; only pixels where ``mask > 0.5`` count.

    Returns
    -------
    float
        Signed value at the absolute peak. NaN when no pixel counts.
    """
    rows = _rows_per_block(plane)
    best = -1.0
    best_index = None
    n_columns = plane.shape[-1]
    for start in range(0, plane.shape[0], rows):
        block = plane[start : start + rows]
        magnitude = np.abs(block)
        magnitude[np.isnan(magnitude)] = -1.0
        if mask is not None:
            magnitude[~(mask[start : start + rows] > 0.5)] = -1.0
        flat = int(np.argmax(magnitude))
        value = float(magnitude.flat[flat])
        if value > best:
            best = value
            best_index = (start + flat // n_columns, flat % n_columns)
    if best_index is None or best < 0.0:
        return float("nan")
    return float(plane[best_index])


def plane_abs_max(plane, mask=None):
    """
    Largest absolute value of a plane, without a copy of the plane.

    Not a number propagates, as with ``np.abs(plane).max()``. With a mask the
    result is ``np.where(mask, np.abs(plane), 0).max()``: a selected pixel that
    is not a number gives NaN, and a plane with no selected pixel gives 0.

    Parameters
    ----------
    plane : numpy.ndarray
        2-D image plane.
    mask : numpy.ndarray, optional
        Array of the same shape; only pixels where ``mask`` is nonzero count.

    Returns
    -------
    float
        ``np.abs(plane).max()``, or ``np.where(mask, np.abs(plane), 0).max()``
        with a mask, computed block by block.
    """
    rows = _rows_per_block(plane)
    best = np.float64(-np.inf)
    for start in range(0, plane.shape[0], rows):
        block = np.abs(plane[start : start + rows])
        if mask is not None:
            block[mask[start : start + rows] == 0] = 0.0
        best = np.maximum(best, block.max())
    return float(best)


def plane_abs_sum(plane):
    """
    Sum of the absolute values of a plane, without a copy of the plane.

    Parameters
    ----------
    plane : numpy.ndarray
        2-D image plane.

    Returns
    -------
    float
        ``np.abs(plane).sum()``, accumulated in float64 block by block.
    """
    rows = _rows_per_block(plane)
    total = np.float64(0.0)
    for start in range(0, plane.shape[0], rows):
        total += np.abs(plane[start : start + rows]).sum(dtype=np.float64)
    return float(total)


def cube_plane_abs_max(cube):
    """
    Largest absolute value of every plane of a cube, without a copy of the
    cube.

    Parameters
    ----------
    cube : numpy.ndarray
        Array whose last two axes are the image plane.

    Returns
    -------
    numpy.ndarray
        ``np.abs(cube).max(axis=(-2, -1))`` as float64, computed plane by
        plane and block by block.
    """
    plane_shape = cube.shape[:-2]
    result = np.empty(plane_shape, dtype=np.float64)
    for index in np.ndindex(plane_shape):
        result[index] = plane_abs_max(cube[index])
    return result


# The statistics of a residual plane (plane_residual_statistics): about zero
# for the RMS, about the mean for the skew and the excess kurtosis.
RESIDUAL_STATISTICS = (
    "rms_in_mask",
    "rms_outside_mask",
    "max_in_mask",
    "min_in_mask",
    "mad_rms_outside_mask",
    "skew_in_mask",
    "excess_kurtosis_in_mask",
)

# Pixels outside the mask the robust noise (mad_rms_outside_mask) is taken
# from, at most: a regular subsample of the plane.
MAD_SAMPLES = 1_000_000


def plane_residual_statistics(plane, mask=None):
    """
    Statistics of a residual plane inside and outside a mask, without a copy
    of the plane.

    Pixels that are not a number are ignored. Without a mask every pixel is
    inside it and the statistics outside it are NaN, as are those of an empty
    region.

    Parameters
    ----------
    plane : numpy.ndarray
        2-D residual plane.
    mask : numpy.ndarray, optional
        Array of the same shape; the mask is where ``mask > 0.5``.

    Returns
    -------
    dict
        Floats keyed by :data:`RESIDUAL_STATISTICS`: the RMS about zero inside
        and outside the mask, the largest and the smallest value inside it,
        the 1.4826 x median absolute deviation outside it (on a regular
        subsample of at most :data:`MAD_SAMPLES` pixels of the plane), and
        the skew and the excess kurtosis inside it.
    """
    rows = _rows_per_block(plane)
    count = {True: 0, False: 0}
    squares = {True: 0.0, False: 0.0}
    total = 0.0
    largest, smallest = -np.inf, np.inf
    for start in range(0, plane.shape[0], rows):
        block = np.asarray(plane[start : start + rows], dtype=np.float64)
        finite = np.isfinite(block)
        inside = finite if mask is None else finite & (mask[start : start + rows] > 0.5)
        for region, selected in ((True, inside), (False, finite & ~inside)):
            values = block[selected]
            count[region] += values.size
            squares[region] += float(np.dot(values, values))
            if region and values.size:
                total += float(values.sum())
                largest = max(largest, float(values.max()))
                smallest = min(smallest, float(values.min()))
    n = count[True]
    mean = total / n if n else np.nan
    central = np.zeros(3)  # second, third and fourth central moments x n
    if n:
        for start in range(0, plane.shape[0], rows):
            block = np.asarray(plane[start : start + rows], dtype=np.float64)
            inside = np.isfinite(block)
            if mask is not None:
                inside &= mask[start : start + rows] > 0.5
            d = block[inside] - mean
            d2 = d * d
            central += (d2.sum(), (d2 * d).sum(), (d2 * d2).sum())
    m2, m3, m4 = central / n if n else (np.nan,) * 3
    step = max(1, int(np.ceil(np.sqrt(plane.size / MAD_SAMPLES))))
    sample = np.asarray(plane[::step, ::step], dtype=np.float64)
    outside = np.isfinite(sample)
    outside &= False if mask is None else ~(mask[::step, ::step] > 0.5)
    sample = sample[outside]
    mad = (
        1.4826 * float(np.median(np.abs(sample - np.median(sample))))
        if sample.size
        else np.nan
    )

    def rms(region):
        return np.sqrt(squares[region] / count[region]) if count[region] else np.nan

    return {
        "rms_in_mask": float(rms(True)),
        "rms_outside_mask": float(rms(False)),
        "max_in_mask": largest if n else np.nan,
        "min_in_mask": smallest if n else np.nan,
        "mad_rms_outside_mask": mad,
        "skew_in_mask": float(m3 / m2**1.5) if n and m2 > 0 else np.nan,
        "excess_kurtosis_in_mask": float(m4 / m2**2 - 3.0) if n and m2 > 0 else np.nan,
    }


def cube_residual_statistics(cube, mask=None):
    """
    :func:`plane_residual_statistics` of every plane of a cube.

    Parameters
    ----------
    cube : numpy.ndarray
        Array whose last two axes are the image plane.
    mask : numpy.ndarray, optional
        Array of the shape of ``cube``.

    Returns
    -------
    dict
        Float64 arrays of shape ``cube.shape[:-2]`` keyed by
        :data:`RESIDUAL_STATISTICS`.
    """
    plane_shape = cube.shape[:-2]
    result = {name: np.empty(plane_shape) for name in RESIDUAL_STATISTICS}
    for index in np.ndindex(plane_shape):
        stats = plane_residual_statistics(
            cube[index], None if mask is None else mask[index]
        )
        for name, value in stats.items():
            result[name][index] = value
    return result


def cube_flux_by_sign(cube):
    """
    Sums of the positive and of the negative values of every plane of a cube,
    without a copy of the cube.

    Parameters
    ----------
    cube : numpy.ndarray
        Array whose last two axes are the image plane.

    Returns
    -------
    positive, negative : numpy.ndarray
        Float64 arrays of shape ``cube.shape[:-2]``.
    """
    plane_shape = cube.shape[:-2]
    positive, negative = np.zeros(plane_shape), np.zeros(plane_shape)
    for index in np.ndindex(plane_shape):
        plane = cube[index]
        rows = _rows_per_block(plane)
        for start in range(0, plane.shape[0], rows):
            block = np.asarray(plane[start : start + rows], dtype=np.float64)
            positive[index] += block[block > 0].sum()
            negative[index] += block[block < 0].sum()
    return positive, negative
