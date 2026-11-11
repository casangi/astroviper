"""Shared helpers for the moments (immoments) tests at all three layers.

Provides a synthetic-image builder and an *independent* nan-aware reference
implementation of every moment (straightforward vectorized full-array numpy,
deliberately different from the streaming plane-by-plane implementation in
``astroviper.processing_functions.image_analysis.moments``).
"""

import warnings

import numpy as np
import xarray as xr
from xradio.image import make_empty_sky_image

RAD_PER_ARCSEC = np.pi / 180 / 3600

ALL_MOMENTS = [
    "mean",
    "integrated",
    "weighted_coord",
    "weighted_dispersion_coord",
    "median",
    "median_coord",
    "standard_deviation",
    "rms",
    "abs_mean_dev",
    "maximum",
    "maximum_coord",
    "minimum",
    "minimum_coord",
]


def make_test_image_xds(
    n_frequency=7,
    n_polarization=2,
    n_l=18,
    n_m=14,
    dtype=np.float32,
    with_nans=True,
    with_mask=True,
    seed=42,
):
    """Build a small in-memory synthetic image dataset with SKY (and MASK)."""
    rng = np.random.default_rng(seed)
    img_xds = make_empty_sky_image(
        phase_center=[0.6, -0.2],
        image_size=[n_l, n_m],
        cell_size=[15 * RAD_PER_ARCSEC, 15 * RAD_PER_ARCSEC],
        frequency_coords=np.linspace(1.4e9, 1.5e9, n_frequency),
        pol_coords=["I", "Q", "U", "V"][:n_polarization],
        time_coords=[0],
    )
    shape = (1, n_frequency, n_polarization, n_l, n_m)
    sky = rng.normal(0.1, 1.0, shape).astype(dtype)
    if with_nans:
        # A few isolated NaNs plus one fully-NaN profile along every axis of
        # interest so the "no contributing planes -> NaN" path is exercised.
        sky[0, 1, 0, 2, 3] = np.nan
        sky[0, :, 0, 5, 6] = np.nan  # all-NaN frequency profile
        if n_polarization > 1:
            sky[0, 2, :, 7, 8] = np.nan  # all-NaN polarization profile
    img_xds["SKY"] = xr.DataArray(
        sky, dims=["time", "frequency", "polarization", "l", "m"]
    )
    img_xds["SKY"].attrs["units"] = "Jy/beam"
    data_group = {"sky": "SKY"}
    if with_mask:
        mask = rng.random(shape) > 0.15
        img_xds["MASK"] = xr.DataArray(
            mask, dims=["time", "frequency", "polarization", "l", "m"]
        )
        data_group["mask"] = "MASK"
    img_xds.attrs["data_groups"] = {"base": data_group}
    return img_xds


def write_test_image(img_xds, path):
    """Write the synthetic image to a Zarr store."""
    from xradio.image import write_image

    write_image(img_xds, imagename=str(path), out_format="zarr", overwrite=True)


def reference_moments(
    sky,
    axis,
    coord_values,
    mask=None,
    include_range=None,
    exclude_range=None,
):
    """Independent nan-aware reference implementation of all moments.

    Parameters
    ----------
    sky : numpy.ndarray
        The full sky array.
    axis : int
        Index of the moment axis in ``sky``.
    coord_values : numpy.ndarray
        Numeric coordinate values of the moment axis.
    mask : numpy.ndarray of bool, optional
        ``True`` = pixel contributes.
    include_range, exclude_range : tuple of float, optional
        ``(low, high)`` pixel-value ranges.

    Returns
    -------
    dict of str -> numpy.ndarray
        Every moment map (float64), with the moment axis removed.
    """
    data = np.moveaxis(sky, axis, -1).astype(np.float64)
    valid = np.isfinite(data)
    if mask is not None:
        valid &= np.moveaxis(mask, axis, -1).astype(bool)
    if include_range is not None:
        valid &= (data >= include_range[0]) & (data <= include_range[1])
    if exclude_range is not None:
        valid &= (data < exclude_range[0]) | (data > exclude_range[1])
    masked = np.where(valid, data, np.nan)

    v = np.asarray(coord_values, dtype=np.float64)
    n = v.size
    widths = np.abs(np.gradient(v)) if n > 1 else np.ones(1)

    count = valid.sum(axis=-1)
    empty = count == 0
    zero_filled = np.where(valid, data, 0.0)
    s1 = zero_filled.sum(axis=-1)

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)

        results = {}
        results["mean"] = np.nanmean(masked, axis=-1)
        results["integrated"] = np.where(empty, np.nan, (zero_filled * widths).sum(-1))
        # Shifted coordinate frame, matching the implementation's numerical
        # conditioning (raw v^2 sums suffer catastrophic cancellation for
        # frequency coordinates ~1e9 Hz).
        v_shifted = v - v.mean()
        weighted_shifted = np.where(
            s1 != 0, (zero_filled * v_shifted).sum(-1) / s1, np.nan
        )
        results["weighted_coord"] = v.mean() + weighted_shifted
        variance = (
            np.where(s1 != 0, (zero_filled * v_shifted**2).sum(-1) / s1, np.nan)
            - weighted_shifted**2
        )
        results["weighted_dispersion_coord"] = np.sqrt(
            np.where(variance >= 0, variance, np.nan)
        )
        results["median"] = np.nanmedian(masked, axis=-1)

        cumulative = np.cumsum(zero_filled, axis=-1)
        crossed = cumulative >= 0.5 * s1[..., None]
        first_crossing = np.argmax(crossed, axis=-1)
        median_coord = np.where(
            (s1 > 0) & crossed.any(axis=-1), v[first_crossing], np.nan
        )
        results["median_coord"] = np.where(empty, np.nan, median_coord)

        std = np.nanstd(masked, axis=-1, ddof=1)
        results["standard_deviation"] = np.where(count > 1, std, np.nan)
        results["rms"] = np.sqrt(np.nanmean(masked**2, axis=-1))
        results["abs_mean_dev"] = np.nanmean(
            np.abs(masked - np.nanmean(masked, axis=-1, keepdims=True)), axis=-1
        )
        results["maximum"] = np.nanmax(
            np.where(empty[..., None], -np.inf, masked), axis=-1
        )
        results["maximum"] = np.where(empty, np.nan, results["maximum"])
        argmax = np.nanargmax(np.where(empty[..., None], 0.0, masked), axis=-1)
        results["maximum_coord"] = np.where(empty, np.nan, v[argmax])
        results["minimum"] = np.nanmin(
            np.where(empty[..., None], np.inf, masked), axis=-1
        )
        results["minimum"] = np.where(empty, np.nan, results["minimum"])
        argmin = np.nanargmin(np.where(empty[..., None], 0.0, masked), axis=-1)
        results["minimum_coord"] = np.where(empty, np.nan, v[argmin])
    return results


def assert_moments_match(moments_img_xds, reference, axis, atol=1e-5):
    """Assert every moment variable in ``moments_img_xds`` matches ``reference``.

    Parameters
    ----------
    moments_img_xds : xarray.Dataset
        Output of the moments processing function / a loaded moments store.
    reference : dict
        Output of :func:`reference_moments`.
    axis : int
        Index of the (collapsed, size-1) moment axis in the output variables.
    atol : float
        Absolute tolerance (the implementation may compute in float32).
    """
    for name, expected in reference.items():
        variable_name = "SKY_MOMENT_" + name.upper()
        if variable_name not in moments_img_xds:
            continue
        actual = np.squeeze(moments_img_xds[variable_name].values, axis=axis)
        np.testing.assert_allclose(
            actual,
            expected,
            atol=atol,
            rtol=1e-4,
            equal_nan=True,
            err_msg=f"moment '{name}' mismatch",
        )
