"""Explicit spectral diagnostics from finalized flat-sky restored coefficients."""

import copy

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import modify_data_groups_xds


def spectral_products(
    img_xds,
    image_data_group_in_name="restored",
    image_data_group_in_residual_name="residual",
    products=("ALPHA", "ALPHA_ERROR"),
):
    """Calculate requested spectral diagnostics independently of imaging.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Public continuum image with flat-sky restored coefficients and residual.
    image_data_group_in_name : str, optional
        Group whose sky role identifies the flat-sky restored stack. A corrected
        image stored as the sole restored product uses the restored group.
    image_data_group_in_residual_name : str, optional
        Group identifying the flat-sky coefficient-space residual.
    products : tuple of str, optional
        Requested ALPHA, ALPHA_ERROR and/or BETA. Alpha requires two Taylor
        coefficients; beta requires three. No unrequested product is allocated.

    Returns
    -------
    xarray.Dataset
        Input with requested diagnostics on a singleton reference-frequency axis.
        Invalid pixels are NaN. ALPHA_ERROR is a residual-based diagnostic,
        not a statistically calibrated one-sigma uncertainty.
    """
    from astroviper.processing_functions.imaging.restore import (
        resolve_restoration_product,
    )

    allowed = {"ALPHA", "ALPHA_ERROR", "BETA"}
    if (
        not products
        or not set(products) <= allowed
        or len(set(products)) != len(products)
    ):
        raise ValueError(
            "products must contain distinct ALPHA, ALPHA_ERROR, and/or BETA names"
        )
    restored_name = resolve_restoration_product(
        img_xds, image_data_group_in_name, "sky"
    )
    residual_name = resolve_restoration_product(
        img_xds, image_data_group_in_residual_name, "sky"
    )
    restored, residual = img_xds[restored_name], img_xds[residual_name]
    for name, variable in ((restored_name, restored), (residual_name, residual)):
        if variable.attrs.get("sky_convention") != "flat_sky":
            raise ValueError(f"{name} must use the flat-sky convention")
        if "frequency_polynomial" not in variable.dims:
            raise ValueError(f"{name} must have a frequency_polynomial axis")
    required = 3 if "BETA" in products else 2
    if restored.sizes["frequency_polynomial"] < required:
        raise ValueError(
            f"Requested spectral products require at least {required} coefficients"
        )
    if restored.dims != residual.dims or restored.shape != residual.shape:
        raise ValueError(
            "Restored coefficients and residual must have identical dimensions and shape"
        )
    xr.align(restored, residual, join="exact", copy=False)
    if residual.attrs.get("residual_representation") != "coefficient_space":
        raise ValueError("Residual must use the coefficient-space representation")
    zero = restored.isel(frequency_polynomial=0, drop=True)
    one = restored.isel(frequency_polynomial=1, drop=True)
    rzero = residual.isel(frequency_polynomial=0, drop=True)
    threshold = abs(rzero).max(dim=("l", "m"), skipna=True) / 10.0
    valid = np.isfinite(zero) & np.isfinite(one) & (abs(zero) > threshold) & (zero != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = xr.where(valid, one / zero, np.nan)
        calculated = {}
        if "ALPHA" in products:
            calculated["ALPHA"] = alpha
        if "ALPHA_ERROR" in products:
            rone = residual.isel(frequency_polynomial=1, drop=True)
            error = np.sqrt(rone**2 + (alpha * rzero) ** 2) / abs(zero)
            calculated["ALPHA_ERROR"] = xr.where(
                valid & np.isfinite(error), error, np.nan
            )
        if "BETA" in products:
            two = restored.isel(frequency_polynomial=2, drop=True)
            beta = two / zero - 0.5 * alpha * (alpha - 1.0)
            calculated["BETA"] = xr.where(valid & np.isfinite(beta), beta, np.nan)
    out = img_xds.copy(deep=False)
    out.attrs = copy.deepcopy(img_xds.attrs)
    if "frequency" in out.dims and out.sizes["frequency"] != 1:
        raise ValueError(
            "Continuum spectral products require a singleton frequency axis"
        )
    reference = float(out.attrs.get("reference_frequency", out.frequency.values[0]))
    for name in products:
        value = (
            calculated[name]
            .astype(restored.dtype)
            .expand_dims(frequency=[reference], axis=1)
        )
        value.attrs = {
            **value.attrs,
            "type": "sky",
            "sub_type": "SpectralIndex" if name == "ALPHA" else "Intensity",
            "units": "",
        }
        value.attrs["reference_frequency"] = reference
        out[name] = value
        modify_data_groups_xds(
            out,
            name.lower(),
            {"sky": name},
            description=f"{name} derived from flat-sky restored coefficients and residual.",
        )
    for name in img_xds.coords:
        if name in out.coords:
            out[name].attrs = copy.deepcopy(img_xds[name].attrs)
    return out
