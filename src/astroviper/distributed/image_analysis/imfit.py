"""Image-plane Gaussian fitting for xradio images.

This module provides :func:`imfit`, the astronomer-facing entry point for
fitting 2-D Gaussians to xradio image Datasets. It wraps the generic
:func:`fit_multi_gaussian2d` fitter, adding xradio-specific input
normalization, sky-coordinate translation, beam deconvolution, and
publication of results in standard astronomical conventions.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr

from .multi_gaussian2d_fit import fit_multi_gaussian2d
from ...utils._gaussian_math import (
    SIG2FWHM,
    FWHM2SIG,
    fwhm_from_sigma,
    sigma_from_fwhm,
    deconvolve_gaussian,
    deconvolve_gaussian_with_errors,
)
from ...utils.coordinate_axes import (
    representative_pixel_scale,
    world_value_to_pixel,
)
from ...utils.sky_coordinates import (
    coerce_angle_to_radians,
    is_scalar_number,
    parse_sky_center_to_radians,
    skycoord_to_lm_from_wcs,
)

Number = Union[int, float]
MaskSpec = Optional[Union[str, Path, np.ndarray, xr.DataArray, da.Array]]

# Angular unit strings recognised when validating beam units.
_ANGULAR_UNITS = {"rad", "arcsec", "arcmin", "deg"}
_CRTF_SHAPES = frozenset(
    {"box", "centerbox", "rotbox", "poly", "circle", "annulus", "ellipse"}
)

# Conversion factors to radians.
_TO_RAD = {
    "rad": 1.0,
    "arcsec": np.pi / (180.0 * 3600.0),
    "arcmin": np.pi / (180.0 * 60.0),
    "deg": np.pi / 180.0,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _convert_world_widths_to_pixel(
    comp: Dict[str, Any],
    l_axis: np.ndarray,
    m_axis: np.ndarray,
) -> None:
    """Convert world-frame width guesses in-place into pixel-frame widths.

    Parameters
    ----------
    comp : dict[str, Any]
        Component guess dictionary to mutate in place.
    l_axis : np.ndarray
        Native ``l`` axis in world units.
    m_axis : np.ndarray
        Native ``m`` axis in world units.

    Returns
    -------
    None
        The input dictionary is modified in place.

    Raises
    ------
    ValueError
        If world-frame width guesses are supplied on a non-square pixel grid.

    Notes
    -----
    World-frame initial widths can be mapped into pixel widths only when the image
    uses square pixels, because the optimizer parameterization assumes a single
    pixel metric along the rotated principal axes.
    """
    width_keys = {
        "fwhm_major",
        "fwhm_minor",
        "sigma_x",
        "sigma_y",
        "sx",
        "sy",
    }
    if not any(key in comp for key in width_keys):
        return
    l_scale = representative_pixel_scale(l_axis, "l")
    m_scale = representative_pixel_scale(m_axis, "m")
    if not np.isclose(l_scale, m_scale, rtol=1e-8, atol=0.0):
        raise ValueError(
            "World-frame initial widths are only supported for square pixels; "
            "non-square pixels cannot safely convert initial FWHM/sigma guesses "
            "into pixel-space widths."
        )
    pixel_scale = 0.5 * (l_scale + m_scale)
    for key in ("fwhm_major", "fwhm_minor", "sigma_x", "sigma_y", "sx", "sy"):
        if key in comp:
            comp[key] = coerce_angle_to_radians(comp[key]) / pixel_scale


def _convert_world_component_guess_to_pixel(
    xds: xr.Dataset,
    comp: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert one world-frame imfit component guess into pixel-frame form.

    Parameters
    ----------
    xds : xr.Dataset
        Input xradio image Dataset that defines the native ``l``/``m`` axes and sky
        metadata.
    comp : Mapping[str, Any]
        One component guess dictionary.

    Returns
    -------
    dict[str, Any]
        Pixel-frame component guess compatible with :func:`fit_multi_gaussian2d`.

    Notes
    -----
    Pixel guesses continue to use ``x0``/``y0`` directly. World guesses may use
    native ``l0``/``m0`` keys, explicit sky keys ``ra``/``dec``,
    ``right_ascension``/``declination``, or generic sky keys ``lon``/``lat`` or
    ``longitude``/``latitude`` in the image frame. String or Astropy angle-valued
    ``x0``/``y0`` inputs are also interpreted as sky coordinates in the dataset
    frame. World-frame centers and widths are converted to zero-based pixel
    coordinates before the fit for numerical stability.
    """
    out = dict(comp)
    l_axis = np.asarray(xds.coords["l"].values, dtype=float)
    m_axis = np.asarray(xds.coords["m"].values, dtype=float)

    has_l0 = "l0" in out
    has_m0 = "m0" in out
    has_lm_keys = has_l0 or has_m0
    has_sky_keys = any(
        key in out
        for key in (
            "ra",
            "dec",
            "right_ascension",
            "declination",
            "lon",
            "lat",
            "longitude",
            "latitude",
        )
    )
    x0_val = out.get("x0")
    y0_val = out.get("y0")
    xy_are_plain_numbers = is_scalar_number(x0_val) and is_scalar_number(y0_val)

    if not has_lm_keys and not has_sky_keys and xy_are_plain_numbers:
        return out

    if has_lm_keys:
        if has_l0 != has_m0:
            raise ValueError("Native world-center guesses require both 'l0' and 'm0'.")
        # In imfit, explicit l0/m0 keys mean native image-world coordinates.
        l0 = coerce_angle_to_radians(out.pop("l0"))
        m0 = coerce_angle_to_radians(out.pop("m0"))
        out.pop("x0", None)
        out.pop("y0", None)
    else:
        if has_sky_keys:
            lon_value = out.pop(
                "ra",
                out.pop(
                    "right_ascension",
                    out.pop("lon", out.pop("longitude", None)),
                ),
            )
            lat_value = out.pop(
                "dec",
                out.pop(
                    "declination",
                    out.pop("lat", out.pop("latitude", None)),
                ),
            )
        else:
            # imfit treats non-numeric x0/y0 tokens as sky-frame longitude/latitude.
            lon_value = out.get("x0")
            lat_value = out.get("y0")
        csinfo = xds.attrs.get("coordinate_system_info", {})
        ref_dir = csinfo.get("reference_direction", {})
        phase_center = ref_dir.get("data")
        if phase_center is None:
            raise ValueError(
                "Sky-coordinate initial guesses require coordinate_system_info.reference_direction."
            )
        frame = ref_dir.get("attrs", {}).get("frame", "icrs")
        projection = csinfo.get("projection", "SIN")
        lon_rad, lat_rad = parse_sky_center_to_radians(lon_value, lat_value, frame)
        # imfit accepts sky-frame seed positions but fits in native l/m.
        l0, m0 = skycoord_to_lm_from_wcs(lon_rad, lat_rad, phase_center, projection)
        out.pop("x0", None)
        out.pop("y0", None)

    out["x0"] = world_value_to_pixel(l0, l_axis, "l")
    out["y0"] = world_value_to_pixel(m0, m_axis, "m")
    # imfit converts world-frame widths into pixel widths only at this boundary.
    _convert_world_widths_to_pixel(out, l_axis, m_axis)
    return out


def _normalize_imfit_initial_guesses(
    xds: xr.Dataset,
    initial_guesses: Any,
) -> Any:
    """Normalize imfit-specific initial guesses into pixel-frame fitter inputs.

    Parameters
    ----------
    xds : xr.Dataset
        Input xradio image Dataset.
    initial_guesses : Any
        Public initial-guess payload accepted by :func:`imfit`.

    Returns
    -------
    Any
        Payload with any world-frame component dictionaries converted into the
        pixel-frame representation expected by :func:`fit_multi_gaussian2d`.

    Notes
    -----
    Array-form guesses remain unchanged because they do not carry semantic field
    names to disambiguate pixel from world coordinates. Dictionary-form component
    guesses can opt into world-coordinate handling via explicit keys.
    """
    if initial_guesses is None:
        return None
    if isinstance(initial_guesses, Mapping):
        if "components" in initial_guesses or "offset" in initial_guesses:
            out = dict(initial_guesses)
            comps = out.get("components")
            if (
                isinstance(comps, (list, tuple))
                and len(comps) > 0
                and isinstance(comps[0], Mapping)
            ):
                out["components"] = [
                    _convert_world_component_guess_to_pixel(xds, comp) for comp in comps
                ]
            return out
        return _convert_world_component_guess_to_pixel(xds, initial_guesses)
    if (
        isinstance(initial_guesses, (list, tuple))
        and len(initial_guesses) > 0
        and isinstance(initial_guesses[0], Mapping)
    ):
        return [
            _convert_world_component_guess_to_pixel(xds, comp)
            for comp in initial_guesses
        ]
    return initial_guesses


def _validate_data_var(xds: xr.Dataset, data_var: str) -> xr.DataArray:
    """Extract and validate the image DataArray from an xradio Dataset.

    Parameters
    ----------
    xds : xr.Dataset
        Input xradio image Dataset.
    data_var : str
        Name of the data variable containing the image.

    Returns
    -------
    xr.DataArray
        The image DataArray, verified to contain ``l`` and ``m`` dimensions.

    Raises
    ------
    KeyError
        If *data_var* is not found in *xds*.
    ValueError
        If the DataArray does not contain ``l`` and ``m`` dimensions.
    """
    if data_var not in xds:
        raise KeyError(f"Data variable {data_var!r} not found in Dataset.")
    da = xds[data_var]
    if "l" not in da.dims or "m" not in da.dims:
        raise ValueError(
            f"Data variable {data_var!r} must have 'l' and 'm' dimensions, "
            f"found {da.dims!r}."
        )
    return da


def _resolve_mask(xds: xr.Dataset, mask_var: MaskSpec) -> Optional[MaskSpec]:
    """Validate the public mask input before shared selection resolution.

    Parameters
    ----------
    xds : xr.Dataset
        Input xradio image Dataset, used only for the legacy missing-name
        warning fallback.
    mask_var : str or pathlib.Path or numpy.ndarray or xarray.DataArray or dask.array.Array or None
        Mask specification. Supported choices are:
        - a Dataset variable name
        - a boolean array/DataArray aligned or broadcastable to the image
        - a CRTF/selection string understood by ``selection.select_mask``
        - a CRTF file path whose contents should be read and applied
        - ``None`` to skip masking
        String/path resolution order is handled centrally by the shared
        selection layer. imfit keeps only its historical warning-and-skip
        fallback for plain unknown variable names.

    Returns
    -------
    same type as ``mask_var`` or None
        The validated mask specification, or ``None`` when no mask should be
        applied.

    Raises
    ------
    TypeError
        If *mask_var* is not one of the supported mask input forms.
    """
    if mask_var is None:
        return None

    if isinstance(mask_var, (np.ndarray, xr.DataArray, da.Array)):
        return mask_var

    if isinstance(mask_var, Path):
        return mask_var

    if not isinstance(mask_var, str):
        raise TypeError(
            "mask_var must be None, a Dataset variable name, a boolean array/DataArray, "
            "a CRTF/selection string, or a CRTF file path; "
            f"got {type(mask_var).__name__}."
        )

    if mask_var in xds or not _looks_like_plain_missing_mask_name(mask_var):
        return mask_var

    warnings.warn(
        f"Mask variable {mask_var!r} not found in Dataset; "
        "proceeding without a mask.",
        stacklevel=3,
    )
    return None


def _looks_like_plain_missing_mask_name(mask_var: str) -> bool:
    """Return whether a string looks like a missing legacy mask variable name.

    Parameters
    ----------
    mask_var : str
        User-provided mask string supplied to :func:`imfit`.

    Returns
    -------
    bool
        ``True`` when the string is plain enough that imfit should preserve its
        historical "missing variable name warns and skips" behavior.
    """
    spec = mask_var.strip()
    if not spec:
        return False
    if spec.startswith("`") and spec.endswith("`"):
        return False
    if any(token in spec for token in "&|^~()[]"):
        return False
    if Path(spec).suffix.lower() == ".crtf" or "/" in spec or "\\" in spec:
        return False
    spec = spec.lstrip("\ufeff \t\r\n")
    if spec.startswith("#CRTF"):
        return False
    match = re.match(r"^([+-])?\s*([A-Za-z]+)\s*\[\[", spec, flags=re.IGNORECASE)
    return not bool(match and match.group(2).lower() in _CRTF_SHAPES)


def _resolve_beam(
    xds: xr.Dataset, beam_var: Optional[str]
) -> Optional[Tuple[xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Extract per-plane beam parameters (bmaj, bmin, bpa) in radians.

    Parameters
    ----------
    xds : xr.Dataset
        Input xradio image Dataset.
    beam_var : str or None
        Name of the beam data variable. ``None`` means no beam.

    Returns
    -------
    tuple of (bmaj, bmin, bpa) DataArrays or None
        Beam FWHM major, FWHM minor, and PA, all in radians with dims
        ``(time, frequency, polarization)`` (or a subset thereof).

    Raises
    ------
    ValueError
        If beam units are present but not a recognised angular unit.
    """
    if beam_var is None:
        return None
    if beam_var not in xds:
        warnings.warn(
            f"Beam variable {beam_var!r} not found in Dataset; "
            "deconvolution will be skipped.",
            stacklevel=3,
        )
        return None
    beam_da = xds[beam_var]
    unit_str = beam_da.attrs.get("units", "rad")
    if unit_str not in _ANGULAR_UNITS:
        raise ValueError(
            f"Beam units {unit_str!r} are not a recognised angular unit. "
            f"Expected one of {_ANGULAR_UNITS}."
        )
    scale = _TO_RAD[unit_str]
    bmaj = beam_da.sel(beam_params_label="major") * scale
    bmin = beam_da.sel(beam_params_label="minor") * scale
    bpa = beam_da.sel(beam_params_label="pa") * scale
    return bmaj, bmin, bpa


def _prune_math_angles(ds: xr.Dataset) -> xr.Dataset:
    """Remove math-convention and redundant PA angle variables.

    Parameters
    ----------
    ds : xr.Dataset
        Result Dataset from the lower-level fitter.

    Returns
    -------
    xr.Dataset
        Dataset with math-convention angles and pixel PA removed.
    """
    to_drop = [
        v
        for v in ds.data_vars
        if "_math" in v  # all math-convention angles
        or v.startswith("theta_pixel")  # pixel PA (≈ world PA)
    ]
    # When angle="pa", theta_world and theta_world_pa are both present and
    # identical. Drop the _pa variant first to avoid rename conflicts.
    for redundant in ("theta_world_pa", "theta_world_pa_err"):
        if redundant in ds.data_vars and redundant not in to_drop:
            to_drop.append(redundant)
    ds = ds.drop_vars(to_drop, errors="ignore")
    # Rename theta_world → pa, theta_world_err → pa_err
    renames = {}
    if "theta_world" in ds:
        renames["theta_world"] = "pa"
    if "theta_world_err" in ds:
        renames["theta_world_err"] = "pa_err"
    if renames:
        ds = ds.rename(renames)
    return ds


def _prepare_lm_radec_interpolation_grids(
    l_coord: np.ndarray,
    m_coord: np.ndarray,
    ra_grid: xr.DataArray,
    dec_grid: xr.DataArray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return l/m axes and RA/Dec grids in ascending interpolation order.

    Parameters
    ----------
    l_coord : np.ndarray
        1-D l coordinate axis values.
    m_coord : np.ndarray
        1-D m coordinate axis values.
    ra_grid : xr.DataArray
        2-D RA grid with its first axis matching *l_coord* and second axis
        matching *m_coord*.
    dec_grid : xr.DataArray
        2-D Dec grid with the same shape and axis convention as *ra_grid*.

    Returns
    -------
    tuple of np.ndarray
        Ascending l axis, ascending m axis, RA grid, and Dec grid.

    Notes
    -----
    xradio l axes are commonly descending. The interpolation arrays are
    reversed only for strictly descending axes, preserving the physical grid
    values while presenting scipy with ascending coordinate axes.
    """
    l1d = np.asarray(l_coord, dtype=float)
    m1d = np.asarray(m_coord, dtype=float)
    ra_np = np.asarray(ra_grid.values)
    dec_np = np.asarray(dec_grid.values)

    if l1d.ndim != 1 or m1d.ndim != 1:
        raise ValueError("l_coord and m_coord must be 1-D coordinate axes.")
    expected_shape = (l1d.size, m1d.size)
    if ra_np.shape != expected_shape or dec_np.shape != expected_shape:
        raise ValueError("RA/Dec grids must have shape (len(l_coord), len(m_coord)).")

    if l1d.size > 1 and np.all(np.diff(l1d) < 0):
        l1d = l1d[::-1]
        ra_np = ra_np[::-1, :]
        dec_np = dec_np[::-1, :]
    if m1d.size > 1 and np.all(np.diff(m1d) < 0):
        m1d = m1d[::-1]
        ra_np = ra_np[:, ::-1]
        dec_np = dec_np[:, ::-1]

    return l1d, m1d, ra_np, dec_np


def _lm_to_radec_from_grids(
    l_vals: xr.DataArray,
    m_vals: xr.DataArray,
    ra_grid: xr.DataArray,
    dec_grid: xr.DataArray,
    l_coord: np.ndarray,
    m_coord: np.ndarray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Interpolate (l, m) center positions into RA/Dec grids.

    Parameters
    ----------
    l_vals : xr.DataArray
        Fitted l-coordinate centers (``x0_world``), with a ``component`` dim.
    m_vals : xr.DataArray
        Fitted m-coordinate centers (``y0_world``), with a ``component`` dim.
    ra_grid : xr.DataArray
        2-D RA grid with dims ``(l, m)``.
    dec_grid : xr.DataArray
        2-D Dec grid with dims ``(l, m)``.
    l_coord : np.ndarray
        1-D l coordinate axis values.
    m_coord : np.ndarray
        1-D m coordinate axis values.

    Returns
    -------
    tuple of (ra, dec) DataArrays
        Interpolated sky coordinates, with the same outer dims as *l_vals*.
    """
    from scipy.interpolate import RegularGridInterpolator

    l1d, m1d, ra_np, dec_np = _prepare_lm_radec_interpolation_grids(
        l_coord, m_coord, ra_grid, dec_grid
    )

    ra_interp = RegularGridInterpolator(
        (l1d, m1d), ra_np, method="linear", bounds_error=False, fill_value=None
    )
    dec_interp = RegularGridInterpolator(
        (l1d, m1d), dec_np, method="linear", bounds_error=False, fill_value=None
    )

    l_np = np.asarray(l_vals.values, dtype=float)
    m_np = np.asarray(m_vals.values, dtype=float)
    pts = np.stack([l_np.ravel(), m_np.ravel()], axis=-1)
    ra_out = ra_interp(pts).reshape(l_np.shape)
    dec_out = dec_interp(pts).reshape(l_np.shape)
    return (
        xr.DataArray(ra_out, dims=l_vals.dims, coords=l_vals.coords),
        xr.DataArray(dec_out, dims=m_vals.dims, coords=m_vals.coords),
    )


def _lm_to_radec_from_wcs(
    l_vals: xr.DataArray,
    m_vals: xr.DataArray,
    phase_center: Sequence[float],
    frame: str,
    projection: str,
) -> Tuple[xr.DataArray, xr.DataArray, Dict[str, str]]:
    """Convert (l, m) direction cosines to RA/Dec via projection inverse.

    Parameters
    ----------
    l_vals : xr.DataArray
        Fitted l-coordinate centers (direction cosines, radians).
    m_vals : xr.DataArray
        Fitted m-coordinate centers (direction cosines, radians).
    phase_center : sequence of float
        ``[ra0, dec0]`` of the phase center in radians.
    frame : str
        Celestial frame (e.g. ``"fk5"``, ``"icrs"``, ``"galactic"``).
    projection : str
        WCS projection code (e.g. ``"SIN"``). Currently only ``"SIN"`` is
        implemented; other projections will warn and fall back to SIN.

    Returns
    -------
    ra : xr.DataArray
        Right ascension (or longitude) in radians.
    dec : xr.DataArray
        Declination (or latitude) in radians.
    coord_attrs : dict
        Attributes to attach to the coordinate variables (frame, etc.).

    Notes
    -----
    For the SIN (orthographic) projection used by radio interferometers,
    the direction cosines (l, m) relate to sky coordinates via:

        l = cos(dec) * sin(ra - ra0)
        m = sin(dec) * cos(dec0) - cos(dec) * sin(dec0) * cos(ra - ra0)

    This function applies the analytic inverse of these equations.
    """
    ra0, dec0 = float(phase_center[0]), float(phase_center[1])
    if projection != "SIN":
        warnings.warn(
            f"Projection {projection!r} not directly supported for l,m → "
            f"RA/Dec conversion; falling back to SIN projection.",
            stacklevel=3,
        )

    l_np = np.asarray(l_vals.values, dtype=float)
    m_np = np.asarray(m_vals.values, dtype=float)

    # Inverse SIN projection:
    # n = sqrt(1 - l² - m²)  (third direction cosine)
    # dec = arcsin(m * cos(dec0) + n * sin(dec0))
    # ra = ra0 + arctan2(l, n * cos(dec0) - m * sin(dec0))
    n = np.sqrt(np.maximum(1.0 - l_np**2 - m_np**2, 0.0))
    cos_dec0 = np.cos(dec0)
    sin_dec0 = np.sin(dec0)

    dec_rad = np.arcsin(np.clip(m_np * cos_dec0 + n * sin_dec0, -1.0, 1.0))
    ra_rad = ra0 + np.arctan2(l_np, n * cos_dec0 - m_np * sin_dec0)

    coord_attrs = {"frame": frame}

    return (
        xr.DataArray(ra_rad, dims=l_vals.dims, coords=l_vals.coords),
        xr.DataArray(dec_rad, dims=m_vals.dims, coords=m_vals.coords),
        coord_attrs,
    )


def _attach_sky_coordinates(ds: xr.Dataset, xds: xr.Dataset) -> xr.Dataset:
    """Add sky coordinates (RA/Dec) to the result Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Result Dataset containing ``x0_world`` (l) and ``y0_world`` (m) centers.
    xds : xr.Dataset
        Original xradio input Dataset, used for coordinate grids and metadata.

    Returns
    -------
    xr.Dataset
        Dataset with sky coordinate variables added.
    """
    if "x0_world" not in ds or "y0_world" not in ds:
        return ds

    l_vals = ds["x0_world"]
    m_vals = ds["y0_world"]

    has_grids = (
        "right_ascension" in xds.coords or "right_ascension" in xds.data_vars
    ) and ("declination" in xds.coords or "declination" in xds.data_vars)

    if has_grids:
        ra_grid = xds["right_ascension"]
        dec_grid = xds["declination"]
        l_coord = xds.coords["l"].values
        m_coord = xds.coords["m"].values
        ra, dec = _lm_to_radec_from_grids(
            l_vals, m_vals, ra_grid, dec_grid, l_coord, m_coord
        )
        ds["right_ascension"] = ra
        ds["declination"] = dec
        ds["right_ascension"].attrs[
            "description"
        ] = "Sky right ascension of component center."
        ds["declination"].attrs["description"] = "Sky declination of component center."
        # Propagate RA/Dec errors from l/m errors via local Jacobian
        if "x0_world_err" in ds and "y0_world_err" in ds:
            _attach_sky_coord_errors_from_grids(
                ds, l_vals, m_vals, ra_grid, dec_grid, l_coord, m_coord
            )
    else:
        csinfo = xds.attrs.get("coordinate_system_info", {})
        ref_dir = csinfo.get("reference_direction", {})
        phase_center = ref_dir.get("data")
        frame = ref_dir.get("attrs", {}).get("frame", "icrs")
        projection = csinfo.get("projection", "SIN")
        if phase_center is None:
            warnings.warn(
                "No sky coordinate grids and no coordinate_system_info with "
                "reference_direction found; sky coordinates cannot be computed.",
                stacklevel=3,
            )
            return ds
        ra, dec, coord_attrs = _lm_to_radec_from_wcs(
            l_vals, m_vals, phase_center, frame, projection
        )
        ds["Right Ascension"] = ra
        ds["Declination"] = dec
        ds["Right Ascension"].attrs[
            "description"
        ] = "Sky right ascension of component center."
        ds["Right Ascension"].attrs.update(coord_attrs)
        ds["Declination"].attrs["description"] = "Sky declination of component center."
        ds["Declination"].attrs.update(coord_attrs)
        # Propagate errors via WCS Jacobian
        if "x0_world_err" in ds and "y0_world_err" in ds:
            _attach_sky_coord_errors_from_wcs(
                ds, l_vals, m_vals, phase_center, frame, projection
            )
    return ds


def _attach_sky_coord_errors_from_grids(
    ds: xr.Dataset,
    l_vals: xr.DataArray,
    m_vals: xr.DataArray,
    ra_grid: xr.DataArray,
    dec_grid: xr.DataArray,
    l_coord: np.ndarray,
    m_coord: np.ndarray,
) -> None:
    """Propagate l/m center errors to RA/Dec errors via finite-difference Jacobian.

    Parameters
    ----------
    ds : xr.Dataset
        Result Dataset (modified in place).
    l_vals, m_vals : xr.DataArray
        Fitted l/m centers.
    ra_grid, dec_grid : xr.DataArray
        2-D RA/Dec grids.
    l_coord, m_coord : np.ndarray
        1-D l/m coordinate axes.
    """
    eps_l = np.mean(np.abs(np.diff(np.asarray(l_coord, dtype=float)))) * 1e-3
    eps_m = np.mean(np.abs(np.diff(np.asarray(m_coord, dtype=float)))) * 1e-3

    l_np = np.asarray(l_vals.values, dtype=float)
    m_np = np.asarray(m_vals.values, dtype=float)

    from scipy.interpolate import RegularGridInterpolator

    l1d, m1d, ra_np, dec_np = _prepare_lm_radec_interpolation_grids(
        l_coord, m_coord, ra_grid, dec_grid
    )
    ra_interp = RegularGridInterpolator(
        (l1d, m1d),
        ra_np,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    dec_interp = RegularGridInterpolator(
        (l1d, m1d),
        dec_np,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    def _eval(interp, l, m):
        pts = np.stack([l.ravel(), m.ravel()], axis=-1)
        return interp(pts).reshape(l.shape)

    dra_dl = (
        _eval(ra_interp, l_np + eps_l, m_np) - _eval(ra_interp, l_np - eps_l, m_np)
    ) / (2 * eps_l)
    dra_dm = (
        _eval(ra_interp, l_np, m_np + eps_m) - _eval(ra_interp, l_np, m_np - eps_m)
    ) / (2 * eps_m)
    ddec_dl = (
        _eval(dec_interp, l_np + eps_l, m_np) - _eval(dec_interp, l_np - eps_l, m_np)
    ) / (2 * eps_l)
    ddec_dm = (
        _eval(dec_interp, l_np, m_np + eps_m) - _eval(dec_interp, l_np, m_np - eps_m)
    ) / (2 * eps_m)

    l_err = np.asarray(ds["x0_world_err"].values, dtype=float)
    m_err = np.asarray(ds["y0_world_err"].values, dtype=float)

    ra_err = np.sqrt((dra_dl * l_err) ** 2 + (dra_dm * m_err) ** 2)
    dec_err = np.sqrt((ddec_dl * l_err) ** 2 + (ddec_dm * m_err) ** 2)

    ds["right_ascension_err"] = xr.DataArray(
        ra_err, dims=ds["x0_world_err"].dims, coords=ds["x0_world_err"].coords
    )
    ds["declination_err"] = xr.DataArray(
        dec_err, dims=ds["y0_world_err"].dims, coords=ds["y0_world_err"].coords
    )
    ds["right_ascension_err"].attrs[
        "description"
    ] = "1-sigma uncertainty of right ascension."
    ds["declination_err"].attrs["description"] = "1-sigma uncertainty of declination."


def _attach_sky_coord_errors_from_wcs(
    ds: xr.Dataset,
    l_vals: xr.DataArray,
    m_vals: xr.DataArray,
    phase_center: Sequence[float],
    frame: str,
    projection: str,
) -> None:
    """Propagate l/m errors to RA/Dec errors via WCS Jacobian.

    Parameters
    ----------
    ds : xr.Dataset
        Result Dataset (modified in place).
    l_vals, m_vals : xr.DataArray
        Fitted l/m centers.
    phase_center : sequence of float
        Phase center ``[ra0, dec0]`` in radians.
    frame : str
        Celestial frame.
    projection : str
        WCS projection code.
    """
    eps = 1e-8  # small step in radians for finite-difference

    l_np = np.asarray(l_vals.values, dtype=float)
    m_np = np.asarray(m_vals.values, dtype=float)

    def _eval_wcs(l, m):
        da_l = xr.DataArray(l, dims=l_vals.dims, coords=l_vals.coords)
        da_m = xr.DataArray(m, dims=m_vals.dims, coords=m_vals.coords)
        ra, dec, _ = _lm_to_radec_from_wcs(da_l, da_m, phase_center, frame, projection)
        return np.asarray(ra.values, dtype=float), np.asarray(dec.values, dtype=float)

    ra_pl, dec_pl = _eval_wcs(l_np + eps, m_np)
    ra_ml, dec_ml = _eval_wcs(l_np - eps, m_np)
    ra_pm, dec_pm = _eval_wcs(l_np, m_np + eps)
    ra_mm, dec_mm = _eval_wcs(l_np, m_np - eps)

    dra_dl = (ra_pl - ra_ml) / (2 * eps)
    dra_dm = (ra_pm - ra_mm) / (2 * eps)
    ddec_dl = (dec_pl - dec_ml) / (2 * eps)
    ddec_dm = (dec_pm - dec_mm) / (2 * eps)

    l_err = np.asarray(ds["x0_world_err"].values, dtype=float)
    m_err = np.asarray(ds["y0_world_err"].values, dtype=float)

    ra_err = np.sqrt((dra_dl * l_err) ** 2 + (dra_dm * m_err) ** 2)
    dec_err = np.sqrt((ddec_dl * l_err) ** 2 + (ddec_dm * m_err) ** 2)

    ra_name = "Right Ascension"
    dec_name = "Declination"
    ds[f"{ra_name} err"] = xr.DataArray(
        ra_err, dims=ds["x0_world_err"].dims, coords=ds["x0_world_err"].coords
    )
    ds[f"{dec_name} err"] = xr.DataArray(
        dec_err, dims=ds["y0_world_err"].dims, coords=ds["y0_world_err"].coords
    )
    ds[f"{ra_name} err"].attrs[
        "description"
    ] = "1-sigma uncertainty of Right Ascension."
    ds[f"{dec_name} err"].attrs["description"] = "1-sigma uncertainty of Declination."


def _deconvolve_and_attach(
    ds: xr.Dataset,
    beam: Tuple[xr.DataArray, xr.DataArray, xr.DataArray],
) -> xr.Dataset:
    """Deconvolve fitted components from the beam and attach results.

    Parameters
    ----------
    ds : xr.Dataset
        Result Dataset containing world-frame source sizes and PA.
    beam : tuple of (bmaj, bmin, bpa) DataArrays
        Beam FWHM major, FWHM minor, and PA in radians. Dims are a subset
        of ``(time, frequency, polarization)``.

    Returns
    -------
    xr.Dataset
        Dataset with deconvolved quantities, flags, and upper limits added.
    """
    bmaj, bmin, bpa = beam

    src_fwhm_maj = ds["fwhm_major_world"]
    src_fwhm_min = ds["fwhm_minor_world"]
    src_pa = ds["pa"]

    # Broadcast beam to match the component dimension
    # Beam has no 'component' dim; xarray broadcast will handle it.
    has_errors = (
        "fwhm_major_world_err" in ds and "fwhm_minor_world_err" in ds and "pa_err" in ds
    )

    if has_errors:
        results = xr.apply_ufunc(
            deconvolve_gaussian_with_errors,
            src_fwhm_maj,
            src_fwhm_min,
            src_pa,
            ds["fwhm_major_world_err"],
            ds["fwhm_minor_world_err"],
            ds["pa_err"],
            bmaj,
            bmin,
            bpa,
            input_core_dims=[["component"]] * 6 + [[]] * 3,
            output_core_dims=[["component"]] * 8,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float, float, float, float, bool, bool],
        )
        (
            d_fwhm_maj,
            d_fwhm_min,
            d_pa,
            d_fwhm_maj_err,
            d_fwhm_min_err,
            d_pa_err,
            is_unresolved,
            is_marginally_resolved,
        ) = results
    else:
        results = xr.apply_ufunc(
            deconvolve_gaussian,
            src_fwhm_maj,
            src_fwhm_min,
            src_pa,
            bmaj,
            bmin,
            bpa,
            input_core_dims=[["component"]] * 3 + [[]] * 3,
            output_core_dims=[["component"]] * 4,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float, bool],
        )
        d_fwhm_maj, d_fwhm_min, d_pa, is_unresolved = results
        is_marginally_resolved = None

    # World-frame deconvolved quantities
    ds["fwhm_major_deconv"] = d_fwhm_maj
    ds["fwhm_minor_deconv"] = d_fwhm_min
    ds["pa_deconv"] = d_pa
    ds["sigma_major_deconv"] = d_fwhm_maj * FWHM2SIG
    ds["sigma_minor_deconv"] = d_fwhm_min * FWHM2SIG
    ds["is_unresolved"] = is_unresolved

    ds["fwhm_major_deconv"].attrs[
        "description"
    ] = "Deconvolved major-axis FWHM in world units. NaN if unresolved."
    ds["fwhm_minor_deconv"].attrs[
        "description"
    ] = "Deconvolved minor-axis FWHM in world units. NaN if unresolved."
    ds["pa_deconv"].attrs[
        "description"
    ] = "Deconvolved position angle (east of north) in radians. NaN if unresolved."
    ds["sigma_major_deconv"].attrs[
        "description"
    ] = "Deconvolved major-axis sigma in world units. NaN if unresolved."
    ds["sigma_minor_deconv"].attrs[
        "description"
    ] = "Deconvolved minor-axis sigma in world units. NaN if unresolved."
    ds["is_unresolved"].attrs[
        "description"
    ] = "True if the source is unresolved (deconvolution failed)."

    # Upper limit: beam major FWHM when unresolved
    fwhm_upper = xr.where(is_unresolved, bmaj, np.nan)
    ds["fwhm_upper_limit"] = fwhm_upper
    ds["fwhm_upper_limit"].attrs["description"] = (
        "Beam major-axis FWHM as upper limit when source is unresolved. "
        "NaN when resolved."
    )

    if has_errors:
        ds["fwhm_major_deconv_err"] = d_fwhm_maj_err
        ds["fwhm_minor_deconv_err"] = d_fwhm_min_err
        ds["pa_deconv_err"] = d_pa_err
        ds["sigma_major_deconv_err"] = d_fwhm_maj_err * FWHM2SIG
        ds["sigma_minor_deconv_err"] = d_fwhm_min_err * FWHM2SIG
        ds["fwhm_major_deconv_err"].attrs[
            "description"
        ] = "1-sigma uncertainty of deconvolved major FWHM."
        ds["fwhm_minor_deconv_err"].attrs[
            "description"
        ] = "1-sigma uncertainty of deconvolved minor FWHM."
        ds["pa_deconv_err"].attrs[
            "description"
        ] = "1-sigma uncertainty of deconvolved PA."
        ds["sigma_major_deconv_err"].attrs[
            "description"
        ] = "1-sigma uncertainty of deconvolved major sigma."
        ds["sigma_minor_deconv_err"].attrs[
            "description"
        ] = "1-sigma uncertainty of deconvolved minor sigma."

    if is_marginally_resolved is not None:
        ds["is_marginally_resolved"] = is_marginally_resolved
        ds["is_marginally_resolved"].attrs["description"] = (
            "True if deconvolution succeeded but the source is close to the "
            "beam size, making error estimates unreliable."
        )

    # Pixel-frame deconvolved sizes: convert world deconvolved widths to pixel
    # using the same local scale as the fitter.
    if "sigma_major_pixel" in ds and "sigma_major_world" in ds:
        # Compute pixel-to-world scale from the existing fitter outputs
        scale_maj = ds["sigma_major_world"] / xr.where(
            ds["sigma_major_pixel"] > 0, ds["sigma_major_pixel"], np.nan
        )
        scale_min = ds["sigma_minor_world"] / xr.where(
            ds["sigma_minor_pixel"] > 0, ds["sigma_minor_pixel"], np.nan
        )
        # Use average scale for deconvolved pixel sizes
        avg_scale = (np.abs(scale_maj) + np.abs(scale_min)) / 2.0
        ds["fwhm_major_deconv_pixel"] = d_fwhm_maj / avg_scale
        ds["fwhm_minor_deconv_pixel"] = d_fwhm_min / avg_scale
        ds["sigma_major_deconv_pixel"] = d_fwhm_maj * FWHM2SIG / avg_scale
        ds["sigma_minor_deconv_pixel"] = d_fwhm_min * FWHM2SIG / avg_scale
        ds["fwhm_major_deconv_pixel"].attrs[
            "description"
        ] = "Deconvolved major-axis FWHM in pixels. NaN if unresolved."
        ds["fwhm_minor_deconv_pixel"].attrs[
            "description"
        ] = "Deconvolved minor-axis FWHM in pixels. NaN if unresolved."
        ds["sigma_major_deconv_pixel"].attrs[
            "description"
        ] = "Deconvolved major-axis sigma in pixels. NaN if unresolved."
        ds["sigma_minor_deconv_pixel"].attrs[
            "description"
        ] = "Deconvolved minor-axis sigma in pixels. NaN if unresolved."

        if has_errors:
            ds["fwhm_major_deconv_pixel_err"] = d_fwhm_maj_err / avg_scale
            ds["fwhm_minor_deconv_pixel_err"] = d_fwhm_min_err / avg_scale
            ds["sigma_major_deconv_pixel_err"] = d_fwhm_maj_err * FWHM2SIG / avg_scale
            ds["sigma_minor_deconv_pixel_err"] = d_fwhm_min_err * FWHM2SIG / avg_scale

    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def imfit(
    xds: xr.Dataset,
    n_components: int,
    *,
    data_var: str = "SKY",
    mask_var: MaskSpec = "FLAGS_SKY",
    beam_var: Optional[str] = "BEAM_FIT_PARAMS_SKY",
    min_threshold: Optional[Number] = None,
    max_threshold: Optional[Number] = None,
    initial_guesses=None,
    bounds=None,
    initial_is_fwhm: bool = True,
    max_nfev: int = 20000,
    return_model: bool = False,
    return_residual: bool = True,
) -> xr.Dataset:
    """Fit 2-D Gaussians to an xradio image Dataset.

    This is the astronomer-facing entry point for image-plane Gaussian
    fitting. It accepts an xradio image Dataset, delegates the numerical
    fit to :func:`fit_multi_gaussian2d`, then enriches the result with sky
    coordinates, beam deconvolution, and clean astronomical-convention
    outputs.

    Parameters
    ----------
    xds : xr.Dataset
        xradio image Dataset. Must contain the specified *data_var* with
        at least ``l`` and ``m`` dimensions.
    n_components : int
        Number of Gaussian components to fit (N >= 1).
    data_var : str
        Name of the image data variable. Default ``"SKY"``.
    mask_var : str or pathlib.Path or numpy.ndarray or xarray.DataArray or dask.array.Array or None
        Mask specification (True = good pixel). A string may name a Dataset
        variable, provide a CRTF/selection string, or name a CRTF file whose
        contents should be read. Boolean array/DataArray inputs must be aligned
        or broadcastable to the image. Default
        ``"FLAGS_SKY"``. String/path resolution order is shared with the generic
        selection layer: Dataset variable name, then CRTF file contents, then
        CRTF/selection text. Set to ``None`` to skip masking.
    beam_var : str or None
        Name of the beam parameter data variable. Default
        ``"BEAM_FIT_PARAMS_SKY"``. Must have a ``beam_params_label``
        dimension with values ``["major", "minor", "pa"]`` (all FWHM).
        Set to ``None`` to skip deconvolution.
    min_threshold : float or None
        Inclusive lower threshold; pixels below this are excluded from the fit.
    max_threshold : float or None
        Inclusive upper threshold; pixels above this are excluded from the fit.
    initial_guesses : array-like or list[dict] or dict or None
        Initial guesses for the fit, passed through to
        :func:`fit_multi_gaussian2d`. Array-form guesses remain pixel-frame.
        Dictionary-form component guesses may use pixel centers ``x0``/``y0``,
        native world centers ``l0``/``m0``, equatorial sky keys
        ``ra``/``dec`` or ``right_ascension``/``declination``, or generic
        frame-aware sky keys ``lon``/``lat`` or ``longitude``/``latitude``.
        Sexagesimal sky strings are accepted. World-frame centers and widths are
        converted into pixel-frame guesses before optimization so the fitter sees
        numerically well-scaled initial values. World-frame initial widths are
        only supported for square pixels.
    bounds : dict or None
        Parameter bounds, passed through to :func:`fit_multi_gaussian2d`.
    initial_is_fwhm : bool
        If True (default), width columns in array-form *initial_guesses*
        are FWHM; if False they are sigma.
    max_nfev : int
        Maximum function evaluations for the optimizer. Default 20000.
    return_model : bool
        If True, include the fitted model plane(s) in the result.
    return_residual : bool
        If True (default), include residual (data - model) planes.

    Returns
    -------
    xr.Dataset
        Result Dataset containing:

        From the lower-level fitter:
          - ``amplitude``, ``amplitude_err`` — component amplitudes
          - ``peak``, ``peak_err`` — peak values (amplitude + offset)
          - ``x0_pixel``, ``y0_pixel`` — pixel-frame centers (+ ``*_err``)
          - ``x0_world``, ``y0_world`` — l/m world centers (+ ``*_err``)
          - ``sigma_major_pixel``, ``sigma_minor_pixel`` — pixel widths (+ ``*_err``)
          - ``sigma_major_world``, ``sigma_minor_world`` — world widths (+ ``*_err``)
          - ``fwhm_major_pixel``, ``fwhm_minor_pixel`` — pixel FWHM (+ ``*_err``)
          - ``fwhm_major_world``, ``fwhm_minor_world`` — world FWHM (+ ``*_err``)
          - ``pa``, ``pa_err`` — position angle (east of north, radians)
          - ``offset``, ``offset_err``, ``success``, ``variance_explained``
          - ``residual`` (if *return_residual*), ``model`` (if *return_model*)

        Added by imfit:
          - ``right_ascension``, ``declination`` (or ``Right Ascension``,
            ``Declination``) — sky coordinates of component centers
          - Corresponding ``*_err`` uncertainty variables

        If a beam is provided:
          - ``fwhm_major_deconv``, ``fwhm_minor_deconv``, ``pa_deconv`` —
            deconvolved source sizes in world units (+ ``*_err``)
          - ``sigma_major_deconv``, ``sigma_minor_deconv`` (+ ``*_err``)
          - ``fwhm_major_deconv_pixel``, ``fwhm_minor_deconv_pixel`` (+ ``*_err``)
          - ``sigma_major_deconv_pixel``, ``sigma_minor_deconv_pixel`` (+ ``*_err``)
          - ``is_unresolved`` — boolean flag per component per plane
          - ``is_marginally_resolved`` — boolean flag for unreliable errors
          - ``fwhm_upper_limit`` — beam major FWHM when unresolved

    Notes
    -----
    The fit is performed internally on zero-based pixel axes even though this
    wrapper accepts world-coordinate initial guesses and publishes world-frame
    results. Position angles are reported in the standard
    astronomical convention: measured east of north (counter-clockwise from
    "up" in a left-handed coordinate system). Math-convention angles are
    not included in the output.

    The beam is matched per-plane: each ``(time, frequency, polarization)``
    slice uses its own beam for deconvolution.
    """
    # Stage 1: validate and extract inputs
    image_da = _validate_data_var(xds, data_var)
    mask_da = _resolve_mask(xds, mask_var)
    beam = _resolve_beam(xds, beam_var)
    initial_guesses = _normalize_imfit_initial_guesses(xds, initial_guesses)

    # Stage 2: delegate to the generic fitter.
    # Use coord_type="pixel" for numerical stability (l/m values can be very
    # small in radians). The fitter still publishes world-frame results by
    # converting pixel results through the attached coordinate metadata.
    ds = fit_multi_gaussian2d(
        image_da,
        n_components,
        dims=("l", "m"),
        mask=mask_da,
        mask_source=xds,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        initial_guesses=initial_guesses,
        bounds=bounds,
        initial_is_fwhm=initial_is_fwhm,
        max_nfev=max_nfev,
        return_model=return_model,
        return_residual=return_residual,
        angle="pa",
        coord_type="pixel",
    )

    # Stage 3: translate l/m centers to sky coordinates
    ds = _attach_sky_coordinates(ds, xds)

    # Stage 4: prune math-convention and redundant pixel-PA angles
    ds = _prune_math_angles(ds)

    # Stage 5: deconvolve from beam
    if beam is not None and "fwhm_major_world" in ds:
        ds = _deconvolve_and_attach(ds, beam)

    # Stage 6: attach metadata
    for attr_name in ("coordinate_system_info",):
        if attr_name in xds.attrs:
            ds.attrs[attr_name] = xds.attrs[attr_name]
    ds.attrs["data_var"] = data_var
    ds.attrs["beam_var"] = beam_var
    ds.attrs["theta_convention"] = "pa"
    ds.attrs["pa_definition"] = "east of north (CCW from +y in left-handed system)"

    return ds
