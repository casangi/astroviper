"""Primary-beam prescriptions for continuum imaging.

The CASA response is adapted from the existing implementation on
``origin/265-port-sirius``. It reproduces PBMath1DAiry's obscuration convention,
radial constants, and lower-sample lookup. The physical Airy path remains the
existing implementation in make_pb_symmetric.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1

SPEED_OF_LIGHT = 299792458.0
CASA_AIRY_TWIDDLE = (180 * 7.016 * SPEED_OF_LIGHT) / ((np.pi**2) * 1e9 * 1.566 * 24.5)
CASA_AIRY_N_SAMPLE = 10000


def casa_airy_disk_response(
    l: np.ndarray,  # noqa: E741
    m: np.ndarray,
    frequency: np.ndarray,
    dish_diameter: float,
    blockage_diameter: float,
    max_rad_1GHz: float,
    ipower: int = 1,
    n_sample: int | None = CASA_AIRY_N_SAMPLE,
) -> np.ndarray:
    """CASA ``PBMath1DAiry`` compatible Airy-disk response.

    CASA tabulates the pattern on ``n_sample`` radii out to ``max_rad_1GHz``
    (scaled to the observing frequency) and truncates to the nearest lower sample;
    :data:`CASA_AIRY_TWIDDLE` reproduces its truncated constants.  Directions
    beyond ``max_rad_1GHz / (frequency / 1 GHz)`` are zeroed, matching the
    support of CASA's lookup table.

    Parameters
    ----------
    l, m : np.ndarray (broadcastable), radians
    frequency : np.ndarray (broadcastable), Hz
    dish_diameter, blockage_diameter : float, metres
    max_rad_1GHz : float, radians
    ipower : int
    n_sample : int or None
        ``None`` disables lookup quantisation while retaining CASA's radial scale.

    Returns
    -------
    np.ndarray
    """
    frequency = np.asarray(frequency, dtype=np.float64)
    k = 2 * np.pi * frequency / SPEED_OF_LIGHT
    aperture = dish_diameter / 2
    rho = np.sqrt(
        np.asarray(l, dtype=np.float64) ** 2 + np.asarray(m, dtype=np.float64) ** 2
    )
    if n_sample is not None:
        # PBMath1D applies its Float image using squared pixel offsets in
        # degrees, a Float radius in arcmin GHz, and truncation to a table bin.
        # Preserving those roundings avoids one-bin errors near boundaries.
        degree_l = np.rad2deg(np.asarray(l, dtype=np.float64))
        degree_m = np.rad2deg(np.asarray(m, dtype=np.float64))
        radius_squared = (degree_l**2).astype(np.float32) + (degree_m**2).astype(
            np.float32
        )
        radius = (np.sqrt(radius_squared) * (60.0 * frequency / 1e9)).astype(np.float32)
        max_arcmin = np.rad2deg(max_rad_1GHz) * 60.0
        index = np.trunc(radius.astype(np.float64) * ((n_sample - 1) / max_arcmin))
        # Evaluate Bessel functions only on the lookup table, not every pixel.
        r = (
            np.arange(n_sample, dtype=np.float64)
            / (n_sample - 1)
            * max_rad_1GHz
            * aperture
            * (2 * np.pi * 1e9 / SPEED_OF_LIGHT)
            * CASA_AIRY_TWIDDLE
        )
    else:
        r = rho * k * aperture * CASA_AIRY_TWIDDLE
    r = np.asarray(r)
    safe = np.where(r == 0, 1.0, r)
    if blockage_diameter == 0.0:
        val = 2.0 * j1(safe) / safe
    else:
        area_ratio = (dish_diameter / blockage_diameter) ** 2
        length_ratio = dish_diameter / blockage_diameter
        val = (
            area_ratio * 2.0 * j1(safe) / safe
            - 2.0 * j1(safe * length_ratio) / (safe * length_ratio)
        ) / (area_ratio - 1.0)
    val = np.where(r == 0, 1.0, val)
    if n_sample is not None:
        # The CASA voltage table and output image are single precision.
        val = val.astype(np.float32)[np.clip(index, 0, n_sample - 1).astype(np.int64)]
    return np.where(rho <= max_rad_1GHz / (frequency / 1e9), val**ipower, 0.0)


def resolve_continuum_primary_beam(image_params, antenna_xds):
    """Resolve a beam prescription without changing user parameters or metadata.

    Parameters
    ----------
    image_params : dict
        Imaging parameters. ``primary_beam_model`` accepts ``auto`` (default),
        ``airy`` (physical aperture), or ``casa_airy``. Explicit dish, blockage,
        and ``primary_beam_max_radius_1ghz`` parameters take precedence.
    antenna_xds : xarray.Dataset
        Antenna metadata with physical diameters and telescope identification.

    Returns
    -------
    dict
        Copied parameters with a resolved prescription and aperture diameters.
        Auto uses CASA's effective aperture only for ALMA/ACA; other telescopes
        retain their physical diameter and existing Airy model.
    """
    params = dict(image_params)
    model = params.get("primary_beam_model", "auto")
    if model not in ("auto", "airy", "casa_airy"):
        raise ValueError("primary_beam_model must be 'auto', 'airy', or 'casa_airy'.")
    if "telescope_name" in antenna_xds:
        names = {
            str(name).strip().upper()
            for name in antenna_xds.telescope_name.values.ravel()
        }
    else:
        names = {
            str(antenna_xds.attrs.get("overall_telescope_name", "")).strip().upper()
        }
    is_alma = bool(names) and names.issubset({"ALMA", "ACA", "ALMASD"})
    if model == "auto":
        model = "casa_airy" if is_alma else "airy"
    params["primary_beam_model"] = model

    diameter_name = "ANTENNA_DISH_DIAMETER"
    physical = None
    if diameter_name in antenna_xds:
        values = np.asarray(antenna_xds[diameter_name].values, dtype=float)
        diameters = np.unique(values[np.isfinite(values)])
        if diameters.size == 1:
            physical = float(diameters[0])
    if params.get("list_dish_diameters") is None:
        if diameter_name not in antenna_xds:
            raise KeyError(
                f"Continuum primary-beam construction requires {diameter_name}."
            )
        if physical is None or physical <= 0:
            raise NotImplementedError(
                "Continuum primary beams require one common positive antenna dish diameter."
            )
        effective = physical
        if is_alma and model == "casa_airy":
            if np.isclose(physical, 12.0):
                effective = 10.7
            elif np.isclose(physical, 7.0):
                effective = 6.25
            else:
                raise ValueError(
                    "CASA-compatible ALMA beams require a 12-m or 7-m aperture, or an explicit effective diameter."
                )
        params["list_dish_diameters"] = [effective]
    if params.get("list_blockage_diameters") is None:
        params["list_blockage_diameters"] = [0.75]
    if model == "casa_airy":
        params.setdefault(
            "primary_beam_max_radius_1ghz",
            np.deg2rad(
                3.568 if physical is not None and np.isclose(physical, 7.0) else 1.784
            ),
        )
    return params


def evaluate_primary_beam(freq_chan, pol, pb_params, grid_params, dtype=None):
    """Evaluate the selected beam in dish/frequency/polarization/l/m order.

    Parameters
    ----------
    freq_chan, pol : array-like
        Frequencies in Hz and polarization labels.
    pb_params : dict
        Dish and blockage diameter lists and voltage/power exponent ``ipower``.
    grid_params : dict
        Image geometry and resolved ``primary_beam_model`` prescription.
    dtype : numpy.dtype, optional
        Output dtype, default float64.

    Returns
    -------
    numpy.ndarray
        Beam array with shape (dish, frequency, polarization, l, m).
    """
    from astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric import (
        airy_disk_rorder_v2,
    )

    model = grid_params.get("primary_beam_model", "airy")
    if model == "airy":
        return airy_disk_rorder_v2(freq_chan, pol, pb_params, grid_params, dtype=dtype)
    if model != "casa_airy":
        raise ValueError("Resolve primary_beam_model before evaluating the beam.")
    if dtype is None:
        dtype = np.float64
    shape = grid_params["image_size"]
    center = grid_params["image_center"]
    cell = grid_params["cell_size"]
    l_axis = (np.arange(shape[0]) - center[0]) * cell[0]
    m_axis = (np.arange(shape[1]) - center[1]) * cell[1]
    frequencies = np.asarray(freq_chan)
    dishes = pb_params["list_dish_diameters"]
    blocks = pb_params["list_blockage_diameters"]
    result = np.empty((len(dishes), len(frequencies), 1, *shape), dtype=dtype)
    maximum_radius = grid_params.get("primary_beam_max_radius_1ghz", np.deg2rad(1.784))
    for dd, (dish, blockage) in enumerate(zip(dishes, blocks, strict=True)):
        for ff, frequency in enumerate(frequencies):
            result[dd, ff, 0] = casa_airy_disk_response(
                l_axis[:, None],
                m_axis[None, :],
                frequency,
                dish,
                blockage,
                maximum_radius,
                ipower=pb_params["ipower"],
            )
    return np.broadcast_to(result, (len(dishes), len(frequencies), len(pol), *shape))
