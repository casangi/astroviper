"""Primary-beam prescriptions for continuum imaging.

These prescriptions are used only by continuum MFS/MVC callers. The shared
``airy_disk`` and ``make_primary_beam`` utilities used by cube imaging and
simulation deliberately retain their existing behavior. This module reproduces
CASA PBMath1DAiry's image-table rounding, support and telescope selection.
EVLA polynomial prescriptions are not implemented here.
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


def resolve_continuum_primary_beam(image_params, antenna_xds, *, specmode="mfs"):
    """Resolve a beam prescription without changing user parameters or metadata.

    Parameters
    ----------
    image_params : dict
        Imaging parameters. ``primary_beam_model`` accepts ``auto`` (default),
        ``airy`` (physical aperture), or ``casa_airy``. Explicit dish, blockage,
        and ``primary_beam_max_radius_1ghz`` parameters take precedence.
    antenna_xds : xarray.Dataset
        Antenna metadata with physical diameters and telescope identification.
    specmode : {"mfs", "mvc"}
        MFS selects the VLA band at the continuum reference frequency; MVC
        uses the first image channel, matching CASA makePBImage. Resolve once
        before partitioning so every task uses the same prescription.

    Returns
    -------
    dict
        Copied parameters with a resolved prescription and aperture diameters.
        Auto uses CASA's effective aperture for ALMA/ACA and its squint-free
        Airy prescriptions for legacy VLA bands (including the NVSS fallback).
        EVLA retains the physical Airy model. Explicit aperture parameters
        remain authoritative.
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
    # PBMath::whichCommonPBtoUse selects these legacy VLA bands. Each uses
    # PBMath1DAiry(25 m, 2.36 m, 0.8564 deg at 1 GHz); EVLA instead uses
    # frequency-dependent polynomial beams. In band gaps CASA selects VLA_NVSS,
    # which instead uses a 24.5-m unobstructed aperture with the same support.
    reference_frequency = params.get(
        "reference_frequency", params.get("reference_frequency_hz")
    )
    frequencies = np.asarray(params.get("frequency_coords", []), dtype=float)
    if specmode == "mvc":
        reference_frequency = frequencies.flat[0] if frequencies.size else np.nan
    elif specmode != "mfs":
        raise ValueError("specmode must be 'mfs' or 'mvc'.")
    elif reference_frequency is None:
        reference_frequency = np.mean(frequencies) if frequencies.size else np.nan
    frequency_ghz = float(reference_frequency) / 1e9
    is_vla = (
        bool(names)
        and all(name.startswith("VLA") for name in names)
        and np.isfinite(frequency_ghz)
        and frequency_ghz > 0
    )
    is_vla_band = is_vla and any(
        lower < frequency_ghz < upper
        for lower, upper in (
            (0, 0.1),
            (0.2, 0.4),
            (1, 2),
            (4, 7),
            (7, 11),
            (11, 19),
            (19, 35),
            (35, 55),
        )
    )
    if model == "auto":
        model = "casa_airy" if is_alma or is_vla else "airy"
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
        if is_vla and model == "casa_airy":
            effective = 25.0 if is_vla_band else 24.5
        params["list_dish_diameters"] = [effective]
    if params.get("list_blockage_diameters") is None:
        blockage = 0.75
        if is_vla and model == "casa_airy":
            blockage = 2.36 if is_vla_band else 0.0
        params["list_blockage_diameters"] = [blockage]
    if model == "casa_airy":
        params.setdefault(
            "primary_beam_max_radius_1ghz",
            np.deg2rad(
                0.8564
                if is_vla
                else (
                    3.568
                    if physical is not None and np.isclose(physical, 7.0)
                    else 1.784
                )
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


def make_continuum_primary_beam_single_field(
    img_xds,
    image_params,
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    list_dish_diameters=None,
    list_blockage_diameters=None,
    ipower=2,
    float_dtype=None,
):
    """Add an azimuthally-symmetric primary beam to a single-field image dataset.

    Evaluates the selected Airy prescription via
    :func:`~astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam.evaluate_primary_beam`
    for every frequency channel and writes it as the ``PRIMARY_BEAM`` data
    variable, registering it under ``image_data_group_out_name``.  The same beam
    is broadcast across polarization (a single dish diameter is assumed).

    The beam is created in whatever polarization basis ``img_xds`` currently
    carries; because ``PRIMARY_BEAM`` is stamped with ``method="airy_disk"`` it
    is skipped by
    :func:`~astroviper.processing_functions.image_analysis.transform_polarization_basis.transform_polarization_basis`,
    so it is invariant to a later basis change.  If the data group already has a
    ``primary_beam`` entry the function is a no-op.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset with ``time``, ``frequency``, ``polarization``, ``l`` and
        ``m`` coordinates.  Modified in place.
    image_params : dict
        Image geometry.  Must contain ``"image_size"`` (``(nx, ny)``) and
        ``"cell_size"`` (l, m cell size in radians).  The image centre is
        derived internally as ``image_size // 2``; the supplied dictionary is
        not mutated. ``primary_beam_model`` selects ``airy`` (the default for
        direct calls) or ``casa_airy``. The continuum driver resolves ``auto``
        from telescope metadata before calling this function.
    image_data_group_in_name : str, optional
        Image data group whose existing entries are carried into the output
        group.  Default ``"residual"``.
    image_data_group_out_name : str, optional
        Name of the data group that the ``PRIMARY_BEAM`` variable is registered
        under.  Default ``"residual"``.
    list_dish_diameters : array-like of float, optional
        Antenna dish diameters in metres.  Default ``numpy.array([10.7])``
        (ALMA 12 m array effective value used by the reference imaging tests).
    list_blockage_diameters : array-like of float, optional
        Sub-reflector blockage diameters in metres.  Default
        ``numpy.array([0.75])``.
    ipower : int, optional
        ``1`` returns the voltage pattern, ``2`` returns the power beam.
        Default ``2`` (power primary beam).
    float_dtype : numpy.dtype, optional
        Floating-point precision for the primary beam.  Defaults to
        ``numpy.float64``.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with ``PRIMARY_BEAM`` added (or unchanged if it
        already existed).
    return_df : pandas.DataFrame
        One-row timing frame with the ``T_primary_beam`` column.

    See Also
    --------
    astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric.airy_disk_rorder_v2
    astroviper.processing_functions.imaging.make_point_spread_function.make_point_spread_function_single_field
    """
    import time

    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    if float_dtype is None:
        float_dtype = np.float64
    if list_dish_diameters is None:
        list_dish_diameters = np.array([10.7])
    if list_blockage_diameters is None:
        list_blockage_diameters = np.array([0.75])

    start = time.time()

    pb_params = {
        "list_dish_diameters": np.asarray(list_dish_diameters),
        "list_blockage_diameters": np.asarray(list_blockage_diameters),
        "ipower": ipower,
    }

    # The airy-disk evaluation needs the image centre in pixels. Build a local
    # copy of image_params so the caller's dictionary is not mutated.
    pb_image_params = {
        **image_params,
        "image_center": (np.array(image_params["image_size"]) // 2).tolist(),
    }

    data_group = img_xds.attrs["data_groups"][image_data_group_in_name]
    if data_group.get("primary_beam", None) is not None:
        return img_xds, pd.DataFrame({"T_primary_beam": [0.0]})

    image_data_group_in, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified={"primary_beam": "PRIMARY_BEAM"},
        overwrite=False,
    )

    img_xds["PRIMARY_BEAM"] = xr.DataArray(
        # Select the first (only) dish diameter and add a leading time axis.
        evaluate_primary_beam(
            img_xds.frequency.values,
            img_xds.polarization.values,
            pb_params,
            pb_image_params,
            dtype=float_dtype,
        )[0, ...][None, ...],
        dims=("time", "frequency", "polarization", "l", "m"),
    )
    img_xds["PRIMARY_BEAM"].attrs["type"] = "primary_beam"
    img_xds["PRIMARY_BEAM"].attrs["method"] = "airy_disk"
    img_xds["PRIMARY_BEAM"].attrs.update(
        beam_model=image_params.get("primary_beam_model", "airy"),
        dish_diameters_m=np.asarray(list_dish_diameters).tolist(),
        blockage_diameters_m=np.asarray(list_blockage_diameters).tolist(),
    )

    modify_data_groups_xds(
        img_xds,
        data_group_out_name=image_data_group_out_name,
        data_group_out=image_data_group_out,
        description="Added primary beam to data group.",
    )

    return img_xds, pd.DataFrame({"T_primary_beam": [time.time() - start]})
