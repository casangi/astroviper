"""Antenna (voltage) beam models: evaluation, Jones images and sampling.

Beam models are described in :mod:`astroviper.utils.beam_models`.  This module
turns them into what the visibility kernel needs:

* analytic Airy-disk responses (``casa_airy`` reproduces CASA ``PBMath1DAiry``
  including its 10000-point lookup quantisation and constant truncation);
* 1-D beam polynomials (CASA ``PBMath1DPoly``);
* Zernike aperture coefficients → Jones beam images
  ``JONES[parallactic_angle, frequency, polarization, l, m]`` via an inverse FFT
  of the aperture illumination (:func:`make_zernike_jones_beam`);
* :func:`evaluate_beam_models` (SIRIUS ``evaluate_beam_models``) which computes
  parallactic angles and converts every Zernike model in a list to Jones images
  on a reduced set of parallactic angles;
* :func:`pack_beam_models` / :func:`sample_jones` — the flat representation and
  the per-direction Jones sampler (analytic / polynomial / bilinear on images)
  used by the NumPy visibility kernel, and :func:`apply_mueller` which scales
  the 4-correlation flux by the Mueller matrix of two sampled Jones vectors.

Jones vectors are the row-major flattened 2x2 matrix ``[J_pp, J_pq, J_qp, J_qq]``
with ``(p, q) = (R, L)`` or ``(X, Y)``; the 16 Mueller elements are numbered
row-wise, ``M[f // 4, f % 4] = J1[a] conj(J2[b])`` with
``(a, b) = MAP_MUELLER_TO_JONES[f]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr
from scipy.special import j1

from astroviper.processing_functions.simulation.calculate_parallactic_angles import (
    calculate_parallactic_angles,
    find_representative_angles,
)
from astroviper.processing_functions.simulation.zernike_polynomials import (
    zernike_surface,
)
from astroviper.utils.beam_models import (
    normalize_beam_model_dict,
    normalize_beam_model_xds,
)
from astroviper.utils.coordinate_transforms import (
    make_rotated_grid,
    rotate_coordinates,
    wrapped_angle_difference,
)
from astroviper.utils.measurement_set_tools import polarization_index

SPEED_OF_LIGHT = 299792458.0
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)
# CASA PBMath truncates some constants when building its 1 GHz lookup table; this
# factor reproduces CASA's Airy disk to ~7 significant figures (SIRIUS casa_twiddle).
CASA_AIRY_TWIDDLE = (180 * 7.016 * SPEED_OF_LIGHT) / ((np.pi**2) * 1e9 * 1.566 * 24.5)
CASA_AIRY_N_SAMPLE = 10000

MAP_MUELLER_TO_JONES = np.array(
    [
        [0, 0], [0, 1], [1, 0], [1, 1],
        [0, 2], [0, 3], [1, 2], [1, 3],
        [2, 0], [2, 1], [3, 0], [3, 1],
        [2, 2], [2, 3], [3, 2], [3, 3],
    ]
)  # fmt: skip

DEFAULT_BEAM_PARAMS = {
    "mueller_selection": [0, 5, 10, 15],
    "pa_radius": 0.2,
    "image_size": [1000, 1000],
    "fov_scaling": 4.0,
    "zernike_freq_interp": "nearest",
}


def resolve_beam_params(beam_params: dict | None) -> dict:
    """Fill missing beam parameters with :data:`DEFAULT_BEAM_PARAMS` (returns a new dict)."""
    params = dict(DEFAULT_BEAM_PARAMS)
    if beam_params:
        params.update(beam_params)
    params["mueller_selection"] = np.asarray(
        params["mueller_selection"], dtype=np.int64
    )
    params["image_size"] = np.asarray(params["image_size"], dtype=np.int64)
    if not (0 in params["mueller_selection"] or 15 in params["mueller_selection"]):
        raise ValueError("mueller_selection must contain element 0 or 15.")
    return params


# ----------------------------------------------------------------------------
# Analytic responses
# ----------------------------------------------------------------------------
def airy_disk_response(
    l: np.ndarray,  # noqa: E741
    m: np.ndarray,
    frequency: np.ndarray,
    dish_diameter: float,
    blockage_diameter: float = 0.0,
    ipower: int = 1,
) -> np.ndarray:
    """Airy-disk (voltage if ``ipower=1``, power if ``2``) response at directions ``(l, m)``.

    Obscured Airy pattern ``(2 J1(r)/r - 2 e J1(e r)/r) / (1 - e^2)`` with
    ``e = blockage / dish`` and ``r = |lm| k D / 2``.

    Parameters
    ----------
    l, m : np.ndarray (broadcastable), radians
    frequency : np.ndarray (broadcastable), Hz
    dish_diameter, blockage_diameter : float, metres
    ipower : int

    Returns
    -------
    np.ndarray
        Broadcast shape of ``l``, ``m`` and ``frequency``.
    """
    k = 2 * np.pi * np.asarray(frequency, dtype=np.float64) / SPEED_OF_LIGHT
    r = (
        np.sqrt(
            np.asarray(l, dtype=np.float64) ** 2 + np.asarray(m, dtype=np.float64) ** 2
        )
        * k
        * (dish_diameter / 2)
    )
    r = np.asarray(r)
    safe = np.where(r == 0, 1.0, r)
    if blockage_diameter == 0.0:
        val = 2.0 * j1(safe) / safe
    else:
        e = blockage_diameter / dish_diameter
        val = (2.0 * j1(safe) / safe - 2.0 * e * j1(safe * e) / safe) / (1.0 - e**2)
    val = np.where(r == 0, 1.0, val)
    return val**ipower


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
    beyond ``max_rad_1GHz / (frequency / 1 GHz)`` are **not** zeroed here (see
    :func:`sample_jones`).

    Parameters
    ----------
    l, m : np.ndarray (broadcastable), radians
    frequency : np.ndarray (broadcastable), Hz
    dish_diameter, blockage_diameter : float, metres
    max_rad_1GHz : float, radians
    ipower : int
    n_sample : int or None
        ``None`` disables the lookup quantisation (exact ``arcsin`` argument).

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
        r_max = max_rad_1GHz / (frequency / 1e9)
        r_inc = r_max / (n_sample - 1)
        r = (np.trunc(rho / r_inc) * r_inc) * aperture * k * CASA_AIRY_TWIDDLE
    else:
        r = np.arcsin(rho * k * aperture)
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
    return val**ipower


def polynomial_beam_response(
    l: np.ndarray,  # noqa: E741
    m: np.ndarray,
    frequency: np.ndarray,
    coefficients: np.ndarray,
    max_rad_1GHz: float,
    ipower: int = 1,
    n_sample: int | None = CASA_AIRY_N_SAMPLE,
) -> np.ndarray:
    """CASA ``PBMath1DPoly`` style 1-D beam: ``(sum_i c_i r^(2 i))^(ipower / 2)``.

    ``r`` is the radius in arcmin scaled to 1 GHz (``|lm| * frequency / 1 GHz``),
    quantised like the CASA lookup when ``n_sample`` is given.  The coefficients
    describe the **power** beam, so ``ipower=1`` returns the voltage response.

    Parameters
    ----------
    l, m : np.ndarray (broadcastable), radians
    frequency : np.ndarray (broadcastable), Hz
    coefficients : np.ndarray, [n_coefficients]
    max_rad_1GHz : float, radians
    ipower : int
    n_sample : int or None

    Returns
    -------
    np.ndarray
    """
    frequency = np.asarray(frequency, dtype=np.float64)
    rho = np.sqrt(
        np.asarray(l, dtype=np.float64) ** 2 + np.asarray(m, dtype=np.float64) ** 2
    )
    if n_sample is not None:
        r_inc = (max_rad_1GHz / (frequency / 1e9)) / (n_sample - 1)
        rho = np.trunc(rho / r_inc) * r_inc
    r = rho * (frequency / 1e9) / ARCMIN_TO_RAD
    beam = np.zeros(np.broadcast(r, frequency).shape, dtype=np.float64)
    for i, c in enumerate(np.asarray(coefficients, dtype=np.float64)):
        beam = beam + c * r ** (2 * i)
    beam = np.where(beam < 0, 0.0, beam)
    return beam ** (0.5 * ipower)


# ----------------------------------------------------------------------------
# Jones beam images
# ----------------------------------------------------------------------------
def beam_image_cell_size(
    max_rad_1GHz: float, frequency: np.ndarray, beam_params: dict
) -> np.ndarray:
    """Cell size ``[dl, dm]`` (radians; ``dl < 0``) of a beam image.

    ``delta = (max_rad_1GHz / min(frequency / 1 GHz)) / image_size * fov_scaling``,
    i.e. the image spans ``fov_scaling`` times the beam cut radius at the lowest
    frequency.
    """
    image_size = np.asarray(beam_params["image_size"], dtype=np.float64)
    delta = (max_rad_1GHz / np.min(np.asarray(frequency) / 1e9)) / image_size
    return np.array([-delta[0], delta[1]]) * beam_params["fov_scaling"]


def beam_image_coordinates(
    image_size: np.ndarray, cell_size: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``l`` and ``m`` coordinate arrays of a beam image (centre pixel ``image_size // 2``)."""
    image_size = np.asarray(image_size)
    center = image_size // 2
    l = np.arange(-center[0], image_size[0] - center[0]) * cell_size[0]  # noqa: E741
    m = np.arange(-center[1], image_size[1] - center[1]) * cell_size[1]
    return l, m


def _jones_image_xds(jones, parallactic_angle, frequency, polarization, l, m, attrs):  # noqa: E741
    xds = xr.Dataset(
        {
            "JONES": (
                ("parallactic_angle", "frequency", "polarization", "l", "m"),
                jones,
            )
        },
        coords={
            "parallactic_angle": np.asarray(parallactic_angle, dtype=np.float64),
            "frequency": np.asarray(frequency, dtype=np.float64),
            "polarization": np.asarray(polarization, dtype=str),
            "l": l,
            "m": m,
        },
        attrs=dict(attrs),
    )
    xds.attrs["model_type"] = "jones_image"
    xds.parallactic_angle.attrs.update({"units": "rad"})
    xds.frequency.attrs.update({"units": "Hz", "type": "spectral_coord"})
    xds.l.attrs.update({"units": "rad"})
    xds.m.attrs.update({"units": "rad"})
    return xds


def make_airy_jones_beam(
    model: dict,
    frequency: np.ndarray,
    beam_params: dict | None = None,
    polarization: Sequence[str] = ("RR", "LL"),
) -> xr.Dataset:
    """Jones beam image of an analytic (Airy) model, for inspection/plotting.

    Parameters
    ----------
    model : dict
        Analytic beam model (see :mod:`astroviper.utils.beam_models`).
    frequency : np.ndarray, [n_frequency], Hz
    beam_params : dict, optional
    polarization : sequence of str
        Labels of the (identical, unpolarised) diagonal Jones elements.

    Returns
    -------
    xr.Dataset
        ``JONES[parallactic_angle=1, frequency, polarization, l, m]``.
    """
    model = normalize_beam_model_dict(model)
    params = resolve_beam_params(beam_params)
    frequency = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    cell_size = beam_image_cell_size(model["max_rad_1GHz"], frequency, params)
    l, m = beam_image_coordinates(params["image_size"], cell_size)  # noqa: E741
    resp = _analytic_response(
        model, l[None, :, None], m[None, None, :], frequency[:, None, None]
    )  # [n_freq, n_l, n_m]
    jones = np.repeat(resp[None, :, None, :, :], len(polarization), axis=2).astype(
        np.complex128
    )
    return _jones_image_xds(jones, [0.0], frequency, polarization, l, m, model)


def make_polynomial_jones_beam(
    bpc_xds: xr.Dataset,
    frequency: np.ndarray,
    beam_params: dict | None = None,
    polarization: Sequence[str] = ("RR", "LL"),
) -> xr.Dataset:
    """Jones beam image of a beam-polynomial model (nearest tabulated frequency)."""
    bpc_xds = normalize_beam_model_xds(bpc_xds)
    params = resolve_beam_params(beam_params)
    frequency = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    max_rad = float(bpc_xds.attrs["max_rad_1GHz"])
    cell_size = beam_image_cell_size(max_rad, frequency, params)
    l, m = beam_image_coordinates(params["image_size"], cell_size)  # noqa: E741
    resp = np.empty((len(frequency), len(l), len(m)))
    for i, f in enumerate(frequency):
        i_chan = int(np.argmin(np.abs(bpc_xds.frequency.values - f)))
        coef = bpc_xds.BPC.values[i_chan, 0, :]
        resp[i] = polynomial_beam_response(
            l[:, None], m[None, :], f, coef, max_rad, 1, n_sample=None
        )
        resp[i][np.hypot(l[:, None], m[None, :]) >= max_rad / (f / 1e9)] = 0.0
    jones = np.repeat(resp[None, :, None, :, :], len(polarization), axis=2).astype(
        np.complex128
    )
    return _jones_image_xds(jones, [0.0], frequency, polarization, l, m, bpc_xds.attrs)


def _interpolate_coefficients(
    zpc_xds: xr.Dataset, frequency: float, method: str
) -> xr.Dataset:
    """Coefficients at ``frequency``; out-of-range frequencies use the nearest tabulated entry."""
    table = zpc_xds.frequency.values
    if method == "nearest" or frequency <= table.min() or frequency >= table.max():
        return zpc_xds.sel(frequency=frequency, method="nearest")
    return zpc_xds.interp(frequency=frequency, method=method)


def make_zernike_jones_beam(
    zpc_xds: xr.Dataset,
    parallactic_angle: np.ndarray,
    frequency: np.ndarray,
    beam_params: dict | None = None,
) -> xr.Dataset:
    """Jones beam images from Zernike aperture coefficients.

    For every (parallactic angle, frequency) the aperture illumination is built
    from the Zernike surface of each needed polarization on a grid rotated by the
    parallactic angle, masked to the dish radius, zero-padded into an
    ``image_size`` uv grid whose cell matches the image cell, and inverse-FFTed.
    The Jones planes are normalised so that the mean of the diagonal peak
    amplitudes is one (SIRIUS ``_calc_ant_jones``).

    Parameters
    ----------
    zpc_xds : xr.Dataset
        Aperture-coefficient model (``ZPC``, ``ETA``; attrs ``dish_diameter``,
        ``max_rad_1GHz``).
    parallactic_angle : np.ndarray, [n_pa], radians
    frequency : np.ndarray, [n_frequency], Hz
    beam_params : dict, optional
        ``mueller_selection`` (decides which Jones elements are needed),
        ``image_size``, ``fov_scaling``, ``zernike_freq_interp``.

    Returns
    -------
    xr.Dataset
        ``JONES[parallactic_angle, frequency, polarization, l, m]`` (complex128).
    """
    zpc_xds = normalize_beam_model_xds(zpc_xds)
    params = resolve_beam_params(beam_params)
    parallactic_angle = np.atleast_1d(np.asarray(parallactic_angle, dtype=np.float64))
    frequency = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    image_size = params["image_size"]
    max_rad = float(zpc_xds.attrs["max_rad_1GHz"])
    dish_diameter = float(zpc_xds.attrs["dish_diameter"])
    cell_size = beam_image_cell_size(max_rad, frequency, params)
    needed_pol = np.unique(np.ravel(MAP_MUELLER_TO_JONES[params["mueller_selection"]]))
    pol_labels = np.asarray(zpc_xds.polarization.values, dtype=str)
    pol_index_of_model = polarization_index(pol_labels)
    model_index = np.array(
        [int(np.where(pol_index_of_model == p)[0][0]) for p in needed_pol]
    )

    jones = np.zeros(
        (
            len(parallactic_angle),
            len(frequency),
            len(needed_pol),
            image_size[0],
            image_size[1],
        ),
        dtype=np.complex128,
    )
    center = image_size // 2
    uv_cell_size = 1.0 / (cell_size * image_size)
    for i_chan, freq in enumerate(frequency):
        interp = _interpolate_coefficients(zpc_xds, freq, params["zernike_freq_interp"])
        eta = float(
            interp.ETA.values[0, 0]
        )  # assumed independent of pol and coefficient
        wavelength = SPEED_OF_LIGHT / freq
        zernike_cell = (2.0 * uv_cell_size * wavelength) / (dish_diameter * eta)
        zernike_size = np.ceil(np.abs(2.0 / zernike_cell)).astype(int)
        zernike_center = zernike_size // 2
        include_last = (zernike_size % 2).astype(int)
        sl_l = slice(
            center[0] - zernike_center[0],
            center[0] + zernike_center[0] + include_last[0],
        )
        sl_m = slice(
            center[1] - zernike_center[1],
            center[1] + zernike_center[1] + include_last[1],
        )
        for i_pa, pa in enumerate(parallactic_angle):
            x_grid, y_grid = make_rotated_grid(zernike_size, zernike_cell, pa)
            outside = (x_grid**2 + y_grid**2) > 1.0
            for i_pol, i_model in enumerate(model_index):
                aperture = zernike_surface(
                    interp.ZPC.values[i_model, :], x_grid, y_grid
                )
                aperture[outside] = 0.0
                plane = np.zeros((image_size[0], image_size[1]), dtype=np.complex128)
                plane[sl_l, sl_m] = aperture
                jones[i_pa, i_chan, i_pol] = np.fft.fftshift(
                    np.fft.ifft2(np.fft.ifftshift(plane))
                ) / (image_size[0] * image_size[1])
            abs_max = np.abs(jones[i_pa, i_chan]).max(axis=(1, 2))
            has_p = 0 in needed_pol
            has_q = 3 in needed_pol
            p_max = abs_max[np.where(needed_pol == 0)[0][0]] if has_p else None
            q_max = abs_max[np.where(needed_pol == 3)[0][0]] if has_q else None
            if p_max is None:
                p_max = q_max
            if q_max is None:
                q_max = p_max
            jones[i_pa, i_chan] *= 2.0 / (p_max + q_max)

    l, m = beam_image_coordinates(image_size, cell_size)  # noqa: E741
    attrs = dict(zpc_xds.attrs)
    return _jones_image_xds(
        jones, parallactic_angle, frequency, pol_labels[model_index], l, m, attrs
    )


def make_mueller_matrix(
    jones_xds_1: xr.Dataset, jones_xds_2: xr.Dataset, mueller_selection: Sequence[int]
) -> xr.Dataset:
    """Mueller-matrix images ``M = J1 (x) conj(J2)`` for the selected elements.

    Parameters
    ----------
    jones_xds_1, jones_xds_2 : xr.Dataset
        Jones beam images on identical ``parallactic_angle``/``frequency``/``l``/``m`` grids.
    mueller_selection : sequence of int
        Row-wise flattened 4x4 element indices.

    Returns
    -------
    xr.Dataset
        ``MUELLER[parallactic_angle, frequency, mueller_element, l, m]`` with
        ``polarization_1``/``polarization_2`` coordinates on ``mueller_element``.
    """
    j1x = normalize_beam_model_xds(jones_xds_1)
    j2x = normalize_beam_model_xds(jones_xds_2)
    idx1 = polarization_index(j1x.polarization.values)
    idx2 = polarization_index(j2x.polarization.values)
    sel = np.asarray(mueller_selection, dtype=np.int64)
    shape = j1x.JONES.shape
    mueller = np.zeros(
        (shape[0], shape[1], len(sel), shape[3], shape[4]), dtype=np.complex128
    )
    pol1, pol2 = [], []
    for i, f in enumerate(sel):
        a, b = MAP_MUELLER_TO_JONES[f]
        ia = int(np.where(idx1 == a)[0][0])
        ib = int(np.where(idx2 == b)[0][0])
        mueller[:, :, i] = j1x.JONES.values[:, :, ia] * np.conj(
            j2x.JONES.values[:, :, ib]
        )
        pol1.append(str(j1x.polarization.values[ia]))
        pol2.append(str(j2x.polarization.values[ib]))
    return xr.Dataset(
        {
            "MUELLER": (
                ("parallactic_angle", "frequency", "mueller_element", "l", "m"),
                mueller,
            )
        },
        coords={
            "parallactic_angle": j1x.parallactic_angle.values,
            "frequency": j1x.frequency.values,
            "mueller_element": sel,
            "polarization_1": ("mueller_element", np.array(pol1)),
            "polarization_2": ("mueller_element", np.array(pol2)),
            "l": j1x.l.values,
            "m": j1x.m.values,
        },
    )


# ----------------------------------------------------------------------------
# Model lists
# ----------------------------------------------------------------------------
def beam_model_kind(model) -> str:
    """``"analytic"``, ``"aperture_polynomial_coefficients"``, ``"beam_polynomial_coefficients"`` or ``"jones_image"``."""
    if isinstance(model, dict):
        return "analytic"
    if isinstance(model, xr.Dataset):
        if "ZPC" in model.data_vars:
            return "aperture_polynomial_coefficients"
        if "BPC" in model.data_vars:
            return "beam_polynomial_coefficients"
        if "JONES" in model.data_vars or "J" in model.data_vars:
            return "jones_image"
    raise TypeError(f"Unsupported beam model {type(model)}: {model!r}")


def normalize_beam_model(model):
    """Canonical form of a beam model (dict with canonical keys, or xds with canonical names)."""
    return (
        normalize_beam_model_dict(model)
        if isinstance(model, dict)
        else normalize_beam_model_xds(model)
    )


def dish_diameters_of_beam_models(beam_models: Sequence) -> np.ndarray:
    """Dish diameter (m) of each model in a list."""
    out = []
    for model in beam_models:
        model = normalize_beam_model(model)
        out.append(
            float(
                model["dish_diameter"]
                if isinstance(model, dict)
                else model.attrs["dish_diameter"]
            )
        )
    return np.array(out)


def evaluate_beam_models(
    beam_models: Sequence,
    time: np.ndarray,
    frequency: np.ndarray,
    phase_center_ra_dec: np.ndarray,
    site_position: np.ndarray,
    beam_params: dict | None = None,
    direction_frame: str = "icrs",
) -> tuple[list, np.ndarray]:
    """Prepare beam models for a chunk: parallactic angles and Zernike → Jones images.

    Parallactic angles are computed only when a Zernike (aperture coefficient)
    model is present; otherwise they are zero.  Zernike models are evaluated on
    the representative subset of parallactic angles returned by
    :func:`~astroviper.processing_functions.simulation.calculate_parallactic_angles.find_representative_angles`
    with ``beam_params["pa_radius"]``.

    Parameters
    ----------
    beam_models : sequence
        Analytic dicts, aperture/beam polynomial datasets or Jones image datasets.
    time : np.ndarray, [n_time]
        UTC times (ISO strings or unix seconds).
    frequency : np.ndarray, [n_frequency], Hz
    phase_center_ra_dec : np.ndarray, [n_time | 1, 2], radians
    site_position : np.ndarray, [3], metres (ITRF)
    beam_params : dict, optional
    direction_frame : str

    Returns
    -------
    evaluated_beam_models : list
        Same order as ``beam_models``; Zernike models replaced by Jones images.
    parallactic_angle : np.ndarray, [n_time], radians
    """
    params = resolve_beam_params(beam_params)
    models = [normalize_beam_model(m) for m in beam_models]
    kinds = [beam_model_kind(m) for m in models]
    time = np.asarray(time)
    if "aperture_polynomial_coefficients" in kinds:
        parallactic_angle = calculate_parallactic_angles(
            time, site_position, phase_center_ra_dec, direction_frame
        )
        pa_subset, _, _ = find_representative_angles(
            parallactic_angle, params["pa_radius"]
        )
    else:
        parallactic_angle = np.zeros(time.shape[0], dtype=np.float64)
        pa_subset = np.array([0.0])
    evaluated = []
    for model, kind in zip(models, kinds, strict=True):
        if kind == "aperture_polynomial_coefficients":
            evaluated.append(
                make_zernike_jones_beam(model, pa_subset, frequency, params)
            )
        else:
            evaluated.append(model)
    return evaluated, parallactic_angle


def pack_beam_models(evaluated_beam_models: Sequence) -> list[dict]:
    """Flatten evaluated beam models into plain dicts/arrays for the visibility kernels.

    Returns
    -------
    list of dict
        Each with ``kind`` plus, depending on the kind: ``func``,
        ``dish_diameter``, ``blockage_diameter``, ``max_rad_1GHz`` (analytic);
        ``frequency``, ``coefficients[n_frequency, n_coefficients]`` (polynomial);
        ``jones[n_pa, n_frequency, n_pol, n_l, n_m]``, ``parallactic_angle``,
        ``frequency``, ``polarization_index``, ``cell_size_l``, ``cell_size_m``
        (Jones image).
    """
    packed = []
    for model in evaluated_beam_models:
        model = normalize_beam_model(model)
        kind = beam_model_kind(model)
        if kind == "analytic":
            packed.append(
                {
                    "kind": "analytic",
                    "func": model["func"],
                    "dish_diameter": float(model["dish_diameter"]),
                    "blockage_diameter": float(model["blockage_diameter"]),
                    "max_rad_1GHz": float(model["max_rad_1GHz"]),
                }
            )
        elif kind == "beam_polynomial_coefficients":
            packed.append(
                {
                    "kind": "polynomial",
                    "frequency": np.asarray(model.frequency.values, dtype=np.float64),
                    "coefficients": np.ascontiguousarray(
                        model.BPC.values[:, 0, :], dtype=np.float64
                    ),
                    "dish_diameter": float(model.attrs["dish_diameter"]),
                    "max_rad_1GHz": float(model.attrs["max_rad_1GHz"]),
                }
            )
        elif kind == "jones_image":
            l = model.l.values  # noqa: E741
            m = model.m.values
            packed.append(
                {
                    "kind": "jones_image",
                    "jones": np.ascontiguousarray(
                        model.JONES.values, dtype=np.complex128
                    ),
                    "parallactic_angle": np.asarray(
                        model.parallactic_angle.values, dtype=np.float64
                    ),
                    "frequency": np.asarray(model.frequency.values, dtype=np.float64),
                    "polarization_index": polarization_index(model.polarization.values),
                    "cell_size_l": float(l[1] - l[0]),
                    "cell_size_m": float(m[1] - m[0]),
                    "dish_diameter": float(model.attrs["dish_diameter"]),
                    "max_rad_1GHz": float(model.attrs["max_rad_1GHz"]),
                }
            )
        else:
            raise ValueError(
                "Zernike aperture models must be converted to Jones images with "
                "evaluate_beam_models() before packing."
            )
    return packed


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
def _analytic_response(model: dict, l, m, frequency):  # noqa: E741
    func = model["func"]
    if func == "casa_airy":
        return casa_airy_disk_response(
            l,
            m,
            frequency,
            model["dish_diameter"],
            model["blockage_diameter"],
            model["max_rad_1GHz"],
            1,
        )
    if func == "airy":
        return airy_disk_response(
            l, m, frequency, model["dish_diameter"], model["blockage_diameter"], 1
        )
    if func == "none":
        return np.ones(
            np.broadcast(np.asarray(l), np.asarray(m), np.asarray(frequency)).shape
        )
    raise ValueError(
        f"Unknown analytic beam function {func!r} (use 'casa_airy', 'airy' or 'none')."
    )


def bilinear_interpolate(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of ``image[..., n_x, n_y]`` at fractional pixel positions.

    Indices are clamped to the image edges (out-of-range positions take edge values).

    Parameters
    ----------
    image : np.ndarray, [..., n_x, n_y]
    x, y : np.ndarray, [n]
        Fractional pixel coordinates along the last-but-one and last axes.

    Returns
    -------
    np.ndarray, [..., n]
    """
    n_x, n_y = image.shape[-2], image.shape[-1]
    x = np.clip(np.asarray(x, dtype=np.float64), 0, n_x - 1)
    y = np.clip(np.asarray(y, dtype=np.float64), 0, n_y - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, n_x - 1)
    y1 = np.minimum(y0 + 1, n_y - 1)
    fx = x - x0
    fy = y - y0
    return (
        (1 - fx) * (1 - fy) * image[..., x0, y0]
        + (1 - fx) * fy * image[..., x0, y1]
        + fx * (1 - fy) * image[..., x1, y0]
        + fx * fy * image[..., x1, y1]
    )


def sample_jones(
    packed_model: dict,
    lm: np.ndarray,
    frequency: np.ndarray,
    parallactic_angle: float,
) -> np.ndarray:
    """Jones vector of a packed beam model at directions ``lm`` for each frequency.

    Parameters
    ----------
    packed_model : dict
        One element of :func:`pack_beam_models`.
    lm : np.ndarray, [n, 2]
        Direction cosines of the sources relative to the antenna pointing.
    frequency : np.ndarray, [n_frequency], Hz
    parallactic_angle : float, radians

    Returns
    -------
    np.ndarray, [n, n_frequency, 4] complex128
        ``[J_pp, J_pq, J_qp, J_qq]``; zero beyond ``max_rad_1GHz / (frequency / 1 GHz)``.
    """
    lm = np.atleast_2d(np.asarray(lm, dtype=np.float64))
    frequency = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    n, n_chan = lm.shape[0], frequency.shape[0]
    jones = np.zeros((n, n_chan, 4), dtype=np.complex128)
    l = lm[:, 0][:, None]  # noqa: E741
    m = lm[:, 1][:, None]
    freq = frequency[None, :]
    kind = packed_model["kind"]
    if kind == "analytic":
        resp = _analytic_response(packed_model, l, m, freq)
        jones[:, :, 0] = resp
        jones[:, :, 3] = resp
    elif kind == "polynomial":
        i_chan = np.argmin(
            np.abs(frequency[:, None] - packed_model["frequency"][None, :]), axis=1
        )
        coef = packed_model["coefficients"][i_chan]  # [n_chan, n_coef]
        rho = np.hypot(l, m)  # [n, 1]
        r_inc = (packed_model["max_rad_1GHz"] / (freq / 1e9)) / (CASA_AIRY_N_SAMPLE - 1)
        r = (
            (np.trunc(rho / r_inc) * r_inc) * (freq / 1e9) / ARCMIN_TO_RAD
        )  # [n, n_chan]
        beam = np.zeros((n, n_chan))
        for i in range(coef.shape[1]):
            beam = beam + coef[None, :, i] * r ** (2 * i)
        beam = np.sqrt(np.where(beam < 0, 0.0, beam))
        jones[:, :, 0] = beam
        jones[:, :, 3] = beam
    elif kind == "jones_image":
        i_pa = int(
            np.argmin(
                wrapped_angle_difference(
                    packed_model["parallactic_angle"], parallactic_angle
                )
            )
        )
        i_chan = np.argmin(
            np.abs(frequency[:, None] - packed_model["frequency"][None, :]), axis=1
        )
        scale = frequency / packed_model["frequency"][i_chan]  # [n_chan]
        delta_pa = parallactic_angle - packed_model["parallactic_angle"][i_pa]
        l_rot, m_rot = rotate_coordinates(
            l * scale[None, :], m * scale[None, :], delta_pa
        )  # [n, n_chan]
        image = packed_model["jones"][i_pa]  # [n_chan_model, n_pol, n_l, n_m]
        n_l, n_m = image.shape[-2], image.shape[-1]
        x = l_rot / packed_model["cell_size_l"] + n_l // 2
        y = m_rot / packed_model["cell_size_m"] + n_m // 2
        pol_index = packed_model["polarization_index"]
        for c in range(n_chan):
            sampled = bilinear_interpolate(
                image[i_chan[c]], x[:, c], y[:, c]
            )  # [n_pol, n]
            for i_p, p in enumerate(pol_index):
                jones[:, c, p] = sampled[i_p]
    else:
        raise ValueError(f"Unknown packed beam model kind {kind!r}.")

    outside = np.hypot(l, m) >= packed_model["max_rad_1GHz"] / (
        freq / 1e9
    )  # [n, n_chan]
    jones[outside] = 0.0
    return jones


def apply_mueller(
    jones_1: np.ndarray,
    jones_2: np.ndarray,
    flux: np.ndarray,
    mueller_selection: np.ndarray,
) -> np.ndarray:
    """Scale 4-correlation fluxes by the Mueller matrix of two sampled Jones vectors.

    Parameters
    ----------
    jones_1, jones_2 : np.ndarray, [..., 4] complex
        Jones vectors of the two antennas of a baseline (broadcastable).
    flux : np.ndarray, [..., 4]
        Flux in the instrumental correlations (broadcastable).
    mueller_selection : np.ndarray of int
        Mueller elements to apply; the others are zero.

    Returns
    -------
    np.ndarray, [..., 4] complex
        ``M @ flux`` for the selected elements.
    """
    jones_1 = np.asarray(jones_1)
    jones_2 = np.asarray(jones_2)
    flux = np.asarray(flux)
    shape = np.broadcast(jones_1[..., 0], jones_2[..., 0], flux[..., 0]).shape
    out = np.zeros(shape + (4,), dtype=np.complex128)
    for f in np.asarray(mueller_selection, dtype=np.int64):
        a, b = MAP_MUELLER_TO_JONES[f]
        row, col = divmod(int(f), 4)
        out[..., row] += jones_1[..., a] * np.conj(jones_2[..., b]) * flux[..., col]
    return out
