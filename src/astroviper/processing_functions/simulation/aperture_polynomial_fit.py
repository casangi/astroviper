"""Zernike aperture coefficients fitted to holography samples.

The coefficients are fitted to the raw samples of holography maps (no gridding
and no aperture-plane step): each sample is compared with the exact transform
of the basis polynomials (:func:`zernike_basis`) on the disc of radius
``eta * dish_diameter / 2``, ``V(l, m) = sum_xy A(x, y) exp(+2 pi i (x l + y m) / lambda)``,
the transform :func:`make_zernike_jones_beam` applies at zero parallactic angle.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr
from scipy.optimize import least_squares

from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
    SPEED_OF_LIGHT,
)
from astroviper.processing_functions.simulation.zernike_polynomials import (
    zernike_basis,
)

_LEGACY_DIM_NAMES = {"chan": "frequency", "pol": "polarization"}


def fit_aperture_polynomial_coefficients(
    maps: Sequence[xr.Dataset],
    dish_diameter: float,
    eta: float = 1.0,
    n_frequency: int = 1,
    aperture_cell: float = 0.1,
    telescope_name: str = "",
) -> xr.Dataset:
    """Fit full-Jones Zernike aperture coefficients to holography maps.

    Each map is centred on its main lobe, from a 2-D Gaussian fit to the mean
    parallel-hand amplitude, and all its correlations are divided by the
    complex parallel-hand gain at the centre, so that maps observed at
    different times, with different pointing offsets and instrumental phases,
    can be fitted together. The coefficients of each correlation and frequency
    then solve the weighted linear least-squares problem over the samples of
    all maps.

    Parameters
    ----------
    maps : sequence of xr.Dataset
        Holography maps of identical antennas, each with
        ``DIRECTIONAL_COSINES[time, lm]`` (radians), ``VIS[time, frequency,
        polarization]`` and optionally ``WEIGHT`` (as ``VIS``), on the same
        frequencies and polarizations. The legacy names ``chan`` and ``pol`` of
        astrohack ``extract_holog`` are accepted.
    dish_diameter : float
        Metres.
    eta : float
        Radius of the unit disc of the polynomials, in units of
        ``dish_diameter / 2``.
    n_frequency : int
        Number of contiguous channel groups, each fitted separately.
    aperture_cell : float
        Cell, in metres, of the aperture grid on which the basis polynomials
        are transformed.
    telescope_name : str

    Returns
    -------
    xr.Dataset
        ``ZPC[frequency, polarization, coefficient]`` (complex128), in units of
        the parallel-hand gain at the beam centre, and ``ETA``. ``max_rad_1GHz``
        is the radius of the circle inscribed in the 1-99 percentile extent of
        the sampled directions of the smallest map, times the lowest frequency
        over 1 GHz.
    """
    maps = [_normalize_map_names(m) for m in maps]
    frequency = np.asarray(maps[0].frequency.values, dtype=np.float64)
    polarization = [str(p) for p in maps[0].polarization.values]
    groups = np.array_split(np.arange(frequency.size), n_frequency)
    group_frequency = np.array([frequency[g].mean() for g in groups])

    radius = eta * dish_diameter / 2
    n_half = int(np.ceil(radius / aperture_cell)) + 1
    axis = np.arange(-n_half, n_half + 1) * aperture_cell
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    disc = np.hypot(x_grid, y_grid) <= radius
    basis = zernike_basis(x_grid / radius, y_grid / radius) * disc
    n_coefficient = basis.shape[0]

    normal = np.zeros(
        (n_frequency, len(polarization), n_coefficient, n_coefficient), complex
    )
    rhs = np.zeros((n_frequency, len(polarization), n_coefficient), complex)
    raster_radius = np.inf
    for holog_map in maps:
        lm = np.asarray(holog_map.DIRECTIONAL_COSINES.values, dtype=np.float64)
        vis = np.asarray(holog_map.VIS.values)
        weight = (
            np.asarray(holog_map.WEIGHT.values, dtype=np.float64)
            if "WEIGHT" in holog_map
            else np.ones(vis.shape)
        )
        low, high = np.percentile(lm, [1, 99], axis=0)
        raster_radius = min(raster_radius, -low[0], high[0], -low[1], high[1])
        centre, _ = _centre_and_gain(
            lm,
            np.nanmean(vis, axis=1),
            polarization,
            SPEED_OF_LIGHT / frequency.mean() / dish_diameter,
        )
        lm = lm - centre
        for i_group, group in enumerate(groups):
            group_vis = np.nanmean(vis[:, group], axis=1)
            group_weight = np.nanmean(weight[:, group], axis=1)
            wavelength = SPEED_OF_LIGHT / group_frequency[i_group]
            _, gain = _centre_and_gain(
                lm, group_vis, polarization, wavelength / dish_diameter
            )
            design = _basis_transform(basis, axis, lm, wavelength) * aperture_cell**2
            for i_pol in range(len(polarization)):
                good = np.isfinite(group_vis[:, i_pol]) & np.isfinite(
                    group_weight[:, i_pol]
                )
                weighted = design[good].conj().T * group_weight[good, i_pol]
                normal[i_group, i_pol] += weighted @ design[good]
                rhs[i_group, i_pol] += weighted @ (group_vis[good, i_pol] / gain)

    zpc = np.linalg.solve(normal, rhs[..., None])[..., 0]
    xds = xr.Dataset(
        {
            "ZPC": (("frequency", "polarization", "coefficient"), zpc),
            "ETA": (
                ("frequency", "polarization", "coefficient"),
                np.full(zpc.shape, float(eta)),
            ),
        },
        coords={
            "frequency": group_frequency,
            "polarization": polarization,
            "coefficient": np.arange(n_coefficient),
        },
        attrs={
            "type": "zernike_polynomial_coefficients",
            "schema_version": "0.0.1",
            "zernike_basis": "sirius",
            "telescope_name": telescope_name,
            "dish_diameter": float(dish_diameter),
            "max_rad_1GHz": float(raster_radius * group_frequency.min() / 1e9),
            "aperture_frame": "x and y are conjugate to the DIRECTIONAL_COSINES l and m of the holography maps",
            "origin": {
                "software": "astroviper fit_aperture_polynomial_coefficients",
                "n_maps": len(maps),
            },
        },
    )
    xds.frequency.attrs.update({"units": "Hz", "type": "spectral_coord"})
    return xds


def _normalize_map_names(holog_map: xr.Dataset) -> xr.Dataset:
    """Rename the ``chan`` and ``pol`` dimensions of astrohack holography maps."""
    holog_map = holog_map.rename(
        {k: v for k, v in _LEGACY_DIM_NAMES.items() if k in holog_map.dims}
    )
    for name in ("DIRECTIONAL_COSINES", "VIS"):
        if name not in holog_map:
            raise ValueError(f"Holography map is missing {name}.")
    return holog_map


def _centre_and_gain(
    lm: np.ndarray, vis: np.ndarray, polarization: list[str], lambda_over_d: float
) -> tuple[np.ndarray, complex]:
    """Main-lobe centre (2-D Gaussian fit to the mean parallel-hand amplitude
    above a quarter of its maximum) and the complex parallel-hand gain there
    (Gaussian-weighted mean within one standard deviation)."""
    parallel = [i for i, p in enumerate(polarization) if p[0] == p[1]]
    mean_parallel = vis[:, parallel].mean(axis=1)
    amplitude = np.abs(mean_parallel)
    bright = np.isfinite(amplitude) & (amplitude > 0.25 * np.nanmax(amplitude))
    l, m, z = lm[bright, 0], lm[bright, 1], amplitude[bright]  # noqa: E741

    def residual(p):
        return (
            p[0] * np.exp(-0.5 * (((l - p[1]) / p[3]) ** 2 + ((m - p[2]) / p[4]) ** 2))
            - z
        )

    start = [
        z.max(),
        l[np.argmax(z)],
        m[np.argmax(z)],
        lambda_over_d / 2,
        lambda_over_d / 2,
    ]
    _, l0, m0, sigma_l, sigma_m = least_squares(residual, start).x
    offset_l = (lm[:, 0] - l0) / sigma_l
    offset_m = (lm[:, 1] - m0) / sigma_m
    near = (np.hypot(offset_l, offset_m) < 1.0) & np.isfinite(mean_parallel)
    taper = np.exp(-0.5 * (offset_l[near] ** 2 + offset_m[near] ** 2))
    gain = np.sum(mean_parallel[near] * taper) / np.sum(taper**2)
    return np.array([l0, m0]), gain


def _basis_transform(
    basis: np.ndarray, axis: np.ndarray, lm: np.ndarray, wavelength: float
) -> np.ndarray:
    """``[n_sample, n_coefficient]`` sums of each basis polynomial on the
    aperture grid times ``exp(+2 pi i (x l + y m) / lambda)``, separated
    along the two axes."""
    k = 2 * np.pi / wavelength
    phase_l = np.exp(1j * k * np.outer(axis, lm[:, 0]))
    phase_m = np.exp(1j * k * np.outer(axis, lm[:, 1]))
    design = np.empty((lm.shape[0], basis.shape[0]), dtype=np.complex128)
    for i_coefficient, polynomial in enumerate(basis):
        design[:, i_coefficient] = np.sum(phase_l * (polynomial @ phase_m), axis=0)
    return design
