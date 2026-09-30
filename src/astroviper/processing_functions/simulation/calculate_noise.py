"""Thermal (additive Gaussian) noise for simulated visibilities."""

from __future__ import annotations

import numpy as np

BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
T_CMB = 2.725  # K

DEFAULT_NOISE_PARAMS = {
    "mode": "tsys-manual",
    "t_atmos": 250.0,
    "tau": 0.1,
    "ant_efficiency": 0.8,
    "spill_efficiency": 0.85,
    "corr_efficiency": 0.88,
    "quantization_efficiency": 0.96,
    "t_receiver": 50.0,
    "t_cmb": T_CMB,
    "random_seed": None,
}


def resolve_noise_params(noise_params: dict | None) -> dict | None:
    """Fill missing noise parameters with :data:`DEFAULT_NOISE_PARAMS` (``None`` stays ``None``)."""
    if noise_params is None:
        return None
    params = dict(DEFAULT_NOISE_PARAMS)
    params.update(noise_params)
    if params["mode"] != "tsys-manual":
        raise ValueError("Only noise_params['mode'] = 'tsys-manual' is implemented.")
    return params


def calculate_noise_sigma(
    dish_diameter_per_antenna: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    channel_width: float,
    integration_time: float,
    noise_params: dict,
) -> np.ndarray:
    """Per-baseline RMS noise (Jy) of one real/imaginary visibility component.

    Follows ``casatools.simulator.setnoise(mode="tsys-manual")``::

        A_eff = eta_ant * pi * D1 * D2 / 4
        T_sys = T_receiver + T_atmos * (1 - eta_spill) + T_cmb
        sigma = sqrt(2) k_B T_sys 1e26 / (eta_corr eta_q A_eff sqrt(channel_width * integration_time))

    The zenith opacity ``tau`` is accepted but not applied (as in CASA's
    ``tsys-manual`` without elevation dependence in SIRIUS).

    Parameters
    ----------
    dish_diameter_per_antenna : np.ndarray, [n_antenna], metres
    antenna1, antenna2 : np.ndarray, [n_baseline]
    channel_width : float, Hz
    integration_time : float, s
    noise_params : dict
        See :data:`DEFAULT_NOISE_PARAMS`.

    Returns
    -------
    np.ndarray, [n_baseline]
    """
    params = resolve_noise_params(noise_params)
    d = np.asarray(dish_diameter_per_antenna, dtype=np.float64)
    a_eff = params["ant_efficiency"] * np.pi * d[antenna1] * d[antenna2] / 4.0
    t_sys = (
        params["t_receiver"]
        + params["t_atmos"] * (1 - params["spill_efficiency"])
        + params["t_cmb"]
    )
    sigma = (
        np.sqrt(2.0)
        * BOLTZMANN_CONSTANT
        * t_sys
        * 1e26
        / (
            params["corr_efficiency"]
            * params["quantization_efficiency"]
            * a_eff
            * np.sqrt(channel_width * integration_time)
        )
    )
    return sigma


def calculate_noise(
    visibility_shape: tuple[int, int, int, int],
    dish_diameter_per_antenna: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    channel_width: float,
    integration_time: float,
    noise_params: dict,
    auto_correlations: bool = False,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian noise realisation plus matching ``WEIGHT`` and ``SIGMA``.

    Parameters
    ----------
    visibility_shape : tuple
        ``(n_time, n_baseline, n_frequency, n_polarization)``.
    dish_diameter_per_antenna : np.ndarray, [n_antenna], metres
    antenna1, antenna2 : np.ndarray, [n_baseline]
    channel_width : float, Hz
    integration_time : float, s
    noise_params : dict
    auto_correlations : bool
        Autocorrelations get real-only noise with ``sqrt(2)`` the cross RMS.
    random_seed : int or None
        Seed for ``numpy.random.default_rng``; ``None`` draws a fresh seed.

    Returns
    -------
    noise : np.ndarray, complex128, ``visibility_shape``
    weight : np.ndarray, float64, ``visibility_shape``
        ``1 / sigma^2``.
    sigma : np.ndarray, float64, ``visibility_shape``
    """
    n_time, n_baseline, n_chan, n_pol = visibility_shape
    sigma_bl = calculate_noise_sigma(
        dish_diameter_per_antenna,
        antenna1,
        antenna2,
        channel_width,
        integration_time,
        noise_params,
    )
    sigma = np.broadcast_to(sigma_bl[None, :, None, None], visibility_shape).astype(
        np.float64
    )
    rng = np.random.default_rng(random_seed)
    noise_re = rng.normal(0.0, sigma)
    noise_im = rng.normal(0.0, sigma)
    if auto_correlations:
        is_auto = np.asarray(antenna1) == np.asarray(antenna2)
        noise_re[:, is_auto] *= np.sqrt(2.0)
        noise_im[:, is_auto] = 0.0
    noise = noise_re + 1j * noise_im
    weight = 1.0 / sigma**2
    return noise, weight, np.array(sigma)
