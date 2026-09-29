"""Airy-disk antenna response functions shared by imaging and simulation.

These are the single source of truth for the analytic Airy voltage/power
patterns in AstroVIPER:

* :func:`airy_disk_response` -- the textbook obscured Airy pattern
  (https://en.wikipedia.org/wiki/Airy_disk, *Obscured Airy pattern* section);
* :func:`casa_airy_disk_response` -- CASA ``PBMath1DAiry`` compatible variant
  (10000-point lookup quantisation and truncated constants) that reproduces the
  ``.pb`` image written by ``tclean`` to float32 precision.

Following the CASA definition, the **primary beam** is the power sensitivity
pattern of the antenna: the square of the absolute value of the voltage
pattern, ``P = |V| ** 2`` (``ipower=2``).  ``ipower=1`` returns the voltage
pattern itself, which is what the simulation subdomain uses to build per-
antenna Jones responses (a baseline of identical antennas then responds with
``V_1 V_2^* = P``).

The simulation module :mod:`astroviper.processing_functions.simulation.antenna_beams`
re-exports these functions; the imaging primary-beam step
(:mod:`astroviper.processing_functions.imaging.primary_beam.make_primary_beam`)
evaluates :func:`casa_airy_disk_response` with ``ipower=2``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1

SPEED_OF_LIGHT = 299792458.0
# CASA PBMath truncates some constants when building its 1 GHz lookup table; this
# factor reproduces CASA's Airy disk to ~7 significant figures (SIRIUS casa_twiddle).
CASA_AIRY_TWIDDLE = (180 * 7.016 * SPEED_OF_LIGHT) / ((np.pi**2) * 1e9 * 1.566 * 24.5)
CASA_AIRY_N_SAMPLE = 10000


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
