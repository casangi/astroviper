"""Elliptical Gaussian sky component: analytic visibility including the w term.

The tangent-plane (``w = 0``) response of a unit-flux elliptical Gaussian is the
imaging restore module's
:func:`~astroviper.processing_functions.imaging.restore.elliptical_gaussian_uv_taper`
(the single source of truth for the ``[major, minor, pa]`` FWHM parametrisation).
This module adds the **w term**.  In the frame of the component (its centre at
the pole, coordinates ``(u, v, w)`` rotated accordingly) the measurement
equation is, to second order in the direction cosines,

.. math::

    T(u, v, w) = \\int b(l, m)\\, e^{-i \\pi w (l^2 + m^2)}\\,
                 e^{2 \\pi i (u l + v m)}\\, dl\\, dm ,

with ``b`` the unit-flux brightness and the ``exp(+2 pi i)`` sign convention of
:mod:`~astroviper.processing_functions.simulation.calculate_visibilities`
(``w (n - 1) ~ -w r^2 / 2``).  The quadratic phase is isotropic on the sky, so
in the Gaussian's principal axes the integral separates and each axis becomes
a Gaussian of *complex* width ``1 / sigma_c^2 = 1 / sigma^2 + 2 pi i w``::

    T = sqrt(g_maj g_min) exp(-2 pi^2 [sigma_maj^2 g_maj u_maj^2
                                      + sigma_min^2 g_min u_min^2]),
    g_j = 1 / (1 + 2 pi i w sigma_j^2),

which reduces to the restore taper at ``w = 0``.  The size of the correction is
set by ``2 pi w sigma^2``; see the simulation memo (version 2) for the
derivation, the validity of the paraxial expansion and worked numbers.
"""

from __future__ import annotations

import numpy as np

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def elliptical_gaussian_uv_response(u, v, major, minor, pa, w=0.0) -> np.ndarray:
    """Normalised visibility of a unit-flux elliptical Gaussian, with the w term.

    Parameters
    ----------
    u, v, w : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in the frame of the component (the phase-centre
        ``uvw`` rotated to the component direction, see
        :func:`~astroviper.processing_functions.simulation.calculate_visibilities.source_frame_uvw`),
        in units of the observing wavelength.  ``w = 0`` (default) is the
        tangent-plane approximation.
    major, minor : float, radians
        FWHM of the major and minor axes on the sky.
    pa : float, radians
        Position angle of the major axis, measured from the ``+m`` axis
        towards the ``+l`` axis (the clean-beam convention).

    Returns
    -------
    numpy.ndarray
        Broadcast shape of ``u``, ``v`` and ``w``.  Real (the restore taper)
        when ``w`` is identically zero, complex otherwise.

    See Also
    --------
    astroviper.processing_functions.imaging.restore.elliptical_gaussian_uv_taper :
        the ``w = 0`` form, reproduced exactly.
    """
    from astroviper.processing_functions.imaging.restore import (
        elliptical_gaussian_uv_taper,
    )

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if not np.any(w):  # tangent plane: exactly the restore taper
        return elliptical_gaussian_uv_taper(u, v, major, minor, pa)
    sin_pa = float(np.sin(pa))
    cos_pa = float(np.cos(pa))
    u_major = u * sin_pa + v * cos_pa
    u_minor = u * cos_pa - v * sin_pa
    sigma_major2 = (float(major) * FWHM_TO_SIGMA) ** 2
    sigma_minor2 = (float(minor) * FWHM_TO_SIGMA) ** 2
    two_pi2 = 2.0 * np.pi**2
    g_major = 1.0 / (1.0 + 2j * np.pi * w * sigma_major2)
    g_minor = 1.0 / (1.0 + 2j * np.pi * w * sigma_minor2)
    return np.sqrt(g_major * g_minor) * np.exp(
        -two_pi2
        * (sigma_major2 * g_major * u_major**2 + sigma_minor2 * g_minor * u_minor**2)
    )


def elliptical_gaussian_image(l, m, major, minor, pa=0.0) -> np.ndarray:  # noqa: E741
    """Unit-total-flux surface brightness of the elliptical Gaussian (per steradian).

    The image-plane twin of :func:`elliptical_gaussian_uv_response`:
    ``exp(-(l_maj^2 / sigma_maj^2 + l_min^2 / sigma_min^2) / 2) / (2 pi sigma_maj sigma_min)``
    with ``l_maj = l sin pa + m cos pa`` and ``l_min = l cos pa - m sin pa``.
    """
    major = float(major)
    minor = float(minor)
    if major <= 0 or minor <= 0:
        raise ValueError(
            f"Gaussian FWHM must be positive; got major={major}, minor={minor}."
        )
    l = np.asarray(l, dtype=np.float64)  # noqa: E741
    m = np.asarray(m, dtype=np.float64)
    sin_pa = float(np.sin(pa))
    cos_pa = float(np.cos(pa))
    l_major = l * sin_pa + m * cos_pa
    l_minor = l * cos_pa - m * sin_pa
    sigma_major = major * FWHM_TO_SIGMA
    sigma_minor = minor * FWHM_TO_SIGMA
    return np.exp(
        -0.5 * ((l_major / sigma_major) ** 2 + (l_minor / sigma_minor) ** 2)
    ) / (2.0 * np.pi * sigma_major * sigma_minor)
