"""Limb-darkened elliptical disk sky component: image-plane profile and analytic visibility.

The brightness of a limb-darkened disk of angular diameter ``D`` follows the
power-law limb-darkening law of Hestroffer (1997, A&A 327, 199)::

    I(mu) = I_0 * mu**alpha,    mu = sqrt(1 - (2 r / D)**2),    r < D / 2

where ``r`` is the angular distance from the disk centre, ``mu`` the cosine of
the angle between the line of sight and the local normal of a spherical surface
and ``alpha`` the limb-darkening exponent:

* ``alpha = 0`` -- uniform disk;
* ``alpha > 0`` -- darker towards the limb (a stellar photosphere, or the smooth
  radial fall-off of a protoplanetary dust disk);
* ``-2 < alpha < 0`` -- brighter towards the limb (``alpha = -1`` is an
  optically thin spherical shell);
* ``alpha = -2`` -- the infinitely thin ring limit (all flux on the rim), e.g.
  a black-hole photon-ring model.

The normalised visibility of the circular disk at a baseline of length ``q``
(wavelengths) has the closed form (Hestroffer 1997, eq. 4)::

    V(x) = Gamma(nu + 1) * (2 / x)**nu * J_nu(x),    nu = alpha / 2 + 1,    x = pi D q

which reduces to ``2 J_1(x) / x`` (uniform disk), ``sin(x) / x`` (shell) and
``J_0(x)`` (thin ring).  An inclined disk is the same profile stretched along a
major axis at position angle ``pa`` -- the ellipse of diameters
``[major, minor]`` -- so its visibility is the circular formula evaluated at the
"deprojected" baseline::

    x = pi * sqrt( major**2 (u sin pa + v cos pa)**2 + minor**2 (u cos pa - v sin pa)**2 )

This is the same quadratic form and position-angle convention as the Gaussian
source component
(:func:`~astroviper.processing_functions.imaging.restore.elliptical_gaussian_uv_taper`):
the major axis lies along ``(sin pa, cos pa)`` in ``(l, m)``.  The unit tests
pin :func:`limb_darkened_disk_uv_response` to the numerical Hankel transform
of the radial profile and to the FFT of :func:`limb_darkened_disk_image`, so
the two forms cannot drift apart.

**The w term.**  In the frame of the component (its centre at the pole, the
phase-centre ``uvw`` rotated accordingly) and to second order in the direction
cosines, the measurement equation multiplies the brightness by the quadratic
phase ``exp(-i pi w (l^2 + m^2))`` (``exp(+2 pi i)`` sign convention of
``calculate_visibilities``, ``w (n - 1) ~ -w r^2 / 2``).  In the face-on
coordinates of an inclined disk that phase is anisotropic,
``exp(-i pi (w_1 x^2 + w_2 y^2))`` with ``w_1 = w`` and ``w_2 = q^2 w``
(``q`` the axis ratio), and the response is the Sonine / Jacobi-Anger series
of :func:`~astroviper.processing_functions.simulation.component_series.uniform_disk_series`
(simulation memo, version 2).  An optional Gaussian blur ``fwhm`` (circular in
the face-on frame; the blurred image is stretched with the disk) is applied
exactly through
:func:`~astroviper.processing_functions.simulation.component_series.blur_transform`.
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.simulation.component_series import (
    SERIES_X_MAX,
    blur_quadrature_nodes,
    blur_transform,
    blurred_radial_profile_image,
    deprojected_baseline,
    deprojected_sky,
    uniform_disk_series,
)
from astroviper.processing_functions.simulation.gaussian_source import FWHM_TO_SIGMA


def _check_disk_shape(major, minor, limb_darkening, minimum_limb_darkening, fwhm=0.0):
    if major < 0 or minor < 0:
        raise ValueError(
            f"disk diameters must be non-negative; got major={major}, minor={minor}."
        )
    if not np.isfinite(limb_darkening) or limb_darkening < minimum_limb_darkening:
        raise ValueError(
            "limb_darkening (the exponent alpha of I ~ mu**alpha) must be a finite "
            f"number >= {minimum_limb_darkening}; got {limb_darkening}."
        )
    if fwhm < 0:
        raise ValueError(f"disk fwhm must be non-negative; got {fwhm}.")


def limb_darkened_disk_uv_response(
    u,
    v,
    major,
    minor,
    pa,
    limb_darkening: float = 0.0,
    w=0.0,
    tolerance: float = 1e-14,
    fwhm: float = 0.0,
) -> np.ndarray:
    """Normalised visibility of a unit-flux limb-darkened elliptical disk.

    Multiplying the visibilities of a point source by this response turns the
    point source into a limb-darkened disk of the same integrated flux centred
    on the same position (the response is ``1`` at ``(u, v, w) = (0, 0, 0)`` and
    for a zero-diameter disk).

    Parameters
    ----------
    u, v, w : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in the frame of the component (see
        :func:`~astroviper.processing_functions.simulation.calculate_visibilities.source_frame_uvw`),
        in units of the observing wavelength.  ``w = 0`` (default) is the
        tangent-plane approximation.
    major, minor : float, radians
        Outer angular diameters of the disk along its major and minor axes
        (an inclined circular disk of diameter ``D`` and inclination ``i`` has
        ``major = D`` and ``minor = D cos i``).
    pa : float, radians
        Position angle of the major axis, measured from the ``+m`` axis towards
        the ``+l`` axis (the clean-beam / Gaussian-source convention).
    limb_darkening : float, optional
        Power-law limb-darkening exponent ``alpha`` (``I ~ mu**alpha``), ``>= -2``.
        ``0`` (default) is a uniform disk, ``-2`` the infinitely thin ring.
    tolerance : float, optional
        Truncation tolerance of the w-term series (module docstring).
    fwhm : float, radians, optional
        FWHM of a circular Gaussian (in the face-on frame) the disk is
        convolved with; ``0`` (default) is the sharp-edged disk.

    Returns
    -------
    numpy.ndarray
        Broadcast shape of ``u``, ``v`` and ``w``.  Real, in ``[-1, 1]``, when
        ``w`` is identically zero and ``fwhm = 0`` (it oscillates through zero
        beyond the first null, unlike a Gaussian taper); complex otherwise.

    See Also
    --------
    limb_darkened_disk_image : the image-plane form (unit total flux).
    astroviper.processing_functions.imaging.restore.elliptical_gaussian_uv_taper :
        the Gaussian analogue sharing the ``[major, minor, pa]`` convention.
    """
    from scipy.special import gamma, jv

    major = float(major)
    minor = float(minor)
    limb_darkening = float(limb_darkening)
    fwhm = float(fwhm)
    _check_disk_shape(
        major, minor, limb_darkening, minimum_limb_darkening=-2.0, fwhm=fwhm
    )
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    nu = limb_darkening / 2.0 + 1.0

    if np.any(w) or fwhm > 0.0:
        if major == 0.0 or minor == 0.0:  # a point: no extent, no phase
            return np.ones(np.broadcast(u, v, w).shape, dtype=np.complex128)
        q = minor / major
        k1, k2 = deprojected_baseline(u, v, pa, q)
        sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
        k1, k2, w1, w2, prefactor = blur_transform(k1, k2, w, q * q * w, sigma2)
        return prefactor * uniform_disk_series(
            k1, k2, w1, w2, major / 2.0, nu, tolerance
        )

    sin_pa = float(np.sin(pa))
    cos_pa = float(np.cos(pa))
    x = np.pi * np.sqrt(
        (major * (u * sin_pa + v * cos_pa)) ** 2
        + (minor * (u * cos_pa - v * sin_pa)) ** 2
    )
    x = np.atleast_1d(x)
    response = np.empty_like(x)
    small = x < SERIES_X_MAX
    # leading terms of Gamma(nu + 1) (2/x)^nu J_nu(x) = 1 - (x/2)^2 / (nu + 1) + ...
    response[small] = 1.0 - x[small] ** 2 / (4.0 * (nu + 1.0))
    large = ~small
    if large.any():
        x_large = x[large]
        response[large] = gamma(nu + 1.0) * (2.0 / x_large) ** nu * jv(nu, x_large)
    return response.reshape(np.broadcast(u, v).shape)


def limb_darkened_disk_image(
    l, m, major, minor, pa, limb_darkening: float = 0.0, fwhm: float = 0.0
) -> np.ndarray:
    """Unit-total-flux surface brightness of a limb-darkened elliptical disk.

    Evaluates ``I(l, m)`` (per steradian) at sky offsets ``(l, m)`` from the disk
    centre; ``flux * I * pixel_area`` is the disk in Jy/pixel on an image grid.
    The image-plane counterpart of :func:`limb_darkened_disk_uv_response`::

        I = (alpha + 2) / (2 pi a b) * (1 - rho^2)^(alpha / 2)    for rho < 1
        rho^2 = ((l sin pa + m cos pa) / a)^2 + ((l cos pa - m sin pa) / b)^2

    with the semi-axes ``a = major / 2`` and ``b = minor / 2``.  With
    ``fwhm > 0`` the profile is convolved with the circular Gaussian in the
    face-on frame (a 1-D quadrature over the radial profile).

    Parameters
    ----------
    l, m : numpy.ndarray (broadcastable), radians
        Direction cosines (sky offsets) relative to the disk centre.
    major, minor : float, radians
        Outer angular diameters along the major and minor axes (``> 0``).
    pa : float, radians
        Position angle of the major axis (from ``+m`` towards ``+l``).
    limb_darkening : float, optional
        Power-law exponent ``alpha`` (``> -2``): the thin-ring limit
        ``alpha = -2`` has no finite surface brightness and is rejected here.
    fwhm : float, radians, optional
        Gaussian blur FWHM (face-on frame); ``0`` is the sharp disk.

    Returns
    -------
    numpy.ndarray
        Broadcast shape of ``l`` and ``m``; zero outside the disk.  For
        ``-2 < alpha < 0`` the brightness diverges (integrably) at the rim.
    """
    major = float(major)
    minor = float(minor)
    limb_darkening = float(limb_darkening)
    fwhm = float(fwhm)
    _check_disk_shape(
        major, minor, limb_darkening, minimum_limb_darkening=-2.0, fwhm=fwhm
    )
    if major == 0 or minor == 0 or limb_darkening == -2.0:
        raise ValueError(
            "limb_darkened_disk_image needs positive diameters and limb_darkening > -2 "
            "(a zero-size disk or the thin-ring limit has no finite surface brightness)."
        )
    l = np.asarray(l, dtype=np.float64)  # noqa: E741
    m = np.asarray(m, dtype=np.float64)
    a = major / 2.0
    b = minor / 2.0
    q = b / a
    x, y = deprojected_sky(l, m, pa, q)  # face-on frame (circular disk of radius a)
    r = np.sqrt(x**2 + y**2)
    nu = limb_darkening / 2.0 + 1.0
    if fwhm == 0.0:
        rho2 = np.atleast_1d((r / a) ** 2)
        image = np.zeros_like(rho2)
        inside = rho2 < 1.0
        image[inside] = ((limb_darkening + 2.0) / (2.0 * np.pi * a * b)) * (
            1.0 - rho2[inside]
        ) ** (limb_darkening / 2.0)
        return image.reshape(np.broadcast(l, m).shape)  # fmt: skip
    # blurred: int_0^a b(r') kernel(r, r') dr' with b(r') = nu / (pi a^2) (1 - r'^2/a^2)^(nu - 1);
    # Gauss-Jacobi in x = r'/a absorbs the (1 - x)^(nu - 1) rim singularity of limb-brightened disks
    from scipy.special import roots_jacobi

    sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        nodes, weights = roots_jacobi(
            blur_quadrature_nodes(a, sigma2), nu - 1.0, 0.0
        )  # (1 - t)^(nu-1) on [-1, 1]
    x_nodes = (nodes + 1.0) / 2.0  # r'/a in [0, 1]
    # (1 - t)^(nu-1) dt = 2^nu (1 - x)^(nu-1) dx; remaining smooth factor (1 + x)^(nu-1)
    profile_values = (
        nu / (np.pi * a**2) * (1.0 + x_nodes) ** (nu - 1.0) * weights / 2.0**nu * a
    )
    return blurred_radial_profile_image(r, x_nodes * a, profile_values, sigma2) / q
