"""The common beam of a set of Gaussian beams, and the beam that convolves one into another.

A beam is ``[major, minor, pa]`` as the ``BEAM_FIT_PARAMS_*`` variables hold it: the full
widths at half maximum and the position angle of the major axis, from north through east,
all in radians.
"""

import math

import numpy as np
import toolviper.utils.logger as logger

# casacore's C::arcsec. CASA builds its enclosing ellipses from the axes in arcsec and
# keeps them in arcsec; the conversions are part of its result.
_ARCSEC = 0.00000484813681109535993589914098765
_FOUR_LN_2 = 4.0 * math.log(2.0)


def common_beam(beams):
    """CASA's common beam of a set of beams.

    The algorithm of ``CasaImageBeamSet::getCommonBeam``, which ``tclean`` uses for
    ``restoringbeam='common'``, as CASA 6.8 has it (with the fixes of CAS-14719 for
    circular and nearly identical beams, which make CASA 6.7 return beams up to several
    times too large). It returns the largest beam (by area) when every other beam fits
    inside it. Otherwise it builds an ellipse around the largest beam and the last beam
    that does not fit, grows it by factors of 1.001 until both fit, puts it in place of the
    largest beam and starts again; when 1000 steps are not enough it gives up and returns
    the largest beam. A beam fits inside another when CASA's ``GaussianDeconvolver``
    deconvolves it without its point-source case. The result is therefore not the
    smallest ellipse that encloses all beams. It depends on the order of the beams, so it
    equals CASA's for the beams in the order of the CASA image's channels.

    Parameters
    ----------
    beams : array_like
        Beams ``(..., 3)``: ``[major, minor, pa]`` in radians. Beams with a zero axis
        (CASA's null beams) or a non-finite one (failed fits) are left out.

    Returns
    -------
    numpy.ndarray
        The common beam ``[major, minor, pa]``, with pa in ``[0, pi)``.

    Raises
    ------
    ValueError
        If no beam has finite, positive axes.
    """
    beams = np.asarray(beams, dtype=np.float64).reshape(-1, 3)
    usable = np.isfinite(beams).all(axis=1) & (beams[:, 0] > 0) & (beams[:, 1] > 0)
    if not usable.any():
        raise ValueError("No beam has finite, positive axes.")
    # (major, minor, pa, in_arcsec): the input beams keep their radians, CASA's
    # enclosing ellipses are kept in arcsec, as CASA keeps them.
    flat = [(b[0], b[1], _unwrap(b[2]), False) for b in beams[usable]]
    while True:
        if all(b == flat[0] for b in flat):
            return _result(flat[0])
        areas = [_area(b, flat[0][3]) for b in flat]
        largest = flat[int(np.argmax(areas))]
        problem = None
        for b in flat:
            if b != largest and not _fits_inside(b, largest):
                problem = b
        if problem is None:
            return _result(largest)
        if abs(problem[2] - largest[2]) == 0.5 * math.pi:
            outer, inner = (largest, problem)
            if _radians(problem)[0] > _radians(largest)[0]:
                outer, inner = problem, largest
            return _result((_radians(outer)[0], _radians(inner)[0], outer[2], False))
        enclosing = _enclosing(largest, problem)
        if enclosing is None:
            logger.warning(
                "No beam enclosing all beams after 1000 steps; using the largest beam."
            )
            return _result(largest)
        flat[flat.index(largest)] = enclosing


def correcting_beam(beam, target_beam, tolerance=1e-10):
    """The Gaussian that convolves ``beam`` into ``target_beam``.

    The correcting beam of CASA Memo 10 (CAS-12148), in covariance form: its covariance is
    the target's minus the beam's, which is a Gaussian when that difference is positive
    semidefinite, that is, when the target contains the beam. Unlike the memo's
    parametrization, this holds when the beams are equal (the correcting beam is then
    ``[0, 0, 0]``, no convolution) or touch (a zero minor axis, a line).

    Parameters
    ----------
    beam, target_beam : array_like
        ``[major, minor, pa]`` in radians.
    tolerance : float, optional
        Negative eigenvalues of the covariance difference down to ``tolerance`` times the
        target's major-axis variance count as zero (rounding when the ellipses touch).

    Returns
    -------
    numpy.ndarray
        ``[major, minor, pa]`` of the correcting beam, pa in ``[0, pi)``.

    Raises
    ------
    ValueError
        If ``target_beam`` does not contain ``beam``.
    """
    target = _covariance(target_beam)
    difference = target - _covariance(beam)
    variances, directions = np.linalg.eigh(difference)
    if variances[0] < -tolerance * np.linalg.eigvalsh(target)[-1]:
        raise ValueError(
            f"The target beam {target_beam} does not contain the beam {beam}."
        )
    variances = np.maximum(variances, 0.0)
    if variances[1] == 0.0:
        return np.zeros(3)
    major_direction = directions[:, 1]
    pa = math.atan2(major_direction[0], -major_direction[1]) % math.pi
    return np.array([*np.sqrt(2.0 * _FOUR_LN_2 * variances[::-1]), pa])


def _covariance(beam):
    """Covariance of a beam in the frame where the major axis lies along (sin pa, -cos pa)."""
    major, minor, pa = (float(x) for x in beam)
    major_axis = np.array([math.sin(pa), -math.cos(pa)])
    minor_axis = np.array([math.cos(pa), math.sin(pa)])
    return (
        major**2 * np.outer(major_axis, major_axis)
        + minor**2 * np.outer(minor_axis, minor_axis)
    ) / (2.0 * _FOUR_LN_2)


def _unwrap(pa):
    """GaussianBeam::_unwrap: the position angle in (-pi/2, pi/2]."""
    if pa > 0.5 * math.pi or pa <= -0.5 * math.pi:
        degrees = math.fmod(math.degrees(pa), 180.0)
        if degrees > 90.0:
            degrees -= 180.0
        elif degrees <= -90.0:
            degrees += 180.0
        return math.radians(degrees)
    return pa


def _radians(beam):
    """The axes of an internal beam in radians, converted as casacore converts them."""
    major, minor, pa, in_arcsec = beam
    if in_arcsec:
        return major / (1.0 / _ARCSEC), minor / (1.0 / _ARCSEC), pa
    return major, minor, pa


def _area(beam, first_in_arcsec):
    """GaussianBeam::getArea in the unit of the first beam (without the common factor)."""
    major, minor, _, in_arcsec = beam
    if in_arcsec == first_in_arcsec:
        return major * minor
    if in_arcsec:
        return (major * minor) / (1.0 / (_ARCSEC * _ARCSEC))
    return (major * minor) / ((_ARCSEC * _ARCSEC) / 1.0)


def _result(beam):
    major, minor, pa = _radians(beam)
    return np.array([major, minor, pa % math.pi])


def _fits_inside(beam, outer):
    """Whether GaussianDeconvolver deconvolves ``beam`` from ``outer`` cleanly."""
    major_o, minor_o, pa_o = _radians(outer)
    major_b, minor_b, pa_b = _radians(beam)
    cos_o, sin_o, cos_b, sin_b = (
        math.cos(pa_o),
        math.sin(pa_o),
        math.cos(pa_b),
        math.sin(pa_b),
    )
    alpha = (
        (major_o * cos_o) ** 2
        + (minor_o * sin_o) ** 2
        - (major_b * cos_b) ** 2
        - (minor_b * sin_b) ** 2
    )
    beta = (
        (major_o * sin_o) ** 2
        + (minor_o * cos_o) ** 2
        - (major_b * sin_b) ** 2
        - (minor_b * cos_b) ** 2
    )
    gamma = 2 * (
        (minor_o**2 - major_o**2) * sin_o * cos_o
        - (minor_b**2 - major_b**2) * sin_b * cos_b
    )
    s = alpha + beta
    t = math.sqrt((alpha - beta) ** 2 + gamma**2)
    # Otherwise CASA returns its point-source case or fails; neither counts as fitting.
    return not (alpha < 0.0 or beta < 0.0 or s < t)


def _transform_ellipse_by_scaling(major, minor, pa, x_scale, y_scale):
    """CasaImageBeamSet::_transformEllipseByScaling of CASA 6.8 (CAS-14719)."""
    cos2, sin2 = math.cos(pa) ** 2, math.sin(pa) ** 2
    a = cos2 / major**2 + sin2 / minor**2
    b = -2 * math.cos(pa) * math.sin(pa) * (1 / major**2 - 1 / minor**2)
    c = sin2 / major**2 + cos2 / minor**2
    r = a / x_scale**2
    s = b * b / (4 * x_scale**2 * y_scale**2)
    t = c / y_scale**2
    u = r - t
    f1 = u * u + 4 * s
    j1 = j2 = 0.5
    # The roots tend to (r + t) / 2 +- sqrt(s) when the scaled ellipse is (nearly)
    # circular, where the general form divides zero by zero.
    if not f1 > 0.0 or abs(u) < 1e-14 * (abs(r) + abs(t)):
        k1, k2 = 0.5 * (r + t) + math.sqrt(s), 0.5 * (r + t) - math.sqrt(s)
    else:
        f2 = math.sqrt(f1) * abs(u)
        j1 = (f2 + f1) / f1 / 2
        j2 = (-f2 + f1) / f1 / 2
        if 2 * j1 - 1 == 0.0 or 2 * j2 - 1 == 0.0:
            k1, k2 = 0.5 * (r + t) + math.sqrt(s), 0.5 * (r + t) - math.sqrt(s)
        else:
            k1 = (j1 * r + j1 * t - t) / (2 * j1 - 1)
            k2 = (j2 * r + j2 * t - t) / (2 * j2 - 1)
    if not (k1 > 0.0 and k2 > 0.0):
        k1 = k2 = 0.5 * (r + t) if 0.5 * (r + t) > 0.0 else 1.0 / (major * minor)
        j1 = j2 = 1.0
    c1, c2 = math.sqrt(1 / k1), math.sqrt(1 / k2)
    sign = 1 if pa >= 0 else -1
    if c1 == c2:
        return c1, c1, 0.0
    if c1 > c2:
        return c1, c2, sign * math.acos(math.sqrt(j1))
    return c2, c1, sign * math.acos(math.sqrt(j2))


def _enclosing(largest, problem):
    """CASA's ellipse around the largest beam and a beam that does not fit inside it:
    scaled so that the largest beam becomes a circle of the same area, it takes the major
    axis and angle of the other beam and the circle's radius as minor axis; it is then
    grown by factors of 1.001 until both beams fit, at most 1000 times (``None`` when
    that is not enough)."""
    major_a = largest[0] if largest[3] else largest[0] / _ARCSEC
    minor_a = largest[1] if largest[3] else largest[1] / _ARCSEC
    major_p = problem[0] if problem[3] else problem[0] / _ARCSEC
    minor_p = problem[1] if problem[3] else problem[1] / _ARCSEC
    radius = math.sqrt(major_a * minor_a)
    x_scale, y_scale = radius / major_a, radius / minor_a
    major_b, _, pa_b = _transform_ellipse_by_scaling(
        major_p, minor_p, problem[2] - largest[2], x_scale, y_scale
    )
    major_c, minor_c, pa_c = _transform_ellipse_by_scaling(
        major_b, radius, pa_b, 1 / x_scale, 1 / y_scale
    )
    pa_c = _unwrap(pa_c + largest[2])
    for _ in range(1001):
        enclosing = (major_c, minor_c, pa_c, True)
        if _fits_inside(largest, enclosing) and _fits_inside(problem, enclosing):
            return enclosing
        major_c *= 1.001
        minor_c *= 1.001
    return None
