"""The common beam of a set of beams, and the Gaussian that takes a beam to a larger one.

A beam is ``[major, minor, pa]`` as the ``BEAM_FIT_PARAMS_*`` variables hold it: the full
widths at half maximum of the major and minor axes and the position angle of the major
axis, north through east, in radians. In the ``(l index, m index)`` frame of an image the
major axis lies along ``(sin pa, -cos pa)``.
"""

import math
from typing import NamedTuple

import numpy as np
import toolviper.utils.logger as logger
import xarray as xr

# casacore's C::arcsec and C::degree, the factors of its units "arcsec" and "deg". CASA keeps
# its enclosing ellipses in arcsec; its unit conversions are part of its result.
_ARCSEC = 0.00000484813681109535993589914098765
_DEGREE = 0.0174532925199432957692369
_AREA_FACTOR = math.pi / (4.0 * math.log(2.0))
_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
_MAX_GROWTH_STEPS = 1000
_CONTAINMENT_TOLERANCE = 1e-6
_MAX_PLANES_NAMED = 10


class _Beam(NamedTuple):
    """A beam as casacore holds it: the axes in the unit of factor ``unit`` (1 for
    radians, C::arcsec for arcsec), pa in radians as stored and unwrapped."""

    major: float
    minor: float
    pa: float
    unit: float
    unwrapped_pa: float


def common_beam(beam_fit_params):
    """The common beam of a set of beams, as CASA computes it.

    The algorithm of CASA's ``CasaImageBeamSet::getCommonBeam``, the beam of ``tclean``'s
    ``restoringbeam='common'``, as CASA 6.8.0.4 has it (CAS-14719). The first beam of
    largest area is the common beam when every other beam deconvolves from it cleanly
    (CASA's ``GaussianDeconvolver``, without its point-source case). Otherwise an ellipse
    is built around it and the last beam that does not, grown by factors of 1.001 until
    both deconvolve from it and put in its place, and the test is repeated; when 1000
    growth steps are not enough, a warning is logged and the beam of largest area is the
    result. The result is not the smallest ellipse that contains every beam, and two CASA
    rules are kept: at an exact right angle between the two beams CASA returns the ellipse
    of their major axes at once, without testing the other beams, and for position angles
    more than 90 degrees apart its enclosing ellipse is mirrored and grows until it
    contains both. :func:`correcting_beam` tells whether the result contains every beam.

    Parameters
    ----------
    beam_fit_params : xarray.DataArray or numpy.ndarray
        Beams ``[major, minor, pa]`` in radians: a DataArray whose last dimension holds
        them and whose other dimensions are among ``time``, ``frequency`` and
        ``polarization`` (``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION``, for example), or an
        array ``(time, frequency, polarization, 3)``. The beams are visited frequency
        fastest, then polarization, then time, as CASA visits the beams of an image, and
        the order can change the result: to reproduce the common beam of a CASA image,
        pass its beams in its channel order. Beams with a zero axis (CASA's null beams)
        or a value that is not finite (failed fits) are left out.

    Returns
    -------
    numpy.ndarray
        The common beam ``[major, minor, pa]`` in radians, pa in ``(-pi/2, pi/2]``.

    Raises
    ------
    ValueError
        If a beam has a negative axis or a minor axis larger than its major axis, if every
        beam is left out, or if ``beam_fit_params`` is not shaped as described.

    See Also
    --------
    correcting_beam : The correcting Gaussian of every plane, with the containment check.

    Notes
    -----
    Given the same doubles in the same units and order, the result is that of casatools
    6.7.6, except where CAS-14719 changed it, up to the rounding of sin, cos and acos, which
    the port and casatools take from the platform's math library. Where an enclosing ellipse
    touches a beam, one ulp of one of them, or of an input (float32 storage of the beams,
    for example), can change the number of growth steps and so the result by a factor of
    1.001: the math libraries of macOS and Linux give such different results for some sets.

    Examples
    --------
    ``tclean``'s ``restoringbeam='common'`` for ``image_cube_single_field``: the common
    beam of the Stokes I PSF beams made with the imager's ``image_params`` and weighting
    (the restore applies the beam of the first polarization to every polarization),
    checked before the run::

        psf = point_spread_function_cube_single_field(
            ps_store, image_params, imaging_weights_params, ...
        )
        beams = psf["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].isel(polarization=[0])
        target = common_beam(beams)
        correcting_beam(beams, target)  # raises if the target misses a plane's beam
        image_cube_single_field(..., restore=True, restoring_beam=target)
    """
    beams = _beams_in_casa_order(beam_fit_params)
    finite = beams[np.isfinite(beams).all(axis=1)]
    invalid = (finite[:, 1] < 0.0) | (finite[:, 1] > finite[:, 0])
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} beam(s) do not have 0 <= minor <= major, the first "
            f"{finite[invalid][0].tolist()}."
        )
    usable = finite[finite[:, 1] > 0.0]
    if usable.size == 0:
        raise ValueError("Every beam has a zero axis or a value that is not finite.")
    beams = [
        _Beam(major, minor, pa, 1.0, unwrapped_pa)
        for (major, minor, pa), unwrapped_pa in zip(
            usable.tolist(), _unwrap(usable[:, 2]).tolist(), strict=True
        )
    ]
    while True:
        first = beams[0]
        if all(_equal(beam, first) for beam in beams):
            return _result(first)
        areas = [_area(beam, first.unit) for beam in beams]
        largest_index = 0
        for index, area in enumerate(areas):
            if area > areas[largest_index]:
                largest_index = index
        largest = beams[largest_index]
        problem = None
        for beam in beams:
            if not _equal(beam, largest) and not _deconvolves_cleanly(largest, beam):
                problem = beam
        if problem is None:
            return _result(largest)
        angle = problem.unwrapped_pa - largest.unwrapped_pa
        if abs(angle) == math.pi / 2:
            if _arcsec(largest.major, largest.unit) >= _arcsec(
                problem.major, problem.unit
            ):
                outer, inner = largest, problem
            else:
                outer, inner = problem, largest
            return np.array(
                [
                    _radians(outer.major, outer.unit),
                    _radians(inner.major, inner.unit),
                    outer.unwrapped_pa,
                ]
            )
        enclosing = _enclosing_beam(largest, problem, angle)
        if enclosing is None:
            logger.warning(
                "Could not build an enclosing beam for the common beam within "
                f"{_MAX_GROWTH_STEPS} growth steps; using the beam of largest area, "
                f"{_result(largest).tolist()} rad."
            )
            return _result(largest)
        beams[largest_index] = enclosing


def correcting_beam(beam_fit_params, restoring_beam):
    """The Gaussian that takes each beam to a larger target beam.

    Its covariance is ``D = S_T - S_p``, the target's minus the beam's (CASA Memo 10,
    CAS-12148, eqs. 16-20, in covariance form, which also holds for equal and touching
    beams): restoring a plane to ``restoring_beam`` convolves its residual with this
    Gaussian and multiplies it by ``area_ratio``. The target must contain every beam,
    ``D`` positive semidefinite; a negative eigenvalue of ``D`` down to ``-1e-6`` times
    the variance of the beam along its eigenvector (the target narrower than the beam by
    up to 5e-7 of the beam's width in that direction) counts as rounding and is set to
    zero. Memo 10's test (eqs. 24-26) lacks the condition ``det D >= 0`` and accepts
    beams the target does not contain.

    Parameters
    ----------
    beam_fit_params : xarray.DataArray or array_like
        Beams ``(..., 3)``, ``[major, minor, pa]`` in radians: the PSF beams of a cube,
        for example.
    restoring_beam : array_like
        The target ``[major, minor, pa]`` in radians, ``major >= minor > 0``,
        ``|pa| <= pi``.

    Returns
    -------
    correcting_beam_params : numpy.ndarray
        ``(..., 3)``: ``[major, minor, pa]`` of the correcting Gaussian of every beam in
        radians, pa in ``(-pi/2, pi/2]``; ``minor = 0`` where the target touches the beam,
        ``[0, 0, 0]`` where the two are equal, NaN for a beam without a fit (a value that
        is not finite, or an axis that is not positive).
    area_ratio : numpy.ndarray
        ``(...)``: the area of the target over that of the beam, NaN for a beam without a
        fit.

    Raises
    ------
    ValueError
        If ``restoring_beam`` is not ``[major, minor, pa]`` with ``major >= minor > 0``
        and ``|pa| <= pi``, or if it does not contain a beam. The message names the
        planes (indices, and frequencies for a DataArray), the beams and the factor by
        which both axes of the target must grow to contain them.

    See Also
    --------
    common_beam : The target of ``tclean``'s ``restoringbeam='common'``, with the recipe.
    """
    from astroviper.processing_functions.imaging.config_check._predicates import (
        check_restoring_beam,
    )

    check_restoring_beam(restoring_beam)
    beams = np.asarray(beam_fit_params, dtype=np.float64)
    if beams.shape[-1:] != (3,):
        raise ValueError(f"beam_fit_params must be shaped (..., 3), not {beams.shape}.")
    target = np.asarray(restoring_beam, dtype=np.float64)
    covariance, area_ratio, state = _correcting_covariance(beams, target)
    planes = [
        index for index in np.ndindex(state.shape) if state[index] == "not contained"
    ]
    if planes:
        raise ValueError(_not_contained_message(beam_fit_params, beams, target, planes))
    return _beam_from_covariance(covariance), area_ratio


def _beams_in_casa_order(beam_fit_params):
    """The beams as rows ``(n, 3)``: frequency fastest, then polarization, then time."""
    if isinstance(beam_fit_params, xr.DataArray):
        planes = beam_fit_params.dims[:-1]
        if beam_fit_params.shape[-1] != 3 or not set(planes) <= {
            "time",
            "frequency",
            "polarization",
        }:
            raise ValueError(
                "beam_fit_params must hold the beams in its last dimension and have its "
                "other dimensions among time, frequency and polarization, not "
                f"{dict(beam_fit_params.sizes)}."
            )
        order = [dim for dim in ("time", "polarization", "frequency") if dim in planes]
        values = beam_fit_params.transpose(*order, beam_fit_params.dims[-1]).values
    else:
        values = np.asarray(beam_fit_params)
        if values.ndim != 4 or values.shape[-1] != 3:
            raise ValueError(
                "beam_fit_params must be shaped (time, frequency, polarization, 3), "
                f"not {values.shape}."
            )
        values = values.transpose(0, 2, 1, 3)
    return np.asarray(values, dtype=np.float64).reshape(-1, 3)


def _radians(value, unit):
    """casacore's ``Quantum::getValue("rad")`` of a value in the unit of factor ``unit``."""
    return value / (1.0 / unit)


def _arcsec(value, unit):
    """casacore's ``Quantum::getValue("arcsec")``."""
    return value / (_ARCSEC / unit)


def _unwrap(pa):
    """casacore's ``GaussianBeam::getPA(true)`` of pa in radians: to (-pi/2, pi/2],
    through degrees."""
    degrees = np.fmod(pa / _DEGREE, 180.0)
    degrees = np.where(
        degrees > 90.0,
        degrees - 180.0,
        np.where(degrees <= -90.0, degrees + 180.0, degrees),
    )
    return np.where(
        (pa > 90.0 * _DEGREE) | (pa <= -90.0 * _DEGREE), degrees * _DEGREE, pa
    )


def _equal(beam, other):
    """casacore's ``GaussianBeam::operator==``: ``other`` in the unit of ``beam``, pa as
    stored."""
    factor = other.unit / beam.unit
    return (
        beam.major == other.major * factor
        and beam.minor == other.minor * factor
        and beam.pa == other.pa
    )


def _area(beam, unit):
    """casacore's ``GaussianBeam::getArea`` in the square of the unit of factor ``unit``."""
    return _AREA_FACTOR * (
        (beam.major * beam.minor) / ((unit * unit) / (beam.unit * beam.unit))
    )


def _result(beam):
    return np.array(
        [
            _radians(beam.major, beam.unit),
            _radians(beam.minor, beam.unit),
            beam.unwrapped_pa,
        ]
    )


def _square(value):
    return value * value


def _deconvolves_cleanly(convolved, beam):
    """Whether CASA's ``GaussianDeconvolver::deconvolve`` takes ``beam`` out of
    ``convolved`` without its point-source case or an error."""
    major_s = _radians(convolved.major, convolved.unit)
    minor_s = _radians(convolved.minor, convolved.unit)
    major_b = _radians(beam.major, beam.unit)
    minor_b = _radians(beam.minor, beam.unit)
    cos_s, sin_s = math.cos(convolved.unwrapped_pa), math.sin(convolved.unwrapped_pa)
    cos_b, sin_b = math.cos(beam.unwrapped_pa), math.sin(beam.unwrapped_pa)
    alpha = (
        _square(major_s * cos_s)
        + _square(minor_s * sin_s)
        - _square(major_b * cos_b)
        - _square(minor_b * sin_b)
    )
    beta = (
        _square(major_s * sin_s)
        + _square(minor_s * cos_s)
        - _square(major_b * sin_b)
        - _square(minor_b * cos_b)
    )
    gamma = 2 * (
        (_square(minor_s) - _square(major_s)) * sin_s * cos_s
        - (_square(minor_b) - _square(major_b)) * sin_b * cos_b
    )
    s = alpha + beta
    t = math.sqrt(_square(alpha - beta) + _square(gamma))
    return not (alpha < 0.0 or beta < 0.0 or s < t)


def _transform_ellipse_by_scaling(major, minor, pa, x_scale, y_scale):
    """CASA's ``CasaImageBeamSet::_transformEllipseByScaling`` after CAS-14719: the
    ellipse with its x and y axes scaled; pa takes the sign of the input pa."""
    cos_pa = math.cos(pa)
    sin_pa = math.sin(pa)
    cos2 = cos_pa * cos_pa
    sin2 = sin_pa * sin_pa
    major2 = major * major
    minor2 = minor * minor
    a = cos2 / major2 + sin2 / minor2
    b = -2 * cos_pa * sin_pa * (1 / major2 - 1 / minor2)
    c = sin2 / major2 + cos2 / minor2
    xs = x_scale * x_scale
    ys = y_scale * y_scale
    r = a / xs
    s = b * b / (4 * xs * ys)
    t = c / ys
    u = r - t
    f1 = u * u + 4 * s
    j1 = j2 = 0.5
    if not f1 > 0.0 or abs(u) < 1e-14 * (abs(r) + abs(t)):
        k1 = 0.5 * (r + t) + math.sqrt(s)
        k2 = 0.5 * (r + t) - math.sqrt(s)
    else:
        f2 = math.sqrt(f1) * abs(u)
        j1 = (f2 + f1) / f1 / 2
        j2 = (-f2 + f1) / f1 / 2
        if 2 * j1 - 1 == 0.0 or 2 * j2 - 1 == 0.0:
            k1 = 0.5 * (r + t) + math.sqrt(s)
            k2 = 0.5 * (r + t) - math.sqrt(s)
        else:
            k1 = (j1 * r + j1 * t - t) / (2 * j1 - 1)
            k2 = (j2 * r + j2 * t - t) / (2 * j2 - 1)
    if not (k1 > 0.0 and k2 > 0.0):
        k1 = k2 = 0.5 * (r + t) if 0.5 * (r + t) > 0.0 else 1.0 / (major * minor)
        j1 = j2 = 1.0
    c1 = math.sqrt(1 / k1)
    c2 = math.sqrt(1 / k2)
    sign = 1 if pa >= 0 else -1
    if c1 == c2:
        return c1, c1, 0.0
    if c1 > c2:
        return c1, c2, sign * math.acos(math.sqrt(j1))
    return c2, c1, sign * math.acos(math.sqrt(j2))


def _enclosing_beam(largest, problem, angle):
    """CASA's ellipse, in arcsec, around the beam of largest area and a beam that does not
    fit in it, grown until both fit; ``None`` after 1000 growth steps or for a size that is
    not finite. Scaled so that the largest beam becomes a circle of the same area, the
    ellipse has the major axis and pa of the other beam and the circle's radius as minor
    axis."""
    major_a = _arcsec(largest.major, largest.unit)
    minor_a = _arcsec(largest.minor, largest.unit)
    radius = math.sqrt(major_a * minor_a)
    x_scale = radius / major_a
    y_scale = radius / minor_a
    major_b, _, pa_b = _transform_ellipse_by_scaling(
        _arcsec(problem.major, problem.unit),
        _arcsec(problem.minor, problem.unit),
        angle,
        x_scale,
        y_scale,
    )
    major, minor, pa = _transform_ellipse_by_scaling(
        major_b, radius, pa_b, 1 / x_scale, 1 / y_scale
    )
    pa += largest.unwrapped_pa
    unwrapped_pa = float(_unwrap(pa))
    for _ in range(_MAX_GROWTH_STEPS + 1):
        if not (math.isfinite(major) and math.isfinite(minor)):
            return None
        enclosing = _Beam(major, minor, pa, _ARCSEC, unwrapped_pa)
        if _deconvolves_cleanly(enclosing, largest) and _deconvolves_cleanly(
            enclosing, problem
        ):
            return enclosing
        major *= 1.001
        minor *= 1.001
    return None


def _covariance(variance_major, variance_minor, pa):
    """Covariance ``(..., 2, 2)``, in the ``(l index, m index)`` frame, of Gaussians with
    the given variances along their axes and the major axis along ``(sin pa, -cos pa)``."""
    sin_pa, cos_pa = np.sin(pa), np.cos(pa)
    cross = (variance_minor - variance_major) * sin_pa * cos_pa
    ll = variance_major * sin_pa * sin_pa + variance_minor * cos_pa * cos_pa
    mm = variance_major * cos_pa * cos_pa + variance_minor * sin_pa * sin_pa
    return np.stack(
        [np.stack([ll, cross], axis=-1), np.stack([cross, mm], axis=-1)], axis=-2
    )


def _beam_covariance(beam_fit_params):
    """Covariance ``(..., 2, 2)`` in rad² of beams ``(..., 3)``. pa is unwrapped first,
    so a beam and its :func:`common_beam` (pa in [0, pi) from the fit, (-pi/2, pi/2]
    from CASA's rule) have the same covariance."""
    beams = np.asarray(beam_fit_params, dtype=np.float64)
    sigma_major = beams[..., 0] / _FWHM_FACTOR
    sigma_minor = beams[..., 1] / _FWHM_FACTOR
    return _covariance(
        sigma_major * sigma_major, sigma_minor * sigma_minor, _unwrap(beams[..., 2])
    )


def _eigenvalues(covariance):
    """The smaller and the larger eigenvalue of symmetric matrices ``(..., 2, 2)``."""
    half_trace = 0.5 * (covariance[..., 0, 0] + covariance[..., 1, 1])
    radius = np.hypot(
        0.5 * (covariance[..., 0, 0] - covariance[..., 1, 1]), covariance[..., 0, 1]
    )
    return half_trace - radius, half_trace + radius


def _orientation(covariance):
    """pa of the major axis of covariances ``(..., 2, 2)``, in (-pi/2, pi/2]."""
    pa = 0.5 * np.arctan2(
        -2.0 * covariance[..., 0, 1], covariance[..., 1, 1] - covariance[..., 0, 0]
    )
    return np.where(pa <= -0.5 * np.pi, pa + np.pi, pa)


def _beam_from_covariance(covariance):
    """Beams ``(..., 3)`` of covariances ``(..., 2, 2)`` in rad²; pa is 0 for a circle and
    negative eigenvalues count as zero."""
    smallest, largest = _eigenvalues(covariance)
    pa = np.where(smallest == largest, 0.0, _orientation(covariance))
    return np.stack(
        [
            _FWHM_FACTOR * np.sqrt(np.maximum(largest, 0.0)),
            _FWHM_FACTOR * np.sqrt(np.maximum(smallest, 0.0)),
            pa,
        ],
        axis=-1,
    )


def _correcting_covariance(beam_fit_params, restoring_beam):
    """The covariance of the correcting Gaussian of every beam, without raising.

    Parameters
    ----------
    beam_fit_params : array_like
        Beams ``(..., 3)``, ``[major, minor, pa]`` in radians.
    restoring_beam : array_like
        The target ``[major, minor, pa]`` in radians.

    Returns
    -------
    covariance : numpy.ndarray
        ``(..., 2, 2)``: ``D = S_T - S_p`` in rad² where the target contains the beam,
        with negative eigenvalues within the tolerance set to zero (``0`` for a beam equal
        to the target); NaN elsewhere.
    area_ratio : numpy.ndarray
        ``(...)``: the area of the target over that of the beam, NaN for a beam without a
        fit.
    state : numpy.ndarray
        ``(...)``: ``"contained"``, ``"not contained"``, or ``"no fit"`` for a beam with a
        value that is not finite or an axis that is not positive.
    """
    beams = np.asarray(beam_fit_params, dtype=np.float64)
    target = np.asarray(restoring_beam, dtype=np.float64)
    fitted = (
        np.isfinite(beams).all(axis=-1) & (beams[..., 0] > 0.0) & (beams[..., 1] > 0.0)
    )
    point_spread_function = _beam_covariance(beams)
    difference = _beam_covariance(target) - point_spread_function
    smallest, largest = _eigenvalues(difference)
    # Each eigenvalue of D against the variance of the beam along its eigenvector:
    # (cos pa, sin pa) for the smaller, (sin pa, -cos pa) for the larger.
    pa = _orientation(difference)
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    ll, lm, mm = (
        point_spread_function[..., 0, 0],
        point_spread_function[..., 0, 1],
        point_spread_function[..., 1, 1],
    )
    contained = (
        fitted
        & (
            smallest
            >= -_CONTAINMENT_TOLERANCE
            * (ll * cos_pa * cos_pa + 2.0 * lm * cos_pa * sin_pa + mm * sin_pa * sin_pa)
        )
        & (
            largest
            >= -_CONTAINMENT_TOLERANCE
            * (ll * sin_pa * sin_pa - 2.0 * lm * cos_pa * sin_pa + mm * cos_pa * cos_pa)
        )
    )
    clipped = _covariance(np.maximum(largest, 0.0), 0.0, pa)
    covariance = np.where(
        contained[..., None, None],
        np.where((smallest < 0.0)[..., None, None], clipped, difference),
        np.nan,
    )
    area_ratio = np.full(fitted.shape, np.nan)
    np.divide(
        target[0] * target[1],
        beams[..., 0] * beams[..., 1],
        out=area_ratio,
        where=fitted,
    )
    state = np.where(
        contained, "contained", np.where(fitted, "not contained", "no fit")
    )
    return covariance, area_ratio, state


def _growth_factor(beam_fit_params, restoring_beam):
    """The factor ``(...)`` by which both axes of the target must grow to contain each
    fitted beam ``(..., 3)``, ``sqrt(max eig(S_T^-1 S_p))``."""
    return np.sqrt(
        np.linalg.eigvals(
            np.linalg.solve(
                _beam_covariance(restoring_beam), _beam_covariance(beam_fit_params)
            )
        ).real.max(axis=-1)
    )


def _not_contained_message(beam_fit_params, beams, target, planes):
    """The error of :func:`correcting_beam` for the planes whose beam the target misses."""
    selected = np.array([beams[index] for index in planes])
    growth = _growth_factor(selected, target)
    lines = [
        f"  {_plane_name(beam_fit_params, index)}: {_format_beam(beam)}, growth "
        f"factor {factor:.7g}"
        for index, beam, factor in zip(
            planes[:_MAX_PLANES_NAMED], selected, growth, strict=False
        )
    ]
    if len(planes) > _MAX_PLANES_NAMED:
        lines.append(f"  and {len(planes) - _MAX_PLANES_NAMED} more")
    return (
        f"restoring_beam {_format_beam(target)} does not contain the beam of "
        f"{len(planes)} of {math.prod(beams.shape[:-1])} plane(s):\n"
        + "\n".join(lines)
        + f"\nBoth axes of restoring_beam must grow by {growth.max():.7g} to contain "
        "them. A beam typed from a rounded printout must be at least as large as the "
        "largest beam along every direction: round it up."
    )


def _plane_name(beam_fit_params, index):
    if not isinstance(beam_fit_params, xr.DataArray):
        return f"plane {index}"
    names = []
    for dim, position in zip(beam_fit_params.dims[:-1], index, strict=True):
        name = f"{dim} {position}"
        if dim == "frequency" and "frequency" in beam_fit_params.coords:
            name += f" ({float(beam_fit_params['frequency'][position]):.10g} Hz)"
        names.append(name)
    return ", ".join(names)


def _format_beam(beam):
    return "[" + ", ".join(f"{value:.10g}" for value in beam) + "] rad"
