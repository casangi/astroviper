import numpy as np
import scipy.fft
import toolviper.utils.logger as logger
import xarray as xr

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from astroviper.utils.param_docs import shares_param_docs

# FWHM = 2 * sqrt(2 * ln 2) * sigma. Matches the factor used by the PSF Gaussian
# fit (astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit)
# so a beam built here round-trips back to the same [major, minor, pa].
FWHM_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))


# -----------------------------------------------------------------------------
# Runtime restoration product roles
# -----------------------------------------------------------------------------


def resolve_restoration_product(img_xds, data_group_name, role, *, fallback=None):
    """Resolve a required restoration role without reading or modifying pixels.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image products and their data-group registrations.
    data_group_name : str
        Group containing the requested role.
    role : str
        Logical role, such as sky, restoring-beam parameters, or primary beam.
    fallback : str, optional
        Explicit compatibility variable used only when the role is unregistered.
        Continuum permits legacy PSF and restoring-beam registrations; cube
        restoration requires its registered roles. A missing registered variable
        never falls back to another product.

    Returns
    -------
    str
        Existing variable name. No data-group entries or defaults are inserted.

    Raises
    ------
    KeyError
        If the data group, required role, or referenced variable is missing.
    """
    # --- Required data group ---
    groups = img_xds.attrs.get("data_groups", {})
    if data_group_name not in groups:
        raise KeyError(
            f"Restoration data group {data_group_name!r} is missing. "
            f"Available groups are {list(groups)}."
        )

    # --- Registered role or explicitly supported legacy product ---
    variable_name = groups[data_group_name].get(role)
    if variable_name is None:
        if fallback is None:
            raise KeyError(
                f"Restoration data group {data_group_name!r} does not define {role!r}."
            )
        if fallback not in img_xds:
            raise KeyError(
                f"Restoration data group {data_group_name!r} does not define {role!r}, "
                f"and {fallback!r} is absent."
            )
        variable_name = fallback

    # --- Actual product availability ---
    if variable_name not in img_xds:
        raise KeyError(
            f"Restoration variable {variable_name!r} for role {role!r} "
            f"in data group {data_group_name!r} is missing."
        )
    return variable_name


# -----------------------------------------------------------------------------
# Clean-beam kernels and plane restoration
# -----------------------------------------------------------------------------


def _elliptical_gaussian_kernel(ny, nx, major_fwhm_pix, minor_fwhm_pix, pa, dtype):
    """Unit-peak elliptical Gaussian on an ``(ny, nx)`` pixel grid, centered.

    The kernel is centred on pixel ``(ny // 2, nx // 2)`` with a peak value of
    ``1.0`` (so convolving a model in Jy/pixel yields a restored image in
    Jy/beam). The orientation convention matches
    :func:`~astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit.point_spread_function_gaussian_fit`:
    that fit reports the position angle of the major axis measured from the
    ``m`` (axis 1) towards the ``l`` (axis 0) pixel axis, i.e. the major axis
    lies along ``(sin pa, -cos pa)`` in ``(l-index, m-index)`` space.  A beam
    built here from a fitted ``[major, minor, pa]`` therefore round-trips back
    to the same parameters through that fit (verified in the unit tests).

    Parameters
    ----------
    ny, nx : int
        Image dimensions along ``l`` (axis 0) and ``m`` (axis 1).
    major_fwhm_pix, minor_fwhm_pix : float
        Major/minor axis FWHM in **pixels**.
    pa : float
        Position angle in radians.
    dtype : numpy dtype
        Output dtype (the image dtype, e.g. ``float32``).

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` unit-peak elliptical Gaussian.
    """
    # Work at the image dtype (never below float32): pixel offsets are exact in
    # float32 up to 2^24 and the Gaussian argument only needs ~7 significant
    # digits, while full-plane float64 temporaries cost ~1 GB each at 11 250².
    work_dtype = np.result_type(dtype, np.float32)
    di = (np.arange(ny, dtype=work_dtype) - ny // 2)[:, None]
    dj = (np.arange(nx, dtype=work_dtype) - nx // 2)[None, :]

    # point_spread_function_gaussian_fit measures the position angle from the m
    # axis (axis 1) towards the l axis (axis 0), the complement of the angle in
    # the rotated-coordinate form below, so build the beam at (pi/2 - pa). With
    # this the major axis lies along (sin pa, -cos pa) and a beam built from a
    # fitted [major, minor, pa] reproduces that fit.
    theta = 0.5 * np.pi - pa
    # Python-float scalars: NumPy-2 promotion keeps work_dtype planes at
    # work_dtype for python-float operands, but a np.float64 scalar would
    # silently promote every float32 plane back to float64.
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    sigma_major = float(major_fwhm_pix) / FWHM_factor
    sigma_minor = float(minor_fwhm_pix) / FWHM_factor

    # Project onto the beam's principal axes (major along (cos theta, -sin theta))
    # and accumulate the Gaussian argument in place, so peak scratch is two
    # (ny, nx) planes at work_dtype (the broadcasts of di/dj are the only
    # full-plane allocations).
    u = di * cos_t - dj * sin_t  # major-axis coordinate (pixels)
    u /= sigma_major
    u *= u
    v = di * sin_t + dj * cos_t  # minor-axis coordinate (pixels)
    v /= sigma_minor
    v *= v
    u += v
    del v
    u *= -0.5
    np.exp(u, out=u)
    return u.astype(dtype, copy=False)


def _inverse_rfft2(spectrum, shape, workers):
    """``scipy.fft.irfft2(spectrum, s=shape, workers=workers)``, element for
    element, without its hidden temporary. ``spectrum`` is overwritten.

    scipy's two-axis inverse real FFT (pocketfft in scipy 1.17, ducc0 in
    1.18) runs the complex inverse transform along the first axis into a
    temporary copy of the whole half spectrum, allocated in C++ (one image
    plane that tracemalloc does not see), then the complex-to-real transform
    along the last axis, scaled by ``1 / (ny * nx)``. The same two transforms
    are done here, the first in place in ``spectrum``, both unscaled
    (``norm="forward"`` of an inverse transform), and the result is then
    multiplied by the factor scipy computes, ``1 / (ny * nx)`` in long double
    rounded to the image dtype: the same operations on the same values, so
    the same result, without the temporary.
    """
    ny, nx = shape
    spectrum = scipy.fft.ifft(
        spectrum, axis=-2, norm="forward", overwrite_x=True, workers=workers
    )
    image = scipy.fft.irfft(spectrum, n=nx, axis=-1, norm="forward", workers=workers)
    image *= image.dtype.type(np.longdouble(1) / np.longdouble(ny * nx))
    return image


def _clean_beam_kernel_ft(beam_params, ny, nx, delta, dtype, workers):
    """FFT of the centred clean beam of one frequency, ``None`` without a beam.

    ``beam_params`` are ``[major, minor, pa]`` (FWHM and position angle in
    radians) and ``delta`` the pixel size in radians. A fit that is not finite
    or not positive has no clean beam.
    """
    major, minor, pa = (float(x) for x in beam_params)
    if not np.isfinite([major, minor, pa]).all() or major <= 0 or minor <= 0:
        return None
    kernel = _elliptical_gaussian_kernel(
        ny, nx, major / delta, minor / delta, pa, dtype
    )
    # centre shifted to the origin; only the transform is kept
    return scipy.fft.rfft2(scipy.fft.ifftshift(kernel), workers=workers)


def _convolve_with_clean_beam(plane, kernel_ft, workers):
    """``plane`` convolved with the clean beam whose transform is ``kernel_ft``.

    A real FFT at the dtype of the plane; the spectrum is multiplied in place
    and inverse transformed with :func:`_inverse_rfft2`, so at most the
    spectrum and the output are live besides the plane.
    """
    ny, nx = plane.shape
    plane_ft = scipy.fft.rfft2(plane, workers=workers)
    np.multiply(plane_ft, kernel_ft, out=plane_ft)
    return _inverse_rfft2(plane_ft, (ny, nx), workers)


def _multiply_by_correcting_transform(spectrum, covariance_pixels, area_ratio, nx):
    """Multiply an ``rfft2`` half spectrum in place by the correcting transform.

    ``H(f) = r exp(-2 pi^2 f^T D f)``, with ``D`` (``covariance_pixels``) the
    covariance of the correcting Gaussian in pixels^2 of the ``(l index,
    m index)`` frame, ``r`` the area ratio, and ``f`` the frequencies of the
    spectrum in cycles per pixel (``fftfreq`` along ``l``, ``rfftfreq(nx)``
    along ``m``). The exponent is evaluated for 256 rows at a time in one
    float64 buffer and cast to the real dtype of the spectrum for the
    multiplication.
    """
    (c00, c01), (_, c11) = covariance_pixels
    f0 = scipy.fft.fftfreq(spectrum.shape[0])
    f1 = scipy.fft.rfftfreq(nx)
    cross = 2.0 * c01 * f1
    along_m = c11 * f1 * f1
    real_dtype = np.finfo(spectrum.dtype).dtype
    exponent = np.empty((min(256, f0.size), f1.size))
    for start in range(0, f0.size, 256):
        rows = f0[start : start + 256]
        block = exponent[: rows.size]
        np.multiply(rows[:, None], cross, out=block)
        block += along_m
        block += (c00 * rows * rows)[:, None]
        block *= -2.0 * np.pi**2
        np.exp(block, out=block)
        block *= area_ratio
        spectrum[start : start + rows.size] *= block.astype(real_dtype, copy=False)


def _correcting_spectrum(residual_plane, covariance_pixels, area_ratio, workers):
    """``rfft2`` of a residual plane times the correcting transform.

    Non-finite residual pixels are zero for the transform, in a copy freed
    right after it. Returns the spectrum and whether every pixel is finite.
    """
    finite = bool(np.isfinite(residual_plane).all())
    spectrum = scipy.fft.rfft2(
        residual_plane
        if finite
        else np.where(np.isfinite(residual_plane), residual_plane, 0),
        workers=workers,
    )
    _multiply_by_correcting_transform(
        spectrum, covariance_pixels, area_ratio, residual_plane.shape[1]
    )
    return spectrum, finite


def _residual_to_beam(residual_plane, covariance_pixels, area_ratio, workers):
    """``R (*) K``: a residual plane taken to the target beam, as a new plane.

    ``D = 0`` scales the residual by the area ratio without a transform.
    Otherwise non-finite residual pixels are NaN in the result.
    """
    if not covariance_pixels.any():
        return residual_plane * residual_plane.dtype.type(area_ratio)
    spectrum, finite = _correcting_spectrum(
        residual_plane, covariance_pixels, area_ratio, workers
    )
    plane = _inverse_rfft2(spectrum, residual_plane.shape, workers)
    del spectrum
    if not finite:
        plane[~np.isfinite(residual_plane)] = np.nan
    return plane


def primary_beam_corrected_plane(
    primary_beam_plane,
    primary_beam_limit,
    model_plane,
    residual_plane,
    kernel_ft=None,
    workers=1,
    out=None,
):
    """One plane of the primary beam corrected restored image.

    With ``P`` the (power) primary beam, ``M`` the model, ``R`` the residual
    and ``B`` the clean beam, the corrected plane is ``(M / P) * B + R / P``:
    the model is divided by the primary beam where it is a pixel value and
    then convolved with the clean beam, so a model that represents the
    apparent sky ``I P`` gives the true sky convolved with the clean beam,
    ``I * B``, exactly. Pixels where the primary beam is below
    ``primary_beam_limit`` are blanked with NaN, and the model is taken as
    zero there before the convolution. Non-finite and non-positive beam
    values are always blanked, including when ``primary_beam_limit`` is zero.

    CASA's ``pbcor`` divides the restored image instead, ``(M * B + R) / P``,
    which differs from this wherever the primary beam changes across the
    clean beam.

    Parameters
    ----------
    primary_beam_plane : numpy.ndarray
        The primary beam power of the plane.
    primary_beam_limit : float
        Cutoff of the primary beam power below which the plane is blanked.
    model_plane, residual_plane : numpy.ndarray
        Model and residual of the plane.
    kernel_ft : numpy.ndarray, optional
        Transform of the clean beam from :func:`_clean_beam_kernel_ft`;
        ``None`` (no clean beam) leaves the divided model unconvolved, as the
        restore step leaves the model unrestored.
    workers : int, optional
        Threads handed to ``scipy.fft``.
    out : numpy.ndarray, optional
        Plane, at the dtype of the residual plane, the result is written to;
        a new one by default.

    Returns
    -------
    numpy.ndarray
        The corrected plane, at the dtype of the residual plane (``out`` when
        given).
    """
    inside = (
        np.isfinite(primary_beam_plane)
        & (primary_beam_plane > 0.0)
        & (primary_beam_plane >= primary_beam_limit)
    )
    if out is None:
        out = np.empty(residual_plane.shape, dtype=residual_plane.dtype)
    # the model divided by the primary beam inside the cutoff, zero outside
    out.fill(0)
    np.divide(model_plane, primary_beam_plane, out=out, where=inside)
    convolve = kernel_ft is not None and out.any()
    if convolve:
        # only the spectrum of the divided model is kept; ``out`` is reused
        scaled = scipy.fft.rfft2(out, workers=workers)
        np.multiply(scaled, kernel_ft, out=scaled)
    else:
        scaled = out.copy()
    out.fill(np.nan)
    np.divide(residual_plane, primary_beam_plane, out=out, where=inside)
    del inside
    if convolve:
        scaled = _inverse_rfft2(scaled, out.shape, workers)
    # NaN outside the cutoff stays NaN
    out += scaled
    return out


def elliptical_gaussian_uv_taper(u, v, major, minor, pa):
    """Analytic visibility taper of an elliptical-Gaussian sky component.

    The Fourier transform of the same elliptical Gaussian that
    :func:`_elliptical_gaussian_kernel` evaluates on the image plane (restore's
    clean beam), normalised to a **unit-total-flux** component, so the taper is
    1 at ``(u, v) = (0, 0)`` and multiplying a point source's visibilities by it
    turns the point source into a Gaussian of the same integrated flux::

        T(u, v) = exp(-(pi^2 / (4 ln 2)) * [ major^2 (u sin pa + v cos pa)^2
                                           + minor^2 (u cos pa - v sin pa)^2 ])

    This is the single source of truth the simulation subdomain uses to
    simulate Gaussian sources (no duplicated Gaussian parametrisation); the
    unit tests pin it to the FFT of :func:`_elliptical_gaussian_kernel`, so the
    two cannot drift apart.  In sky coordinates the major axis lies along
    ``(sin pa, cos pa)`` in ``(l, m)`` -- position angle measured from the
    ``+m`` axis towards the ``+l`` axis -- which is the same beam that
    ``[major, minor, pa]`` describes in ``BEAM_FIT_PARAMS`` / the restore step
    (their pixel-index convention differs only by the sign of the ``l`` axis,
    under which the Gaussian is invariant).

    Parameters
    ----------
    u, v : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in units of the observing wavelength.
    major, minor : float, radians
        FWHM of the major and minor axes on the sky.
    pa : float, radians
        Position angle of the major axis.

    Returns
    -------
    numpy.ndarray
        Broadcast shape of ``u`` and ``v``; real taper in ``(0, 1]``.

    See Also
    --------
    _elliptical_gaussian_kernel : the image-plane form (unit peak).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    sin_pa = float(np.sin(pa))
    cos_pa = float(np.cos(pa))
    factor = np.pi**2 / (4.0 * np.log(2.0))
    return np.exp(
        -factor
        * (
            (float(major) * (u * sin_pa + v * cos_pa)) ** 2
            + (float(minor) * (u * cos_pa - v * sin_pa)) ** 2
        )
    )


def _divide_taylor_polynomials(
    apparent_da,
    primary_beam_taylor_da,
    limit=0.2,
    outside_value=None,
):
    """Recover intrinsic Taylor coefficients from ``J(x) = A(x) I(x)``.

    Coefficients are solved recursively,

    ``I_t = (J_t - sum(A_p I_(t-p), p=1..t)) / A_0``.

    Pixels whose zeroth-order beam is non-finite, non-positive, or below
    ``limit`` receive ``outside_value`` in every output coefficient. Restored
    residuals use NaN; models use zero so masked model pixels do not
    contaminate the clean-beam FFT.
    """
    if outside_value is None:
        outside_value = np.nan

    image_dims = ("time", "taylor_term", "polarization", "l", "m")
    missing_image_dims = [dim for dim in image_dims if dim not in apparent_da.dims]
    missing_beam_dims = [
        dim for dim in image_dims if dim not in primary_beam_taylor_da.dims
    ]
    if missing_image_dims or missing_beam_dims:
        raise ValueError(
            "Taylor polynomial division requires dimensions "
            f"{image_dims}; image missing {missing_image_dims}, beam missing "
            f"{missing_beam_dims}."
        )

    apparent = apparent_da.transpose(*image_dims)
    primary_beam = primary_beam_taylor_da.transpose(*image_dims)
    if primary_beam.sizes["taylor_term"] < apparent.sizes["taylor_term"]:
        raise ValueError(
            "PRIMARY_BEAM_TAYLOR must contain at least as many coefficients as "
            "the apparent-sky Taylor stack."
        )
    for dimension in ("time", "polarization", "l", "m"):
        if primary_beam.sizes[dimension] != apparent.sizes[dimension]:
            raise ValueError(
                f"PRIMARY_BEAM_TAYLOR {dimension!r} size does not match the image."
            )

    # This triangular division is exact for coefficients of the full formal
    # Taylor series. In MFS, however, J and A are independent finite-order
    # weighted fits over the selected band. Fitting, truncating, and dividing
    # do not commute: omitted orders can alias differently into the retained
    # coefficients and bias I_0, alpha, and especially beta. The effect grows
    # with fractional bandwidth and off-axis beam chromaticity, and can be much
    # larger when the reference frequency is away from the weighted band
    # centre; a flat intrinsic spectrum remains exact.
    apparent_values = np.asarray(apparent.values)
    beam_values = np.asarray(primary_beam.values)
    intrinsic = np.empty_like(apparent_values)
    beam_zero = beam_values[:, 0]
    supported = np.isfinite(beam_zero) & (beam_zero > 0.0) & (beam_zero >= float(limit))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for order in range(apparent.sizes["taylor_term"]):
            coefficient = apparent_values[:, order].copy()
            for beam_order in range(1, order + 1):
                coefficient -= (
                    beam_values[:, beam_order] * intrinsic[:, order - beam_order]
                )
            intrinsic[:, order] = np.where(
                supported,
                coefficient / beam_zero,
                outside_value,
            )
    intrinsic[~np.isfinite(intrinsic)] = outside_value
    result = apparent.copy(data=intrinsic).transpose(*apparent_da.dims)
    result.attrs = apparent_da.attrs.copy()
    result.attrs.update(
        {
            "primary_beam_correction": "taylor_polynomial_division",
            "primary_beam_limit": float(limit),
        }
    )
    return result


def _correcting_transforms(beam, restoring_beam, pixel_size):
    """The correcting transform of every plane whose residual goes to the target.

    Parameters
    ----------
    beam : numpy.ndarray
        PSF beams ``(time, plane, 3)``: ``[major, minor, pa]`` in radians.
    restoring_beam : array_like
        Target ``[major, minor, pa]`` in radians.
    pixel_size : float
        Absolute image-cell size in radians.

    Returns
    -------
    dict
        ``(time, plane)`` to ``(covariance_pixels, area_ratio)`` of
        :func:`_multiply_by_correcting_transform`. A plane whose PSF beam is
        the target, has no fit or is not contained by the target is left out:
        its residual is kept unconvolved and unscaled. The last two are named
        in one warning.
    """
    from astroviper.processing_functions.image_analysis.common_beam import (
        _correcting_covariance,
    )

    covariance, area_ratio, state = _correcting_covariance(beam, restoring_beam)
    transforms, not_contained, no_fit = {}, [], []
    for index in np.ndindex(beam.shape[:2]):
        if state[index] == "no fit":
            no_fit.append(index)
        elif state[index] == "not contained":
            not_contained.append(index)
        elif covariance[index].any() or area_ratio[index] != 1.0:
            transforms[index] = (
                covariance[index] / pixel_size**2,
                float(area_ratio[index]),
            )
    if not_contained or no_fit:
        logger.warning(
            f"restoring_beam {list(map(float, restoring_beam))} rad: the residual "
            "of these (time, plane) planes is kept unconvolved and unscaled. "
            "Not contained by the target (growth factor that would contain "
            f"them): {_growth_factors(beam, restoring_beam, not_contained)}. "
            f"Without a PSF fit: {no_fit}."
        )
    return transforms


def _growth_factors(beam, restoring_beam, planes):
    """``(time, plane)`` with the uniform growth of the target that would
    contain the plane's PSF beam, ``sqrt(max eig(S_T^-1 S_p))``."""
    from astroviper.processing_functions.image_analysis.common_beam import (
        _growth_factor,
    )

    if not planes:
        return []
    growth = _growth_factor(np.array([beam[index] for index in planes]), restoring_beam)
    return [
        (index, round(float(factor), 6))
        for index, factor in zip(planes, growth, strict=True)
    ]


def _primary_beam_inside(primary_beam_plane, primary_beam_limit):
    """Pixels where the primary beam is finite, positive and not below the cutoff."""
    return (
        np.isfinite(primary_beam_plane)
        & (primary_beam_plane > 0.0)
        & (primary_beam_plane >= primary_beam_limit)
    )


def _restore_plane_to_beam(
    model_plane,
    residual_plane,
    kernel_ft,
    covariance_pixels,
    area_ratio,
    workers,
    apparent_plane=None,
    corrected_plane=None,
    primary_beam_plane=None,
    primary_beam_limit=0.2,
):
    """Restore one plane to the target beam into the given output planes.

    The model is convolved with the target's clean beam (``kernel_ft``,
    ``None`` for a zero model) and the residual is taken to the target,
    ``R (*) K``. The corrected plane is ``(M / P) (*) B_T + (R (*) K) / P``,
    NaN where the primary beam is below ``primary_beam_limit``. The model
    parts are made first, so the model plane may be the apparent output plane,
    and at most three planes besides the outputs are live.

    Returns
    -------
    float
        Wall-clock seconds spent on the corrected plane alone.
    """
    import time

    convolve_model = kernel_ft is not None and model_plane.any()
    if corrected_plane is None and covariance_pixels.any():
        # One inverse transform of rfft2(R) H + rfft2(M) B_T.
        spectrum, finite = _correcting_spectrum(
            residual_plane, covariance_pixels, area_ratio, workers
        )
        if convolve_model:
            model_spectrum = scipy.fft.rfft2(model_plane, workers=workers)
            model_spectrum *= kernel_ft
            spectrum += model_spectrum
            del model_spectrum
        apparent_plane[...] = _inverse_rfft2(spectrum, apparent_plane.shape, workers)
        del spectrum
        if not finite:
            apparent_plane[~np.isfinite(residual_plane)] = np.nan
        return 0.0

    corrected_seconds = 0.0
    if corrected_plane is not None:
        start = time.time()
        corrected_plane.fill(0)
        np.divide(
            model_plane,
            primary_beam_plane,
            out=corrected_plane,
            where=_primary_beam_inside(primary_beam_plane, primary_beam_limit),
        )
        if kernel_ft is not None and corrected_plane.any():
            corrected_plane[...] = _convolve_with_clean_beam(
                corrected_plane, kernel_ft, workers
            )
        corrected_seconds += time.time() - start
    if apparent_plane is not None:
        if convolve_model:
            apparent_plane[...] = _convolve_with_clean_beam(
                model_plane, kernel_ft, workers
            )
        else:
            apparent_plane.fill(0)
    residual_to_beam = _residual_to_beam(
        residual_plane, covariance_pixels, area_ratio, workers
    )
    if apparent_plane is not None:
        apparent_plane += residual_to_beam
    if corrected_plane is not None:
        start = time.time()
        inside = _primary_beam_inside(primary_beam_plane, primary_beam_limit)
        np.divide(
            residual_to_beam, primary_beam_plane, out=residual_to_beam, where=inside
        )
        residual_to_beam[~inside] = np.nan
        del inside
        corrected_plane += residual_to_beam
        corrected_seconds += time.time() - start
    return corrected_seconds


def _correct_residual_arrays(
    residual, beam, restoring_beam, pixel_size, processing_function_threads=1
):
    """Residual planes taken to the target beam, ``R (*) K``, as a new array.

    Parameters
    ----------
    residual : numpy.ndarray
        Residual planes ``(time, plane, polarization, l, m)``.
    beam : numpy.ndarray
        PSF beams ``(time, plane, polarization, 3)``; the beam of polarization
        0 applies to every polarization.
    restoring_beam : array_like
        Target ``[major, minor, pa]`` in radians.
    pixel_size : float
        Absolute image-cell size in radians.
    processing_function_threads : int, optional
        Number of workers supplied to SciPy FFT operations. Values ``<= 0``
        use all available cores.

    Returns
    -------
    numpy.ndarray
        ``R (*) K`` of every plane, at the dtype of ``residual``. A plane whose
        PSF beam is the target, has no fit or is not contained by the target
        keeps its residual (:func:`_correcting_transforms`).
    """
    workers = (
        processing_function_threads
        if processing_function_threads and processing_function_threads > 0
        else -1
    )
    transforms = _correcting_transforms(beam[:, :, 0], restoring_beam, pixel_size)
    corrected = np.empty_like(residual)
    for index in np.ndindex(residual.shape[:3]):
        transform = transforms.get(index[:2])
        corrected[index] = (
            residual[index]
            if transform is None
            else _residual_to_beam(residual[index], *transform, workers)
        )
    return corrected


def _restore_image_arrays(
    residual,
    model,
    beam,
    pixel_size,
    processing_function_threads=1,
    primary_beam=None,
    primary_beam_limit=0.2,
    output="apparent",
    consume_model=False,
    restoring_beam=None,
):
    """Restore generic image planes without assuming an axis coordinate name.

    Cube and continuum restoration both have arrays shaped as ``(time, plane,
    polarization, l, m)``; only the meaning of ``plane`` differs (frequency or
    Taylor term). This core owns their common clean-beam convolution, safe
    primary-beam division, cutoff masking, and output allocation.

    With ``restoring_beam`` every plane is restored to that beam: the model is
    convolved with its clean beam, built once per call, and the residual is
    taken from the plane's beam to it (:func:`_restore_plane_to_beam`), except
    where :func:`_correcting_transforms` keeps the residual.

    Parameters
    ----------
    residual, model : numpy.ndarray
        Five-dimensional image arrays with identical shape.
    beam : numpy.ndarray
        Restoring-beam parameters with shape
        ``(time, plane, polarization, beam_params)``.
    pixel_size : float
        Absolute image-cell size in radians.
    processing_function_threads : int, optional
        Number of workers supplied to SciPy FFT operations. Values ``<= 0``
        use all available cores.
    primary_beam : numpy.ndarray, optional
        Primary-beam power with the same image-plane layout as ``residual``.
        Required for a primary-beam-corrected output.
    primary_beam_limit : float, optional
        Beam cutoff below which corrected pixels are set to NaN.
    output : {"apparent", "primary_beam_corrected", "both"}, optional
        Restored products to calculate.
    consume_model : bool, optional
        Reuse the model buffer for the apparent restored output when possible.
    restoring_beam : array_like, optional
        Target beam ``[major, minor, pa]`` in radians; ``None`` restores each
        plane with its own beam.

    Returns
    -------
    products : dict[str, numpy.ndarray]
        Requested arrays keyed by ``"apparent"`` and/or
        ``"primary_beam_corrected"``.
    timing : dict[str, float]
        Wall-clock seconds spent on the apparent and corrected calculations.
    """
    import time

    allowed_outputs = {"apparent", "primary_beam_corrected", "both"}
    if output not in allowed_outputs:
        raise ValueError(
            f"output must be one of {sorted(allowed_outputs)}; received {output!r}."
        )
    if residual.ndim != 5 or model.shape != residual.shape:
        raise ValueError(
            "residual and model must be identically shaped five-dimensional arrays."
        )
    expected_beam_shape = residual.shape[:3] + (3,)
    if beam.shape != expected_beam_shape:
        raise ValueError(
            "beam must have shape (time, plane, polarization, 3); received "
            f"{beam.shape}, expected {expected_beam_shape}."
        )

    make_apparent = output in {"apparent", "both"}
    make_corrected = output in {"primary_beam_corrected", "both"}
    if make_corrected:
        if primary_beam is None:
            raise ValueError(
                "primary_beam is required for a primary-beam-corrected output."
            )
        if primary_beam.ndim != 5 or primary_beam.shape[:2] != residual.shape[:2]:
            raise ValueError(
                "primary_beam must have time and plane axes matching the image."
            )
        if primary_beam.shape[-2:] != residual.shape[-2:]:
            raise ValueError(
                "primary_beam spatial axes must match the image spatial axes."
            )
        if primary_beam.shape[2] not in (1, residual.shape[2]):
            raise ValueError(
                "primary_beam polarization must be singleton or match the image."
            )

    workers = (
        processing_function_threads
        if processing_function_threads and processing_function_threads > 0
        else -1
    )
    nt, nplane, npol, ny, nx = residual.shape
    consume = (
        make_apparent
        and consume_model
        and model.dtype == residual.dtype
        and model is not residual
    )
    apparent = model if consume else np.empty_like(residual) if make_apparent else None
    corrected = np.empty_like(residual) if make_corrected else None
    apparent_seconds = 0.0
    corrected_seconds = 0.0
    total_start = time.time()
    if restoring_beam is not None:
        target_kernel_ft = (
            _clean_beam_kernel_ft(
                restoring_beam, ny, nx, pixel_size, residual.dtype, workers
            )
            if model.any()
            else None
        )
        transforms = _correcting_transforms(beam[:, :, 0], restoring_beam, pixel_size)

    for time_index in range(nt):
        for plane_index in range(nplane):
            if restoring_beam is None:
                kernel_ft = _clean_beam_kernel_ft(
                    beam[time_index, plane_index, 0],
                    ny,
                    nx,
                    pixel_size,
                    residual.dtype,
                    workers,
                )
                transform = None
            else:
                kernel_ft = target_kernel_ft
                transform = transforms.get((time_index, plane_index))
            for polarization_index in range(npol):
                model_plane = model[time_index, plane_index, polarization_index]
                residual_plane = residual[time_index, plane_index, polarization_index]
                if transform is not None:
                    beam_polarization = (
                        polarization_index
                        if make_corrected and primary_beam.shape[2] == npol
                        else 0
                    )
                    corrected_seconds += _restore_plane_to_beam(
                        model_plane,
                        residual_plane,
                        kernel_ft,
                        *transform,
                        workers,
                        apparent_plane=(
                            apparent[time_index, plane_index, polarization_index]
                            if make_apparent
                            else None
                        ),
                        corrected_plane=(
                            corrected[time_index, plane_index, polarization_index]
                            if make_corrected
                            else None
                        ),
                        primary_beam_plane=(
                            primary_beam[time_index, plane_index, beam_polarization]
                            if make_corrected
                            else None
                        ),
                        primary_beam_limit=primary_beam_limit,
                    )
                    continue
                # Corrected restoration must precede an apparent restoration
                # that consumes the model buffer.
                if make_corrected:
                    start = time.time()
                    beam_polarization = (
                        polarization_index if primary_beam.shape[2] == npol else 0
                    )
                    primary_beam_corrected_plane(
                        primary_beam[time_index, plane_index, beam_polarization],
                        primary_beam_limit,
                        model_plane,
                        residual_plane,
                        kernel_ft=kernel_ft,
                        workers=workers,
                        out=corrected[time_index, plane_index, polarization_index],
                    )
                    corrected_seconds += time.time() - start

                if make_apparent:
                    start = time.time()
                    apparent_plane = apparent[
                        time_index, plane_index, polarization_index
                    ]
                    if kernel_ft is None or not model_plane.any():
                        apparent_plane[...] = residual_plane
                    else:
                        apparent_plane[...] = _convolve_with_clean_beam(
                            model_plane, kernel_ft, workers
                        )
                        apparent_plane += residual_plane
                    apparent_seconds += time.time() - start

    total_seconds = time.time() - total_start
    if make_apparent:
        # Match the historical timing split: shared setup (notably the clean
        # beam FFT) belongs to restoration, while corrected plane work is
        # reported separately.
        apparent_seconds = total_seconds - corrected_seconds
    elif make_corrected:
        corrected_seconds = total_seconds

    products = {}
    if make_apparent:
        products["apparent"] = apparent
    if make_corrected:
        products["primary_beam_corrected"] = corrected
    return products, {
        "apparent": apparent_seconds,
        "primary_beam_corrected": corrected_seconds,
        "consumed_model": consume,
    }


def _restored_beam_variable(beam_da, restoring_beam):
    """``BEAM_FIT_PARAMS_SKY_RESTORED``: the target beam in every plane of the
    PSF beam variable ``beam_da``, whose last dimension holds the parameters."""
    restored = beam_da.copy(
        data=np.broadcast_to(
            np.asarray(restoring_beam, dtype=np.float64), beam_da.shape
        ).copy()
    )
    restored.attrs = {"type": "beam_fit_params_sky", "units": "rad"}
    return restored


@shares_param_docs
def restore_image(
    img_xds: xr.Dataset,
    image_data_group_in_residual_name: str = "residual",
    image_data_group_in_model_name: str = "model",
    image_data_group_out_restore_name: str = "restored",
    image_data_group_out_modified: dict | None = None,
    beam_fit_params_key: str = "beam_fit_params_point_spread_function",
    beam_polarization_index: int = 0,
    processing_function_threads: int = 1,
    consume_model: bool = False,
    restoration_output: str = "apparent",
    restoring_beam: list[float] | None = None,
    primary_beam_limit: float = 0.2,
    primary_beam_key: str = "primary_beam",
    overwrite: bool = True,
    *,
    input_checked: bool = False,
):
    """Restore an image: model convolved with the clean beam plus the residual.

    For every frequency a 2-D elliptical Gaussian "clean beam" is built on the
    ``(l, m)`` plane from the ``[major, minor, pa]`` beam-fit parameters stored
    in the **residual** data group (the Gaussian fit to the point spread
    function; see the `CASA synthesized-beam definition
    <https://casadocs.readthedocs.io/en/stable/notebooks/casa-fundamentals.html#Definition-Synthesized-Beam>`_).
    The same beam is used for both polarizations of that frequency.  The beam is
    convolved (via FFT) with the sky in the **model** data group, and the sky in
    the **restored** output data group is that convolved model plus the sky in
    the residual data group::

        SKY_RESTORED = (clean_beam * SKY_MODEL) + SKY_RESIDUAL

    The clean beam is normalised to unit peak, so a model point source of flux
    ``F`` (Jy) becomes a Gaussian of peak ``F`` Jy/beam.

    With ``restoration_output="primary_beam_corrected"`` only the corrected
    image is made; ``"both"`` also retains the apparent image. The corrected
    image is stored as
    ``SKY_RESTORED_PRIMARY_BEAM_CORRECTED`` (role ``sky_primary_beam_corrected``
    of the restored group): the model and the residual are divided by the
    primary beam before the convolution, ``(SKY_MODEL / P) * B + SKY_RESIDUAL
    / P``, which is exact for the model part (see
    :func:`primary_beam_corrected_plane`; CASA's ``pbcor`` divides the
    restored image instead).

    Corrected-only output does not allocate an apparent restored cube and
    leaves the input model intact even when ``consume_model=True``.

    Efficiency
    ----------
    The work is done plane by plane so no full-cube temporaries are allocated
    beyond the requested restored cubes.  The convolution is a real FFT
    (``scipy.fft.rfft2`` / ``irfft2``, the inverse done in two in-place steps
    that give the same result without scipy's hidden copy of the spectrum,
    see ``_inverse_rfft2``) at the image dtype (single-precision
    images stay single precision), and the clean beam's FFT is computed once per
    frequency and reused for every polarization (the beam plane itself is freed
    as soon as it is transformed).  Per polarization at most two extra planes
    are live at once -- the model spectrum, multiplied by the beam in place,
    and the inverse-transform output, with the residual added into the restored
    cube without a further temporary.  Planes whose model is entirely zero skip
    the convolution (the restored plane is just the residual).  With
    ``consume_model=True`` and apparent output, its cube allocation is avoided: the
    model cube's buffer is reused and the model variable dropped.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset with dims ``(time, frequency, polarization, l, m)``
        containing the residual and model data groups (and the residual group's
        beam-fit parameters).  Modified in place: the restored sky variable is
        added and ``attrs["data_groups"]`` gains the restored group.
    image_data_group_in_residual_name : str, optional
        Key of the residual input data group.  Supplies the residual sky
        (``"sky"`` role) and the clean-beam fit (``beam_fit_params_key`` role).
        Default ``"residual"``.
    image_data_group_in_model_name : str, optional
        Key of the model input data group.  Supplies the model sky (``"sky"``
        role) that is convolved with the clean beam.  Default ``"model"``.
    image_data_group_out_restore_name : str, optional
        Key under which the restored output data group is registered.  Default
        ``"restored"``.
    image_data_group_out_modified : dict, optional
        Role overrides layered on top of the residual group to form the restored
        group.  Default ``{"sky": "SKY_RESTORED"}`` so the restored sky is stored
        in the ``SKY_RESTORED`` data variable.
    beam_fit_params_key : str, optional
        Role key in the residual data group holding the ``[major, minor, pa]``
        beam-fit parameters (FWHM major/minor and position angle, in radians),
        with dims ``(time, frequency, polarization, beam_params)``.  Default
        ``"beam_fit_params_point_spread_function"``.
    beam_polarization_index : int, optional
        Polarization index of the beam-fit parameters used to build the clean
        beam (the same beam is applied to every polarization).  Default ``0``
        (Stokes I).
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    consume_model : bool, optional
        If ``True`` and apparent output is requested, that cube is written
        into the **model** variable's
        buffer instead of a freshly allocated cube (each model plane is fully
        read into its forward FFT before its slot is overwritten), and the
        model data variable is removed from ``img_xds`` afterwards — the model
        is destroyed.  Only set this when the model is not needed after the
        restore (e.g. it is not written to the output store).  Saves one full
        image cube of peak memory.  Requires the model and residual dtypes to
        match; otherwise a fresh cube is allocated as for ``False``.  Default
        ``False`` (model preserved).
    restoration_output : {"apparent", "primary_beam_corrected", "both"}, optional
        Restored products to compute: ``"apparent"``, ``"primary_beam_corrected"``,
        or ``"both"``. Cube outputs are ``SKY_RESTORED`` and
        ``SKY_RESTORED_PRIMARY_BEAM_CORRECTED``; continuum outputs are
        ``SKY_RESTORED_FLAT_NOISE`` and ``SKY_RESTORED_FLAT_SKY``.
        Cube corrects the model and residual by the channel power primary beam
        before convolution. MFS divides their Taylor polynomials; MVC divides
        by its common power primary beam. Corrected pixels below
        ``primary_beam_limit`` are blanked with NaN. Imaging workflows use this
        selector only when ``restore=True``.
    restoring_beam : list of float, optional
        ``None`` restores each plane with its own PSF fit [CASA ``tclean``
        ``restoringbeam=''``]. ``[major, minor, pa]`` (FWHM and position angle
        in radians, as ``BEAM_FIT_PARAMS``; ``major >= minor > 0``) restores
        every plane to that beam as ``tclean`` does since CASA 5.6 (CAS-12148):
        the model is convolved with the unit-peak target beam, and the residual
        with the Gaussian that takes the plane's PSF beam to the target, times
        the ratio of the beam areas, so it is in Jy per target beam.
        ``SKY_RESIDUAL`` is unchanged. A plane whose PSF beam the target does
        not contain, or that has no PSF fit, keeps its residual unconvolved and
        unscaled, as ``tclean`` does, and is named in a warning. The continuum
        imager refuses a target that does not contain its PSF beam before its
        imaging cycles. The beam is recorded as ``BEAM_FIT_PARAMS_SKY_RESTORED``.
        For ``tclean``'s ``restoringbeam='common'``, see
        :func:`~astroviper.processing_functions.image_analysis.common_beam.common_beam`.
    primary_beam_limit : float, optional
        Primary beam (power) cutoff below which the corrected image is blanked
        with NaN, as a fraction of the beam peak.  Default ``0.2`` (the CASA
        ``pblimit`` default).
    primary_beam_key : str, optional
        Role key of the primary beam in the residual data group.  Default
        ``"primary_beam"``.
    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.
    overwrite : bool, optional
        If ``True`` an existing restored data group / output variable is
        overwritten.  Default ``True``.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with the restored sky variable added and the restored
        data group registered. With ``restoring_beam`` it also holds
        ``BEAM_FIT_PARAMS_SKY_RESTORED`` (the target in every plane, role
        ``beam_fit_params_sky`` of the restored group).
    return_df : pandas.DataFrame
        One-row timing frame with a ``T_restore`` column (wall-clock seconds of
        the clean-beam convolution), matching the other imaging processing
        functions, and a ``T_correct_sky_by_primary_beam`` column when a
        primary beam corrected image is made.

    Notes
    -----
    A square pixel grid (``|delta_l| == |delta_m|``) is assumed, as in the PSF
    Gaussian fit.  Frequencies whose beam-fit parameters are non-finite or
    non-positive have no defined clean beam; their restored plane falls back to
    the residual (the model cannot be restored without a beam).

    With ``restoring_beam`` the residual is multiplied in the uv plane by
    ``H = (Omega_T / Omega_p) exp(-2 pi^2 f^T D f)``, with ``D`` the covariance
    of the target minus that of the PSF beam (CASA Memo 10 in covariance form),
    which also holds for equal and touching beams. The corrected image is
    ``(SKY_MODEL / P) * B_T + (SKY_RESIDUAL * K) / P``. Non-finite residual
    pixels are zero for the transform and NaN in the restored images.
    """
    import pandas as pd

    if not input_checked:
        from astroviper.processing_functions.imaging.config_check._predicates import (
            check_primary_beam_limit,
            check_restoration_output,
            check_restoring_beam,
        )

        check_restoration_output(restoration_output)
        check_primary_beam_limit(primary_beam_limit)
        if restoring_beam is not None:
            check_restoring_beam(restoring_beam)

    make_apparent = restoration_output in ("apparent", "both")
    correct = restoration_output in ("primary_beam_corrected", "both")
    if image_data_group_out_modified is None:
        image_data_group_out_modified = {
            "sky": "SKY_RESTORED"
            if make_apparent
            else "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"
        }

    # Resolve every required runtime product before building output metadata.
    # A cube consumes the fitted beam directly and does not require a PSF.
    residual_sky_name = resolve_restoration_product(
        img_xds, image_data_group_in_residual_name, "sky"
    )
    model_sky_name = resolve_restoration_product(
        img_xds, image_data_group_in_model_name, "sky"
    )
    beam_name = resolve_restoration_product(
        img_xds, image_data_group_in_residual_name, beam_fit_params_key
    )
    if correct:
        primary_beam_name = resolve_restoration_product(
            img_xds, image_data_group_in_residual_name, primary_beam_key
        )

    # The restored output inherits the residual roles and replaces only sky.
    _, restored_group = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_residual_name,
        data_group_out_name=image_data_group_out_restore_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=overwrite,
    )
    restored_sky_name = restored_group["sky"]
    if correct:
        primary_beam = img_xds[primary_beam_name].values
        corrected_sky_name = (
            "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"
            if make_apparent
            else restored_sky_name
        )
        restored_group["sky_primary_beam_corrected"] = corrected_sky_name

    residual_da = img_xds[residual_sky_name]
    # ``.values`` are views into the dataset; only written to when the model
    # buffer is consumed as the restored cube (``consume_model``).
    residual = residual_da.values
    model = img_xds[model_sky_name].values
    # Beam-fit parameters: (time, frequency, polarization, beam_params)
    # = [major, minor, pa] FWHM/angle in radians.
    beam = img_xds[beam_name].values

    # Pixel size in radians (square pixels assumed, as in the PSF fit).
    l = img_xds["l"].values
    delta = abs(float(l[1] - l[0]))
    products, array_timing = _restore_image_arrays(
        residual,
        model,
        beam[:, :, beam_polarization_index : beam_polarization_index + 1].repeat(
            residual.shape[2], axis=2
        ),
        delta,
        processing_function_threads=processing_function_threads,
        primary_beam=primary_beam if correct else None,
        primary_beam_limit=primary_beam_limit,
        output=restoration_output,
        consume_model=(consume_model and model_sky_name != residual_sky_name),
        restoring_beam=restoring_beam,
    )
    consume = bool(array_timing["consumed_model"])
    if correct:
        corrected = products["primary_beam_corrected"]
        T_correct = float(array_timing["primary_beam_corrected"])

    # Store the restored sky, preserving the residual's dims, coords and attrs.
    if make_apparent:
        img_xds[restored_sky_name] = residual_da.copy(data=products["apparent"])
    if correct:
        img_xds[corrected_sky_name] = residual_da.copy(data=corrected)
        img_xds[corrected_sky_name].attrs["type"] = "sky"

    if consume and model_sky_name != restored_sky_name:
        # The model buffer now lives on as the restored sky; drop the stale
        # model data variable so nothing reads the overwritten planes.
        del img_xds[model_sky_name]

    if restoring_beam is None:
        description = "Restored image: clean-beam-convolved model plus residual."
    else:
        img_xds["BEAM_FIT_PARAMS_SKY_RESTORED"] = _restored_beam_variable(
            img_xds[beam_name], restoring_beam
        )
        restored_group["beam_fit_params_sky"] = "BEAM_FIT_PARAMS_SKY_RESTORED"
        description = (
            "Restored image: model convolved with the restoring beam "
            f"{list(map(float, restoring_beam))} rad plus residual taken to it."
        )
    if correct:
        description += (
            " Primary beam corrected restored image "
            f"(primary_beam_limit {primary_beam_limit})."
        )
    modify_data_groups_xds(
        img_xds,
        image_data_group_out_restore_name,
        restored_group,
        description=description,
    )

    timing = {"T_restore": [float(array_timing["apparent"])]}
    if correct:
        timing["T_correct_sky_by_primary_beam"] = [T_correct]
    return_df = pd.DataFrame(timing)

    return img_xds, return_df
