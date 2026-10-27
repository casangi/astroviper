import numpy as np
import scipy.fft
import xarray as xr

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)

# FWHM = 2 * sqrt(2 * ln 2) * sigma. Matches the factor used by the PSF Gaussian
# fit (astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit)
# so a beam built here round-trips back to the same [major, minor, pa].
FWHM_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))


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
    zero there before the convolution.

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
    inside = primary_beam_plane >= primary_beam_limit
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

    Pixels whose zeroth-order beam is non-finite or below ``limit`` receive
    ``outside_value`` in every output coefficient. Restored residuals use NaN;
    models use zero so masked model pixels do not contaminate the clean-beam
    FFT.
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

    apparent_values = np.asarray(apparent.values)
    beam_values = np.asarray(primary_beam.values)
    intrinsic = np.empty_like(apparent_values)
    beam_zero = beam_values[:, 0]
    supported = np.isfinite(beam_zero) & (beam_zero >= float(limit))
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
):
    """Restore generic image planes without assuming an axis coordinate name.

    Cube and continuum restoration both have arrays shaped as ``(time, plane,
    polarization, l, m)``; only the meaning of ``plane`` differs (frequency or
    Taylor term). This core owns their common clean-beam convolution, safe
    primary-beam division, cutoff masking, and output allocation.

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

    for time_index in range(nt):
        for plane_index in range(nplane):
            kernel_ft = _clean_beam_kernel_ft(
                beam[time_index, plane_index, 0],
                ny,
                nx,
                pixel_size,
                residual.dtype,
                workers,
            )
            for polarization_index in range(npol):
                model_plane = model[time_index, plane_index, polarization_index]
                residual_plane = residual[time_index, plane_index, polarization_index]
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
    primary_beam_correction: bool = False,
    primary_beam_limit: float = 0.2,
    primary_beam_key: str = "primary_beam",
    overwrite: bool = True,
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

    With ``primary_beam_correction`` the primary beam corrected restored image
    is made in the same pass and stored as
    ``SKY_RESTORED_PRIMARY_BEAM_CORRECTED`` (role ``sky_primary_beam_corrected``
    of the restored group): the model and the residual are divided by the
    primary beam before the convolution, ``(SKY_MODEL / P) * B + SKY_RESIDUAL
    / P``, which is exact for the model part (see
    :func:`primary_beam_corrected_plane`; CASA's ``pbcor`` divides the
    restored image instead).

    Efficiency
    ----------
    The work is done plane by plane so no full-cube temporaries are allocated
    beyond the single restored cube.  The convolution is a real FFT
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
    ``consume_model=True`` even the restored cube allocation is avoided: the
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
        Number of worker threads handed to ``scipy.fft`` for the per-plane
        FFTs.  Values ``<= 0`` use all available cores.  Default ``1``.
    consume_model : bool, optional
        If ``True`` the restored cube is written into the **model** variable's
        buffer instead of a freshly allocated cube (each model plane is fully
        read into its forward FFT before its slot is overwritten), and the
        model data variable is removed from ``img_xds`` afterwards — the model
        is destroyed.  Only set this when the model is not needed after the
        restore (e.g. it is not written to the output store).  Saves one full
        image cube of peak memory.  Requires the model and residual dtypes to
        match; otherwise a fresh cube is allocated as for ``False``.  Default
        ``False`` (model preserved).
    primary_beam_correction : bool, optional
        If ``True`` also make the primary beam corrected restored image
        (needs the ``primary_beam_key`` role in the residual data group).
        Default ``False``.
    primary_beam_limit : float, optional
        Primary beam (power) cutoff below which the corrected image is blanked
        with NaN, as a fraction of the beam peak.  Default ``0.2`` (the CASA
        ``pblimit`` default).
    primary_beam_key : str, optional
        Role key of the primary beam in the residual data group.  Default
        ``"primary_beam"``.
    overwrite : bool, optional
        If ``True`` an existing restored data group / output variable is
        overwritten.  Default ``True``.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with the restored sky variable added and the restored
        data group registered.
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
    """
    import pandas as pd

    if image_data_group_out_modified is None:
        image_data_group_out_modified = {"sky": "SKY_RESTORED"}

    # Resolve the residual group as the "input" and build the restored output
    # group from it (so the restored group inherits the residual roles and just
    # overrides the sky variable).
    residual_group, restored_group = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_residual_name,
        data_group_out_name=image_data_group_out_restore_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=overwrite,
    )

    # The model is the second input group; resolve it directly.
    assert image_data_group_in_model_name in img_xds.attrs["data_groups"], (
        "Model data group "
        + image_data_group_in_model_name
        + " not found in img_xds data_groups: "
        + str(list(img_xds.attrs["data_groups"].keys()))
    )
    model_group = img_xds.attrs["data_groups"][image_data_group_in_model_name]

    assert beam_fit_params_key in residual_group, (
        "Beam-fit parameters '"
        + beam_fit_params_key
        + "' not found in the residual data group '"
        + image_data_group_in_residual_name
        + "'. Run point_spread_function_gaussian_fit first."
    )

    residual_sky_name = residual_group["sky"]
    model_sky_name = model_group["sky"]
    beam_name = residual_group[beam_fit_params_key]
    restored_sky_name = restored_group["sky"]

    correct = bool(primary_beam_correction)
    if correct:
        assert primary_beam_key in residual_group, (
            "Data group '"
            + image_data_group_in_residual_name
            + "' has no "
            + primary_beam_key
            + " entry; run make_primary_beam_single_field first."
        )
        primary_beam = img_xds[residual_group[primary_beam_key]].values
        corrected_sky_name = "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"
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
        output="both" if correct else "apparent",
        consume_model=(consume_model and model_sky_name != residual_sky_name),
    )
    restored = products["apparent"]
    consume = bool(array_timing["consumed_model"])
    if correct:
        corrected = products["primary_beam_corrected"]
        T_correct = float(array_timing["primary_beam_corrected"])

    # Store the restored sky, preserving the residual's dims, coords and attrs.
    img_xds[restored_sky_name] = residual_da.copy(data=restored)
    if correct:
        img_xds[corrected_sky_name] = residual_da.copy(data=corrected)
        img_xds[corrected_sky_name].attrs["type"] = "sky"

    if consume and model_sky_name != restored_sky_name:
        # The model buffer now lives on as the restored sky; drop the stale
        # model data variable so nothing reads the overwritten planes.
        del img_xds[model_sky_name]

    description = "Restored image: clean-beam-convolved model plus residual."
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
