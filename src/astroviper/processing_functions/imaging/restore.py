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

    Efficiency
    ----------
    The work is done plane by plane so no full-cube temporaries are allocated
    beyond the single restored cube.  The convolution is a real FFT
    (``scipy.fft.rfft2`` / ``irfft2``) at the image dtype (single-precision
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
        functions.

    Notes
    -----
    A square pixel grid (``|delta_l| == |delta_m|``) is assumed, as in the PSF
    Gaussian fit.  Frequencies whose beam-fit parameters are non-finite or
    non-positive have no defined clean beam; their restored plane falls back to
    the residual (the model cannot be restored without a beam).
    """
    import time

    import pandas as pd

    if image_data_group_out_modified is None:
        image_data_group_out_modified = {"sky": "SKY_RESTORED"}

    start = time.time()

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

    residual_da = img_xds[residual_sky_name]
    # ``.values`` are views into the dataset; only written to when the model
    # buffer is consumed as the restored cube (``consume_model``).
    residual = residual_da.values
    model = img_xds[model_sky_name].values
    # Beam-fit parameters: (time, frequency, polarization, beam_params)
    # = [major, minor, pa] FWHM/angle in radians.
    beam = img_xds[beam_name].values

    nt, nf, npol, ny, nx = residual.shape

    # Pixel size in radians (square pixels assumed, as in the PSF fit).
    l = img_xds["l"].values
    delta = abs(float(l[1] - l[0]))

    # scipy.fft uses one worker by default; <= 0 means "all cores".
    workers = (
        processing_function_threads
        if (processing_function_threads and processing_function_threads > 0)
        else -1
    )

    # Output cube, filled plane by plane below. With ``consume_model`` the
    # model cube's buffer is reused (safe: every model plane is fully read
    # into its forward FFT, or found empty, before its slot is overwritten);
    # otherwise a single fresh cube is allocated.
    consume = (
        consume_model
        and model.dtype == residual.dtype
        and model_sky_name != residual_sky_name
    )
    restored = model if consume else np.empty_like(residual)

    for tt in range(nt):
        for ff in range(nf):
            major, minor, pa = (float(x) for x in beam[tt, ff, beam_polarization_index])

            # No valid clean beam -> the model cannot be restored; fall back to
            # the residual for every polarization of this frequency.
            if not np.isfinite([major, minor, pa]).all() or major <= 0 or minor <= 0:
                restored[tt, ff] = residual[tt, ff]
                continue

            kernel = _elliptical_gaussian_kernel(
                ny, nx, major / delta, minor / delta, pa, residual.dtype
            )
            # FFT of the centred clean beam (centre shifted to the origin),
            # computed once and reused for every polarization. Only the
            # transform is used below, so the kernel plane is dropped now.
            kernel_ft = scipy.fft.rfft2(scipy.fft.ifftshift(kernel), workers=workers)
            del kernel

            for pp in range(npol):
                model_plane = model[tt, ff, pp]
                if not model_plane.any():
                    # Nothing cleaned in this plane: restored == residual.
                    restored[tt, ff, pp] = residual[tt, ff, pp]
                    continue
                # At most two extra planes are live at any point: the model
                # spectrum (beam applied in place) and the irfft2 output; the
                # residual is added into the restored cube without a temporary.
                model_ft = scipy.fft.rfft2(model_plane, workers=workers)
                np.multiply(model_ft, kernel_ft, out=model_ft)
                convolved_model = scipy.fft.irfft2(
                    model_ft, s=(ny, nx), workers=workers
                )
                del model_ft
                restored[tt, ff, pp] = convolved_model
                del convolved_model
                restored[tt, ff, pp] += residual[tt, ff, pp]

    # Store the restored sky, preserving the residual's dims, coords and attrs.
    img_xds[restored_sky_name] = residual_da.copy(data=restored)

    if consume and model_sky_name != restored_sky_name:
        # The model buffer now lives on as the restored sky; drop the stale
        # model data variable so nothing reads the overwritten planes.
        del img_xds[model_sky_name]

    modify_data_groups_xds(
        img_xds,
        image_data_group_out_restore_name,
        restored_group,
        description="Restored image: clean-beam-convolved model plus residual.",
    )

    return_df = pd.DataFrame({"T_restore": [time.time() - start]})

    return img_xds, return_df
