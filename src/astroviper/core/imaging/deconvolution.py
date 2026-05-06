import numpy as np
import xarray as xr
import copy
from typing import Optional, Tuple

from astroviper.core.imaging.deconvolvers import hogbom
from astroviper.core.image_analysis import image_statistics as imgstats
from astroviper.core.image_analysis.point_spread_function_gaussian_fit import extract_main_lobe
from astroviper.core.imaging.utils.return_dict import ReturnDict

import logging
import toolviper.utils.logger as logger

# lg = logger.get_logger()
# lg.setLevel(logging.DEBUG)

# XXX : TODO: As of 2025-10-07 there is no way to supply an initial model image to the deconvolver


from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)


def get_phase_center(residual_image_xds):
    """
    Get the phase center from the residual image coordinates.

    Parameters
    ----------
    residual_image_xds : xarray.Dataset
        Residual image dataset with 2-D ``right_ascension`` and
        ``declination`` coordinate arrays.

    Returns
    -------
    phase_center : str
        The phase center in the format ``"RA,Dec"`` (e.g.
        ``"12.345,-67.890"``), taken from the central pixel of the
        coordinate grids.
    """
    ra_shape = residual_image_xds.coords["right_ascension"].shape
    dec_shape = residual_image_xds.coords["declination"].shape

    cx_ra, cy_ra = ra_shape[0] // 2, ra_shape[1] // 2
    cx_dec, cy_dec = dec_shape[0] // 2, dec_shape[1] // 2

    ra0 = residual_image_xds.coords["right_ascension"].values[cx_ra, cy_ra]
    dec0 = residual_image_xds.coords["declination"].values[cx_dec, cy_dec]
    return f"{ra0},{dec0}"


def progress_callback(
    iter_num: int,
    px: int,
    py: int,
    peak: float,
    niter_log: int = 100,
):
    """
    Log CLEAN progress at a fixed iteration cadence.

    Parameters
    ----------
    iter_num : int
        Current iteration number.
    px : int
        X-coordinate of the current peak.
    py : int
        Y-coordinate of the current peak.
    peak : float
        Value of the current peak.
    niter_log : int, optional
        Frequency (in iterations) at which to emit log messages.
        Default is 100.
    """
    if iter_num % niter_log == 0:
        logger.info(f"  Iteration {iter_num}, peak at ({px}, {py}): {peak:.6f}")


def _validate_deconvolve_params(deconvolve_params):
    """
    Validate a deconvolution parameter dict and fill in defaults.

    Parameters
    ----------
    deconvolve_params : dict or None
        Parameter dict; missing keys are populated with defaults. If
        ``None``, a fresh dict of defaults is returned.

        Supported keys

        - ``gain`` : float, CLEAN loop gain in ``(0, 1]``. Default 0.1.
        - ``niter`` : int, maximum number of iterations. Default 1000.
        - ``threshold`` : float, stopping threshold, non-negative.
          Default 0.0.
        - ``clean_box`` : 4-tuple ``(xmin, xmax, ymin, ymax)``. Default
          ``(-1, -1, -1, -1)`` meaning the full image.

    Returns
    -------
    dict
        The validated parameter dict, with defaults applied.

    Raises
    ------
    ValueError
        If any supplied value is outside its allowed range.
    """
    # NOTE : XXX : This should probably not live here. This validation
    # function should live in utils or something.

    if deconvolve_params is None:
        deconvolve_params = {}

    default_params = {
        "gain": 0.1,
        "niter": 1000,
        "threshold": 0.0,
        "clean_box": (-1, -1, -1, -1),
    }

    for key, default_value in default_params.items():
        if key not in deconvolve_params:
            logger.info(
                f"Deconvolution parameter '{key}' not specified. Using default: {default_value}"
            )
            deconvolve_params[key] = default_value
            continue

        value = deconvolve_params[key]
        if key == "gain":
            if not (0 < value <= 1):
                raise ValueError("CLEAN gain must be between 0 and 1.")
        elif key == "niter":
            if not (isinstance(value, int) and value > 0):
                raise ValueError(
                    "Maximum number of iterations must be a positive integer."
                )
        elif key == "threshold":
            if value is not None and value < 0:
                raise ValueError("Threshold must be non-negative or None.")
        elif key == "clean_box":
            if value is not None and not (
                isinstance(value, tuple) and len(value) == 4
            ):
                raise ValueError(
                    "Clean box must be a 4-tuple (xmin, xmax, ymin, ymax) or None."
                )

    return deconvolve_params


def _plane_peak_abs_signed(arr, mask=None):
    """
    Return the signed value at the absolute-value peak of a 2-D array,
    optionally restricted to pixels where ``mask > 0.5``.

    Parameters
    ----------
    arr : numpy.ndarray
        2-D image plane.
    mask : numpy.ndarray, optional
        2-D mask of the same shape as ``arr``. If provided, only pixels
        where ``mask > 0.5`` are considered.

    Returns
    -------
    float
        Signed value of ``arr`` at its absolute-value maximum within the
        mask. NaN if every pixel is masked.
    """
    if mask is None:
        idx = np.unravel_index(np.abs(arr).argmax(), arr.shape)
        return float(arr[idx])
    valid = mask > 0.5
    if not np.any(valid):
        return float("nan")
    absvals = np.where(valid, np.abs(arr), -np.inf)
    idx = np.unravel_index(absvals.argmax(), arr.shape)
    return float(arr[idx])


def deconvolve(
    img_xds: xr.Dataset = None,
    algorithm: str = "hogbom",
    deconvolve_params: Optional[dict] = None,
    num_threads: int = 1,
    image_data_group_in_name: str = "residual",
    image_data_group_out_name: str = "model",
    image_data_group_out_modified: Optional[dict] = None,
):
    """
    Deconvolve a residual image cube, in place, with planes parallelized
    in C++.

    The residual and model image variables of ``img_xds`` are mutated in
    place: the residual has CLEAN components subtracted from it, and the
    model has the same components added. The per-plane ``(time,
    frequency, polarization)`` loop is delegated to the C++ binding,
    which dispatches planes across ``num_threads`` ``std::thread``
    workers. No per-plane copies of the ``(ny, nx)`` arrays are retained
    on the Python side; the full ``(nt, nf, np, ny, nx)`` numpy buffers
    are handed directly to the CLEAN routine.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset containing the residual, PSF, and (optionally)
        model variables with dimensions
        ``(time, frequency, polarization, y, x)``. The residual and
        model variables are updated in place.
    algorithm : str, optional
        Deconvolution algorithm to use. Only ``"hogbom"`` is currently
        supported.
    deconvolve_params : dict, optional
        Algorithm-specific parameters. See
        :func:`_validate_deconvolve_params` for supported keys and
        defaults.
    num_threads : int, optional
        Number of ``std::thread`` workers to use for per-plane CLEAN
        execution. Values ``<= 1`` disable threading and run in the
        calling thread. Values larger than the number of planes are
        clamped to the plane count. Default 1.
    image_data_group_in_name : str, optional
        Name of the input data group in ``img_xds`` that provides the
        residual sky image and PSF. Default is ``"residual"``.
    image_data_group_out_name : str, optional
        Name of the output data group to create or update for the model
        image. Default is ``"model"``.
    image_data_group_out_modified : dict, optional
        Mapping from logical names to data-variable names for the output
        data group. Defaults to ``{"sky": "SKY_MODEL"}``.

    Returns
    -------
    returndict : ReturnDict
        Per-plane deconvolution statistics, indexed by
        ``(time, chan, pol)``. Each entry contains iteration count, peak
        residuals before and after, model fluxes, and PSF bookkeeping.
    img_xds : xarray.Dataset
        The same dataset passed in, with residual and model variables
        updated in place.

    Raises
    ------
    ValueError
        If ``algorithm`` is not recognized, or if the PSF polarization
        dimension is incompatible with the image polarization dimension.

    Notes
    -----
    - The PSF may be single-polarization (Stokes I only), in which case
      it is broadcast across all image polarizations.
    - Deconvolution is performed independently on each
      ``(time, frequency, polarization)`` plane; parallelization across
      planes happens inside the C++ binding.
    - The underlying CLEAN C++ binding operates directly on the
      Python-owned numpy buffers: the full ``(nt, nf, np, ny, nx)``
      residual and model cubes are the only per-plane allocations,
      and they are updated in place.
    """
    if algorithm.lower() != "hogbom":
        raise ValueError(f"Deconvolution algorithm '{algorithm}' not recognized.")

    if image_data_group_out_modified is None:
        image_data_group_out_modified = {"sky": "SKY_MODEL"}

    data_group_in, data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=True,
    )

    residual_name = data_group_in["sky"]
    psf_name = data_group_in["point_spread_function"]
    model_name = data_group_out["sky"]

    if model_name not in img_xds.data_vars:
        img_xds[model_name] = xr.zeros_like(img_xds[residual_name])
        zero_model = True
        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            data_group_out,
            description="Created model.",
        )
    else:
        zero_model = False

    ntime = img_xds.sizes["time"]
    nchan = img_xds.sizes["frequency"]
    npol = img_xds.sizes["polarization"]
    npol_psf = img_xds[psf_name].sizes["polarization"]

    broadcast_psf = npol_psf == 1 and npol > 1
    if broadcast_psf:
        logger.info(f"PSF is single-plane, will broadcast to {npol} polarizations")
    elif npol_psf != npol:
        raise ValueError(
            "PSF should have same number of polarizations as the input image, "
            "or be Stokes I only. "
            f"(npol_psf = {npol_psf}, npol_image = {npol})"
        )

    masksum = imgstats.get_image_masksum(img_xds, dv=residual_name)

    deconvolve_params = _validate_deconvolve_params(deconvolve_params)

    max_psf_fraction = 0.8
    min_psf_fraction = 0.1
    max_psf_sidelobe = None  # TODO: compute per (time, freq) via extract_main_lobe

    residual_arr = img_xds[residual_name].values
    psf_arr = img_xds[psf_name].values
    model_arr = img_xds[model_name].values

    # Optional mask cube, matching the residual's shape. The C++ binding
    # accepts a bool mask directly; Python keeps ownership of the buffer.
    mask_name = img_xds.attrs["data_groups"][image_data_group_in_name].get("mask", None)
    mask_arr = None
    if mask_name in img_xds:
        mask_arr = img_xds[mask_name].values

    # Collect per-plane starting statistics in-place (no copies, just
    # reads). The actual CLEAN iteration is driven in C++.
    start_peakres = np.empty((ntime, nchan, npol), dtype=np.float64)
    start_peakres_nomask = np.empty((ntime, nchan, npol), dtype=np.float64)
    start_model_flux = np.empty((ntime, nchan, npol), dtype=np.float64)
    for tt in range(ntime):
        for nn in range(nchan):
            for pp in range(npol):
                rp = residual_arr[tt, nn, pp]
                mp = mask_arr[tt, nn, pp] if mask_arr is not None else None
                start_peakres[tt, nn, pp] = _plane_peak_abs_signed(rp, mask=mp)
                start_peakres_nomask[tt, nn, pp] = _plane_peak_abs_signed(rp)
                start_model_flux[tt, nn, pp] = (
                    0.0 if zero_model else float(model_arr[tt, nn, pp].sum())
                )

    # Drive the CLEAN loop in C++. The helper owns the full
    # (time, frequency, polarization) iteration and the parallel worker
    # pool.
    results = hogbom_clean(
        residual_cube=residual_arr,
        psf_cube=psf_arr,
        model_cube=model_arr,
        deconvolve_params=deconvolve_params,
        num_threads=num_threads,
        mask_cube=mask_arr,
    )

    iters = np.asarray(results["iterations_performed"])
    final_peaks = np.asarray(results["final_peak"])

    returndict = ReturnDict()
    pol_vals = img_xds.coords["polarization"].values
    freq_vals = img_xds.coords["frequency"].values
    time_vals = img_xds.coords["time"].values

    for tt in range(ntime):
        for nn in range(nchan):
            for pp in range(npol):
                rp = residual_arr[tt, nn, pp]
                mp = mask_arr[tt, nn, pp] if mask_arr is not None else None
                peakres = _plane_peak_abs_signed(rp, mask=mp)
                peakres_nomask = _plane_peak_abs_signed(rp)
                model_flux = float(model_arr[tt, nn, pp].sum())

                returnvals = {
                    "niter": deconvolve_params.get("niter", None),
                    "threshold": deconvolve_params.get("threshold", None),
                    "iter_done": int(iters[tt, nn, pp]),
                    "loop_gain": deconvolve_params.get("gain", None),
                    "min_psf_fraction": min_psf_fraction,
                    "max_psf_fraction": max_psf_fraction,
                    "max_psf_sidelobe": max_psf_sidelobe,
                    "stop_code": None,
                    "stokes": pol_vals[pp],
                    "frequency": freq_vals[nn],
                    "time": time_vals[tt],
                    "start_model_flux": start_model_flux[tt, nn, pp],
                    "model_flux": model_flux,
                    "start_peakres": start_peakres[tt, nn, pp],
                    "start_peakres_nomask": start_peakres_nomask[tt, nn, pp],
                    "peakres": peakres,
                    "peakres_nomask": peakres_nomask,
                    "masksum": masksum,
                }

                returndict.add(returnvals, time=tt, pol=pp, chan=nn)
    return returndict


def hogbom_clean(
    residual_cube: np.ndarray,
    psf_cube: np.ndarray,
    model_cube: np.ndarray,
    deconvolve_params: Optional[dict] = None,
    num_threads: int = 1,
    mask_cube: Optional[np.ndarray] = None,
):
    """
    Run Hogbom CLEAN over an entire ``(time, frequency, polarization,
    y, x)`` image cube, parallelized across planes in C++.

    The residual and model cubes are updated in place: the residual has
    CLEAN components subtracted from it, and the model accumulates the
    same components. The per-plane loop and the ``std::thread`` worker
    pool live entirely in the C++ binding
    (:func:`hogbom.clean_cube`); this Python layer only validates inputs
    and forwards them.

    Parameters
    ----------
    residual_cube : numpy.ndarray
        5-D residual array with shape ``(nt, nf, np, ny, nx)``. Must be
        C-contiguous, writeable, and float32 or float64. Updated in
        place with the post-CLEAN residual.
    psf_cube : numpy.ndarray
        5-D PSF array with shape ``(nt, nf, np_psf, ny, nx)`` where
        ``np_psf`` equals ``np`` or is ``1`` (Stokes-I broadcast across
        all image polarizations). Must be C-contiguous and share the
        dtype of ``residual_cube``. Not modified.
    model_cube : numpy.ndarray
        5-D model array with shape ``(nt, nf, np, ny, nx)``. CLEAN
        components are **added** into this array in place, allowing
        callers to start from a non-zero initial model or accumulate
        across calls. Must be C-contiguous, writeable, and share the
        dtype of ``residual_cube``.
    deconvolve_params : dict, optional
        Algorithm parameters. See :func:`_validate_deconvolve_params`
        for supported keys and defaults.
    num_threads : int, optional
        Number of ``std::thread`` workers used to run per-plane CLEAN in
        parallel. ``num_threads <= 1`` disables threading; values larger
        than the number of planes are clamped to the plane count.
        Default 1.
    mask_cube : numpy.ndarray, optional
        5-D bool mask array with the same shape as ``residual_cube``.
        Must be C-contiguous and of dtype ``bool`` (the C++ binding
        accepts the buffer directly without a copy). Pixels with mask
        value ``True`` are considered in the peak search; others are
        ignored. ``None`` disables masking.

    Returns
    -------
    dict
        Per-plane summary arrays with shape ``(nt, nf, np)`` and keys
        ``iterations_performed`` (int), ``final_peak`` (float),
        ``total_flux_cleaned`` (float), and ``converged`` (bool). The
        final residual and model cubes are the caller-supplied arrays,
        updated in place.

    Raises
    ------
    ValueError
        If the input arrays are not 5-D, do not share a floating-point
        dtype, or have incompatible shapes.

    Notes
    -----
    The 5-D layout is the same one used by ``xarray`` image datasets in
    astroviper, so a plain ``img_xds[name].values`` call produces an
    array that can be passed to this function without any copy.
    """
    deconvolve_params = _validate_deconvolve_params(deconvolve_params)

    for name, arr in (
        ("residual_cube", residual_cube),
        ("psf_cube", psf_cube),
        ("model_cube", model_cube),
    ):
        if not isinstance(arr, np.ndarray) or arr.ndim != 5:
            raise ValueError(
                f"{name} must be a 5D numpy array with shape "
                "(nt, nf, np, ny, nx)"
            )

    if residual_cube.shape != model_cube.shape:
        raise ValueError(
            "residual_cube and model_cube must have same shape. "
            f"Got {residual_cube.shape} and {model_cube.shape}"
        )

    # PSF cube may broadcast on the polarization axis.
    nt, nf, npol_img, ny, nx = residual_cube.shape
    nt_p, nf_p, npol_psf, ny_p, nx_p = psf_cube.shape
    if (nt_p, nf_p, ny_p, nx_p) != (nt, nf, ny, nx):
        raise ValueError(
            "psf_cube must match residual_cube on (time, frequency, y, x). "
            f"Got psf shape {psf_cube.shape} vs residual shape "
            f"{residual_cube.shape}"
        )
    if npol_psf not in (npol_img, 1):
        raise ValueError(
            "psf_cube polarization axis must equal residual_cube "
            "polarization axis or be 1 (Stokes I broadcast). "
            f"Got np_psf={npol_psf}, np_img={npol_img}"
        )

    if mask_cube is not None:
        if mask_cube.shape != residual_cube.shape:
            raise ValueError(
                "mask_cube must have the same shape as residual_cube. "
                f"Got {mask_cube.shape} and {residual_cube.shape}"
            )

    logger.debug(f"Residual cube shape: {residual_cube.shape}")
    logger.debug(f"PSF cube shape: {psf_cube.shape}")
    logger.info("Running Hogbom CLEAN on cube with num_threads=%d" % num_threads)

    clean_box = deconvolve_params["clean_box"]
    if clean_box is None:
        clean_box = (-1, -1, -1, -1)

    mask_arg = (
        mask_cube
        if mask_cube is not None
        else np.array([], dtype=residual_cube.dtype)
    )

    return hogbom.clean_cube(
        residual_cube=residual_cube,
        psf_cube=psf_cube,
        model_cube=model_cube,
        mask_cube=mask_arg,
        clean_box=clean_box,
        max_iter=deconvolve_params["niter"],
        gain=deconvolve_params["gain"],
        threshold=deconvolve_params["threshold"],
        num_threads=int(num_threads),
    )
