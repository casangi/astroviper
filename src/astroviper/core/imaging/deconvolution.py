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


def deconvolve(
    img_xds: xr.Dataset = None,
    algorithm: str = "hogbom",
    deconvolve_params: Optional[dict] = None,
    image_data_group_in_name: str = "residual",
    image_data_group_out_name: str = "model",
    image_data_group_out_modified: Optional[dict] = None,
):
    """
    Deconvolve a residual image cube plane-by-plane, in place.

    The residual and model image variables of ``img_xds`` are mutated in
    place: the residual has CLEAN components subtracted from it, and the
    model has the same components added. No per-plane copies of the
    ``(ny, nx)`` arrays are retained on the Python side; 2-D views of
    the underlying numpy arrays are passed directly to the CLEAN routine.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset containing the residual, PSF, and (optionally) model
        variables with dimensions
        ``(time, frequency, polarization, y, x)``. The residual and model
        variables are updated in place.
    algorithm : str, optional
        Deconvolution algorithm to use. Only ``"hogbom"`` is currently
        supported.
    deconvolve_params : dict, optional
        Algorithm-specific parameters. See
        :func:`_validate_deconvolve_params` for supported keys and
        defaults.
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
      ``(time, frequency, polarization)`` plane.
    - The underlying CLEAN C++ binding operates directly on the
      Python-owned numpy buffers: the full ``(nt, nf, np, ny, nx)``
      residual and model cubes are the only per-plane allocations,
      and they are updated in place.
    """
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

    returndict = ReturnDict()

    # No model provided: allocate once and fill with zeros. We then update
    # this single allocation in place from every plane's CLEAN call.
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
    # phase_center = get_phase_center(img_xds)

    if algorithm.lower() == "hogbom":
        _deconvolver = hogbom_clean
    else:
        raise ValueError(f"Deconvolution algorithm '{algorithm}' not recognized.")

    # Normalize params once so the per-plane call does not re-fill defaults.
    deconvolve_params = _validate_deconvolve_params(deconvolve_params)
    
    max_psf_fraction = 0.8
    min_psf_fraction = 0.1
    max_psf_sidelobe = None  # TODO: compute per (time, freq) via extract_main_lobe

    # Obtain direct views of the underlying numpy buffers. Integer indexing
    # into these views yields 2-D views (no copy), which the CLEAN routine
    # can consume directly and write back to.
    residual_arr = img_xds[residual_name].values
    psf_arr = img_xds[psf_name].values
    model_arr = img_xds[model_name].values

    pol_vals = img_xds.coords["polarization"].values
    freq_vals = img_xds.coords["frequency"].values
    time_vals = img_xds.coords["time"].values

    for tt in range(ntime):
        for nn in range(nchan):
            for pp in range(npol):
                logger.debug(
                    f"Deconvolving time {tt+1}/{ntime}, "
                    f"freq {nn+1}/{nchan}, pol {pp+1}/{npol}"
                )

                pidx = 0 if broadcast_psf else pp

                residual_plane = residual_arr[tt, nn, pp]
                psf_plane = psf_arr[tt, nn, pidx]
                model_plane = model_arr[tt, nn, pp]

                # Per-plane starting statistics. Scalar .isel() returns a
                # 2-D DataArray view, not a copy.
                residual_slice = img_xds.isel(
                    time=tt, frequency=nn, polarization=pp
                )
                start_peakres = imgstats.image_peak_residual(
                    residual_slice, per_plane_stats=False, use_mask=True, dv=residual_name
                )
                start_peakres_nomask = imgstats.image_peak_residual(
                    residual_slice, per_plane_stats=False, use_mask=False, dv=residual_name
                )
                start_model_flux = (
                    0.0 if zero_model else float(model_plane.sum())
                )
                
                # Run CLEAN on the plane views; residual_plane and
                # model_plane are updated in place inside the helper.
                results = _deconvolver(
                    residual_image=residual_plane,
                    psf=psf_plane,
                    model=model_plane,
                    deconvolve_params=deconvolve_params,
                )

                model_flux = float(model_plane.sum())
                peakres = imgstats.image_peak_residual(
                    residual_slice, per_plane_stats=False, use_mask=True
                )
                peakres_nomask = imgstats.image_peak_residual(
                    residual_slice, per_plane_stats=False, use_mask=False
                )

                returnvals = {
                    "niter": deconvolve_params.get("niter", None),
                    "threshold": deconvolve_params.get("threshold", None),
                    "iter_done": results.get("iterations_performed", None),
                    "loop_gain": deconvolve_params.get("gain", None),
                    "min_psf_fraction": min_psf_fraction,
                    "max_psf_fraction": max_psf_fraction,
                    "max_psf_sidelobe": max_psf_sidelobe,
                    "stop_code": None,
                    "stokes": pol_vals[pp],
                    "frequency": freq_vals[nn],
                    #"phase_center": phase_center,
                    "time": time_vals[tt],
                    "start_model_flux": start_model_flux,
                    "model_flux": model_flux,
                    "start_peakres": start_peakres,
                    "start_peakres_nomask": start_peakres_nomask,
                    "peakres": peakres,
                    "peakres_nomask": peakres_nomask,
                    "masksum": masksum,
                }

                returndict.add(returnvals, time=tt, pol=pp, chan=nn)

    return returndict


def hogbom_clean(
    residual_image: np.ndarray,
    psf: np.ndarray,
    model: np.ndarray,
    deconvolve_params: Optional[dict] = None,
):
    """
    Run Hogbom CLEAN on a single 2-D plane, updating the residual and
    model arrays in place.

    The underlying C++ routine writes the residual and model directly
    into the caller-supplied arrays with no copies; both arrays must be
    C-contiguous, writeable, and share a floating-point dtype.

    Parameters
    ----------
    residual_image : numpy.ndarray
        2-D residual image array with shape ``(ny, nx)``. Updated in
        place with the post-CLEAN residual.
    psf : numpy.ndarray
        2-D PSF array with shape ``(ny, nx)``. Not modified.
    model : numpy.ndarray
        2-D model image array with shape ``(ny, nx)``. CLEAN components
        found during this call are **added** to this array in place,
        allowing the caller to accumulate components across calls or to
        start from a non-zero initial model.
    deconvolve_params : dict, optional
        Algorithm parameters. See :func:`_validate_deconvolve_params`
        for supported keys. If ``None``, defaults are used.

    Returns
    -------
    dict
        Summary dictionary from the underlying CLEAN binding:
        ``iterations_performed``, ``final_peak``, ``total_flux_cleaned``,
        and ``converged``. The final model and residual are already in
        the caller-supplied arrays.

    Raises
    ------
    ValueError
        If the inputs are not 2-D numpy arrays of matching shape.

    Notes
    -----
    Iteration over ``(time, frequency, polarization)`` is the caller's
    responsibility (see :func:`deconvolve`).
    """
    deconvolve_params = _validate_deconvolve_params(deconvolve_params)

    if not isinstance(residual_image, np.ndarray) or residual_image.ndim != 2:
        raise ValueError("residual_image must be a 2D numpy array with shape (ny, nx)")
    if not isinstance(psf, np.ndarray) or psf.ndim != 2:
        raise ValueError("psf must be a 2D numpy array with shape (ny, nx)")
    if not isinstance(model, np.ndarray) or model.ndim != 2:
        raise ValueError("model must be a 2D numpy array with shape (ny, nx)")
    if residual_image.shape != psf.shape:
        raise ValueError(
            "residual_image and psf must have same shape. "
            f"Got {residual_image.shape} and {psf.shape}"
        )
    if model.shape != residual_image.shape:
        raise ValueError(
            "model and residual_image must have same shape. "
            f"Got {model.shape} and {residual_image.shape}"
        )

    logger.debug(f"Residual image shape: {residual_image.shape}")
    logger.debug(f"PSF shape: {psf.shape}")

    fmin, fmax = hogbom.maximg(residual_image)
    initial_peak = max(abs(fmin), abs(fmax))
    logger.debug(f"Initial peak flux: {initial_peak:.6f}")

    logger.info("Running Hogbom CLEAN algorithm...")

    clean_box = deconvolve_params["clean_box"]
    if clean_box is None:
        clean_box = (-1, -1, -1, -1)

    results = hogbom.clean(
        dirty_image=residual_image,
        psf=psf,
        model=model,
        mask=np.array([], dtype=residual_image.dtype),
        clean_box=clean_box,
        max_iter=deconvolve_params["niter"],
        gain=deconvolve_params["gain"],
        threshold=deconvolve_params["threshold"],
        progress_callback=progress_callback,
        stop_callback=None,
    )

    return results
