import numpy as np
import xarray as xr
import copy
from typing import Optional, Tuple

from astroviper.core.imaging.deconvolvers import hogbom
from astroviper.core.image_analysis import image_statistics as imgstats
from astroviper.core.image_analysis.psf_gaussian_fit import extract_main_lobe
from astroviper.core.imaging.imaging_utils.return_dict import ReturnDict

import logging
import toolviper.utils.logger as logger

# lg = logger.get_logger()
# lg.setLevel(logging.DEBUG)

# XXX : TODO: As of 2025-10-07 there is no way to supply an initial model image to the deconvolver


def get_phase_center(dirty_image_xds):
    """
    Get the phase center from the dirty image coordinates.
    Parameters:
    -----------
    dirty_image_xds: xarray.Dataset
        The dirty image dataset with coordinates 'right_ascension' and 'declination'.
    Returns:
    --------
    phase_center: str
        The phase center in the format "RA,Dec" (e.g., "12.345,-67.890").
    """

    phase_center = ""
    ra_shape = dirty_image_xds.coords["right_ascension"].shape
    dec_shape = dirty_image_xds.coords["declination"].shape

    cx_ra, cy_ra = ra_shape[0] // 2, ra_shape[1] // 2
    cx_dec, cy_dec = dec_shape[0] // 2, dec_shape[1] // 2

    ra0 = dirty_image_xds.coords["right_ascension"].values[cx_ra, cy_ra]
    dec0 = dirty_image_xds.coords["declination"].values[cx_dec, cy_dec]
    phase_center = f"{ra0},{dec0}"
    return phase_center


def progress_callback(
    iter_num: int,
    px: int,
    py: int,
    peak: float,
    niter_log: int = 100,
):
    """
    Callback function to log progress during the CLEAN iterations.
    Logs progress every 100 iterations.

    Parameters:
    -----------
    iter_num: int
        Current iteration number.
    px: int
        X-coordinate of the current peak.
    py: int
        Y-coordinate of the current peak.
    peak: float
        Value of the current peak.
    niter_log: int
        Frequency of logging iterations (default is every 100 iterations).
    """
    if iter_num % niter_log == 0:
        logger.info(f"  Iteration {iter_num}, peak at ({px}, {py}): {peak:.6f}")


def _validate_deconv_params(deconv_params):
    """
    Validate and set default deconvolution parameters if unspecified.
    deconv_params: Dictionary containing deconvolution parameters such as:
        - 'gain': CLEAN gain (float, default=0.1)
        - 'niter': Maximum number of iterations (int, default=1000)
        - 'threshold': Stopping threshold (float, default=None)
    """

    # NOTE : XXX : This should probably not live here. This validation function
    # should live in utils or something

    default_params = {
        "gain": 0.1,
        "niter": 1000,
        "threshold": 0.0,
        "clean_box": (-1, -1, -1, -1),  # No clean box by default
    }

    for key, default_value in default_params.items():
        if key not in deconv_params:
            logger.info(
                f"Deconvolution parameter '{key}' not specified. Using default: {default_value}"
            )
            deconv_params[key] = default_value
        else:
            if key == "gain":
                if not (0 < deconv_params[key] <= 1):
                    raise ValueError("CLEAN gain must be between 0 and 1.")
            elif key == "niter":
                if not (isinstance(deconv_params[key], int) and deconv_params[key] > 0):
                    raise ValueError(
                        "Maximum number of iterations must be a positive integer."
                    )
            elif key == "threshold":
                if deconv_params[key] is not None and deconv_params[key] < 0:
                    raise ValueError("Threshold must be non-negative or None.")
            elif key == "clean_box":
                if deconv_params[key] is not None and not (
                    isinstance(deconv_params[key], tuple)
                    and len(deconv_params[key]) == 4
                ):
                    raise ValueError(
                        "Clean box must be a 4-tuple (xmin, xmax, ymin, ymax) or None."
                    )

    return deconv_params


def deconvolve(
    dirty_image_xds: xr.Dataset,
    psf_xds: xr.Dataset,
    model_xds: Optional[xr.Dataset] = None,
    algorithm: str = "hogbom",
    deconv_params: Optional[dict] = None,
    output_dir: str = ".",
):
    """
    Run the chosen deconvolution algorithm on the dirty image using the provided PSF.

    This function iterates over all (time, frequency, polarization) planes in the input
    datasets, deconvolving each plane independently, and accumulates results into full
    multi-dimensional output datasets.

    Parameters:
    -----------
    dirty_image_xds: xarray.Dataset
        The dirty image dataset with dimensions (time, frequency, polarization, y, x).
    psf_xds: xarray.Dataset
        The PSF dataset with dimensions (time, frequency, polarization, y, x).
    model_xds: xarray.Dataset or None
        The initial model image dataset with same dimensions as dirty_image_xds.
        If None, starts with zero model. If provided, deconvolution starts from
        this initial model (useful for resuming or multi-scale approaches).
    algorithm: str
        The deconvolution algorithm to use ('hogbom' supported).
    deconv_params: dict or None
        Dictionary containing deconvolution parameters specific to the chosen algorithm:
        - gain (float): CLEAN gain factor (default: 0.1)
        - niter (int): Maximum iterations per plane (default: 1000)
        - threshold (float): Stopping threshold (default: 0.0)
        - clean_box (tuple): Region to clean (xmin, xmax, ymin, ymax) or (-1,-1,-1,-1) for full image
    output_dir: str
        Directory to save intermediate outputs (not used in this implementation).

    Returns:
    --------
    returndict: ReturnDict
        A dictionary containing per-plane deconvolution statistics indexed by (time, pol, chan).
        Each entry contains: iterations performed, peak residual, model flux, PSF sidelobe levels, etc.
    model_xds: xarray.Dataset
        The model image dataset after deconvolution, with dimensions (time, frequency, polarization, y, x).
        Contains the accumulated CLEAN components from all planes.
    residual_xds: xarray.Dataset
        The residual image dataset after deconvolution, with dimensions (time, frequency, polarization, y, x).
        Contains the dirty image minus the convolved model for all planes.

    Notes:
    ------
    - Currently, only the 'hogbom' algorithm is implemented.
    - Deconvolution is performed independently on each (time, freq, pol) plane.
    - PSF sidelobe statistics are computed per (time, freq) and shared across polarizations.
    - All output datasets preserve the coordinate structure of the input dirty_image_xds.
    """

    returndict = ReturnDict()

    ntime = dirty_image_xds.sizes["time"]
    nchan = dirty_image_xds.sizes["frequency"]
    npol = dirty_image_xds.sizes["polarization"]

    masksum = imgstats.get_image_masksum(dirty_image_xds)
    phase_center = get_phase_center(dirty_image_xds)

    if algorithm.lower() == "hogbom":
        _deconvolver = hogbom_clean
    else:
        raise ValueError(f"Deconvolution algorithm '{algorithm}' not recognized.")

    psf_fit_window = (41, 41)
    psf_fit_cutoff = 0.35

    max_psf_fraction = 0.8
    min_psf_fraction = 0.1

    # Pre-allocate full model and residual datasets
    # Initialize from dirty image structure to preserve coordinates
    full_model_xds = dirty_image_xds.copy(deep=True)
    full_residual_xds = dirty_image_xds.copy(deep=True)

    # Zero out the model (residual starts as dirty image copy)
    full_model_xds["SKY"].values[:] = 0.0

    # If initial model provided, use it as starting point
    if model_xds is not None:
        full_model_xds["SKY"].values[:] = model_xds["SKY"].values[:]

    for tt in range(ntime):
        for nn in range(nchan):
            # Compute PSF sidelobe, same for all pols
            _psf_values = psf_xds.isel(time=tt, frequency=nn)["SKY"].values
            _psf_values = _psf_values[
                None, None, ...
            ]  # Add dummy axes for freq and time
            _main_lobe, _blc, _trc, max_psf_sidelobe = extract_main_lobe(
                psf_fit_window, psf_fit_cutoff, _psf_values
            )

            for pp in range(npol):
                logger.info(
                    f"Deconvolving time {tt+1}/{ntime}, freq {nn+1}/{nchan}, pol {pp+1}/{npol}"
                )

                # Extract current plane slices
                dirty_slice = dirty_image_xds.isel(
                    time=tt, frequency=nn, polarization=pp
                )
                psf_slice = psf_xds.isel(time=tt, frequency=nn, polarization=pp)

                # Compute per-plane starting statistics
                start_peakres = imgstats.image_peak_residual(
                    dirty_slice, per_plane_stats=False, use_mask=True
                )
                start_peakres_nomask = imgstats.image_peak_residual(
                    dirty_slice, per_plane_stats=False, use_mask=False
                )

                # Get starting model flux for this plane
                if model_xds is not None:
                    start_model_flux = float(
                        model_xds["SKY"]
                        .isel(time=tt, frequency=nn, polarization=pp)
                        .sum()
                        .values
                    )
                else:
                    start_model_flux = 0.0

                # Run deconvolution on single plane
                # Extract numpy arrays from xarray datasets
                results, model_array, residual_array = _deconvolver(
                    dirty_image=dirty_slice["SKY"].values,
                    psf=psf_slice["SKY"].values,
                    deconv_params=deconv_params,
                    output_dir=output_dir,
                )

                # Insert results back into full datasets at correct plane location
                full_model_xds["SKY"].values[tt, nn, pp, :, :] = model_array
                full_residual_xds["SKY"].values[tt, nn, pp, :, :] = residual_array

                # Create temporary xarray for peak residual calculation
                # (to reuse existing imgstats functions)
                temp_residual = dirty_slice.copy()
                temp_residual["SKY"].values[:] = residual_array

                peakres = imgstats.image_peak_residual(
                    temp_residual, per_plane_stats=False, use_mask=True
                )
                peakres_nomask = imgstats.image_peak_residual(
                    temp_residual, per_plane_stats=False, use_mask=False
                )

                # Extract coordinate values for this plane
                stokes = dirty_slice.coords["polarization"].values
                freq = dirty_slice.coords["frequency"].values
                time = dirty_slice.coords["time"].values

                returnvals = {
                    "niter": deconv_params.get("niter", None),
                    "threshold": deconv_params.get("threshold", None),
                    "iter_done": results.get("iterations_performed", None),
                    "loop_gain": deconv_params.get("gain", None),
                    "min_psf_fraction": min_psf_fraction,
                    "max_psf_fraction": max_psf_fraction,
                    "max_psf_sidelobe": max_psf_sidelobe,
                    "stop_code": None,
                    "stokes": stokes,
                    "frequency": freq,
                    "phase_center": phase_center,
                    "time": time,
                    "start_model_flux": start_model_flux,
                    "start_peakres": start_peakres,
                    "start_peakres_nomask": start_peakres_nomask,
                    "peakres": peakres,
                    "peakres_nomask": peakres_nomask,
                    "masksum": masksum,
                }

                returndict.add(returnvals, time=tt, pol=pp, chan=nn)

    return returndict, full_model_xds, full_residual_xds


def hogbom_clean(
    dirty_image: np.ndarray,
    psf: np.ndarray,
    deconv_params: Optional[dict] = None,
    output_dir: str = ".",
):
    """
    Perform Hogbom CLEAN deconvolution on a dirty image using the provided PSF.
    The input dirty image and PSF are expected to be **single plane** 2D numpy arrays.
    Any iteration over time, frequency, polarization is done outside this function.

    Parameters:
    -----------
    dirty_image: np.ndarray
        The 2D dirty image array with shape (ny, nx).
    psf: np.ndarray
        The 2D PSF array with shape (ny, nx).
    deconv_params: dict or None
        Dictionary containing deconvolution parameters:
        - gain (float): CLEAN gain factor (0 < gain <= 1).
        - niter (int): Maximum number of CLEAN iterations.
        - threshold (float): Stopping threshold for CLEANing.
        - clean_box (Tuple[int] or None): Clean box defined as (xmin, xmax, ymin, ymax) or None for no box.
        If None, default parameters will be used.
    output_dir: str
        Directory to save intermediate outputs (not used in this implementation).

    Returns:
    --------
    results: dict
        A dictionary containing deconvolution statistics (iterations_performed, final_peak, etc.)
        excluding the image arrays.
    model_array: np.ndarray
        The 2D model image array after deconvolution (shape: [ny, nx]).
    residual_array: np.ndarray
        The 2D residual image array after deconvolution (shape: [ny, nx]).

    Notes:
    ------
    - Input arrays must be 2D numpy arrays (ny, nx).
    - Returns numpy arrays for efficient accumulation by caller.
    - Iteration over time, frequency, and polarization is handled by the caller (deconvolve function).
    """

    # Validate and set default deconvolution parameters
    deconv_params = _validate_deconv_params(deconv_params)

    # Validate input arrays
    if not isinstance(dirty_image, np.ndarray) or dirty_image.ndim != 2:
        raise ValueError("dirty_image must be a 2D numpy array with shape (ny, nx)")
    if not isinstance(psf, np.ndarray) or psf.ndim != 2:
        raise ValueError("psf must be a 2D numpy array with shape (ny, nx)")
    if dirty_image.shape != psf.shape:
        raise ValueError(
            f"dirty_image and psf must have same shape. Got {dirty_image.shape} and {psf.shape}"
        )

    logger.debug(f"Dirty image shape: {dirty_image.shape}")
    logger.debug(f"PSF shape: {psf.shape}")

    # Find initial peak in dirty image
    fmin, fmax = hogbom.maximg(dirty_image)
    initial_peak = max(abs(fmin), abs(fmax))
    logger.debug(f"Initial peak flux: {initial_peak:.6f}")

    # Run CLEAN with full interface (including callbacks)
    logger.info("\nRunning Hogbom CLEAN algorithm...")
    results = hogbom.clean(
        dirty_image=dirty_image,
        psf=psf,
        mask=np.array([], dtype=np.float32),
        gain=deconv_params["gain"],
        threshold=deconv_params["threshold"],
        max_iter=deconv_params["niter"],
        clean_box=(
            deconv_params["clean_box"]
            if deconv_params["clean_box"]
            else (-1, -1, -1, -1)
        ),
        progress_callback=progress_callback,
        stop_callback=None,
    )

    # Extract arrays from results
    model_array = results["model_image"]
    residual_array = results["residual_image"]

    # Return statistics dict (without image arrays) and the arrays separately
    final_results = {
        key: value
        for key, value in results.items()
        if key not in ["model_image", "residual_image"]
    }

    return final_results, model_array, residual_array
