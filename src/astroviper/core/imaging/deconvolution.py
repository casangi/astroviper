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

#lg = logger.get_logger()
#lg.setLevel(logging.DEBUG)

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
        logger.info(
            f"  Iteration {iter_num}, peak at ({px}, {py}): {peak:.6f}"
        )


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
                    and len(deconv_params[key]) == 2
                ):
                    raise ValueError("Clean box must be a tuple of slices or None.")

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

    Parameters:
    -----------
    dirty_image_xds: xarray.Dataset
        The dirty image dataset with dimensions (time, frequency, polarization, y, x).
    psf_xds: xarray.Dataset
        The PSF dataset with dimensions (time, frequency, polarization, y, x).
    model_xds: xarray.Dataset or None
        The initial model image dataset
    algorithm: str
        The deconvolution algorithm to use ('hogbom' supported).
    deconv_params: dict or None
        Dictionary containing deconvolution parameters specific to the chosen algorithm.
    output_dir: str
        Directory to save intermediate outputs (not used in this implementation).
    Returns:
    --------
    results: dict
        A dictionary containing the model image and residual image after deconvolution.
    model_xds: xarray.Dataset
        The model image dataset after deconvolution.
    residual_xds: xarray.Dataset
        The residual image dataset after deconvolution.

    Notes:
    ------
    - Currently, only the 'hogbom' algorithm is implemented.
    """

    returndict = ReturnDict()

    start_model_flux = 0.0
    start_peakres = 0.0

    ntime = dirty_image_xds.dims["time"]
    nchan = dirty_image_xds.dims["frequency"]
    npol = dirty_image_xds.dims["polarization"]

    # XXX : TODO : This should be within the time/pol/chan loop
    if model_xds is not None:
        start_model_flux = float(model_xds["SKY"].sum().values)

    masksum = imgstats.get_image_masksum(dirty_image_xds)
    phase_center = get_phase_center(dirty_image_xds)

    logger.info(f"Starting peak residual (with mask): {start_peakres:.6f}")

    if algorithm.lower() == "hogbom":
        _deconvolver = hogbom_clean
    else:
        raise ValueError(f"Deconvolution algorithm '{algorithm}' not recognized.")

    psf_fit_window = (41, 41)
    psf_fit_cutoff = 0.35

    max_psf_fraction = 0.8
    min_psf_fraction = 0.1


    for tt in range(ntime):
        for nn in range(nchan):
            # Compute PSF sidelobe, same for all pols
            _psf_values = psf_xds.isel(time=tt, frequency=nn)["SKY"].values
            _psf_values = _psf_values[None, None, ...]  # Add dummy axes for freq and time
            _main_lobe, _blc, _trc, max_psf_sidelobe = extract_main_lobe(
                psf_fit_window, psf_fit_cutoff, _psf_values
            )

            for pp in range(npol):
                logger.info(
                    f"Deconvolving time {tt+1}/{ntime}, freq {nn+1}/{nchan}, pol {pp+1}/{npol}"
                )

                start_peakres = imgstats.image_peak_residual(
                    dirty_image_xds, per_plane_stats=False, use_mask=True
                )
                start_peakres_nomask = imgstats.image_peak_residual(
                    dirty_image_xds, per_plane_stats=False, use_mask=False
                )

                dirty_slice = dirty_image_xds.isel(
                    time=tt, frequency=nn, polarization=pp
                )
                psf_slice = psf_xds.isel(time=tt, frequency=nn, polarization=pp)

                results, model_xds, residual_xds = _deconvolver(
                    dirty_image_xds=dirty_slice,
                    psf_xds=psf_slice,
                    deconv_params=deconv_params,
                    output_dir=output_dir,
                )

                peakres = imgstats.image_peak_residual(
                    residual_xds, per_plane_stats=False, use_mask=True
                )
                peakres_nomask = imgstats.image_peak_residual(
                    residual_xds, per_plane_stats=False, use_mask=False
                )

                stokes = residual_xds.coords["polarization"].values
                freq = residual_xds.coords["frequency"].values
                time = residual_xds.coords["time"].values

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

    # TODO : Need to add min/max psf sidelobe levels to the return dict
    return returndict, model_xds, residual_xds


def hogbom_clean(
    dirty_image_xds: xr.Dataset,
    psf_xds: xr.Dataset,
    deconv_params: Optional[dict] = None,
    output_dir: str = ".",
):
    """
    Perform Hogbom CLEAN deconvolution on a dirty image using the provided PSF.
    The input dirty image and PSF are expected to be **single plane** images (2D, spatial dimensions only).
    Any iteration over time, frequency, polarization is done outside this function.

    Parameters:
    -----------
    dirty_image_xds: xarray.Dataset
        The dirty image dataset with dimensions (y, x) - a single 2D plane
        (obtained after selecting specific time, frequency, and polarization indices).
    psf_xds: xarray.Dataset
        The PSF dataset with dimensions (y, x) - a single 2D plane
        (obtained after selecting specific time, frequency, and polarization indices).
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
        A dictionary containing the model image and residual image after deconvolution.
    model_xds: xarray.Dataset
        The model image dataset after deconvolution.
    residual_xds: xarray.Dataset
        The residual image dataset after deconvolution.

    Notes:
    ------
    - Input arrays must be 2D (y, x dimensions only).
    - Iteration over time, frequency, and polarization is handled by the caller (deconvolve function).
    """

    # Validate and set default deconvolution parameters
    deconv_params = _validate_deconv_params(deconv_params)

    # Initialize model and residual xds from dirty image
    # XXX : Potential hotspot for memory usage here
    model_xds = dirty_image_xds.copy(deep=True)
    residual_xds = dirty_image_xds.copy(deep=True)

    # Get the underlying values to pass into
    # pybinded function
    dirty_slice = dirty_image_xds["SKY"].values
    psf_slice = psf_xds["SKY"].values

    logger.debug(f"Dirty image shape: {dirty_slice.shape}")
    logger.debug(f"PSF shape: {psf_slice.shape}")

    # Find initial peak in dirty image
    fmin, fmax = hogbom.maximg(dirty_slice)
    initial_peak = max(abs(fmin), abs(fmax))
    logger.debug(f"Initial peak flux: {initial_peak:.6f}")

    # Run CLEAN with full interface (including callbacks)
    logger.info("\nRunning Hogbom CLEAN algorithm...")
    results = hogbom.clean(
        dirty_image=dirty_slice,
        psf=psf_slice,
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

    model_xds["SKY"].values[:] = copy.deepcopy(results["model_image"])
    residual_xds["SKY"].values[:] = copy.deepcopy(results["residual_image"])

    # XXX : TODO : This will only return the results from the last slice cleaned
    final_results = {
        key: value
        for key, value in results.items()
        if key not in ["model_image", "residual_image"]
    }

    return final_results, model_xds, residual_xds
