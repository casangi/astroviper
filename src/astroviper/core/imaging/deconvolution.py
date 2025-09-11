import numpy as np
import xarray as xr
import copy
from typing import Optional, Tuple

from astroviper.core.imaging.deconvolvers import hogbom

import logging
import toolviper.utils.logger as logger


def progress_callback(
    npol: int,
    pol: int,
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
    npol: int
        Total number of polarizations.
    pol: int
        Current polarization index.
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
            f"  Iteration {iter_num}, pol {pol}, peak at ({px}, {py}): {peak:.6f}"
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


def hogbom_clean(
    dirty_image_xds: xr.Dataset,
    psf_xds: xr.Dataset,
    deconv_params: Optional[dict] = None,
    output_dir: str = ".",
):
    """
    Perform Hogbom CLEAN deconvolution on a dirty image using the provided PSF.

    Parameters:
    -----------
    dirty_image_xds: xarray.Dataset
        The dirty image dataset with dimensions (time, frequency, polarization, y, x).
    psf_xds: xarray.Dataset
        The PSF dataset with dimensions (time, frequency, polarization, y, x).
    deconv_params: dict or None
        Dictionary containing deconvolution parameters:
        - gain (float): CLEAN gain factor (0 < gain <= 1).
        - niter (int): Maximum number of CLEAN iterations.
        - threshold (float): Stopping threshold for CLEANing.
        - clean_box (Tuple[int] or None): Clean box defined as (ymin, ymax, xmin, xmax) or None for no box.
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
    - This function assumes that the PSF is the same for all polarizations.
    - The function loops over time and frequency dimensions to perform deconvolution on each slice.
    """

    ntime = dirty_image_xds.dims["time"]
    nchan = dirty_image_xds.dims["frequency"]
    npol = dirty_image_xds.dims["polarization"]

    # Validate and set default deconvolution parameters
    deconv_params = _validate_deconv_params(deconv_params)

    # Initialize model and residual xds from dirty image
    # XXX : Potential hotspot for memory usage here
    model_xds = dirty_image_xds.copy(deep=True)
    residual_xds = dirty_image_xds.copy(deep=True)

    # Deconvolution will loop over each time, chan, slice
    for tt in range(ntime):
        for cc in range(nchan):
            dirty_slice = dirty_image_xds["SKY"].isel(time=tt, frequency=cc).values
            psf_slice = (
                psf_xds["SKY"].isel(time=tt, frequency=cc, polarization=0).values
            )  # Assuming PSF is same for all polns

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

            model_xds["SKY"].values[tt, cc, :, :, :] = copy.deepcopy(
                results["model_image"]
            )
            residual_xds["SKY"].values[tt, cc, :, :, :] = copy.deepcopy(
                results["residual_image"]
            )

    final_results = {
        key: value
        for key, value in results.items()
        if key not in ["model_image", "residual_image"]
    }

    return final_results, model_xds, residual_xds
