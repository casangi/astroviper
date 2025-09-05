import numpy as np
#from xradio.image import load_image, write_image, make_empty_sky_image
from astroviper.core.imaging.deconvolvers import hogbom

import toolviper.utils.logger as logger


# Progress callback function
def progress_callback(npol, pol, iter_num, px, py, peak):
    if iter_num % 100 == 0:
        logger.info(
            f"  Iteration {iter_num}, pol {pol}, peak at ({px}, {py}): {peak:.6f}"
        )


def _validate_dconv_params(deconv_params):
    """
    Validate and set default deconvolution parameters if unspecified.
    deconv_params: Dictionary containing deconvolution parameters such as:
        - 'gain': CLEAN gain (float, default=0.1)
        - 'niter': Maximum number of iterations (int, default=1000)
        - 'threshold': Stopping threshold (float, default=None)
    """

    # NOTE : XXX : This should probably not live here. This validation should happen much earlier

    default_params = {
        "gain": 0.1,
        "niter": 1000,
        "threshold": None,
        "clean_box": None,  # No clean box by default
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


def hogbom_clean(dirty_image_xds, psf_xds, deconv_params, output_dir="."):
    """
    Run Hogbom CLEAN algorithm on the dirty image using the provided PSF.

    Inputs:
    dirty_image_xds: xarray Dataset containing the dirty image.
    psf_xds: xarray Dataset containing the point spread function (PSF).
    deconv_params: Dictionary containing deconvolution parameters such as:
        - 'gain': CLEAN gain (float)
        - 'niter': Maximum number of iterations (int)
        - 'threshold': Stopping threshold (float)
        - 'clean_box': Optional clean box (tuple of slices or None)
    output_dir: Directory to save the output images (str)

    Returns:
    results: Dictionary containing deconvolution results, including:
        - 'model_image': The CLEAN model image (numpy array)
        - 'residual_image': The residual image after CLEAN (numpy array)
        - 'iterations_performed': Number of iterations performed (int)
        - 'final_peak': Final peak flux in the residual image (float)
        - 'total_flux_cleaned': Total flux in the model image (float)
        - 'converged': Boolean indicating if CLEAN converged (bool)
    """

    ntime = dirty_image_xds.dims["time"]
    nchan = dirty_image_xds.dims["frequency"]
    npol = dirty_image_xds.dims["polarization"]

    # Validate and set default deconvolution parameters
    deconv_params = _validate_dconv_params(deconv_params)

    # Initialize model and residual xds from dirty image
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
                mask = np.array([], dtype=np.float32),
                gain=deconv_params["gain"],
                threshold=deconv_params["threshold"],
                max_iter=deconv_params["niter"],
                clean_box=deconv_params["clean_box"] or (-1,-1,-1,-1),
                progress_callback=progress_callback,
                stop_callback=None,
            )

            model_xds["SKY"][tt, cc, ...] = results["model_image"]
            residual_xds["SKY"][tt, cc, ...] = results["residual_image"]

    return results
