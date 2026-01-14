"""
An end-to-end imaging script, including gridding, deconvolution,
and iteration control.
"""

import numpy as np
import xarray as xr
from typing import Optional, Dict, Any, Tuple
import logging

from astroviper.core.imaging.iteration_control import (
    IterationController,
    StopCode,
    MAJOR_CONTINUE,
    MAJOR_ITER_LIMIT,
    MAJOR_THRESHOLD,
    MAJOR_ZERO_MASK,
    MAJOR_CYCLE_LIMIT,
)
from astroviper.core.imaging.imaging_utils.return_dict import ReturnDict, Key
from astroviper.core.imaging.psf_analysis import analyze_psf_xds
from astroviper.core.imaging.deconvolution import hogbom_clean
from astroviper.core.imaging.make_visibility_grid import make_visibility_grid
from astroviper.core.imaging.fft_norm_img_xds import fft_norm_img_xds

logger = logging.getLogger(__name__)


def clean(
    ms_xds: xr.Dataset,
    vis_sel_params: Dict[str, Any],
    grid_params: Dict[str, Any],
    iteration_control_params: Dict[str, Any],
    gridder: str = "standard",
    deconvolver: str = "hogbom",
    image_size: Optional[Tuple[int, int]] = [256, 256],
    niter: int = 0,
    nmajor: int = 0,
    threshold: float = 0.0,
    gain: float = 0.1,
    cyclefactor: float = 1.5,
    cycleniter: int = 10,
) -> Tuple[xr.Dataset, ReturnDict]:
    """
    Perform deconvolution on the image dataset using the specified parameters.

    Parameters
    ----------
    ms_xds : xr.Dataset
        Measurement set dataset.
    gridder : str, optional
        The gridding algorithm to use. Default is 'standard'.
    deconvolver : str, optional
        The deconvolution algorithm to use. Default is 'hogbom'.
    image_size : Tuple[int, int], optional
        Size of the image to be cleaned. Default is (256, 256).
    niter : int, optional
        Number of iterations for the deconvolution. Default is 0.
    vis_sel_params : Dict[str, Any]
        Selection parameters for visibility data.
    grid_params : Dict[str, Any]
        Parameters for gridding.
    iteration_control_params : Dict[str, Any]
        Parameters for controlling the iteration process.

    Returns
    -------
    Tuple[xr.Dataset, ReturnDict]
        The cleaned image dataset and a return dictionary with details of the cleaning process.
    """

    # Initialize return dictionary
    ret_dict = ReturnDict()

    # Initialize iteration controller
    iter_controller = IterationController(iteration_control_params)

    # Initialize iteration controller
    iter_controller = IterationController(
        niter=niter,
        nmajor=nmajor,
        threshold=threshold,
        gain=gain,
        cyclefactor=cyclefactor,
        cycleniter=cycleniter,
        minpsffraction=minpsffraction,
        maxpsffraction=maxpsffraction,
        nsigma=nsigma,
    )

    # Create visibility grid
    vis_grid_xds = make_visibility_grid(
        ms_xds, gcf_xds, img_xds, vis_sel_params, img_sel_params, grid_params
    )

    # Normalize image dataset after FFT
    img_xds = fft_norm_img_xds(img_xds, grid_params)

    major_cycle = 0
    stop_code = StopCode.CONTINUE

    while stop_code == StopCode.CONTINUE:
        major_cycle += 1
        logger.info(f"Starting major cycle {major_cycle}")

        # Perform Hogbom clean
        img_xds, clean_info = hogbom_clean(
            img_xds, vis_grid_xds, clean_params, img_sel_params, grid_params
        )

        # Update return dictionary with clean info
        # ret_dict[Key.CLEAN_INFO] =
