"""
End-to-end imaging loop similar to CASA tclean.

This script orchestrates gridding, FFT, deconvolution, and degridding
in a major/minor cycle loop with visibility-domain residual calculation.

Usage:
    Assumes pre-loaded ms4 xarray DataTree and image parameters.
"""

import numpy as np
import xarray as xr
from typing import Optional, Tuple, Dict, Any
from copy import deepcopy

# Astroviper imports
from astroviper.core.imaging.imaging_utils.standard_grid import (
    standard_grid_numpy_wrap_input_checked,
    grid2image_spheroid_ms4,
)
from astroviper.core.imaging.imaging_utils.standard_degrid import degrid_spheroid_ms4
from astroviper.core.imaging.imaging_utils.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel_1D,
    create_prolate_spheroidal_kernel,
)
from astroviper.core.imaging.imaging_utils.corr_to_stokes import (
    image_corr_to_stokes,
    image_stokes_to_corr,
)
from astroviper.core.imaging.imaging_utils.make_point_spread_function import make_psf
from astroviper.core.imaging.imaging_utils.iteration_control import (
    IterationController,
    ReturnDict,
)
from astroviper.core.imaging.deconvolution import deconvolve
from astroviper.core.imaging.fft import fft_lm_to_uv
from astroviper.core.imaging.ifft import ifft_uv_to_lm
from xradio.image import make_empty_sky_image


def _get_param(params: Dict, key: str):
    """Get parameter with fallback to default."""

    # Default imaging parameters
    default_params = {
        # Image geometry
        "image_size": (256, 256),
        "cell_size": None,
        # Gridding parameters
        "support": 7,
        "oversampling": 100,
        # Deconvolution parameters
        "algorithm": "hogbom",
        "gain": 0.1,
        "niter": 1000,
        "threshold": 0.0,
        # Major cycle parameters
        "nmajor": 10,
        "cyclefactor": 1.5,
        "cycleniter": -1,  # -1 means adaptive
        "minpsffraction": 0.05,
        "maxpsffraction": 0.8,
        # Spectral mode
        "chan_mode": "continuum",  # "continuum" or "cube"
        # Polarization
        "corr_type": "linear",
    }
    if key in params:
        return params[key]
    else:
        return default_params.get(key)


def grid_visibilities(
    ms4: xr.DataTree,
    image_size: Tuple[int, int],
    cell_size: Tuple[float, float],
    support: int = 7,
    oversampling: int = 100,
    chan_mode: str = "continuum",
    column: str = "VISIBILITY",
) -> np.ndarray:
    """
    Grid visibilities from ms4 to image plane.

    Parameters
    ----------
    ms4 : xr.DataTree
        Measurement set v4 xarray DataTree.
    image_size : tuple
        (ny, nx) image dimensions in pixels.
    cell_size : tuple
        (dy, dx) cell size in radians.
    support : int
        Convolution function support size.
    oversampling : int
        Convolution function oversampling factor.
    chan_mode : str
        "continuum" or "cube".
    column : str
        Which data column to grid: "VISIBILITY" or "RESIDUAL".

    Returns
    -------
    dirty_image : np.ndarray
        Dirty image array with shape depending on chan_mode.
        continuum: (1, n_pol, ny, nx)
        cube: (n_chan, n_pol, ny, nx)
    """
    n_chan = 1 if chan_mode == "continuum" else len(ms4.coords["frequency"])
    n_pol = len(ms4.coords["polarization"])
    ny, nx = image_size

    # Initialize output array
    dirty_image = np.zeros([n_chan, n_pol, ny, nx], dtype=float)

    pixelincr = np.array([cell_size[0], cell_size[1]])

    grid2image_spheroid_ms4(
        vis=ms4,
        resid_array=dirty_image,
        pixelincr=pixelincr,
        support=support,
        sampling=oversampling,
        dopsf=False,
        column=column,
        chan_mode=chan_mode,
    )

    return dirty_image


def compute_residual_visibilities(ms4: xr.DataTree) -> None:
    """
    Compute residual visibilities: RESIDUAL = VISIBILITY - VISIBILITY_MODEL.

    Modifies ms4 in-place by adding/updating the RESIDUAL data variable.

    Parameters
    ----------
    ms4 : xr.DataTree
        Measurement set with VISIBILITY and VISIBILITY_MODEL columns.
    """
    vis_data = ms4["VISIBILITY"].values
    vis_model = ms4["VISIBILITY_MODEL"].values

    residual = vis_data - vis_model

    # Add or update RESIDUAL in ms4
    ms4["RESIDUAL"] = xr.DataArray(
        residual,
        dims=ms4["VISIBILITY"].dims,
        coords=ms4["VISIBILITY"].coords,
    )


def predict_model_visibilities(
    ms4: xr.DataTree,
    model_image: np.ndarray,
    cell_size: Tuple[float, float],
    support: int = 7,
    oversampling: int = 100,
    incremental: bool = False,
) -> None:
    """
    Degrid model image to predict model visibilities.

    FFTs the model image to UV plane, then degrids onto ms4.VISIBILITY_MODEL.

    Parameters
    ----------
    ms4 : xr.DataTree
        Measurement set v4 xarray DataTree.
    model_image : np.ndarray
        Model image in image plane, shape (n_chan, n_pol, ny, nx).
    cell_size : tuple
        (dy, dx) cell size in radians.
    support : int
        Convolution function support size.
    oversampling : int
        Convolution function oversampling factor.
    incremental : bool
        If True, add to existing VISIBILITY_MODEL; if False, replace.
    """
    # FFT model image to UV plane
    model_grid = fft_lm_to_uv(model_image, axes=[2, 3])

    pixelincr = np.array([cell_size[0], cell_size[1]])

    # Degrid to visibilities
    degrid_spheroid_ms4(
        vis=ms4,
        grid=model_grid,
        pixelincr=pixelincr,
        support=support,
        sampling=oversampling,
        incremental=incremental,
    )


def run_imaging_loop(
    ms4: xr.DataTree,
    params: Dict[str, Any],
    initial_model: Optional[np.ndarray] = None,
    output_dir: str = ".",
) -> Tuple[np.ndarray, np.ndarray, ReturnDict, IterationController]:
    """
    Run the full imaging loop with major/minor cycles.

    Parameters
    ----------
    ms4 : xr.DataTree
        Measurement set v4 xarray DataTree with VISIBILITY column.
    params : dict
        Imaging parameters dictionary. Required keys:
            - cell_size: tuple (dy, dx) in radians
        Optional keys (with defaults from DEFAULT_IMAGING_PARAMS):
            - image_size: tuple (ny, nx), default (256, 256)
            - support: int, default 7
            - oversampling: int, default 100
            - algorithm: str, default "hogbom"
            - gain: float, default 0.1
            - niter: int, default 1000
            - threshold: float, default 0.0
            - nmajor: int, default 10
            - cyclefactor: float, default 1.5
            - cycleniter: int, default -1 (adaptive)
            - minpsffraction: float, default 0.05
            - maxpsffraction: float, default 0.8
            - chan_mode: str, "continuum" or "cube", default "continuum"
            - corr_type: str, "linear" or "circular", default "linear"
    initial_model : np.ndarray, optional
        Initial model image to start from. If None, starts from zero.
    output_dir : str
        Directory for output files.

    Returns
    -------
    model : np.ndarray
        Final model image (Stokes basis).
    residual : np.ndarray
        Final residual image (Stokes basis).
    return_dict : ReturnDict
        Deconvolution statistics and convergence history.
    controller : IterationController
        Iteration controller with final state.
    """
    # Extract parameters with defaults
    cell_size = _get_param(params, "cell_size")
    image_size = _get_param(params, "image_size")
    support = _get_param(params, "support")
    oversampling = _get_param(params, "oversampling")
    algorithm = _get_param(params, "algorithm")
    gain = _get_param(params, "gain")
    niter = _get_param(params, "niter")
    threshold = _get_param(params, "threshold")
    nmajor = _get_param(params, "nmajor")
    cyclefactor = _get_param(params, "cyclefactor")
    cycleniter = _get_param(params, "cycleniter")
    minpsffraction = _get_param(params, "minpsffraction")
    maxpsffraction = _get_param(params, "maxpsffraction")
    chan_mode = _get_param(params, "chan_mode")
    corr_type = _get_param(params, "corr_type")

    # Validate cell_size
    if cell_size is None:
        raise ValueError("cell_size must be specified in params")

    ny, nx = image_size
    n_chan = 1 if chan_mode == "continuum" else len(ms4.coords["frequency"])

    # Extract coordinates from ms4 for image xds creation
    phase_center = ms4.attrs.get("phase_center", (0.0, 0.0))
    if chan_mode == "continuum":
        # Use mean frequency for continuum
        freq_coords = [float(np.mean(ms4.coords["frequency"].values))]
    else:
        freq_coords = ms4.coords["frequency"].values
    time_coords = [float(ms4.coords["time"].values[0])]  # Single time for now
    pol_coords = ["I", "Q", "U", "V"]  # Stokes parameters

    print(f"Imaging configuration:")
    print(f"  Image size: {ny} x {nx}")
    print(f"  Cell size: {cell_size[0]*206265:.4f} x {cell_size[1]*206265:.4f} arcsec")
    print(f"  Channels: {n_chan} ({chan_mode})")
    print(f"  Max iterations: {niter}")
    print(f"  Max major cycles: {nmajor}")
    print(f"  Threshold: {threshold} Jy")
    print()

    # Initialize iteration controller
    controller = IterationController(
        niter=niter,
        nmajor=nmajor,
        threshold=threshold,
        gain=gain,
        cyclefactor=cyclefactor,
        minpsffraction=minpsffraction,
        maxpsffraction=maxpsffraction,
        cycleniter=cycleniter,
    )

    # Make PSF (once, doesn't change during loop)
    print("Making PSF...")
    im_params = {
        "cell_size": cell_size,
        "image_size": image_size,
        "phase_center": phase_center,
        "chan_mode": chan_mode,
    }
    grid_params = {
        "sampling": oversampling,
        "complex_grid": True,
        "support": support,
    }
    psf_da = make_psf(ms4, im_params, grid_params)

    # Convert to Stokes (keep 5D: time, freq, pol, l, m)
    psf_stokes = image_corr_to_stokes(psf_da.values, corr_type=corr_type, pol_axis=2)

    # Create PSF xarray dataset with Stokes polarization
    psf_xds = make_empty_sky_image(
        phase_center=phase_center,
        image_size=image_size,
        cell_size=cell_size,
        frequency_coords=freq_coords,
        pol_coords=pol_coords,
        time_coords=time_coords,
    )
    psf_xds["POINT_SPREAD_FUNCTION"] = xr.DataArray(
        psf_stokes,
        dims=["time", "frequency", "polarization", "l", "m"],
    )

    # Initialize model (Stokes basis)
    if initial_model is not None:
        model_stokes = initial_model.copy()
    else:
        model_stokes = np.zeros([n_chan, 4, ny, nx], dtype=float)  # 4 Stokes params

    # Return dict for tracking convergence
    combined_return_dict = ReturnDict()

    # Create dirty_xds once and reuse (updated with new residuals each cycle)
    dirty_xds = None

    major_cycle = 0

    print("Starting major cycle loop...")
    print()

    while controller.stopcode.major == 0:
        major_cycle += 1
        print(f"=== Major Cycle {major_cycle} ===")

        # Step 1: Get dirty/residual image
        if major_cycle == 1:
            # First cycle: grid original visibilities
            print("  Gridding VISIBILITY column...")
            dirty_corr = grid_visibilities(
                ms4,
                image_size,
                cell_size,
                support=support,
                oversampling=oversampling,
                chan_mode=chan_mode,
                column="VISIBILITY",
            )
        else:
            # Subsequent cycles: grid residual visibilities
            print("  Gridding RESIDUAL column...")
            dirty_corr = grid_visibilities(
                ms4,
                image_size,
                cell_size,
                support=support,
                oversampling=oversampling,
                chan_mode=chan_mode,
                column="RESIDUAL",
            )

        # Step 2: Convert to Stokes
        print("  Converting to Stokes basis...")
        dirty_stokes = image_corr_to_stokes(dirty_corr, corr_type=corr_type, pol_axis=1)

        # Create dirty_xds once, then just update the data values on subsequent cycles
        if dirty_xds is None:
            dirty_xds = make_empty_sky_image(
                phase_center=phase_center,
                image_size=image_size,
                cell_size=cell_size,
                frequency_coords=freq_coords,
                pol_coords=pol_coords,
                time_coords=time_coords,
            )
            dirty_5d = dirty_stokes[np.newaxis, ...]  # Add time dimension
            dirty_xds["RESIDUAL"] = xr.DataArray(
                dirty_5d,
                dims=["time", "frequency", "polarization", "l", "m"],
            )
        else:
            # Update existing dataset with new residual values
            dirty_xds["RESIDUAL"].values[0, ...] = dirty_stokes

        # Step 3: Calculate cycle controls
        if major_cycle == 1:
            # First cycle: need to create initial return dict
            # Get peak residual from dirty image
            peak_res = np.max(np.abs(dirty_stokes))
            temp_rd = ReturnDict()
            temp_rd.add(
                {
                    "peakres": peak_res,
                    "peakres_nomask": peak_res,
                    "masksum": ny * nx,
                    "iter_done": 0,
                    "max_psf_sidelobe": 0.2,
                    "loop_gain": gain,
                },
                time=0,
                pol=0,
                chan=0,
            )
            cycle_niter, cyclethresh = controller.calculate_cycle_controls(temp_rd)
        else:
            cycle_niter, cyclethresh = controller.calculate_cycle_controls(
                combined_return_dict
            )

        print(f"  Cycle controls: niter={cycle_niter}, threshold={cyclethresh:.6f} Jy")

        # Step 4: Create model xds from current accumulated model (5D with singleton time)
        if np.any(model_stokes != 0):
            model_xds = make_empty_sky_image(
                phase_center=phase_center,
                image_size=image_size,
                cell_size=cell_size,
                frequency_coords=freq_coords,
                pol_coords=pol_coords,
                time_coords=time_coords,
            )
            model_5d = model_stokes[np.newaxis, ...]  # Add time dimension
            model_xds["MODEL"] = xr.DataArray(
                model_5d,
                dims=["time", "frequency", "polarization", "l", "m"],
            )
        else:
            model_xds = None

        # Step 5: Run deconvolution
        print(f"  Running {algorithm} deconvolution...")
        deconv_params = {
            "gain": gain,
            "niter": cycle_niter,
            "threshold": cyclethresh,
        }

        return_dict, new_model_xds, residual_xds = deconvolve(
            dirty_image_xds=dirty_xds,
            psf_xds=psf_xds,
            model_xds=model_xds,
            algorithm=algorithm,
            deconv_params=deconv_params,
            output_dir=output_dir,
        )

        # Update accumulated model (Stokes)
        model_stokes = new_model_xds["MODEL"].values[0, ...]  # Remove time dim

        # Get residual (Stokes)
        residual_stokes = residual_xds["RESIDUAL"].values[0, ...]  # Remove time dim

        # Report progress
        peak_res = np.max(np.abs(residual_stokes))
        model_flux = np.sum(model_stokes[:, 0, :, :])  # Stokes I flux
        print(f"  Peak residual: {peak_res:.6f} Jy")
        print(f"  Model flux: {model_flux:.6f} Jy")

        # Step 6: Update iteration controller
        controller.update_counts(return_dict)

        # Step 7: Merge return dict for history tracking
        for key, value in return_dict.data.items():
            combined_return_dict.add(value, time=key[0], pol=key[1], chan=key[2])

        # Step 8: Check convergence
        stopcode, stopdesc = controller.check_convergence(return_dict)

        if stopcode != 0:
            print(f"\n  *** CONVERGED: {stopdesc} ***")
            break

        # Step 9: Prepare for next major cycle - predict model visibilities
        print("  Predicting model visibilities...")

        # Convert accumulated model to correlation basis for degridding
        model_corr_accumulated = image_stokes_to_corr(
            model_stokes, corr_type=corr_type, pol_axis=1
        )

        # Degrid model to VISIBILITY_MODEL
        predict_model_visibilities(
            ms4,
            model_corr_accumulated,
            cell_size,
            support=support,
            oversampling=oversampling,
            incremental=False,
        )

        # Step 10: Compute residual visibilities
        print("  Computing residual visibilities...")
        compute_residual_visibilities(ms4)

    print("=== Final Results ===")
    print(f"Total major cycles: {controller.major_done}")
    print(f"Total iterations: {controller.total_iter_done}")
    print(f"Final peak residual: {np.max(np.abs(residual_stokes)):.6f} Jy")
    print(f"Final model flux: {np.sum(model_stokes[:, 0, :, :]):.6f} Jy")
    print(f"Stop code: {controller.stopcode}")
    print(f"Stop reason: {controller.stopdescription}")

    return model_stokes, residual_stokes, combined_return_dict, controller
