"""
End-to-end imaging loop similar to CASA tclean.

Orchestrates gridding, FFT, deconvolution, and degridding in major/minor cycles.
"""

import numpy as np
import xarray as xr
from typing import Optional, Tuple, Dict, Any

from astroviper.core.imaging.imaging_utils.standard_grid import grid2image_spheroid_ms4
from astroviper.core.imaging.imaging_utils.standard_degrid import degrid_spheroid_ms4
from astroviper.core.imaging.imaging_utils.corr_to_stokes import (
    image_corr_to_stokes,
    image_stokes_to_corr,
)
from astroviper.core.imaging.imaging_utils.make_point_spread_function import make_psf
from astroviper.core.imaging.imaging_utils.iteration_control import (
    IterationController,
    ReturnDict,
    merge_return_dicts,
)
from astroviper.core.imaging.deconvolution import deconvolve
from astroviper.core.imaging.fft import fft_lm_to_uv
from xradio.image import make_empty_sky_image

# Default imaging parameters
_DEFAULT_PARAMS = {
    "image_size": (256, 256),
    "cell_size": None,
    "support": 7,
    "oversampling": 100,
    "algorithm": "hogbom",
    "gain": 0.1,
    "niter": 1000,
    "threshold": 0.0,
    "nmajor": 10,
    "cyclefactor": 1.5,
    "cycleniter": -1,
    "minpsffraction": 0.05,
    "maxpsffraction": 0.8,
    "chan_mode": "cube",
    "corr_type": "linear",
    "stokes": None,
}


def _get_param(params: Dict, key: str):
    """Get parameter with fallback to default."""
    return params.get(key, _DEFAULT_PARAMS.get(key))


def grid_visibilities(
    ms4: xr.DataTree,
    image_size: Tuple[int, int],
    cell_size: Tuple[float, float],
    support: int = 7,
    oversampling: int = 100,
    chan_mode: str = "cube",
    column: str = "VISIBILITY",
) -> np.ndarray:
    """Grid visibilities from ms4 to image plane. Returns (n_chan, n_pol, ny, nx)."""
    n_chan = len(ms4.coords["frequency"])
    n_pol = len(ms4.coords["polarization"])
    ny, nx = image_size

    dirty_image = np.zeros([n_chan, n_pol, ny, nx], dtype=float)
    grid2image_spheroid_ms4(
        vis=ms4,
        resid_array=dirty_image,
        pixelincr=np.array(cell_size),
        support=support,
        sampling=oversampling,
        dopsf=False,
        column=column,
        chan_mode=chan_mode,
    )
    return dirty_image


def compute_residual_visibilities(ms4: xr.DataTree) -> None:
    """Compute RESIDUAL = VISIBILITY - VISIBILITY_MODEL in-place."""

    return ms4["RESIDUAL"] = ms4["VISIBILITY"] - ms4['VISIBILITY_MODEL']


def predict_model_visibilities(
    ms4: xr.DataTree,
    model_image: np.ndarray,
    cell_size: Tuple[float, float],
    support: int = 7,
    oversampling: int = 100,
    incremental: bool = False,
) -> None:
    """FFT model image to UV plane and degrid onto ms4.VISIBILITY_MODEL."""
    model_grid = fft_lm_to_uv(model_image, axes=[2, 3])
    degrid_spheroid_ms4(
        vis=ms4,
        grid=model_grid,
        pixelincr=np.array(cell_size),
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
        Measurement set with VISIBILITY column.
    params : dict
        Required: cell_size (dy, dx in radians).
        Optional: image_size, support, oversampling, algorithm, gain, niter,
        threshold, nmajor, cyclefactor, cycleniter, minpsffraction,
        maxpsffraction, chan_mode, corr_type, stokes. See _DEFAULT_PARAMS.
    initial_model : np.ndarray, optional
        Initial model image. If None, starts from zero.
    output_dir : str
        Directory for output files.

    Returns
    -------
    tuple : (model, residual, return_dict, controller)
    """
    # Extract parameters with defaults
    p = {k: _get_param(params, k) for k in _DEFAULT_PARAMS}
    cell_size, image_size = p["cell_size"], p["image_size"]
    support, oversampling = p["support"], p["oversampling"]
    algorithm, gain, niter, threshold = (
        p["algorithm"],
        p["gain"],
        p["niter"],
        p["threshold"],
    )
    nmajor, cyclefactor, cycleniter = p["nmajor"], p["cyclefactor"], p["cycleniter"]
    minpsffraction, maxpsffraction = p["minpsffraction"], p["maxpsffraction"]
    chan_mode, corr_type, stokes = p["chan_mode"], p["corr_type"], p["stokes"]

    if cell_size is None:
        raise ValueError("cell_size must be specified in params")

    ny, nx = image_size
    n_chan = len(ms4.coords["frequency"])
    n_corr = len(ms4.coords["polarization"])

    # Auto-detect stokes if not provided
    if stokes is None:
        stokes = (
            ["I", "Q"]
            if n_corr == 2 and corr_type == "linear"
            else ["I", "V"] if n_corr == 2 else ["I", "Q", "U", "V"]
        )
    n_stokes = len(stokes)

    # Coordinates for image xds creation
    phase_center = ms4.attrs.get("phase_center", (0.0, 0.0))
    freq_coords = ms4.coords["frequency"].values
    time_coords = [float(ms4.coords["time"].values[0])]

    print(
        f'Imaging: {ny}x{nx}, {cell_size[0]*206265:.4f}" cells, {n_chan} chan, '
        f"{n_corr}-corr ({corr_type}) → {stokes}"
    )
    print(f"  niter={niter}, nmajor={nmajor}, threshold={threshold} Jy")

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

    # Make PSF (Stokes I only, same for all polarizations)
    print("Making PSF...")
    psf_da = make_psf(
        ms4,
        {
            "cell_size": cell_size,
            "image_size": image_size,
            "phase_center": phase_center,
            "chan_mode": chan_mode,
        },
        {"sampling": oversampling, "complex_grid": True, "support": support},
    )
    psf_stokes_i = image_corr_to_stokes(
        psf_da.values, corr_type=corr_type, pol_axis=2, stokes_out=["I"]
    )
    psf_xds = make_empty_sky_image(
        phase_center=phase_center,
        image_size=image_size,
        cell_size=cell_size,
        frequency_coords=freq_coords,
        pol_coords=["I"],
        time_coords=time_coords,
    )
    psf_xds["POINT_SPREAD_FUNCTION"] = xr.DataArray(
        psf_stokes_i,
        dims=["time", "frequency", "polarization", "l", "m"],
    )

    # Initialize model and tracking
    model_stokes = (
        initial_model.copy()
        if initial_model is not None
        else np.zeros([n_chan, n_stokes, ny, nx], dtype=float)
    )
    combined_return_dict = ReturnDict()
    dirty_xds = None

    # Helper to create sky image xds
    def _make_sky_xds(pol_coords_arg):
        return make_empty_sky_image(
            phase_center=phase_center,
            image_size=image_size,
            cell_size=cell_size,
            frequency_coords=freq_coords,
            pol_coords=pol_coords_arg,
            time_coords=time_coords,
        )

    major_cycle = 0
    while controller.stopcode.major == 0:
        major_cycle += 1
        column = "VISIBILITY" if major_cycle == 1 else "RESIDUAL"
        print(f"=== Major Cycle {major_cycle} === (gridding {column})")

        dirty_corr = grid_visibilities(
            ms4,
            image_size,
            cell_size,
            support=support,
            oversampling=oversampling,
            chan_mode=chan_mode,
            column=column,
        )
        dirty_stokes = image_corr_to_stokes(
            dirty_corr, corr_type=corr_type, pol_axis=1, stokes_out=stokes
        )

        # Create dirty_xds once, then update data on subsequent cycles
        if dirty_xds is None:
            dirty_xds = _make_sky_xds(stokes)
            dirty_xds["RESIDUAL"] = xr.DataArray(
                dirty_stokes[np.newaxis, ...],
                dims=["time", "frequency", "polarization", "l", "m"],
            )
        else:
            dirty_xds["RESIDUAL"].values[0, ...] = dirty_stokes

        # Calculate cycle controls
        if major_cycle == 1:
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

        # Create model xds if model is non-zero
        if np.any(model_stokes != 0):
            model_xds = _make_sky_xds(stokes)
            model_xds["MODEL"] = xr.DataArray(
                model_stokes[np.newaxis, ...],
                dims=["time", "frequency", "polarization", "l", "m"],
            )
        else:
            model_xds = None

        # Run deconvolution
        return_dict, new_model_xds, residual_xds = deconvolve(
            dirty_image_xds=dirty_xds,
            psf_xds=psf_xds,
            model_xds=model_xds,
            algorithm=algorithm,
            deconv_params={
                "gain": gain,
                "niter": cycle_niter,
                "threshold": cyclethresh,
            },
            output_dir=output_dir,
        )

        model_stokes = new_model_xds["MODEL"].values[0, ...]
        residual_stokes = residual_xds["RESIDUAL"].values[0, ...]

        peak_res = np.max(np.abs(residual_stokes))
        model_flux = np.sum(model_stokes[:, 0, :, :])
        print(
            f"  peak={peak_res:.6f} Jy, model_flux={model_flux:.6f} Jy, niter={cycle_niter}"
        )

        controller.update_counts(return_dict)
        combined_return_dict = merge_return_dicts([combined_return_dict, return_dict])

        stopcode, stopdesc = controller.check_convergence(return_dict)
        if stopcode.major != 0:
            print(f"  *** CONVERGED: {stopdesc} ***")
            break

        # Predict model visibilities for next cycle
        model_corr = image_stokes_to_corr(
            model_stokes, corr_type=corr_type, pol_axis=1, stokes_in=stokes
        )
        predict_model_visibilities(
            ms4,
            model_corr,
            cell_size,
            support=support,
            oversampling=oversampling,
        )
        compute_residual_visibilities(ms4)

    stokes_i_idx = stokes.index("I") if "I" in stokes else 0
    print(
        f"=== Done: {controller.major_done} major, {controller.total_iter_done} iter, "
        f"peak={np.max(np.abs(residual_stokes)):.6f} Jy, "
        f"flux={np.sum(model_stokes[:, stokes_i_idx, :, :]):.6f} Jy ==="
    )

    return model_stokes, residual_stokes, combined_return_dict, controller
