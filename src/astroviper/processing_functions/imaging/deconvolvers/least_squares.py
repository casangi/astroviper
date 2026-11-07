"""Least-squares solve of the sky model inside the clean mask."""

import numpy as np
import scipy.fft

from astroviper.processing_functions.image_analysis.make_auto_mask import _bounding_box
from astroviper.processing_functions.imaging.utils.cycle_control import (
    MODEL_UPDATE_ALGORITHM,
    MODEL_UPDATE_DIVERGENCE,
    MODEL_UPDATE_MAX_ITER_PER_CYCLE,
)

# A plane stops when the RMS of its residual outside the mask, which noise
# fitted inside the mask cannot lower, has fallen by less than STOP_MIN_FALL
# over the last STOP_WINDOW steps. Checked every STOP_EVERY steps.
STOP_EVERY = 25
STOP_WINDOW = 200
STOP_MIN_FALL = 0.05

# Energy below this share of the first step's is rounding, not residual.
ROUNDING = 1e-24


def _psf_spectrum(psf, height, width, workers):
    """Spectrum of ``psf`` for a linear convolution on a ``height x width`` box.

    ``psf`` has twice the image size with its peak at the centre, so it holds
    every offset between two pixels of the box.
    """
    row, column = (size // 2 for size in psf.shape)
    if not np.isclose(psf[row, column], 1.0, atol=1e-3):
        raise ValueError(f"The PSF peaks at {psf[row, column]} in its centre, not 1.")
    shape = (
        scipy.fft.next_fast_len(2 * height - 1),
        scipy.fft.next_fast_len(2 * width - 1, real=True),
    )
    kernel = np.zeros(shape)
    kernel[
        np.ix_(
            np.arange(1 - height, height) % shape[0],
            np.arange(1 - width, width) % shape[1],
        )
    ] = psf[row + 1 - height : row + height, column + 1 - width : column + width]
    return scipy.fft.rfft2(kernel, workers=workers), shape


def _solve_plane(residual, model, mask, psf, max_steps, workers, record):
    """Conjugate residual on ``M P M d = M R``, ``residual`` and ``model`` in place.

    Returns
    -------
    steps : int
    stop_code : int
    history : dict
        ``rms_outside`` of the residual in the box outside the mask, one value
        per check, and with ``record`` per step the ``peak`` and ``rms`` of the
        residual in the mask and the ``flux`` of the step.
    """
    box = _bounding_box(mask)
    inside = mask[box]
    outside = ~inside
    n_inside = np.count_nonzero(inside)
    height, width = inside.shape
    spectrum, shape = _psf_spectrum(psf, height, width, workers)

    def convolve(image):
        transform = scipy.fft.rfft(image, n=shape[1], axis=1, workers=workers)
        transform = scipy.fft.fft(
            transform, n=shape[0], axis=0, workers=workers, overwrite_x=True
        )
        transform *= spectrum
        transform = scipy.fft.ifft(transform, axis=0, workers=workers, overwrite_x=True)
        return scipy.fft.irfft(transform[:height], n=shape[1], axis=1, workers=workers)[
            :, :width
        ]

    current = residual[box].astype(np.float64)
    update = np.zeros_like(current)
    search = np.where(inside, current, 0.0)
    direction = search.copy()
    search_response = convolve(search)
    direction_response = search_response.copy()
    energy = first_energy = float(np.vdot(search, search_response))

    history = {"peak": [], "rms": [], "flux": [], "rms_outside": []}
    window = STOP_WINDOW // STOP_EVERY
    stop_code = MODEL_UPDATE_MAX_ITER_PER_CYCLE
    steps = 0
    while steps < max_steps:
        masked_response = direction_response[inside]
        curvature = float(np.vdot(masked_response, masked_response))
        if not (energy > ROUNDING * first_energy and curvature > 0):
            stop_code = (
                MODEL_UPDATE_ALGORITHM
                if np.isfinite(energy)
                else MODEL_UPDATE_DIVERGENCE
            )
            break
        length = energy / curvature
        if record:
            history["peak"].append(float(search.flat[np.abs(search).argmax()]))
            history["flux"].append(length * float(direction.sum()))
        update += length * direction
        current -= length * direction_response
        search = np.where(inside, current, 0.0)
        steps += 1
        if record:
            history["rms"].append(float(np.sqrt(np.vdot(search, search) / n_inside)))
        if steps % STOP_EVERY == 0 and outside.any():
            rms_outside = history["rms_outside"]
            rms_outside.append(float(current[outside].std()))
            if (
                len(rms_outside) > window
                and rms_outside[-1] > (1 - STOP_MIN_FALL) * rms_outside[-1 - window]
            ):
                stop_code = MODEL_UPDATE_ALGORITHM
                break
        if steps == max_steps:
            break
        search_response = convolve(search)
        previous, energy = energy, float(np.vdot(search, search_response))
        beta = energy / previous
        direction = search + beta * direction
        direction_response = search_response + beta * direction_response

    model[box] += update.astype(model.dtype)
    residual[box] = current.astype(residual.dtype)
    return steps, stop_code, history


def _iteration_history(history, steps):
    """Per step, as the deconvolvers report it: lists ``[t][f][p]`` of arrays.

    A step is no component: its pixel is -1 and its scale -2.
    """

    def nested(name, fill=0.0):
        nt, nf, npol = steps.shape
        return [
            [
                [
                    np.asarray(history[t, f, p][name])
                    if (t, f, p) in history and name in history[t, f, p]
                    else np.full(steps[t, f, p], fill)
                    for p in range(npol)
                ]
                for f in range(nf)
            ]
            for t in range(nt)
        ]

    return {
        "iteration_peak_residual": nested("peak"),
        "iteration_residual_rms": nested("rms"),
        "iteration_component_flux": nested("flux"),
        "iteration_component_l": nested("l", -1.0),
        "iteration_component_m": nested("m", -1.0),
        "iteration_component_scale": nested("scale", -2.0),
    }


def least_squares_solve(
    residual_cube,
    psf_cube,
    model_cube,
    deconvolve_params,
    processing_function_threads=1,
    mask_cube=None,
    *,
    input_checked=False,
):
    """Solve the model inside the clean mask, plane by plane, in place.

    Parameters
    ----------
    residual_cube : numpy.ndarray
        ``(nt, nf, np, ny, nx)`` residual, updated in the bounding box of each
        plane's mask.
    psf_cube : numpy.ndarray
        ``(nt, nf, np_psf, 2 ny, 2 nx)`` PSF of twice the image size.
    model_cube : numpy.ndarray
        ``(nt, nf, np, ny, nx)`` model, to which the solution is added.
    deconvolve_params : dict
        ``least_squares_params``, and ``record_iteration_history``.
    processing_function_threads : int, optional
        FFT workers.
    mask_cube : numpy.ndarray
        ``(nt, nf, np, ny, nx)`` clean mask. A plane without a mask pixel, with
        a residual that is zero or not finite in its mask, is left as it is.
    input_checked : bool, optional
        ``True`` when a caller has checked ``least_squares_params``.

    Returns
    -------
    dict
        ``iterations_performed`` and ``stop_code`` per plane, the component
        counts, ``least_squares_rms_outside_mask``, and the iteration history
        when it is asked for.
    """
    least_squares_params = deconvolve_params["least_squares_params"]
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check._predicates import (
            check_least_squares_params,
        )

        check_least_squares_params(least_squares_params)
    record = deconvolve_params.get("record_iteration_history", False)
    planes = residual_cube.shape[:3]
    steps = np.zeros(planes, dtype=int)
    stop_code = np.zeros(planes, dtype=int)
    history = {}
    for index in np.ndindex(planes):
        mask = mask_cube[index].astype(bool, copy=False)
        values = residual_cube[index][mask]
        if not values.any() or not np.isfinite(values).all():
            continue
        psf = psf_cube[index[:2] + (index[2] if psf_cube.shape[2] > 1 else 0,)]
        steps[index], stop_code[index], history[index] = _solve_plane(
            residual_cube[index],
            model_cube[index],
            mask,
            psf,
            least_squares_params["max_steps"],
            processing_function_threads,
            record,
        )
    results = {
        "iterations_performed": steps,
        "stop_code": stop_code,
        "n_point_components": np.zeros_like(steps),
        "n_scale_components": np.zeros_like(steps),
        "least_squares_rms_outside_mask": {
            index: fields["rms_outside"] for index, fields in history.items()
        },
    }
    if record:
        results.update(_iteration_history(history, steps))
    return results
