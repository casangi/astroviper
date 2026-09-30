"""CASA-compatible PSF Gaussian beam fit (C++, port of CAS-13022).

Wraps the ``_psf_gaussian_fit_ext`` pybind11 module, a port of casa6
``StokesImageUtil::FitGaussianPSF`` -- the fit behind ``tclean``'s restoring
beam -- selected in the imaging chain with ``psf_fitting_method="casa"``.
"""

from __future__ import annotations

import numpy as np


def casa_psf_fit_available() -> bool:
    """Whether the C++ extension was built and can be imported."""
    try:
        from astroviper.processing_functions.image_analysis.psf_gaussian_fit_cpp import (  # noqa: F401
            _psf_gaussian_fit_ext,
        )
    except ImportError:
        return False
    return True


def fit_psf_beam(
    point_spread_function: np.ndarray,
    delta: np.ndarray,
    psfcutoff: float = 0.35,
    processing_function_threads: int = 1,
) -> np.ndarray:
    """Fit every PSF plane with the CASA algorithm.

    Parameters
    ----------
    point_spread_function : np.ndarray, [time, frequency, polarization, l, m]
        PSF cube (float32 or float64, C-contiguous).
    delta : np.ndarray, [2], radians
        (l, m) pixel increments (signs are ignored, as in CASA).
    psfcutoff : float
        Main-lobe cutoff as a fraction of the peak (CASA ``psfcutoff``,
        default 0.35).
    processing_function_threads : int
        Planes are partitioned over this many C++ threads.

    Returns
    -------
    np.ndarray, [time, frequency, polarization, 3] float64
        ``[major FWHM, minor FWHM, position angle]`` per plane, radians, in the
        same pixel-frame position-angle convention as astroviper's
        ``BEAM_FIT_PARAMS`` (casacore ``Gaussian2D``, mod pi).
    """
    from astroviper.processing_functions.image_analysis.psf_gaussian_fit_cpp import (
        _psf_gaussian_fit_ext,
    )

    cube = np.ascontiguousarray(point_spread_function)
    n_time, n_frequency, n_polarization, nx, ny = cube.shape
    planes = cube.reshape(n_time * n_frequency * n_polarization, nx, ny)
    beams = np.empty((planes.shape[0], 3), dtype=np.float64)
    _psf_gaussian_fit_ext.fit_psf_beam(
        planes,
        float(abs(delta[0])),
        float(abs(delta[1])),
        float(psfcutoff),
        beams,
        int(processing_function_threads),
    )
    return beams.reshape(n_time, n_frequency, n_polarization, 3)
