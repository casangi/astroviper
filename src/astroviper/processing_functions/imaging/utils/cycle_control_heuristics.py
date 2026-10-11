"""Cycle control parameters for ALMA single field cubes from the observables of a first run.

A model update runs until the peak residual of a plane falls to its
``threshold_per_cycle``, the PSF fraction
``f = clamp(psf_sidelobe_factor * max_psf_sidelobe, min_psf_fraction, max_psf_fraction)``
of the peak it started from, so the imaging cycles needed to bring the peak
residual ``P0`` of the dirty image down to a level ``L`` are about
``ceil(ln(P0 / L) / ln(1 / f))`` (:func:`predicted_imaging_cycles`). The total
number of iterations is set by the stopping level and the gain, not by how they
are split into cycles, and every imaging cycle costs one residual update, so the
fastest controls run each model update to its ``threshold_per_cycle`` without a
cap and with a small PSF fraction. The values below were measured on simulated
ALMA data (C8 configuration at 950 GHz, point sources, dynamic range 1e3 to 1e5);
see the RADPS memo on the imaging accuracy of the single field imager.

The observables come from a first run with ``max_iter = 0``: the recorded
``max_psf_sidelobe`` and the peak residual of the dirty image of a channel.
"""

from __future__ import annotations

import math

import numpy as np

PSF_SIDELOBE_FACTOR = 0.5
MIN_PSF_FRACTION = 0.05
MAX_PSF_FRACTION = 0.8
EXTRA_CYCLES = 3
NOISY_MAX_ITER_PER_CYCLE = 100
NOISY_CYCLE_SAFETY = 3


def psf_fraction(max_psf_sidelobe, psf_sidelobe_factor=PSF_SIDELOBE_FACTOR):
    """The clamped PSF fraction of ``threshold_per_cycle`` for a recorded PSF sidelobe."""
    return min(
        max(psf_sidelobe_factor * max_psf_sidelobe, MIN_PSF_FRACTION), MAX_PSF_FRACTION
    )


def predicted_imaging_cycles(peak_residual, reference_level, psf_fraction):
    """Imaging cycles that bring the peak residual down to ``reference_level``.

    Parameters
    ----------
    peak_residual : float
        Peak residual of the dirty image of the plane, ``P0`` (Jy).
    reference_level : float
        The stopping threshold for noise free data, the thermal noise for
        noisy data (Jy).
    psf_fraction : float
        The PSF fraction ``f`` of ``threshold_per_cycle``, in ``(0, 1)``.

    Returns
    -------
    int
        ``ceil(ln(P0 / reference_level) / ln(1 / f))``, at least 1.

    Raises
    ------
    ValueError
        For a non positive peak or level, or ``f`` outside ``(0, 1)``.
    """
    if not (peak_residual > 0.0 and reference_level > 0.0):
        raise ValueError(
            "peak_residual and reference_level must be positive; got "
            f"{peak_residual!r}, {reference_level!r}"
        )
    if not 0.0 < psf_fraction < 1.0:
        raise ValueError(f"psf_fraction must lie in (0, 1); got {psf_fraction!r}")
    if peak_residual <= reference_level:
        return 1
    return max(
        1,
        int(
            math.ceil(
                math.log(peak_residual / reference_level) / math.log(1.0 / psf_fraction)
            )
        ),
    )


def alma_cycle_control_params(
    max_psf_sidelobe,
    peak_residual,
    threshold,
    *,
    gain=0.1,
    noisy=False,
    thermal_noise=None,
    max_iter=200000,
    primary_beam_limit=0.2,
):
    """``cycle_control_params`` for an ALMA single field cube.

    Parameters
    ----------
    max_psf_sidelobe : float
        The ``max_psf_sidelobe`` a ``max_iter = 0`` run records for the channel.
    peak_residual : float
        Peak residual of the dirty image of the channel (Jy), Stokes I.
    threshold : float or sequence of float
        Stopping threshold of a noise free run (Jy), a little above the
        numerical floor of the imaging cycle (for example 1.2 times the peak
        of the residual the imager leaves for the true model), one value or
        one per Stokes plane (the first, Stokes I, sets the predicted
        cycles); not used when ``noisy`` (the entropy stop ends the clean at
        the noise).
    gain : float, optional
        CLEAN loop gain; 0.2 halves the iterations, with a smaller margin of
        the divergence test.
    noisy : bool, optional
        Data dominated by thermal noise: threshold 0, the entropy stop and a
        cap on the iterations of a model update.
    thermal_noise : float, optional
        The thermal noise of a Stokes I plane (Jy); required when ``noisy``.
    max_iter : int, optional
        Total iteration budget of a plane.
    primary_beam_limit : float, optional
        Primary beam mask cutoff.

    Returns
    -------
    dict
        Every key of ``cycle_control_params`` with Python numbers.

    Raises
    ------
    ValueError
        For ``noisy`` without a positive ``thermal_noise``, or an invalid
        observable (see :func:`predicted_imaging_cycles`).
    """
    fraction = psf_fraction(max_psf_sidelobe)
    params = {
        "max_iter": int(max_iter),
        "primary_beam_limit": float(primary_beam_limit),
        "gain": float(gain),
        "psf_sidelobe_factor": PSF_SIDELOBE_FACTOR,
        "min_psf_fraction": MIN_PSF_FRACTION,
        "max_psf_fraction": MAX_PSF_FRACTION,
        "max_iter_divergence": 1,
    }
    if noisy:
        if thermal_noise is None or not thermal_noise > 0.0:
            raise ValueError("noisy=True needs a positive thermal_noise")
        cycles = predicted_imaging_cycles(peak_residual, thermal_noise, fraction)
        params.update(
            {
                "threshold": 0.0,
                "max_cycles": NOISY_CYCLE_SAFETY * cycles,
                "max_iter_per_cycle": NOISY_MAX_ITER_PER_CYCLE,
                "entropy_stop": True,
            }
        )
    else:
        thresholds = [float(value) for value in np.atleast_1d(threshold)]
        cycles = predicted_imaging_cycles(peak_residual, thresholds[0], fraction)
        params.update(
            {
                "threshold": thresholds[0] if np.ndim(threshold) == 0 else thresholds,
                "max_cycles": cycles + EXTRA_CYCLES,
                "max_iter_per_cycle": -1,
                "entropy_stop": False,
            }
        )
    return params
