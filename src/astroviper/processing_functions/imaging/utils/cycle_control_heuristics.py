"""Cycle control parameters for single field cubes from the observables of a first run.

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
ALMA data (the C8 configuration at 950 GHz of
``tests/component/test_point_source_flux_recovery.py``, point sources, dynamic
range 1e3 to 1e5); their use for other telescopes (the VLA in
:func:`~astroviper.distributed_applications.imaging.image_cube_single_field_delphi.image_cube_single_field_delphi`)
is assumed, not validated.

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
    """The clamped PSF fraction of ``threshold_per_cycle`` for a recorded PSF sidelobe.

    Parameters
    ----------
    max_psf_sidelobe : float
        The ``max_psf_sidelobe`` a ``max_iter = 0`` run records, in units of
        the PSF peak.
    psf_sidelobe_factor : float, optional
        Factor on the sidelobe.

    Returns
    -------
    float
        ``psf_sidelobe_factor * max_psf_sidelobe`` clamped to
        ``[MIN_PSF_FRACTION, MAX_PSF_FRACTION]``.
    """
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


def noisy_cycle_prediction(point_spread_function_xds, n_sigma, max_dynamic_range):
    """PSF fraction and predicted imaging cycles of :func:`noisy_cycle_control_params`.

    Parameters
    ----------
    point_spread_function_xds : xarray.Dataset
        The result of the PSF pass, with ``MAX_SIDELOBE_POINT_SPREAD_FUNCTION``.
    n_sigma : float
        Threshold in units of the sensitivity of the plane.
    max_dynamic_range : float
        Largest ratio of the peak residual of a dirty plane to its sensitivity
        the controls allow for.

    Returns
    -------
    tuple of (float, int)
        The PSF fraction of the largest ``max_psf_sidelobe`` of the cube and
        the imaging cycles that bring a peak ``max_dynamic_range`` times the
        sensitivity down to ``n_sigma`` times it.

    Raises
    ------
    ValueError
        For a non positive ``n_sigma`` or ``max_dynamic_range``.
    """
    fraction = psf_fraction(
        float(
            np.nanmax(point_spread_function_xds["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"])
        )
    )
    return fraction, predicted_imaging_cycles(max_dynamic_range, n_sigma, fraction)


def noisy_cycle_control_params(
    point_spread_function_xds,
    *,
    n_sigma=2.0,
    max_dynamic_range=1e5,
    gain=0.1,
    max_iter=200000,
    primary_beam_limit=0.2,
):
    """``cycle_control_params`` for noisy data from the PSF pass of the cube.

    The threshold of every plane is ``n_sigma`` times its theoretical point
    source sensitivity, the entropy stop is on and every model update is
    capped at ``NOISY_MAX_ITER_PER_CYCLE`` iterations (the noisy controls of
    :func:`alma_cycle_control_params`). ``max_cycles`` is the number of
    imaging cycles that bring a peak
    ``max_dynamic_range`` times the sensitivity down to ``n_sigma`` times it
    (:func:`predicted_imaging_cycles`) with the PSF fraction of the largest
    ``max_psf_sidelobe`` of the cube, plus ``EXTRA_CYCLES``.

    Parameters
    ----------
    point_spread_function_xds : xarray.Dataset
        The result of
        :func:`~astroviper.distributed_applications.imaging.point_spread_function_cube_single_field.point_spread_function_cube_single_field`:
        ``SENSITIVITY`` and ``MAX_SIDELOBE_POINT_SPREAD_FUNCTION`` with dims
        ``(frequency, polarization)``.
    n_sigma : float, optional
        Threshold in units of the sensitivity of the plane.
    max_dynamic_range : float, optional
        Largest ratio of the peak residual of a dirty plane to its sensitivity
        the controls allow for.
    gain : float, optional
        CLEAN loop gain.
    max_iter : int, optional
        Total iteration budget of a plane.
    primary_beam_limit : float, optional
        Primary beam mask cutoff.

    Returns
    -------
    dict
        Every key of ``cycle_control_params`` with Python numbers; ``threshold``
        is a per channel threshold, a nested list of shape
        ``(n_frequency, n_polarization)`` in the order of the dataset's
        ``frequency`` and ``polarization`` coordinates.

    Raises
    ------
    ValueError
        For a non positive ``n_sigma`` or ``max_dynamic_range``.
    """
    sensitivity = point_spread_function_xds["SENSITIVITY"].transpose(
        "frequency", "polarization"
    )
    _, cycles = noisy_cycle_prediction(
        point_spread_function_xds, n_sigma, max_dynamic_range
    )
    return {
        "max_iter": int(max_iter),
        "max_cycles": cycles + EXTRA_CYCLES,
        "threshold": np.nan_to_num(
            n_sigma * np.asarray(sensitivity, dtype=float), posinf=0.0
        ).tolist(),
        "gain": float(gain),
        "psf_sidelobe_factor": PSF_SIDELOBE_FACTOR,
        "min_psf_fraction": MIN_PSF_FRACTION,
        "max_psf_fraction": MAX_PSF_FRACTION,
        "max_iter_per_cycle": NOISY_MAX_ITER_PER_CYCLE,
        "max_iter_divergence": 1,
        "primary_beam_limit": float(primary_beam_limit),
        "entropy_stop": True,
    }
