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
cap and with a small PSF fraction. ``PSF_SIDELOBE_FACTOR`` to
``NOISY_CYCLE_SAFETY``, the constants of :func:`alma_cycle_control_params` and
of ``max_cycles="auto"``, were measured on simulated ALMA data (the C8
configuration at 950 GHz of
``tests/component/test_point_source_flux_recovery.py``, point sources, dynamic
range 1e3 to 1e5) at ``threshold_tolerance`` 0. ``NOISY_GAIN`` to
``THRESHOLD_TOLERANCE``, the controls of :func:`noisy_cycle_control_params` that
:func:`~astroviper.distributed_applications.imaging.image_cube_single_field_delphi.image_cube_single_field_delphi`
uses, were measured at ``NOISY_N_SIGMA``, an assumed choice, on simulated noisy
ALMA 12 m, VLA and ngVLA core cubes (RADPS Memo 13; scope in
:func:`noisy_cycle_control_params`).

The observables of :func:`alma_cycle_control_params` and ``max_cycles="auto"``
come from a first run with ``max_iter = 0``: the recorded ``max_psf_sidelobe``
and the peak residual of the dirty image of a channel. Those of
:func:`noisy_cycle_control_params` are the ``SENSITIVITY`` of the PSF pass per
channel and Stokes plane; the imager forms the PSF fraction of each plane from
its ``max_psf_sidelobe``.
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
# Assumed (a science choice); the other noisy controls were measured at 3.
NOISY_N_SIGMA = 3.0
# Measured against 0.2: 13 to 33 percent fewer iterations in 10 of 10 runs, an
# extra point bias of -0.11 to -0.15 sigma on 4 of 10; 0.4 and 0.5 fail the
# accuracy test on the VLA D (single runs at the gate margin).
NOISY_GAIN = 0.3
# Measured: 0.5 fails the accuracy test on ALMA C-3 and the VLA D and passes on
# ALMA C-8, the VLA C line skies and the ngVLA core; 1.5 passes every gated held
# out line sky. 1.5 is the floor of an in-sample guard on the PSF level beyond the
# Hogbom window (ALMA C-3 within 2 percent of it).
NOISY_PSF_SIDELOBE_FACTOR = 1.5
# A safety limit (measured: at most 31 model updates in any uncapped run,
# max_psf_sidelobe 0.034 to 0.20). At the clamp MAX_PSF_FRACTION a dynamic range of
# 1e5 / 3 needs 47 imaging cycles (derived): a plane stopped by NOISY_MAX_CYCLES is
# not validated.
NOISY_MAX_CYCLES = 40
# Measured against 0 on 30 noisy run pairs (RADPS Memo 13; Hogbom at 3 sigma, gain
# 0.2 and 0.3, exact_psf_subtraction off): cost lower in 29, accuracy verdicts
# unchanged in 30; the last fresh masked peak at most 1.0197 times the threshold;
# point effects isolated on 5 of 10 stores. With exact_psf_subtraction run in one
# configuration only (accuracy test passed on 14 of 14 judged skies). Other
# thresholds and noise free data: cost replayed from records only; Asp and
# max_cycles="auto": not run.
THRESHOLD_TOLERANCE = 0.02


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


def noisy_cycle_control_params(
    point_spread_function_xds,
    *,
    n_sigma=NOISY_N_SIGMA,
    gain=NOISY_GAIN,
    max_iter=200000,
    primary_beam_limit=0.2,
):
    """``cycle_control_params`` for noisy data from the PSF pass of the cube.

    * ``threshold``: ``n_sigma`` times the theoretical point source
      sensitivity of every channel and Stokes plane, 0 for a plane without
      data (infinite sensitivity). The sensitivity is a noise level only when
      ``WEIGHT`` holds absolute ``1 / sigma**2`` weights.
    * ``threshold_tolerance``: ``THRESHOLD_TOLERANCE``; a plane stops once the
      peak of its fresh residual inside the clean mask is at most
      ``threshold * (1 + THRESHOLD_TOLERANCE)``.
    * ``psf_sidelobe_factor``: ``NOISY_PSF_SIDELOBE_FACTOR``; the imager
      makes the PSF fraction of every plane from its ``max_psf_sidelobe``.
    * ``max_cycles``: ``NOISY_MAX_CYCLES``, a safety limit; the threshold
      stops the imaging.
    * ``max_iter_per_cycle`` -1, no cap: a cap of 100 did not converge on 4
      of the 10 training skies and cost 1.4 to 4.5 times as much, a cap of
      1000 did not converge on the bright ngVLA core sky, and with a cap the
      iterations follow the run to run noise of the residual updates
      (measured).
    * ``entropy_stop`` False: at 3 sigma the entropy stop had no material
      effect on accuracy and cost 14 to 20 percent more on ALMA C-3 and the
      VLA D, 6 to 13 percent less on the bright ALMA C-8 sky and the ngVLA
      core (measured); untested with ``threshold_tolerance``.
    * ``exact_psf_subtraction`` False, the code default (a decision; measured,
      the option fails the paired accuracy test on the bright ALMA C-3 sky
      while closer to the truth, worsens the Gaussian on the VLA C line skies
      and raises the peak memory on ALMA C-8 from about 4 to 7.3 GiB).

    Measured (RADPS Memo 13; Hogbom) on simulated noisy cubes of ALMA C-3
    (100 GHz) and C-8 (950 GHz), the VLA in D (45 GHz) and C (33 GHz) and the
    ngVLA core (41 GHz, 10 h), with ``max_psf_sidelobe`` 0.034 to 0.20: the
    training skies (seed 1001; the bright VLA C sky is limited by its w term
    floor and not judged, the bright VLA D and ngVLA core skies are judged
    only with a floor allowance above the threshold) and five held out line
    skies (seed 2001), one run each. Accuracy was judged on the Stokes I
    point sources; the flux of the Gaussian source was reported, not judged
    (errors of -33 to +7 percent). The held out bright skies converged but
    are off grid limited and not judged; the shifted training layout (seed
    1201) and the ngVLA core plus spiral were not run.

    Parameters
    ----------
    point_spread_function_xds : xarray.Dataset
        The result of
        :func:`~astroviper.distributed_applications.imaging.point_spread_function_cube_single_field.point_spread_function_cube_single_field`:
        ``SENSITIVITY`` with dims ``(frequency, polarization)``.
    n_sigma : float, optional
        Threshold in units of the sensitivity of the plane.
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
        For a non positive ``n_sigma``.
    """
    if not n_sigma > 0.0:
        raise ValueError(f"n_sigma must be positive; got {n_sigma!r}")
    sensitivity = point_spread_function_xds["SENSITIVITY"].transpose(
        "frequency", "polarization"
    )
    return {
        "max_iter": int(max_iter),
        "max_cycles": NOISY_MAX_CYCLES,
        "threshold": np.nan_to_num(
            n_sigma * np.asarray(sensitivity, dtype=float), posinf=0.0
        ).tolist(),
        "threshold_tolerance": THRESHOLD_TOLERANCE,
        "gain": float(gain),
        "psf_sidelobe_factor": NOISY_PSF_SIDELOBE_FACTOR,
        "min_psf_fraction": MIN_PSF_FRACTION,
        "max_psf_fraction": MAX_PSF_FRACTION,
        "max_iter_per_cycle": -1,
        "max_iter_divergence": 1,
        "primary_beam_limit": float(primary_beam_limit),
        "entropy_stop": False,
        "exact_psf_subtraction": False,
    }
