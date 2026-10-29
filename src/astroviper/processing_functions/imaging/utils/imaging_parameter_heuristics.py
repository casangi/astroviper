"""Image geometry and polarization of a single field cube from the processing set metadata.

Pure functions behind
:func:`~astroviper.distributed_applications.imaging.image_cube_single_field_delphi.image_cube_single_field_delphi`:

- the primary beam model of the telescope (:func:`primary_beam_telescope`) and
  the CASA pipeline ``robust`` of a cube (:func:`default_robust`);
- the instrument polarization basis and the Stokes planes the strict
  polarization modes of the imager make from the correlations of the data
  (:func:`instrument_polarization_basis_from_correlations`,
  :func:`stokes_for_correlations`);
- the field of view, ``FIELD_OF_VIEW_MARGIN`` times the diameter at which the
  primary beam the imager uses falls to a given level at the lowest frequency
  (:func:`primary_beam_radius`, :func:`field_of_view`);
- a first cell, ``1 / (3 u_max)`` with ``u_max`` the longest antenna
  separation at the highest frequency, so that no sample falls off the uv grid
  of the PSF pass (:func:`cell_size_guess`), the final cell from the fitted
  beam (:func:`cell_size_from_beam`, :func:`cell_size_from_beam_fit`) and the
  largest cell whose uv grid holds those samples
  (:func:`max_cell_size_for_uv_coverage`);
- the image size, the next even 2, 3, 5 smooth size that covers the field of
  view (:func:`image_size_for_field_of_view`);
- the gridding kernel and FFT padding (``GRIDDING_KERNEL_PARAMS``,
  ``FFT_PADDING``).
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.imaging.utils.imaging_polarization import (
    CORRELATIONS,
    STOKES,
)

SPEED_OF_LIGHT = 299792458.0
# Assumed.
FIELD_OF_VIEW_MARGIN = 1.1
# Derived: at least 2 puts every baseline on the uv grid of the PSF pass; 3 adds an
# assumed margin.
CELL_GUESS_SAMPLES_PER_U_MAX = 3.0
# Between the 7 m (ACA) and 12 m antennas of ALMA (metres).
ACA_DISH_DIAMETER_LIMIT = 9.5
# CASA pipeline robust of a cube.
DEFAULT_ROBUST = {"alma": 0.5, "aca": 0.5, "vla": 2.0}
# The dynamic range rule of RADPS Memo 13a: kernel errors at most 0.1 sigma at a
# dynamic range of 1e4, i.e. a far floor (the residual RMS away from the sources after
# one residual update at the true model, w term off, over the Stokes I peak) of at most
# 1e-5, worst case over its ALMA C-8, ALMA C-1 and 7 m array skies, a source anywhere
# from 0.75 of the half width kept to the 0.2 primary beam radius, and every Stokes
# plane. Measured worst case 2.7e-7 (the 7 m array, a source at 0.85), a margin of 36
# (35 between pixel centres and out to the clean mask edge, predicted); prolate
# spheroidal 9/300/linear 5.7e-7, least misfit 7/300/linear 6.1e-6, CASA's kernel
# 1.3e-4. Cost order: grid memory (fft_padding 1.2), then time (support 9 costs 51
# percent more than 7, which leaves too little margin for ten sources near the edge,
# derived). It replaces requirement (a) of RADPS Memo 13 (the kernel floor at most a
# third of the w term floor: 0.002 sigma at 1e4 on the 7 m array, met only at
# fft_padding 1.5; this kernel reaches 0.45).
GRIDDING_KERNEL_PARAMS = {
    "type": "least_misfit",
    "support": 9,
    "oversampling": 300,
    "interpolation": "linear",
}
FFT_PADDING = 1.2


def primary_beam_telescope(telescope_name, dish_diameters=None):
    """``primary_beam_params["telescope"]`` of an observatory.

    Parameters
    ----------
    telescope_name : str
        ``observation_info["telescope_name"]`` of the processing set.
    dish_diameters : array_like of float, optional
        Dish diameters of the antennas (metres); for ALMA the smallest sets
        the primary beam (as the CASA pipeline does for mixed arrays): below
        ``ACA_DISH_DIAMETER_LIMIT`` it is ``"aca"``, otherwise ``"alma"``.

    Returns
    -------
    str
        ``"alma"``, ``"aca"``, ``"vla"`` or ``"ngvla"``.

    Raises
    ------
    ValueError
        For an observatory without a primary beam model.
    """
    name = str(telescope_name).strip().upper()
    if name in ("ALMA", "ACA"):
        diameters = np.asarray(
            [] if dish_diameters is None else dish_diameters, dtype=float
        )
        diameters = diameters[np.isfinite(diameters) & (diameters > 0.0)]
        if name == "ACA" or (
            diameters.size and diameters.min() < ACA_DISH_DIAMETER_LIMIT
        ):
            return "aca"
        return "alma"
    if name in ("VLA", "EVLA"):
        return "vla"
    if name == "NGVLA":
        return "ngvla"
    raise ValueError(
        f"no primary beam model for telescope {telescope_name!r}; expected ALMA, "
        "ACA, VLA, EVLA or ngVLA"
    )


def default_robust(telescope):
    """The CASA pipeline Briggs ``robust`` of a cube: 0.5 (ALMA, ACA), 2.0 (VLA).

    Parameters
    ----------
    telescope : str
        ``primary_beam_params["telescope"]``, as :func:`primary_beam_telescope`
        returns it.

    Returns
    -------
    float
        The Briggs ``robust`` parameter.

    Raises
    ------
    ValueError
        For a telescope without a pipeline value (pass ``robust``).
    """
    if telescope not in DEFAULT_ROBUST:
        raise ValueError(
            f"no default robust for telescope {telescope!r}; pass robust explicitly"
        )
    return DEFAULT_ROBUST[telescope]


def instrument_polarization_basis_from_correlations(correlations):
    """``"linear"`` for ``XX, XY, YX, YY`` correlations, ``"circular"`` for ``RR, RL, LR, LL``.

    Parameters
    ----------
    correlations : str or sequence of str
        Correlation labels of the data.

    Returns
    -------
    str
        ``"linear"`` or ``"circular"``.

    Raises
    ------
    ValueError
        For correlations of both or of neither basis.
    """
    labels = {str(label) for label in np.atleast_1d(correlations)}
    bases = [
        basis for basis, by_count in CORRELATIONS.items() if labels <= set(by_count[4])
    ]
    if len(bases) != 1:
        raise ValueError(
            f"correlations {sorted(labels)} are not all linear or all circular"
        )
    return bases[0]


def stokes_for_correlations(correlations):
    """Stokes planes the imager makes from the correlations of the data.

    All four correlations give ``I, Q, U, V``; the two parallel hands give
    ``I, Q`` (linear) or ``I, V`` (circular), see
    :mod:`~astroviper.processing_functions.imaging.utils.imaging_polarization`.

    Parameters
    ----------
    correlations : str or sequence of str
        Correlation labels of the data.

    Returns
    -------
    tuple of (str, list of str)
        The instrument polarization basis and the Stokes planes.

    Raises
    ------
    ValueError
        When the parallel hands are not both present.
    """
    basis = instrument_polarization_basis_from_correlations(correlations)
    labels = {str(label) for label in np.atleast_1d(correlations)}
    for n_correlation in (4, 2):
        if set(CORRELATIONS[basis][n_correlation]) <= labels:
            return basis, list(STOKES[basis][n_correlation])
    raise ValueError(
        f"correlations {sorted(labels)} do not hold both parallel hands "
        f"{CORRELATIONS[basis][2]}"
    )


def primary_beam_radius(frequency, level, primary_beam_params=None, n_scan=4096):
    """Radius at which the imager's primary beam (power) falls to ``level``.

    The beam is the one the imager makes for ``primary_beam_params``
    (:func:`~astroviper.processing_functions.imaging.primary_beam.make_primary_beam.resolve_primary_beam_params`
    and the power pattern of
    :func:`~astroviper.processing_functions.simulation.antenna_beams.symmetric_beam_response`,
    zero beyond ``max_rad_1GHz / (frequency / 1 GHz)``, if any), and the radius is the
    first crossing of the main lobe, found on a scan out to ``1.5 lambda / D``
    for an Airy disk (else to just past ``max_rad_1GHz / (frequency / 1 GHz)``)
    and refined with Brent's method.

    Parameters
    ----------
    frequency : float
        Frequency (Hz).
    level : float
        Primary beam power level, in ``(0, 1)``.
    primary_beam_params : dict, optional
        As ``primary_beam_params`` of the imager; ``None`` is ALMA 12 m (the
        imager's default beam).
    n_scan : int, optional
        Samples of the scan.

    Returns
    -------
    float
        Radius (radians).

    Raises
    ------
    ValueError
        For a level outside ``(0, 1)`` or below the beam on the whole scan.
    """
    from scipy.optimize import brentq

    from astroviper.processing_functions.imaging.primary_beam.make_primary_beam import (
        pack_primary_beam_model,
        resolve_primary_beam_params,
    )
    from astroviper.processing_functions.simulation.antenna_beams import (
        symmetric_beam_response,
    )

    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1); got {level!r}")
    resolved = resolve_primary_beam_params(
        {"telescope": "alma"} if primary_beam_params is None else primary_beam_params
    )
    packed_model = pack_primary_beam_model(resolved["beam_model"], float(frequency))
    if packed_model["kind"] == "analytic" and packed_model["func"] != "none":
        scan_radius = (
            1.5 * SPEED_OF_LIGHT / float(frequency) / packed_model["dish_diameter"]
        )
    else:
        # end just past the cut, where the beam is zero
        scan_radius = np.nextafter(
            packed_model["max_rad_1GHz"] / (float(frequency) / 1e9), np.inf
        )

    def excess(radius):
        return (
            float(
                symmetric_beam_response(
                    packed_model, radius, 0.0, float(frequency), ipower=2
                )
            )
            - level
        )

    radii = np.linspace(0.0, scan_radius, n_scan)
    below = np.flatnonzero([excess(radius) < 0.0 for radius in radii])
    if below.size == 0:
        raise ValueError(
            f"the primary beam does not fall to {level} within 1.5 lambda/D"
        )
    return float(brentq(excess, radii[below[0] - 1], radii[below[0]]))


def field_of_view(frequency_min, level, primary_beam_params=None):
    """``FIELD_OF_VIEW_MARGIN`` times the diameter where the primary beam falls to ``level``.

    Parameters
    ----------
    frequency_min : float
        Lowest frequency of the cube (Hz), where the primary beam is widest.
    level : float
        Primary beam power level, in ``(0, 1)``.
    primary_beam_params : dict, optional
        As ``primary_beam_params`` of the imager; ``None`` is ALMA 12 m.

    Returns
    -------
    float
        Field of view (radians).

    Raises
    ------
    ValueError
        As :func:`primary_beam_radius`.
    """
    return (
        FIELD_OF_VIEW_MARGIN
        * 2.0
        * primary_beam_radius(frequency_min, level, primary_beam_params)
    )


def longest_antenna_separation(antenna_positions):
    """Largest distance between two antennas.

    Parameters
    ----------
    antenna_positions : array_like of float
        ``(n_antenna, 3)`` antenna positions (metres), as ``ANTENNA_POSITION``.

    Returns
    -------
    float
        Largest separation (metres).
    """
    positions = np.asarray(antenna_positions, dtype=float).reshape(-1, 3)
    separations = positions[:, None, :] - positions[None, :, :]
    return float(np.sqrt((separations**2).sum(axis=-1)).max())


def cell_size_guess(max_antenna_separation, frequency_max):
    """First cell, ``1 / (3 u_max)``, with ``u_max`` in wavelengths at ``frequency_max``.

    Every projected baseline is at most the antenna separation, so the uv grid,
    which spans ``1 / cell = 3 u_max``, holds every sample of the PSF pass.

    Parameters
    ----------
    max_antenna_separation : float
        Longest antenna separation (metres), see
        :func:`longest_antenna_separation`.
    frequency_max : float
        Highest frequency of the cube (Hz).

    Returns
    -------
    float
        Cell size (radians).
    """
    u_max = float(max_antenna_separation) * float(frequency_max) / SPEED_OF_LIGHT
    return 1.0 / (CELL_GUESS_SAMPLES_PER_U_MAX * u_max)


def image_size_for_field_of_view(field_of_view, cell_size):
    """The next even 2, 3, 5 smooth image size that covers ``field_of_view`` at ``cell_size``.

    Parameters
    ----------
    field_of_view : float
        Field of view (radians).
    cell_size : float
        Cell size (radians).

    Returns
    -------
    int
        Image size (pixels) along each axis.
    """
    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        next_fft_friendly_size,
    )

    return next_fft_friendly_size(
        int(np.ceil(float(field_of_view) / float(cell_size))), even=True
    )


def cell_size_from_beam(minor_axis, pixels_per_beam):
    """Cell that puts ``pixels_per_beam`` pixels across the fitted minor axis.

    Parameters
    ----------
    minor_axis : float
        Minor axis of the fitted beam (radians).
    pixels_per_beam : float
        Pixels across the minor axis.

    Returns
    -------
    float
        Cell size (radians).

    Raises
    ------
    ValueError
        For a non positive ``pixels_per_beam``.
    """
    if not pixels_per_beam > 0.0:
        raise ValueError(f"pixels_per_beam must be positive; got {pixels_per_beam!r}")
    return float(minor_axis) / float(pixels_per_beam)


def cell_size_from_beam_fit(beam_fit_params, pixels_per_beam):
    """:func:`cell_size_from_beam` of the highest frequency channel with a valid fit.

    The minor axis is that of the Stokes I fit. A fully flagged channel has an
    all zero fit, so channels whose minor axis is not finite and positive are
    skipped.

    Parameters
    ----------
    beam_fit_params : xarray.DataArray
        ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION`` of the PSF pass, with dims
        ``frequency``, ``polarization`` and ``beam_params_label`` (radians).
    pixels_per_beam : float
        Pixels across the minor axis.

    Returns
    -------
    float
        Cell size (radians).

    Raises
    ------
    ValueError
        When no channel has a valid fit, or for a non positive
        ``pixels_per_beam``.
    """
    minor = beam_fit_params.sel(polarization="I", beam_params_label="minor")
    minor_axis = minor.values
    valid = np.isfinite(minor_axis) & (minor_axis > 0)
    if not valid.any():
        raise ValueError("No channel has a valid beam fit (every channel flagged?).")
    return cell_size_from_beam(
        float(minor_axis[valid][np.argmax(minor.frequency.values[valid])]),
        pixels_per_beam,
    )


def max_cell_size_for_uv_coverage(cell_guess):
    """Largest cell whose uv grid holds every baseline of :func:`cell_size_guess`.

    The uv grid spans ``|u| <= 1 / (2 cell)`` and no baseline is longer than
    ``u_max = 1 / (CELL_GUESS_SAMPLES_PER_U_MAX * cell_guess)``.

    Parameters
    ----------
    cell_guess : float
        The cell of the PSF pass, :func:`cell_size_guess` (radians).

    Returns
    -------
    float
        ``1 / (2 u_max)`` (radians).
    """
    return 0.5 * CELL_GUESS_SAMPLES_PER_U_MAX * cell_guess
