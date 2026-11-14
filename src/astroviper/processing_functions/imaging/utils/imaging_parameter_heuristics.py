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
  of the PSF pass (:func:`cell_size_guess`), the beam fits the uv coverage and
  the image allow (:func:`plausible_beam_fits`), the cells at which the PSF of
  the highest channel is fitted again (:func:`beam_refit_cells`), the minor
  axis of those refits (:func:`refitted_minor_axis`), the final cell from it
  (:func:`cell_size_from_beam`) and the largest cell whose uv grid holds the
  samples of the PSF pass (:func:`max_cell_size_for_uv_coverage`);
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
from astroviper.utils.param_docs import shares_param_docs

SPEED_OF_LIGHT = 299792458.0
# Assumed.
FIELD_OF_VIEW_MARGIN = 1.1
# Derived: at least 2 puts every baseline on the uv grid of the PSF pass; 3 adds an
# assumed margin.
CELL_GUESS_SAMPLES_PER_U_MAX = 3.0
# Pixels across the minor axis of the first refit of the PSF. Measured on simulated
# ALMA cubes (dev/COMMON_BEAM/delphi_sampling_fix): with 2 to 3.25 pixels 12 of 60
# fits are junk, with 3.5 to 10 pixels 8 of 150, and the others at 3.5 to 10 pixels
# are within 1.5 percent of their median; 5 keeps the refit above 4.3 pixels for an
# estimate 14 percent too large, the worst measured.
REFIT_PIXELS_PER_BEAM = 5.0
# The second refit samples the minor axis this many times as finely (assumed); of
# the 190 pairs of those fits from 3.5 pixels sampled 1.2 to 1.3 times as finely as
# each other, none was junk twice.
REFIT_SAMPLING_RATIO = 1.25
# Measured: plausible fits of the PSF pass are within 14 percent of the refits, the
# minor axes of junk fits are 2.0 to 9.0 times the median of the good ones.
BEAM_FIT_AGREEMENT = 1.25
# Derived: the half power width of a uniformly weighted disk of radius u_max,
# 2 x 2.2151 / (2 pi u_max) (2 J1(x) / x = 1/2 at x = 2.2151), in units of 1 / u_max.
UNIFORM_DISK_HALF_POWER_WIDTH = 0.7051
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
    from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
        check_telescope_prescription,
    )

    name = str(telescope_name).strip().upper()
    check_telescope_prescription(name)
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
    from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
        check_telescope_prescription,
    )

    check_telescope_prescription(telescope, default_robust=True)
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


@shares_param_docs
def primary_beam_radius(
    frequency, level, primary_beam_params=None, n_scan=4096, *, input_checked=False
):
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
        The primary beam of the imager, the power pattern that gives the clean
        mask and the primary beam correction: ``telescope`` (``"alma"``,
        ``"aca"`` or ``"ngvla"``, the textbook obscured Airy disk with the
        diameters of :func:`~astroviper.utils.beam_models.airy_disk_model`, no
        cut; ``"vla"``, the EVLA beam polynomial of CASA's ``PBMath1DEVLA``,
        with the coefficients CASA picks for each channel (of its two 10052 MHz
        rows, the first), at the exact radius and zero beyond 58 arcmin at
        1 GHz, past the 5 percent power level its polynomials were fitted
        to), ``effective_dish_diameter`` and
        ``blockage_diameter`` (metres; refused for ``"vla"``) and
        ``max_rad_1GHz`` (radians at 1 GHz, scaled as 1 / frequency, beyond
        which the beam is zero; ``None`` is no cut for an Airy disk). Numbers
        given with a telescope override that telescope's values;
        no telescope means ``"alma"``. ``None`` is the ALMA 12 m beam (10.7 m,
        0.75 m, no cut). Or ``beam_model`` alone: an analytic dict
        or a beam-polynomial dataset as ``simulate_processing_set`` takes it in
        ``beam_models`` (``"vla"`` is
        ``read_beam_polynomial_coefficients("EVLA_")``). The beam is the square
        of the voltage pattern the simulator applies for the model, computed by
        the same function
        (:func:`~astroviper.processing_functions.simulation.antenna_beams.symmetric_beam_response`);
        an Airy disk (``func="airy"``) the imager interpolates from a radial
        table, within 1e-9 of it.
        The CASA variants, to compare with the ``.pb`` of ``tclean``, are such
        models: ``airy_disk_model(telescope, func="casa_airy")`` and
        ``read_beam_polynomial_coefficients("EVLA_", n_sample=10000)``.
        Zernike aperture models and Jones images are not supported: they are
        not azimuthally symmetric and depend on the parallactic angle.
    n_scan : int, optional
        Samples of the scan.

    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

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

    if not input_checked:
        from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
            check_beam_level,
        )

        check_beam_level(level)
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


@shares_param_docs
def field_of_view(
    frequency_min, level, primary_beam_params=None, *, input_checked=False
):
    """``FIELD_OF_VIEW_MARGIN`` times the diameter where the primary beam falls to ``level``.

    Parameters
    ----------
    frequency_min : float
        Lowest frequency of the cube (Hz), where the primary beam is widest.
    level : float
        Primary beam power level, in ``(0, 1)``.
    primary_beam_params : dict, optional
        The primary beam of the imager, the power pattern that gives the clean
        mask and the primary beam correction: ``telescope`` (``"alma"``,
        ``"aca"`` or ``"ngvla"``, the textbook obscured Airy disk with the
        diameters of :func:`~astroviper.utils.beam_models.airy_disk_model`, no
        cut; ``"vla"``, the EVLA beam polynomial of CASA's ``PBMath1DEVLA``,
        with the coefficients CASA picks for each channel (of its two 10052 MHz
        rows, the first), at the exact radius and zero beyond 58 arcmin at
        1 GHz, past the 5 percent power level its polynomials were fitted
        to), ``effective_dish_diameter`` and
        ``blockage_diameter`` (metres; refused for ``"vla"``) and
        ``max_rad_1GHz`` (radians at 1 GHz, scaled as 1 / frequency, beyond
        which the beam is zero; ``None`` is no cut for an Airy disk). Numbers
        given with a telescope override that telescope's values;
        no telescope means ``"alma"``. ``None`` is the ALMA 12 m beam (10.7 m,
        0.75 m, no cut). Or ``beam_model`` alone: an analytic dict
        or a beam-polynomial dataset as ``simulate_processing_set`` takes it in
        ``beam_models`` (``"vla"`` is
        ``read_beam_polynomial_coefficients("EVLA_")``). The beam is the square
        of the voltage pattern the simulator applies for the model, computed by
        the same function
        (:func:`~astroviper.processing_functions.simulation.antenna_beams.symmetric_beam_response`);
        an Airy disk (``func="airy"``) the imager interpolates from a radial
        table, within 1e-9 of it.
        The CASA variants, to compare with the ``.pb`` of ``tclean``, are such
        models: ``airy_disk_model(telescope, func="casa_airy")`` and
        ``read_beam_polynomial_coefficients("EVLA_", n_sample=10000)``.
        Zernike aperture models and Jones images are not supported: they are
        not azimuthally symmetric and depend on the parallactic angle.

    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

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
        * primary_beam_radius(
            frequency_min, level, primary_beam_params, input_checked=input_checked
        )
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


@shares_param_docs
def cell_size_from_beam(minor_axis, pixels_per_beam, *, input_checked=False):
    """Cell that puts ``pixels_per_beam`` pixels across the fitted minor axis.

    Parameters
    ----------
    minor_axis : float
        Minor axis of the fitted beam (radians).
    pixels_per_beam : float
        Pixels across the minor axis.

    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

    Returns
    -------
    float
        Cell size (radians).

    Raises
    ------
    ValueError
        For a non positive ``pixels_per_beam``.
    """
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
            check_pixels_per_beam,
        )

        check_pixels_per_beam(pixels_per_beam)
    return float(minor_axis) / float(pixels_per_beam)


def plausible_beam_fits(point_spread_function_xds):
    """Which beam fits of a PSF pass the uv coverage and the image allow.

    A fit is plausible when its parameters are finite, ``0 < minor <= major``,
    the minor axis is at least ``1 / (3 u_max)`` and the major axis is shorter
    than the image:

    * every sample adds ``cos(2 pi u . l) >= 1/2`` to the PSF where
      ``|u . l| <= 1/6``, so the main lobe is at least ``1 / (3 u_max)`` wide at
      half power, with ``u_max`` the longest baseline of the channel
      (``UV_DISTANCE_MAX``); a channel without data (``NaN``) has no plausible
      fit;
    * a main lobe longer than the image cannot be measured on it; the junk
      fits of the PSF pass have major axes of 16 to 94 times the image
      (measured).

    Parameters
    ----------
    point_spread_function_xds : xarray.Dataset
        A result of
        :func:`~astroviper.distributed_applications.imaging.point_spread_function_cube_single_field.point_spread_function_cube_single_field`:
        ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION`` (radians), ``UV_DISTANCE_MAX``
        (wavelengths) and the attributes ``image_size`` and ``cell_size``.

    Returns
    -------
    xarray.DataArray of bool
        Dims ``frequency`` and ``polarization``.
    """
    beam = point_spread_function_xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"]
    major = beam.sel(beam_params_label="major")
    minor = beam.sel(beam_params_label="minor")
    attrs = point_spread_function_xds.attrs
    image_extent = min(
        size * abs(cell)
        for size, cell in zip(attrs["image_size"], attrs["cell_size"], strict=True)
    )
    with np.errstate(divide="ignore"):
        minor_axis_limit = 1.0 / (3.0 * point_spread_function_xds["UV_DISTANCE_MAX"])
    return (
        np.isfinite(beam).all("beam_params_label")
        & (minor > 0.0)
        & (minor <= major)
        & (minor >= minor_axis_limit)
        & (major < image_extent)
    )


def beam_refit_cells(point_spread_function_xds):
    """The channel whose PSF is fitted again, its expected minor axis and the cells of the refits.

    The channel is the highest frequency with data (a finite
    ``UV_DISTANCE_MAX``). The expected minor axis is the median, over the
    channels whose Stokes I fit is plausible (:func:`plausible_beam_fits`), of
    the fitted minor axis times the frequency of the channel over that of the
    refits (the beam scales with the wavelength). The cells put
    ``REFIT_PIXELS_PER_BEAM`` and ``REFIT_SAMPLING_RATIO`` times as many
    pixels across it, or, when no fit is plausible, across the half power
    width of a uniformly weighted disk out to the longest baseline of the
    channel, ``UNIFORM_DISK_HALF_POWER_WIDTH / u_max``.

    Parameters
    ----------
    point_spread_function_xds : xarray.Dataset
        A result of
        :func:`~astroviper.distributed_applications.imaging.point_spread_function_cube_single_field.point_spread_function_cube_single_field`,
        as :func:`plausible_beam_fits` takes it.

    Returns
    -------
    tuple of (float, float or None, list of float)
        The frequency of the refits (Hz), the expected minor axis (radians;
        ``None`` when no fit is plausible) and the two cells (radians).

    Raises
    ------
    ValueError
        When no channel has data.
    """
    frequency = point_spread_function_xds.frequency.values
    uv_distance_max = point_spread_function_xds["UV_DISTANCE_MAX"].values
    with_data = np.flatnonzero(np.isfinite(uv_distance_max))
    if with_data.size == 0:
        raise ValueError("No channel has data (every channel flagged?).")
    refit_channel = with_data[np.argmax(frequency[with_data])]
    refit_frequency = float(frequency[refit_channel])
    plausible = (
        plausible_beam_fits(point_spread_function_xds).sel(polarization="I").values
    )
    minor = (
        point_spread_function_xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"]
        .sel(polarization="I", beam_params_label="minor")
        .values
    )
    minor_axis = None
    if plausible.any():
        minor_axis = float(
            np.median(minor[plausible] * frequency[plausible]) / refit_frequency
        )
    sampled_axis = (
        UNIFORM_DISK_HALF_POWER_WIDTH / float(uv_distance_max[refit_channel])
        if minor_axis is None
        else minor_axis
    )
    return (
        refit_frequency,
        minor_axis,
        [
            sampled_axis / (REFIT_PIXELS_PER_BEAM * ratio)
            for ratio in (1.0, REFIT_SAMPLING_RATIO)
        ],
    )


def refitted_minor_axis(refits, minor_axis_estimate):
    """The Stokes I minor axis given by refits of the PSF of one channel.

    A refit counts when its fit is plausible (:func:`plausible_beam_fits`)
    and its minor axis lies within a factor ``BEAM_FIT_AGREEMENT`` of that of
    another plausible refit or of ``minor_axis_estimate``; the minor axis is
    the smallest one that counts, so that the cell samples every such fit at
    least as finely as asked.

    Parameters
    ----------
    refits : sequence of xarray.Dataset
        Results of
        :func:`~astroviper.distributed_applications.imaging.point_spread_function_cube_single_field.point_spread_function_cube_single_field`
        for the same single channel at different cells, as
        :func:`plausible_beam_fits` takes them.
    minor_axis_estimate : float or None
        The minor axis expected from the PSF pass (radians), as
        :func:`beam_refit_cells` returns it.

    Returns
    -------
    float
        Minor axis (radians).

    Raises
    ------
    ValueError
        When no refit counts.
    """
    beams = [
        refit["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].sel(polarization="I").values[0]
        for refit in refits
    ]
    candidates = [
        beam[1]
        for beam, refit in zip(beams, refits, strict=True)
        if plausible_beam_fits(refit).sel(polarization="I").values[0]
    ]

    def agree(minor_axis, other):
        return max(minor_axis, other) <= BEAM_FIT_AGREEMENT * min(minor_axis, other)

    counted = [
        minor_axis
        for index, minor_axis in enumerate(candidates)
        if any(
            agree(minor_axis, other)
            for other_index, other in enumerate(candidates)
            if other_index != index
        )
        or (minor_axis_estimate is not None and agree(minor_axis, minor_axis_estimate))
    ]
    if not counted:
        raise ValueError(
            "The refits of the PSF give no plausible minor axis that agrees with "
            f"another refit or with {minor_axis_estimate} rad, the estimate of the "
            f"PSF pass: [major, minor, pa] {[beam.tolist() for beam in beams]} rad; "
            "the cell cannot be derived, so image_cube_single_field needs the "
            "image geometry given explicitly."
        )
    return float(min(counted))


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
