"""Single field cube imaging with every imaging parameter derived from the data.

:func:`image_cube_single_field_delphi` builds no graph itself. It reads the
metadata of the processing set (telescope, dish diameters, antenna positions,
correlations, frequencies, phase centre), runs one PSF pass
(:func:`~astroviper.distributed_applications.imaging.point_spread_function_cube_single_field.point_spread_function_cube_single_field`)
and derives from them every argument of
:func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`.
The arguments are written to a JSON parameters file, from which
``image_cube_single_field(parameters_file=...)`` reproduces the run, and the
cube is then imaged from that file.
"""

import importlib.metadata
import inspect
import json
import math
import os
from typing import Any

import toolviper.utils.parameter

from astroviper.distributed_applications.imaging.image_cube_single_field import (
    image_cube_single_field,
)
from astroviper.utils.param_docs import shares_param_docs

_PARAM_CONFIG_DIR = os.path.dirname(__file__)

# The kernel of the RADPS imaging accuracy memo.
GRIDDING_KERNEL_PARAMS = {
    "type": "exponential_semicircle",
    "support": 7,
    "oversampling": 300,
    "interpolation": "linear",
}
FFT_PADDING = 1.2
BEAM_PARAMS_LABEL = ["major", "minor", "pa"]
PSF_FITTING_METHOD = "astroviper"
IMAGE_DATA_VARIABLES_KEEP = [
    "sky_model",
    "sky_residual",
    "point_spread_function",
    "primary_beam",
    "beam_fit_params_point_spread_function",
]

# The variables of the PSF pass written to the "derived" block.
_POINT_SPREAD_FUNCTION_VARIABLES = (
    "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
    "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
    "SENSITIVITY",
    "SENSITIVITY_NATURAL",
    "SUM_OF_WEIGHTS",
    "EFFECTIVE_NUMBER_OF_VISIBILITIES",
    "UV_DISTANCE_MAX",
    "UV_DISTANCE_MIN",
    "ABS_W_MAX",
    "FLAGGED_FRACTION",
)


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def image_cube_single_field_delphi(
    ps_store: str,
    image_store: str,
    pixels_per_beam: float = 5.0,
    image_edge_primary_beam_limit: float = 0.2,
    weighting: str = "briggs",
    robust: float | None = None,
    n_sigma: float = 2.0,
    max_dynamic_range: float = 1e5,
    processing_set_data_group_name: str = "corrected",
    field_name: str | None = None,
    scan_intents: list[str] | None = ["OBSERVE_TARGET#ON_SOURCE"],  # noqa: B006 - None means every intent, so the default stays a list; never mutated
    calculate_parameters_only: bool = False,
    parameters_file: str | None = None,
    overwrite: bool = False,
    n_mapping_parallelism: dict[str, int | None] | None = None,
    compute_backend: str = "dask",
    mpi_cluster_setup: dict[str, Any] | None = None,
    processing_function_threads: int = 1,
) -> dict:
    """
    Image a single field cube with parameters derived from the data.

    The derivations, for ALMA (12 m and 7 m arrays) and the VLA:

    * ``primary_beam_params``: ``{"telescope": ...}`` from the telescope
      name (of the observation, else of the antennas) and the dish diameters
      (the smallest dish of a mixed ALMA array), see
      :func:`~astroviper.processing_functions.imaging.utils.imaging_parameter_heuristics.primary_beam_telescope`.
    * ``instrument_polarization_basis`` and the Stokes planes from the
      correlations every MSv4 holds: ``I, Q`` (linear) or ``I, V`` (circular)
      for the two parallel hands, ``I, Q, U, V`` for four correlations.
    * The frequency coordinate is every channel of the processing set and the
      phase direction is that of ``field_name`` (the centre field for
      ``None``).
    * Geometry: the field of view is 1.1 times the diameter at which the
      primary beam of the imager falls to ``image_edge_primary_beam_limit`` at
      the lowest frequency. The PSF pass uses the cell ``1 / (3 u_max)``, with
      ``u_max`` the longest antenna separation at the highest frequency, so
      that no sample falls off the uv grid; the cube then uses the cell that
      puts ``pixels_per_beam`` pixels across the fitted minor axis of the
      Stokes I PSF of the highest frequency. Each image size is the next even
      2, 3, 5 smooth size that covers the field of view.
    * Imaging weights: ``weighting`` with ``robust`` (``None``: 0.5 for ALMA
      and ACA, 2.0 for the VLA, the CASA pipeline values of a cube).
    * Gridding kernel: exponential semicircle, support 7, oversampling 300,
      linear interpolation, with ``fft_padding`` 1.2.
    * Cycle controls for noisy data
      (:func:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics.noisy_cycle_control_params`):
      a per channel threshold of ``n_sigma`` times the theoretical
      sensitivity of every channel and Stokes plane, the entropy stop, and
      ``max_cycles`` from ``max_dynamic_range`` and the largest PSF sidelobe.
      The clean mask is the primary beam above
      ``image_edge_primary_beam_limit``.
    * Products: the model, residual, PSF, primary beam and beam fit, the
      restored cube and its primary beam correction.

    Parameters
    ----------
    ps_store : str
        Path of the input processing set (Zarr store).
    image_store : str
        Path/URL of the on-disk Zarr image cube.
    pixels_per_beam : float, optional
        Pixels across the fitted minor axis of the Stokes I PSF at the highest
        frequency. Default 5.
    image_edge_primary_beam_limit : float, optional
        Primary beam power at the radius whose 1.1 times sets the half width
        of the image, at the lowest frequency; also the primary beam limit of
        the clean mask. Default 0.2.
    weighting : str, optional
        ``"briggs"`` (default), ``"natural"`` or ``"uniform"``.
    robust : float, optional
        Briggs robust parameter; ``None`` (default) takes the CASA pipeline
        value of the telescope.
    n_sigma : float, optional
        Threshold of every channel and Stokes plane in units of its
        theoretical sensitivity. Default 2.
    max_dynamic_range : float, optional
        Largest ratio of the peak residual of a dirty plane to its sensitivity
        that ``max_cycles`` allows for. Default 1e5.
    processing_set_data_group_name : str
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    field_name : str, optional
        The field whose phase centre is the image centre; ``None`` (default)
        takes the centre field of the processing set.
    scan_intents : list[str] or None
        The scan intents to image. ``None`` images every scan intent.
    calculate_parameters_only : bool, optional
        If ``True``, run only the PSF pass and the derivations, write the
        parameters file and return the parameters without imaging.
    parameters_file : str, optional
        Path of the JSON parameters file; ``None`` (default) is
        ``f"{image_store}.parameters.json"``. An existing file is refused
        unless ``overwrite``.
    overwrite : bool
        Whether to overwrite an existing parameters file and image. Default
        is False.
    n_mapping_parallelism : dict, optional
        Mapping parallelism of the PSF pass and of the imaging, as
        ``{"frequency": n_chunks}``; ``None`` (default) auto-determines it.
    compute_backend : str
        ``"dask"`` (default) or ``"mpi"``, for the PSF pass and the imaging.
    mpi_cluster_setup : dict, optional
        Options of the MPI backend (see
        :func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`).
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.

    Returns
    -------
    dict
        With ``calculate_parameters_only`` the parameters: the content of the
        parameters file, ``{"image_cube_single_field": {every argument of
        image_cube_single_field but parameters_file}, "derived": {the
        observables of the PSF pass per channel and Stokes plane and the
        derived geometry}, "provenance": {"astroviper": version,
        "delphi_inputs": {the arguments of this call}}}``. Otherwise the result
        of :func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`
        with the parameters added under ``"parameters"``.

    Raises
    ------
    FileExistsError
        For an existing parameters file without ``overwrite``.
    ValueError
        For a telescope without a primary beam model (or without a pipeline
        robust when ``robust`` is ``None``), or correlations the imager cannot
        image.
    """
    import numpy as np
    import toolviper.utils.logger as logger
    from xradio.measurement_set import open_processing_set

    from astroviper.distributed_applications.imaging.point_spread_function_cube_single_field import (
        point_spread_function_cube_single_field,
    )
    from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
        noisy_cycle_control_params,
        predicted_imaging_cycles,
        psf_fraction,
    )
    from astroviper.processing_functions.imaging.utils.imaging_parameter_heuristics import (
        cell_size_from_beam,
        cell_size_guess,
        default_robust,
        field_of_view,
        image_size_for_field_of_view,
        longest_antenna_separation,
        primary_beam_telescope,
        stokes_for_correlations,
    )

    call_locals = dict(locals())
    delphi_inputs = {
        name: call_locals[name]
        for name in inspect.signature(image_cube_single_field_delphi).parameters
    }
    for name in ("pixels_per_beam", "n_sigma", "max_dynamic_range"):
        if not delphi_inputs[name] > 0:
            raise ValueError(f"{name} must be positive, got {delphi_inputs[name]!r}")
    if parameters_file is None:
        parameters_file = f"{image_store}.parameters.json"
    if os.path.exists(parameters_file) and not overwrite:
        raise FileExistsError(
            f"parameters file {parameters_file} exists; pass overwrite=True"
        )

    # Metadata only: nothing is loaded.
    ps_xdt = open_processing_set(ps_store, scan_intents=scan_intents)
    metadata = _processing_set_metadata(ps_xdt, field_name)
    telescope = primary_beam_telescope(
        metadata["telescope_name"], metadata["dish_diameters"]
    )
    primary_beam_params = {"telescope": telescope}
    instrument_polarization_basis, polarization_coords = stokes_for_correlations(
        metadata["correlations"]
    )
    frequency_coords = metadata["frequency_coords"]
    imaging_weights_params = {"weighting": weighting}
    if weighting == "briggs":
        imaging_weights_params["robust"] = float(
            default_robust(telescope) if robust is None else robust
        )

    fov = field_of_view(
        float(frequency_coords.min()),
        image_edge_primary_beam_limit,
        primary_beam_params,
    )
    max_antenna_separation = longest_antenna_separation(metadata["antenna_positions"])
    cell_guess = cell_size_guess(max_antenna_separation, float(frequency_coords.max()))
    image_size_guess = image_size_for_field_of_view(fov, cell_guess)
    image_params = {
        "image_size": [image_size_guess, image_size_guess],
        "cell_size": np.array([-cell_guess, cell_guess]),
        "phase_direction": metadata["phase_direction"],
        "frequency_coords": frequency_coords,
        "polarization_coords": polarization_coords,
        "time_coords": [0],
        "fft_padding": FFT_PADDING,
    }
    point_spread_function_xds = point_spread_function_cube_single_field(
        ps_store,
        image_params,
        imaging_weights_params,
        image_store=None,
        instrument_polarization_basis=instrument_polarization_basis,
        scan_intents=scan_intents,
        field_name=field_name,
        processing_set_data_group_name=processing_set_data_group_name,
        gridding_kernel_params=GRIDDING_KERNEL_PARAMS,
        primary_beam_params=primary_beam_params,
        psf_fitting_method=PSF_FITTING_METHOD,
        n_mapping_parallelism=n_mapping_parallelism,
        processing_function_threads=processing_function_threads,
        compute_backend=compute_backend,
        # graphviper arms teardown_force_exit_seconds at the end of a compute
        # and never disarms it, so only the last compute gets it.
        mpi_cluster_setup=mpi_cluster_setup
        if mpi_cluster_setup is None or calculate_parameters_only
        else {
            k: v
            for k, v in mpi_cluster_setup.items()
            if k != "teardown_force_exit_seconds"
        },
    )

    # The last axis holds (major, minor, pa), as the imager's beam fit (whose
    # dimension name differs from its label coordinate).
    beam = point_spread_function_xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"]
    beam = beam.transpose("frequency", "polarization", ...)
    # A fully flagged channel has an all zero fit: take the highest frequency
    # channel with a valid one.
    minor = beam.values[
        :, list(beam.polarization.values).index("I"), BEAM_PARAMS_LABEL.index("minor")
    ]
    valid = np.isfinite(minor) & (minor > 0)
    if not valid.any():
        raise ValueError("No channel has a valid beam fit (every channel flagged?).")
    minor_axis = float(minor[valid][np.argmax(beam.frequency.values[valid])])
    cell = cell_size_from_beam(minor_axis, pixels_per_beam)
    # The uv grid spans |u| <= 1 / (2 cell), and no baseline is longer than
    # u_max = 1 / (3 cell_guess).
    if cell > 1.5 * cell_guess:
        logger.warning(
            f"The cell {cell:.4g} rad of {pixels_per_beam} pixels per beam is "
            f"larger than 1 / (2 u_max) = {1.5 * cell_guess:.4g} rad: the "
            "longest baselines may fall off the uv grid."
        )
    image_size = image_size_for_field_of_view(fov, cell)
    image_params = {
        **image_params,
        "image_size": [image_size, image_size],
        "cell_size": np.array([-cell, cell]),
    }
    cycle_control_params = noisy_cycle_control_params(
        point_spread_function_xds,
        n_sigma=n_sigma,
        max_dynamic_range=max_dynamic_range,
        primary_beam_limit=image_edge_primary_beam_limit,
    )
    fraction = psf_fraction(
        float(
            np.nanmax(point_spread_function_xds["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"])
        )
    )

    arguments = {
        name: parameter.default
        for name, parameter in inspect.signature(
            image_cube_single_field
        ).parameters.items()
        if name != "parameters_file"
    }
    arguments.update(
        ps_store=ps_store,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params=imaging_weights_params,
        cycle_control_params=cycle_control_params,
        instrument_polarization_basis=instrument_polarization_basis,
        scan_intents=scan_intents,
        field_name=field_name,
        image_data_variables_keep=IMAGE_DATA_VARIABLES_KEEP,
        processing_set_data_group_name=processing_set_data_group_name,
        processing_function_threads=processing_function_threads,
        n_mapping_parallelism=n_mapping_parallelism,
        overwrite=overwrite,
        restore=True,
        primary_beam_correction=True,
        psf_fitting_method=PSF_FITTING_METHOD,
        gridding_kernel_params=GRIDDING_KERNEL_PARAMS,
        primary_beam_params=primary_beam_params,
        compute_backend=compute_backend,
        mpi_cluster_setup=mpi_cluster_setup,
    )
    derived = {
        "telescope_name": metadata["telescope_name"],
        "telescope": telescope,
        "frequency": frequency_coords,
        "polarization": polarization_coords,
        "field_of_view": fov,
        "longest_antenna_separation": max_antenna_separation,
        "cell_size_guess": cell_guess,
        "image_size_guess": image_size_guess,
        "cell_size": cell,
        "image_size": image_size,
        "psf_fraction": fraction,
        "predicted_imaging_cycles": predicted_imaging_cycles(
            max_dynamic_range, n_sigma, fraction
        ),
        "beam_params_label": BEAM_PARAMS_LABEL,
    }
    for name in _POINT_SPREAD_FUNCTION_VARIABLES:
        if name in point_spread_function_xds:
            derived[name.lower()] = point_spread_function_xds[name].values
    parameters = _json_compatible(
        {
            "image_cube_single_field": arguments,
            "derived": derived,
            "provenance": {
                "astroviper": importlib.metadata.version("astroviper"),
                "delphi_inputs": delphi_inputs,
            },
        }
    )
    with open(parameters_file, "w") as file:
        json.dump(parameters, file, indent=2, allow_nan=False)
    logger.info(f"Imaging parameters written to {parameters_file}")
    if calculate_parameters_only:
        return parameters

    result = image_cube_single_field(parameters_file=parameters_file)
    result["parameters"] = parameters
    return result


def _processing_set_metadata(ps_xdt, field_name):
    """Telescope, dish diameters, antenna positions, correlations, frequencies
    and phase direction of a processing set (metadata only).

    The telescope name is the ``telescope_name`` of the observation info of
    the first MSv4, else the ``overall_telescope_name`` of the antennas, else
    the ``telescope_name`` of the first antenna.
    """
    import numpy as np

    first_ms = next(iter(ps_xdt.values()))
    antenna_xds = ps_xdt.xr_ps.get_combined_antenna_xds()
    telescope_name = first_ms.ds.attrs.get("observation_info", {}).get(
        "telescope_name"
    ) or antenna_xds.attrs.get("overall_telescope_name")
    if not telescope_name and "telescope_name" in antenna_xds.coords:
        telescope_name = str(antenna_xds.telescope_name.values[0])
    dish_diameters = (
        antenna_xds["ANTENNA_DISH_DIAMETER"].values
        if "ANTENNA_DISH_DIAMETER" in antenna_xds
        else None
    )
    # The Stokes planes are made from the correlations every MSv4 holds.
    correlations = None
    for ms_xdt in ps_xdt.values():
        labels = {str(label) for label in ms_xdt.ds.polarization.values}
        correlations = labels if correlations is None else correlations & labels
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    if combined.sizes.get("field_name", 1) > 1:
        raise ValueError(
            "The imager makes one field; the selected scan intents leave the "
            f"fields {list(combined.field_name.values)}."
        )
    if field_name is None:
        field_name = combined.attrs["center_field_name"]
    return {
        "telescope_name": telescope_name,
        "dish_diameters": dish_diameters,
        "antenna_positions": antenna_xds["ANTENNA_POSITION"].values,
        "correlations": sorted(correlations),
        "frequency_coords": np.asarray(ps_xdt.xr_ps.get_freq_axis().values, float),
        "phase_direction": np.asarray(
            combined.FIELD_PHASE_CENTER_DIRECTION.sel(field_name=field_name).values,
            float,
        ),
    }


def _json_compatible(value):
    """``value`` with NumPy arrays and scalars as Python lists and numbers,
    tuples as lists and dict keys as strings."""
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
