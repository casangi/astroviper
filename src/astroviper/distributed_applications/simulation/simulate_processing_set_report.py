"""Report of a :func:`~astroviper.distributed_applications.simulation.simulate_processing_set.simulate_processing_set` run."""

from __future__ import annotations

import os

import numpy as np

SPEED_OF_LIGHT = 299792458.0  # m / s

# The node task timing columns of the task stream.
SIMULATION_TASK_STREAM_COLUMNS = {
    "task": "T_simulate_task",
    "load": (),
    "write": "T_write",
}

# Pixels across the synthesized beam (wavelength over the longest baseline at
# the highest frequency) and pixels of the image of the point spread function,
# when the simulation has no sky_image_params.
PIXELS_PER_BEAM = 5.0
POINT_SPREAD_FUNCTION_IMAGE_SIZE = 256
# Largest side of the image of the sky components.
MAX_MODEL_IMAGE_SIZE = 2048
# Pixels of the images made for the sensitivity of every channel (natural
# weighting: the sensitivity does not depend on the grid).
SENSITIVITY_IMAGE_SIZE = 128
# At most this many uv samples (times x baselines) are read for the summary
# and the uv coverage; longer observations are read every k-th time.
MAX_UV_SAMPLES = 2_000_000


def simulate_processing_set_report(
    return_dict,
    ps_store,
    report="html",
    sky_components=None,
    noise_params=None,
    sky_image_params=None,
    processing_function_threads=1,
    compute_backend="dask",
    mpi_cluster_setup=None,
    thread_info=None,
):
    """Make the report of a ``simulate_processing_set`` run.

    The processing section is the one of every distributed application
    (:func:`~astroviper.utils.report.add_processing_section`). The results
    section, worked out from the processing set written and the sky
    components:

    - a summary of the array and the observation, with the shortest and
      longest baseline;
    - the natural weighting point source sensitivity per channel and Stokes
      plane (``SENSITIVITY_NATURAL`` of
      :func:`~astroviper.distributed_applications.imaging.point_spread_function_cube_single_field.point_spread_function_cube_single_field`);
    - the antenna positions (east and north, and the height of every antenna);
    - the uv coverage at the centre frequency;
    - cross sections of the point spread function (natural weighting, Stokes
      I, centre channel) along the major axis, the minor axis and the diagonal
      between them, with the fitted Gaussian;
    - the image of the sum of the sky components at the centre channel and
      their total flux per channel, per Stokes plane (no imaging).

    The point spread function is made on the grid of ``sky_image_params`` or,
    without them, on a grid of :data:`POINT_SPREAD_FUNCTION_IMAGE_SIZE` pixels
    with :data:`PIXELS_PER_BEAM` pixels across the synthesized beam of the
    longest baseline at the highest frequency.

    Parameters
    ----------
    return_dict : dict
        The return dict of ``simulate_processing_set``.
    ps_store : str
        The processing set it wrote.
    report : str, optional
        ``"html"`` (default), ``"pdf"`` or ``"notebook"``.
    sky_components : list of dict, optional
        The sky components simulated (normalised or not); ``None`` leaves out
        the image and the spectrum of the components.
    noise_params : dict, optional
        The noise parameters of the simulation; ``None`` (no noise) leaves out
        the sensitivity, which is not a noise level for unit weights.
    sky_image_params : dict, optional
        The ``sky_image_params`` of the simulation: their ``image_size``,
        ``cell_size`` and ``phase_direction`` give the grid of the point spread
        function and of the image of the components.
    processing_function_threads : int, optional
        Passed to ``point_spread_function_cube_single_field``.
    compute_backend : str, optional
        Passed to ``point_spread_function_cube_single_field``.
    mpi_cluster_setup : dict, optional
        Passed to ``point_spread_function_cube_single_field``.
    thread_info : dict, optional
        Passed to ``point_spread_function_cube_single_field``.

    Returns
    -------
    str or None
        The path of the report written (``<name>_report.html`` or ``.pdf``
        next to ``ps_store``); ``None`` for ``"notebook"``. The point spread
        function of the centre channel is kept in ``<name>_report_files``
        next to ``ps_store``.
    """
    import datetime

    from xradio.measurement_set import open_processing_set

    from astroviper.distributed_applications.simulation.simulate_processing_set import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
    )
    from astroviper.utils.data_tree import release_data_tree
    from astroviper.utils.report import (
        Report,
        add_processing_section,
        report_files_directory,
        report_path,
    )

    path = report_path(ps_store, report)
    files_directory = report_files_directory(ps_store)
    ps_xdt = open_processing_set(ps_store)
    try:
        observation = _observation(ps_xdt)
    finally:
        release_data_tree(ps_xdt)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with Report(
        "simulate_processing_set report",
        report,
        path,
        subtitle=f"Processing set {ps_store}, report made {stamp}.",
    ) as document:
        add_processing_section(
            document,
            return_dict,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            task_columns=SIMULATION_TASK_STREAM_COLUMNS,
            task_name="(time, frequency) chunk",
        )
        document.section("Results")
        grid = _grid(observation, sky_image_params)
        stokes = _stokes(observation["polarization"])
        _add_summary(document, observation, noise_params)
        _add_antenna_positions(document, observation)
        _add_uv_coverage(document, observation)
        if stokes is None:
            document.note(
                "The simulated correlations "
                f"{list(observation['polarization'])} cannot form Stokes I: no "
                "point spread function or sensitivity."
            )
        else:
            _add_point_spread_function(
                document,
                ps_store,
                observation,
                grid,
                stokes,
                noise_params,
                files_directory,
                processing_function_threads,
                compute_backend,
                mpi_cluster_setup,
                thread_info,
            )
        _add_sky_components(
            document,
            observation,
            grid,
            stokes or ["I"],
            sky_components,
            return_dict.get("sky_image_store"),
        )
    return path


def _observation(ps_xdt):
    """What the report needs of the (first) measurement set of ``ps_xdt``."""
    from astroviper.utils.telescope_layout import observatory_position

    ms_name, ms_xdt = next(iter(ps_xdt.items()))
    ms_xds = ms_xdt.ds
    antenna_xds = ms_xdt["antenna_xds"].ds
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    phase_direction = combined.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=combined.attrs["center_field_name"]
    ).values
    frequency = ms_xds["frequency"].values
    centre = len(frequency) // 2
    antenna_position = np.asarray(antenna_xds["ANTENNA_POSITION"].values, dtype=float)
    telescope_name = str(antenna_xds.attrs.get("overall_telescope_name", "unknown"))
    time = ms_xds["time"].values
    time_stride = max(
        1, int(np.ceil(len(time) * ms_xds.sizes["baseline_id"] / MAX_UV_SAMPLES))
    )
    samples = ms_xds.isel(time=slice(None, None, time_stride), frequency=centre)
    flagged = samples["FLAG"].values.any(axis=-1)
    uvw = samples["UVW"].values
    return {
        "ms_name": ms_name,
        "telescope_name": telescope_name,
        "antenna_name": [str(name) for name in antenna_xds["antenna_name"].values],
        "antenna_position": antenna_position,
        "site_position": observatory_position(telescope_name, antenna_position),
        "frequency": frequency,
        "channel_width": float(
            np.asarray(ms_xds["frequency"].attrs["channel_width"]["data"]).ravel()[0]
        ),
        "polarization": [str(label) for label in ms_xds["polarization"].values],
        "time": time,
        "integration_time": float(
            np.asarray(ms_xds["time"].attrs["integration_time"]["data"]).ravel()[0]
        ),
        "n_baseline": ms_xds.sizes["baseline_id"],
        "phase_direction": np.asarray(phase_direction, dtype=float).reshape(2),
        # The unflagged cross correlations at the centre frequency (metres),
        # of every time_stride-th time.
        "time_stride": time_stride,
        "uvw": uvw[~flagged & np.any(uvw != 0.0, axis=-1)],
        "weight": samples["WEIGHT"].values[~flagged],
    }


def _stokes(polarization):
    """The Stokes planes the correlations form, ``None`` without Stokes I."""
    labels = set(polarization)
    if labels >= {"XX", "XY", "YX", "YY"} or labels >= {"RR", "RL", "LR", "LL"}:
        return ["I", "Q", "U", "V"]
    if labels >= {"XX", "YY"}:
        return ["I", "Q"]
    if labels >= {"RR", "LL"}:
        return ["I", "V"]
    return None


def _baselines(observation):
    """Shortest and longest baseline, metres, with their antenna pairs."""
    position = observation["antenna_position"]
    first, second = np.triu_indices(len(position), k=1)
    length = np.linalg.norm(position[first] - position[second], axis=-1)
    shortest, longest = int(np.argmin(length)), int(np.argmax(length))
    names = observation["antenna_name"]
    return (
        (length[shortest], names[first[shortest]], names[second[shortest]]),
        (length[longest], names[first[longest]], names[second[longest]]),
    )


def _grid(observation, sky_image_params):
    """``(image_size, cell_size, phase_direction)`` of the point spread function."""
    if sky_image_params is not None:
        phase_direction = sky_image_params.get("phase_direction")
        return (
            [int(n) for n in np.ravel(sky_image_params["image_size"])],
            np.asarray(sky_image_params["cell_size"], dtype=float).ravel(),
            observation["phase_direction"]
            if phase_direction is None
            else np.asarray(phase_direction, dtype=float).reshape(2),
        )
    longest = _baselines(observation)[1][0]
    wavelength = SPEED_OF_LIGHT / float(np.max(observation["frequency"]))
    cell = wavelength / longest / PIXELS_PER_BEAM
    size = POINT_SPREAD_FUNCTION_IMAGE_SIZE
    return [size, size], np.array([-cell, cell]), observation["phase_direction"]


def _add_summary(document, observation, noise_params):
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    (short, short_1, short_2), (long, long_1, long_2) = _baselines(observation)
    centre_frequency = float(
        observation["frequency"][len(observation["frequency"]) // 2]
    )
    uv_distance = (
        np.hypot(observation["uvw"][:, 0], observation["uvw"][:, 1])
        * centre_frequency
        / SPEED_OF_LIGHT
    )
    time = Time(observation["time"], format="unix", scale="utc")
    direction = SkyCoord(*observation["phase_direction"], unit="rad")
    frequency = observation["frequency"] / 1e9
    rows = [
        ("measurement set", observation["ms_name"]),
        ("telescope", observation["telescope_name"]),
        ("antennas", len(observation["antenna_name"])),
        ("baselines", observation["n_baseline"]),
        ("shortest baseline", f"{short:.4g} m ({short_1}-{short_2})"),
        ("longest baseline", f"{long:.4g} m ({long_1}-{long_2})"),
        (
            "uv distance, centre frequency",
            (
                f"{uv_distance.min() / 1e3:.4g} to {uv_distance.max() / 1e3:.4g} "
                "kilo wavelengths"
                if uv_distance.size
                else "no unflagged cross correlation"
            )
            + (
                f" (every {observation['time_stride']}th time)"
                if observation["time_stride"] > 1
                else ""
            ),
        ),
        (
            "times",
            f"{len(time)} of {observation['integration_time']:.4g} s, "
            f"{time[0].isot} to {time[-1].isot} UTC",
        ),
        (
            "channels",
            f"{len(frequency)} of {observation['channel_width'] / 1e6:.4g} MHz, "
            f"{frequency.min():.6g} to {frequency.max():.6g} GHz",
        ),
        ("polarizations", ", ".join(observation["polarization"])),
        ("phase centre", direction.to_string("hmsdms", precision=2)),
    ]
    if noise_params is None:
        rows.append(("noise", "none (noise_params=None, unit weights)"))
    else:
        weight = observation["weight"]
        weight = weight[np.isfinite(weight) & (weight > 0)]
        rows.append(
            (
                "noise of one visibility, centre frequency",
                f"{np.median(1.0 / np.sqrt(weight)):.4g} Jy (median over the "
                "baselines, real or imaginary part)"
                if weight.size
                else "no unflagged weight",
            )
        )
    document.subsection("Observation")
    document.table(
        pd.DataFrame(rows, columns=["", "value"]).set_index("").rename_axis(None)
    )


def _add_antenna_positions(document, observation):
    from astroviper.utils.report_plots import plot_antenna_positions
    from astroviper.utils.telescope_layout import itrf_to_local_tangent_plane

    local = itrf_to_local_tangent_plane(
        observation["antenna_position"], observation["site_position"]
    )
    document.subsection("Antennas")
    document.figure(
        plot_antenna_positions(
            observation["antenna_name"],
            local[:, 0],
            local[:, 1],
            local[:, 2],
            title=f"{observation['telescope_name']}: "
            f"{len(observation['antenna_name'])} antennas",
        ),
        caption="Antenna positions in the local tangent plane of the array "
        "reference position (east, north and up).",
    )


def _add_uv_coverage(document, observation):
    from astroviper.utils.report_plots import plot_uv_coverage

    document.subsection("uv coverage")
    frequency = observation["frequency"]
    centre_frequency = float(frequency[len(frequency) // 2])
    uv = observation["uvw"][:, :2] * centre_frequency / SPEED_OF_LIGHT
    document.figure(
        plot_uv_coverage(
            uv[:, 0], uv[:, 1], title=f"uv coverage at {centre_frequency / 1e9:.6g} GHz"
        ),
        caption="The unflagged cross correlations and their conjugates at the "
        "centre frequency.",
    )


def _add_point_spread_function(
    document,
    ps_store,
    observation,
    grid,
    stokes,
    noise_params,
    files_directory,
    processing_function_threads,
    compute_backend,
    mpi_cluster_setup,
    thread_info,
):
    import xarray as xr

    from astroviper.distributed_applications.imaging.point_spread_function_cube_single_field import (
        point_spread_function_cube_single_field,
    )
    from astroviper.processing_functions.simulation.sky_components import (
        polarization_basis_of,
    )
    from astroviper.utils.report_plots import (
        plot_point_spread_function_cross_sections,
        plot_spectra,
    )

    image_size, cell_size, phase_direction = grid
    basis = polarization_basis_of(observation["polarization"])
    frequency = observation["frequency"]
    centre = len(frequency) // 2

    def point_spread_function(frequency_coords, image_size, image_store=None):
        return point_spread_function_cube_single_field(
            ps_store=ps_store,
            image_params={
                "image_size": image_size,
                "cell_size": cell_size,
                "phase_direction": phase_direction,
                "frequency_coords": np.asarray(frequency_coords, dtype=float),
                "polarization_coords": stokes,
                "time_coords": [0],
                "fft_padding": 1.2,
            },
            imaging_weights_params={"weighting": "natural"},
            image_store=image_store,
            instrument_polarization_basis=basis,
            scan_intents=None,
            processing_set_data_group_name="base",
            thread_info=thread_info,
            processing_function_threads=processing_function_threads,
            overwrite=True,
            compute_backend=compute_backend,
            mpi_cluster_setup=mpi_cluster_setup,
        )

    document.subsection("Noise")
    if noise_params is None:
        document.note(
            "No noise was simulated (noise_params=None): the weights are 1, so "
            "there is no noise level against frequency."
        )
    else:
        # Every channel, small images and no image store: only the sensitivity.
        properties = point_spread_function(
            frequency, [SENSITIVITY_IMAGE_SIZE, SENSITIVITY_IMAGE_SIZE]
        )
        document.figure(
            plot_spectra(
                properties["frequency"].values,
                properties["SENSITIVITY_NATURAL"]
                .transpose("frequency", "polarization")
                .values,
                stokes,
                "Jy/beam",
                title="Point source sensitivity, natural weighting",
            ),
            caption="The theoretical point source noise of an image of each "
            "channel and Stokes plane with natural weights, from the weights "
            "of the processing set.",
        )

    document.subsection("Point spread function")
    image_store = os.path.join(files_directory, "point_spread_function.img.zarr")
    properties = point_spread_function(
        [frequency[centre]], image_size, image_store=image_store
    )
    beam = np.asarray(
        properties["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"]
        .isel(frequency=0)
        .sel(polarization="I")
        .values,
        dtype=float,
    )
    if not np.all(np.isfinite(beam)) or np.any(beam[:2] <= 0):
        document.note(f"The beam fit of the centre channel failed ({beam}).")
        return
    psf = (
        xr.open_zarr(image_store)["POINT_SPREAD_FUNCTION"]
        .isel(time=0, frequency=0)
        .sel(polarization="I")
        .transpose("l", "m")
    )
    arcsec = 180.0 / np.pi * 3600.0
    document.figure(
        plot_point_spread_function_cross_sections(
            psf.values,
            beam,
            (cell_size[0], cell_size[1]),
            title=(
                f"Stokes I, natural weighting, {frequency[centre] / 1e9:.6g} GHz: "
                f'beam {beam[0] * arcsec:.4g}" x {beam[1] * arcsec:.4g}", position '
                f"angle {np.degrees(beam[2]):.1f} deg"
            ),
        ),
        caption="The point spread function through its peak (solid) and the "
        "fitted Gaussian (dashed) along the major axis, the minor axis and the "
        f"diagonal between them ({image_store}).",
    )


def _model_grid(observation, grid, components):
    """The grid of the image of the components: the grid of the point spread
    function, widened (with the same cell, at most
    :data:`MAX_MODEL_IMAGE_SIZE` pixels, then coarser) to hold every
    component position."""
    from astroviper.utils.coordinate_transforms import sin_project

    image_size, cell_size, phase_direction = grid
    offsets = [0.0]
    for component in components:
        for ra_dec in np.asarray(component["ra_dec"], dtype=float).reshape(-1, 2):
            offsets.extend(np.abs(sin_project(phase_direction, ra_dec)))
    half_width = 1.25 * max(offsets)
    cell = np.abs(np.asarray(cell_size, dtype=float))
    size = [
        max(int(n), int(np.ceil(2.0 * half_width / c)) + 1)
        for n, c in zip(image_size, cell, strict=True)
    ]
    scale = max(1.0, max(size) / MAX_MODEL_IMAGE_SIZE)
    if scale > 1.0:
        size = [int(np.ceil(n / scale)) for n in size]
        cell_size = np.asarray(cell_size, dtype=float) * scale
    return size, cell_size, phase_direction


def _add_sky_components(
    document, observation, grid, stokes, sky_components, sky_image_store
):
    import xarray as xr
    from xradio.image import make_empty_sky_image

    from astroviper.processing_functions.simulation.sky_components import (
        normalize_sky_components,
        polarization_basis_of,
        stokes_sky_model_images,
        total_stokes_flux,
    )
    from astroviper.utils.report_plots import plot_spectra, plot_stokes_images

    document.subsection("Sky components")
    if not sky_components:
        document.note("The sky components were not given to the report.")
        return
    components = normalize_sky_components(sky_components)
    basis = polarization_basis_of(observation["polarization"])
    frequency = observation["frequency"]
    centre = len(frequency) // 2
    if sky_image_store is not None:
        sky = xr.open_zarr(sky_image_store)["SKY"].isel(time=0, frequency=centre)
        source = f"the sky image of the simulation ({sky_image_store})"
    else:
        image_size, cell_size, phase_direction = _model_grid(
            observation, grid, components
        )
        img_xds = make_empty_sky_image(
            phase_center=phase_direction,
            image_size=image_size,
            cell_size=cell_size,
            frequency_coords=[float(frequency[centre])],
            pol_coords=stokes,
            time_coords=[0.0],
            do_sky_coords=False,
        )
        sky = xr.DataArray(
            stokes_sky_model_images(
                components,
                img_xds["l"].values,
                img_xds["m"].values,
                phase_direction,
                basis,
                stokes,
                frequency_index=centre,
            ),
            dims=("polarization", "l", "m"),
            coords={
                "polarization": stokes,
                "l": img_xds["l"].values,
                "m": img_xds["m"].values,
            },
        )
        source = "the components rasterised on the grid of the point spread function"
    document.figure(
        plot_stokes_images(
            sky.load(),
            title=f"Sum of the sky components, {frequency[centre] / 1e9:.6g} GHz",
            colorbar_label="Jy/pixel",
            stretch="asinh",
        ),
        caption=f"The sky the visibilities were simulated from, without the "
        f"primary beam: {source}. Point sources fill one pixel; the color scale "
        "is linear up to a hundredth of the peak and logarithmic above.",
    )
    document.figure(
        plot_spectra(
            frequency,
            total_stokes_flux(components, basis, stokes, len(frequency)),
            stokes,
            "Jy",
            title="Total flux of the sky components",
        ),
        caption="The sum of the component fluxes per channel and Stokes "
        "parameter, without the primary beam.",
    )
