"""Continuum channel finding of the ALMA pipeline (``hif_findcont``).

Runs the pipeline's findContinuum (vendored in ``findContinuum.py``) on a dirty
cube in an xradio image store. findContinuum's CASA tasks and tools are those of
:mod:`casa_image_tasks`; every reduction of the cube is its own map-reduce over
frequency chunks, so the cube is read as often as the pipeline reads it.
"""

import contextlib
import glob
import io
import os
import time

import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter

_PARAM_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


class _LogStream(io.TextIOBase):
    """Sends findContinuum's printed lines to the logger at debug level."""

    def __init__(self):
        self.pending = ""

    def write(self, text):
        *lines, self.pending = (self.pending + text).split("\n")
        for line in lines:
            if line.strip():
                logger.debug(line)
        return len(text)


def _cube_metadata(lazy_xds, variable, beam_variable, telescope_name):
    frequency = lazy_xds.frequency.values
    l_values, m_values = lazy_xds.l.values, lazy_xds.m.values
    direction = lazy_xds.attrs["coordinate_system_info"]["reference_direction"]
    beams = None
    if beam_variable is not None:
        beam = lazy_xds[beam_variable]
        if beam.attrs.get("units", "rad") != "rad":
            raise ValueError(f"{beam_variable} must be in radians")
        beams = beam.sel(polarization="I").values[0].tolist()
    return {
        "shape": [l_values.size, m_values.size, 1, frequency.size],
        "increment": [
            float(l_values[1] - l_values[0]),
            float(m_values[1] - m_values[0]),
            1.0,
            float(frequency[1] - frequency[0]) if frequency.size > 1 else 1.0,
        ],
        "reference_value": [
            float(direction["data"][0]),
            float(direction["data"][1]),
            1.0,
            float(frequency[0]),
        ],
        "reference_pixel": [
            float(np.argmin(np.abs(l_values))),
            float(np.argmin(np.abs(m_values))),
            0.0,
            0.0,
        ],
        "rest_frequency": float(lazy_xds.frequency.attrs["rest_frequency"]["data"]),
        "spectral_frame": lazy_xds.frequency.attrs["frame"],
        "direction_frame": "J2000"
        if direction["attrs"]["frame"].lower() == "fk5"
        else direction["attrs"]["frame"].upper(),
        "beams": beams,
        "brightness_unit": lazy_xds[variable].attrs.get("units", ""),
        "telescope": telescope_name,
        "object": "",
        "epoch_mjd": None,
    }


@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def find_continuum(
    input_image_store: str,
    atmospheric_transmission,
    atmospheric_sky_temperature,
    work_directory: str,
    image_data_group_in_name: str = "residual",
    telescope_name: str = "ALMA",
    single_continuum: bool = False,
    n_bin: int = 1,
    spectral_dynamic_range_bandwidth: float | None = None,
    primary_beam_limit: float | None = 0.2,
    n_mapping_parallelism: dict | None = None,
    *,
    input_checked: bool = False,
) -> dict:
    """Find the continuum channels of a dirty cube as the ALMA pipeline does.

    Parameters
    ----------
    input_image_store : str
        xradio image store of the dirty cube (Stokes I is used).
    atmospheric_transmission : list or numpy.ndarray
        Atmospheric transmission at every channel (0 to 1), as CASA's
        atmosphere model gives it for the observation.
    atmospheric_sky_temperature : list or numpy.ndarray
        Sky brightness temperature at every channel (K).
    work_directory : str
        Directory for findContinuum's plots, spectra and intermediate images.
        It must not hold the products of an earlier run on a cube of the same
        name: findContinuum would reuse them, as in the pipeline.
    image_data_group_in_name : str, default "residual"
        Image data group with the roles ``sky``, ``primary_beam``,
        ``point_spread_function`` and ``beam_fit_params_point_spread_function``.
    telescope_name : str, default "ALMA"
        Telescope, as findContinuum reads it from a CASA image header.
    single_continuum : bool, default False
        The spectral window is a single-continuum setup [findContinuum
        ``singleContinuum``].
    n_bin : int, default 1
        Channels to average in the mean spectrum [findContinuum ``nbin``].
    spectral_dynamic_range_bandwidth : float, optional
        The requested spectral dynamic range bandwidth (Hz) [findContinuum
        ``spectralDynamicRangeBandWidth``].
    primary_beam_limit : float, default 0.2
        Pixels where the primary beam is below this level are masked in the
        dirty cube and the primary beam, as in the cube ``hif_findcont`` makes
        with ``tclean(pblimit=0.2)``; ``None`` reads the cubes as they are.
    n_mapping_parallelism : dict, optional
        ``{"frequency": n}``, the number of frequency chunks of every
        map-reduce; one chunk if not given.
    input_checked : bool, default False
        Skip the configuration checks already done by the calling layer. The
        checks of the opened image and of the work directory always run.

    Returns
    -------
    dict
        ``channel_selection`` (CASA channel ranges, ``"a~b;c~d"``),
        ``aggregate_bandwidth`` (GHz), ``all_continuum``, ``warnings``,
        ``joint_mask`` (path of the final joint mask), ``moment_difference_snr``,
        ``plot`` (path of the summary plot), ``n_cube_reductions`` (map-reduce
        passes over the image store), ``cube_reads`` (data variable to the
        number of passes that read it) and ``timing_node_tasks``.

    Raises
    ------
    ValueError
        If the configuration or the image fails the checks of
        :mod:`~astroviper.processing_functions.imaging.config_check.param_check_find_continuum`,
        or ``work_directory`` holds the products of an earlier run.

    Notes
    -----
    findContinuum runs in ``work_directory`` (the working directory of the
    process changes while it runs) and draws with pyplot, so a process runs
    one call at a time; spectral windows run in parallel in separate
    processes.
    """
    import dask
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    from graphviper.graph_tools import generate_dask_workflow, map, reduce
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )

    import astroviper.node_tasks.imaging.find_continuum as node_tasks
    from astroviper.distributed_applications.imaging.find_continuum import (
        casa_image_tasks,
        findContinuum,
    )
    from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
        check_find_continuum_config,
        check_find_continuum_inputs,
    )
    from astroviper.processing_functions.imaging.find_continuum import (
        finalize_find_continuum_reductions,
    )

    if not input_checked:
        check_find_continuum_config(n_bin, primary_beam_limit, n_mapping_parallelism)
    # findContinuum runs in its work directory, as in the pipeline, and node
    # tasks may run elsewhere, so every path is made absolute.
    input_image_store = os.path.abspath(input_image_store)
    work_directory = os.path.abspath(work_directory)
    stem = os.path.join(
        work_directory, os.path.basename(input_image_store).split(".")[0]
    )
    if glob.glob(glob.escape(stem) + ".*"):
        raise ValueError(
            f"{work_directory} holds the products of an earlier run on "
            f"{os.path.basename(stem)}, which findContinuum would reuse"
        )
    lazy_xds = xr.open_zarr(input_image_store, chunks=None)
    check_find_continuum_inputs(
        lazy_xds,
        image_data_group_in_name,
        atmospheric_transmission,
        atmospheric_sky_temperature,
    )
    data_group = lazy_xds.attrs["data_groups"][image_data_group_in_name]
    nchan = lazy_xds.sizes["frequency"]

    os.makedirs(work_directory, exist_ok=True)
    beam_variable = data_group["beam_fit_params_point_spread_function"]
    cubes = {}
    for suffix, role, beam in (
        (".residual", "sky", beam_variable),
        (".pb", "primary_beam", None),
        (".psf", "point_spread_function", beam_variable),
    ):
        path = stem + suffix
        os.makedirs(path, exist_ok=True)
        cubes[path] = {
            "store": input_image_store,
            "variable": data_group[role],
            "metadata": _cube_metadata(
                lazy_xds, data_group["sky"], beam, telescope_name
            ),
        }
    cubes[stem + ".residual"]["atmosphere"] = {
        "transmission": np.asarray(atmospheric_transmission, dtype=np.float64),
        "tsky": np.asarray(atmospheric_sky_temperature, dtype=np.float64),
    }

    index_xds = lazy_xds.assign_coords(frequency=np.arange(nchan, dtype=np.float64))
    n_chunks = (
        1 if n_mapping_parallelism is None else n_mapping_parallelism["frequency"]
    )
    parallel_coords = {
        "frequency": make_parallel_coord(coord=index_xds.frequency, n_chunks=n_chunks)
    }
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, {"img": index_xds}
    )
    timing, cube_reads = [], {}
    primary_beam = data_group["primary_beam"]

    def run_reductions(requests):
        variables, image_paths = {}, {}
        for request in requests:
            variables[request["image"]] = cubes[request["image"]]["variable"]
            small_images, mask_cubes = casa_image_tasks._mask_references(
                request["mask"]
            )
            for name in mask_cubes:
                variables[name] = cubes[name]["variable"]
            image_paths.update({name: os.path.abspath(name) for name in small_images})
        read = set(variables.values())
        if primary_beam_limit is not None:
            read.add(primary_beam)
        for variable in read:
            cube_reads[variable] = cube_reads.get(variable, 0) + 1
        viper_graph = map(
            input_data={"img": index_xds},
            node_task_data_mapping=node_task_data_mapping,
            node_task=node_tasks.find_continuum_cube_reductions,
            input_params={
                "input_image_store": input_image_store,
                "variables": variables,
                "image_paths": image_paths,
                "requests": requests,
                "primary_beam": primary_beam,
                "primary_beam_limit": primary_beam_limit,
            },
            in_memory_compute=False,
        )
        viper_graph = reduce(
            viper_graph,
            node_tasks.merge_find_continuum_cube_reductions,
            {},
            mode="tree",
        )
        start = time.time()
        output = dask.compute(generate_dask_workflow(viper_graph))[0]
        logger.debug(
            f"find_continuum reduction {[r['kind'] for r in requests]} in {time.time() - start:.2f} s"
        )
        timing.append(output["timing_node_tasks"].assign(reduction=len(timing)))
        return finalize_find_continuum_reductions(output["reductions"], requests)

    figures = set(plt.get_fignums())
    log_stream = _LogStream()
    try:
        with (
            contextlib.chdir(work_directory),
            contextlib.redirect_stdout(log_stream),
            casa_image_tasks.cube_session(cubes, run_reductions),
        ):
            result = findContinuum.findContinuum(
                img=stem + ".residual",
                pbcube=stem + ".pb",
                psfcube=stem + ".psf",
                singleContinuum=single_continuum,
                returnAllContinuumBoolean=True,
                returnWarnings=True,
                vis="",
                nbin=n_bin,
                spectralDynamicRangeBandWidth=spectral_dynamic_range_bandwidth,
                returnMomDiffSNR=True,
            )
    finally:
        log_stream.write("\n")
        for number in set(plt.get_fignums()) - figures:
            plt.close(number)
    (
        selection,
        plot,
        aggregate_bandwidth,
        all_continuum,
        warnings,
        joint_mask,
        moment_difference_snr,
    ) = result
    return {
        "channel_selection": selection,
        "aggregate_bandwidth": aggregate_bandwidth,
        "all_continuum": bool(all_continuum),
        "warnings": list(warnings),
        "joint_mask": joint_mask,
        "moment_difference_snr": moment_difference_snr,
        "plot": plot,
        "n_cube_reductions": len(timing),
        "cube_reads": cube_reads,
        "timing_node_tasks": pd.concat(timing, ignore_index=True)
        if timing
        else pd.DataFrame(),
    }
