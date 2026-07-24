"""Distributed single-field continuum imaging application.

The notebook-facing setup remains outside, while graph construction and execution
live in :func:`image_continuum_single_field`.
"""

import os
from typing import Any

import toolviper.utils.parameter
import zarr
from numcodecs import Blosc

import astroviper.node_tasks as node_tasks
from astroviper.utils.param_docs import shares_param_docs

_PARAM_CONFIG_DIR = os.path.dirname(__file__)

DISTRIBUTED_APPLICATION_TIMING_PHASES = [
    (
        "DISTRIBUTED APPLICATION (driver)",
        None,
        [
            ("create empty image xds", "T_make_empty_image_xds"),
            ("write empty image to disk", "T_write_empty_image"),
            (
                "determine chunks + parallel coords",
                "T_determine_chunks_and_parallel_coords",
            ),
            ("create empty data vars on disk", "T_create_empty_data_variables"),
            ("open processing set", "T_open_processing_set"),
            ("interpolate data coords", "T_interpolate_data_coords"),
            ("create map/reduce/append graph", "T_create_map_reduce_graph"),
            ("generate dask graph", "T_generate_dask_graph"),
            ("compute graph", "T_compute_dask_graph"),
            ("consolidate metadata", "T_consolidate_metadata"),
        ],
    ),
]

DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY = "T_total"

###############################################################################
# The main functions to build and reduce graphviper graphs
###############################################################################


def compute_continuum_graph(
    *,
    ps_xdt,
    node_task_data_mapping,
    cycle_input_params,
    reduce_input_params,
    disk_chunk_sizes,
    processing_set_data_group_name,
    monitor_resources_seconds,
    task_priorities,
    reduce_mode,
    reduce_n_batch,
    append_node=None,
    append_input_params=None,
):
    """The graph performs

    map -> reduce -> append_node

     Parameters
     ----------
     ps_xdt
         Processing set used as the map input.
     node_task_data_mapping
         Mapping between processing-set coordinates and map-task coordinates.
     cycle_input_params : dict
         Parameters forwarded to each residual major-cycle map task.
     reduce_input_params : dict
         Parameters forwarded to :func:`combine_continuum_chunks`.
     disk_chunk_sizes : dict or None
         Native disk-level chunk sizes used by the GraphVIPER map stage.
     processing_set_data_group_name : str
         Processing-set data group loaded by each map task.
     monitor_resources_seconds : float or None
         Resource-monitor sampling interval for map tasks.
     task_priorities
         Optional GraphVIPER task priorities.
     reduce_mode : str
         GraphVIPER reduction mode.
     reduce_n_batch : int
         Number of inputs combined per reduction batch.
     append_node : callable, optional
         Global node executed after reduction.
     append_input_params : dict, optional
         Parameters forwarded to ``append_node``.

     Returns
     -------
     dict
         Computed result of the map/reduce or map/reduce/append graph.

     Raises
     ------
     ValueError
         If only one of ``append_node`` and ``append_input_params`` is supplied.
    """
    import time

    import dask
    from graphviper.graph_tools import append, generate_dask_workflow, map, reduce

    # Some sanity checks
    if append_node is None and append_input_params is not None:
        raise ValueError("append_input_params was supplied without an append_node.")

    if append_node is not None and append_input_params is None:
        raise ValueError("append_node was supplied without append_input_params.")

    timings = {}

    start = time.time()

    # Mapping stage: Residual update (calculation of taylor order uv grids)
    viper_graph = map(
        input_data=ps_xdt,
        node_task_data_mapping=node_task_data_mapping,
        node_task=node_tasks.imaging.residual_update_continuum_single_field,
        input_params=cycle_input_params,
        in_memory_compute=False,
        data_loading_task=None,
        disk_chunk_sizes=disk_chunk_sizes,
        load_node_input_params={
            "processing_set_data_group_name": processing_set_data_group_name,
        },
        monitor_resources_seconds=monitor_resources_seconds,
        task_priorities=task_priorities,
    )

    # Reduce stage: Combine uv grids
    viper_graph = reduce(
        viper_graph,
        combine_continuum_chunks,
        reduce_input_params,
        mode=reduce_mode,
        n_batch=reduce_n_batch,
    )

    # Append node: Either minor cycle or finalization
    if append_node is not None:
        viper_graph = append(
            viper_graph,
            append_node,
            append_input_params,
        )

    timings["T_create_map_reduce_append_graph"] = time.time() - start

    start = time.time()
    dask_graph = generate_dask_workflow(viper_graph)
    timings["T_generate_dask_graph"] = time.time() - start

    start = time.time()
    graph_result = dask.compute(dask_graph)[0]
    timings["T_compute_dask_graph"] = time.time() - start

    return graph_result, timings


def combine_continuum_chunks(input_data, input_params):
    """Combine frequency-chunk continuum map results.

    This function is intended for use as the ``combine`` function passed to
    :func:`graphviper.graph_tools.reduce`. It is associative: the returned object
    has the same structure as each input element, allowing reductions using
    ``mode="tree"``, ``mode="tree_n"``, or ``mode="single_node"``.

    Each input element is expected to have the form

    .. code-block:: python

        {
            "image": img_xds,
            "timing_node_tasks": timing_df,
            "deconvolution": deconvolution_return_dict,
        }

    where ``img_xds`` contains the continuum imaging products produced by one
    frequency chunk.

    By default, the reducer performs an element-wise sum of the additive image
    products

    * ``VISIBILITY``;
    * ``VISIBILITY_NORMALIZATION``;
    * ``UV_SAMPLING``;
    * ``UV_SAMPLING_NORMALIZATION``.

    With the current continuum implementation these variables have dimensions

    ``VISIBILITY``
        ``(time, taylor_term, polarization, u, v)``

    ``VISIBILITY_NORMALIZATION``
        ``(time, taylor_term, polarization)``

    ``UV_SAMPLING``
        ``(time, psf_taylor_order, polarization, u, v)``

    ``UV_SAMPLING_NORMALIZATION``
        ``(time, psf_taylor_order, polarization)``.

    All remaining dataset variables (for example metadata, coordinates, primary
    beam products, and other non-additive quantities) are copied from the first
    input dataset after verifying consistency across all inputs.

    Parameters
    ----------
    input_data : list of dict
        Leaf map-task results or partially reduced results.

    input_params : dict, optional
        Optional reducer configuration. Supported entries are

        ``additive_variables`` : sequence of str, optional
            Dataset variables that are accumulated by element-wise addition.
            Defaults to

            ``("VISIBILITY", "VISIBILITY_NORMALIZATION",
            "UV_SAMPLING", "UV_SAMPLING_NORMALIZATION")``.

        ``strict`` : bool, optional
            If ``True``, every additive variable must exist in every input and
            dimensions, coordinates, and selected metadata must agree exactly.
            Defaults to ``True``.

        ``copy_image_deep`` : bool, optional
            If ``True``, the first image dataset is deep-copied before
            accumulation. This avoids modifying an input object at the cost of
            additional temporary memory. Defaults to ``True``.

    Returns
    -------
    dict
        Dictionary with the same structure as a map-task result.

        ``"image"``
            Dataset containing the accumulated continuum products.

        ``"timing_node_tasks"``
            Concatenated timing dataframe containing one row per original map
            task.

        ``"deconvolution"``
            Combined deconvolution metadata.

    Notes
    -----
    This reducer performs an unnormalized accumulation of continuum products.
    Any normalization by imaging weights, sum-of-weights, or Hessian terms is
    performed later, after all frequency chunks have been reduced.

    The reducer assumes that all inputs were produced using identical continuum
    imaging parameters (for example, reference frequency, Taylor expansion, and
    image geometry). These metadata are validated when available before
    accumulation.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.processing_functions.imaging.utils.iteration_control import (
        merge_return_dicts,
    )

    if input_params is None:
        input_params = {}

    additive_variables = tuple(
        input_params.get(
            "additive_variables",
            (
                "VISIBILITY",
                "VISIBILITY_NORMALIZATION",
                "UV_SAMPLING",
                "UV_SAMPLING_NORMALIZATION",
            ),
        )
    )
    strict = bool(input_params.get("strict", True))
    copy_image_deep = bool(input_params.get("copy_image_deep", True))

    if not input_data:
        raise ValueError("combine_continuum_chunks received no inputs.")

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    # Kept inline due ti their shortness
    # returns dictionary of metadata
    def _continuum_metadata(dataset):
        metadata = dataset.attrs.get("continuum_imaging", {})

        return {
            "nterms": metadata.get(
                "nterms",
                dataset.attrs.get("nterms"),
            ),
            "reference_frequency_hz": metadata.get(
                "reference_frequency_hz",
                dataset.attrs.get("reference_frequency_hz"),
            ),
            "n_psf_taylor_terms": metadata.get(
                "n_psf_taylor_terms",
                dataset.attrs.get("n_psf_taylor_terms"),
            ),
        }

    # validate metadata
    def _validate_metadata(reference_dataset, candidate_dataset, input_index):
        reference = _continuum_metadata(reference_dataset)
        candidate = _continuum_metadata(candidate_dataset)

        for key in (
            "nterms",
            "n_psf_taylor_terms",
        ):
            reference_value = reference[key]
            candidate_value = candidate[key]

            if (
                reference_value is not None
                and candidate_value is not None
                and int(reference_value) != int(candidate_value)
            ):
                raise ValueError(
                    f"Continuum metadata mismatch for {key!r}: "
                    f"reference={reference_value}, "
                    f"input[{input_index}]={candidate_value}."
                )

        reference_frequency = reference["reference_frequency_hz"]
        candidate_frequency = candidate["reference_frequency_hz"]

        if (
            reference_frequency is not None
            and candidate_frequency is not None
            and not np.isclose(
                float(reference_frequency),
                float(candidate_frequency),
                rtol=1.0e-12,
                atol=0.0,
            )
        ):
            raise ValueError(
                "All continuum chunks must use the same reference frequency: "
                f"reference={reference_frequency}, "
                f"input[{input_index}]={candidate_frequency}."
            )

    # validate that additive variables match in dimensions
    def _validate_additive_variable(
        reference_array,
        candidate_array,
        variable_name,
        input_index,
    ):
        if reference_array.dims != candidate_array.dims:
            raise ValueError(
                f"Dimension mismatch for {variable_name!r}: "
                f"reference={reference_array.dims}, "
                f"input[{input_index}]={candidate_array.dims}."
            )

        if reference_array.shape != candidate_array.shape:
            raise ValueError(
                f"Shape mismatch for {variable_name!r}: "
                f"reference={reference_array.shape}, "
                f"input[{input_index}]={candidate_array.shape}."
            )

        # join="exact" catches mismatched Taylor, spatial, polarization, and
        # time coordinates while avoiding silent coordinate reindexing.
        try:
            xr.align(
                reference_array,
                candidate_array,
                join="exact",
                copy=False,
            )
        except ValueError as exc:
            raise ValueError(
                f"Coordinate mismatch for {variable_name!r} in "
                f"input[{input_index}]."
            ) from exc

    # ------------------------------------------------------------------
    # Timing and deconvolution metadata
    # ------------------------------------------------------------------

    # Concatenate timing and deconvolution return dictionaries
    combined_timing = pd.DataFrame()
    deconvolution_dicts = []

    for result in input_data:
        if "timing_node_tasks" not in result:
            raise KeyError(
                "Every continuum map/reduce result must contain " "'timing_node_tasks'."
            )

        timing = result["timing_node_tasks"]

        # Preserve GraphViper's optional leaf-level resource monitor output in
        # the same way as the existing cube reducer. Partially reduced inputs
        # have already folded this information into their timing dataframe.
        resource_usage = result.get("resource_usage")

        if resource_usage is not None:
            timing = timing.copy()

            for key, value in resource_usage.items():
                timing[key] = [value] if isinstance(value, list) else value

        combined_timing = pd.concat(
            [combined_timing, timing],
            ignore_index=True,
        )

        if "deconvolution" in result:
            deconvolution_dicts.append(result["deconvolution"])

    # ------------------------------------------------------------------
    # Taylor-image reduction
    # ------------------------------------------------------------------

    first_result = input_data[0]

    if "image" not in first_result:
        raise KeyError(
            "Every continuum map/reduce result must contain an 'image' dataset."
        )

    first_image = first_result["image"]

    if not isinstance(first_image, xr.Dataset):
        raise TypeError(
            "result['image'] must be an xarray.Dataset; received "
            f"{type(first_image).__name__}."
        )

    # Make a copy for the combined image
    # In this way, dimensions, shapes, metadata and coordinates are going to be correct
    combined_image = first_image.copy(deep=copy_image_deep)

    for variable_name in additive_variables:
        if variable_name not in combined_image:
            if strict:
                raise KeyError(
                    f"Additive variable {variable_name!r} is absent from the "
                    "first continuum image."
                )
            continue

        # Ensure the accumulator owns a writable array. This is particularly
        # relevant when copy_image_deep=False or when an input is backed by a
        # read-only array.
        combined_image[variable_name] = combined_image[variable_name].copy(deep=True)

    # Main loop: Loop over input data to combine
    for input_index, result in enumerate(input_data[1:], start=1):
        if "image" not in result:
            raise KeyError(f"input[{input_index}] does not contain an 'image' dataset.")

        candidate_image = result["image"]

        # Sanity checks
        if not isinstance(candidate_image, xr.Dataset):
            raise TypeError(
                f"input[{input_index}]['image'] must be an xarray.Dataset; "
                f"received {type(candidate_image).__name__}."
            )

        _validate_metadata(
            combined_image,
            candidate_image,
            input_index,
        )

        # combine for every additive variable
        for variable_name in additive_variables:
            accumulator_has_variable = variable_name in combined_image
            candidate_has_variable = variable_name in candidate_image

            if not accumulator_has_variable or not candidate_has_variable:
                if strict:
                    raise KeyError(
                        f"Additive variable {variable_name!r} must be present "
                        f"in every input. Present in accumulator: "
                        f"{accumulator_has_variable}; present in "
                        f"input[{input_index}]: {candidate_has_variable}."
                    )
                continue

            accumulator = combined_image[variable_name]
            candidate = candidate_image[variable_name]

            _validate_additive_variable(
                accumulator,
                candidate,
                variable_name,
                input_index,
            )

            # xarray keeps dimension labels and coordinates while performing
            # the numerical sum. Assignment avoids relying on in-place
            # behavior for lazy, read-only, or non-NumPy-backed arrays.
            combined_image[variable_name] = accumulator + candidate

    # Record how many original map-task rows contributed to this partial or
    # complete reduction. Because timing rows are concatenated at each tree
    # level, this remains correct for partially reduced inputs.
    combined_image.attrs["n_continuum_chunks_combined"] = int(len(combined_timing))
    combined_image.attrs["continuum_additive_variables"] = list(additive_variables)

    if deconvolution_dicts:
        combined_deconvolution = merge_return_dicts(deconvolution_dicts)
    else:
        # Keep the output schema stable even when no input supplied
        # deconvolution metadata.
        from astroviper.processing_functions.imaging.utils.return_dict import ReturnDict

        combined_deconvolution = ReturnDict()

    return {
        "image": combined_image,
        "timing_node_tasks": combined_timing,
        "deconvolution": combined_deconvolution,
    }


###############################################################################
# Generic Helper Functions
###############################################################################


def calculate_number_of_chunks_for_continuum_imaging(
    img_xds, single_precision_image, n_chunks, thread_info
):
    """Determine the number of frequency chunks for continuum imaging.

    Computes the memory required per single-frequency chunk and delegates to
    :func:`calculate_data_chunking` to find a chunk count that satisfies both
    memory and parallelism constraints. If ``n_chunks`` is already provided it
    is returned unchanged.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Empty image dataset whose ``sizes`` attribute provides the grid dimensions.
    single_precision_image : bool
        If ``True``, use single-precision (complex64 / float32) memory estimates
        for the image-domain arrays; otherwise double-precision
        (complex128 / float64).
    n_chunks : int or None
        If not ``None``, this value is returned directly without any computation.
    thread_info : dict or None
        Thread information as returned by :func:`get_thread_info`.
        If ``None``, thread information is queried automatically.

    Returns
    -------
    int
        Number of frequency chunks to use for the parallel imaging graph.
    """
    import toolviper.utils.logger as logger

    if n_chunks is None:
        # Calculate n_chunks
        from astroviper.utils.data_partitioning import bytes_in_dtype

        ## Determine the amount of memory required by the node task if all dimensions that chunking will occur on are singleton.
        ## For example cube_imaging does chunking only only frequency, so memory_singleton_chunk should be the amount of memory requered by _feather when there is a single frequency channel.

        n_pixels_single_frequency = (
            img_xds.sizes["l"]
            * img_xds.sizes["m"]
            * img_xds.sizes["polarization"]
            * img_xds.sizes["time"]
        )
        fudge_factor = 1.2
        if single_precision_image:
            memory_singleton_chunk = fudge_factor * (
                3
                * n_pixels_single_frequency
                * bytes_in_dtype["complex64"]
                / (1024**3)
                + 3
                * n_pixels_single_frequency
                * bytes_in_dtype["float32"]
                / (1024**3)
            )
        else:
            memory_singleton_chunk = fudge_factor * (
                3
                * n_pixels_single_frequency
                * bytes_in_dtype["complex128"]
                / (1024**3)
                + 3
                * n_pixels_single_frequency
                * bytes_in_dtype["float64"]
                / (1024**3)
            )

        logger.info(
            "Memory required for a single frequency channel: "
            + str(memory_singleton_chunk)
            + " GiB"
        )

        chunking_dims_sizes = {
            "frequency": img_xds.sizes["frequency"]
        }  # Need to know how many frequency channels there are.
        from astroviper.utils.data_partitioning import (
            calculate_data_chunking,
            get_thread_info,
        )

        if thread_info is None:
            thread_info = get_thread_info()
            logger.info("Thread info " + str(thread_info))
        n_chunks = calculate_data_chunking(
            memory_singleton_chunk,
            chunking_dims_sizes,
            thread_info,
            constant_memory=0,
            tasks_per_thread=4,
        )["frequency"]
        logger.info(
            "Number of frequency chunks: "
            + str(n_chunks)
            + " frequency channels: "
            + str(chunking_dims_sizes)
        )
    return n_chunks


###############################################################################
# Main distributed layer level function call
###############################################################################


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def image_continuum_single_field(
    ps_store: str,
    image_store: str,
    image_params: dict[str, Any],
    imaging_weights_params: dict[str, Any],
    iteration_control_params: dict[str, Any],
    gridder: str = "prolate_spheroidal",
    deconvolver: str = "hogbom",
    instrument_polarization_basis: str = "linear",
    scan_intents: list[str] = ["OBSERVE_TARGET#ON_SOURCE"],
    field_name: str | None = None,
    image_data_variables_keep: list[str] = [
        "sky_residual",
        "point_spread_function",
        "primary_beam",
        "beam_fit_params_point_spread_function",
    ],
    compressor=Blosc(cname="lz4", clevel=5),
    processing_set_data_group_name: str = "base",
    single_precision_image: bool = False,
    thread_info: dict | None = None,
    processing_function_threads: int = 1,
    n_chunks: int | None = None,
    overwrite: bool = False,
    memory_mode: str = "in_memory",
    cache_directory: str | None = None,
    write_visibility_model_to_ps: bool = False,
    write_imaging_weights_to_ps: bool = False,
    clear_cache: bool = True,
    vizualize_graph: bool = False,
    disk_chunk_sizes: dict[str, int] | str | None = None,
    fft_backend: str = "pyfftw",
    restore: bool = False,
    skunk_works: bool = False,
    compute_backend: str = "dask",
    mpi_cluster_setup: dict[str, Any] | None = None,
    reduce_mode: str = "tree",
    reduce_n_batch: int = 2,
    output_shard_channels: int | None = None,
    task_time_kill_switch_seconds: float | None = None,
    monitor_resources_seconds: float | None = None,
) -> dict:
    """
    Distributed MT-MFS continuum imaging.

    Pipeline
    --------

    Initialization
        - prepare static imaging quantities
        - build Dask graph

    Major cycle
        - predict model visibilities
        - compute residual visibilities
        - grid Taylor residuals
        - reduce across frequency partitions
        - inverse FFT
        - minor cycle

    Finalization
        - final residual image
        - restoration
        - write products

    Unlike cube imaging, FFTs are performed only once after each
    minor cycle. Workers operate directly on UV-domain Taylor grids.
    """
    import time

    import toolviper.utils.logger as logger
    from graphviper.graph_tools.coordinate_utils import (
        get_disk_chunk_sizes,
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )
    from xradio.image import make_empty_sky_image, write_image
    from xradio.measurement_set import open_processing_set

    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        prepare_model_uv_continuum_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        IMAGING_TIMING_PHASES,
        IMAGING_TIMING_TOTAL_KEY,
        IterationController,
    )
    from astroviper.utils.data_group_tools import modify_data_groups_xds
    from astroviper.utils.io import (
        create_empty_data_variables_on_disk,
        image_data_groups_for_kept_variables,
    )
    from astroviper.utils.timing import format_timing_summary

    assert (
        memory_mode == "in_memory"
    ), "Currently only in_memory is supported for memory_mode is implemented."

    # Sharded output is written by the concurrent direct-blob (skunk_works) writer;
    # the standard write path cannot safely write partial shards concurrently, so
    # creating sharded arrays without it would corrupt the output. Fail fast rather
    # than silently create sharded arrays a non-concurrent writer will clobber.
    if output_shard_channels is not None and not skunk_works:
        raise ValueError(
            "output_shard_channels requires skunk_works=True (sharded output is "
            "written by the concurrent direct-blob writer)."
        )

    # When restoring, the restored sky must be created on disk and written, so
    # ensure it is in the keep list (without mutating the caller's list).
    if restore and "sky_restored" not in image_data_variables_keep:
        image_data_variables_keep = list(image_data_variables_keep) + ["sky_restored"]

    # Every driver step is timed into ``timing_distributed_application``; the
    # individual per-step timing log messages are replaced by the formatted
    # summary logged just before returning.
    timing_distributed_application = {}
    application_start = time.time()

    # Create an empty image on disk with the correct coordinates and dimensions.
    start = time.time()
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=image_params["frequency_coords"],
        pol_coords=image_params["polarization_coords"],
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )
    timing_distributed_application["T_make_empty_image_xds"] = time.time() - start

    # Register the image data groups for the kept variables so the on-disk
    # store carries the same group layout the node tasks build in memory
    # (make_empty_sky_image only stamps an empty "base" placeholder, which is
    # dropped here in favor of the real groups).
    img_xds.attrs.get("data_groups", {}).pop("base", None)
    for data_group_name, data_group in image_data_groups_for_kept_variables(
        image_data_variables_keep
    ).items():
        modify_data_groups_xds(
            img_xds,
            data_group_out_name=data_group_name,
            data_group_out=data_group,
            description="Created by the image_cube_single_field driver; "
            "populated by its node tasks.",
        )

    start = time.time()
    write_image(img_xds, imagename=image_store, out_format="zarr", overwrite=overwrite)
    timing_distributed_application["T_write_empty_image"] = time.time() - start

    # Determine number of chunks
    start = time.time()
    n_chunks = int(
        calculate_number_of_chunks_for_continuum_imaging(
            img_xds, single_precision_image, n_chunks, thread_info
        )
    )

    # Make Parallel Coords
    parallel_coords = {}
    parallel_coords["frequency"] = make_parallel_coord(
        coord=img_xds.frequency, n_chunks=n_chunks
    )
    logger.info(
        "Number of frequency chunks ... : "
        + str(len(parallel_coords["frequency"]["data_chunks"]))
    )
    timing_distributed_application["T_determine_chunks_and_parallel_coords"] = (
        time.time() - start
    )

    # Add nan images (these will be overwritten with the actual image data but this ensures the coordinates and dtypes are correct and allows for lazy writing of the data)
    # create_empty_data_varable_on_disk(zarr_store, dv_names, dims, shape, chunk, variable_dtype, compressor)
    start = time.time()
    create_empty_data_variables_on_disk(
        image_store,
        image_data_variables_keep,
        shape_dict=img_xds.sizes,
        parallel_coords=parallel_coords,
        compressor=compressor,
        double_precision=not single_precision_image,
        data_variable_definitions="imaging",
        shard_channels=output_shard_channels,
    )
    timing_distributed_application["T_create_empty_data_variables"] = (
        time.time() - start
    )

    zarr_meta = {}

    input_params = {}
    input_params["image_params"] = image_params
    input_params["imaging_weights_params"] = imaging_weights_params
    input_params["zarr_meta"] = zarr_meta
    input_params["to_disk"] = True
    input_params["polarization"] = img_xds.polarization.data
    input_params["time"] = [0]
    input_params["compressor"] = compressor
    input_params["image_store"] = image_store
    input_params["input_data_store"] = ps_store
    input_params["processing_set_data_group_name"] = processing_set_data_group_name
    input_params["image_data_variables_keep"] = image_data_variables_keep
    input_params["memory_mode"] = memory_mode
    input_params["cache_directory"] = cache_directory
    input_params["write_visibility_model_to_ps"] = write_visibility_model_to_ps
    input_params["write_imaging_weights_to_ps"] = write_imaging_weights_to_ps
    input_params["clear_cache"] = clear_cache
    input_params["processing_function_threads"] = processing_function_threads
    input_params["iteration_control_params"] = iteration_control_params
    input_params["gridder"] = gridder
    input_params["is_n_iter_0"] = True
    input_params["deconvolver"] = deconvolver
    input_params["instrument_polarization_basis"] = instrument_polarization_basis
    input_params["single_precision_image"] = single_precision_image
    input_params["fft_backend"] = fft_backend
    input_params["restore"] = restore
    input_params["skunk_works"] = skunk_works
    input_params["output_shard_channels"] = output_shard_channels
    input_params["task_time_kill_switch_seconds"] = task_time_kill_switch_seconds

    controller = IterationController(
        niter=iteration_control_params["niter"],
        nmajor=iteration_control_params["nmajor"],
        threshold=iteration_control_params["threshold"],
        gain=iteration_control_params["gain"],
        cyclefactor=iteration_control_params["cyclefactor"],
        minpsffraction=iteration_control_params["minpsffraction"],
        maxpsffraction=iteration_control_params["maxpsffraction"],
        cycleniter=iteration_control_params["cycleniter"],
    )

    start = time.time()
    ps_xdt = open_processing_set(ps_store, scan_intents=scan_intents)
    timing_distributed_application["T_open_processing_set"] = time.time() - start

    # The skunk-works node-task I/O path reconstructs the processing set from the
    # data group's variables only, so it needs the resolved role->variable
    # mapping. Read it once here (from the first MS) and forward it to every
    # node task rather than re-reading it per task.
    if skunk_works:
        first_ms = next(iter(ps_xdt.values()))
        input_params["data_group"] = first_ms.ds.attrs["data_groups"][
            processing_set_data_group_name
        ]

    start = time.time()
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps_xdt
    )
    timing_distributed_application["T_interpolate_data_coords"] = time.time() - start

    # Auto-detect native on-disk chunk sizes if not supplied by the caller.
    if disk_chunk_sizes == "Auto":
        disk_chunk_sizes = get_disk_chunk_sizes(ps_xdt, parallel_coords)
        logger.info("Auto-detected disk chunk sizes: " + str(disk_chunk_sizes))
    elif isinstance(disk_chunk_sizes, str):
        # If disk_chunk_sizes is a string but not "Auto", treat as None
        disk_chunk_sizes = None

    # frequency_coords is not used by node tasks (they use task_coords["frequency"]["data"])
    # so remove it to avoid embedding the full frequency axis in every task in the graph.
    input_params["image_params"] = {
        k: v for k, v in image_params.items() if k != "frequency_coords"
    }

    # Sharded output: dispatch the tasks in shard-interleaved "waves" (each wave
    # touches every shard file once) so the concurrently running tasks write to
    # all shard files -- and hence all Lustre OSTs -- instead of piling onto the
    # few shards that consecutive task_ids share. Derived from the on-disk shard
    # layout of the first kept variable, for any combination of sharded dims.
    task_priorities = None
    if skunk_works and output_shard_channels:
        from astroviper.node_tasks.imaging.utils import compute_shard_task_priorities

        task_priorities = compute_shard_task_priorities(
            image_store, image_data_variables_keep[0], node_task_data_mapping
        )

    # =============================================================
    # Distributed major/minor-cycle loop
    # =============================================================

    # These timing entries accumulate over all major cycles
    timing_distributed_application["T_create_map_reduce_append_graph"] = 0.0

    timing_distributed_application["T_generate_dask_graph"] = 0.0

    timing_distributed_application["T_compute_dask_graph"] = 0.0

    # State carried between independently computed distributed graphs.
    is_n_iter_0 = True
    static_xds = None
    model_xds = None
    model_uv_xds = None
    last_minor_return_dict = None
    n_major_cycles = 0

    # The IterationController is updated inside the append node. A nonzero
    # major stop code means that the CLEAN loop has converged or reached one
    # of its configured limits.
    while controller.stopcode.major == 0:

        n_major_cycles += 1

        logger.debug(f"Starting continuum major cycle {n_major_cycles}.")

        # ---------------------------------------------------------
        # Configure the distributed residual/major-cycle map tasks.
        # ---------------------------------------------------------
        cycle_input_params = dict(input_params)

        cycle_input_params["is_n_iter_0"] = is_n_iter_0
        cycle_input_params["restore"] = False

        if not is_n_iter_0:
            if model_xds is None:
                raise RuntimeError(
                    "No accumulated continuum model is available for "
                    f"major cycle {n_major_cycles}."
                )

            if model_uv_xds is None:
                raise RuntimeError("No Fourier-domain model is available.")

            if static_xds is None:
                raise RuntimeError(
                    "No static continuum products are available for "
                    f"major cycle {n_major_cycles}."
                )

            cycle_input_params["model_uv_xds"] = model_uv_xds
            cycle_input_params["static_xds"] = static_xds

        # During the first major cycle the PSF and residual Taylor products
        # are reduced. Later cycles only produce new residual products; the
        # static PSF/PB products are supplied by continuum_append_node.
        if is_n_iter_0:
            reduce_input_params = {}
        else:
            reduce_input_params = {
                "additive_variables": (
                    "VISIBILITY",
                    "VISIBILITY_NORMALIZATION",
                ),
            }

        # ---------------------------------------------------------
        # Configure the global continuum minor-cycle append node.
        # ---------------------------------------------------------
        append_input_params = {
            "iteration_control_params": iteration_control_params,
            "deconvolver": deconvolver,
            "processing_function_threads": processing_function_threads,
            "is_n_iter_0": is_n_iter_0,
            "controller": controller,
            "image_data_group_in_name": "residual",
            "image_data_group_out_name": "model",
            "image_params": image_params,
            "image_data_variables_keep": image_data_variables_keep,
            "fft_backend": fft_backend,
            "single_precision_image": single_precision_image,
        }

        # In later major loops, a static_xds should be present
        # This holds static quantities such as PSF and PB
        if not is_n_iter_0:
            append_input_params["static_xds"] = static_xds

        # ---------------------------------------------------------
        # Execute one major cycle followed by one minor cycle.
        # ---------------------------------------------------------

        # Call the graph with continuum_minor_cycle_node
        cycle_return_dict, graph_timings = compute_continuum_graph(
            ps_xdt=ps_xdt,
            node_task_data_mapping=node_task_data_mapping,
            cycle_input_params=cycle_input_params,
            reduce_input_params=reduce_input_params,
            disk_chunk_sizes=disk_chunk_sizes,
            processing_set_data_group_name=processing_set_data_group_name,
            monitor_resources_seconds=monitor_resources_seconds,
            task_priorities=task_priorities,
            reduce_mode=reduce_mode,
            reduce_n_batch=reduce_n_batch,
            append_node=node_tasks.imaging.continuum_minor_cycle_node,
            append_input_params=append_input_params,
        )

        # Gather timing information
        for key, value in graph_timings.items():
            timing_distributed_application[key] += value

        # Get current status for bookkeeping
        last_minor_return_dict = cycle_return_dict
        controller = cycle_return_dict["controller"]

        # ---------------------------------------------------------
        # Initialize or update the state carried to the next cycle.
        # ---------------------------------------------------------
        if is_n_iter_0:
            if "static_xds" not in cycle_return_dict:
                raise KeyError(
                    "The first continuum append node did not return " "'static_xds'."
                )

            # Static holding quantities that are only computed in the first major loop
            # PSF, PB, PSF sidelobe level ...
            static_xds = cycle_return_dict["static_xds"]

            # The first minor-cycle result is the initial accumulated model.
            model_xds = cycle_return_dict["image"][["SKY_MODEL"]].copy(deep=True)

        else:
            # Later minor cycles return a model increment.
            model_increment = cycle_return_dict["image"]["SKY_MODEL"]

            if model_increment.dims != model_xds["SKY_MODEL"].dims:
                raise ValueError(
                    "The continuum model increment dimensions do not match "
                    "the accumulated model: "
                    f"{model_increment.dims} != "
                    f"{model_xds['SKY_MODEL'].dims}."
                )

            if model_increment.shape != model_xds["SKY_MODEL"].shape:
                raise ValueError(
                    "The continuum model increment shape does not match "
                    "the accumulated model: "
                    f"{model_increment.shape} != "
                    f"{model_xds['SKY_MODEL'].shape}."
                )

            # Update only the data so that coordinates such as the Stokes
            # polarization labels remain unchanged.
            accumulated_model = model_xds["SKY_MODEL"]

            # Sum old model and model increment
            model_xds["SKY_MODEL"].data = accumulated_model.data + model_increment.data

            model_xds["SKY_MODEL"].attrs = accumulated_model.attrs.copy()

        # Prepare global Fourier-domain Taylor model grids for degridding
        model_uv_xds = prepare_model_uv_continuum_single_field(
            model_xds,
            image_params=image_params,
            instrument_polarization_basis=instrument_polarization_basis,
            single_precision_image=single_precision_image,
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
        )

        # Update global iteration control information and break loop if converged
        is_n_iter_0 = False

        stopcode = cycle_return_dict["stopcode"]
        stopdesc = cycle_return_dict["stopdesc"]

        if stopcode.major != 0:
            logger.debug(
                "Continuum major/minor-cycle loop stopped after "
                f"{n_major_cycles} major cycles: {stopdesc}"
            )
            break

    if last_minor_return_dict is None:
        raise RuntimeError(
            "The continuum major/minor-cycle loop completed without "
            "executing a minor cycle."
        )

    # =============================================================
    # Final major cycle: recompute residual and restore
    # =============================================================

    assert model_uv_xds is not None
    assert model_xds is not None

    final_input_params = dict(input_params)

    final_input_params["is_n_iter_0"] = False
    final_input_params["restore"] = True
    final_input_params["model_uv_xds"] = model_uv_xds
    final_input_params["model_xds"] = model_xds
    final_input_params["static_xds"] = static_xds

    # Call the graph with continuum_finalize_node
    final_return_dict, graph_timings = compute_continuum_graph(
        ps_xdt=ps_xdt,
        node_task_data_mapping=node_task_data_mapping,
        cycle_input_params=final_input_params,
        reduce_input_params={
            "additive_variables": (
                "VISIBILITY",
                "VISIBILITY_NORMALIZATION",
            ),
        },
        disk_chunk_sizes=disk_chunk_sizes,
        processing_set_data_group_name=processing_set_data_group_name,
        monitor_resources_seconds=monitor_resources_seconds,
        task_priorities=task_priorities,
        reduce_mode=reduce_mode,
        reduce_n_batch=reduce_n_batch,
        append_node=node_tasks.imaging.continuum_finalize_node,
        append_input_params=final_input_params,
    )

    # Gather timing information
    for key, value in graph_timings.items():
        timing_distributed_application[key] += value

    # =============================================================
    # Assemble the final application result
    # =============================================================

    return_dict = final_return_dict

    # The final major-cycle graph computes the final residual/restored image,
    # while the accumulated model comes from all preceding minor cycles.
    return_dict["image"]["SKY_MODEL"] = model_xds["SKY_MODEL"].copy(deep=True)

    # Convergence and deconvolution state come from the last minor cycle,
    # because the final graph contains no model-update append node.
    for key in (
        "controller",
        "deconvolution",
        "stopcode",
        "stopdesc",
        "is_n_iter_0",
    ):
        return_dict[key] = last_minor_return_dict[key]

    return_dict["static_xds"] = static_xds
    return_dict["n_major_cycles"] = n_major_cycles

    # Consolidate metadata
    start = time.time()
    zarr.consolidate_metadata(image_store)
    timing_distributed_application["T_consolidate_metadata"] = time.time() - start

    timing_distributed_application["T_total"] = time.time() - application_start

    # The reduce already produced ``{"timing_node_tasks", "deconvolution"}``; add
    # the driver-level timing so the full return dict carries timing for both the
    # distributed application (this driver) and the per-chunk node tasks.
    return_dict["timing_distributed_application"] = timing_distributed_application

    # Driver-level ("distributed application") timing breakdown.
    logger.info(
        format_timing_summary(
            timing_distributed_application,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            total_key=DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
            title="AstroVIPER distributed-application timing (driver, seconds)",
            total_label="TOTAL (driver wall time)",
        )
    )

    # Per-node-task timing summarized across all frequency chunks: the mean of
    # each timing column over all chunks, then the max (the slowest chunk).
    timing_node_tasks = return_dict["timing_node_tasks"]
    logger.info(
        format_timing_summary(
            timing_node_tasks.mean(numeric_only=True).to_dict(),
            IMAGING_TIMING_PHASES,
            total_key=IMAGING_TIMING_TOTAL_KEY,
            title="AstroVIPER node-task timing: MEAN over frequency chunks (seconds)",
        )
    )
    logger.info(
        format_timing_summary(
            timing_node_tasks.max(numeric_only=True).to_dict(),
            IMAGING_TIMING_PHASES,
            total_key=IMAGING_TIMING_TOTAL_KEY,
            title="AstroVIPER node-task timing: MAX over frequency chunks (seconds)",
        )
    )

    return return_dict
