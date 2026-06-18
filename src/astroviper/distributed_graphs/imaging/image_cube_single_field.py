import os
from numcodecs import Blosc
import xarray as xr
from typing import Optional, Dict, Any, Tuple, Union
from xradio.image import make_empty_sky_image
import numpy as np
import zarr
import toolviper.utils.parameter

import astroviper.node_tasks as node_tasks
from astroviper.utils.param_docs import shares_param_docs

# The toolviper parameter-check schema lives next to this module (rather than in
# a central config/ directory) so it is easy to find; point the validator at it.
_PARAM_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


# Phase layout for the driver-level ("distributed application") timing dict that
# :func:`image_cube_single_field` accumulates and logs before returning.  It is
# consumed by :func:`astroviper.utils.timing.format_timing_summary`: a single
# phase lists every driver step as a leaf and ``T_total`` holds the grand total.
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
            ("create map/reduce graph", "T_create_map_reduce_graph"),
            ("generate dask graph", "T_generate_dask_graph"),
            ("compute dask graph", "T_compute_dask_graph"),
            ("consolidate metadata", "T_consolidate_metadata"),
        ],
    ),
]

DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY = "T_total"


def _load_processing_set_chunk(load_params):
    """Load a disk-level chunk of the processing set for the data loading layer.

    Used as the ``data_loading_task`` argument of
    :func:`graphviper.graph_tools.map.map` when ``disk_chunk_sizes`` is provided.
    Each call reads one contiguous block of frequency channels (the native on-disk
    chunk) and returns the loaded datasets as a plain dict.  The framework
    sub-selects each map task's slice before invoking
    :func:`NT_image_cube_single_field`, which uses the pre-loaded data directly
    instead of re-reading from disk.

    Parameters
    ----------
    load_params : dict
        Must contain:

        * ``"input_data_store"`` – path to the processing set Zarr store.
        * ``"data_selection"`` – ``{xds_name: {dim: slice}}`` at disk-chunk
          granularity as produced by
          :func:`graphviper.graph_tools.map._build_load_stage`.
        * ``"processing_set_data_group_name"`` – data group forwarded to
          :func:`xradio.measurement_set.load_processing_set`.

    Returns
    -------
    dict
        ``{xds_name: xarray.Dataset}`` with the loaded disk-chunk data.
    """
    from xradio.measurement_set.load_processing_set import load_processing_set

    return load_processing_set(
        load_params["input_data_store"],
        sel_parms=load_params["data_selection"],
        data_group_name=load_params.get("processing_set_data_group_name"),
        load_sub_datasets=False,
    )


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def image_cube_single_field(
    ps_store: str,
    image_store: str,
    image_params: Dict[str, Any],
    imaging_weights_params: Dict[str, Any],
    iteration_control_params: Dict[str, Any],
    gridder="prolate_spheroidal",
    deconvolver="hogbom",
    instrument_polarization_basis: str = "linear",
    scan_intents: list[str] = ["OBSERVE_TARGET#ON_SOURCE"],
    field_name: str = None,
    image_data_variables_keep: list[str] = [
        "sky_deconvolved",
        "sky_residual",
        "sky_model",
        "point_spread_function",
        "primary_beam",
    ],
    compressor=Blosc(cname="lz4", clevel=5),
    processing_set_data_group_name: str = "corrected",
    single_precision_image: bool = True,
    thread_info: dict = None,
    processing_function_threads: int = 1,
    n_chunks: Optional[int] = None,
    overwrite: bool = False,
    memory_mode: str = "in_memory",
    cache_directory: str = None,
    write_visibility_model_to_ps: bool = False,
    write_imaging_weights_to_ps: bool = False,
    clear_cache: bool = True,
    vizualize_graph: bool = False,
    disk_chunk_sizes: Optional[Union[Dict[str, int], str]] = None,
    fft_backend="pyfftw",
    restore: bool = False,
    skunk_works: bool = False,
):  # -> Tuple[xr.Dataset, ReturnDict]:
    """
    Create a spectral cube.

    Parameters
    ----------
    ps_store : str
        String of the path and name of the processing set.
    image_store : str
        Path/URL of the on-disk Zarr image cube.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor.
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    iteration_control_params : dict
        CLEAN iteration controls: ``niter``, ``nmajor``, ``threshold`` (the
        deconvolver stopping threshold), ``primary_beam_limit`` (primary-beam
        mask cutoff as a fraction of the peak primary beam, distinct from
        ``threshold``), ``gain``, ``cyclefactor``, ``cycleniter``,
        ``minpsffraction`` and ``maxpsffraction``.
    gridder : str
        The gridder to use. Default ``"prolate_spheroidal"`` (a prolate
        spheroidal gridding convolution kernel with support 7x7 and oversampling
        of 100).
    deconvolver : str
        Deconvolution algorithm for the minor cycle. One of ``"hogbom"`` (C++, threaded across planes), ``"hogbom_many_threads"``
        (numba, threaded across *and* within planes -- faster when there are
        few planes, e.g. single-channel imaging) or ``"asp"``.
    instrument_polarization_basis : str
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` (``XX``/``YY``) or ``"circular"`` (``RR``/``LL``). The
        output image is always produced in the Stokes basis.
    scan_intents : list[str]
        The scan intents to image.
    field_name : str
        The field to image. If None, the first field in the processing set will be used
    compressor : numcodecs compressor
        Compressor applied to each chunk of the on-disk image.
    processing_set_data_group_name : str
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    single_precision_image : bool
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the minor
        cycle runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    thread_info : dict, optional
        Thread information as returned by
        :func:`~astroviper.utils.data_partitioning.get_thread_info`. Queried
        automatically when ``None``.
    n_chunks : int, optional
            Number of frequency chunks to use for parallel processing. If None (default), the chunk count is auto-determined based on the image size, memory constraints, and available parallelism.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / Numba /
        FFT) kernels.
    overwrite : bool
        Whether to overwrite existing image. Default is False.
    memory_mode : str
        The memory mode to use. Default is "in_memory".
        The image data partition is always kept in memory until the final image is written to disk. The memory mode determines how the processing set (PS) partition is handled during imaging.
        Options are:
        - "in_memory": The PS partition is kept entirely in memory.
        - "in_place": Only a single MSv4 of the PS partition is kept in memory. Imaging weights and the model are written to the original on-disk PS and removed after imaging, unless `write_visibility_model_to_ps` or `write_imaging_weights_to_ps` is `True`. In this mode, the visibility data is read in each cycle (higher I/O).
        - "cache": Only a single MSv4 of the PS partition is kept in memory. In each cycle, the cache is queried to check whether the relevant partition of the PS is present. If not, the required partition is read from the original PS and written to the cache. The visibility model and imaging weights are always written to the cache. After imaging, the cache is cleared if `clear_cache` is `True`. In this mode, the visibility data is read in each cycle, but the cache is always checked first. Therefore, if there is a data scatter stage in a workflow, the PS partition will only be read from the cache. This could provide better performance than `"in_place"` if the cache resides on fast local storage.
    cache_directory : str, optional
        The directory to use for caching. Default is None.
    write_visibility_model_to_ps : bool
        Whether to write the visibility model to the processing set. Default is False.
    write_imaging_weights_to_ps : bool
        Whether to write the imaging weights to the processing set. Default is False.
    clear_cache : bool
        Whether to clear the cache after imaging. Default is True.
    vizualize_graph : bool
        Whether to vizualize the graph. Default is False.
    disk_chunk_sizes : Union[Dict[str, int], str], optional
        Native on-disk chunk sizes for parallel dimensions, e.g.
        ``{"frequency": 200}``.  The graph gains a data loading layer: one load
        node is created per unique disk chunk, each reading the full native
        chunk once.  The mapping nodes then receive their sub-selected slice
        from the pre-loaded chunk via ``input_params["input_data"]``, avoiding
        redundant disk reads when many small mapping tasks fall within the same
        on-disk chunk.  If ``Auto`` (default), the chunk sizes are
        auto-detected from the processing set using
        :func:`graphviper.graph_tools.coordinate_utils.get_disk_chunk_sizes`.
    fft_backend : str
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    restore : bool
        If ``True`` produce a restored image after deconvolution: the model
        convolved with the clean beam (the Gaussian fit to the PSF) plus the
        residual, written to the ``sky_restored`` (``SKY_RESTORED``) variable.
    skunk_works : bool
        If ``True`` use the experimental performance I/O path in each node task:
        load only the data group's data variables straight from the Zarr chunk
        blobs (reconstructing coordinates from the inputs, ignoring all
        sub-datasets) and write each result chunk's blob directly to the
        pre-created image store, bypassing the asyncio Zarr dataset API. The
        processing set and image are assumed to share the same frequency
        coordinate. Default ``False``.
    Returns
    -------
    dict
        ``{"timing_node_tasks", "deconvolution", "timing_distributed_application"}``:

        * ``"timing_node_tasks"`` is a :class:`pandas.DataFrame` with one row per
          frequency chunk and a ``T_*`` column per processing function (the
          per-node-task timings concatenated across chunks).
        * ``"deconvolution"`` is the merged per-plane
          :class:`~astroviper.processing_functions.imaging.utils.return_dict.ReturnDict`
          of convergence statistics (global channel numbering).
        * ``"timing_distributed_application"`` is a dict of the driver-level
          step timings (``T_*`` seconds: building/writing the empty image,
          building the graph, computing it, consolidating metadata) plus the
          grand total ``T_total``.
    """

    import numpy as np
    import xarray as xr
    import dask
    import os
    import toolviper.utils.logger as logger
    from xradio.measurement_set import open_processing_set
    from graphviper.graph_tools.coordinate_utils import make_parallel_coord
    from graphviper.graph_tools import generate_dask_workflow, generate_airflow_workflow
    from graphviper.graph_tools import map, reduce
    from xradio.image import make_empty_sky_image
    from xradio.image import write_image
    import zarr
    from astroviper.utils.io import create_empty_data_variables_on_disk
    from astroviper.utils.data_partitioning import (
        calculate_data_chunking,
        get_thread_info,
    )
    import time

    assert (
        memory_mode == "in_memory"
    ), "Currently only in_memory is supported for memory_mode is implemented."

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

    start = time.time()
    write_image(img_xds, imagename=image_store, out_format="zarr", overwrite=overwrite)
    timing_distributed_application["T_write_empty_image"] = time.time() - start

    # Determine number of chunks
    start = time.time()
    n_chunks = int(
        calculate_number_of_chunks_for_cube_imaging(
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
    input_params["deconvolver"] = deconvolver
    input_params["instrument_polarization_basis"] = instrument_polarization_basis
    input_params["single_precision_image"] = single_precision_image
    input_params["fft_backend"] = fft_backend
    input_params["restore"] = restore
    input_params["skunk_works"] = skunk_works

    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        get_disk_chunk_sizes,
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

    # Create Map Graph. The node task has an explicit, NumPy-documented,
    # standalone signature; graphviper's map() adapts it to the single
    # ``input_params`` dict calling convention automatically (forwarding only the
    # keys the node task declares), so it can be passed directly.
    start = time.time()
    viper_graph = map(
        input_data=ps_xdt,
        node_task_data_mapping=node_task_data_mapping,
        node_task=node_tasks.imaging.image_cube_single_field,
        input_params=input_params,
        in_memory_compute=False,
        # data_loading_task=_load_processing_set_chunk,
        data_loading_task=None,
        disk_chunk_sizes=disk_chunk_sizes,
        load_node_input_params={
            "processing_set_data_group_name": processing_set_data_group_name
        },
    )

    reduce_input_params = {}

    viper_graph = reduce(
        viper_graph, combine_return_data_frames, reduce_input_params, mode="tree"
    )
    timing_distributed_application["T_create_map_reduce_graph"] = time.time() - start

    # Compute cube
    start = time.time()
    dask_graph = generate_dask_workflow(viper_graph)
    timing_distributed_application["T_generate_dask_graph"] = time.time() - start

    if vizualize_graph:
        dask.visualize(dask_graph, filename="cube_imaging.png")

    start = time.time()
    return_dict = dask.compute(dask_graph)[0]
    timing_distributed_application["T_compute_dask_graph"] = time.time() - start

    start = time.time()
    zarr.consolidate_metadata(image_store)
    timing_distributed_application["T_consolidate_metadata"] = time.time() - start

    timing_distributed_application["T_total"] = time.time() - application_start

    # The reduce already produced ``{"timing_node_tasks", "deconvolution"}``; add
    # the driver-level timing so the full return dict carries timing for both the
    # distributed application (this driver) and the per-chunk node tasks.
    return_dict["timing_distributed_application"] = timing_distributed_application

    from astroviper.utils.timing import format_timing_summary
    from astroviper.processing_functions.imaging.utils import (
        IMAGING_TIMING_PHASES,
        IMAGING_TIMING_TOTAL_KEY,
    )

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


def combine_return_data_frames(input_data, input_params):
    """Reduce per-chunk ``{"timing_node_tasks", "deconvolution"}`` results into one dict.

    Each node task returns a single dict with a ``"timing_node_tasks"`` one-row
    :class:`pandas.DataFrame` and a ``"deconvolution"``
    :class:`~astroviper.processing_functions.imaging.utils.return_dict.ReturnDict`
    (already remapped to global channel numbers) for its channel chunk. This
    reducer concatenates the timing frames (one row per chunk) and merges the
    per-chunk deconvolution dicts with :func:`merge_return_dicts`. Because every
    chunk covers a disjoint global channel range, the merge never collides.

    Returns the same ``{"timing_node_tasks", "deconvolution"}`` shape so it
    composes under tree-mode reduction (its own output becomes an input at the
    next level).

    Parameters
    ----------
    input_data : list of dict
        Per-chunk (or partially reduced) results, each with
        ``"timing_node_tasks"`` and ``"deconvolution"`` keys.
    input_params : dict
        Unused; present for the GraphVIPER reduce signature.

    Returns
    -------
    dict
        ``{"timing_node_tasks": pandas.DataFrame, "deconvolution": ReturnDict}``.
    """
    import pandas as pd
    from astroviper.processing_functions.imaging.utils.iteration_control import (
        merge_return_dicts,
    )

    combined_timing = pd.DataFrame()
    deconvolve_dicts = []

    for result in input_data:
        combined_timing = pd.concat(
            [combined_timing, result["timing_node_tasks"]], ignore_index=True
        )
        deconvolve_dicts.append(result["deconvolution"])

    return {
        "timing_node_tasks": combined_timing,
        "deconvolution": merge_return_dicts(deconvolve_dicts),
    }


def calculate_number_of_chunks_for_cube_imaging(
    img_xds, single_precision_image, n_chunks, thread_info
):
    """Determine the number of frequency chunks for cube imaging.

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
                3 * n_pixels_single_frequency * bytes_in_dtype["complex64"] / (1024**3)
                + 3 * n_pixels_single_frequency * bytes_in_dtype["float32"] / (1024**3)
            )
        else:
            memory_singleton_chunk = fudge_factor * (
                3 * n_pixels_single_frequency * bytes_in_dtype["complex128"] / (1024**3)
                + 3 * n_pixels_single_frequency * bytes_in_dtype["float64"] / (1024**3)
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
