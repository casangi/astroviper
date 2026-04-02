from numcodecs import Blosc
import xarray as xr
from typing import Optional, Dict, Any, Tuple
from xradio.image import make_empty_sky_image
import numpy as np
import zarr

from astroviper.task.imaging.image_cube_single_field import (
    NT_image_cube_single_field,
)


def image_cube_single_field(
    ps_store: str,
    image_store: str,
    image_params: Dict[str, Any],
    imaging_weights_params: Dict[str, Any],
    iteration_control_params: Dict[str, Any],
    gridder="prolate_spheroidal",
    deconvolver="hogbom",
    fft_padding="1.0",
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
    double_precision: bool = True,
    thread_info: dict = None,
    n_chunks: Optional[int] = None,
    overwrite: bool = False,
    memory_mode: str = "in_memory",
    cache_directory: str = None,
    write_visibility_model_to_ps: bool = False,
    write_imaging_weights_to_ps: bool = False,
    clear_cache: bool = True,
    vizualize_graph: bool = False,
):  # -> Tuple[xr.Dataset, ReturnDict]:
    """
    Create a spectral cube.

    Parameters
    ----------
    ps_store : str
        String of the path and name of the processing set.
    image_store : str
        String of the path and name of the spectral cube image that will be created.
    image_params : dict
        Image parameters used to create the coordinates. Must include:
            - ``image_size`` : array-like of int
                grid size as ``(x, y)``.
            - ``cell_size`` : array-like of float
                Angular cell size (radians).
            - ``phase_center`` : array-like of float
            - ``frequency_coords``  : array-like of float
            - ``time_coords`` : array-like of float
            - ``pol_coords`` : array-like of str
    imaging_weights_params : dict
        Weighting scheme configuration. Must include:
            - ``weighting`` : {"natural", "briggs"}
                Type of weighting to apply.
            - ``robust`` : float, optional
                Briggs robust parameter (ignored if ``"natural"``).
    iteration_control_params : dict

    gridder : str
        The gridder to use. Default prolate_spheroidal.
        Add Schwab reference
        prolate_spheroidal: Uses a prolate spheriodal gridding convolution kernel with support 7x7 and oversmapling of 100.
        nearest: Snaps visibility to nearest element in the grid.
    deconvolver : str
        The image deconvolver to use. Default hogbom.
    scan_intents : list[str]
        The scan intent to use
    field_name : str
        The field to image. If None, the first field in the processing set will be used
    compressor :
    processing_set_data_group_name :
        Data group in the processing set to use for imaging for example base, corrected ect. Default is base
    double_precision :
        Use single or double precision math when gridding and deconvolving. Default = true
    thread_info :
    n_chunks : int, optional
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
    Returns
    -------
    deconvolution_stats :
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
    logger.info(
        "Time to create empty image xds: " + str(time.time() - start) + " seconds"
    )

    start = time.time()
    write_image(img_xds, imagename=image_store, out_format="zarr", overwrite=overwrite)
    logger.info(
        "Time to write empty image to disk: " + str(time.time() - start) + " seconds"
    )

    # Determine number of chunks
    start = time.time()
    n_chunks = calculate_number_of_chunks_for_cube_imaging(
        img_xds, double_precision, n_chunks, thread_info
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
    logger.info(
        "Time to determine number of chunks and make parallel coords: "
        + str(time.time() - start)
        + " seconds"
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
        double_precision=double_precision,
        data_variable_definitions="imaging",
    )
    logger.info(
        "Time to create empty data variables on disk: "
        + str(time.time() - start)
        + " seconds"
    )

    zarr_meta = {}

    input_parms = {}
    input_parms["image_params"] = image_params
    input_parms["imaging_weights_params"] = imaging_weights_params
    input_parms["zarr_meta"] = zarr_meta
    input_parms["to_disk"] = True
    input_parms["polarization"] = img_xds.polarization.data
    input_parms["time"] = [0]
    input_parms["compressor"] = compressor
    input_parms["image_store"] = image_store
    input_parms["input_data_store"] = ps_store
    input_parms["processing_set_data_group_name"] = processing_set_data_group_name
    input_parms["image_data_variables_keep"] = image_data_variables_keep
    input_parms["memory_mode"] = memory_mode
    input_parms["cache_directory"] = cache_directory
    input_parms["write_visibility_model_to_ps"] = write_visibility_model_to_ps
    input_parms["write_imaging_weights_to_ps"] = write_imaging_weights_to_ps
    input_parms["clear_cache"] = clear_cache

    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
    )

    start = time.time()
    ps_xdt = open_processing_set(ps_store, scan_intents=scan_intents)
    logger.info("Time to open processing set: " + str(time.time() - start) + " seconds")

    start = time.time()
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps_xdt
    )
    logger.info(
        "Time to interpolate data coords onto parallel coords: "
        + str(time.time() - start)
        + " seconds"
    )

    # frequency_coords is not used by node tasks (they use task_coords["frequency"]["data"])
    # so remove it to avoid embedding the full frequency axis in every task in the graph.
    input_parms["image_params"] = {
        k: v for k, v in image_params.items() if k != "frequency_coords"
    }

    # Create Map Graph
    start = time.time()
    viper_graph = map(
        input_data=ps_xdt,
        node_task_data_mapping=node_task_data_mapping,
        node_task=NT_image_cube_single_field,
        # node_task=test_task,
        input_params=input_parms,
        in_memory_compute=False,
    )

    input_params = {}

    viper_graph = reduce(
        viper_graph, combine_return_data_frames, input_params, mode="tree"
    )
    logger.info(
        "Time to create map reduce graph: " + str(time.time() - start) + " seconds"
    )

    # Compute cube
    start = time.time()
    dask_graph = generate_dask_workflow(viper_graph)
    logger.info("Time to generate dask graph: " + str(time.time() - start) + " seconds")

    if vizualize_graph:
        dask.visualize(dask_graph, filename="cube_imaging.png")

    start = time.time()
    return_dict = dask.compute(dask_graph)[0]
    logger.info("Time to compute dask graph: " + str(time.time() - start) + " seconds")

    start = time.time()
    zarr.consolidate_metadata(image_store)
    logger.info(
        "Time to consolidate metadata: " + str(time.time() - start) + " seconds"
    )

    return return_dict


def combine_return_data_frames(input_data, input_parms):
    import pandas as pd

    combined_df = pd.DataFrame()

    for df in input_data:
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df


def calculate_number_of_chunks_for_cube_imaging(
    img_xds, double_precision, n_chunks, thread_info
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
    double_precision : bool
        If ``True``, use double-precision (complex128 / float64) memory estimates;
        otherwise single-precision (complex64 / float32).
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
        if double_precision:
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
