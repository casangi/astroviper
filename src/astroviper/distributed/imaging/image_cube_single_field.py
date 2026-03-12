from numcodecs import Blosc
import xarray as xr
from astroviper.core.imaging.imaging_utils.return_dict import ReturnDict
from typing import Optional, Dict, Any, Tuple
from xradio.image import make_empty_sky_image
import numpy as np
import zarr

from astroviper.task.imaging.image_cube_single_field_node_task import (
    image_cube_single_field_node_task,
)

def get_rss_gb():
    import psutil, os
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

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
    data_group_name: str = "base",
    double_precision: bool = True,
    thread_info: dict = None,
    n_chunks: Optional[int] = None,
    overwrite: bool = False,
    memory_mode: str = "in_memory",
    cache_directory: str = None,
    write_visibility_model_to_ps: bool = False,
    write_imaging_weights_to_ps: bool = False,
    clear_cache: bool = True,
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
    data_group_name :
        Data group to use for imaging for example base, corrected ect. Default is base
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
    Returns
    -------
    deconvolution_stats :
    """

    import numpy as np
    import xarray as xr
    import dask
    import os
    from xradio.measurement_set import open_processing_set
    from graphviper.graph_tools.coordinate_utils import make_parallel_coord
    from graphviper.graph_tools import generate_dask_workflow, generate_airflow_workflow
    from graphviper.graph_tools import map, reduce
    from xradio.image import make_empty_sky_image
    from xradio.image import write_image
    import zarr
    import toolviper.utils.logger as logger
    from astroviper.utils.io import create_empty_data_variables_on_disk
    from astroviper.utils.data_partitioning import (
        calculate_data_chunking,
        get_thread_info,
    )
    import time

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
    logger.info("Time to create empty image xds: " + str(time.time() - start) + " seconds")

    start = time.time()
    write_image(img_xds, imagename=image_store, out_format="zarr", overwrite=overwrite)     
    logger.info("Time to write empty image to disk: " + str(time.time() - start) + " seconds")

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
    logger.info("Time to determine number of chunks and make parallel coords: " + str(time.time() - start) + " seconds")

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
    logger.info("Time to create empty data variables on disk: " + str(time.time() - start) + " seconds")
    
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
    input_parms["data_group_name"] = data_group_name
    input_parms["image_data_variables_keep"] = image_data_variables_keep
    input_parms["memory_mode"] = memory_mode
    input_parms["cache_directory"] = cache_directory
    input_parms["write_visibility_model_to_ps"] = write_visibility_model_to_ps
    input_parms["write_imaging_weights_to_ps"] = write_imaging_weights_to_ps
    input_parms["clear_cache"] = clear_cache

    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        interpolate_data_coords_onto_parallel_coords_v2
    )

    start = time.time()
    ps_xdt = open_processing_set(ps_store, scan_intents=scan_intents)
    logger.info("Time to open processing set: " + str(time.time() - start) + " seconds")

    start = time.time()
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords_v2(
        parallel_coords, ps_xdt
    )
    logger.info("Time to interpolate data coords onto parallel coords: " + str(time.time() - start) + " seconds")

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
        node_task=wrap_image_cube_single_field_node_task,
        #node_task=test_task,
        input_params=input_parms,
        in_memory_compute=False,
    )
    logger.info("Time to create map graph: " + str(time.time() - start) + " seconds")

    input_params = {}

    viper_graph = reduce(
        viper_graph, combine_return_data_frames, input_params, mode="tree"
    )
    #Compute cube
    
    start = time.time()
    dask_graph = generate_dask_workflow(viper_graph)
    logger.info("Time to generate dask graph: " + str(time.time() - start) + " seconds")
    #dask.visualize(dask_graph, filename="cube_imaging.png")
    return_dict = dask.compute(dask_graph)[0]

    start = time.time()
    zarr.consolidate_metadata(image_store)
    logger.info("Time to consolidate metadata: " + str(time.time() - start) + " seconds")  
    
    return return_dict

from memory_profiler import profile

def fft_lm_to_uv(image):
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(2, 3)), axes=(2, 3)), axes=(2, 3)
    ).real

def test_task(input_params):
    logger.info("Memory usage before creation of array: " + str(get_rss_gb()) + " GB")
    test_random_array = np.random.rand(3,2,12000, 12000)
    logger.info("Memory usage after creation of array: " + str(get_rss_gb()) + " GB")
    fft_array = fft_lm_to_uv(test_random_array)
    logger.info("Memory usage after fft: " + str(get_rss_gb()) + " GB")
    #logger.info("Running test task with input params: " + str(input_params))
    
    import pandas as pd
    return_dict = {
        "task_id": [input_params["task_id"]],
        "n_channels": [len(input_params["task_coords"]["frequency"]["data"])],
        "T_load": 42.0,
    }
    df = pd.DataFrame(return_dict)
    return df

# [2026-03-11 09:49:55,206]    DEBUG    worker_0:  Memory usage before creation of array: 0.348659712 GB
# [2026-03-11 09:49:56,225]    DEBUG    worker_0:  Memory usage after creation of array: 1.500672 GB
# [2026-03-11 09:50:05,014]    DEBUG    worker_0:  Memory usage after fft: 3.805011968 GB

#@profile(precision=1)
def wrap_image_cube_single_field_node_task(input_params):
    import ctypes
    # Pin glibc's mmap threshold BEFORE any large allocations so they use mmap
    # and are returned to the OS immediately on free (no heap fragmentation).
    # Must run at the start of the task, not after, or fragmentation is already done.
    # M_MMAP_THRESHOLD = -3
    ctypes.CDLL("libc.so.6").mallopt(-3, 131072)
    import toolviper.utils.logger as logger   
    
    logger.info("Memory usage at start of wrap_image_cube_single_field_node_task: " + str(get_rss_gb()) + " GB")

    from xradio.image import make_empty_sky_image
    from xradio.measurement_set.load_processing_set import ProcessingSetIterator

    image_params = input_params["image_params"]
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=input_params["task_coords"]["frequency"]["data"],
        pol_coords=image_params["polarization_coords"],
        time_coords=image_params["time_coords"],
    )

    data_group_name = input_params["data_group_name"]
    if data_group_name == "base":
        data_variables = ["FLAG", "UVW", "VISIBILITY", "WEIGHT"]
    elif data_group_name == "corrected":
        data_variables = ["FLAG", "UVW", "VISIBILITY_CORRECTED", "WEIGHT"]
    else:
        raise ValueError("Invalid data group: " + str(data_group_name))

    if input_params["memory_mode"] == "in_memory":
        in_memory = True
    else:
        in_memory = False

    assert (
        in_memory
    ), "Currently only in_memory is supported for memory_mode is implemented."
    
    
    # logger.info("PS data selection: " + str(input_params["data_selection"]))
    # logger.info("PS data selection: " + str(input_params["input_data_store"]) + " data group: " + str(data_group_name) + " variables: " + str(data_variables))

    ps_iter = ProcessingSetIterator(
        input_data_store=input_params["input_data_store"],
        sel_parms=input_params["data_selection"],
        data_group_name=data_group_name,
        include_variables=data_variables,
        load_sub_datasets=False,
        in_memory=in_memory,
    )
    
    logger.info("********** Processing set iterator created with partitions.")
    result = image_cube_single_field_node_task(input_params, ps_iter, img_xds)

    # Release the outer references to img_xds and ps_iter NOW, so that glibc
    # can actually return those pages to the OS when malloc_trim fires.
    # The malloc_trim inside image_cube_single_field_node_task runs while
    # these outer references are still live, so it cannot reclaim that memory.
    import psutil, os

    p = psutil.Process(os.getpid())
    
    logger.info("Memory usage after completing node task, before releasing references: " + str(get_rss_gb()) + " GB")
    # for m in p.memory_maps():
    #     logger.info(m.path + ": " + str(m.rss/1e6) + " MB")
    
    import gc
    import ctypes
    img_xds = None
    ps_iter = None
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    # Fix glibc's mmap threshold so large arrays (>=128 KB) always use mmap
    # and are returned to the OS immediately on free, rather than going into
    # the heap arena where they cause fragmentation. M_MMAP_THRESHOLD = -3.
    libc.mallopt(-3, 131072)
    libc.malloc_trim(0)
    logger.info("Outer wrap: released img_xds/ps_iter, fixed mmap threshold, trimmed heap")
    
    logger.info("*********************************************************")
    
    # memory_diagnostic_dump(
    #     logger=logger,
    #     top_maps=20,
    #     min_array_mb=50,
    #     min_smaps_rss_mb=100,
    #     dump_malloc_info=True,
    #     trim_heap=True,
    # )
    
    import os
    import gc
    import mmap
    import psutil
    import logging

    p = psutil.Process(os.getpid())

    def _fmt_gb(x):
        return f"{x / (1024**3):.3f} GB"

    def _fmt_mb(x):
        return f"{x / (1024**2):.1f} MB"

    logger.info("========== NATIVE BUFFER DIAGNOSTIC START ==========")
    logger.info(f"RSS: {_fmt_gb(p.memory_info().rss)}")

    # 1) Show biggest anonymous mappings
    maps = sorted(p.memory_maps(grouped=False), key=lambda m: m.rss, reverse=True)
    logger.info("Top memory maps:")
    for m in maps[:10]:
        path = m.path if m.path else "[anon]"
        logger.info(
            f"  path={path} rss={_fmt_mb(m.rss)} "
            f"private_dirty={_fmt_mb(getattr(m, 'private_dirty', 0))}"
        )

    # 2) Force GC
    gc.collect()

    # 3) Look for Python objects that commonly wrap native memory
    suspects = []
    for o in gc.get_objects():
        try:
            t = type(o)
            mod = getattr(t, "__module__", "")
            name = getattr(t, "__name__", str(t))

            size = None
            extra = ""

            if isinstance(o, memoryview):
                try:
                    size = o.nbytes
                except Exception:
                    size = None
                extra = f" readonly={getattr(o, 'readonly', '?')} contiguous={getattr(o, 'c_contiguous', '?')}"

            elif isinstance(o, mmap.mmap):
                try:
                    size = len(o)
                except Exception:
                    size = None

            elif mod.startswith("pyarrow"):
                # PyArrow often owns native memory outside Python heap
                try:
                    size = o.nbytes
                except Exception:
                    try:
                        size = o.size
                    except Exception:
                        size = None

            elif mod.startswith("builtins") and name in {"bytearray", "bytes"}:
                try:
                    size = len(o)
                except Exception:
                    size = None

            # generic fallback for objects exposing nbytes
            elif hasattr(o, "nbytes"):
                try:
                    size = int(o.nbytes)
                except Exception:
                    size = None

            if size is not None and size >= 50 * 1024**2:
                suspects.append((size, mod, name, id(o), extra, repr(o)[:200]))
        except Exception:
            pass

    suspects.sort(reverse=True, key=lambda x: x[0])

    logger.info(f"Large Python-visible native/buffer suspects >= 50 MB: {len(suspects)}")
    for size, mod, name, oid, extra, rep in suspects[:50]:
        logger.info(
            f"  {mod}.{name} id={oid} size={_fmt_mb(size)}{extra} repr={rep}"
        )

    # 4) PyArrow pool stats if available
    try:
        import pyarrow as pa
        logger.info(
            f"PyArrow total_allocated_bytes={_fmt_mb(pa.total_allocated_bytes())}"
        )
        try:
            pool = pa.default_memory_pool()
            logger.info(f"PyArrow backend={pool.backend_name}")
            logger.info(f"PyArrow pool bytes_allocated={_fmt_mb(pool.bytes_allocated())}")
        except Exception:
            pass
    except Exception as e:
        logger.info(f"PyArrow stats unavailable: {e}")

    # 5) Numba / llvmlite presence
    try:
        import numba
        logger.info(f"Numba imported: version={numba.__version__}")
    except Exception as e:
        logger.info(f"Numba not available: {e}")

    logger.info("========== NATIVE BUFFER DIAGNOSTIC END ==========")
    
    logger.info("*********************************************************")

    # for m in p.memory_maps():
    #     logger.info(m.path + ": " + str(m.rss/1e6) + " MB")
    # logger.info("Memory usage after completing node task, before releasing references: " + str(get_rss_gb()) + " GB")
    
    return result
    #return [0]


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



###Memory diagnostics

# Drop this near the end of your function, just before return.
# It will print:
# - RSS before/after cleanup
# - top memory maps
# - live large NumPy arrays
# - a rough fragmentation estimate
# - glibc malloc_info() XML
# - selected /proc/self/smaps entries for large anonymous mappings
#
# Linux/glibc only for malloc_info(), mallopt(), malloc_trim(), /proc/self/smaps.

import os
import gc
import sys
import psutil
import ctypes
import tempfile
import logging

try:
    import numpy as np
except Exception:
    np = None

logger = logging.getLogger(__name__)
p = psutil.Process(os.getpid())


def _rss_gb():
    return p.memory_info().rss / (1024**3)


def _fmt_mb(n):
    return f"{n / (1024**2):.1f} MB"


def _fmt_gb(n):
    return f"{n / (1024**3):.3f} GB"


def memory_diagnostic_dump(
    logger,
    top_maps=20,
    min_array_mb=50,
    min_smaps_rss_mb=100,
    dump_malloc_info=True,
    trim_heap=True,
):
    logger.info("=" * 80)
    logger.info("MEMORY DIAGNOSTIC START")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"RSS before cleanup: {_fmt_gb(p.memory_info().rss)}")

    # ------------------------------------------------------------------
    # 1) Show largest memory maps before cleanup
    # ------------------------------------------------------------------
    try:
        maps = sorted(p.memory_maps(grouped=False), key=lambda m: m.rss, reverse=True)
        logger.info(f"Top {top_maps} memory maps BEFORE cleanup:")
        for m in maps[:top_maps]:
            path = m.path if m.path else "[anon]"
            logger.info(
                f"  path={path} rss={_fmt_mb(m.rss)} "
                f"private_dirty={_fmt_mb(getattr(m, 'private_dirty', 0))} "
                f"private_clean={_fmt_mb(getattr(m, 'private_clean', 0))} "
                f"shared_dirty={_fmt_mb(getattr(m, 'shared_dirty', 0))} "
                f"shared_clean={_fmt_mb(getattr(m, 'shared_clean', 0))}"
            )
    except Exception as e:
        logger.info(f"Could not read memory_maps BEFORE cleanup: {e}")

    # ------------------------------------------------------------------
    # 2) Show live large NumPy arrays before cleanup
    # ------------------------------------------------------------------
    numpy_live_bytes = 0
    live_arrays = []
    if np is not None:
        try:
            for o in gc.get_objects():
                try:
                    if isinstance(o, np.ndarray):
                        n = int(o.nbytes)
                        numpy_live_bytes += n
                        if n >= min_array_mb * 1024**2:
                            live_arrays.append((n, o.shape, o.dtype, id(o)))
                except Exception:
                    pass

            live_arrays.sort(reverse=True, key=lambda x: x[0])
            logger.info(
                f"Live NumPy arrays BEFORE cleanup: total={_fmt_gb(numpy_live_bytes)} "
                f"count={len(live_arrays)} arrays>={min_array_mb}MB={len(live_arrays)}"
            )
            for n, shape, dtype, oid in live_arrays[:20]:
                logger.info(
                    f"  ndarray id={oid} shape={shape} dtype={dtype} nbytes={_fmt_mb(n)}"
                )
        except Exception as e:
            logger.info(f"Could not inspect NumPy arrays BEFORE cleanup: {e}")
    else:
        logger.info("NumPy not importable; skipping ndarray inspection")

    # ------------------------------------------------------------------
    # 3) Remove likely large locals if present in local scope
    #    Add your own names here.
    # ------------------------------------------------------------------
    try:
        # Common suspects from your example:
        for name in [
            "img_xds",
            "ps_iter",
            "xds",
            "dataset",
            "arr",
            "data",
            "model",
            "weights",
            "cube",
            "image",
        ]:
            if name in locals():
                locals()[name] = None
    except Exception:
        pass

    # IMPORTANT:
    # To actually clear your own function locals, explicitly set them to None
    # above this helper call, e.g.:
    #   img_xds = None
    #   ps_iter = None

    # ------------------------------------------------------------------
    # 4) Run GC and optionally trim glibc heap
    # ------------------------------------------------------------------
    try:
        unreachable = gc.collect()
        logger.info(f"gc.collect() reclaimed {unreachable} unreachable objects")
    except Exception as e:
        logger.info(f"gc.collect() failed: {e}")

    trim_result = None
    if trim_heap:
        try:
            libc = ctypes.CDLL("libc.so.6")

            # Fix mmap threshold for future allocations:
            # M_MMAP_THRESHOLD = -3
            libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
            libc.mallopt.restype = ctypes.c_int
            mallopt_res = libc.mallopt(-3, 131072)

            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
            trim_result = libc.malloc_trim(0)

            logger.info(
                f"mallopt(M_MMAP_THRESHOLD, 131072) -> {mallopt_res}, malloc_trim(0) -> {trim_result}"
            )
        except Exception as e:
            logger.info(f"glibc mallopt/malloc_trim failed: {e}")

    rss_after = p.memory_info().rss
    logger.info(f"RSS after cleanup: {_fmt_gb(rss_after)}")

    # ------------------------------------------------------------------
    # 5) Show largest memory maps after cleanup
    # ------------------------------------------------------------------
    try:
        maps_after = sorted(p.memory_maps(grouped=False), key=lambda m: m.rss, reverse=True)
        logger.info(f"Top {top_maps} memory maps AFTER cleanup:")
        for m in maps_after[:top_maps]:
            path = m.path if m.path else "[anon]"
            logger.info(
                f"  path={path} rss={_fmt_mb(m.rss)} "
                f"private_dirty={_fmt_mb(getattr(m, 'private_dirty', 0))} "
                f"private_clean={_fmt_mb(getattr(m, 'private_clean', 0))} "
                f"shared_dirty={_fmt_mb(getattr(m, 'shared_dirty', 0))} "
                f"shared_clean={_fmt_mb(getattr(m, 'shared_clean', 0))}"
            )
    except Exception as e:
        logger.info(f"Could not read memory_maps AFTER cleanup: {e}")

    # ------------------------------------------------------------------
    # 6) Show live large NumPy arrays after cleanup
    # ------------------------------------------------------------------
    numpy_live_bytes_after = 0
    live_arrays_after = []
    if np is not None:
        try:
            for o in gc.get_objects():
                try:
                    if isinstance(o, np.ndarray):
                        n = int(o.nbytes)
                        numpy_live_bytes_after += n
                        if n >= min_array_mb * 1024**2:
                            live_arrays_after.append((n, o.shape, o.dtype, id(o)))
                except Exception:
                    pass

            live_arrays_after.sort(reverse=True, key=lambda x: x[0])
            logger.info(
                f"Live NumPy arrays AFTER cleanup: total={_fmt_gb(numpy_live_bytes_after)} "
                f"arrays>={min_array_mb}MB={len(live_arrays_after)}"
            )
            for n, shape, dtype, oid in live_arrays_after[:20]:
                logger.info(
                    f"  ndarray id={oid} shape={shape} dtype={dtype} nbytes={_fmt_mb(n)}"
                )
        except Exception as e:
            logger.info(f"Could not inspect NumPy arrays AFTER cleanup: {e}")

    # ------------------------------------------------------------------
    # 7) Rough fragmentation / unmanaged-memory estimate
    # ------------------------------------------------------------------
    try:
        approx_unaccounted = max(0, rss_after - numpy_live_bytes_after)
        logger.info(
            "Approximate unaccounted RSS after cleanup "
            f"(RSS - live NumPy bytes) = {_fmt_gb(approx_unaccounted)}"
        )
    except Exception as e:
        logger.info(f"Could not compute approximate fragmentation estimate: {e}")

    # ------------------------------------------------------------------
    # 8) Dump glibc malloc_info() XML
    # ------------------------------------------------------------------
    if dump_malloc_info:
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
            libc.malloc_info.restype = ctypes.c_int
            libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            libc.fopen.restype = ctypes.c_void_p
            libc.fclose.argtypes = [ctypes.c_void_p]
            libc.fclose.restype = ctypes.c_int

            with tempfile.NamedTemporaryFile(prefix="malloc_info_", suffix=".xml", delete=False) as tf:
                xml_path = tf.name

            fp = libc.fopen(xml_path.encode("utf-8"), b"w")
            if fp:
                rc = libc.malloc_info(0, fp)
                libc.fclose(fp)
                logger.info(f"malloc_info() rc={rc} path={xml_path}")

                with open(xml_path, "r", encoding="utf-8", errors="replace") as f:
                    xml_text = f.read()

                # Log in chunks so logger doesn't truncate too badly
                chunk_size = 4000
                for i in range(0, len(xml_text), chunk_size):
                    logger.info(f"malloc_info XML chunk {i // chunk_size + 1}:\n{xml_text[i:i+chunk_size]}")
            else:
                logger.info("malloc_info(): fopen failed")
        except Exception as e:
            logger.info(f"malloc_info() failed: {e}")

    # ------------------------------------------------------------------
    # 9) Parse /proc/self/smaps and print large anonymous / heap mappings
    # ------------------------------------------------------------------
    try:
        with open(f"/proc/{os.getpid()}/smaps", "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        current_header = None
        current = {}
        blocks = []

        def flush_current():
            if current_header is not None:
                blocks.append((current_header, dict(current)))

        for line in lines:
            if not line.startswith(("Rss:", "Pss:", "Private_Clean:", "Private_Dirty:",
                                    "Shared_Clean:", "Shared_Dirty:", "AnonHugePages:",
                                    "Swap:", "Size:", "KernelPageSize:", "MMUPageSize:")):
                # New mapping header
                if "-" in line[:20]:
                    flush_current()
                    current_header = line.strip()
                    current = {}
                continue

            parts = line.split()
            key = parts[0].rstrip(":")
            try:
                value_kb = int(parts[1])
            except Exception:
                value_kb = 0
            current[key] = value_kb

        flush_current()

        logger.info(f"Large /proc/{os.getpid()}/smaps mappings AFTER cleanup:")
        for header, info in blocks:
            rss_mb = info.get("Rss", 0) / 1024
            if rss_mb < min_smaps_rss_mb:
                continue

            is_heap = "[heap]" in header
            is_anon = (
                "[anon]" in header
                or header.endswith(" 0")
                or " /" not in header and "[heap]" not in header and "[" not in header.split()[-1]
            )

            if is_heap or is_anon:
                logger.info(
                    f"  {header}\n"
                    f"    Size={info.get('Size', 0)/1024:.1f} MB "
                    f"Rss={info.get('Rss', 0)/1024:.1f} MB "
                    f"Pss={info.get('Pss', 0)/1024:.1f} MB "
                    f"Private_Dirty={info.get('Private_Dirty', 0)/1024:.1f} MB "
                    f"Private_Clean={info.get('Private_Clean', 0)/1024:.1f} MB "
                    f"Shared_Dirty={info.get('Shared_Dirty', 0)/1024:.1f} MB "
                    f"Shared_Clean={info.get('Shared_Clean', 0)/1024:.1f} MB "
                    f"AnonHugePages={info.get('AnonHugePages', 0)/1024:.1f} MB "
                    f"Swap={info.get('Swap', 0)/1024:.1f} MB"
                )
    except Exception as e:
        logger.info(f"Could not parse /proc/self/smaps: {e}")

    logger.info("MEMORY DIAGNOSTIC END")
    logger.info("=" * 80)


