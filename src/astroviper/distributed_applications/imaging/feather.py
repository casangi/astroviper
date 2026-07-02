#!/usr/bin/env python
# coding: utf-8
"""Feather workflow driver (graph layer).

Builds and runs the GraphVIPER map graph that combines a single-dish and an
interferometer image with the feather algorithm.  It follows the same approach
as :func:`astroviper.distributed_applications.imaging.image_cube_single_field`:

1. create the empty output image on disk (coordinates + the interferometer beam),
2. pre-allocate the ``SKY`` data variable on disk with
   :func:`astroviper.utils.io.create_empty_data_variables_on_disk`,
3. map the per-frequency-chunk node task
   :func:`astroviper.node_tasks.imaging.feather`, which writes its own ``SKY``
   slice in parallel via
   :func:`astroviper.utils.io.write_result_chunk_to_disk_using_zarr`.

The previous implementation used XRADIO's low-level
``create_data_variable_meta_data`` / ``write_chunk`` helpers, which are not
Zarr v3 compatible.
"""

import copy
import os
import time
from typing import Union

import numpy as np
import xarray as xr
from numcodecs import Blosc
import toolviper.utils.logger as logger

import astroviper.node_tasks as node_tasks

_sky = "SKY"
_beam = "BEAM_FIT_PARAMS_SKY"


def _rename_beam_to_convention(xds):
    """Normalize legacy ``BEAM`` / ``beam_param`` naming to the convention used
    throughout astroviper (``BEAM_FIT_PARAMS_SKY`` / ``beam_params_label``)."""
    if "BEAM" in xds:
        xds = xds.rename(
            {"BEAM": "BEAM_FIT_PARAMS_SKY", "beam_param": "beam_params_label"}
        )
    if "BEAM_FIT_PARAMS" in xds:
        xds = xds.rename({"BEAM_FIT_PARAMS": "BEAM_FIT_PARAMS_SKY"})
    return xds


def _open_input_image(path, selection):
    """Open an input image (zarr or XRADIO image) lazily and normalize naming."""
    from xradio.image import open_image

    if not isinstance(path, str):
        raise TypeError(
            "feather highres/lowres must be string paths to the on-disk images "
            "(the parallel node task reads each frequency chunk from disk)."
        )
    if "zarr" in path:
        xds = xr.open_zarr(path).isel(selection)
    else:
        xds = open_image({"sky": path}, selection=selection)
    return _rename_beam_to_convention(xds)


def feather(
    outim: Union[dict, None],
    highres: str,
    lowres: str,
    sdfactor: float = 1.0,
    selection: dict = {},
    thread_info=None,
    processing_function_threads: int = 1,
    fft_backend: str = "scipy",
    compressor=Blosc(cname="lz4", clevel=5),
):
    """Create an image from a single dish and interferometer image (feather).

    Parameters
    ----------
    outim : dict
        Output image information.  Must contain ``"name"`` (the directory the
        Zarr image is written to).  Optional ``"overwrite"`` (bool, default
        ``False``).  Only the Zarr format is written; convert afterwards if
        another format is needed.  The store is created up front and written to
        chunk-by-chunk in parallel.
    highres : str
        Path to the interferometer (high-resolution) image (a ``.zarr`` store or
        an XRADIO image directory).
    lowres : str
        Path to the single-dish (low-resolution) image.
    sdfactor : float, default 1.0
        Single-dish scaling factor applied in the feather combination.
    selection : dict, optional
        ``xarray`` ``isel`` selection applied to both inputs (e.g. an ``l``/``m``
        sub-window).  Applied both when sizing the output and when each chunk is
        loaded.
    thread_info : dict, optional
        Thread information as returned by
        :func:`~astroviper.utils.data_partitioning.get_thread_info`; queried
        automatically when ``None``.
    processing_function_threads : int, default 1
        Number of FFT worker threads used inside each node task.
    fft_backend : {"scipy", "pyfftw"}, default "scipy"
        FFT backend used by the feather science function. ``scipy`` is the
        default (it is always available and measurably faster on the cube sizes
        typical for feather); pass ``"pyfftw"`` to use the FFTW-backed backend.
    compressor : numcodecs compressor
        Compressor applied to each on-disk ``SKY`` chunk.
    """
    import dask
    from graphviper.graph_tools import generate_dask_workflow
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )
    from graphviper.graph_tools.map import map
    from xradio.image import write_image
    import zarr

    from astroviper.utils.data_partitioning import (
        bytes_in_dtype,
        calculate_data_chunking,
        get_thread_info,
    )
    from astroviper.utils.io import create_empty_data_variables_on_disk

    # --- Validate output spec -------------------------------------------------
    if outim is None or not isinstance(outim, dict):
        raise ValueError(
            "outim must be a dictionary with required key 'name' and optional "
            "key 'overwrite'."
        )
    if "name" not in outim:
        raise ValueError("outim dict must have key 'name'.")
    if "overwrite" not in outim:
        outim["overwrite"] = False
    elif not isinstance(outim["overwrite"], bool):
        raise TypeError("If specified, outim['overwrite'] must be a boolean value")
    if not outim["overwrite"] and os.path.exists(outim["name"]):
        raise RuntimeError(
            f"Already existing file {outim['name']} will not be overwritten. To "
            "overwrite it, set outim['overwrite'] = True"
        )

    # --- Read input image metadata -------------------------------------------
    sd_xds = _open_input_image(lowres, selection)
    int_xds = _open_input_image(highres, selection)
    if sd_xds[_sky].shape != int_xds[_sky].shape:
        raise RuntimeError("Image shapes differ")

    # Output adopts the interferometer dtype-or-lower precision, matching the
    # feather science function (feather_core).
    if sd_xds[_sky].dtype.itemsize <= int_xds[_sky].dtype.itemsize:
        out_dtype = sd_xds[_sky].dtype
    else:
        out_dtype = int_xds[_sky].dtype
    if out_dtype == np.float64:
        double_precision = True
    elif out_dtype == np.float32:
        double_precision = False
    else:
        raise ValueError(
            "Unsupported image dtype "
            + str(out_dtype)
            + "; expected float32 or float64."
        )

    # --- Determine frequency chunking ----------------------------------------
    # Memory for a single-frequency chunk: two input images + one output image.
    singleton_chunk_sizes = dict(sd_xds[_sky].sizes)
    del singleton_chunk_sizes["frequency"]
    fudge_factor = 1.1
    n_images_in_memory = 3.0
    memory_singleton_chunk = (
        n_images_in_memory
        * np.prod(np.array(list(singleton_chunk_sizes.values())))
        * fudge_factor
        * bytes_in_dtype[str(sd_xds[_sky].dtype)]
        / (1024**3)
    )
    if thread_info is None:
        thread_info = get_thread_info()
    logger.debug("Thread info " + str(thread_info))
    n_chunks_dict = calculate_data_chunking(
        memory_singleton_chunk,
        {"frequency": int_xds[_sky].sizes["frequency"]},
        thread_info,
        constant_memory=0,
        tasks_per_thread=4,
    )

    parallel_coords = {
        "frequency": make_parallel_coord(
            coord=int_xds.frequency, n_chunks=n_chunks_dict["frequency"]
        )
    }
    logger.info(
        "Number of frequency chunks: "
        + str(len(parallel_coords["frequency"]["data_chunks"]))
    )

    # --- Create the empty image on disk --------------------------------------
    # Coordinates and attributes are copied from the interferometer image. The
    # beam is written whole (not chunked): its parallel dim is frequency but the
    # feathered image simply adopts the interferometer beam unchanged.
    featherd_img_xds = xr.Dataset(coords=int_xds.coords)
    featherd_img_xds.attrs = copy.deepcopy(int_xds.attrs)
    featherd_img_xds[_beam] = int_xds[_beam].copy()
    # Drop encoding inherited from the source Zarr store: it carries a numcodecs
    # compressor that Zarr v3's to_zarr rejects. Cleared so xarray picks a
    # Zarr-v3-compatible default codec.
    for name in featherd_img_xds.variables:
        featherd_img_xds[name].encoding = {}
    write_image(
        featherd_img_xds,
        imagename=outim["name"],
        out_format="zarr",
        overwrite=outim["overwrite"],
    )

    # Pre-allocate the SKY data variable (NaN-filled) so each map task can lazily
    # write its own frequency slice in parallel (Zarr v3 compatible).
    create_empty_data_variables_on_disk(
        outim["name"],
        ["sky"],
        shape_dict=int_xds.sizes,
        parallel_coords=parallel_coords,
        compressor=compressor,
        double_precision=double_precision,
        data_variable_definitions="imaging",
    )

    # --- Build and run the map graph -----------------------------------------
    input_data = {"sd": sd_xds, "int": int_xds}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, input_data
    )

    input_params = {
        "input_data_store": {"sd": lowres, "int": highres},
        "image_store": outim["name"],
        "image_data_variables_keep": ["sky"],
        "axes": ("l", "m"),
        "sdfactor": sdfactor,
        "selection": selection,
        "fft_backend": fft_backend,
        "processing_function_threads": processing_function_threads,
        "to_disk": True,
        "compressor": compressor,
    }

    t0 = time.time()
    viper_graph = map(
        input_data=input_data,
        node_task_data_mapping=node_task_data_mapping,
        node_task=node_tasks.imaging.feather,
        input_params=input_params,
        in_memory_compute=False,
    )
    dask_graph = generate_dask_workflow(viper_graph)
    logger.debug("Time to create feather graph " + str(time.time() - t0) + "s")

    t0 = time.time()
    dask.compute(dask_graph)
    logger.info("Time to compute() feather " + str(time.time() - t0) + "s")

    zarr.consolidate_metadata(outim["name"])
