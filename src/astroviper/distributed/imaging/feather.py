#!/usr/bin/env python
# coding: utf-8

import copy
import dask
import dask.array as da
from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
)
from graphviper.graph_tools.coordinate_utils import make_parallel_coord
from graphviper.graph_tools.map import map

# from toolviper.utils.display import dict_to_html
import numpy as np
import os
import time
from typing import Union
import xarray as xr
from xradio.image import read_image
from xradio.image import write_image
import toolviper.utils.logger as logger
from numcodecs import Blosc

from astroviper.core.imaging.feather import feather_core

_sky = "SKY"
_beam = "BEAM"


def feather(
    outim: Union[dict, None],
    highres: str,
    lowres: str,
    sdfactor: float,
    selection: dict = {},
    thread_info=None,
    compressor=Blosc(cname="lz4", clevel=5),
):
    """
    Create an image from a single dish and interferometer image using the
    feather algorithm
    Parameters
    ----------
    outim : output image information, dict or None
        if None, no image is written (probably only useful for debugging). if
        a dict it must have a "name" key. "name" is the directory to which the
        zarr format image is written. An "overwrite" boolean parameter is
        optional. If it does not exist, it is assumed that the user does not
        want to overwrite an already extant image of the same name. Note that
        feather only writes zarr format images. If another output format is
        desired, the user/caller must convert the zarr format image to the
        desired format after running this function. The zarr file is used to
        accumulate the results of the computation, chunk by chunk,and so it is
        needed at the start of the computation and is not written in total at
        the end but rather is written to as each chunk is computed.
    highres : interferometer image, string or xr.Dataset.
        If str, an image file by that name will be read from disk.
        If xr.Dataset, that xds will be used for the interferometer image.
    lowres : Single dish image, string or xr.Dataset.
        If str, an image file by that name will be read from disk.
        If xr.Dataset, that xds will be used for the single dish image.
    """
    if outim is not None:
        if type(outim) != dict:
            raise ValueError(
                "If specified, outim must be a dictionary with required key "
                "'name' and optional key 'overwrite'."
            )
        if "name" not in outim:
            raise ValueError("If specfied, outim dict must have key 'name'.")
        if "overwrite" not in outim:
            outim["overwrite"] = False
        elif type(outim["overwrite"]) != bool:
            raise TypeError("If specified, outim['overwrite'] must be a boolean value")
        if not outim["overwrite"]:
            if os.path.exists(outim["name"]):
                raise RuntimeError(
                    f"Already existing file {outim['name']} will not be "
                    "overwritten. To overwrite it, set outim['overwrite'] = True"
                )

    # Read in input images
    # single dish image
    sd_xds = (
        read_image(lowres, selection=selection) if isinstance(lowres, str) else lowres
    )
    # interferometer image
    int_xds = (
        read_image(highres, selection=selection)
        if isinstance(highres, str)
        else highres
    )
    if sd_xds[_sky].shape != int_xds[_sky].shape:
        raise RuntimeError("Image shapes differ")

    # Determine chunking
    from astroviper.utils.data_partitioning import bytes_in_dtype

    ## Determine the amount of memory required by the node task if all dimensions that chunking will occur on are singleton.
    ## For example feather does chunking only only frequency, so memory_singleton_chunk should be the amount of memory requered by _feather when there is a single frequency channel.
    singleton_chunk_sizes = dict(sd_xds[_sky].sizes)
    del singleton_chunk_sizes[
        "frequency"
    ]  # Remove dimensions that will be chuncked on.
    fudge_factor = 1.1
    n_images_in_memory = 3.0  # Two input and one output image.
    memory_singleton_chunk = (
        n_images_in_memory
        * np.prod(np.array(list(singleton_chunk_sizes.values())))
        * fudge_factor
        * bytes_in_dtype[str(sd_xds[_sky].dtype)]
        / (1024**3)
    )

    chunking_dims_sizes = {
        "frequency": int_xds[_sky].sizes["frequency"]
    }  # Need to know how many frequency channels there are.
    from astroviper.utils.data_partitioning import (
        calculate_data_chunking,
        get_thread_info,
    )

    if thread_info is None:
        thread_info = get_thread_info()
    logger.debug("Thread info " + str(thread_info))
    n_chunks_dict = calculate_data_chunking(
        memory_singleton_chunk,
        chunking_dims_sizes,
        thread_info,
        constant_memory=0,
        tasks_per_thread=4,
    )

    parallel_coords = {}
    parallel_coords["frequency"] = make_parallel_coord(
        coord=sd_xds.frequency, n_chunks=n_chunks_dict["frequency"]
    )
    # display(HTML(dict_to_html(parallel_coords["frequency"])))

    # FIXME need to do this for int_xds as well, can I do in one go
    # by making "img" a list of xdses?
    # JW says multiple items in the dict, the key names aren't important so can
    # be anything contextual to the problem at hand
    input_data = {"sd": sd_xds, "int": int_xds}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, input_data
    )
    # display(HTML(dict_to_html(node_task_data_mapping)))

    # Create empty image on disk
    to_disk = True
    if to_disk:
        # create new xarray.Dataset with coordinates same as int_xds
        # but with no data

        """
        from xradio.image import make_empty_sky_image

        # the phase center is not the same as lon/lat pole
        phase_center = [
            sd_xds.attrs["direction"]["lonpole"]["data"],
            int_xds.attrs["direction"]["latpole"]["data"],
        ]

        featherd_img_xds = make_empty_sky_image(
            phase_center=phase_center,
            image_size=[int_xds.sizes["l"], int_xds.sizes["m"]],
            cell_size=int_xds.attrs["direction"]["reference"]["cdelt"],
            chan_coords=parallel_coords["frequency"]["data"],
            pol_coords=int_xds.polarization.values,
            time_coords=[0],
        )
        """

        featherd_img_xds = xr.Dataset(coords=int_xds.coords)
        featherd_img_xds.attrs = copy.deepcopy(int_xds.attrs)
        # we cannot build the beam in parallel because its parallel dims are no l, m
        # so just copy the whole thing here
        featherd_img_xds[_beam] = int_xds[_beam].copy()

        write_image(
            featherd_img_xds,
            imagename=outim["name"],
            out_format="zarr",
            overwrite=outim["overwrite"],
        )

        # Create the empty data variable SKY.
        from xradio.image._util._zarr.zarr_low_level import (
            create_data_variable_meta_data,
        )

        if int_xds[_sky].dtype == np.float32:
            from xradio.image._util._zarr.zarr_low_level import (
                image_data_variables_and_dims_single_precision as image_data_variables_and_dims,
            )
        elif int_xds[_sky].dtype == np.float64:
            from xradio.image._util._zarr.zarr_low_level import (
                image_data_variables_and_dims_double_precision as image_data_variables_and_dims,
            )
        else:
            error_message = (
                "Unsupported data type of image "
                + str(int_xds[_sky].dtype)
                + " expected float32 or float64."
            )
            logger.error(error_message)
            raise Exception(error_message)

        xds_dims = dict(int_xds.sizes)
        # right now the keys are lower case, but the associated values are all caps
        # the beam cannot be written in chunks because
        # ValueError: could not broadcast input array from shape (1,4,1,3) into shape (1,4,1,1024,1024)

        data_variables = ["sky"]
        data_varaibles_and_dims_sel = {
            key: image_data_variables_and_dims[key] for key in data_variables
        }
        zarr_meta = create_data_variable_meta_data(
            outim["name"],
            data_varaibles_and_dims_sel,
            xds_dims,
            parallel_coords,
            compressor,
        )

    input_params = {}
    input_params["input_data_store"] = {"sd": lowres, "int": highres}
    input_params["axes"] = ("l", "m")  # (3,4)
    input_params["image_file"] = outim["name"]
    input_params["s"] = 1

    if to_disk:
        input_params["to_disk"] = to_disk
        input_params["compressor"] = compressor
        input_params["zarr_meta"] = zarr_meta

    t0 = time.time()
    viper_graph = map(
        input_data=input_data,
        node_task_data_mapping=node_task_data_mapping,
        node_task=feather_core,
        input_params=input_params,
        in_memory_compute=False,
    )
    from graphviper.graph_tools import generate_dask_workflow

    dask_graph = generate_dask_workflow(viper_graph)
    logger.debug("Time to create graph " + str(time.time() - t0))

    # dask.visualize(graph, filename="map_graph")
    t0 = time.time()
    res = dask.compute(dask_graph)
    logger.info("Time to compute() feather " + str(time.time() - t0) + "s")

    import zarr

    zarr.consolidate_metadata(outim["name"])
