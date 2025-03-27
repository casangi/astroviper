#!/usr/bin/env python
# coding: utf-8

from astropy import units as u
from astroviper._domain._imaging._fft import _fft_lm_to_uv
from astroviper._domain._imaging._ifft import _ifft_uv_to_lm
import copy
import dask
import dask.array as da
from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
)
from graphviper.graph_tools.coordinate_utils import make_parallel_coord
from graphviper.graph_tools.map import map
from toolviper.utils.display import dict_to_html
from IPython.display import HTML, display
import numpy as np
import os
import shutil
import time
from typing import Union
import xarray as xr
from xradio.image import read_image
from xradio.image import make_empty_aperture_image
from xradio.image import load_image
from xradio.image import write_image
import toolviper.utils.logger as logger

_sky = "SKY"
_beam = "BEAM"


def _compute_u_v(xds):
    shape = [xds.dims["l"], xds.dims["m"]]
    # sics = np.abs(xds.attrs["direction"]["reference"]["cdelt"])
    sics = np.abs(2 * [xds.l[1] - xds.l[0]])
    w_xds = make_empty_aperture_image(
        phase_center=[0, 0],
        image_size=shape,
        sky_image_cell_size=sics,
        chan_coords=[1],
        pol_coords=["I"],
        time_coords=[0],
    )
    u = np.zeros(shape)
    v = np.zeros(shape)
    for i, ux in enumerate(w_xds.coords["u"]):
        u[i, :] = ux
    for i, vx in enumerate(w_xds.coords["v"]):
        v[:, i] = vx
    return (u, v)


def _feather(input_params):
    # display(HTML(dict_to_html(input_params)))

    def _compute_w_multiple_beams(xds, uv):
        """xds is the single dish xds"""
        beams = xds[_beam]
        logger.debug("beams " + str(beams))
        w = np.zeros(xds[_sky].shape)
        bunit = beams.attrs["units"]
        bmaj = beams.sel(beam_param="major")
        logger.debug("bmaj orig shape" + str(bmaj.shape))
        # add l and m dims
        bmaj = np.expand_dims(bmaj, -1)
        bmaj = np.expand_dims(bmaj, -1)
        logger.debug("bmaj shape " + str(bmaj.shape))
        alpha = bmaj * u.Unit(bunit)
        alpha = alpha.to(u.rad).value
        bmin = beams.sel(beam_param="minor")
        bmin = np.expand_dims(bmin, -1)
        bmin = np.expand_dims(bmin, -1)
        beta = bmin * u.Unit(bunit)
        beta = beta.to(u.rad).value
        bpa = beams.sel(beam_param="pa")
        bpa = np.expand_dims(bpa, -1)
        bpa = np.expand_dims(bpa, -1)
        phi = bpa * u.Unit(bunit)
        phi = phi.to(u.rad).value

        alpha2 = alpha * alpha
        beta2 = beta * beta
        # u -> uu, v -> vv because we've already used
        # u for astropy.units
        uu, vv = uv
        uu = uu[np.newaxis, np.newaxis, np.newaxis, :, :]
        vv = vv[np.newaxis, np.newaxis, np.newaxis, :, :]
        aterm2 = (uu * np.sin(phi) - vv * np.cos(phi)) ** 2
        bterm2 = (uu * np.cos(phi) + vv * np.sin(phi)) ** 2
        w = np.exp(
            -np.pi * np.pi / 4.0 / np.log(2) * (alpha2 * aterm2 + beta2 * bterm2)
        )
        # w is an np.array
        return w

    # if input_params["input_data"] is None: #Load
    dtypes = {"sd": np.int32, "int": np.int32}
    for k in ["sd", "int"]:
        # the "data_selection" key is set in
        # interpolate_data_coords_onto_parallel_coords()
        # print("data store", input_params["input_data_store"][k])
        # print("block_des", input_params["data_selection"][k])
        xds = load_image(
            input_params["input_data_store"][k],
            block_des=input_params["data_selection"][k],
        )
        # print("load image for", k, "complete")
        # print("completed load_image()")
        fft_plane = (
            xds[_sky].dims.index(input_params["axes"][0]),
            xds[_sky].dims.index(input_params["axes"][1]),
        )
        # print("completed fft_plane")
        # else:
        #   img_xds = input_params["input_data"]['img'] #In memory
        aperture = _fft_lm_to_uv(xds[_sky], fft_plane)
        # print("completed _fft_im_to_uv()")
        dtypes[k] = xds[_sky].dtype
        if k == "int":
            int_ap = aperture
            int_xds = xds
            logger.debug("int_xds beam " + str(int_xds[_beam]))
        else:
            sd_ap = aperture
            sd_xds = xds
            logger.debug("sd_xds beam " + str(sd_xds[_beam]))
    mytype = dtypes["sd"] if dtypes["sd"] < dtypes["int"] else dtypes["int"]

    uv = _compute_u_v(sd_xds)
    w = _compute_w_multiple_beams(sd_xds, uv)

    one_minus_w = 1 - w
    s = input_params["s"]
    if _beam in int_xds.data_vars:
        int_ba = int_xds[_beam].sel(beam_param="major") * int_xds[_beam].sel(
            beam_param="minor"
        )
    else:
        error_message = "Unable to find BEAM data variable in interferometer image."
        logger.error(error_message)
        raise Exception(error_message)

    if _beam in sd_xds.data_vars:
        sd_ba = sd_xds[_beam].sel(beam_param="major") * sd_xds[_beam].sel(
            beam_param="minor"
        )
    else:
        error_message = "Unable to find BEAM data variable in single dish image."
        logger.error(error_message)
        raise Exception(error_message)
    # need to use values becuase the obs times will in general be different
    # which would cause a resulting shape with the time dimension having
    # length 0
    beam_ratio_values = int_ba.values / sd_ba.values
    # use interferometer coords
    beam_ratio = xr.DataArray(
        beam_ratio_values, dims=int_ba.dims, coords=int_ba.coords.copy()
    )

    beam_ratio = np.expand_dims(beam_ratio, -1)
    beam_ratio = np.expand_dims(beam_ratio, -1)

    term = (one_minus_w * int_ap + s * beam_ratio * sd_ap) / (one_minus_w + s * w)
    feather_npary = _ifft_uv_to_lm(term, fft_plane).astype(mytype)
    from xradio.image._util._zarr.zarr_low_level import write_chunk

    from xradio.image import make_empty_sky_image

    """
    # FIXME lon/latpole is not the phase center
    phase_center = [
        sd_xds.attrs["direction"]["longpole"]["value"],
        int_xds.attrs["direction"]["latpole"]["value"],
    ]

    featherd_img_chunk_xds = make_empty_sky_image(
        phase_center=phase_center,
        image_size=[int_xds.sizes["l"], int_xds.sizes["m"]],
        cell_size=int_xds.attrs["direction"]["reference"]["cdelt"],
        chan_coords=int_xds.frequency.values,
        pol_coords=int_xds.polarization.values,
        time_coords=[0],
    )
    """

    featherd_img_chunk_xds = xr.Dataset(coords=int_xds.coords)
    # we need an xradio function to return an ordered list of dimensions
    featherd_img_chunk_xds[_sky] = xr.DataArray(
        feather_npary, dims=["time", "frequency", "polarization", "l", "m"]
    )

    parallel_dims_chunk_id = dict(
        zip(input_params["parallel_dims"], input_params["chunk_indices"])
    )

    # print('input_params["zarr_meta"]',input_params["zarr_meta"])
    if input_params["to_disk"]:
        for data_variable, meta in input_params["zarr_meta"].items():
            write_chunk(
                featherd_img_chunk_xds,
                meta,
                parallel_dims_chunk_id,
                input_params["compressor"],
                input_params["image_file"],
            )

        results_dict = {}
        return results_dict
    else:
        results_dict = {"featherd_img_chunk_xds": featherd_img_chunk_xds}
        return featherd_img_chunk_xds


from numcodecs import Blosc


def feather_v2(
    outim: Union[dict, None],
    highres: str,
    lowres: str,
    sdfactor: float,
    selection: dict = {},
    memory_per_thread=None,
    thread_info=None,
    compressor=Blosc(cname="lz4", clevel=5),
):
    """
    Create an image from a single dish and interferometer image using the
    feather algorithm
    Parameters
    ----------
    outim : output image information, dict or None
        if None, no image is written. if dict must have keys "name" and
        "format" keys. "name" is the file name to which the image is written,
        and "format" is the format (casa or zarr) to write the image. An
        "overwrite" boolean parameter is optional. If it does not exist,
        it is assumed that the user does not want to overwrite an already
        extant image of the same name.
    highres : interferometer image, string or xr.Dataset.
        If str, an image file by that name will be read from disk.
        If xr.Dataset, that xds will be used for the interferometer image.
    lowres : Single dish image, string or xr.Dataset.
        If str, an image file by that name will be read from disk.
        If xr.Dataset, that xds will be used for the single dish image.
    """
    print("hi")
    if outim is not None:
        if type(outim) != dict:
            raise ValueError(
                "If specified, outim must be a dictionary with keys "
                "'name' and 'format'."
            )
        if "name" not in outim or "format" not in outim:
            raise ValueError(
                "If specfied, outim dict must have keys 'name' and 'format'"
            )
        im_format = outim["format"].lower()
        if not (im_format == "casa" or im_format == "zarr"):
            raise ValueError(
                f"Output image type {outim['format']} is not supported. "
                "Please choose either casa or zarr"
            )
        if "overwrite" not in outim or not outim["overwrite"]:
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
    from astroviper._utils.data_partitioning import bytes_in_dtype

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
    from astroviper._utils.data_partitioning import (
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

    # DEBUG
    # if w:
    #    from matplotlib import pyplot as plt
    #    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    #    plt.rcParams["figure.autolayout"] = True
    #    im = plt.imshow(w, cmap="copper_r")
    #    plt.colorbar(im)
    #    plt.show()

    # zn = highres
    # zo = lowres
    # DEBUG
    # imgs = [int_xds, sd_xds]
    # if "beam" in int_xds.data_vars:
    #    zn = "int_mb.zarr"
    #    if not os.path.exists(zn):
    #        write_image(int_xds, zn, "zarr")
    #
    # if "beam" in sd_xds.data_vars:
    #    zo = "sd_mb_1.zarr"
    #    if not os.path.exists(zo):
    #        write_image(sd_xds, zo, "zarr")

    # zarr_names = [zn, zo]
    # debug = read_image(zn)
    # print("beam shape", debug.beam.shape)

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
        featherd_img_xds.attrs["active_mask"] = ""

        write_image(
            featherd_img_xds,
            imagename=outim["name"],
            out_format=outim["format"],
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

        xds_dims = dict(int_xds.dims)
        # right now the keys are lower case, but the associated values are all caps
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
        node_task=_feather,
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
