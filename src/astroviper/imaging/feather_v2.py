#!/usr/bin/env python
# coding: utf-8

from astropy import units as u
from astroviper._domain._imaging._fft import _fft_lm_to_uv
from astroviper._domain._imaging._ifft import _ifft_uv_to_lm
import copy
import dask
import dask.array as da
from graphviper.graph_tools.coordinate_utils import interpolate_data_coords_onto_parallel_coords
from graphviper.graph_tools.coordinate_utils import make_parallel_coord
from graphviper.graph_tools.map import map
from graphviper.utils.display import dict_to_html
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
from xradio.image._util.common import _set_multibeam_array
import graphviper.utils.logger as logger

def _has_single_beam(xds):
    return "beam" in xds.attrs and xds.attrs["beam"]


def _beam_area_single_beam(xds):
    beam = xds.attrs["beam"]
    bmaj = beam["major"]
    major = u.Quantity(
        f"{bmaj['value']}{bmaj['units']}"
    )
    bmin = beam["minor"]
    minor = u.Quantity(
        f"{bmin['value']}{bmin['units']}"
    )
    area = major * minor
    return area.to(u.rad*u.rad)


def _compute_u_v(xds):
    shape = [xds.dims["l"], xds.dims["m"]]
    sics = np.abs(
        xds.attrs["direction"]["reference"]["cdelt"]
    )
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
        u[i,:] = ux
    for i, vx in enumerate(w_xds.coords["v"]):
        v[:, i] = vx
    return (u, v)


def _compute_w_single_beam(xds):
    """xds is the single dish (low res) xds"""
    (uu, vv) = _compute_u_v(xds)
    pi2 = np.pi * np.pi
    shape = [xds.dims["l"], xds.dims["m"]]
    w = np.zeros(shape)
    bmaj = xds.attrs["beam"]["major"]
    alpha = u.Quantity(
        f"{bmaj['value']}{bmaj['units']}"
    )
    alpha = alpha.to(u.rad).value
    bmin = xds.attrs["beam"]["minor"]
    beta = u.Quantity(
        f"{bmin['value']}{bmin['units']}"
    )
    beta = beta.to(u.rad).value
    bpa = xds.attrs["beam"]["pa"]
    phi = u.Quantity(
        f"{bpa['value']}{bpa['units']}"
    )
    phi = phi.to(u.rad).value
    alpha2 = alpha*alpha
    beta2 = beta*beta
    aterm2 = (uu*np.sin(phi) - vv*np.cos(phi))**2
    bterm2 = (uu*np.cos(phi) + vv*np.sin(phi))**2
    w = np.exp(
        -pi2/4.0/np.log(2)
        * (alpha2*aterm2 + beta2*bterm2)
    )
    # w is an np.array
    return w.astype(xds["sky"].dtype)


def _feather(input_params):
    #display(HTML(dict_to_html(input_params)))

    def _compute_w_multiple_beams(xds, uv):
        """xds is the single dish xds"""
        beams = xds["beam"]
        logger.debug("Beams shape" + str(beams.shape))
        w = np.zeros(xds["sky"].shape)
        bunit = beams.attrs["units"]
        bmaj = beams.sel(beam_param="major")
        # add l and m dims
        bmaj = np.expand_dims(bmaj, -1)
        bmaj = np.expand_dims(bmaj, -1)
        logger.debug("bmaj shape " + str(bmaj.shape))
        alpha = bmaj * u.Unit(bunit)
        alpha = alpha.to(u.rad).value
        bmin = beams.sel(beam_param="minor")
        bmin = np.expand_dims(bmin, -1)
        bmin - np.expand_dims(bmin, -1)
        beta = bmin * u.Unit(bunit)
        beta = beta.to(u.rad).value
        bpa = beams.sel(beam_param="pa")
        logger.debug("bpa shape " + str(bpa.shape))
        bpa = np.expand_dims(bpa, -1)
        bpa = np.expand_dims(bpa, -1)
        phi = bpa * u.Unit(bunit)
        phi = phi.to(u.rad).value

        alpha2 = alpha*alpha
        beta2 = beta*beta
        # u -> uu, v -> vv because we've already used
        # u for astropy.units
        uu, vv = uv
        uu = uu[np.newaxis, np.newaxis, np.newaxis, :, :]
        vv = vv[np.newaxis, np.newaxis, np.newaxis, :, :]
        logger.debug("u v shape" + str(uu.shape) + ", " + str(vv.shape))
        aterm2 = (uu*np.sin(phi) - vv*np.cos(phi))**2
        logger.debug("aterm2, shape" + str(aterm2.shape))
        bterm2 = (uu*np.cos(phi) + vv*np.sin(phi))**2
        logger.debug("bterm2 shape" + str(bterm2.shape))
        w = np.exp(
            -np.pi*np.pi/4.0/np.log(2)
            * (
                alpha2*aterm2 + beta2*bterm2
            )
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
            block_des=input_params["data_selection"][k]
        )
        # print("load image for", k, "complete")
        # print("completed load_image()")
        fft_plane = (
            xds['sky'].dims.index(input_params["axes"][0]),
            xds['sky'].dims.index(input_params["axes"][1])
        )
        # print("completed fft_plane")
        # else:
        #   img_xds = input_params["input_data"]['img'] #In memory
        aperture = _fft_lm_to_uv(xds["sky"], fft_plane)
        # print("completed _fft_im_to_uv()")
        dtypes[k] = xds["sky"].dtype
        if k == "int":
            int_ap = aperture
            int_xds = xds
        else:
            sd_ap = aperture
            sd_xds = xds
    logger.debug("fft loop complete")
    mytype = dtypes["sd"] if dtypes["sd"] < dtypes["int"] else dtypes["int"]

    if _has_single_beam(sd_xds):
        # w is a shape [l, m] np.array. 
        w = _compute_w_single_beam(sd_xds)
    else:
        # w must be computed on a per-plane basis.
        uv = compute_u_v(sd_xds)
        w = _compute_w_multiple_beams(sd_xds, uv)
    
    one_minus_w = 1 - w
    s = input_params["s"]
    beam_ratio = input_params["beam_ratio"]
    if beam_ratio is None:
        if "beam" in int_xds.data_vars:
            # compute area for multiple beams
            int_ba = (
                int_xds["beam"].sel(beam_param="major")
                * int_xds["beam"].sel(beam_param="minor")
            )
        else:
            int_ba = _beam_area_single_beam(int_xds)

        if "beam" in sd_xds.data_vars:
            # compute area for multiple beams
            sd_ba = (
                sd_xds["beam"].sel(beam_param="major")
                * sd_xds["beam"].sel(beam_param="minor")
            )
        else:
            sd_ba = _beam_area_single_beam(sd_xds)
        beam_ratio = int_ba/sd_ba
        beam_ratio = np.expand_dims(beam_ratio, -1)
        beam_ratio = np.expand_dims(beam_ratio, -1)

    # print("one_minus_w shape", one_minus_w.shape)
    # print("int_ap shape", int_ap.shape)
    # print("beam ratio shape", beam_ratio.shape)
    # print("sd_ap shape", sd_ap.shape)
    # print("w shape", w.shape)
    term = (
        (
            one_minus_w * int_ap
            + s * beam_ratio * sd_ap
        )
        / (one_minus_w + s * w)
    )
    feather_npary = _ifft_uv_to_lm(term, fft_plane).astype(mytype)

    from xradio.image._util._zarr.zarr_low_level import (
        write_chunk
    )


    from xradio.image import make_empty_sky_image

    phase_center = [sd_xds.attrs['direction']['longpole']['value'], int_xds.attrs['direction']['latpole']['value']]

    featherd_img_chunk_xds = make_empty_sky_image(
        phase_center=phase_center,
        image_size=[int_xds.sizes['l'],int_xds.sizes['m']],
        cell_size=int_xds.attrs['direction'][ 'reference']['cdelt'],
        chan_coords=int_xds.frequency.values,
        pol_coords=int_xds.polarization.values,
        time_coords=[0],
    )

    featherd_img_chunk_xds["SKY"] = xr.DataArray(feather_npary,dims=['time', 'polarization', 'frequency', 'l', 'm'])

    #print(featherd_img_chunk_xds["SKY"])

    parallel_dims_chunk_id = dict(
        zip(input_params["parallel_dims"], input_params["chunk_indices"])
    )

    #print('input_params["zarr_meta"]',input_params["zarr_meta"])
    if input_params["to_disk"]:
        for data_variable, meta in input_params["zarr_meta"].items():
            write_chunk(featherd_img_chunk_xds,meta,parallel_dims_chunk_id,input_params["compressor"],input_params["image_file"])

        results_dict = {}
        return results_dict
    else:
        results_dict = {'featherd_img_chunk_xds':featherd_img_chunk_xds}
        return featherd_img_chunk_xds


    # feather_xds = copy.deepcopy(int_xds)
    # # display(feather_xds)
    # feather_xrary = xr.DataArray(
    #     da.from_array(feather_npary, chunks=int_xds["sky"].shape),
    #     coords=int_xds["sky"].coords,
    #     dims=int_xds["sky"].dims
    # )
    # """
    # feather_xrary = xr.DataArray(
    #     da.from_array(feather_npary, chunks=int_xds["sky"].shape),
    #     coords=int_xds["sky"].coords,
    #     dims=int_xds["sky"].dims
    # )
    # """
    # feather_xrary.rename("sky")
    # feather_xds["sky"] = feather_xrary
    # # print("sky shape", feather_xds["sky"].shape)
    # return feather_xds


from numcodecs import Blosc

def feather_v2(
    outim: Union[dict, None], highres: str,
    lowres: str, sdfactor: float, selection: dict={},
    memory_per_thread = None,
    thread_info = None,
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
    sd_xds = read_image(lowres, selection=selection) if isinstance(lowres, str) else lowres
    # interferometer image
    int_xds = read_image(highres, selection=selection) if isinstance(highres, str) else highres
    if sd_xds["sky"].shape != int_xds["sky"].shape:
        raise RuntimeError("Image shapes differ")
    
    # Determine chunking
    from astroviper._utils.data_partitioning import bytes_in_dtype
    ## Determine the amount of memory required by the node task if all dimensions that chunking will occur on are singleton.
    ## For example feather does chunking only only frequency, so memory_singleton_chunk should be the amount of memory requered by _feather when there is a single frequency channel.
    singleton_chunk_sizes = dict(sd_xds['sky'].sizes) 
    del singleton_chunk_sizes['frequency'] #Remove dimensions that will be chuncked on. 
    fudge_factor = 1.1
    n_images_in_memory = 3.0  # Two input and one output image.
    memory_singleton_chunk = n_images_in_memory*np.prod(np.array(list(singleton_chunk_sizes.values())))*fudge_factor*bytes_in_dtype[str(sd_xds['sky'].dtype)]/(1024**3)
    
    chunking_dims_sizes = {'frequency':int_xds["sky"].sizes['frequency']} #Need to know how many frequency channels there are.
    from astroviper._utils.data_partitioning import calculate_data_chunking, get_thread_info
    if thread_info is None:
        thread_info = get_thread_info()
    logger.debug("Thread info " + str(thread_info))
    n_chunks_dict = calculate_data_chunking(memory_singleton_chunk, chunking_dims_sizes, thread_info, constant_memory=0, tasks_per_thread=4)


    # chans_per_chunk = 2**26/(sd_xds.dims["l"]*sd_xds.dims["m"])
    # chans_per_chunk = min(sd_xds.dims["frequency"], chans_per_chunk)
    # print("chans_per_chunk", chans_per_chunk)
    # chunksize = {
    #     "time": sd_xds.dims["time"], "polarization":
    #     sd_xds.dims["polarization"], "frequency": chans_per_chunk,
    #     "l": sd_xds.dims["l"], "m": sd_xds.dims["m"]
    # }
    # TODO either deep copy this, or put the chunks back as they were when
    # done
    # Comment: don't need.
    # sd_xds["sky"].chunk(chunksize)
    # int_xds["sky"].chunk(chunksize)
    # TODO set masked values to zero (might be better to do a deep copy then
    # if it doesn't take too long)

    ### DEBUG add multibeam
    #del sd_xds.attrs["beam"]
    # sky_shape = sd_xds["sky"].shape
    # beam_shape = sky_shape[:3] + (3,)
    # beam_ary = da.zeros(beam_shape)
    # beam_ary[:, :, :, 0:2] = 0.0040724349213201025
    # sd_xds = _set_multibeam_array(sd_xds, beam_ary, "rad")
    ###
    # sd_xds
    # interferometer image
    # int_xds = read_image("int.zarr")
    # chunksize = {
    #    "frequency": chans_per_chunk, "l": sd_xds.dims["l"],
    #    "m": sd_xds.dims["m"]
    # }
    # int_xds["sky"].chunk(chunksize)
    # int_xds.beam

    # add multibeam
    # from xradio.image._util.common import _set_multibeam_array
    # import dask.array as da
    # del int_xds.attrs["beam"]
    # beam_ary = da.zeros(beam_shape)
    # beam_ary[:, :, :, 0] = 0.0003175938332518091
    # beam_ary[:, :, :, 1] = 0.00031237226207452706
    # beam_ary[:, :, :, 2] = -0.054892581207198074
    # int_xds = _set_multibeam_array(int_xds, beam_ary, "rad")
    ###
    # int_xds

    # beam_ratio will be a scalar if both images have a single
    # beam, if not this computation needs to be done in the node
    # task since it will be per plane
    beam_ratio = None
    if _has_single_beam(sd_xds) and _has_single_beam(int_xds):
        beam_ratio = (
            _beam_area_single_beam(int_xds)
            /_beam_area_single_beam(sd_xds)
        )
    # print(beam_ratio)
    
    parallel_coords = {}
    # TODO need smarter way to get n_chunks
    # n_chunks = int(sd_xds.dims["frequency"]//chans_per_chunk)
    # if sd_xds.dims["frequency"] % chans_per_chunk > 0:
    #     n_chunks += 1
    # print("n_chunks", n_chunks)
    parallel_coords["frequency"] = make_parallel_coord(
        coord=sd_xds.frequency, n_chunks=n_chunks_dict['frequency']
    )
    # display(HTML(dict_to_html(parallel_coords["frequency"])))

    # FIXME need to do this for int_xds as well, can I do in one go
    # by making "img" a list of xdses?
    # JW says multiple items in the dict, the key names aren't important so can
    # be anything contextual to the problem at hand
    input_data = {"sd": sd_xds, "int": int_xds}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords, input_data)
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


    #Create empty image on disk
    to_disk = True
    if to_disk:
        from xradio.image import make_empty_sky_image

        phase_center = [sd_xds.attrs['direction']['longpole']['value'], int_xds.attrs['direction']['latpole']['value']]

        featherd_img_xds = make_empty_sky_image(
            phase_center=phase_center,
            image_size=[int_xds.sizes['l'],int_xds.sizes['m']],
            cell_size=int_xds.attrs['direction'][ 'reference']['cdelt'],
            chan_coords=parallel_coords["frequency"]["data"],
            pol_coords=int_xds.polarization.values,
            time_coords=[0],
        )
        write_image(featherd_img_xds, imagename=outim['name'], out_format=outim["format"], overwrite=outim["overwrite"])

        #Create the empty data variable SKY.
        from xradio.image._util._zarr.zarr_low_level import create_data_variable_meta_data_on_disk
        if int_xds.sky.dtype == np.float32:
            from xradio.image._util._zarr.zarr_low_level import image_data_variables_and_dims_single_precision as image_data_variables_and_dims
        elif int_xds.sky.dtype == np.float64:
            from xradio.image._util._zarr.zarr_low_level import image_data_variables_and_dims_double_precision as image_data_variables_and_dims
        else:
            error_message = "Unsupported data type of image " + str(int_xds.sky.dtype) + " expected float32 or float64."  
            logger.error(error_message)
            raise Exception(error_message)

        xds_dims = dict(int_xds.dims)
        #data_variables=["sky", "point_spread_function", "primary_beam"]
        data_variables=["sky"]
        data_varaibles_and_dims_sel = {
            key: image_data_variables_and_dims[key] for key in data_variables
        }
        zarr_meta = create_data_variable_meta_data_on_disk(
            outim['name'], data_varaibles_and_dims_sel, xds_dims, parallel_coords, compressor
        )

    input_params = {}
    input_params["input_data_store"] = {"sd": lowres, "int": highres}
    input_params["axes"] = ('l','m')#(3,4)
    input_params["image_file"] = outim['name']
    # beam_ratio should be computed inside _feather if
    # at least one image has multiple beams
    input_params["beam_ratio"] = beam_ratio
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
        in_memory_compute=False
    )
    from graphviper.graph_tools import generate_dask_workflow
    dask_graph = generate_dask_workflow(viper_graph)
    logger.debug("Time to create graph " + str(time.time() - t0))

    #dask.visualize(graph, filename="map_graph")
    t0 = time.time()
    res = dask.compute(dask_graph)
    logger.info("Time to compute() feather " + str(time.time() - t0) + "s")

    import zarr
    zarr.consolidate_metadata(outim['name'])