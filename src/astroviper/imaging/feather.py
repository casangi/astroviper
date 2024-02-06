#!/usr/bin/env python
# coding: utf-8

# <h1>Feather Tutorial</h1>

# In[38]:


# from graphviper.dask.client import local_client
# viper_client = local_client(cores=4, memory_limit="4GB")

import dask
from astropy import units as u
from graphviper.graph_tools.coordinate_utils import interpolate_data_coords_onto_parallel_coords
from graphviper.graph_tools.coordinate_utils import make_parallel_coord
from graphviper.graph_tools.map import map
from graphviper.utils.display import dict_to_html
from IPython.display import HTML, display
import numpy as np
import xarray as xr
from xradio.image import load_image
from xradio.image import make_empty_apeture_image
from xradio.image import read_image

def _beam_area(xds):
    # This is not the true area, but just the product of bmaj*bmin
    if xds.attrs["beam"]:
        beam = xds.attrs["beam"]
        major = u.Quantity(
            f"{beam['major']['value']}{beam['major']['units']}"
        )
        minor = u.Quantity(
            f"{beam['minor']['value']}{beam['minor']['units']}"
        )
        area = major * minor
        return area.to(u.rad*u.rad)
    elif "beams" in xds.data_vars:
        area = xr.DataArray(
            (
                xds.beam.sel(beam_param=["major"]).values
                * xds.beam.sel(beam_param=["minor"]).values
            ).squeeze(3),
            dims=["time", "polarization", "frequency"],
            coords=dict(
                time=xds.time,
                polarization=xds.polarization,
                frequency=xds.frequency
            )
        )
        bu = u.Unit(xds["beam"].attrs["units"])
        units = bu * bu
        f = units.to(u.rad * u.rad)
        area *= f
        area.attrs["units"] = u.rad * u.rad
        return area
    else:
        raise RuntimeError("xds has no beam (single or multiple")

def _compute_w_single_beam(xds):
    """xds is the single dish (low res) xds"""
    pi2 = np.pi * np.pi
    shape = [xds.dims.l, xds.dims.m]
    sics = np.abs(
        xds.attrs["direction"]["reference"]["cdelt"]
    )
    w_xds = make_empty_apeture_image(
        phase_center=[0, 0],
        image_size=[xds.dims.l, xds.dims.m],
        sky_image_cell_size=sics,
        chan_coords=[1],
        pol_coords="I",
        time_coords=[0],
    )
    w = np.zeros(shape)
    maj = xds.attrs["beam"]["major"]
    alpha = u.Quantity(
        f"{maj['value']){maj['units']}"
    )
    alpha = alpha.to(u.rad)
    bmin = xds.attrs["beam"]["minor"]
    beta = u.Quantity(
        f"{bmin['value']){bmin['units']}"
    )
    beta = beta.to(u.rad)
    bpa = xds.attrs["beam"]["pa"]
    phi = u.Quantity(
        f"{bpa['value']){bpa['units']}"
    )
    phi = phi.to(u.rad)
    u = np.zeros(shape)
    v = np.zeros(shape)
    for i, uu in enumerate(w_xds.coords["u"]):
        u[i,:] = uu
    for i, vv in enumerate(w_xds.coords["v"]):
        v[:, i] = vv
    alpha2 = alpha*alpha
    beta2 = beta*beta
    aterm2 = (u*np.sin(phi) - v*np.cos(phi))**2
    bterm2 = (u*np.cos(phi) + v*np.sin(phi))**2
    w = np.exp(
        -pi2/4.0/np.log(2)
        * (
            alpha2*(aterm2 + bterm2)
        )
    )
    # w is an np.array
    return w


def _feather(input_parms):
    # display(HTML(dict_to_html(input_parms)))
    def _fft(xds):
        # display(xds)
        fft_plane = (
            xds['sky'].dims.index(input_parms["axes"][0]),
            xds['sky'].dims.index(input_parms["axes"][1])
        )
        # print('fft_plane',fft_plane)
        aperture = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(xds.sky, axes=fft_plane),
                axes=fft_plane
            ), axes=fft_plane
        ).real
        # img_xds['APERTURE'] = xr.DataArray(aperture, dims=('time','polarization','frequency','u','v'))
        return aperture


    def _compute_w_multiple_beams(xds):
        """xds is the single dish xds"""
        beams = xds["beams"]
        w = np.zeros(xds["sky"].shape)
        bunit = beams.attrs["units"]
        bmaj = beams.sel(beam_param="major").squeeze(5)
        alpha = u.Quantity(
            f"{bmaj.values){bunit})"
        )
        alpha = alpha.to(u.rad)
        bmin = beams.sel(beam_param="minor").squeeze(5)
        beta = u.Quantity(
            f"{bmin.values}{bunit}"
        )
        beta = beta.to(u.rad)
        bpa = beams.sel(beam_param="pa").squeeze(5)
        phi = u.Quantity(
            f"{bpa.values}{bunit}"
        phi = phi.to(u.rad)
        shape = xds["sky"].shape
        u = np.zeros(shape)
        v = np.zeros(shape)
        for i, uu in enumerate(w_xds.coords["u"]):
            u[:, :, :, i, :] = uu
        for i, vv in enumerate(w_xds.coords["v"]):
            v[:, :, :, :, i] = vv
        alpha2 = alpha*alpha
        beta2 = beta*beta
        aterm2 = (u*np.sin(phi) - v*np.cos(phi))**2
        bterm2 = (u*np.cos(phi) + v*np.sin(phi))**2
        w = np.exp(
            -pi2/4.0/np.log(2)
            * (
                alpha2*(aterm2 + bterm2)
            )
        )
        # w is an xr.DataArray which wraps a dask array
        return w




    # if input_parms["input_data"] is None: #Load
    for data_store in input_parms["input_data_store"]:
        xds = load_image(
            data_store,
            block_des=input_parms["data_selection"]["img"]
        )
        # else:
        #   img_xds = input_parms["input_data"]['img'] #In memory
        aperture = _fft(xds)
        if i == 0:
            int_ap = aperture
        else:
            sd_ap = aperture
            sd_xds = xds
        w = (
            input_parms["w"]
            if "w" in input_parms
            else _compute_w_multiple_beams(sd_xds)
        )
        one_minus_w = 1 - w
        s = input_parms["s"]
        beam_ratio = input_parms["beam_ratio"]
        term = (
            (
                one_minus_w * int_ap["aperture"]
                + s * beam_ratio] * sd_ap["aperture"]
            )
            / (one_minus_w) + s * w
        )

def _init_dask():
    dask.config.set(scheduler="synchronous")
    # dask.config.set(scheduler="threads")


def feather(imagename=None,highres=None,lowres=None, sdfactor=None):
    _init_dask()
    # interferometer image
    int_xds = read_image(highres)
    # single dish image
    sd_xds = read_image(lowres)
    if sd_xds["sky"].shape != int_xds["sky"].shape:
        raise RuntimeError("Image shapes differ")

    npol = sd_xds.dims["polarization"]
    chans_per_chunk = 2**(28//npol)/(sd_xds.dims["l"]*sd_xds.dims["m"])
    chans_per_chunk = min(sd_xds.dims["frequency"], chans_per_chunk)
    chunksize = {
        "polarization": npol, "frequency": chans_per_chunk,
        "l": sd_xds.dims["l"], "m": sd_xds.dims["m"]
    }
    # TODO either deep copy this, or put the chunks back as they were when
    # done
    sd_xds["sky"].chunk(chunksize)
    int_xds["sky"].chunk(chunksize)
    # TODO set masked values to zero (might be better to do a deep copy then
    # if it doesn't take too long)

    # beam_ratio will be a scalar if both images have a single
    # beam, or an DataArray if at least one of them has
    # multiple beams
    beam_ratio = _beam_area(int_xds)/_beam_area(sd_xds)

    parallel_coords = {}
    # TODO n_chunks needs to be more general
    n_chunks = min(16, sd_xds.dims["frequency"])
    parallel_coords["frequency"] = make_parallel_coord(
        coord=sd_xds.frequency, n_chunks=n_chunks
    )
    # display(HTML(dict_to_html(parallel_coords["frequency"])))
    input_data = {"img": sd_xds}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, input_data
    )
    # display(HTML(dict_to_html(node_task_data_mapping)))
    w = None
    if "beam" in sd_xds.attrs and sd_xds.attrs["beam"]:
        # w is a [1, 1, 1, l, m] np.array. If not computed
        # here, it will be computed on a per chunk basis
        # inside the node task
        w = _compute_w_single_beam(sd_xds)

    imgs = [int_xds, sd_xds]
    zarr_names = [highres, lowres]
    input_parms = {}
    input_parms["input_data_store"] = zarr_names
    input_parms["axes"] = ('l','m')#(3,4)
    input_parms["beam_ratio"] = beam_ratio
    input_parms["w"] = w

    graph = map(
        input_data=input_data,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_feather,
        input_parms=input_parms,
        in_memory_compute=False
    )

    # dask.visualize(graph, filename="map_graph")
    graph.compute()



