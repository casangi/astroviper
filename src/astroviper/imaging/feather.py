#!/usr/bin/env python
# coding: utf-8

# <h1>Feather Tutorial</h1>

# In[44]:


# from graphviper.dask.client import local_client
# viper_client = local_client(cores=4, memory_limit="4GB")

import dask
dask.config.set(scheduler="synchronous")
# dask.config.set(scheduler="threads")


# Read in input images

# In[45]:


from xradio.image import read_image
# single dish image
sd_xds = read_image("sd.zarr")
chans_per_chunk = 2**28/(sd_xds.dims["l"]*sd_xds.dims["m"])
chans_per_chunk = min(sd_xds.dims["frequency"], chans_per_chunk)
chunksize = {
    "frequency": chans_per_chunk, "l": sd_xds.dims["l"],
    "m": sd_xds.dims["m"]
}
sd_xds["sky"].chunk(chunksize)
sd_xds


# In[46]:


# interferometer image
int_xds = read_image("int.zarr")
chunksize = {
    "frequency": chans_per_chunk, "l": sd_xds.dims["l"],
    "m": sd_xds.dims["m"]
}
int_xds["sky"].chunk(chunksize)
int_xds


# In[47]:


if sd_xds["sky"].shape != int_xds["sky"].shape:
    raise RuntimeError("Image shapes differ")


# In[48]:


from astropy import units as u

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

    """
    elif "beams" in xds.data_vars:
        # TODO deal when there are multiple beams
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
    """
    

def has_single_beam(xds):
    return "beam" in xds.attrs and xds.attrs["beam"]
    
# beam_ratio will be a scalar if both images have a single
# beam, if not this computation needs to be done in the node
# task since it will be per plane
beam_ratio = None
if has_single_beam(sd_xds) and has_single_beam(int_xds):
    beam_ratio = (
        _beam_area_single_beam(int_xds
        )/_beam_area_single_beam(sd_xds)
    )
beam_ratio


# In[49]:


from graphviper.graph_tools.coordinate_utils import make_parallel_coord
from graphviper.utils.display import dict_to_html
from IPython.display import HTML, display

parallel_coords = {}
n_chunks = min(16, sd_xds.dims["frequency"])
parallel_coords["frequency"] = make_parallel_coord(
    coord=sd_xds.frequency, n_chunks=n_chunks
)
display(HTML(dict_to_html(parallel_coords["frequency"])))


# In[50]:


from graphviper.graph_tools.coordinate_utils import interpolate_data_coords_onto_parallel_coords

input_data = {"img": sd_xds}
node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords, input_data)
display(HTML(dict_to_html(node_task_data_mapping)))


# In[52]:


from xradio.image import make_empty_apeture_image
import numpy as np
from astropy import units as u


def compute_u_v(xds):
    shape = [xds.dims["l"], xds.dims["m"]]
    sics = np.abs(
        xds.attrs["direction"]["reference"]["cdelt"]
    )
    w_xds = make_empty_apeture_image(
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

def compute_w_single_beam(xds):
    """xds is the single dish (low res)
    xds"""
    (uu, vv) = compute_u_v(xds)
    pi2 = np.pi * np.pi
    shape = [xds.dims["l"], xds.dims["m"]]
    """
    sics = np.abs(
        xds.attrs["direction"]["reference"]["cdelt"]
    )
    w_xds = make_empty_apeture_image(
        phase_center=[0, 0],
        image_size=shape,
        sky_image_cell_size=sics,
        chan_coords=[1],
        pol_coords=["I"],
        time_coords=[0],
    )
    """
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
    return w


w = None
uv = (None, None)
if has_single_beam(sd_xds):
    # w is a shape [l, m] np.array. If not computed
    # here, it will be computed on a per chunk basis 
    # inside the node task
    w = compute_w_single_beam(sd_xds)
else:
    # w must be computed on a per-plane basis, but
    # u and v can be computed once here and then
    # input to the node task; they do not need to be
    # computed per-plane
    uv = compute_u_v(sd_xds)

# comment out the following code if the single dish image
# has multiple beams
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
im = plt.imshow(w, cmap="copper_r")
plt.colorbar(im)
plt.show()




# In[92]:


import copy
from graphviper.graph_tools.map import map
import dask
import numpy as np
import xarray as xr


def _feather(input_parms):
    display(HTML(dict_to_html(input_parms)))

    
    def _fft(xds):
        display(xds)

        fft_plane = (
            xds['sky'].dims.index(input_parms["axes"][0]),
            xds['sky'].dims.index(input_parms["axes"][1])
        )
        print('fft_plane',fft_plane)
        aperture = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(xds.sky, axes=fft_plane),
                axes=fft_plane
            ), axes=fft_plane
         ).real
    
        # img_xds['APERTURE'] = xr.DataArray(aperture, dims=('time','polarization','frequency','u','v'))
        return aperture

    def _ifft(ary):
        plane = (3, 4)
        img = np.fft.ifftshift(
            np.fft.ifft2(
                np.fft.fftshift(ary, axes=plane),
                axes=plane
            ), axes=plane
        ).real
        return img
        
    
    def _compute_w_multiple_beams(xds, uv):
        """xds is the single dish xds"""
        beams = xds["beams"]
        w = np.zeros(xds["sky"].shape)
        bunit = beams.attrs["units"]
        bmaj = beams.sel(beam_param="major").squeeze(5)
        alpha = u.Quantity(
            f"{bmaj.values}{bunit}"
        )
        alpha = alpha.to(u.rad).value
        bmin = beams.sel(beam_param="minor").squeeze(5)
        beta = u.Quantity(
            f"{bmin.values}{bunit}"
        )
        beta = beta.to(u.rad).value
        bpa = beams.sel(beam_param="pa").squeeze(5)
        phi = u.Quantity(f"{bpa.values}{bunit}")
        phi = phi.to(u.rad).value
        """
        shape = xds["sky"].shape
        uu = np.zeros(shape)
        vv = np.zeros(shape)
        for i, ux in enumerate(w_xds.coords["u"]):
            uu[:, :, :, i, :] = ux
        for i, vx in enumerate(w_xds.coords["v"]):
            vv[:, :, :, :, i] = vx
        """
        alpha2 = alpha*alpha
        beta2 = beta*beta
        # u -> uu, v -> vv because we've already used
        # u for astropy.units
        uu, vv = uv
        aterm2 = (uu*np.sin(phi) - vv*np.cos(phi))**2
        bterm2 = (uu*np.cos(phi) + vv*np.sin(phi))**2
        w = np.exp(
            -pi2/4.0/np.log(2)
            * (
                alpha2*aterm2 + beta2*bterm2
            )
        )
        # w is an np.array
        return w

    


    from xradio.image import load_image
    # if input_parms["input_data"] is None: #Load 
    print(f"selection {input_parms['data_selection']['img']}")
    for i, data_store in enumerate(input_parms["input_data_store"]):
        xds = load_image(
            data_store, 
            block_des=input_parms["data_selection"]["img"]
        )
        print(data_store)
        # else:
        #   img_xds = input_parms["input_data"]['img'] #In memory
        aperture = _fft(xds)
        print(f"aperture shape {aperture.shape}")
        if i == 0:
            int_ap = aperture
            int_xds = xds
        else:
            sd_ap = aperture
            sd_xds = xds
    w = (
        input_parms["w"]
        if "w" in input_parms
        else _compute_w_multiple_beams(sd_xds, input_parms["uv"])
    )
    print(f"w shape {w.shape}")
    one_minus_w = 1 - w
    print(f"one_minus_w shape {one_minus_w.shape}")
    s = input_parms["s"]
    print(f"s {s}")
    beam_ratio = input_parms["beam_ratio"]
    print(f"beam_ratio {beam_ratio}")
    print(f"int_ap shape {int_ap.shape}")
    term = (
        (
            one_minus_w * int_ap
            + s * beam_ratio * sd_ap
        )
        / (one_minus_w + s * w)
    )
    print(f"term shape {term.shape}")
    feather_npary = _ifft(term)
    print(f"feather npary shape {feather_npary.shape}")
    feather_xds = copy.deepcopy(int_xds)
    display(feather_xds)
    feather_xrary = xr.DataArray(
        feather_npary, coords=int_xds["sky"].coords,
        dims=int_xds["sky"].dims
    )
    print(f"feather_xrary shape {feather_xrary.shape}")
    feather_xrary.rename("sky")
    feather_xds["sky"] = feather_xrary
    return feather_xds  


imgs = [int_xds, sd_xds]
zarr_names = ["int.zarr", "sd.zarr"]

input_parms = {}
input_parms["input_data_store"] = zarr_names
input_parms["axes"] = ('l','m')#(3,4)
# beam_ratio should be computed inside _feather if
# at least one image has multiple beams
input_parms["beam_ratio"] = beam_ratio
input_parms["w"] = w
input_parms["uv"] = uv
input_parms["s"] = 1

graph = map(
    input_data=input_data,
    node_task_data_mapping=node_task_data_mapping,
    node_task=_feather,
    input_parms=input_parms,
    in_memory_compute=False
)

dask.visualize(graph, filename="map_graph")


# In[94]:


res = dask.compute(graph)


# In[99]:


type(res), type(res[0]),type(res[0][0]), type(res[0][0][0])


# In[103]:


len(res), len(res[0]), len(res[0][0])


# In[107]:


res[0][0][0]["sky"].plot()


# In[115]:


final_xds = xr.concat(res[0][0], "frequency")
final_xds


# In[117]:


0.0003175938332518091*180/np.pi


# In[120]:


u.Quantity("0.00031237226207452706rad").to(u.arcsec)


# In[123]:


u.Quantity("-7.27220521664304e-05rad").to("arcsec")


# In[ ]:




