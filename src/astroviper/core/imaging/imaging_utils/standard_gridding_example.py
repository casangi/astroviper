import xradio
import numpy as np
import xarray as xr
import os
from toolviper.utils.data import download
from astropy import units as u
from astropy import constants as const
from astroviper.core.imaging.imaging_utils.standard_grid import *
from astroviper.core.imaging.imaging_utils.gcf_prolate_spheroidal import *
from astroviper.core.imaging.fft import fft_lm_to_uv


def generate_ms4_with_point_sources(
    nsources: int = 2, flux: np.ndarray = np.array([1.0, 5.0])
):
    """
    Parameters
    ----------
    nsources : int,
        DESCRIPTION. The default is 2.
    flux : np.ndarray, length of nsources
        DESCRIPTION. The default is np.array([1.0, 5.0]).

    Returns
    -------
    A tuple of (npix, cell), ms4

    """
    npix = 200
    if not os.path.exists("Antennae_fld1_casa_lsrk.ps.zarr"):
        download(file="Antennae_fld1_casa_lsrk.ps.zarr", decompress=True)
    ps_xdt = xr.open_datatree("Antennae_fld1_casa_lsrk.ps.zarr")
    origms = ps_xdt["Antennae_fld1_casa_lsrk_0"]
    # Select the first frequency from the origms dataset
    origms_subset = origms.isel(frequency=slice(0, 1))
    f = origms_subset.coords["frequency"].values[0] * u.Unit(
        origms_subset.coords["frequency"].units
    )
    invlam = f / const.c

    # Let's make a uvgrid from npointsources
    mod_im = np.zeros((npix, npix), dtype=np.float64)

    # limit the sources in inner region as sources further out will not really
    # obey ft
    sources = [
        (np.random.sample(nsources) * npix / 4 + npix * 3 / 8).astype(int),
        (np.random.sample(nsources) * npix / 4 + npix * 3 / 8).astype(int),
        flux,
    ]
    # DEBUG
    # sources = [[501], [502], [10]]
    mod_im[
        sources[0],
        sources[1],
    ] = sources[2]
    # convolve it with a gaussian of 5 pixels
    # mod_im = gaussian_filter(mod_im, sigma=5)
    mod_im = np.pad(mod_im, pad_width=((1000, 1000), (1000, 1000)), constant_values=0.0)
    ft_mod = fft_lm_to_uv(mod_im, axes=[0, 1])

    uv_components = origms_subset.UVW.sel(uvw_label=["u", "v"])
    uv_squared = uv_components**2

    # Sum the squares along the uvw_label dimension
    sum_square_uv = uv_squared.sum(dim="uvw_label")
    max_uvdist = np.sqrt(sum_square_uv.max()).values
    cell = ((0.5 / ((invlam).to(u.Unit("/m")).value * max_uvdist)) * u.rad).to("arcsec")
    ny, nx = ft_mod.shape

    x_axis = np.linspace(-max_uvdist, max_uvdist, nx)
    y_axis = np.linspace(-max_uvdist, max_uvdist, ny)

    # Create the DataArray
    ft_mod_da = xr.DataArray(
        ft_mod,
        coords=[x_axis, y_axis],
        dims=[
            "u",
            "v",
        ],  # Assign appropriate dimension names (e.g., 'v' for y, 'u' for x)
    )

    # Create a copy of the VISIBILITY data to modify
    replaced_visibility = origms_subset.VISIBILITY.copy()
    uv_coords = origms_subset.UVW.sel(uvw_label=["u", "v"])

    # Iterate through the time and baseline_id dimensions of the visibility data
    # Note: This iteration can be slow for large datasets.
    # More efficient approaches might involve xarray's advanced indexing or interpolation methods
    # if the data structure allows for it. However, given the request, we'll proceed with iteration
    # to explicitly show the nearest neighbor replacement logic.

    # Get the dimensions
    num_time = replaced_visibility.sizes["time"]
    num_baseline = replaced_visibility.sizes["baseline_id"]
    num_frequency = replaced_visibility.sizes[
        "frequency"
    ]  # Should be 1 for origms_subset
    num_polarization = replaced_visibility.sizes["polarization"]

    # Get the coordinates from ft_mod_da for nearest neighbor lookup
    u_coords_ft = ft_mod_da.u.values
    v_coords_ft = ft_mod_da.v.values

    # Iterate through each element of the visibility data
    for t in range(num_time):
        for b in range(num_baseline):
            # Get the 'u' and 'v' coordinate for the current time and baseline
            current_u = uv_coords.isel(time=t, baseline_id=b).sel(uvw_label="u").values
            current_v = uv_coords.isel(time=t, baseline_id=b).sel(uvw_label="v").values

            # Handle potential NaN values in UVW coordinates
            if np.isnan(current_u) or np.isnan(current_v):
                # If UV is NaN, we might want to keep the original value or set it to NaN
                # For this example, we'll just skip replacement for NaN UVW.
                continue

            # Find the index of the nearest 'u' coordinate in ft_mod_da
            nearest_u_index = np.argmin(np.abs(u_coords_ft - current_u))

            # Find the index of the nearest 'v' coordinate in ft_mod_da
            nearest_v_index = np.argmin(np.abs(v_coords_ft - current_v))

            # Get the value from ft_mod_da at the nearest 'u' and 'v' coordinates
            nearest_ft_value = ft_mod_da.isel(
                u=nearest_u_index, v=nearest_v_index
            ).values
            # Replace the VISIBILITY value for all frequencies and polarizations at this time and baseline
            # Since origms_subset has frequency dimension of size 1, we can iterate or use slicing
            for f in range(num_frequency):
                for p in range(num_polarization):
                    replaced_visibility[t, b, f, p] = np.conj(nearest_ft_value)
    origms_subset["VISIBILITY"] = replaced_visibility

    return sources, npix, cell, origms_subset


def make_standard_grid_image(dopsf=False):
    nsources = 4
    sources, npix, cell, ms4 = generate_ms4_with_point_sources(
        nsources, np.ones(nsources)
    )
    vis_data = ms4.VISIBILITY.data
    uvw = ms4.UVW.data
    # setting all the weights to 1
    dims = ms4.dims
    weight = np.ones([dims["time"], dims["baseline_id"], dims["frequency"], 1])
    freq_chan = ms4.coords["frequency"].values

    params = {}
    params["image_size_padded"] = np.array([npix, npix], dtype=int)
    params["cell_size"] = np.array([cell.to("rad").value, cell.to("rad").value])
    params["complex_grid"] = True
    params["oversampling"] = 100
    params["support"] = 7
    params["do_psf"] = dopsf
    params["chan_mode"] = "continuum"

    cgk_1D = create_prolate_spheroidal_kernel_1D(
        params["oversampling"], params["support"]
    )
    grid, sumwgt = standard_grid_numpy_wrap(
        vis_data, uvw, weight, freq_chan, cgk_1D, params
    )

    kernel, corrTerm = create_prolate_spheroidal_kernel(
        params["oversampling"], params["support"], params["image_size_padded"]
    )
    dirty_im = (
        np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid[0, 0, :, :]))))
        / corrTerm
        * npix
        * npix
        / sumwgt
    )
    for k in range(nsources):
        print(
            f"source {k}  at [{sources[0][k], sources[1][k]}] flux  in image {dirty_im[sources[0][k], sources[1][k]]}, should be {sources[2][k]}"
        )
    ppos = np.where(dirty_im > 0.9)
    for k in range(len(ppos[0])):
        print(
            f"Peak found at [{ppos[0][k], ppos[1][k]}] and value is {dirty_im[ppos[0][k], ppos[1][k]]} "
        )
