import xradio
import numpy as np
import xarray as xr
import matplotlib.pylab as pl
from toolviper.utils.data import download
from astropy import units as u
from astropy import constants as const
from astroviper.core.imaging.imaging_utils.standard_grid import *


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
    nx = 1000
    download(file="Antennae_fld1_casa_lsrk.ps.zarr")
    ps_xdt = xr.open_datatree("Antennae_fld1_casa_lsrk.ps.zarr")
    origms = ps_xdt["Antennae_fld1_casa_lsrk_0"]
    # Select the first frequency from the origms dataset
    origms_subset = origms.isel(frequency=slice(0, 1))
    f = origms_subset.coords.frequency.values[0] * u.Unit(
        origms_subset.coords.frequency.units
    )
    invlam = f / const.c

    # Let's make a uvgrid from npointsources
    mod_im = np.zeros((nx, nx), dtype=np.float64)
    mod_im[
        (np.random.sample(nsources) * nx).astype(int),
        (np.random.sample(nsources) * nx).astype(int),
    ] = flux

    ft_mod = np.fft.fftshift((np.fft.fft2(mod_im)))

    uv_components = origms_subset.UVW.sel(uvw_label=["u", "v"])
    uv_squared = uv_components**2

    # Sum the squares along the uvw_label dimension
    sum_square_uv = uv_squared.sum(dim="uvw_label")
    max_uvdist = np.sqrt(sum_square_uv.max())
    cell = (0.5 / ((invlam).to(u.Unit("/m")).value * max_uvdist) * u.rad).to(
        "arcsec"
    )
    ny, nx = ft_mod.shape

    x_axis = np.linspace(-max_uvdist, max_uvdist, nx)
    y_axis = np.linspace(-max_uvdist, max_uvdist, ny)

    # Create the DataArray
    ft_mod_da = xr.DataArray(
        ft_mod,
        coords=[y_axis, x_axis],
        dims=[
            "v",
            "u",
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
            current_u = (
                uv_coords.isel(time=t, baseline_id=b).sel(uvw_label="u").values
            )
            current_v = (
                uv_coords.isel(time=t, baseline_id=b).sel(uvw_label="v").values
            )

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
                    replaced_visibility[t, b, f, p] = nearest_ft_value
    origms_subset["VISIBILITY"] = replaced_visibility
    return nx, cell, origms_subset


def test_standard_grid():
    npix, cell, ms4 = generate_ms4_with_point_sources(
        4, np.array(np.arange(1, 5))
    )
    vis = ms4.VISIBILITY
    params = {}
    params["image_size"] = npix
    params["cell"] = cell.to("rad").value
    params["complex_grid"] = True
    params
