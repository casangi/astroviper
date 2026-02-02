def make_psf(vis, im_params, grid_params):
    """
    Generate psf

    Parameters:
    ----------
    vis : xarray.Dataset
        The visibility data as an xarray Dataset.
    im_params : dict
        Imaging parameters: must include
        'cell_size': angular size of a pixel 2d tuple (rad, rad),
        'image_size': int 2d tuple (nx, ny),
        'image_center': image center pixel coordinates int 2d tuple (x, y),
        'phase_center': phase reference center (RA, Dec) in radians,
        'chan_mode': channel mode for imaging.
    grid_params : dict
        Gridding parameters: must include
        'sampling': sampling factor for gridding,
        'complex_grid': boolean indicating if complex grid is used,
        'support': support size for gridding.
    Returns:
        xarray.DataArray
    """
    from xradio.image import make_empty_sky_image
    from astroviper.core.imaging.imaging_utils.standard_grid import (
        grid2image_spheroid_ms4,
    )
    import xarray as xr
    import numpy as np

    vis_data = vis.VISIBILITY.data
    uvw = vis.UVW.data

    dims = vis.dims
    freq_chan = vis.coords["frequency"].values
    nfreq = len(freq_chan)
    image_size = im_params["image_size"]
    cell_size = im_params["cell_size"]
    phase_center = im_params["phase_center"]
    complex_grid = True
    do_psf = True
    chan_mode = im_params["chan_mode"]
    sampling = grid_params["sampling"]
    complex_grid = grid_params["complex_grid"]
    support = grid_params["support"]
    time_coords = vis.coords["time"].values[0]
    pol = vis.coords["polarization"].values
    npol = len(pol)
    incr = cell_size[0]

    psf_data = np.zeros([nfreq, npol, image_size[0], image_size[1]], dtype=float)
    grid2image_spheroid_ms4(
        vis=vis,
        resid_array=psf_data,
        pixelincr=np.array([-incr, incr]),
        support=support,
        sampling=sampling,
        dopsf=True,
        chan_mode=chan_mode,
    )

    psf_data_reshaped = np.expand_dims(psf_data, axis=0)

    psf_xds = make_empty_sky_image(
        phase_center=phase_center,
        image_size=image_size,
        cell_size=cell_size,
        frequency_coords=freq_chan,
        pol_coords=pol,
        time_coords=time_coords,
        direction_reference="fk5",
        projection="SIN",
        do_sky_coords=True,
    )
    new_dims = tuple(d for d in psf_xds.dims if d != "beam_params_label")
    coords = psf_xds.drop_vars("beam_params_label").coords
    psf_da = xr.DataArray(
        psf_data_reshaped,
        dims=new_dims,
        coords=coords,
        name="POINT_SPREAD_FUNCTION",
    )

    psf_xds["POINT_SPREAD_FUNCTION"] = psf_da

    return psf_xds
