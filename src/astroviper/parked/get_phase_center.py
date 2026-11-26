def get_phase_center(residual_image_xds):
    """
    Get the phase center from the residual image coordinates.

    Parameters
    ----------
    residual_image_xds : xarray.Dataset
        Residual image dataset with 2-D ``right_ascension`` and
        ``declination`` coordinate arrays.

    Returns
    -------
    phase_center : str
        The phase center in the format ``"RA,Dec"`` (e.g.
        ``"12.345,-67.890"``), taken from the central pixel of the
        coordinate grids.
    """
    ra_shape = residual_image_xds.coords["right_ascension"].shape
    dec_shape = residual_image_xds.coords["declination"].shape

    cx_ra, cy_ra = ra_shape[0] // 2, ra_shape[1] // 2
    cx_dec, cy_dec = dec_shape[0] // 2, dec_shape[1] // 2

    ra0 = residual_image_xds.coords["right_ascension"].values[cx_ra, cy_ra]
    dec0 = residual_image_xds.coords["declination"].values[cx_dec, cy_dec]
    return f"{ra0},{dec0}"
