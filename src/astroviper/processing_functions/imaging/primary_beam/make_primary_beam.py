"""Primary-beam creation for the single-field cube imager."""


def make_primary_beam_single_field(
    img_xds,
    image_params,
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    list_dish_diameters=None,
    list_blockage_diameters=None,
    max_rad_1GHz=0.03113667385557884,
    ipower=2,
    float_dtype=None,
):
    """Add an azimuthally-symmetric primary beam to a single-field image dataset.

    Follows the CASA definition: the primary beam is the **power** sensitivity
    pattern of the antenna, the square of the absolute value of the voltage
    pattern (``P = |V| ** 2``).  The CASA ``PBMath1DAiry`` compatible obscured
    Airy pattern is evaluated per frequency channel with
    :func:`~astroviper.processing_functions.imaging.primary_beam.airy_disk.casa_airy_disk_response`
    -- the same function the simulation subdomain uses for its antenna voltage
    patterns -- and written as the ``PRIMARY_BEAM`` data variable (matching the
    ``.pb`` image of ``tclean`` to float32 precision), registered under
    ``image_data_group_out_name``.  The same beam is broadcast across
    polarization (a single dish diameter is assumed).

    The beam is created in whatever polarization basis ``img_xds`` currently
    carries; because ``PRIMARY_BEAM`` is stamped with ``method="airy_disk"`` it
    is skipped by
    :func:`~astroviper.processing_functions.image_analysis.transform_polarization_basis.transform_polarization_basis`,
    so it is invariant to a later basis change.  If the data group already has a
    ``primary_beam`` entry the function is a no-op.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset with ``time``, ``frequency``, ``polarization``, ``l`` and
        ``m`` coordinates.  Modified in place.
    image_params : dict
        Image geometry.  Must contain ``"image_size"`` (``(nx, ny)``) and
        ``"cell_size"`` (l, m cell size in radians).  The image centre is
        derived internally as ``image_size // 2``; the supplied dictionary is
        not mutated.
    image_data_group_in_name : str, optional
        Image data group whose existing entries are carried into the output
        group.  Default ``"residual"``.
    image_data_group_out_name : str, optional
        Name of the data group that the ``PRIMARY_BEAM`` variable is registered
        under.  Default ``"residual"``.
    list_dish_diameters : array-like of float, optional
        Antenna dish diameters in metres.  Default ``numpy.array([10.7])``
        (ALMA 12 m array effective value used by the reference imaging tests).
    list_blockage_diameters : array-like of float, optional
        Sub-reflector blockage diameters in metres.  Default
        ``numpy.array([0.75])``.
    max_rad_1GHz : float, optional
        CASA ``PBMath1DAiry`` maximum tabulated radius at 1 GHz in radians
        (scaled to the observing frequency).  Default is CASA's ALMA value.
    ipower : int, optional
        ``2`` returns the (power) primary beam -- the CASA definition;
        ``1`` returns the voltage pattern.  Default ``2``.
    float_dtype : numpy.dtype, optional
        Floating-point precision for the primary beam.  Defaults to
        ``numpy.float64``.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with ``PRIMARY_BEAM`` added (or unchanged if it
        already existed).
    return_df : pandas.DataFrame
        One-row timing frame with the ``T_primary_beam`` column.

    See Also
    --------
    astroviper.processing_functions.imaging.primary_beam.airy_disk.casa_airy_disk_response
    astroviper.processing_functions.imaging.make_point_spread_function.make_point_spread_function_single_field
    astroviper.processing_functions.imaging.correct_sky_by_primary_beam.correct_sky_by_primary_beam
    """
    import time

    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
        casa_airy_disk_response,
    )
    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    if float_dtype is None:
        float_dtype = np.float64
    if list_dish_diameters is None:
        list_dish_diameters = np.array([10.7])
    if list_blockage_diameters is None:
        list_blockage_diameters = np.array([0.75])

    start = time.time()

    data_group = img_xds.attrs["data_groups"][image_data_group_in_name]
    if data_group.get("primary_beam", None) is not None:
        return img_xds, pd.DataFrame({"T_primary_beam": [0.0]})

    image_data_group_in, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified={"primary_beam": "PRIMARY_BEAM"},
        overwrite=False,
    )

    # Evaluate the CASA-compatible Airy beam of the first (only) dish diameter
    # on the image's (l, m) grid for every channel, then broadcast over
    # polarization and add a leading time axis.
    pb = casa_airy_disk_response(
        img_xds.l.values[None, :, None],
        img_xds.m.values[None, None, :],
        img_xds.frequency.values[:, None, None],
        float(np.asarray(list_dish_diameters)[0]),
        float(np.asarray(list_blockage_diameters)[0]),
        max_rad_1GHz,
        ipower=ipower,
    ).astype(float_dtype, copy=False)
    img_xds["PRIMARY_BEAM"] = xr.DataArray(
        np.tile(pb[None, :, None, :, :], (1, 1, img_xds.sizes["polarization"], 1, 1)),
        dims=("time", "frequency", "polarization", "l", "m"),
    )
    img_xds["PRIMARY_BEAM"].attrs["type"] = "primary_beam"
    img_xds["PRIMARY_BEAM"].attrs["method"] = "airy_disk"

    modify_data_groups_xds(
        img_xds,
        data_group_out_name=image_data_group_out_name,
        data_group_out=image_data_group_out,
        description="Added primary beam to data group.",
    )

    return img_xds, pd.DataFrame({"T_primary_beam": [time.time() - start]})
