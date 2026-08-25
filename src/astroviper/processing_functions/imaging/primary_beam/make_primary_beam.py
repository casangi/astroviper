"""Primary-beam creation for the single-field cube imager."""


def make_primary_beam_single_field(
    img_xds,
    image_params,
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    list_dish_diameters=None,
    list_blockage_diameters=None,
    ipower=1,
    float_dtype=None,
):
    """Add an azimuthally-symmetric primary beam to a single-field image dataset.

    Evaluates the obscured-Airy-disk voltage pattern (via
    :func:`~astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric.airy_disk_rorder_v2`)
    for every frequency channel and writes it as the ``PRIMARY_BEAM`` data
    variable, registering it under ``image_data_group_out_name``.  The same beam
    is broadcast across polarization (a single dish diameter is assumed).

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
    ipower : int, optional
        ``1`` returns the voltage pattern, ``2`` returns the power beam.
        Default ``1``.
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
    astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric.airy_disk_rorder_v2
    astroviper.processing_functions.imaging.make_point_spread_function.make_point_spread_function_single_field
    """
    import time

    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric import (
        airy_disk_rorder_v2,
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

    pb_params = {
        "list_dish_diameters": np.asarray(list_dish_diameters),
        "list_blockage_diameters": np.asarray(list_blockage_diameters),
        "ipower": ipower,
    }

    # The airy-disk evaluation needs the image centre in pixels. Build a local
    # copy of image_params so the caller's dictionary is not mutated.
    pb_image_params = {
        **image_params,
        "image_center": (np.array(image_params["image_size"]) // 2).tolist(),
    }

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

    img_xds["PRIMARY_BEAM"] = xr.DataArray(
        # Select the first (only) dish diameter and add a leading time axis.
        airy_disk_rorder_v2(
            img_xds.frequency.values,
            img_xds.polarization.values,
            pb_params,
            pb_image_params,
            dtype=float_dtype,
        )[0, ...][None, ...],
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
