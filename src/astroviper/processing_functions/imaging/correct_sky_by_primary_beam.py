"""Primary-beam correction of the restored sky image."""


def correct_sky_by_primary_beam(
    img_xds,
    primary_beam_limit=0.2,
    image_data_group_in_name="restored",
    image_data_group_out_name="restored",
    overwrite=True,
):
    """Divide the restored sky by the primary beam (CASA ``pbcor``).

    The apparent sky of an interferometric image is attenuated by the (power)
    primary beam; dividing by it recovers the true sky flux scale away from the
    pointing centre.  Following the CASA definition the ``PRIMARY_BEAM`` data
    variable holds the power pattern (``P = |V| ** 2``), so the correction is a
    single division::

        SKY_RESTORED_PRIMARY_BEAM_CORRECTED = SKY_RESTORED / PRIMARY_BEAM

    Pixels where the primary beam is below ``primary_beam_limit`` are blanked
    with NaN (as ``tclean``/``impbcor`` blank below ``pblimit``); there the
    correction amplifies noise without bound.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset containing the input data group with ``sky`` (the
        restored sky) and ``primary_beam`` roles.  Modified in place: the
        corrected sky variable is added and registered on the output data
        group under the ``sky_primary_beam_corrected`` role.
    primary_beam_limit : float, optional
        Primary-beam (power) cutoff below which the corrected image is blanked
        with NaN, as a fraction of the beam peak.  Default ``0.2`` (the CASA
        ``pblimit`` default).
    image_data_group_in_name : str, optional
        Data group supplying the restored sky (``sky`` role) and the primary
        beam (``primary_beam`` role).  Default ``"restored"`` (the restored
        group inherits ``primary_beam`` from the residual group).
    image_data_group_out_name : str, optional
        Data group the corrected sky is registered under.  Default
        ``"restored"``.
    overwrite : bool, optional
        If ``True`` an existing corrected variable / group entry is
        overwritten.  Default ``True``.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with ``SKY_RESTORED_PRIMARY_BEAM_CORRECTED`` added.
    return_df : pandas.DataFrame
        One-row timing frame with the ``T_correct_sky_by_primary_beam`` column.

    See Also
    --------
    astroviper.processing_functions.imaging.primary_beam.make_primary_beam.make_primary_beam_single_field
    astroviper.processing_functions.imaging.restore.restore_image
    """
    import time

    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    start = time.time()

    image_data_group_in, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified={
            "sky_primary_beam_corrected": "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"
        },
        overwrite=overwrite,
    )

    assert "primary_beam" in image_data_group_in, (
        "Data group '"
        + image_data_group_in_name
        + "' has no primary_beam entry; run make_primary_beam_single_field first."
    )
    sky = img_xds[image_data_group_in["sky"]].values
    primary_beam = img_xds[image_data_group_in["primary_beam"]].values

    corrected = np.where(primary_beam >= primary_beam_limit, primary_beam, np.nan)
    corrected = (sky / corrected).astype(sky.dtype, copy=False)

    img_xds["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"] = xr.DataArray(
        corrected, dims=img_xds[image_data_group_in["sky"]].dims
    )
    img_xds["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"].attrs["type"] = "sky"

    modify_data_groups_xds(
        img_xds,
        data_group_out_name=image_data_group_out_name,
        data_group_out=image_data_group_out,
        description=(
            "Added primary-beam-corrected restored sky "
            f"(primary_beam_limit {primary_beam_limit})."
        ),
    )

    return img_xds, pd.DataFrame(
        {"T_correct_sky_by_primary_beam": [time.time() - start]}
    )
