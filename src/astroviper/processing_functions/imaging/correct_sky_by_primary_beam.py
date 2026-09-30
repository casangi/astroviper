"""Primary beam correction of the restored sky image."""


def correct_sky_by_primary_beam(
    img_xds,
    primary_beam_limit=0.2,
    primary_beam_correction_order="correct_then_restore",
    image_data_group_in_name="restored",
    image_data_group_out_name="restored",
    image_data_group_in_model_name="model",
    image_data_group_in_residual_name="residual",
    beam_fit_params_key="beam_fit_params_point_spread_function",
    beam_polarization_index=0,
    processing_function_threads=1,
    overwrite=True,
):
    """Primary beam corrected restored sky, in one of two conventions.

    The apparent sky of an interferometric image is attenuated by the (power)
    primary beam ``P`` (``PRIMARY_BEAM`` holds ``|V| ** 2``, the CASA
    definition). With ``M`` the model, ``R`` the residual and ``B`` the clean
    beam, the corrected restored image is

    ``"correct_then_restore"`` (default)
        ``SKY_RESTORED_PRIMARY_BEAM_CORRECTED = (M / P) * B + R / P``. The
        model is divided by the primary beam where it is a pixel value and
        then convolved with the clean beam, so for a model that represents
        the apparent sky ``I P`` the model part is the true sky convolved with
        the clean beam, ``I * B``, exactly.
    ``"restore_then_correct"``
        ``SKY_RESTORED_PRIMARY_BEAM_CORRECTED = SKY_RESTORED / P``, the
        convention of CASA ``pbcor``. Multiplication by ``P`` and convolution
        with ``B`` do not commute, so this is exact only where the primary
        beam is flat across the clean beam: a point source of flux ``F`` at
        ``x0`` comes out as ``F B(x - x0) P(x0) / P(x)``, exact at the source
        and off beside it by up to the clean beam sigma times the gradient of
        ``ln P``, several percent near the half power radius of a compact
        configuration, with a bias of about a percent on the flux integrated
        over the source. Kept for comparisons with CASA products.

    The residual part ``R / P`` is the same in both conventions. Pixels where
    the primary beam is below ``primary_beam_limit`` are blanked with NaN (as
    ``tclean``/``impbcor`` blank below ``pblimit``); there the correction
    amplifies noise without bound.

    :func:`~astroviper.processing_functions.imaging.restore.restore_image`
    makes the same corrected image in its own pass when asked to, which is
    what the imaging cycle uses; this function serves an image that has been
    restored already.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset with the restored data group (``sky`` and
        ``primary_beam`` roles) and, for ``"correct_then_restore"``, the
        model data group and the residual data group with the clean beam fit
        (``beam_fit_params_key`` role).  Modified in place: the corrected sky
        variable is added and registered on the output data group under the
        ``sky_primary_beam_corrected`` role.
    primary_beam_limit : float, optional
        Primary beam (power) cutoff below which the corrected image is blanked
        with NaN, as a fraction of the beam peak.  Default ``0.2`` (the CASA
        ``pblimit`` default).
    primary_beam_correction_order : str, optional
        ``"correct_then_restore"`` (default) or ``"restore_then_correct"``,
        see above.
    image_data_group_in_name : str, optional
        Data group supplying the restored sky (``sky`` role) and the primary
        beam (``primary_beam`` role).  Default ``"restored"`` (the restored
        group inherits ``primary_beam`` from the residual group).
    image_data_group_out_name : str, optional
        Data group the corrected sky is registered under.  Default
        ``"restored"``.
    image_data_group_in_model_name, image_data_group_in_residual_name : str, optional
        Data groups of the model and of the residual, read for
        ``"correct_then_restore"``.  Defaults ``"model"`` and ``"residual"``.
    beam_fit_params_key : str, optional
        Role key in the residual data group holding the ``[major, minor, pa]``
        clean beam fit.  Default ``"beam_fit_params_point_spread_function"``.
    beam_polarization_index : int, optional
        Polarization index of the beam fit used for the clean beam.  Default
        ``0``.
    processing_function_threads : int, optional
        Threads handed to ``scipy.fft`` for the convolution.  Default ``1``.
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
    astroviper.processing_functions.imaging.restore.primary_beam_corrected_plane
    """
    import time

    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.processing_functions.imaging.restore import (
        PRIMARY_BEAM_CORRECTION_ORDERS,
        _clean_beam_kernel_ft,
        primary_beam_corrected_plane,
    )
    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    start = time.time()

    if primary_beam_correction_order not in PRIMARY_BEAM_CORRECTION_ORDERS:
        raise ValueError(
            "primary_beam_correction_order must be one of "
            f"{PRIMARY_BEAM_CORRECTION_ORDERS}; got {primary_beam_correction_order!r}."
        )

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
    sky_da = img_xds[image_data_group_in["sky"]]
    primary_beam = img_xds[image_data_group_in["primary_beam"]].values
    nt, nf, npol = sky_da.shape[:3]
    workers = (
        processing_function_threads
        if (processing_function_threads and processing_function_threads > 0)
        else -1
    )

    if primary_beam_correction_order == "correct_then_restore":
        data_groups = img_xds.attrs["data_groups"]
        for name in (image_data_group_in_model_name, image_data_group_in_residual_name):
            assert name in data_groups, (
                f"Data group '{name}' not found in img_xds data_groups: "
                + str(list(data_groups.keys()))
            )
        residual_group = data_groups[image_data_group_in_residual_name]
        assert beam_fit_params_key in residual_group, (
            "Beam-fit parameters '"
            + beam_fit_params_key
            + "' not found in the residual data group '"
            + image_data_group_in_residual_name
            + "'. Run point_spread_function_gaussian_fit first."
        )
        model = img_xds[data_groups[image_data_group_in_model_name]["sky"]].values
        residual = img_xds[residual_group["sky"]].values
        beam = img_xds[residual_group[beam_fit_params_key]].values
        l = img_xds["l"].values
        delta = abs(float(l[1] - l[0]))
        ny, nx = residual.shape[-2:]
        corrected = np.empty_like(residual)
        for tt in range(nt):
            for ff in range(nf):
                kernel_ft = _clean_beam_kernel_ft(
                    beam[tt, ff, beam_polarization_index],
                    ny,
                    nx,
                    delta,
                    residual.dtype,
                    workers,
                )
                for pp in range(npol):
                    pb_pol = pp if primary_beam.shape[2] == npol else 0
                    corrected[tt, ff, pp] = primary_beam_corrected_plane(
                        primary_beam[tt, ff, pb_pol],
                        primary_beam_limit,
                        primary_beam_correction_order,
                        model_plane=model[tt, ff, pp],
                        residual_plane=residual[tt, ff, pp],
                        kernel_ft=kernel_ft,
                        workers=workers,
                    )
    else:
        sky = sky_da.values
        corrected = np.empty_like(sky)
        for tt in range(nt):
            for ff in range(nf):
                for pp in range(npol):
                    pb_pol = pp if primary_beam.shape[2] == npol else 0
                    corrected[tt, ff, pp] = primary_beam_corrected_plane(
                        primary_beam[tt, ff, pb_pol],
                        primary_beam_limit,
                        primary_beam_correction_order,
                        restored_plane=sky[tt, ff, pp],
                    )

    img_xds["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"] = xr.DataArray(
        corrected, dims=sky_da.dims
    )
    img_xds["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"].attrs["type"] = "sky"

    modify_data_groups_xds(
        img_xds,
        data_group_out_name=image_data_group_out_name,
        data_group_out=image_data_group_out,
        description=(
            "Added primary-beam-corrected restored sky ("
            f"{primary_beam_correction_order}, primary_beam_limit {primary_beam_limit})."
        ),
    )

    return img_xds, pd.DataFrame(
        {"T_correct_sky_by_primary_beam": [time.time() - start]}
    )
