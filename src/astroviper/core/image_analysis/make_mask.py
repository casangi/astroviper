import xarray as xr
import numpy as np


from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)


def make_mask(
    img_xds: xr.Dataset,
    threshold: float,
    image_data_group_in_name="image",
    image_data_group_out_name="image",
    image_data_group_out_modified={"mask": "MASK"},
    combine_mask=False,
    overwrite=False,
):
    """Create or update a primary-beam threshold mask on an image dataset.

    The mask is built from the ``primary_beam`` data variable referenced by
    the input data group: pixels whose primary-beam response meets or
    exceeds ``threshold * peak(primary_beam)`` are marked ``True``.  The
    result is stored on ``img_xds`` under the data variable named by
    ``image_data_group_out_modified["mask"]`` and the output data group is
    registered (with a UTC timestamp and description) in
    ``img_xds.attrs["data_groups"]``.

    Currently only a primary-beam cutoff mask is supported; the function is
    structured so other mask types can be added later.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset.  Must contain the primary-beam data variable named
        by the input data group, and (when ``combine_mask=True``) the
        existing mask data variable likewise referenced by the input data
        group.  Modified in place: the new mask variable is added (or
        overwritten) and ``attrs["data_groups"]`` is updated.
    threshold : float
        Primary-beam cutoff expressed as a fraction of the peak primary-beam
        value.  Pixels with
        ``primary_beam >= threshold * nanmax(primary_beam)`` fall inside the
        mask.
    image_data_group_in_name : str, optional
        Key in ``img_xds.attrs["data_groups"]`` identifying the data group
        whose ``primary_beam`` (and, if combining, ``mask``) entries drive
        the computation.  Default ``"image"``.
    image_data_group_out_name : str, optional
        Key under which the resulting data group will be stored in
        ``img_xds.attrs["data_groups"]``.  Default ``"image"`` — i.e. update
        the input group in place.
    image_data_group_out_modified : dict, optional
        Role-to-variable-name overrides applied on top of the input data
        group when building the output data group.  Default
        ``{"mask": "MASK"}``: the ``mask`` role resolves to the ``"MASK"``
        data variable on the dataset.
    combine_mask : bool, optional
        If ``True``, combine (logical OR) the new threshold mask with the
        existing mask referenced by the input data group's ``mask`` role.
        Requires ``overwrite=True`` because the mask data variable is being
        modified in place.  Default ``False``.
    overwrite : bool, optional
        If ``True``, an existing output data group or output mask data
        variable is silently overwritten.  If ``False``, their presence
        raises ``AssertionError`` (see
        :func:`astroviper.utils.data_group_tools.create_data_groups_in_and_out`).
        Default ``False``.

    Returns
    -------
    None
        ``img_xds`` is modified in place: the new mask data variable is
        added (or overwritten) and ``attrs["data_groups"]`` is updated.

    Notes
    -----
    To use ``combine_mask=True``, ``overwrite`` must also be ``True`` so
    that the existing mask variable can be replaced.
    """

    # Resolve and validate the input/output data groups.  The helper checks
    # that image_data_group_in_name exists in img_xds.attrs["data_groups"],
    # guards against accidental overwrites when overwrite=False, and builds
    # data_group_out by layering image_data_group_out_modified on top of
    # data_group_in.
    image_data_group_in, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=overwrite,
    )

    if combine_mask:
        # Union the threshold mask with the existing mask: a pixel ends up
        # True if either it passes the primary-beam cutoff or it was already
        # True in the input mask.  np.nanmax ignores NaNs so they do not
        # poison the peak value used for the cutoff.
        img_xds[image_data_group_out["mask"]] = np.logical_or(
            img_xds[image_data_group_in["mask"]],
            img_xds[image_data_group_in["primary_beam"]].where(
                img_xds[image_data_group_in["primary_beam"]]
                >= threshold * np.nanmax(img_xds[image_data_group_in["primary_beam"]].values),
                False,
            ).astype(bool),
        )
        description = f"Mask created with threshold {threshold}."
    else:
        # Build the threshold mask from scratch: keep primary-beam values
        # that meet the cutoff and replace the rest with False, then cast
        # to bool so the kept (non-zero) pixels become True.
        img_xds[image_data_group_out["mask"]] = img_xds[
            image_data_group_in["primary_beam"]
        ].where(
            img_xds[image_data_group_in["primary_beam"]]
            >= threshold * np.nanmax(img_xds[image_data_group_in["primary_beam"]].values),
            False,
        ).astype(bool)
        description = f"Updated mask with threshold {threshold}."

    # Register the output data group on img_xds.attrs["data_groups"] and
    # append a UTC timestamp + the description above as an audit-trail
    # entry on the data group itself.
    modify_data_groups_xds(
        img_xds,
        image_data_group_out_name,
        image_data_group_out,
        description=description,
    )
    
    
    
    # # for simplicity for now, assume the relevant data are all in
    # # 'SKY' data variable
    # dv = "SKY"
    # # After image schema update use PRIMARY_BEAM data variable but
    # # for now use SKY data variable
    # mask_image = input_image.copy(deep=True)

    # if threshold < 0.0 or threshold > 1.0:
    #     raise ValueError("threshold must be between 0.0 and 1.0")
    # # if target_image is provided check dimensions
    # # check for l, m sizes also check for other dimensions (time, frequency, polarization)
    # if target_image is None:
    #     raise ValueError("Target image must be provided to check dimensions")
    # else:
    #     if (mask_image.coords["l"].size != target_image.coords["l"].size) or (
    #         mask_image.coords["m"].size != target_image.coords["m"].size
    #     ):
    #         raise ValueError(
    #             "Input mask image dimensions do not match target image dimensions"
    #         )
    #     # test for other dimensions of time, frequency, polarization
    #     if "frequency" in target_image.coords:
    #         if (
    #             mask_image.coords["frequency"].size
    #             != target_image.coords["frequency"].size
    #         ):
    #             raise ValueError(
    #                 "Input mask image frequency dimension does not match target image frequency dimension"
    #             )
    #     if "time" in target_image.coords:
    #         if mask_image.coords["time"].size != target_image.coords["time"].size:
    #             raise ValueError(
    #                 "Input mask image time dimension does not match target image time dimension"
    #             )
    #     if "polarization" in target_image.coords:
    #         if (
    #             mask_image.coords["polarization"].size
    #             != target_image.coords["polarization"].size
    #         ):
    #             raise ValueError(
    #                 "Input mask image polarization dimension does not match target image polarization dimension"
    #             )

    # if combine_mask:
    #     hasmask = True
    #     # check if target_image has existing mask and check attribute 'active_mask' that points the data variable contains MASK
    #     if "active_mask" in target_image[dv].attrs:
    #         print("Found 'active_mask' attr.")
    #         active_mask_dv = target_image[dv].attrs["active_mask"]
    #         if active_mask_dv in target_image:
    #             existing_mask = target_image[active_mask_dv].copy(deep=True)
    #             if active_mask_dv == "MASK0":
    #                 print("Inverting existing MASK0")
    #                 # CASA mask definition is reverse of xradio mask definition
    #                 existing_mask = ~existing_mask
    #             # combine existing mask with new mask using logical AND and ignoring time axis
    #             pb_thesh_mask = mask_image[dv] >= threshold

    #             combined_mask = mask_image[dv].where(
    #                 pb_thesh_mask.drop_vars("time") & existing_mask.drop_vars("time")
    #             )
    #         else:
    #             raise KeyError(
    #                 f"Active mask data variable '{active_mask_dv}' not found in target image"
    #             )
    #     else:
    #         if "MASK0" in target_image or "MASK" in target_image:
    #             # if no "active_mask" attr, check MASKO first then MASK
    #             if "MASK0" in target_image:
    #                 # CASA mask definition is reverse of xradio mask definition
    #                 existing_mask = ~target_image["MASK0"].copy(deep=True)
    #             else:
    #                 existing_mask = target_image["MASK"].copy(deep=True)
    #             # combine existing mask with new mask using logical AND and ignoring time axis
    #             pb_thesh_mask = mask_image[dv] >= threshold
    #             combined_mask = mask_image[dv].where(
    #                 pb_thesh_mask.drop_vars("time") & existing_mask.drop_vars("time")
    #             )
    #             # pb_thesh_mask = mask_image[dv] >= threshold
    #             # combined_mask = pb_thesh_mask & existing_mask
    #         else:
    #             hasmask = False
    #             # no existing mask found, print warning and proceed
    #             print(
    #                 "Warning: No existing mask found in target image. Creating new mask."
    #             )
    #             mask_image[dv] = mask_image[dv].where(mask_image[dv] >= threshold)
    # else:
    #     mask_image[dv] = mask_image[dv].where(mask_image[dv] >= threshold)

    # if combine_mask and hasmask:
    #     # print("coords of combined mask:", combined_mask.coords)x
    #     mask_image["MASK"] = mask_image[dv].where(combined_mask > 0, False).astype(bool)
    # else:
    #     mask_image["MASK"] = mask_image[dv] > 0

    # # generate mask only xds
    # mask_only_image = mask_image["MASK"]
    # if apply_on_target:
    #     # apply the mask on copy of target_image
    #     target_image_copy = target_image.copy(deep=True)
    #     if "MASK" in target_image_copy:
    #         target_image_copy["MASK"].values = mask_only_image
    #     else:
    #         target_image_copy["MASK"] = mask_only_image
    #     target_image_copy[dv].attrs["active_mask"] = "MASK"
    #     mask_xds = target_image_copy
    # else:
    #     mask_xds = xr.Dataset()
    #     mask_xds["MASK"] = mask_only_image
    # if output_image_name != "":
    #     from xradio.image import write_image

    #     write_image(mask_xds, output_image_name, out_format=output_format)
    #     return True
    # else:
    #     return mask_xds
