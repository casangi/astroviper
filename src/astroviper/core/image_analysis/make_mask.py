import numpy as np
import xarray as xr


def make_mask(
    input_image: xr.DataArray,
    pb_threshold: float,
    target_image: xr.DataArray = None,
    apply_on_target: bool = False,
    combine_mask: bool = False,
    output_image_name: str = "",
    output_format: str = "zarr",
):
    """
    Create or update mask image

    Parameters:
    input_image : xarray.DataArray
        Input image name (assume pb image for now)
    pb_threshold : float
        PB cutoff with respect to the peak value
    target_image : xarray.DataArray
        Target image to the generated mask to be used
        to check dimensions and existing mask
    apply_on_target : boolean
        Whether to apply the mask on target_image (default: False)
    combine_mask : boolean
        Whether to combine with existing mask in target_image (default: False)
    output_image_name : str (optional)
        Output mask image name

    Returns:
        xarray.Dataset
        Boolean array stored in the data variable 'MASK' in the xradio image
        if apply_on_target is True, the returned xds contains the target_image data with
        added 'MASK' data variable. Otherwise xds with only 'MASK' data variable is returned.
    """
    # for simplicity for now, assume the relevant data are all in
    # 'SKY' data variable
    dv = "SKY"
    # aster image schema update use PRIMARY_BEAM data variable but
    # for now use SKY data variable
    mask_image = input_image.copy(deep=True)

    # if target_image is provided check dimensions
    # check for l, m sizes also check for other dimensions (time, frequency, polarization)
    if target_image is None:
        raise ValueError("Target image must be provided to check dimensions")
    else:
        if (mask_image.coords["l"].size != target_image.coords["l"].size) or (
            mask_image.coords["m"].size != target_image.coords["m"].size
        ):
            raise ValueError(
                "Input mask image dimensions do not match target image dimensions"
            )
        # test for other dimensions of time, frequency, polarization
        if "frequency" in target_image.coords:
            if (
                mask_image.coords["frequency"].size
                != target_image.coords["frequency"].size
            ):
                raise ValueError(
                    "Input mask image frequency dimension does not match target image frequency dimension"
                )
        if "time" in target_image.coords:
            if mask_image.coords["time"].size != target_image.coords["time"].size:
                raise ValueError(
                    "Input mask image time dimension does not match target image time dimension"
                )
        if "polarization" in target_image.coords:
            if (
                mask_image.coords["polarization"].size
                != target_image.coords["polarization"].size
            ):
                raise ValueError(
                    "Input mask image polarization dimension does not match target image polarization dimension"
                )

    if combine_mask:
        # check if target_image has existing mask and check attribute 'active_mask' that points the data variable contains MASK
        if "active_mask" in target_image.attrs:
            active_mask_dv = target_image.attrs["active_mask"]
            if active_mask_dv in target_image:
                existing_mask = target_image[active_mask_dv]
                # combine existing mask with new mask using logical OR
                combined_mask = np.where(
                    (mask_image[dv] >= pb_threshold) | (existing_mask > 0), 1, 0
                )
                mask_image = combined_mask
            else:
                raise KeyError(
                    f"Active mask data variable '{active_mask_dv}' not found in target image"
                )
        else:
            if "MASK0" in target_image or "MASK" in target_image:
                if "MASK0" in target_image:
                    # CASA mask definition is reverse of xradio mask definition
                    existing_mask = ~target_image["MASK0"]
                else:
                    existing_mask = target_image["MASK"]
                # combine existing mask with new mask using logical OR
                combined_mask = mask_image[dv].where(
                    (mask_image[dv] >= pb_threshold) & existing_mask
                )
                mask_image[dv] = combined_mask
            else:
                # no existing mask found, print warning and proceed
                print(
                    "Warning: No existing mask found in target image. Creating new mask."
                )
                mask_image[dv] = mask_image[dv].where(mask_image[dv] >= pb_threshold)
    else:
        mask_image[dv] = mask_image[dv].where(mask_image[dv] >= pb_threshold)

    mask_image["MASK"] = mask_image[dv] > 0

    # generate mask only xds
    mask_only_image = mask_image["MASK"]
    if apply_on_target:
        # apply the mask on target_image
        target_image["MASK"] = mask_only_image
        mask_xds = target_image
    else:
        mask_xds = xr.Dataset()
        mask_xds["MASK"] = mask_only_image
    if output_image_name != "":
        from xradio.image import write_image

        write_image(mask_xds, output_image_name, out_format=output_format)
        return True
    else:
        return mask_xds
