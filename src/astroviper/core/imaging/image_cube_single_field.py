

def PF_image_cube_single_field(input_params, ps_iter, img_xds):
    import pandas as pd
    from xradio.image import make_empty_sky_image
    from xradio.measurement_set import load_processing_set
    import xarray as xr
    import zarr
    from xradio.measurement_set.load_processing_set import ProcessingSetIterator
    import toolviper.utils.logger as logger
    from astroviper.core.imaging.residual_cycle import residual_cycle_cube_single_field
    from astroviper.core.image_analysis.transform_polarization_basis import transform_polarization_basis

    logger.debug("Processing chunk " + str(input_params["task_id"]))

    # while loop here
    img_xds, return_df = residual_cycle_cube_single_field(
        ps_iter, img_xds, input_params, is_n_iter_0=True
    )
    
    # print("XXimg_xds ", img_xds["SKY_RESIDUAL"].sel(polarization="XX").values)
    # print("YY img_xds ", img_xds["SKY_RESIDUAL"].sel(polarization="YY").values)
    
    print("img_xds stokes values " + str(img_xds.coords["polarization"].values))
    img_xds = transform_polarization_basis(img_xds, new_polarization_basis="stokes", overwrite=True)
    
    # print("I img_xds ", img_xds["SKY_RESIDUAL"].sel(polarization="I").values)
    # print("Q img_xds ", img_xds["SKY_RESIDUAL"].sel(polarization="Q").values)
    # print("&&&&&&&&&&&&")

    return_df["task_id"] = input_params["task_id"]
    return_df["n_channels"] = len(input_params["task_coords"]["frequency"]["data"])

    # #Write Data chunk to disk
    return img_xds, return_df

