




def PF_image_cube_single_field(input_params, ps_iter, img_xds):
    import toolviper.utils.logger as logger
    from astroviper.core.imaging.residual_cycle import residual_cycle_cube_single_field
    from astroviper.core.imaging.model_update_cycle import model_update_cycle_cube_single_field


    logger.debug("Processing chunk " + str(input_params["task_id"]))

    # while loop here
    img_xds, return_df = residual_cycle_cube_single_field(
        ps_iter, img_xds, input_params, is_n_iter_0=True
    )
    
    if input_params["iteration_control_params"]["niter"] > 0:
        model_update_cycle_cube_single_field(img_xds, input_params, is_n_iter_0=True)
    
    print(input_params["iteration_control_params"]["niter"])
    print("@@@@@@@@@@@@@@@@@", img_xds.data_vars)
    

    
    return_df["task_id"] = input_params["task_id"]
    return_df["n_channels"] = len(input_params["task_coords"]["frequency"]["data"])
    logger.debug("Timing info " + str(return_df))

    # #Write Data chunk to disk
    return img_xds, return_df
