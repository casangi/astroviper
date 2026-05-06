







def PF_image_cube_single_field(input_params, ps_xdt, img_xds):
    import toolviper.utils.logger as logger
    from astroviper.core.imaging.residual_cycle import residual_cycle_cube_single_field
    from astroviper.core.imaging.model_update_cycle import model_update_cycle_cube_single_field
    import time


    logger.debug("Processing chunk " + str(input_params["task_id"]))
    
    is_n_iter_0 = True

    for i in range(input_params['iteration_control_params']['cycleniter']):
        print("$$$$************" * 10, i)
        start = time.time()
        img_xds, return_df = residual_cycle_cube_single_field(
            ps_xdt, img_xds, input_params, is_n_iter_0=is_n_iter_0
        )
        T_residual_cycle = time.time() - start
    
        
        if input_params["iteration_control_params"]["niter"] > 0:
            start = time.time()
            model_update_cycle_cube_single_field(img_xds, input_params, is_n_iter_0=is_n_iter_0, num_threads=input_params["processing_function_threads"], img_data_group_in_name = "residual", img_data_group_out_name = "model")
            T_model_update_cycle = time.time() - start
        else:
            T_model_update_cycle = 0.0
            
        is_n_iter_0 = False

    
    return_df["task_id"] = input_params["task_id"]
    return_df["n_channels"] = len(input_params["task_coords"]["frequency"]["data"])
    return_df["T_residual_cycle"] = T_residual_cycle
    return_df["T_model_update_cycle"] = T_model_update_cycle
    logger.debug("Timing info " + str(return_df))

    # #Write Data chunk to disk
    return img_xds, return_df



