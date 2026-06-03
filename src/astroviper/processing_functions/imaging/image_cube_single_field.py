

def image_cube_single_field(input_params, ps_xdt, img_xds):
    import toolviper.utils.logger as logger
    from astroviper.processing_functions.imaging.residual_cycle import residual_cycle_cube_single_field
    from astroviper.processing_functions.imaging.model_update_cycle import model_update_cycle_cube_single_field
    import time
    from astroviper.processing_functions.imaging.iteration_control import (
        IterationController,
        ReturnDict,
        merge_return_dicts,
    )


    logger.debug("Processing chunk " + str(input_params["task_id"]))
    
    controller = IterationController(
        niter=input_params["iteration_control_params"]["niter"],
        nmajor=input_params["iteration_control_params"]["nmajor"],
        threshold=input_params["iteration_control_params"]["threshold"],
        gain=input_params["iteration_control_params"]["gain"],
        cyclefactor=input_params["iteration_control_params"]["cyclefactor"],
        minpsffraction=input_params["iteration_control_params"]["minpsffraction"],
        maxpsffraction=input_params["iteration_control_params"]["maxpsffraction"],
        cycleniter=input_params["iteration_control_params"]["cycleniter"],
    )
    combined_return_dict = ReturnDict()

    is_n_iter_0 = True
    T_residual_cycle = 0.0
    T_model_update_cycle = 0.0
    i_cycles = 0
    while controller.stopcode.major == 0:
        i_cycles += 1
        print("$$$$************" * 10, i_cycles)
        start = time.time()
        img_xds, return_df = residual_cycle_cube_single_field(
            ps_xdt, img_xds, input_params, is_n_iter_0=is_n_iter_0
        )
        T_residual_cycle = T_residual_cycle + time.time() - start

        if input_params["iteration_control_params"]["niter"] > 0:
            logger.debug("Doing model update")
            print("^^^^^^"*10)
            print(combined_return_dict)
            print("^^^^^^"*10)
            cycle_niter, cyclethresh = get_calculate_cycle_controls(controller, combined_return_dict, img_xds, is_n_iter_0, iteration_control_params=input_params["iteration_control_params"])
            
            input_params["iteration_control_params"]["cycleniter"] = cycle_niter
            input_params["iteration_control_params"]["threshold"] = cyclethresh
            start = time.time()
            deconvolve_dict = model_update_cycle_cube_single_field(img_xds, input_params, is_n_iter_0=is_n_iter_0, num_threads=input_params["processing_function_threads"], img_data_group_in_name = "residual", img_data_group_out_name = "model")
            T_model_update_cycle = T_model_update_cycle + time.time() - start
        else:
            deconvolve_dict = ReturnDict()
            
        is_n_iter_0 = False
        
        controller.update_counts(deconvolve_dict)
        combined_return_dict = merge_return_dicts([combined_return_dict, deconvolve_dict])

        stopcode, stopdesc = controller.check_convergence(deconvolve_dict)
        if stopcode.major != 0:
            logger.debug(f"  *** CONVERGED: {stopdesc} ***")
            break
                
    #Last residual cycle to calcultate final residual image after last model update cycle.
    if input_params["iteration_control_params"]["niter"] > 0:       
        start = time.time()
        img_xds, return_df = residual_cycle_cube_single_field(
            ps_xdt, img_xds, input_params, is_n_iter_0=is_n_iter_0
        )
        T_residual_cycle = T_residual_cycle + time.time() - start
        

    return_df["task_id"] = input_params["task_id"]
    return_df["n_channels"] = len(input_params["task_coords"]["frequency"]["data"])
    return_df["T_residual_cycle"] = T_residual_cycle
    return_df["T_model_update_cycle"] = T_model_update_cycle

    # #Write Data chunk to disk
    return img_xds, return_df

def get_calculate_cycle_controls(controller,combined_return_dict, img_xds, is_n_iter_0, iteration_control_params, residual_data_group_name = "residual",):
    from astroviper.processing_functions.imaging.iteration_control import (
        IterationController,
        ReturnDict,
        merge_return_dicts,
    )
    import numpy as np
    residual_data_group = img_xds.attrs["data_groups"][residual_data_group_name]
    if is_n_iter_0:
        peak_res = np.max(np.abs(img_xds[residual_data_group["sky"]].values))
        temp_rd = ReturnDict()
        temp_rd.add(
            {
                "peakres": peak_res,
                "peakres_nomask": peak_res,
                "masksum": img_xds.sizes["l"] * img_xds.sizes["m"],
                "iter_done": 0,
                "max_psf_sidelobe": iteration_control_params["maxpsffraction"],
                "loop_gain": iteration_control_params["gain"],
            },
            time=0,
            pol=0,
            chan=0,
        )
        cycle_niter, cyclethresh = controller.calculate_cycle_controls(temp_rd)
    else:
        cycle_niter, cyclethresh = controller.calculate_cycle_controls(
            combined_return_dict
        )
        
    return cycle_niter, cyclethresh

