

def image_cube_single_field(input_params, ps_xdt, img_xds):
    import toolviper.utils.logger as logger
    from astroviper.processing_functions.imaging.residual_cycle import residual_cycle_cube_single_field
    from astroviper.processing_functions.imaging.model_update_cycle import model_update_cycle_cube_single_field
    import time
    from astroviper.processing_functions.imaging.iteration_control import (
        IterationController,
        ReturnDict,
        merge_return_dicts,
        print_deconvolve_dict,
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
    combined_deconvolve_dict = ReturnDict()

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
            # Size the controller's per-plane state to this cube so iteration
            # control (niter and threshold) is tracked independently for every
            # (time, frequency, polarization) plane before the deconvolver runs.
            controller.ensure_planes(
                img_xds.sizes["time"],
                img_xds.sizes["frequency"],
                img_xds.sizes["polarization"],
            )
            cycle_niter, cyclethresh, threshold_per_plane = get_calculate_cycle_controls(controller, combined_deconvolve_dict, img_xds, is_n_iter_0, iteration_control_params=input_params["iteration_control_params"])

            input_params["iteration_control_params"]["cycleniter"] = cycle_niter
            input_params["iteration_control_params"]["threshold"] = cyclethresh
            # Per-plane iteration control handed to the deconvolver: remaining
            # iterations per plane and the per-plane cyclethreshold.
            input_params["iteration_control_params"]["niter_per_plane"] = controller.niter
            input_params["iteration_control_params"]["threshold_per_plane"] = threshold_per_plane
            start = time.time()
            deconvolve_dict = model_update_cycle_cube_single_field(img_xds, input_params, is_n_iter_0=is_n_iter_0, num_threads=input_params["processing_function_threads"], img_data_group_in_name = "residual", img_data_group_out_name = "model")
            T_model_update_cycle = T_model_update_cycle + time.time() - start
        else:
            deconvolve_dict = ReturnDict()
            
        is_n_iter_0 = False
        
        controller.update_counts(deconvolve_dict)

        # check_convergence stamps the stop code into deconvolve_dict, so run
        # it before the merge to carry that stop code into the combined dict.
        stopcode, stopdesc = controller.check_convergence(deconvolve_dict)
        combined_deconvolve_dict = merge_return_dicts([combined_deconvolve_dict, deconvolve_dict])

        if stopcode.major != 0:
            logger.info(f"  *** CONVERGED: {stopdesc} ***")
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
    
    print("@@@@@@@@@ Combined Deconvolve Dict:")
    print_deconvolve_dict(combined_deconvolve_dict)
    print("***************")

    # #Write Data chunk to disk
    # combined_deconvolve_dict carries per-plane convergence stats for this
    # channel chunk. Its channel labels are chunk-local (0-based); the node
    # task remaps them to global channel numbers before the reduce.
    return img_xds, return_df, combined_deconvolve_dict

def get_calculate_cycle_controls(controller,combined_deconvolve_dict, img_xds, is_n_iter_0, iteration_control_params, residual_data_group_name = "residual",):
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
        rd = temp_rd
    else:
        rd = combined_deconvolve_dict

    cycle_niter, cyclethresh = controller.calculate_cycle_controls(rd)
    # Per-plane cyclethreshold so each (time, chan, pol) plane can use its own
    # threshold (falls back to the global value for planes without data).
    threshold_per_plane = controller.per_plane_cycle_threshold(rd)

    return cycle_niter, cyclethresh, threshold_per_plane

