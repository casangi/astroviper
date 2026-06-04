

def single_field_imaging_preperation(input_params, ps_xdt, img_xds):
    """Set up the iteration controller and run the once-per-chunk imaging setup.

    Everything that is done a single time per chunk happens here, before the
    major-cycle loop in :func:`image_cube_single_field`:

    * the :class:`IterationController` and the (empty) combined return dict, and
    * the imaging weights, the point spread function and the primary beam (via
      :func:`single_field_imaging_setup`).

    The dirty image and the first model update are deliberately NOT done here --
    they are the first iteration of the loop in :func:`image_cube_single_field`.

    Returns
    -------
    controller : IterationController
        Freshly constructed controller (``stopcode.major == 0``).
    img_xds
        Image dataset with the PSF and primary beam, in the Stokes basis.
    return_df
        Timing/metadata dataframe from the setup step.
    combined_deconvolve_dict : ReturnDict
        Empty accumulator for the per-plane convergence stats.
    T_residual_cycle, T_model_update_cycle : float
        Accumulated timers (model-update timer starts at zero).
    """
    import toolviper.utils.logger as logger
    from astroviper.processing_functions.imaging.residual_cycle import single_field_imaging_setup
    import time
    from astroviper.processing_functions.imaging.iteration_control import (
        IterationController,
        ReturnDict,
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

    T_residual_cycle = 0.0
    T_model_update_cycle = 0.0

    # Once-only imaging setup: imaging weights, PSF and primary beam. The dirty
    # image and the model update are NOT done here.
    start = time.time()
    img_xds, return_df = single_field_imaging_setup(ps_xdt, img_xds, input_params)
    T_residual_cycle = T_residual_cycle + time.time() - start

    return (
        controller,
        img_xds,
        return_df,
        combined_deconvolve_dict,
        T_residual_cycle,
        T_model_update_cycle,
    )


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

    # All once-only work -- controller setup, imaging weights, PSF and primary
    # beam creation -- happens in the preparation step before the major-cycle
    # loop. The dirty image and the first model update are the first iteration
    # of the loop below.
    (
        controller,
        img_xds,
        return_df,
        combined_deconvolve_dict,
        T_residual_cycle,
        T_model_update_cycle,
    ) = single_field_imaging_preperation(input_params, ps_xdt, img_xds)

    is_n_iter_0 = True
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
