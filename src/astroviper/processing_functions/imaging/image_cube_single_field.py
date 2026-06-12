def imaging_preparation_single_field(input_params, ps_xdt, img_xds):
    """Run the once-per-chunk imaging setup before the major-cycle loop.

    Everything that is done a single time per chunk happens here:

    * construction of the :class:`IterationController` and the (empty) combined
      return dict, and
    * the imaging weights, the point spread function and the primary beam (via
      :func:`~astroviper.processing_functions.imaging.residual_cycle.imaging_setup_single_field`).

    The dirty image and the first model update are deliberately NOT done here --
    they are the first iteration of the loop in :func:`image_cube_single_field`.

    Parameters
    ----------
    input_params : dict
        Imaging parameters (see :func:`image_cube_single_field`).
    ps_xdt : xarray.DataTree
        Visibility data for this chunk.
    img_xds : xarray.Dataset
        Empty image dataset for this chunk.

    Returns
    -------
    controller : IterationController
        Freshly constructed controller (``stopcode.major == 0``).
    img_xds : xarray.Dataset
        Image dataset with the PSF and primary beam, in the Stokes basis.
    return_df : pandas.DataFrame
        One-row timing frame from the setup step.
    combined_deconvolve_dict : ReturnDict
        Empty accumulator for the per-plane convergence statistics.
    T_setup : float
        Wall-clock time of the setup step (seconds).
    """
    import time
    import toolviper.utils.logger as logger
    from astroviper.processing_functions.imaging.residual_cycle import (
        imaging_setup_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
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

    # Once-only imaging setup: imaging weights, PSF and primary beam. The dirty
    # image and the model update are NOT done here.
    start = time.time()
    img_xds, return_df = imaging_setup_single_field(ps_xdt, img_xds, input_params)
    T_setup = time.time() - start

    return controller, img_xds, return_df, combined_deconvolve_dict, T_setup


def image_cube_single_field(input_params, ps_xdt, img_xds):
    """Run the major/minor cycle CLEAN loop for one single-field image chunk.

    Performs the once-per-chunk setup (imaging weights, PSF, primary beam), then
    iterates residual (major) and model-update (minor) cycles under the
    :class:`IterationController` until convergence, finishing with a last
    residual cycle that produces the final residual image.  Every processing
    function is timed; the totals are returned as a one-row timing frame.

    Parameters
    ----------
    input_params : dict
        Imaging parameters injected by the node task (image params, iteration
        control params, deconvolver, precision, FFT backend, thread count, ...).
    ps_xdt : xarray.DataTree
        Visibility data for this chunk.
    img_xds : xarray.Dataset
        Empty image dataset for this chunk.

    Returns
    -------
    img_xds : xarray.Dataset
        Image dataset with the final residual image, sky model, PSF and primary
        beam (in the Stokes basis).
    timing_df : pandas.DataFrame
        One-row frame with a ``T_*`` column per processing function plus
        ``task_id``, ``n_channels`` and ``n_major_cycles``.
    combined_deconvolve_dict : ReturnDict
        Per-plane convergence statistics for this chunk.  Channel labels are
        chunk-local (0-based); the node task remaps them to global channel
        numbers before the reduce.
    """
    import time
    import toolviper.utils.logger as logger
    import pandas as pd
    from astroviper.processing_functions.imaging.residual_cycle import (
        residual_cycle_cube_single_field,
    )
    from astroviper.processing_functions.imaging.model_update_cycle import (
        model_update_cycle_cube_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        ReturnDict,
        merge_return_dicts,
        accumulate_timing,
        get_calculate_cycle_controls,
    )

    # All once-only work -- controller setup, imaging weights, PSF and primary
    # beam creation -- happens in the preparation step before the major-cycle
    # loop. The dirty image and the first model update are the first iteration
    # of the loop below.
    (
        controller,
        img_xds,
        setup_return_df,
        combined_deconvolve_dict,
        T_setup,
    ) = imaging_preparation_single_field(input_params, ps_xdt, img_xds)

    timing = {"T_setup": T_setup}
    accumulate_timing(timing, setup_return_df)

    T_residual_cycle = 0.0
    T_model_update_cycle = 0.0

    is_n_iter_0 = True
    n_major_cycles = 0
    while controller.stopcode.major == 0:
        n_major_cycles += 1
        start = time.time()
        img_xds, residual_return_df = residual_cycle_cube_single_field(
            ps_xdt, img_xds, input_params, is_n_iter_0=is_n_iter_0
        )
        T_residual_cycle += time.time() - start
        accumulate_timing(timing, residual_return_df)

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
            cycle_niter, cyclethresh, threshold_per_plane = (
                get_calculate_cycle_controls(
                    controller,
                    combined_deconvolve_dict,
                    img_xds,
                    is_n_iter_0,
                    iteration_control_params=input_params["iteration_control_params"],
                )
            )

            input_params["iteration_control_params"]["cycleniter"] = cycle_niter
            input_params["iteration_control_params"]["threshold"] = cyclethresh
            # Per-plane iteration control handed to the deconvolver: remaining
            # iterations per plane and the per-plane cyclethreshold.
            input_params["iteration_control_params"][
                "niter_per_plane"
            ] = controller.niter
            input_params["iteration_control_params"][
                "threshold_per_plane"
            ] = threshold_per_plane

            start = time.time()
            deconvolve_dict, model_update_return_df = (
                model_update_cycle_cube_single_field(
                    img_xds,
                    input_params,
                    is_n_iter_0=is_n_iter_0,
                    num_threads=input_params["processing_function_threads"],
                    image_data_group_in_name="residual",
                    image_data_group_out_name="model",
                )
            )
            T_model_update_cycle += time.time() - start
            accumulate_timing(timing, model_update_return_df)
        else:
            deconvolve_dict = ReturnDict()

        is_n_iter_0 = False

        controller.update_counts(deconvolve_dict)

        # check_convergence stamps the stop code into deconvolve_dict, so run
        # it before the merge to carry that stop code into the combined dict.
        stopcode, stopdesc = controller.check_convergence(deconvolve_dict)
        combined_deconvolve_dict = merge_return_dicts(
            [combined_deconvolve_dict, deconvolve_dict]
        )

        if stopcode.major != 0:
            logger.info(f"  *** CONVERGED: {stopdesc} ***")
            break

    # Last residual cycle to compute the final residual image after the last
    # model-update cycle.
    if input_params["iteration_control_params"]["niter"] > 0:
        start = time.time()
        img_xds, residual_return_df = residual_cycle_cube_single_field(
            ps_xdt, img_xds, input_params, is_n_iter_0=is_n_iter_0
        )
        T_residual_cycle += time.time() - start
        accumulate_timing(timing, residual_return_df)

    timing["T_residual_cycle"] = T_residual_cycle
    timing["T_model_update_cycle"] = T_model_update_cycle
    timing["task_id"] = input_params["task_id"]
    timing["n_channels"] = len(input_params["task_coords"]["frequency"]["data"])
    timing["n_major_cycles"] = n_major_cycles

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df, combined_deconvolve_dict
