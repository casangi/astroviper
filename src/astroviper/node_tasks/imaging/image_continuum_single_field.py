from astroviper.utils.param_docs import shares_param_docs


def _write_task_kill_switch_log(
    timing_df, task_total_time, threshold, image_store, task_id, hostname
):
    """Dump an overrunning node task's full timing breakdown to an error log file.

    Written next to the image store (its parent directory), best-effort. Returns
    the path written (or a placeholder string if writing failed). Used by the
    ``task_time_kill_switch_seconds`` watchdog before it raises to abort the run.

    ``image_store`` is retained in the node-task API for compatibility and as a
    convenient location for watchdog logs, even though this continuum map task
    does not write its image products to disk.
    """
    import os
    import time as _time

    try:
        parent = os.path.dirname(os.path.abspath(image_store)) or "."
        stamp = _time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        path = os.path.join(
            parent, f"KILL_SWITCH_task_{task_id}_{hostname}_{pid}_{stamp}.log"
        )
        lines = [
            "TASK TIME KILL SWITCH TRIPPED",
            f"task_id: {task_id}",
            f"hostname: {hostname}",
            f"pid: {pid}",
            f"task wall time: {task_total_time:.2f} s   (threshold: {threshold} s)",
            f"image_store: {image_store}",
            "",
            "per-task timing breakdown:",
            timing_df.to_string(index=False),
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(str(line) for line in lines) + "\n")
        return path
    except Exception as exc:
        # Never allow logging failure to mask the kill-switch exception.
        return f"(failed to write kill-switch log: {exc!r})"


def _remap_deconvolve_dict_to_global_channels(
    combined_deconvolve_dict,
    data_selection,
):
    """Shift chunk-local deconvolution metadata onto global channel numbers.

    The deconvolution machinery labels channels ``0..N-1`` within the local
    frequency chunk. The global channel offset is the start of the frequency
    slice in ``data_selection``.

    This helper is retained because the continuum processing function still
    receives the deconvolver and iteration-control configuration. In the first
    implementation, the caller may enforce ``niter=0``, in which case the
    returned deconvolution dictionary will normally be empty.
    """
    from astroviper.processing_functions.imaging.utils.return_dict import (
        Key,
        ReturnDict,
    )

    chan_offset = 0

    for selection in (data_selection or {}).values():
        frequency_selection = (
            selection.get("frequency") if isinstance(selection, dict) else None
        )

        if (
            isinstance(frequency_selection, slice)
            and frequency_selection.start is not None
        ):
            chan_offset = int(frequency_selection.start)
            break

    if chan_offset == 0:
        return combined_deconvolve_dict

    remapped = ReturnDict()

    for key, value in combined_deconvolve_dict.data.items():
        remapped.data[
            Key(
                time=key.time,
                pol=key.pol,
                chan=key.chan + chan_offset,
            )
        ] = value

    return remapped


@shares_param_docs
def image_continuum_single_field(
    image_params,
    imaging_weights_params,
    iteration_control_params,
    task_coords,
    data_selection,
    image_store,
    input_data_store,
    processing_set_data_group_name="corrected",
    deconvolver="hogbom",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    restore=False,
    memory_mode="in_memory",
    skunk_works=False,
    data_group=None,
    task_id=0,
    input_data=None,
    graph_mode=True,
    output_shard_channels=None,
    task_time_kill_switch_seconds=None,
):
    """Compute one frequency chunk's continuum Taylor products in memory.

    This is the continuum-map counterpart of the existing cube node task while
    intentionally retaining the public node-task name and signature.

    The task:

    1. constructs the empty per-chunk image in the instrument-correlation basis;
    2. loads or receives this frequency chunk's visibility data;
    3. calls
       :func:`astroviper.processing_functions.imaging.image_continuum_single_field`;
    4. returns the resulting :class:`xarray.Dataset` and metadata in memory.

    Unlike the cube node task, this function does **not** write a frequency
    slice into the output image store. The returned dataset is intended to hold
    the chunk-local continuum products that will be numerically combined by the
    GraphViper reduce stage. For ``nterms=2``, the processing function is
    expected to place the local residual Taylor terms and PSF/Hessian Taylor
    terms in this dataset, for example:

    - residual Taylor terms ``R_0`` and ``R_1``;
    - PSF/Hessian Taylor terms ``H_0``, ``H_1`` and ``H_2``;
    - any normalization or sum-of-weight products required by the reducer.

    The deconvolver, iteration-control, and restoration arguments are preserved
    and forwarded to the processing function. The caller may initially enforce
    ``iteration_control_params["niter"] == 0`` so that only dirty/residual and
    PSF Taylor products are generated.

    Parameters
    ----------
    image_params : dict
        Image geometry and output coordinates. In addition to the cube-imaging
        parameters, the continuum processing function may require entries such
        as ``reference_frequency`` and ``nterms``.
    imaging_weights_params : dict
        Weighting scheme configuration.
    iteration_control_params : dict
        Major/minor-cycle iteration controls. These are forwarded unchanged.
    task_coords : dict
        Per-chunk coordinate mapping. ``task_coords["frequency"]["data"]``
        supplies this chunk's frequency axis.
    data_selection : dict
        Per-chunk visibility selection injected by GraphViper.
    image_store : str
        Retained for API compatibility and watchdog log placement. No image
        products are written by this node task.
    input_data_store : str
        Processing-set store used when ``input_data`` is not supplied.
    processing_set_data_group_name : str, optional
        Processing-set data group to image.
    deconvolver : str, optional
        Deconvolver forwarded to the continuum processing function.
    instrument_polarization_basis : str, optional
        Correlation basis used during gridding.
    single_precision_image : bool, optional
        Whether image-domain arrays use single precision.
    processing_function_threads : int, optional
        Threads supplied to processing kernels.
    fft_backend : str, optional
        FFT backend used by image normalization.
    image_data_variables_keep : list of str, optional
        Logical image products to retain in the returned dataset.
    restore : bool, optional
        Forwarded unchanged to the continuum processing function.
    memory_mode : str, optional
        Only ``"in_memory"`` is currently supported.
    skunk_works : bool, optional
        Use the experimental direct Zarr loading path.
    data_group : dict, optional
        Resolved role-to-variable mapping for the skunk-works loader.
    task_id : int, optional
        Identifier of the frequency chunk.
    input_data : dict, optional
        Preloaded visibility data for this chunk.
    graph_mode : bool, optional
        Retained for signature compatibility. No writing occurs in either mode.
    output_shard_channels : int, optional
        Retained for signature compatibility. Unused because no output shards
        are written.
    task_time_kill_switch_seconds : float, optional
        Abort the distributed calculation if this task exceeds the threshold.

    Returns
    -------
    dict
        Dictionary containing:

        ``"image"``
            The chunk-local continuum :class:`xarray.Dataset`, including the
            Taylor residual and PSF/Hessian products.

        ``"timing_node_tasks"``
            One-row timing dataframe.

        ``"deconvolution"``
            Deconvolution metadata returned by the processing function. This is
            retained even when the initial implementation runs with ``niter=0``.
    """
    import socket
    import time

    import toolviper.utils.logger as logger
    from toolviper.utils.memory_management import get_rss_gb
    from xradio.image import make_empty_sky_image

    import astroviper.processing_functions as pf

    task_start = time.time()

    logger.debug(
        "Memory usage at start of image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )

    assert (
        memory_mode == "in_memory"
    ), "Currently only memory_mode='in_memory' is implemented."

    if image_data_variables_keep is None:
        image_data_variables_keep = [
            "sky_residual",
            "point_spread_function",
            "primary_beam",
        ]

    # Build the empty image in the correlation basis expected by the gridder.
    correlation_pol_coords = {
        "linear": ["XX", "YY"],
        "circular": ["RR", "LL"],
    }[instrument_polarization_basis]

    start = time.time()

    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=task_coords["frequency"]["data"],
        pol_coords=correlation_pol_coords,
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )

    T_make_empty_image = time.time() - start

    # Load this map task's visibility partition.
    start = time.time()

    if input_data is not None:
        # The optional GraphViper data-loading layer has already performed the
        # task-level sub-selection.
        ps_xdt = input_data

    elif skunk_works:
        from astroviper.node_tasks.imaging.utils import load_processing_set_skunk_works

        ps_xdt = load_processing_set_skunk_works(
            input_data_store,
            sel_parms=data_selection,
            data_group=data_group,
            processing_set_data_group_name=processing_set_data_group_name,
            frequency_coords=task_coords["frequency"]["data"],
            instrument_polarization_basis=instrument_polarization_basis,
            processing_function_threads=processing_function_threads,
        )

    else:
        from xradio.measurement_set.load_processing_set import load_processing_set

        ps_xdt = load_processing_set(
            input_data_store,
            sel_parms=data_selection,
            data_group_name=processing_set_data_group_name,
            load_sub_datasets=False,
        )

    T_load = time.time() - start

    # Run the continuum processing function. This call intentionally retains
    # the deconvolver and iteration-control interface; the caller can enforce
    # niter=0 while the first implementation focuses only on the major-cycle
    # Taylor products.
    (
        img_xds,
        timing_df,
        combined_deconvolve_dict,
    ) = pf.imaging.image_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        imaging_weights_params,
        iteration_control_params,
        processing_set_data_group_name=processing_set_data_group_name,
        deconvolver=deconvolver,
        instrument_polarization_basis=instrument_polarization_basis,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
        restore=restore,
        task_id=task_id,
    )

    # Preserve the cube implementation's global channel remapping for the
    # deconvolution metadata. This is harmless for an empty ReturnDict.
    combined_deconvolve_dict = _remap_deconvolve_dict_to_global_channels(
        combined_deconvolve_dict,
        data_selection,
    )

    # No output image writing is performed here. The Taylor products stay in
    # memory and flow directly into the GraphViper reduction stage.
    T_write = 0.0

    # Visibility data are no longer needed once the local Taylor products have
    # been constructed. Keep img_xds alive because it is the map-task output.
    ps_xdt = None

    logger.debug(
        "Memory usage after image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )

    task_total_time = time.time() - task_start

    timing_df["T_make_empty_image"] = T_make_empty_image
    timing_df["T_load"] = T_load

    # Keep this column for compatibility with the existing timing schema while
    # making it explicit that no writing occurred.
    timing_df["T_write"] = T_write
    timing_df["T_image_cube_task"] = task_total_time

    hostname = socket.gethostname()
    timing_df["hostname"] = hostname

    if (
        task_time_kill_switch_seconds is not None
        and task_total_time > task_time_kill_switch_seconds
    ):
        log_path = _write_task_kill_switch_log(
            timing_df,
            task_total_time,
            task_time_kill_switch_seconds,
            image_store,
            task_id,
            hostname,
        )

        message = (
            f"task_time_kill_switch tripped: task {task_id} on {hostname} took "
            f"{task_total_time:.1f}s > "
            f"{task_time_kill_switch_seconds}s threshold. "
            f"Aborting the run. Timing log written to: {log_path}"
        )

        logger.error(message)
        raise RuntimeError(message)

    from astroviper.processing_functions.imaging.utils import (
        IMAGING_TIMING_PHASES,
        IMAGING_TIMING_TOTAL_KEY,
    )
    from astroviper.utils.timing import print_timing_summary

    print_timing_summary(
        timing_df,
        IMAGING_TIMING_PHASES,
        total_key=IMAGING_TIMING_TOTAL_KEY,
        printer=logger.debug,
    )

    return {
        "image": img_xds,
        "timing_node_tasks": timing_df,
        "deconvolution": combined_deconvolve_dict,
    }


@shares_param_docs
def model_update_continuum_single_field(
    input_data,
    input_params,
):
    """Run one continuum model-update node entirely in memory.

        This node is intended for the GraphViper append stage. It receives the
        globally reduced continuum Taylor products from the preceding reduce stage,
        calculates the minor-cycle controls, runs the continuum model-update
        processing function, updates convergence information, and returns the
        modified image dataset in memory.

        No image or processing-set data are read from or written to disk.

        reduce output dictionary
        │
        ├── input_data["image"]
        ├── input_data["deconvolution"]
        └── input_data["timing_node_tasks"]
                │
                ▼
    model_update_continuum_single_field
                │
                ├── calculates cycle controls
                ├── calls model_update_cycle_mtmfs_single_field
                ├── updates convergence
                └── changes SKY_MODEL[taylor_term=0]
                │
                ▼
    updated result dictionary remains in memory

        Parameters
        ----------
        input_data : dict
            Output of the preceding GraphViper reduce stage. It must contain

            ``input_data["image"]``
                Globally reduced continuum image dataset.

            It may additionally contain

            ``input_data["deconvolution"]``
                Previously accumulated deconvolution statistics.

            ``input_data["timing_node_tasks"]``
                Timing information from the map/reduce stages.

        input_params : dict
            Append-node configuration. Expected entries are

            ``iteration_control_params``
                Global CLEAN iteration-control parameters.

            Optional entries are

            ``deconvolver``
                Deconvolver name. Default is ``"hogbom"``.

            ``processing_function_threads``
                Number of threads passed to the model-update processing function.

            ``is_n_iter_0``
                Whether this is the first model update. Default is ``True``.

            ``controller``
                Existing :class:`IterationController`. If omitted, a new controller
                is constructed from ``iteration_control_params``.

            ``image_data_group_in_name``
                Residual data-group name. Default is ``"residual"``.

            ``image_data_group_out_name``
                Model data-group name. Default is ``"model"``.

        Returns
        -------
        dict
            Dictionary containing the updated in-memory image, model-update timing,
            deconvolution statistics, controller, and convergence state.
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        model_update_cycle_mtmfs_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        IterationController,
        ReturnDict,
        get_calculate_cycle_controls,
        merge_return_dicts,
    )

    node_start = time.time()

    # GraphViper append normally supplies the preceding graph result directly.
    # Accept a singleton list as well, in case the append execution wrapper
    # retains the reduce-style list convention.
    if isinstance(input_data, (list, tuple)):
        if len(input_data) != 1:
            raise ValueError(
                "The continuum model-update append node expects exactly one "
                f"reduced input, but received {len(input_data)}."
            )
        input_data = input_data[0]

    if not isinstance(input_data, dict):
        raise TypeError(
            "input_data must be the dictionary returned by the continuum "
            f"reduce stage, not {type(input_data).__name__}."
        )

    if "image" not in input_data:
        raise KeyError(
            "The continuum reduce result does not contain the required "
            "'image' entry."
        )

    if "iteration_control_params" not in input_params:
        raise KeyError("input_params must contain 'iteration_control_params'.")

    img_xds = input_data["image"]
    iteration_control_params = input_params["iteration_control_params"]

    deconvolver = input_params.get("deconvolver", "hogbom")
    processing_function_threads = input_params.get(
        "processing_function_threads",
        1,
    )
    is_n_iter_0 = input_params.get("is_n_iter_0", True)

    image_data_group_in_name = input_params.get(
        "image_data_group_in_name",
        "residual",
    )
    image_data_group_out_name = input_params.get(
        "image_data_group_out_name",
        "model",
    )

    # -------------------------------------------------------------
    # Obtain or temporarily construct the iteration controller.
    #
    # Once controller ownership is moved to the application level, the
    # existing controller should be supplied in input_params.
    # -------------------------------------------------------------
    controller = input_params.get("controller")

    if controller is None:
        controller = IterationController(
            niter=iteration_control_params["niter"],
            nmajor=iteration_control_params["nmajor"],
            threshold=iteration_control_params["threshold"],
            gain=iteration_control_params["gain"],
            cyclefactor=iteration_control_params["cyclefactor"],
            minpsffraction=iteration_control_params["minpsffraction"],
            maxpsffraction=iteration_control_params["maxpsffraction"],
            cycleniter=iteration_control_params["cycleniter"],
        )

    combined_deconvolve_dict = input_data.get("deconvolution")

    if combined_deconvolve_dict is None:
        combined_deconvolve_dict = ReturnDict()

    timing = {
        "T_iteration_control": 0.0,
        "T_model_update_cycle": 0.0,
        "T_convergence": 0.0,
    }

    # -------------------------------------------------------------
    # Calculate the controls for this minor cycle.
    #
    # The temporary Taylor-0 Högbom implementation has one effective
    # frequency plane. Independently controlled planes are therefore
    # time x 1 x polarization.
    # -------------------------------------------------------------
    start = time.time()

    controller.ensure_planes(
        img_xds.sizes["time"],
        1,
        img_xds.sizes["polarization"],
    )

    (
        cycle_niter,
        cyclethreshold,
        cyclethreshold_per_plane,
    ) = get_calculate_cycle_controls(
        controller,
        combined_deconvolve_dict,
        img_xds,
        is_n_iter_0,
        iteration_control_params=iteration_control_params,
    )

    timing["T_iteration_control"] = time.time() - start

    deconvolve_params = {
        **iteration_control_params,
        "cycleniter": cycle_niter,
        "cyclethreshold": cyclethreshold,
        "niter_per_plane": controller.niter.clip(max=cycle_niter),
        "cyclethreshold_per_plane": cyclethreshold_per_plane,
    }

    # -------------------------------------------------------------
    # Run the model-update processing function.
    #
    # img_xds is already in memory and is modified in place.
    # -------------------------------------------------------------
    start = time.time()

    (deconvolve_dict, model_update_return_df,) = model_update_cycle_mtmfs_single_field(
        img_xds,
        deconvolver,
        deconvolve_params,
        is_n_iter_0=is_n_iter_0,
        processing_function_threads=processing_function_threads,
        image_data_group_in_name=image_data_group_in_name,
        image_data_group_out_name=image_data_group_out_name,
    )

    timing["T_model_update_cycle"] = time.time() - start

    # -------------------------------------------------------------
    # Update convergence bookkeeping.
    # -------------------------------------------------------------
    start = time.time()

    controller.update_counts(deconvolve_dict)

    stopcode, stopdesc = controller.check_convergence(deconvolve_dict)

    combined_deconvolve_dict = merge_return_dicts(
        [
            combined_deconvolve_dict,
            deconvolve_dict,
        ]
    )

    timing["T_convergence"] = time.time() - start
    timing["T_model_update_node_task"] = time.time() - node_start

    logger.debug(
        "Continuum model update finished with "
        f"major stop code {stopcode.major}: {stopdesc}"
    )

    node_timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    if model_update_return_df is not None:
        node_timing_df = pd.concat(
            [
                node_timing_df.reset_index(drop=True),
                model_update_return_df.reset_index(drop=True),
            ],
            axis=1,
        )

    # Preserve the map/reduce timing separately rather than mixing rows from
    # different types of node task.
    return {
        "image": img_xds,
        "timing_node_tasks": input_data.get("timing_node_tasks"),
        "timing_model_update": node_timing_df,
        "deconvolution": combined_deconvolve_dict,
        "controller": controller,
        "stopcode": stopcode,
        "stopdesc": stopdesc,
        "is_n_iter_0": False,
    }
