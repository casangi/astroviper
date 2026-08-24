from astroviper.utils.param_docs import shares_param_docs


def _write_task_kill_switch_log(
    timing_df, task_total_time, threshold, image_store, task_id, hostname
):
    """Dump an overrunning node task's full timing breakdown to an error log file.

    Written next to the image store (its parent directory), best-effort. Returns
    the path written (or a placeholder string if writing failed). Used by the
    ``task_time_kill_switch_seconds`` watchdog before it raises to abort the run.
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
            fh.write("\n".join(str(ln) for ln in lines) + "\n")
        return path
    except Exception as exc:  # never let logging failure mask the kill-switch raise
        return f"(failed to write kill-switch log: {exc!r})"


def _log_task_io_failure(phase, exc, task_id, image_store, data_selection, task_coords):
    """Log a NON-FATAL node-task I/O failure; return its timing-row marker columns.

    A chunk whose data cannot be read or whose result cannot be written (e.g. a
    Lustre client eviction that outlives the reader/writer retry schedules, or a
    corrupt input shard) must not abort the whole multi-node run: the caller
    logs it here, marks the chunk's row in the timing frame with these columns
    (``task_failed_phase`` / ``task_error`` / ``failed_channel_start``), and the
    run continues with this chunk's channels left at the image store's fill
    value. Failures stay queryable per run from the saved node-task frame.
    """
    import socket

    import toolviper.utils.logger as logger

    hostname = socket.gethostname()
    chan_start = 0
    for sel in (data_selection or {}).values():
        freq_sel = sel.get("frequency") if isinstance(sel, dict) else None
        if isinstance(freq_sel, slice) and freq_sel.start is not None:
            chan_start = int(freq_sel.start)
            break
    n_channels = len(task_coords["frequency"]["data"])
    logger.error(
        f"node task {task_id} on {hostname}: {phase} FAILED for channels "
        f"[{chan_start}, {chan_start + n_channels}) of {image_store}; skipping "
        f"this chunk and continuing the run. Error: {exc!r}"
    )
    return {
        "hostname": hostname,
        "task_id": task_id,
        "n_channels": n_channels,
        "task_failed_phase": phase,
        "task_error": repr(exc)[:500],
        "failed_channel_start": chan_start,
    }


def _remap_deconvolve_dict_to_global_channels(combined_deconvolve_dict, data_selection):
    """Shift a chunk-local deconvolve ReturnDict onto global channel numbers.

    The per-chunk ``image_cube_single_field`` labels channels ``0..N-1`` within
    the chunk. The global channel offset for this chunk is the start of the
    ``frequency`` slice in ``data_selection`` (frequency and channel are the
    same axis), e.g. ``{'ms_name': {'frequency': slice(2, 4)}}`` -> offset 2.

    Returns a new ReturnDict whose ``Key.chan`` values are global channel
    numbers. A no-op returning the input unchanged when the offset is 0 (e.g. a
    single chunk starting at channel 0) or no frequency slice is present.
    """
    from astroviper.processing_functions.imaging.utils.return_dict import (
        Key,
        ReturnDict,
    )

    chan_offset = 0
    for sel in (data_selection or {}).values():
        freq_sel = sel.get("frequency") if isinstance(sel, dict) else None
        if isinstance(freq_sel, slice) and freq_sel.start is not None:
            chan_offset = int(freq_sel.start)
            break

    if chan_offset == 0:
        return combined_deconvolve_dict

    remapped = ReturnDict()
    for key, value in combined_deconvolve_dict.data.items():
        remapped.data[Key(time=key.time, pol=key.pol, chan=key.chan + chan_offset)] = (
            value
        )
    return remapped


@shares_param_docs
def image_cube_single_field(
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
    output_image_format="zarr",
    task_time_kill_switch_seconds=None,
):
    """Image one frequency chunk of a single-field cube and write it to disk.

    Thin node task: builds the empty per-chunk
    image in the correlation (instrument) polarization basis, loads (or receives)
    this chunk's visibilities, runs the science
    :func:`~astroviper.processing_functions.imaging.image_cube_single_field.image_cube_single_field`,
    writes the result slice to the Zarr image store, and returns the timing and
    deconvolution metadata.

    This function has a fully spelled-out signature so it can be called directly
    (standalone) outside of a graph.  When driven by
    :func:`graphviper.graph_tools.map.map`, graphviper adapts it automatically
    (via :func:`graphviper.graph_tools.map.make_graph_node_task`), expanding the
    single ``input_params`` dict it passes into these keyword arguments.

    Parameters
    ----------
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor.
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    iteration_control_params : dict
        CLEAN minor/major-cycle iteration controls, matching the meaning of the
        corresponding CASA ``tclean`` parameters. Iteration control is performed
        **independently per** ``(time, frequency, polarization)`` **plane**: each
        plane carries its own iteration budget and stopping thresholds, and the
        major-cycle loop continues until *every* selected plane has stopped --
        the one deliberate difference from CASA, whose ``niter`` budget is global
        across the image. Keys:

        - ``niter`` : Maximum number of minor-cycle CLEAN iterations (flux
          components) per plane, summed over all major cycles. A plane stops once
          it has spent this budget; ``niter=0`` makes only the dirty image (no
          deconvolution).
        - ``nmajor`` : Maximum number of deconvolving major cycles (each a
          residual update followed by a minor cycle). ``nmajor=N`` performs ``N``
          deconvolutions -- the dirty image is computed inside the first such
          cycle, matching CASA's ``nmajor`` -- and ``nmajor=-1`` removes the
          major-cycle limit. Shared across planes (not tracked per plane).
        - ``threshold`` : Absolute stopping threshold, given as a float in Jy. A
          plane stops when its peak residual inside the clean mask falls to or
          below ``threshold``; the value is also a hard floor on the
          per-minor-cycle ``cyclethreshold`` (below). ``threshold=0`` disables
          the absolute stop.
        - ``primary_beam_limit`` : Primary-beam mask cutoff as a fraction of the
          peak primary beam, in ``[0, 1]`` (the analogue of CASA's ``pblimit`` /
          ``pbmask``). Pixels where the primary beam is below this fraction are
          excluded from cleaning. A masking cutoff, distinct from ``threshold``.
        - ``gain`` : CLEAN loop gain -- the fraction of the selected peak flux
          subtracted from the residual image each minor iteration
          (``0 < gain <= 1``).
        - ``cyclefactor`` : Scaling applied to the brightest PSF sidelobe level
          when setting the minor-cycle stopping depth (see ``cyclethreshold``
          below). Larger values trigger the next major cycle sooner; smaller
          values clean deeper before each residual update.
        - ``cycleniter`` : Maximum number of minor-cycle iterations a plane may
          run before a major cycle is triggered. ``cycleniter=-1`` lets the
          adaptive ``cyclethreshold`` govern the depth instead; otherwise the
          count is clamped to never exceed the plane's remaining ``niter``.
        - ``minpsffraction`` : Lower clamp on the PSF fraction used to set the
          minor-cycle threshold ``cyclethreshold = clamp(max_psf_sidelobe *
          cyclefactor, minpsffraction, maxpsffraction) * peak_residual`` (then
          floored at ``threshold``). Raising it limits how deep a single minor
          cycle cleans.
        - ``maxpsffraction`` : Upper clamp on that same PSF fraction; it
          guarantees a minimum amount of cleaning per minor cycle even when the
          PSF sidelobe level is high.
    task_coords : dict
        Per-chunk coordinate mapping; ``task_coords[<parallel dim>]`` supplies
        this chunk's parallel coordinate values (``"data"``) and its
        ``"slice"`` into the full output array (for cube imaging the
        parallel dim is ``frequency``).
    data_selection : dict
        Per-chunk ``{ms_name: {dim: slice}}`` selection injected by graphviper;
        used to load this chunk's data and to remap chunk-local channel numbers
        to global ones.
    image_store : str
        Path/URL of the on-disk Zarr image cube.
    input_data_store : str
        Path/URL of the processing-set Zarr store to load this chunk's
        visibilities from (used only when ``input_data`` is ``None``).
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    deconvolver : str, optional
        Deconvolution algorithm for the minor cycle. One of ``"hogbom"`` (C++, threaded across planes), ``"hogbom_many_threads"``
        (C++, threaded across *and* within planes -- faster when there are
        few planes, e.g. single-channel imaging) or ``"asp"``.
    instrument_polarization_basis : str, optional
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` (``XX``/``YY``) or ``"circular"`` (``RR``/``LL``). The
        output image is always produced in the Stokes basis.
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the minor
        cycle runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    restore : bool, optional
        If ``True`` produce a restored image after deconvolution: the model
        convolved with the clean beam (the Gaussian fit to the PSF) plus the
        residual, written to the ``sky_restored`` (``SKY_RESTORED``) variable.
    memory_mode : str, optional
        Only ``"in_memory"`` is implemented.  Default ``"in_memory"``.
    skunk_works : bool, optional
        If ``True`` use the experimental performance I/O path: load only the
        data group's data variables straight from the Zarr chunk blobs with
        :func:`~astroviper.node_tasks.imaging.utils.load_processing_set_skunk_works`
        (reconstructing coordinates from the inputs) and write each result chunk
        with
        :func:`~astroviper.node_tasks.imaging.utils.write_result_chunk_to_disk_using_zarr_skunk_works`.
        Both the skunk-works load and write spread their per-array / per-variable
        I/O and (de)compression concurrently across ``processing_function_threads``
        threads.  Requires ``data_group``.  Default ``False`` (production I/O).
    data_group : dict, optional
        Resolved role->variable mapping for ``processing_set_data_group_name``
        (e.g. ``{"correlated_data": "VISIBILITY", "uvw": "UVW", ...}``), supplied
        by the distributed graph; only used when ``skunk_works`` is ``True``.
    task_id : int, optional
        Identifier of the parallel chunk being processed.
    input_data : dict, optional
        Pre-loaded data for this chunk (supplied by the data-loading layer); when
        ``None`` (default) the data is loaded from ``input_data_store``.
    graph_mode : bool, optional
        If ``True`` (default) each kept variable's slice is written into the
        pre-allocated Zarr store with
        :func:`~astroviper.utils.io.write_result_chunk_to_disk_using_zarr`.  If
        ``False`` the whole chunk image is written with ``to_zarr``.
    output_image_format : str, optional
        On-disk format of the image store this task writes into: ``"zarr"``
        (default) or ``"fits"``.  With ``"fits"`` the chunk is ``pwrite``-en
        directly into the pre-created XRADIO-conformant FITS files
        (:func:`~astroviper.node_tasks.imaging.utils.write_result_chunk_to_fits_skunk_works`;
        the driver created them with
        :func:`~astroviper.node_tasks.imaging.utils.create_empty_fits_images`).

    Returns
    -------
    dict
        Single dict with two keys:

        * ``"timing_node_tasks"`` : one-row :class:`pandas.DataFrame` with a
          ``T_*`` column per processing function (load, image build, weights,
          PSF, primary beam, gridding, FFT normalization, degridding,
          deconvolution, write, ...) plus ``task_id``, ``n_channels``,
          ``n_major_cycles`` and the total ``T_image_cube_task``.
        * ``"deconvolution"`` : the per-plane deconvolution
          :class:`~astroviper.processing_functions.imaging.utils.return_dict.ReturnDict`,
          with channels remapped to global channel numbers.
        * ``"image_statistics"`` : ``{image_variable_key: xarray.Dataset}`` of
          NaN-ignoring per-plane statistics of the image-domain variables
          present in memory (``sky_residual``, ``sky_restored``, ``sky_model``,
          ...), computed over ``(l, m)`` *before* the chunk is written. Each
          dataset has dims ``(time, frequency, polarization)`` for this chunk's
          frequencies and one variable per statistic (``mean``, ``median``,
          ``max``, ``min``, ``peak``, ``sum``, ``rms``, ``std``, ``mad_sigma``,
          ``n_pixels`` and their ``_masked`` twins restricted to the clean
          mask, or to ``PRIMARY_BEAM > primary_beam_limit`` when no mask
          exists, e.g. ``niter=0``); see
          :func:`~astroviper.processing_functions.image_analysis.plane_statistics.calculate_plane_statistics`.
          The reduce concatenates the chunks along ``frequency``.
    """
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

    assert memory_mode == "in_memory", (
        "Currently only memory_mode='in_memory' is implemented."
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = [
            "sky_residual",
            "point_spread_function",
            "primary_beam",
        ]

    # Build the empty per-chunk image in the correlation (instrument)
    # polarization basis the gridder works in. The two-feed correlation labels
    # follow ``instrument_polarization_basis`` ("linear" -> XX/YY,
    # "circular" -> RR/LL); the image is transformed to the Stokes output basis
    # (image_params["polarization_coords"]) inside the science function.
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

    start = time.time()
    try:
        if input_data is not None:
            # Data was pre-loaded by the data loading layer (disk-chunk granularity
            # I/O coalescing). The framework has already applied the task-level
            # sub-selection, so use the dict directly.
            ps_xdt = input_data
        elif skunk_works:
            # Experimental performance path: read only this chunk's data-group
            # variables straight from the Zarr chunk blobs and reconstruct the
            # coordinates from the inputs (no datatree/coords/sub-datasets open).
            from astroviper.node_tasks.imaging.utils import (
                load_processing_set_skunk_works,
            )

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
    except Exception as exc:
        # A chunk whose data cannot be read is skipped -- logged + marked in the
        # timing frame -- instead of aborting the whole run (dask/MPI would
        # otherwise tear down every node after this task exhausts its retries).
        import pandas as pd

        from astroviper.processing_functions.imaging.utils.return_dict import ReturnDict

        row = _log_task_io_failure(
            "load", exc, task_id, image_store, data_selection, task_coords
        )
        row.update(
            {
                "T_make_empty_image": T_make_empty_image,
                "T_load": time.time() - start,
                "T_image_cube_task": time.time() - task_start,
                "start_unixtime": task_start,
            }
        )
        return {
            "timing_node_tasks": pd.DataFrame({k: [v] for k, v in row.items()}),
            "deconvolution": ReturnDict(),
            "image_statistics": {},
        }
    T_load = time.time() - start

    img_xds, timing_df, combined_deconvolve_dict = pf.imaging.image_cube_single_field(
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

    # The deconvolve dict's channels are chunk-local (0-based); remap them to
    # global channel numbers so the reduce can merge chunks correctly.
    combined_deconvolve_dict = _remap_deconvolve_dict_to_global_channels(
        combined_deconvolve_dict, data_selection
    )

    # Per-plane (l, m) statistics of every image-domain variable in memory,
    # taken BEFORE the write so they describe exactly what goes to disk (and
    # survive a skipped write). Channels carry their global frequency values,
    # so the reduce can concatenate chunks along ``frequency``.
    start = time.time()
    from astroviper.processing_functions.image_analysis.plane_statistics import (
        calculate_plane_statistics,
    )

    image_statistics = calculate_plane_statistics(
        img_xds,
        # Masked statistics use the clean MASK when present; a niter=0 run has
        # none, so the fallback mask PRIMARY_BEAM > primary_beam_limit (the
        # same valid-sky cutoff the deconvolver would use) applies.
        primary_beam_limit=iteration_control_params.get("primary_beam_limit", 0.2),
    )
    T_image_statistics = time.time() - start

    start = time.time()
    write_exc = None
    try:
        if graph_mode and output_image_format == "fits":
            # FITS performance path: pwrite this chunk's contiguous channel
            # block (and its BEAMS-table rows) directly into the pre-created
            # XRADIO-conformant FITS files -- disjoint byte ranges across
            # tasks, no locking, no file creation.
            from astroviper.node_tasks.imaging.utils import (
                write_result_chunk_to_fits_skunk_works,
            )

            write_result_chunk_to_fits_skunk_works(
                image_store,
                image_data_variables_keep,
                task_coords,
                img_xds,
                processing_function_threads=processing_function_threads,
            )
        elif graph_mode and skunk_works and output_shard_channels:
            # Sharded performance path: write this chunk's inner-chunk blob(s) into
            # shared, pre-created Zarr v3 shard files (far fewer files -> metadata-server
            # relief; the "single parallel file" pattern).
            from astroviper.node_tasks.imaging.utils import (
                write_result_chunk_to_disk_sharded_skunk_works,
            )

            write_result_chunk_to_disk_sharded_skunk_works(
                image_store,
                image_data_variables_keep,
                task_coords,
                img_xds,
                processing_function_threads=processing_function_threads,
            )
        elif graph_mode and skunk_works:
            # Experimental performance path: encode and write only this chunk's
            # blob(s) directly to the pre-created Zarr image store (no open_group).
            from astroviper.node_tasks.imaging.utils import (
                write_result_chunk_to_disk_using_zarr_skunk_works,
            )

            write_result_chunk_to_disk_using_zarr_skunk_works(
                image_store,
                image_data_variables_keep,
                task_coords,
                img_xds,
                processing_function_threads=processing_function_threads,
            )
        elif graph_mode:
            from astroviper.utils.io import write_result_chunk_to_disk_using_zarr

            write_result_chunk_to_disk_using_zarr(
                image_store,
                image_data_variables_keep,
                task_coords,
                img_xds,
            )
        else:
            img_xds.to_zarr(image_store, consolidated=True)
    except Exception as exc:
        # A chunk whose result cannot be written is skipped -- logged + marked
        # below (after the timing columns are folded in) -- instead of aborting
        # the whole run; its channels keep the image store's fill value.
        write_exc = exc
    T_write = time.time() - start

    # Two reference-cycle classes pin this task's gigabytes past `= None`
    # (2026-08-12 findings; each survives until a full gc pass otherwise):
    # 1. DataTree parent<->child links (the loaded chunk's tree), and
    # 2. the xarray cached-accessor cycle on the image dataset
    #    (_cache['xr_img'] <-> xradio ImageXds._xds, created by the
    #    img_xds.xr_img.* calls in the processing functions).
    # Sever both so everything dies by refcount right here. Both helpers are
    # no-ops on the load-layer dict path / cache-less datasets.
    from astroviper.utils.data_tree import clear_cached_accessors, release_data_tree

    release_data_tree(ps_xdt)
    clear_cached_accessors(img_xds)
    img_xds = None
    ps_xdt = None

    logger.debug(
        "Memory usage after image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )

    # Fold the node-task timings (image build, load, write, total) into the
    # per-chunk timing frame produced by the science function.
    task_total_time = time.time() - task_start
    timing_df["T_make_empty_image"] = T_make_empty_image
    timing_df["T_load"] = T_load
    timing_df["T_image_statistics"] = T_image_statistics
    timing_df["T_write"] = T_write
    timing_df["T_image_cube_task"] = task_total_time
    # Wall-clock anchor so the task-stream analysis can place this task on the
    # run's common timeline without needing the resource monitor (whose own
    # anchor, recorded a hair earlier around the whole task, overwrites this
    # column when monitor_resources_seconds is set).
    timing_df["start_unixtime"] = task_start
    # Record which node ran this task so the per-chunk timing frame can be grouped
    # by host (identify stragglers / a slow node in the sweep), plus the exact
    # execution slot (process + thread + Dask worker name) so the task-stream
    # analysis can reconstruct TRUE per-worker lanes -- and place reduce nodes
    # (which record the same identity) on the lane they actually ran on --
    # instead of inferring lanes by interval packing. worker_name is None
    # outside a Dask worker (the MPI ranks).
    import os
    import socket
    import threading

    hostname = socket.gethostname()
    timing_df["hostname"] = hostname
    timing_df["process_pid"] = os.getpid()
    timing_df["thread_native_id"] = threading.get_native_id()
    try:
        from distributed import get_worker

        timing_df["worker_name"] = str(get_worker().name)
    except Exception:
        timing_df["worker_name"] = None

    if write_exc is not None:
        for key, value in _log_task_io_failure(
            "write", write_exc, task_id, image_store, data_selection, task_coords
        ).items():
            timing_df[key] = value

    # Timing kill switch: if this task overran the watchdog threshold, dump its
    # full timing breakdown to an error log and raise -- aborting the whole
    # distributed computation (fail fast on a pathological node/I-O stall rather
    # than hang the run). A task already skipped for a write failure is exempt:
    # its (long) retry schedule must not re-escalate into the abort this
    # skip-and-log path exists to avoid.
    if (
        write_exc is None
        and task_time_kill_switch_seconds is not None
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
        msg = (
            f"task_time_kill_switch tripped: task {task_id} on {hostname} took "
            f"{task_total_time:.1f}s > {task_time_kill_switch_seconds}s threshold. "
            f"Aborting the run. Timing log written to: {log_path}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # Debug: phase-grouped timing breakdown for this chunk. The generic
    # formatter lives in the top-level utils; the imaging phase layout
    # parameterizes it.
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
        "timing_node_tasks": timing_df,
        "deconvolution": combined_deconvolve_dict,
        "image_statistics": image_statistics,
    }
