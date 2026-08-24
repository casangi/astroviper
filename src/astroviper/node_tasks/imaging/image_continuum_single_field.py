from astroviper.utils.param_docs import shares_param_docs

###############################################################################
# Generic helper functions
###############################################################################


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


def _unwrap_continuum_reduce_result(input_data, node_name):
    """Return the dictionary produced by the continuum reduce stage."""
    if isinstance(input_data, (list, tuple)):
        if len(input_data) != 1:
            raise ValueError(
                f"{node_name} expects exactly one reduced input, "
                f"but received {len(input_data)}."
            )
        input_data = input_data[0]

    if not isinstance(input_data, dict):
        raise TypeError(
            f"{node_name} expects the dictionary returned by the reduce "
            f"stage, not {type(input_data).__name__}."
        )

    if "image" not in input_data:
        raise KeyError(
            f"The continuum reduce result supplied to {node_name} "
            "does not contain 'image'."
        )

    return input_data


def _resolve_continuum_append_configuration(input_params):
    """Resolve validated continuum parameters used by global append nodes."""
    import numpy as np

    if "image_params" not in input_params:
        raise KeyError("A continuum append node requires input_params['image_params'].")

    image_params = input_params["image_params"]

    if "nterms" not in image_params:
        raise KeyError("image_params must contain 'nterms'.")

    nterms = int(image_params["nterms"])

    if nterms < 1:
        raise ValueError(f"image_params['nterms'] must be positive; received {nterms}.")

    if "reference_frequency" in image_params:
        reference_frequency = float(image_params["reference_frequency"])
    elif "reference_frequency_hz" in image_params:
        reference_frequency = float(image_params["reference_frequency_hz"])
    else:
        raise KeyError(
            "image_params must contain 'reference_frequency' or "
            "'reference_frequency_hz'."
        )

    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "The continuum reference frequency must be finite and positive; "
            f"received {reference_frequency}."
        )

    single_precision_image = bool(input_params.get("single_precision_image", True))

    return {
        "image_params": image_params,
        "nterms": nterms,
        "n_psf_taylor_terms": 2 * nterms - 1,
        "reference_frequency": reference_frequency,
        "complex_dtype": (np.complex64 if single_precision_image else np.complex128),
        "processing_function_threads": int(
            input_params.get("processing_function_threads", 1)
        ),
        "fft_backend": input_params.get("fft_backend", "pyfftw"),
        "image_data_variables_keep": input_params.get(
            "image_data_variables_keep",
            [],
        ),
        "image_data_group_in_name": input_params.get(
            "image_data_group_in_name",
            "residual",
        ),
    }


def _install_static_continuum_products(
    img_xds,
    static_xds,
    image_data_group_name="residual",
):
    """Install cached PSF, PB, and beam-fit products into a residual image."""
    import xarray as xr

    static_variable_names = (
        "POINT_SPREAD_FUNCTION",
        "PRIMARY_BEAM",
        "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
        "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
    )

    for name in static_variable_names:
        if name not in static_xds:
            raise KeyError(f"static_xds does not contain {name!r}.")

        if name not in img_xds:
            source = static_xds[name]

            if (
                "frequency" in source.dims
                and "frequency" in img_xds.dims
                and source.sizes["frequency"] != img_xds.sizes["frequency"]
            ):
                frequency_users = [
                    variable_name
                    for variable_name, variable in img_xds.data_vars.items()
                    if "frequency" in variable.dims
                ]

                if frequency_users:
                    raise ValueError(
                        "Cannot replace the chunk frequency coordinate while "
                        f"variables still use it: {frequency_users}."
                    )

                img_xds = img_xds.drop_dims("frequency")

            source_coords = {
                dim: source.coords[dim]
                for dim in source.dims
                if dim in source.coords and dim not in img_xds.coords
            }

            if source_coords:
                img_xds = img_xds.assign_coords(source_coords)

            img_xds[name] = xr.Variable(
                source.dims,
                source.data.copy(),
                attrs=source.attrs.copy(),
            )

    data_groups = img_xds.attrs.setdefault("data_groups", {})
    residual_group = data_groups.setdefault(image_data_group_name, {})

    residual_group.update(
        {
            "point_spread_function": "POINT_SPREAD_FUNCTION",
            "primary_beam": "PRIMARY_BEAM",
            "beam_fit_params_point_spread_function": (
                "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
            ),
            "max_sidelobe_point_spread_function": (
                "MAX_SIDELOBE_POINT_SPREAD_FUNCTION"
            ),
        }
    )

    return img_xds


def _attach_imaging_weights_continuum(
    ps_xdt,
    weight_datasets,
    processing_set_data_group_name="corrected",
    weight_imaging_name="WEIGHT_IMAGING",
):
    """Attach cached imaging weights to a loaded processing-set chunk.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Freshly loaded processing-set chunk.

    weight_datasets : dict
        Mapping from processing-set child name to an xarray Dataset containing
        ``weight_imaging_name``.

    processing_set_data_group_name : str, optional
        Processing-set data group receiving the logical
        ``"weight_imaging"`` registration.

    weight_imaging_name : str, optional
        Name of the cached imaging-weight variable.

    Returns
    -------
    xarray.DataTree
        Processing-set chunk with cached imaging weights attached.
    """
    attached = 0

    for ms_name, ms_xdt in ps_xdt.items():
        if ms_name not in weight_datasets:
            raise KeyError(
                f"No cached imaging weights were found for processing-set "
                f"child {ms_name!r}. Available children are "
                f"{list(weight_datasets)}."
            )

        weight_xds = weight_datasets[ms_name]

        if weight_imaging_name not in weight_xds:
            raise KeyError(
                f"Cached dataset for {ms_name!r} does not contain "
                f"{weight_imaging_name!r}."
            )

        weight_da = weight_xds[weight_imaging_name]

        if (
            weight_da.dims
            != ms_xdt[
                ms_xdt.attrs["data_groups"][processing_set_data_group_name]["weight"]
            ].dims
        ):
            raise ValueError(
                f"Cached imaging weights for {ms_name!r} have dimensions "
                f"{weight_da.dims}, which do not match the visibility-weight "
                "dimensions."
            )

        for dim, size in weight_da.sizes.items():
            if dim not in ms_xdt.sizes:
                raise ValueError(
                    f"Cached imaging weights for {ms_name!r} contain unknown "
                    f"dimension {dim!r}."
                )

            if ms_xdt.sizes[dim] != size:
                raise ValueError(
                    f"Cached imaging weights for {ms_name!r} have size "
                    f"{size} along {dim!r}; the loaded processing-set chunk "
                    f"has size {ms_xdt.sizes[dim]}."
                )

        # Use a direct Variable assignment to avoid coordinate alignment.
        ms_xdt[weight_imaging_name] = weight_da.variable.copy(deep=False)

        data_groups = ms_xdt.attrs.setdefault("data_groups", {})

        if processing_set_data_group_name not in data_groups:
            raise KeyError(
                f"Processing-set child {ms_name!r} does not contain data group "
                f"{processing_set_data_group_name!r}."
            )

        data_groups[processing_set_data_group_name][
            "weight_imaging"
        ] = weight_imaging_name

        attached += 1

    if attached == 0:
        raise RuntimeError(
            "No imaging-weight datasets were attached to the processing-set " "chunk."
        )

    return ps_xdt


def _finalize_reference_primary_beam(
    img_xds,
    *,
    reference_frequency,
    image_data_group_name="residual",
):
    """Install the static MFS primary beam evaluated at reference frequency."""
    import numpy as np
    import xarray as xr

    reference_name = "PRIMARY_BEAM_REFERENCE"
    if reference_name not in img_xds:
        raise KeyError(f"MFS image is missing {reference_name!r}.")
    primary_beam = img_xds[reference_name]

    # The globally reduced continuum products use Taylor dimensions, not
    # the chunk-local frequency dimension. Remove the stale frequency
    # coordinate before creating the effective one-channel PB.
    if "frequency" in img_xds.dims:
        variables_with_frequency = [
            name
            for name, data_array in img_xds.data_vars.items()
            if "frequency" in data_array.dims
        ]

        if variables_with_frequency:
            raise ValueError(
                "Cannot replace the chunk-local frequency dimension because "
                "these variables still depend on it: "
                f"{variables_with_frequency}."
            )

        img_xds = img_xds.drop_dims(
            "frequency",
            errors="ignore",
        )

    primary_beam = primary_beam.expand_dims(
        frequency=np.asarray(
            [float(reference_frequency)],
            dtype=np.float64,
        ),
        axis=1,
    )

    primary_beam.attrs = img_xds[reference_name].attrs.copy()
    primary_beam.attrs.update(
        {
            "description": (
                "Continuum zeroth-order primary beam evaluated directly "
                "at the MT-MFS reference frequency."
            ),
            "type": "primary_beam",
            "continuum_pb_order": 0,
            "effective_frequency_hz": float(reference_frequency),
            "effective_frequency_interpretation": "direct_evaluation",
        }
    )

    # Direct Variable assignment is safe now because the dataset has no
    # conflicting frequency coordinate.
    img_xds["PRIMARY_BEAM"] = xr.Variable(
        primary_beam.dims,
        primary_beam.data,
        attrs=primary_beam.attrs.copy(),
    )

    data_groups = img_xds.attrs.setdefault("data_groups", {})
    image_data_group = data_groups.setdefault(
        image_data_group_name,
        {},
    )

    image_data_group["primary_beam"] = "PRIMARY_BEAM"
    image_data_group.pop("primary_beam_reference", None)

    img_xds = img_xds.drop_vars(
        [reference_name],
        errors="ignore",
    )

    return img_xds


###############################################################################
# Node task level functionality related to the residual update
###############################################################################


@shares_param_docs
def residual_update_continuum_single_field(
    image_params,
    imaging_weights_params,
    task_coords,
    data_selection,
    image_store,
    input_data_store,
    specmode="mfs",
    pb_cache_mapping=None,
    model_xds=None,
    processing_set_data_group_name="corrected",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    memory_mode="in_memory",
    skunk_works=False,
    data_group=None,
    is_n_iter_0=True,
    model_uv_xds=None,
    task_id=0,
    pblimit=0.2,
    input_data=None,
    task_time_kill_switch_seconds=None,
    weight_cache_mapping=None,
):
    """Compute one frequency chunk's continuum products in memory.

    This node task constructs the per-chunk image dataset, loads the corresponding
    visibility partition, and calls the continuum residual processing function.

    Unlike the cube imaging node task, this function performs no image writing.
    MFS returns chunk-local UV-domain products for numerical reduction. MVC
    normalizes and inverse-transforms its exclusively owned channel grids,
    applies the channel primary-beam convention, and forms additive Taylor
    numerators locally. Polarization conversion, final global normalization,
    the minor cycle, and restoration remain append operations.

    The MFS dataset typically contains

    - ``VISIBILITY``;
    - ``VISIBILITY_NORMALIZATION``;
    - ``UV_SAMPLING``;
    - ``UV_SAMPLING_NORMALIZATION``;

    MVC instead returns ``MVC_RESIDUAL_TAYLOR_NUMERATOR`` and
    ``MVC_RESIDUAL_WEIGHT_SUM``. During the first major cycle it also returns
    the additive Taylor PSF numerator, PSF weight sum, and weighted PB sum.

    Parameters
    ----------
    image_params : dict
        Image geometry and output coordinates. In addition to the standard imaging
        parameters, the continuum processing function may require entries such as
        ``reference_frequency`` and ``nterms``.

    imaging_weights_params : dict
        Imaging-weight configuration.

    task_coords : dict
        Per-chunk coordinate mapping. ``task_coords["frequency"]["data"]``
        supplies the frequency coordinates assigned to this worker.

    data_selection : dict
        Visibility selection injected by GraphViper.

    image_store : str
        Retained for task watchdog logging. This node task does not write image
        products.

    input_data_store : str
        Processing-set store used when ``input_data`` is not supplied.

    processing_set_data_group_name : str, optional
        Processing-set data group to image.

    instrument_polarization_basis : str, optional
        Instrument correlation basis used during gridding.

    single_precision_image : bool, optional
        Whether image-domain arrays use single precision.

    processing_function_threads : int, optional
        Number of threads supplied to the imaging kernels.

    fft_backend : str, optional
        FFT backend used by the processing function.

    image_data_variables_keep : list of str, optional
        Logical image products retained in the returned dataset.

    memory_mode : str, optional
        Currently only ``"in_memory"`` is supported.

    skunk_works : bool, optional
        Use the experimental direct-Zarr loading path.

    data_group : dict, optional
        Role-to-variable mapping used by the skunk-works loader.

    is_n_iter_0 : bool, optional
        Indicates whether this is the first major cycle.

    model_uv_xds : xarray.Dataset, optional
        Precomputed Taylor-domain model visibility grids used for degridding.
        Ignored during the first major cycle.

    task_id : int, optional
        Identifier of the frequency chunk.

    pblimit : float, optional
        Channel primary-beam cutoff used for MVC map-local Taylor products.

    input_data : dict, optional
        Preloaded visibility data for this chunk.

    task_time_kill_switch_seconds : float, optional
        Abort the distributed calculation if this task exceeds the specified wall
        time.

    Returns
    -------
    dict
        Dictionary containing

        ``"image"``
            Chunk-local continuum UV-domain products.

        ``"timing_node_tasks"``
            One-row dataframe summarizing task timing information."""

    import socket
    import time

    import numpy as np
    import toolviper.utils.logger as logger
    import xarray as xr
    from toolviper.utils.memory_management import get_rss_gb
    from xradio.image import make_empty_sky_image

    import astroviper.processing_functions as pf
    from astroviper.processing_functions.imaging.utils import (
        IMAGING_TIMING_PHASES,
        IMAGING_TIMING_TOTAL_KEY,
    )
    from astroviper.utils.timing import print_timing_summary

    # =============================================================
    # Initialization
    # =============================================================

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

    specmode = str(specmode).lower()

    if specmode not in ("mfs", "mvc"):
        raise ValueError(
            "specmode must be either 'mfs' or 'mvc'; " f"received {specmode!r}."
        )

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
        spectral_reference=image_params.get("spectral_reference", "lsrk"),
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

    # =============================================================
    # Load Imaging Cache
    # =============================================================

    if weight_cache_mapping is not None:
        task_id = int(task_id)

        if task_id not in weight_cache_mapping:
            raise KeyError(f"No cached imaging weights were found for task {task_id}.")

        ps_xdt = _attach_imaging_weights_continuum(
            ps_xdt,
            weight_cache_mapping[task_id],
            processing_set_data_group_name=(processing_set_data_group_name),
            weight_imaging_name="WEIGHT_IMAGING",
        )

    # =============================================================
    # Resolve the task-local MVC primary beam
    # =============================================================

    specmode = str(specmode).lower()

    if specmode not in ("mfs", "mvc"):
        raise ValueError(
            "specmode must be either 'mfs' or 'mvc'; " f"received {specmode!r}."
        )

    primary_beam_xds = None

    if specmode == "mvc" and not is_n_iter_0:
        if pb_cache_mapping is None:
            raise ValueError("Later MVC major cycles require pb_cache_mapping.")

        task_id = int(task_id)

        if task_id not in pb_cache_mapping:
            raise KeyError(
                "No cached MVC primary beam exists for "
                f"task {task_id}. Available task identifiers are "
                f"{sorted(pb_cache_mapping)}."
            )

        primary_beam_xds = pb_cache_mapping[task_id]

        if not isinstance(primary_beam_xds, xr.Dataset):
            raise TypeError(
                "The cached MVC primary beam for task "
                f"{task_id} must be an xarray.Dataset; received "
                f"{type(primary_beam_xds).__name__}."
            )

        primary_beam_name = primary_beam_xds.attrs.get(
            "primary_beam_name",
            "PRIMARY_BEAM",
        )

        if primary_beam_name not in primary_beam_xds:
            raise KeyError(
                "The cached MVC primary-beam dataset for task "
                f"{task_id} does not contain "
                f"{primary_beam_name!r}."
            )

        primary_beam = primary_beam_xds[primary_beam_name]

        if "frequency" not in primary_beam.dims:
            raise ValueError(
                "The cached MVC primary beam must contain a " "frequency dimension."
            )

        expected_frequency = np.asarray(
            task_coords["frequency"]["data"],
            dtype=np.float64,
        )

        actual_frequency = np.asarray(
            primary_beam.coords["frequency"].values,
            dtype=np.float64,
        )

        if not np.array_equal(
            actual_frequency,
            expected_frequency,
        ):
            raise ValueError(
                "The cached MVC primary-beam frequencies do not "
                "match the frequencies assigned to this map task: "
                f"cached={actual_frequency}, "
                f"task={expected_frequency}."
            )

    # =============================================================
    # Run processing function
    # =============================================================

    # Run the continuum processing function.
    img_xds, timing_df = pf.imaging.residual_update_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        imaging_weights_params,
        processing_set_data_group_name=(processing_set_data_group_name),
        instrument_polarization_basis=(instrument_polarization_basis),
        single_precision_image=single_precision_image,
        processing_function_threads=(processing_function_threads),
        fft_backend=fft_backend,
        image_data_variables_keep=(image_data_variables_keep),
        is_n_iter_0=is_n_iter_0,
        model_uv_xds=model_uv_xds,
        model_xds=model_xds,
        primary_beam_xds=primary_beam_xds,
        specmode=specmode,
        task_id=task_id,
        pblimit=pblimit,
    )

    # Retain the task-local frequency-dependent PB for MVC
    pb_xds = None

    if specmode == "mvc" and is_n_iter_0:
        import xarray as xr

        image_data_groups = img_xds.attrs.get(
            "data_groups",
            {},
        )

        if "residual" not in image_data_groups:
            raise KeyError("MVC setup did not create the residual image data group.")

        residual_data_group = image_data_groups["residual"]
        primary_beam_name = residual_data_group.get("primary_beam")

        if primary_beam_name is None:
            raise KeyError(
                "The MVC residual data group does not register a " "primary beam."
            )

        if primary_beam_name not in img_xds:
            raise KeyError(
                f"The MVC residual data group registers primary beam "
                f"{primary_beam_name!r}, but that variable is absent."
            )

        primary_beam = img_xds[primary_beam_name]

        if "frequency" not in primary_beam.dims:
            raise ValueError(
                "The MVC primary beam must retain its frequency " "dimension."
            )

        expected_frequency = np.asarray(
            task_coords["frequency"]["data"],
            dtype=np.float64,
        )
        actual_frequency = np.asarray(
            primary_beam.coords["frequency"].values,
            dtype=np.float64,
        )

        if not np.array_equal(
            actual_frequency,
            expected_frequency,
        ):
            raise ValueError(
                "The MVC primary-beam frequencies do not match the "
                "map-task frequencies."
            )

        pb_xds = xr.Dataset(
            {
                primary_beam_name: xr.Variable(
                    dims=primary_beam.dims,
                    data=primary_beam.data,
                    attrs=primary_beam.attrs.copy(),
                )
            },
            coords={
                dim: primary_beam.coords[dim]
                for dim in primary_beam.dims
                if dim in primary_beam.coords
            },
            attrs={
                "primary_beam_name": primary_beam_name,
                "task_id": int(task_id),
                "specmode": "mvc",
            },
        )

    if specmode == "mvc":
        # Taylor numerators and normalization sums have already been formed by
        # this map task.  Keep the channel PB only in pb_xds for later model
        # prediction and do not send any frequency-sized image through reduce.
        frequency_variables = [
            name
            for name, data_array in img_xds.data_vars.items()
            if "frequency" in data_array.dims
        ]
        img_xds = img_xds.drop_vars(frequency_variables, errors="ignore")
        if "frequency" in img_xds.dims:
            img_xds = img_xds.drop_dims("frequency", errors="ignore")

    # Preserve the first-cycle imaging weights for reuse by later major cycles.
    # They may have been loaded from the processing set or calculated in setup.
    weight_datasets = {}

    if is_n_iter_0 and weight_cache_mapping is None:
        import xarray as xr

        for ms_name, ms_xds in ps_xdt.items():
            data_group = ms_xds.attrs["data_groups"][processing_set_data_group_name]

            weight_name = data_group.get("weight_imaging")

            if weight_name is None or weight_name not in ms_xds:
                raise RuntimeError(
                    f"No imaging weights were available for {ms_name!r} "
                    "after first-cycle setup."
                )

            weight_da = ms_xds[weight_name]

            weight_datasets[ms_name] = xr.Dataset(
                {
                    weight_name: xr.Variable(
                        dims=weight_da.dims,
                        data=weight_da.data,
                        attrs=weight_da.attrs.copy(),
                    )
                },
                attrs={
                    "weight_imaging_name": weight_name,
                    "processing_set_data_group_name": (processing_set_data_group_name),
                    "task_id": int(task_id),
                },
            )

    # No output image writing is performed here. The Taylor products stay in
    # memory and flow directly into the GraphViper reduction stage.

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

    print_timing_summary(
        timing_df,
        IMAGING_TIMING_PHASES,
        total_key=IMAGING_TIMING_TOTAL_KEY,
        printer=logger.debug,
    )

    return_dict = {
        "image": img_xds,
        "timing_node_tasks": timing_df,
    }

    # If we have computed weights in the first major loop locally, we need to communicate that back to the driver
    if weight_datasets:
        return_dict["task_id"] = int(task_id)
        return_dict["weight_datasets"] = weight_datasets

    # Return primary beams per channel if present
    if pb_xds is not None:
        return_dict["task_id"] = int(task_id)
        return_dict["pb_xds"] = pb_xds

    return return_dict


@shares_param_docs
def grid_imaging_weight_density_continuum_node(
    image_params,
    imaging_weights_params,
    task_coords,
    data_selection,
    input_data_store,
    processing_set_data_group_name="corrected",
    instrument_polarization_basis="linear",
    single_precision_gridding=False,
    processing_function_threads=1,
    skunk_works=False,
    data_group=None,
    input_data=None,
    task_id=0,
):
    """Grid one visibility partition's contribution to the global weight density.

    This node task implements the map stage of the distributed continuum
    imaging-weight calculation. It loads one processing-set partition, creates
    the corresponding local image geometry, and calls
    ``grid_imaging_weight_density_continuum`` to produce the partition-local
    weight-density grid and sum-of-weight contribution.

    The node does not calculate Briggs factors, degrid imaging weights, or
    create ``WEIGHT_IMAGING``. Those operations occur only after all local
    density contributions have been combined by the GraphViper reduce stage.

    Returns
    -------
    dict
        Dictionary containing

        ``"weight_density"``
            An :class:`xarray.Dataset` containing
            ``WEIGHT_DENSITY_GRID`` and ``SUM_WEIGHT``.

        ``"timing_node_tasks"``
            One-row timing dataframe for this map task.

        ``"task_id"``
            Integer task identifier.
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger
    from xradio.image import make_empty_sky_image

    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        grid_imaging_weight_density_continuum,
    )

    task_id = int(task_id)
    task_start = time.time()

    # ------------------------------------------------------------------
    # Load this task's visibility partition.
    # ------------------------------------------------------------------
    start = time.time()

    if input_data is not None:
        # GraphViper or another caller already loaded and selected the
        # processing-set partition.
        ps_xdt = input_data

    elif skunk_works:
        from astroviper.node_tasks.imaging.utils import load_processing_set_skunk_works

        ps_xdt = load_processing_set_skunk_works(
            input_data_store,
            sel_parms=data_selection,
            data_group=data_group,
            processing_set_data_group_name=(processing_set_data_group_name),
            frequency_coords=task_coords["frequency"]["data"],
            instrument_polarization_basis=(instrument_polarization_basis),
            processing_function_threads=(processing_function_threads),
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

    # ------------------------------------------------------------------
    # Construct the local image geometry used by the density gridder.
    # ------------------------------------------------------------------
    if "frequency" not in task_coords:
        raise KeyError(
            "grid_imaging_weight_density_continuum_node requires "
            "task_coords['frequency']."
        )

    if "data" not in task_coords["frequency"]:
        raise KeyError(
            "task_coords['frequency'] must contain the local frequency "
            "coordinate under the key 'data'."
        )

    correlation_pol_coords = {
        "linear": ["XX", "YY"],
        "circular": ["RR", "LL"],
    }

    if instrument_polarization_basis not in correlation_pol_coords:
        raise ValueError(
            "instrument_polarization_basis must be 'linear' or "
            f"'circular'; received "
            f"{instrument_polarization_basis!r}."
        )

    start = time.time()

    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=task_coords["frequency"]["data"],
        pol_coords=correlation_pol_coords[instrument_polarization_basis],
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )

    # Required by the xradio image accessor used to retrieve the image
    # cell size.
    img_xds.attrs["type"] = "image_dataset"

    T_make_empty_image = time.time() - start

    # ------------------------------------------------------------------
    # Grid the partition-local density contribution.
    # ------------------------------------------------------------------
    start = time.time()

    weight_density_xds = grid_imaging_weight_density_continuum(
        ps_xdt,
        img_xds,
        imaging_weights_params,
        ms_data_group_in_name=processing_set_data_group_name,
        single_precision_gridding=single_precision_gridding,
        processing_function_threads=processing_function_threads,
    )

    T_grid_weight_density = time.time() - start

    # ------------------------------------------------------------------
    # Validate the map-task output before returning it to the reducer.
    # ------------------------------------------------------------------
    required_variables = (
        "WEIGHT_DENSITY_GRID",
        "SUM_WEIGHT",
    )

    missing_variables = [
        name for name in required_variables if name not in weight_density_xds
    ]

    if missing_variables:
        raise RuntimeError(
            "The weight-density processing function did not create all "
            f"required products. Missing: {missing_variables}."
        )

    if "frequency" not in weight_density_xds.coords:
        raise RuntimeError(
            "The partition-local weight-density result does not contain "
            "a frequency coordinate."
        )

    task_total_time = time.time() - task_start

    timing_df = pd.DataFrame(
        {
            "task_id": [task_id],
            "T_load": [T_load],
            "T_make_empty_image": [T_make_empty_image],
            "T_grid_weight_density": [T_grid_weight_density],
            "T_weight_density_node": [task_total_time],
            "n_frequency_channels": [weight_density_xds.sizes.get("frequency", 0)],
        }
    )

    logger.debug(
        "Finished continuum weight-density task "
        f"{task_id}: "
        f"{weight_density_xds.sizes.get('frequency', 0)} channels, "
        f"{task_total_time:.3f} s."
    )

    # The visibility partition is not needed after its density
    # contribution has been constructed.
    ps_xdt = None
    img_xds = None

    return {
        "task_id": task_id,
        "weight_density": weight_density_xds,
        "timing_node_tasks": timing_df,
    }


@shares_param_docs
def degrid_imaging_weights_continuum_node(
    image_params,
    imaging_weights_params,
    global_weighting_xds,
    task_coords,
    data_selection,
    input_data_store,
    processing_set_data_group_name="corrected",
    instrument_polarization_basis="linear",
    processing_function_threads=1,
    skunk_works=False,
    data_group=None,
    input_data=None,
    task_id=0,
):
    """Create final imaging weights for one visibility partition.

    This node implements the map stage of the second distributed continuum
    weighting graph. It loads one processing-set partition, selects the
    corresponding planes from the globally reduced weight-density products,
    degrids the global Briggs weighting solution onto the local visibility
    samples, and returns only the resulting imaging-weight arrays.

    No density gridding or Briggs-factor calculation is performed here.
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger
    import xarray as xr
    from xradio.image import make_empty_sky_image

    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        degrid_imaging_weights_continuum,
    )

    task_id = int(task_id)
    task_start = time.time()

    if not isinstance(global_weighting_xds, xr.Dataset):
        raise TypeError(
            "global_weighting_xds must be an xarray.Dataset; received "
            f"{type(global_weighting_xds).__name__}."
        )

    # -------------------------------------------------------------
    # Load this task's visibility partition.
    # -------------------------------------------------------------
    start = time.time()

    if input_data is not None:
        ps_xdt = input_data

    elif skunk_works:
        from astroviper.node_tasks.imaging.utils import load_processing_set_skunk_works

        ps_xdt = load_processing_set_skunk_works(
            input_data_store,
            sel_parms=data_selection,
            data_group=data_group,
            processing_set_data_group_name=(processing_set_data_group_name),
            frequency_coords=task_coords["frequency"]["data"],
            instrument_polarization_basis=(instrument_polarization_basis),
            processing_function_threads=(processing_function_threads),
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

    # -------------------------------------------------------------
    # Construct the local image geometry needed by the degridder.
    # -------------------------------------------------------------
    if "frequency" not in task_coords:
        raise KeyError(
            "degrid_imaging_weights_continuum_node requires "
            "task_coords['frequency']."
        )

    if "data" not in task_coords["frequency"]:
        raise KeyError(
            "task_coords['frequency'] must contain the local channel "
            "coordinate under 'data'."
        )

    correlation_pol_coords = {
        "linear": ["XX", "YY"],
        "circular": ["RR", "LL"],
    }

    if instrument_polarization_basis not in correlation_pol_coords:
        raise ValueError(
            "instrument_polarization_basis must be 'linear' or "
            f"'circular'; received {instrument_polarization_basis!r}."
        )

    start = time.time()

    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=task_coords["frequency"]["data"],
        pol_coords=correlation_pol_coords[instrument_polarization_basis],
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )
    img_xds.attrs["type"] = "image_dataset"

    T_make_empty_image = time.time() - start

    # -------------------------------------------------------------
    # Degrid the global weighting products.
    # -------------------------------------------------------------
    start = time.time()

    ps_xdt = degrid_imaging_weights_continuum(
        ps_xdt,
        img_xds,
        global_weighting_xds,
        imaging_weights_params,
        ms_data_group_in_name=processing_set_data_group_name,
        ms_data_group_out_name=processing_set_data_group_name,
        ms_data_group_out_modified={
            "weight_imaging": "WEIGHT_IMAGING",
        },
        overwrite=True,
        processing_function_threads=processing_function_threads,
    )

    T_degrid_imaging_weights = time.time() - start

    # -------------------------------------------------------------
    # Extract only the task-local imaging-weight arrays.
    # -------------------------------------------------------------
    start = time.time()

    weight_datasets = {}

    for ms_name, ms_xds in ps_xdt.items():
        data_groups = ms_xds.attrs.get("data_groups", {})

        if processing_set_data_group_name not in data_groups:
            raise KeyError(
                f"Processing-set child {ms_name!r} does not contain "
                f"data group {processing_set_data_group_name!r}."
            )

        data_group = data_groups[processing_set_data_group_name]
        weight_name = data_group.get("weight_imaging")

        if weight_name is None:
            raise KeyError(
                f"The degridding stage did not register "
                f"'weight_imaging' for child {ms_name!r}."
            )

        if weight_name not in ms_xds:
            raise KeyError(
                f"The registered imaging-weight variable "
                f"{weight_name!r} is absent from child {ms_name!r}."
            )

        weight_da = ms_xds[weight_name]

        # Create an independent lightweight dataset containing only the
        # per-visibility imaging weights. The visibility partition itself
        # must not remain reachable from the map result.
        weight_datasets[ms_name] = xr.Dataset(
            {
                weight_name: xr.Variable(
                    dims=weight_da.dims,
                    data=weight_da.data,
                    attrs=weight_da.attrs.copy(),
                )
            },
            attrs={
                "weight_imaging_name": weight_name,
                "processing_set_data_group_name": (processing_set_data_group_name),
                "task_id": task_id,
            },
        )

    if not weight_datasets:
        raise RuntimeError(
            f"No imaging-weight datasets were produced for task {task_id}."
        )

    T_extract_imaging_weights = time.time() - start
    task_total_time = time.time() - task_start

    timing_df = pd.DataFrame(
        {
            "task_id": [task_id],
            "T_load": [T_load],
            "T_make_empty_image": [T_make_empty_image],
            "T_degrid_imaging_weights": [T_degrid_imaging_weights],
            "T_extract_imaging_weights": [T_extract_imaging_weights],
            "T_imaging_weight_degrid_node": [task_total_time],
            "n_frequency_channels": [img_xds.sizes.get("frequency", 0)],
            "n_processing_set_children": [len(weight_datasets)],
        }
    )

    logger.debug(
        "Finished continuum imaging-weight degrid task "
        f"{task_id}: {len(weight_datasets)} processing-set children, "
        f"{task_total_time:.3f} s."
    )

    ps_xdt = None
    img_xds = None

    return {
        "task_id": task_id,
        "weight_datasets": weight_datasets,
        "timing_node_tasks": timing_df,
    }


###############################################################################
# Node task level functionality related to the append node
###############################################################################


def _prepare_continuum_image(
    img_xds,
    input_params,
    *,
    initialize_static_products,
    pb_cache_mapping=None,
):
    """Convert globally reduced MFS or MVC products to image-domain products.

    For ``specmode='mfs'``, the reduced visibility grids already represent
    Taylor residual terms. They are inverse Fourier-transformed directly into
    ``SKY_RESIDUAL(taylor_term)``.

    For ``specmode='mvc'``, map tasks have already normalized and inverse
    Fourier-transformed their exclusively owned channels, applied the channel
    primary-beam convention, and constructed additive Taylor numerators. The
    reducer sums those compact products; this function performs only their
    global normalization and constructs the effective primary beam.

    MFS retains its direct Taylor-weighted PSF path.  MVC constructs its Taylor
    PSFs from the channel PSF cube.  Both are fitted only during the first major
    cycle; later cycles reinstall the cached static products.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Globally reduced image dataset produced by the distributed reduce stage.

    input_params : dict
        Parameters supplied to the append node. Must contain ``image_params``.
        For later cycles, it must also contain ``static_xds``.

    initialize_static_products : bool
        If true, construct and cache the global PSF, restoring-beam fit, and
        primary-beam products. This should be true only for the first major
        cycle.

    pb_cache_mapping : dict, optional
        MVC channel-primary-beam cache carried through the append node for
        later map-task model prediction. Taylor image preparation itself uses
        the already reduced PB sum rather than assembling this cache.

    Returns
    -------
    img_xds : xarray.Dataset
        Image-domain dataset containing Taylor residuals and static products.

    static_xds : xarray.Dataset
        Cached PSF, primary beam, beam-fit parameters, and maximum sidelobe.

    psf_fit_return_df : pandas.DataFrame or None
        PSF Gaussian-fit timing information. It is non-null only when static
        products are initialized.
    """
    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        ifft_norm_img_xds,
    )
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        finalize_mvc_taylor_normal_equations,
        point_spread_function_gaussian_fit_continuum,
    )
    from astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field import (
        _rename_psf_frequency_axis_to_taylor_order,
    )

    config = _resolve_continuum_append_configuration(input_params)

    image_params = config["image_params"]
    nterms = config["nterms"]
    n_psf_taylor_terms = config["n_psf_taylor_terms"]
    reference_frequency = config["reference_frequency"]
    image_data_group_name = config["image_data_group_in_name"]

    specmode = str(
        input_params.get(
            "specmode",
            image_params.get("specmode", "mfs"),
        )
    ).lower()

    if specmode not in ("mfs", "mvc"):
        raise ValueError(
            "specmode must be either 'mfs' or 'mvc'; " f"received {specmode!r}."
        )

    if pb_cache_mapping is None:
        pb_cache_mapping = input_params.get("pb_cache_mapping")

    # MFS still reduces Taylor-weighted UV grids and therefore performs its
    # inverse FFT globally. MVC map tasks already returned Taylor products.
    residual_output_name = "SKY_RESIDUAL"

    if specmode == "mfs":
        img_xds = ifft_norm_img_xds(
            img_xds,
            image_params=image_params,
            image_data_group_in_name=image_data_group_name,
            image_data_group_out_name=image_data_group_name,
            image_data_group_out_modified={
                "sky": residual_output_name,
            },
            image_data_variables_keep=config["image_data_variables_keep"],
            processing_function_threads=config["processing_function_threads"],
            fft_backend=config["fft_backend"],
            complex_dtype=config["complex_dtype"],
        )

    if specmode == "mfs" and residual_output_name not in img_xds:
        raise RuntimeError(
            "The global residual inverse FFT did not "
            f"create {residual_output_name!r}."
        )

    # ------------------------------------------------------------------
    # Direct MFS: the inverse-FFT result is already a Taylor stack.
    # ------------------------------------------------------------------
    if specmode == "mfs":

        residual = img_xds["SKY_RESIDUAL"]

        if "taylor_term" not in residual.dims:
            raise RuntimeError(
                "SKY_RESIDUAL does not contain the 'taylor_term' dimension."
            )

        actual_nterms = img_xds["SKY_RESIDUAL"].sizes["taylor_term"]

        if actual_nterms != nterms:
            raise ValueError(
                "The residual Taylor stack is inconsistent with nterms: "
                f"received {actual_nterms}, expected {nterms}."
            )

        residual.attrs.update(
            {
                "description": "Continuum residual Taylor products.",
                "nterms": nterms,
                "reference_frequency": reference_frequency,
                "placeholder": False,
            }
        )

    elif specmode == "mvc":
        effective_primary_beam = None
        if not initialize_static_products:
            static_primary_beam = input_params["static_xds"]["PRIMARY_BEAM"]
            effective_primary_beam = static_primary_beam.isel(
                frequency=0,
                drop=True,
            ).transpose("time", "polarization", "l", "m")

        (
            residual_taylor,
            psf_taylor,
            effective_primary_beam,
        ) = finalize_mvc_taylor_normal_equations(
            img_xds,
            effective_primary_beam=effective_primary_beam,
        )

        img_xds["SKY_RESIDUAL"] = residual_taylor
        img_xds.attrs["data_groups"][image_data_group_name]["sky"] = "SKY_RESIDUAL"

        if initialize_static_products:
            if psf_taylor is None:
                raise RuntimeError(
                    "The first MVC reduction did not contain Taylor PSF "
                    "contributions."
                )
            img_xds["POINT_SPREAD_FUNCTION"] = psf_taylor
            img_xds.attrs["data_groups"][image_data_group_name][
                "point_spread_function"
            ] = "POINT_SPREAD_FUNCTION"

        # ----------------------------------------------------------
        # Construct the effective PB used only by the minor loop.
        # ----------------------------------------------------------

        contribution_variables = [
            name for name in img_xds.data_vars if name.startswith("MVC_")
        ]
        img_xds = img_xds.drop_vars(contribution_variables, errors="ignore")

        # ----------------------------------------------------------
        # Recreate the weighted common PB as a singleton-frequency image.
        # ----------------------------------------------------------

        if "time" in effective_primary_beam.dims:
            frequency_axis = effective_primary_beam.dims.index("time") + 1
        else:
            frequency_axis = 0

        effective_primary_beam = effective_primary_beam.expand_dims(
            frequency=np.asarray(
                [reference_frequency],
                dtype=np.float64,
            ),
            axis=frequency_axis,
        )

        effective_primary_beam.attrs.update(
            {
                "description": "Effective MVC primary beam",
                "type": "primary_beam",
                "specmode": "mvc",
                "primary_beam_usage": "common_taylor_convention",
                "method": "airy_disk",
            }
        )

        img_xds["PRIMARY_BEAM"] = xr.Variable(
            dims=effective_primary_beam.dims,
            data=effective_primary_beam.data,
            attrs=effective_primary_beam.attrs.copy(),
        )

        residual_group = img_xds.attrs.setdefault("data_groups", {},).setdefault(
            image_data_group_name,
            {},
        )

        residual_group["primary_beam"] = "PRIMARY_BEAM"

    psf_fit_return_df = None

    # ------------------------------------------------------------------
    # First major cycle: form and cache global static products.
    # ------------------------------------------------------------------

    if initialize_static_products:
        if specmode == "mfs":
            img_xds = ifft_norm_img_xds(
                img_xds,
                image_params=image_params,
                image_data_group_in_name=image_data_group_name,
                image_data_group_out_name=image_data_group_name,
                image_data_group_out_modified={
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                },
                image_data_variables_keep=config["image_data_variables_keep"],
                processing_function_threads=config["processing_function_threads"],
                fft_backend=config["fft_backend"],
                complex_dtype=config["complex_dtype"],
            )

            # Direct MFS PSF Taylor terms may emerge from the generic FFT helper
            # on a frequency-like axis.  Re-label that axis without changing data.
            img_xds = _rename_psf_frequency_axis_to_taylor_order(
                img_xds,
                image_data_group_out_name=image_data_group_name,
                n_psf_taylor_terms=n_psf_taylor_terms,
            )

        # register name of data group
        psf_name = img_xds.attrs["data_groups"][image_data_group_name][
            "point_spread_function"
        ]

        if psf_name not in img_xds:
            raise RuntimeError(
                "The global PSF inverse FFT did not " f"create {psf_name!r}."
            )

        psf = img_xds[psf_name]

        if "psf_taylor_order" not in psf.dims:
            raise RuntimeError(
                f"{psf_name!r} does not contain the " "'psf_taylor_order' dimension."
            )

        actual_psf_terms = img_xds[psf_name].sizes["psf_taylor_order"]

        if actual_psf_terms != n_psf_taylor_terms:
            raise ValueError(
                "The PSF Taylor stack is inconsistent with nterms: "
                f"received {actual_psf_terms}, "
                f"expected {n_psf_taylor_terms}."
            )

        # Update attributes
        img_xds[psf_name].attrs.update(
            {
                "type": "point_spread_function",
                "nterms": nterms,
                "n_psf_taylor_terms": n_psf_taylor_terms,
                "reference_frequency": reference_frequency,
            }
        )

        # Fit point spread function
        (img_xds, psf_fit_return_df,) = point_spread_function_gaussian_fit_continuum(
            img_xds,
            image_data_group_in_name=image_data_group_name,
            image_data_group_out_name=image_data_group_name,
            processing_function_threads=config["processing_function_threads"],
        )

        # ----------------------------------------------------------
        # Construct or validate the PB used by the minor-loop/static API.
        # ----------------------------------------------------------

        if specmode == "mfs":
            # Install the static PB evaluated at the Taylor reference frequency.
            img_xds = _finalize_reference_primary_beam(
                img_xds,
                reference_frequency=reference_frequency,
                image_data_group_name=image_data_group_name,
            )

        elif specmode == "mvc":
            # MVC already constructed PRIMARY_BEAM immediately after the
            # frequency-cube-to-Taylor conversion.
            if "PRIMARY_BEAM" not in img_xds:
                raise RuntimeError(
                    "MVC image preparation did not construct PRIMARY_BEAM."
                )

            primary_beam = img_xds["PRIMARY_BEAM"]

            expected_dims = (
                "time",
                "frequency",
                "polarization",
                "l",
                "m",
            )

            if primary_beam.dims != expected_dims:
                raise ValueError(
                    "MVC PRIMARY_BEAM has unexpected dimensions: "
                    f"{primary_beam.dims}; expected {expected_dims}."
                )

            if primary_beam.sizes["frequency"] != 1:
                raise ValueError(
                    "MVC PRIMARY_BEAM must have exactly one effective "
                    "frequency plane."
                )

        else:
            raise ValueError(
                "specmode must be either 'mfs' or 'mvc'; " f"received {specmode!r}."
            )

        # Static variables to be shared with the minor loop controls
        static_variable_names = (
            "POINT_SPREAD_FUNCTION",
            "PRIMARY_BEAM",
            "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
        )

        missing = [name for name in static_variable_names if name not in img_xds]

        if missing:
            raise KeyError(
                "The first continuum global node could "
                "not construct all static products. "
                f"Missing: {missing}."
            )

        # Keep the correlation-basis cache independent: img_xds is converted
        # to Stokes below, while later residual updates reinstall this cache
        # and perform that conversion exactly once per cycle.
        static_xds = img_xds[list(static_variable_names)].copy(deep=True)
        static_xds.attrs = dict(img_xds.attrs)

    # ------------------------------------------------------------------
    # Later major cycles: reinstall static products.
    # ------------------------------------------------------------------

    else:
        # At later iterations, static_xds needs to be present already
        if "static_xds" not in input_params:
            raise KeyError(
                "static_xds must be supplied when static products are "
                "not initialized in this node."
            )

        static_xds = input_params["static_xds"]

        # Write static variables into img_xds
        img_xds = _install_static_continuum_products(
            img_xds,
            static_xds,
            image_data_group_name=image_data_group_name,
        )

    # This transformation is performed exactly once, after the global
    # inverse FFT. The map/reduce products must remain in correlation basis.
    img_xds = transform_polarization_basis(
        img_xds,
        new_polarization_basis="stokes",
        overwrite=True,
    )

    # transform_polarization_basis may place polarization after the spatial
    # dimensions. Restore the canonical layouts required by iteration control,
    # masking, deconvolution, and restoration.
    canonical_dimension_orders = {
        "SKY_RESIDUAL": (
            "time",
            "taylor_term",
            "polarization",
            "l",
            "m",
        ),
        "POINT_SPREAD_FUNCTION": (
            "time",
            "psf_taylor_order",
            "polarization",
            "l",
            "m",
        ),
        "PRIMARY_BEAM": (
            "time",
            "frequency",
            "polarization",
            "l",
            "m",
        ),
        "SKY_MODEL": (
            "time",
            "taylor_term",
            "polarization",
            "l",
            "m",
        ),
    }

    for variable_name, expected_dims in canonical_dimension_orders.items():
        if variable_name not in img_xds:
            continue

        data_array = img_xds[variable_name]

        if set(data_array.dims) != set(expected_dims):
            raise ValueError(
                f"{variable_name!r} has dimensions "
                f"{data_array.dims}; expected the dimension set "
                f"{expected_dims}."
            )

        img_xds[variable_name] = data_array.transpose(*expected_dims)

    return img_xds, static_xds, psf_fit_return_df


def _prepare_post_update_continuum_model_state(model_increment_xds, input_params):
    """Accumulate the model and prepare MFS Fourier state inside the append node.

    The distributed application supplies the previous image-domain model as
    append input and only forwards the two state objects returned here. All
    numerical accumulation, polarization conversion, and Fourier preparation
    are delegated to processing functions.

    Parameters
    ----------
    model_increment_xds : xarray.Dataset
        Image dataset produced by the current model-update cycle.
    input_params : dict
        Append-node configuration. Later cycles must provide ``model_xds``.

    Returns
    -------
    model_xds : xarray.Dataset
        Fully accumulated image-domain Taylor model.
    model_uv_xds : xarray.Dataset or None
        Fourier-domain MFS Taylor model, or ``None`` for MVC.
    """
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        accumulate_continuum_model,
        prepare_model_uv_continuum_single_field,
    )

    is_n_iter_0 = bool(input_params.get("is_n_iter_0", True))
    previous_model_xds = input_params.get("model_xds")

    if not is_n_iter_0 and previous_model_xds is None:
        raise KeyError(
            "Later continuum append nodes require input_params['model_xds']."
        )

    specmode = str(input_params.get("specmode", "mfs")).lower()
    model_xds = accumulate_continuum_model(
        model_increment_xds,
        previous_model_xds=previous_model_xds,
        specmode=specmode,
    )

    if specmode == "mfs":
        model_uv_xds = prepare_model_uv_continuum_single_field(
            model_xds,
            image_params=input_params["image_params"],
            instrument_polarization_basis=input_params.get(
                "instrument_polarization_basis",
                "linear",
            ),
            single_precision_image=input_params.get(
                "single_precision_image",
                True,
            ),
            processing_function_threads=input_params.get(
                "processing_function_threads",
                1,
            ),
            fft_backend=input_params.get("fft_backend", "pyfftw"),
        )
    elif specmode == "mvc":
        model_uv_xds = None
    else:
        raise ValueError(
            "specmode must be either 'mfs' or 'mvc'; " f"received {specmode!r}."
        )

    return model_xds, model_uv_xds


@shares_param_docs
def continuum_minor_cycle_node(
    input_data,
    input_params,
):
    """Prepare the globally reduced continuum image and execute one minor cycle.

    This node is executed exactly once per outer cycle control after all frequency chunks
    have been combined by the GraphViper reduce stage. Unlike the map node tasks,
    which operate independently on individual frequency partitions, this node has
    access to the globally accumulated continuum products.

    The node performs the following steps:

    1. converts the reduced UV-domain continuum products into image-domain
       quantities by applying the global inverse FFT;
    2. normalizes the residual image and PSF using the accumulated imaging-weight
       products;
    3. converts the image from the instrument correlation basis to the requested
       Stokes basis;
    4. installs the static continuum products (for example, the primary beam and
       fitted restoring beam parameters);
    5. executes one continuum minor cycle, updating the sky model and producing the
       corresponding model increment;
    6. accumulates that increment into the persistent image-domain model and, for
       MFS, prepares its Fourier-domain Taylor grids for the next residual update.

    The accumulated image-domain model and the optional Fourier-domain MFS model
    are returned to the distributed application as state objects. The application
    forwards them to the next graph without performing numerical model operations.

    This node performs no distributed processing. It operates on the single,
    globally reduced continuum dataset produced by the GraphViper reduce stage.
    """

    import time

    input_data = _unwrap_continuum_reduce_result(
        input_data,
        "continuum_minor_cycle_node",
    )

    is_n_iter_0 = bool(input_params.get("is_n_iter_0", True))

    # prepare continuum image, this is doing 1.-4.
    pb_cache_mapping = input_data.get(
        "pb_cache_mapping",
        input_params.get("pb_cache_mapping"),
    )
    if pb_cache_mapping is None and "pb_xds" in input_data:
        pb_cache_mapping = {int(input_data["task_id"]): input_data["pb_xds"]}

    (img_xds, static_xds, psf_fit_return_df,) = _prepare_continuum_image(
        input_data["image"],
        input_params,
        initialize_static_products=is_n_iter_0,
        pb_cache_mapping=pb_cache_mapping,
    )

    input_data["image"] = img_xds

    model_update_input_params = dict(input_params)
    model_update_input_params.pop("static_xds", None)

    # A residual-update reduce has no deconvolution work of its own, but keeps
    # the result schema stable by returning an empty ReturnDict. Do not let
    # that placeholder hide the history supplied by the previous model update.
    reduced_deconvolution = input_data.get("deconvolution")
    previous_deconvolution = input_params.get("deconvolution")
    if (
        reduced_deconvolution is not None
        and not reduced_deconvolution.data
        and previous_deconvolution is not None
        and previous_deconvolution.data
    ):
        input_data["deconvolution"] = previous_deconvolution

    # Preserve imaging weights collected during the first major-cycle reduce.
    weight_cache_mapping = input_data.get("weight_cache_mapping")
    if weight_cache_mapping is None and "weight_datasets" in input_data:
        weight_cache_mapping = {
            int(input_data["task_id"]): input_data["weight_datasets"]
        }

    # Run the model update.
    return_dict = model_update_continuum_single_field(
        input_data,
        model_update_input_params,
    )

    # The append stage owns all numerical state preparation needed by the next
    # residual-update graph. The distributed application only forwards these
    # returned objects.
    start = time.time()
    model_xds, model_uv_xds = _prepare_post_update_continuum_model_state(
        return_dict["image"],
        input_params,
    )
    T_prepare_model_state = time.time() - start

    return_dict["model_xds"] = model_xds
    return_dict["model_uv_xds"] = model_uv_xds

    timing_model_update = return_dict.get("timing_model_update")
    if timing_model_update is not None:
        timing_model_update["T_prepare_model_state"] = T_prepare_model_state

    # model_update_continuum_single_field constructs its own return dictionary,
    # so explicitly carry the first-cycle imaging-weight cache through the
    # append node to the distributed driver.
    if weight_cache_mapping is not None:
        return_dict["weight_cache_mapping"] = weight_cache_mapping

    # do the same with the channelized primary beams, if present
    if pb_cache_mapping is not None:
        return_dict["pb_cache_mapping"] = pb_cache_mapping

    return_dict["static_xds"] = static_xds
    return_dict["timing_psf_fit"] = (
        None if psf_fit_return_df is None else psf_fit_return_df.reset_index(drop=True)
    )

    return return_dict


@shares_param_docs
def continuum_finalize_node(
    input_data,
    input_params,
):
    """Finalize the continuum imaging after the last major cycle.

    This node is executed once after the final GraphViper reduce stage has
    completed. It converts the globally accumulated continuum products into the
    final image-domain representation and produces the restored continuum image.

    The node performs the following steps:

    1. converts the reduced UV-domain continuum products into the final residual
       image by applying the global inverse FFT;
    2. converts the image from the instrument correlation basis to the requested
       Stokes basis;
    3. uses the cached static imaging products (for example, the primary beam and
       fitted restoring beam parameters);
    4. restores the image using the accumulated sky model already provided by the
       distributed application;
    5. removes intermediate products that are not requested in the final output.

    Unlike the continuum minor-cycle node, this function performs no
    deconvolution, no iteration-controller updates, and no distributed
    computation. It operates only on the single globally reduced continuum dataset
    produced by the final GraphViper reduce stage.
    """

    from astroviper.utils.data_group_tools import modify_data_groups_xds

    input_data = _unwrap_continuum_reduce_result(
        input_data,
        "continuum_finalize_node",
    )

    # sanity check whether products from previous cycles are present
    if "static_xds" not in input_params:
        raise KeyError("continuum_finalize_node requires input_params['static_xds'].")

    if "model_xds" not in input_params:
        raise KeyError("continuum_finalize_node requires input_params['model_xds'].")

    # shared functionality with the minor loop
    pb_cache_mapping = input_data.get(
        "pb_cache_mapping",
        input_params.get("pb_cache_mapping"),
    )

    img_xds, static_xds, _ = _prepare_continuum_image(
        input_data["image"],
        input_params,
        initialize_static_products=False,
        pb_cache_mapping=pb_cache_mapping,
    )

    model_xds = input_params["model_xds"]

    if "SKY_MODEL" not in model_xds:
        raise KeyError("model_xds does not contain 'SKY_MODEL'.")

    if "SKY_MODEL" in img_xds:
        img_xds = img_xds.drop_vars("SKY_MODEL")

    img_xds["SKY_MODEL"] = model_xds["SKY_MODEL"].copy(deep=True)

    modify_data_groups_xds(
        img_xds,
        data_group_out_name="model",
        data_group_out={
            "sky": "SKY_MODEL",
        },
        description="Accumulated continuum model installed for final restoration.",
    )

    # restore
    if input_params.get("restore", False):
        from astroviper.processing_functions.imaging.image_continuum_single_field import (
            restore_image,
        )

        img_xds, restore_timing_df = restore_image(
            img_xds,
            image_data_group_in_residual_name=(
                input_params.get(
                    "image_data_group_in_name",
                    "residual",
                )
            ),
            image_data_group_in_model_name="model",
            image_data_group_out_restore_name="restored",
            processing_function_threads=input_params.get(
                "processing_function_threads",
                1,
            ),
        )

        if input_params.get("pbcor", False):
            from astroviper.processing_functions.imaging.image_continuum_single_field import (
                primary_beam_correct_restored_continuum,
            )

            img_xds = primary_beam_correct_restored_continuum(
                img_xds,
                pblimit=input_params.get("pblimit", 0.2),
                primary_beam_name="PRIMARY_BEAM",
                restored_data_group_name="restored",
                output_data_group_name="restored_pbcor",
                output_variable_name="SKY_RESTORED_PBCOR",
            )

        input_data["timing_restore"] = restore_timing_df.reset_index(drop=True)

    else:
        input_data["timing_restore"] = None

    input_data["image"] = img_xds
    input_data["static_xds"] = static_xds

    # Preserve the final accumulated model as a separate driver-visible
    # state object as well as installing it in the output image.
    input_data["model_xds"] = model_xds

    return input_data


@shares_param_docs
def model_update_continuum_single_field(
    input_data,
    input_params,
):
    """Run one continuum model-update node entirely in memory.

    This node is intended for the GraphViper append stage following the global
    reduce operation. It receives the globally accumulated continuum image
    products, determines the minor-cycle parameters using the iteration
    controller, executes one continuum model-update step, and returns the updated
    image dataset in memory.

    No image or processing-set data are read from or written to disk.

    reduce output dictionary
    │
    ├── input_data["image"]
    └── input_data["timing_node_tasks"]
            │
            ▼
    model_update_continuum_single_field
            │
            ├── updates the iteration controller
            ├── determines the minor-cycle parameters
            ├── calls model_update_cycle_mtmfs_single_field
            ├── updates the continuum sky model
            └── updates convergence information
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

        ``input_data["timing_node_tasks"]``
            Timing information accumulated during the distributed map/reduce
            stages.

    input_params : dict
        Append-node configuration. Expected entries are

        ``iteration_control_params``
            Global CLEAN iteration-control parameters.

        Optional entries are

        ``deconvolver``
            Deconvolver name. Defaults to ``"hogbom"``.

        ``processing_function_threads``
            Number of threads passed to the model-update processing function.

        ``is_n_iter_0``
            Whether this is the first major cycle. Defaults to ``True``.

        ``controller``
            Existing :class:`IterationController`. If omitted, a new controller is
            constructed from ``iteration_control_params``.

        ``image_data_group_in_name``
            Residual image data-group name. Defaults to ``"residual"``.

        ``image_data_group_out_name``
            Model image data-group name. Defaults to ``"model"``.

    Returns
    -------
    dict
        Dictionary containing

        ``"image"``
            Updated continuum image dataset containing the new sky model.

        ``"timing_node_tasks"``
            Timing dataframe including the model-update stage.

        ``"controller"``
            Updated iteration controller.

        ``"converged"``
            Boolean indicating whether the imaging has converged.
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        model_update_mtmfs_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
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

    # Sanity checks
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

    # Prepare control parameters
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
    # Obtain the iteration controller.
    # -------------------------------------------------------------
    controller = input_params.get("controller")

    # A fresh map/reduce graph does not contain the preceding append node's
    # convergence history. Accept that persistent state through input_params so
    # later cycle controls retain the measured peak and PSF sidelobe.
    combined_deconvolve_dict = input_data.get(
        "deconvolution",
        input_params.get("deconvolution"),
    )

    if combined_deconvolve_dict is None:
        combined_deconvolve_dict = ReturnDict()

    timing = {
        "T_iteration_control": 0.0,
        "T_model_update": 0.0,
        "T_convergence": 0.0,
    }

    # A dirty-image request still passes through the append node so static
    # products and model state are prepared consistently, but it must not call
    # a deconvolver whose contract requires a positive iteration count.
    if int(iteration_control_params["niter"]) == 0:
        import xarray as xr

        from astroviper.processing_functions.imaging.utils.iteration_control import (
            MAJOR_ITER_LIMIT,
            MAJOR_STOPCODE_DESCRIPTIONS,
            MINOR_CONTINUE,
            StopCode,
        )

        if "SKY_MODEL" not in img_xds:
            img_xds["SKY_MODEL"] = xr.zeros_like(img_xds["SKY_RESIDUAL"])
        img_xds.attrs.setdefault("data_groups", {})[image_data_group_out_name] = {
            "sky": "SKY_MODEL"
        }

        controller.ensure_planes(
            img_xds.sizes["time"],
            1,
            img_xds.sizes["polarization"],
        )
        controller.niter[...] = 0
        controller.stopcode_major[...] = MAJOR_ITER_LIMIT
        controller.stopcode_minor[...] = MINOR_CONTINUE
        stopcode = StopCode(MAJOR_ITER_LIMIT, MINOR_CONTINUE)
        stopdesc = MAJOR_STOPCODE_DESCRIPTIONS[MAJOR_ITER_LIMIT]
        controller.stopcode = stopcode
        controller.stopdescription = stopdesc

        timing["T_model_update_node_task"] = time.time() - node_start
        return {
            "image": img_xds,
            "timing_node_tasks": input_data.get("timing_node_tasks"),
            "timing_model_update": pd.DataFrame(
                {key: [value] for key, value in timing.items()}
            ),
            "deconvolution": combined_deconvolve_dict,
            "controller": controller,
            "stopcode": stopcode,
            "stopdesc": stopdesc,
            "is_n_iter_0": False,
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

    # placeholder for mtmfs
    (deconvolve_dict, model_update_return_df,) = model_update_mtmfs_single_field(
        img_xds,
        deconvolver,
        deconvolve_params,
        is_n_iter_0=is_n_iter_0,
        processing_function_threads=processing_function_threads,
        image_data_group_in_name=image_data_group_in_name,
        image_data_group_out_name=image_data_group_out_name,
    )

    timing["T_model_update"] = time.time() - start

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
