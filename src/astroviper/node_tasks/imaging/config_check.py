"""Read only the visibility metadata needed for imaging preflight."""


def preflight_imaging_inputs(processing_set, config, *, mode):
    """Check actual selected imaging weights before creating output or graphs.

    Parameters
    ----------
    processing_set : xarray.DataTree or mapping
        Lazy input Measurement Sets. Visibility data are never loaded here.
    config : mapping
        Validated image, weighting, polarization, and data-group configuration.
    mode : ImagingMode
        Imaging kind and field layout.

    Returns
    -------
    dict
        Spectral-design diagnostics based on the same weight kernels as imaging.

    Notes
    -----
    Weight/flag/UVW data are read in bounded time blocks. Global Briggs uses
    two passes over the selected metadata; local Briggs processes one frequency
    at a time so the density grid never grows with the channel count.
    """
    import numpy as np

    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        _equalize_parallel_hand_weights,
    )
    from astroviper.processing_functions.imaging.config_check._common import (
        imaging_weighting_scope,
        selected_frequency_mask,
    )
    from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
        check_input_metadata,
        check_spectral_sampling,
    )
    from astroviper.processing_functions.imaging.imaging_weighting.briggs_weighting import (
        calculate_briggs_params,
    )
    from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights import (
        degrid_imaging_weights,
        grid_imaging_weights,
    )
    from astroviper.processing_functions.imaging.utils.imaging_polarization import (
        correlation_selection,
        correlations_for_stokes,
    )

    image_params = config["image_params"]
    weighting_params = config["imaging_weights_params"]
    requested_frequencies = np.asarray(image_params["frequency_coords"])
    correlation_coords = correlations_for_stokes(
        image_params["polarization_coords"],
        config.get("instrument_polarization_basis", "linear"),
    )
    group_name = config.get("processing_set_data_group_name", "corrected")
    weighting = weighting_params.get("weighting", "natural")
    scope = imaging_weighting_scope(weighting_params, mode=mode)
    truncate_uv_cells = mode.imaging_kind == "continuum"
    image_size = np.asarray(image_params["image_size"], dtype=np.int64)
    cell_size = np.asarray(image_params["cell_size"], dtype=np.float64)
    threads = config.get("processing_function_threads", 1)
    samples = {}
    longest_projected_baselines = {}
    needs_spatial_preflight = (
        mode.imaging_kind == "continuum"
        and config.get("deconvolver", "hogbom") == "mtmfs"
    )
    fields = set()
    spectral_windows = set()

    # --- Selection and coordinate metadata ---
    # Resolve channels, correlations, fields, and spectral windows once, without
    # loading visibility, weight, flag, or UVW arrays.
    selected_inputs = []
    for ms_name, child in processing_set.items():
        source = child.ds if hasattr(child, "ds") else child
        group = source.attrs.get("data_groups", {}).get(group_name)
        if group is None:
            raise KeyError(
                f"Preflight: data group {group_name!r} is missing from {ms_name!r}."
            )
        for role in ("weight", "flag", "uvw"):
            if role not in group or group[role] not in source:
                raise KeyError(
                    f"Preflight: {ms_name!r} data group {group_name!r} lacks {role!r}."
                )
        units = source.frequency.attrs.get("units", ["Hz"])
        if set(np.atleast_1d(units)) != {"Hz"}:
            raise ValueError(
                f"Selected frequencies in {ms_name!r} must be expressed in Hz; received units={units!r}."
            )
        frequencies = np.asarray(source.frequency.values)
        selected = selected_frequency_mask(frequencies, requested_frequencies)
        indices = np.flatnonzero(selected)
        if not indices.size:
            continue
        # Inspect role targets and dimensions before reading any weight blocks.
        # Existing guards here and in the processing functions remain active.
        check_input_metadata(source, config, mode=mode, measurement_set=ms_name)
        pol_selection = correlation_selection(
            source.polarization.values, correlation_coords, ms_name
        )
        partition_info = (
            child.xr_ms.get_partition_info(data_group_name=group_name)
            if hasattr(child, "xr_ms")
            else source.attrs.get("partition_info", {})
        )
        for name in np.atleast_1d(partition_info.get("field_name", [])):
            fields.add(str(name))
        for name in np.atleast_1d(
            partition_info.get(
                "spectral_window_name", partition_info.get("spectral_window_id", [])
            )
        ):
            spectral_windows.add(str(name))
        selected_inputs.append(
            (ms_name, source, group, indices, pol_selection, frequencies)
        )

    # --- Single-field contract ---
    # Reject multiple selected fields before reading any weight or flag arrays.
    if len(fields) > 1:
        raise ValueError(
            f"Single-field imaging selected multiple fields: {sorted(fields)}."
        )

    # --- Selected metadata blocks ---
    # Keep I/O here in the node layer. Each pass reads fresh bounded blocks;
    # flagging and equalization never modify the source Processing Set.
    def blocks(selected_frequency=None):
        for (
            ms_name,
            source,
            group,
            indices,
            pol_selection,
            frequencies,
        ) in selected_inputs:
            if selected_frequency is not None:
                indices = indices[frequencies[indices] == selected_frequency]
            if not indices.size:
                continue
            selection = {"frequency": indices}
            if pol_selection is not None:
                selection["polarization"] = pol_selection
            selected_source = source.isel(selection)
            values_per_time = max(
                1,
                selected_source.sizes["baseline_id"]
                * len(indices)
                * len(correlation_coords),
            )
            times_per_block = max(1, (64 * 1024**2) // (values_per_time * 24))
            for start in range(0, source.sizes["time"], times_per_block):
                block = selected_source.isel(time=slice(start, start + times_per_block))
                raw_weights = np.array(
                    block[group["weight"]]
                    .transpose("time", "baseline_id", "frequency", "polarization")
                    .values,
                    copy=True,
                )
                flags = np.asarray(
                    block[group["flag"]]
                    .transpose("time", "baseline_id", "frequency", "polarization")
                    .values
                )
                uvw = np.ascontiguousarray(block[group["uvw"]].values, dtype=np.float64)

                # XRADIO's sparse time/baseline grid contains absent rows.
                # All three metadata markers must agree before excluding one;
                # a real row with corrupt weights or UVW must still fail.
                absent_rows = np.zeros(raw_weights.shape[:2], dtype=bool)
                if "TIME_CENTROID" in block:
                    absent_rows = (
                        ~np.isfinite(
                            block.TIME_CENTROID.transpose("time", "baseline_id").values
                        )
                        & np.isnan(uvw).all(axis=-1)
                        & np.isnan(raw_weights).all(axis=(2, 3))
                    )
                sample_flagged = np.any(flags != 0, axis=-1, keepdims=True)
                sample_flagged |= absent_rows[:, :, None, None]
                if np.any(absent_rows):
                    uvw = uvw.copy()
                    uvw[absent_rows] = 0.0  # No sample contributes at these rows.
                active = np.broadcast_to(~sample_flagged, raw_weights.shape)
                if np.any((~np.isfinite(raw_weights) | (raw_weights < 0)) & active):
                    raise ValueError(
                        f"Spectral-design preflight: unflagged weights must be finite and non-negative; field={sorted(fields)}, spectral_window={sorted(spectral_windows)}, measurement_set={ms_name!r}, channel_selection={indices.tolist()}, polarization={correlation_coords}, nterms={image_params.get('n_taylor_terms', 1)}, reference_frequency={image_params.get('reference_frequency')}, usable/distinct/rank/condition=unavailable."
                    )
                raw_weights[np.broadcast_to(sample_flagged, raw_weights.shape)] = np.nan
                equalized = _equalize_parallel_hand_weights(
                    raw_weights,
                    weighting_params.get("casa_weighting_implementation", False),
                )
                if not np.all(np.isfinite(uvw)):
                    raise ValueError(
                        f"Spectral-design preflight: non-finite UVW coordinates in {ms_name!r}."
                    )
                baselines = np.ones(block.sizes["baseline_id"], dtype=bool)
                if (
                    "baseline_antenna1_name" in block
                    and "baseline_antenna2_name" in block
                ):
                    baselines = np.asarray(
                        block.baseline_antenna1_name != block.baseline_antenna2_name
                    )
                yield (
                    ms_name,
                    indices,
                    frequencies[indices],
                    uvw,
                    np.ascontiguousarray(equalized, dtype=np.float64),
                    sample_flagged,
                    baselines,
                )

    # --- Effective-weight accumulation ---
    # Flagged samples and auto-correlations do not contribute Taylor constraints.
    def accumulate(block, effective_weights):
        ms_name, indices, frequencies, uvw, _, flagged, baselines = block
        active = np.broadcast_to(~flagged, effective_weights.shape).copy()
        active[:, ~baselines, :, :] = False
        if np.any((~np.isfinite(effective_weights) | (effective_weights < 0)) & active):
            raise ValueError(
                f"Spectral-design preflight: negative or non-finite effective imaging weights in {ms_name!r}; channel_selection={indices.tolist()}, polarization={correlation_coords}."
            )
        channel_weights = np.where(active, effective_weights, 0).sum(
            axis=(0, 1, 3), dtype=np.float64
        )
        # Gaussian preflight uses resolution lambda / Bmax at each frequency.
        # Accumulate only usable projected baselines in the existing metadata
        # pass; flagged, zero-weight, and autocorrelation samples cannot define
        # the resolution. No visibility reads or additional pass are needed.
        if needs_spatial_preflight:
            positive_samples = np.any(active & (effective_weights > 0), axis=-1)
            projected_lengths = np.hypot(uvw[:, :, 0], uvw[:, :, 1])
            channel_baselines = np.where(
                positive_samples, projected_lengths[:, :, None], 0
            ).max(axis=(0, 1))
            for frequency, longest_baseline in zip(
                frequencies, channel_baselines, strict=True
            ):
                longest_projected_baselines[float(frequency)] = max(
                    longest_projected_baselines.get(float(frequency), 0.0),
                    float(longest_baseline),
                )
        for index, frequency, weight in zip(
            indices, frequencies, channel_weights, strict=True
        ):
            key = (ms_name, int(index), float(frequency))
            samples[key] = samples.get(key, 0.0) + float(weight)

    # --- Natural weighting ---
    # This is exactly the flagged, equalized statistical weight used by imaging.
    if weighting == "natural":
        for block in blocks():
            accumulate(block, block[4])

    # --- Briggs and uniform weighting ---
    # Use the production gridding/degridding kernels. The same recipe handles
    # one global density plane or independent per-frequency density planes.
    else:
        density_frequencies = (
            [None] if scope == "global" else np.unique(requested_frequencies)
        )
        for selected_frequency in density_frequencies:
            density = np.zeros((1, 1, *image_size), dtype=np.float64)
            sum_weight = np.zeros((1, 1), dtype=np.float64)
            for block in blocks(selected_frequency):
                _, _, frequencies, uvw, weights, _, _ = block
                grid_imaging_weights(
                    density,
                    sum_weight,
                    uvw,
                    weights,
                    frequencies,
                    image_size,
                    cell_size,
                    processing_function_threads=threads,
                    frequency_map=np.zeros(len(frequencies), dtype=np.int64),
                    truncate_uv_cells=truncate_uv_cells,
                )
            if not np.any(sum_weight > 0):
                # Preserve zero-weight channels in the final rank diagnostic.
                for block in blocks(selected_frequency):
                    accumulate(block, np.zeros_like(block[4]))
                continue
            factors = calculate_briggs_params(density, sum_weight, weighting_params)
            for block in blocks(selected_frequency):
                _, _, frequencies, uvw, weights, _, _ = block
                effective = degrid_imaging_weights(
                    density,
                    uvw,
                    weights,
                    factors,
                    frequencies,
                    image_size,
                    cell_size,
                    processing_function_threads=threads,
                    frequency_map=np.zeros(len(frequencies), dtype=np.int64),
                    truncate_uv_cells=truncate_uv_cells,
                )
                accumulate(block, effective)

    if not samples:
        raise ValueError("Spectral-design preflight: no selected channels have data.")
    context = {
        "field": sorted(fields),
        "spectral_window": sorted(spectral_windows),
        "channel_selection": [[name, index] for name, index, _ in samples],
        "polarization": image_params["polarization_coords"],
    }
    frequencies = np.array([frequency for _, _, frequency in samples])
    weights = np.array(list(samples.values()))
    precision = "float32" if config.get("single_precision_image", True) else "float64"
    if mode.imaging_kind == "continuum":
        diagnostics = check_spectral_sampling(
            frequencies,
            weights,
            n_taylor_terms=image_params["n_taylor_terms"],
            reference_frequency=image_params["reference_frequency"],
            precision=precision,
            context=context,
        )
        # The driver calls this before output creation or graph execution.
        # Reuse the same selected metadata for Gaussian spatial PSF blocks;
        # the append node will still validate the actual PSF-derived Hessians.
        if config.get("deconvolver", "hogbom") == "mtmfs":
            from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
                check_taylor_hessian_approximation,
            )

            diagnostics["taylor_hessian_approximation"] = (
                check_taylor_hessian_approximation(
                    frequencies,
                    weights,
                    n_taylor_terms=image_params["n_taylor_terms"],
                    reference_frequency=image_params["reference_frequency"],
                    longest_projected_baselines=np.array(
                        [
                            longest_projected_baselines[float(frequency)]
                            for frequency in frequencies
                        ]
                    ),
                    image_size=image_params["image_size"],
                    cell_size=image_params["cell_size"],
                    precision=precision,
                    context=context,
                )
            )
        return diagnostics

    # --- Cube channel usability ---
    # Each independently imaged channel needs at least one usable sample.
    diagnostics = []
    for frequency in requested_frequencies:
        selected = np.isclose(frequencies, frequency, rtol=1e-12, atol=0)
        diagnostics.append(
            check_spectral_sampling(
                frequencies[selected],
                weights[selected],
                n_taylor_terms=1,
                reference_frequency=float(frequency),
                precision=precision,
                context={
                    **context,
                    "channel_frequency_hz": float(frequency),
                    "channel_selection": [
                        [name, index]
                        for (name, index, _), matches in zip(
                            samples, selected, strict=True
                        )
                        if matches
                    ],
                },
            )
        )
    return {"cube_channel_sampling": diagnostics}
