"""Inspect selected imaging coordinates and structure without reading sample arrays."""


def preflight_imaging_inputs(processing_set, config, *, mode):
    """Check nominal spectral support using metadata and equal sample weights.

    Parameters
    ----------
    processing_set : xarray.DataTree or mapping
        Lazy input Measurement Sets. Only coordinates, attributes, shapes, and
        dtypes are inspected; visibility, weight, flag, time-centroid, and UVW
        arrays remain unloaded.
    config : mapping
        Validated image, polarization, and data-group configuration.
    mode : ImagingMode
        Imaging kind and field layout.

    Returns
    -------
    dict
        Equal-weight spectral diagnostics, including their approximation and
        scope. Cube diagnostics describe each requested channel separately.

    Notes
    -----
    The approximation q_nu = 1 tests the nominal selected frequency coverage.
    It does not certify actual positive-weight support after flags, the final
    coefficient errors, or a PSF Hessian. No density gridding or spatial-PSF
    approximation is performed here. Data-dependent checks require arrays
    produced or loaded by the imaging tasks.
    """
    import numpy as np

    from astroviper.processing_functions.imaging.config_check._common import (
        selected_frequency_mask,
    )
    from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
        check_input_metadata,
        check_selected_fields,
        check_selected_samples,
        check_spectral_sampling,
    )
    from astroviper.processing_functions.imaging.utils.imaging_polarization import (
        correlation_selection,
        correlations_for_stokes,
    )

    image_params = config["image_params"]
    requested_frequencies = np.asarray(image_params["frequency_coords"])
    correlation_coords = correlations_for_stokes(
        image_params["polarization_coords"],
        config.get("instrument_polarization_basis", "linear"),
    )
    group_name = config.get("processing_set_data_group_name", "corrected")
    fields = set()
    spectral_windows = set()
    samples = []
    output_channels = []

    # --- Input structure and selected frequency coordinates ---
    # Channel mapping follows imaging's coordinate selection. Array values are
    # irrelevant to this equal-weight approximation and must remain lazy.
    for ms_name, child in processing_set.items():
        source = child.ds if hasattr(child, "ds") else child
        check_input_metadata(source, config, mode=mode, measurement_set=ms_name)
        frequencies = np.asarray(source.frequency.values)
        if mode.imaging_kind == "cube":
            from graphviper.graph_tools.coordinate_utils import (
                map_to_nearest_coordinate,
            )

            image_channels = map_to_nearest_coordinate(
                frequencies, requested_frequencies
            )
            indices = np.flatnonzero(image_channels >= 0)
            output_channels.extend(image_channels[indices].tolist())
        else:
            indices = np.flatnonzero(
                selected_frequency_mask(frequencies, requested_frequencies, mode=mode)
            )
        if not indices.size:
            continue
        correlation_selection(source.polarization.values, correlation_coords, ms_name)
        partition_info = (
            child.xr_ms.get_partition_info(data_group_name=group_name)
            if hasattr(child, "xr_ms")
            else source.attrs.get("partition_info", {})
        )
        fields.update(
            str(name) for name in np.atleast_1d(partition_info.get("field_name", []))
        )
        spectral_windows.update(
            str(name)
            for name in np.atleast_1d(
                partition_info.get(
                    "spectral_window_name", partition_info.get("spectral_window_id", [])
                )
            )
        )
        samples.extend(
            (ms_name, int(index), float(frequencies[index])) for index in indices
        )

    # --- Selection contract ---
    # Missing coordinates and mixed fields are known without inspecting flags.
    check_selected_fields(fields, mode=mode)
    check_selected_samples(samples)
    context = {
        "field": sorted(fields),
        "spectral_window": sorted(spectral_windows),
        "channel_selection": [[name, index] for name, index, _ in samples],
        "polarization": np.asarray(image_params["polarization_coords"]).tolist(),
        "weight_assumption": "equal",
        "support_assumption": "selected_frequency_coordinates",
    }
    frequencies = np.array([frequency for _, _, frequency in samples])
    unit_weights = np.ones(frequencies.size, dtype=np.float64)
    precision = "float32" if config.get("single_precision_image", True) else "float64"

    # --- Equal-weight continuum approximation ---
    # The rank diagnostic is float64 regardless of requested image precision.
    if mode.imaging_kind == "continuum":
        return check_spectral_sampling(
            frequencies,
            unit_weights,
            n_taylor_terms=image_params["n_taylor_terms"],
            reference_frequency=image_params["reference_frequency"],
            precision=precision,
            context=context,
        )

    # --- Nominal cube channel coverage ---
    # A requested channel needs source coordinates assigned to it. This is not
    # a test for unflagged, positive-weight data, which metadata cannot establish.
    output_channels = np.asarray(output_channels)
    diagnostics = []
    for image_channel, frequency in enumerate(requested_frequencies):
        selected = output_channels == image_channel
        diagnostics.append(
            check_spectral_sampling(
                frequencies[selected],
                unit_weights[selected],
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
