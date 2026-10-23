"""Check workflow capabilities and output-policy relationships."""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    require_boolean,
    require_finite_real,
    require_integer,
    require_mapping,
    selected_frequency_mask,
)
from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def check_driver_workflow(config, *, mode, require_complete=False):
    """Validate workflow settings without modifying the configuration.

    Parameters
    ----------
    config : mapping
        Workflow configuration. Only supplied keys are checked. Residual-update
        and model-update settings are checked by their separate validators.
    mode : ImagingMode
        Imaging kind (cube or continuum) and field layout.

    require_complete : bool, optional
        Require all coordinates and cycle controls needed by a distributed driver.
        Lower-level entry points check only the configuration they receive.

    Raises
    ------
    TypeError
        If the primary-beam limit is not a real scalar.
    ValueError
        If memory mode, backend, or output policy is invalid.
    NotImplementedError
        If multifield imaging is requested.
    """
    # --- Supported imaging mode ---
    # Reject multifield requests before creating outputs or constructing a graph.
    mode.require_supported()
    require_mapping("config", config)

    # --- Processing-set memory mode ---
    # Both current imaging workflows require the general in-memory path.
    if "memory_mode" in config and config["memory_mode"] != "in_memory":
        raise ValueError("Only memory_mode='in_memory' is implemented.")

    # --- Continuum compute backend ---
    # Continuum graph execution currently implements Dask only.
    if (
        mode.imaging_kind == "continuum"
        and "compute_backend" in config
        and config["compute_backend"] != "dask"
    ):
        raise ValueError("Continuum imaging supports only compute_backend='dask'.")

    # --- Restoration and primary-beam correction ---
    # The corrected output is a restored image, so restoration must be enabled.
    primary_beam_correction = config.get("primary_beam_correction", False)
    restore = config.get("restore", False)
    if primary_beam_correction and not restore:
        raise ValueError("primary_beam_correction requires restore=True.")

    # --- Primary-beam cutoff ---
    # The fractional cutoff must be one finite value in the interval [0, 1).
    if "primary_beam_limit" in config:
        primary_beam_limit = require_finite_real(
            "primary_beam_limit", config["primary_beam_limit"], scalar=True
        )
        if not 0 <= primary_beam_limit < 1:
            raise ValueError(
                "primary_beam_limit must satisfy 0 <= primary_beam_limit < 1."
            )

    # --- Required configuration groups ---
    # A distributed driver must not defer missing input groups to a worker.
    if require_complete:
        for name in ("image_params", "imaging_weights_params", "cycle_control_params"):
            if name not in config:
                raise KeyError(f"Imaging configuration must contain {name!r}.")
            require_mapping(name, config[name])

    # --- Boolean switches ---
    # False-like strings and integers must not silently enable a different path.
    for name in (
        "input_checked",
        "single_precision_image",
        "restore",
        "primary_beam_correction",
        "skunk_works",
        "overwrite",
        "clear_cache",
        "vizualize_graph",
        "write_visibility_model_to_ps",
        "write_imaging_weights_to_ps",
    ):
        if name in config:
            require_boolean(name, config[name])

    # --- Supported graph backend ---
    # Cube supports Dask and MPI; continuum is restricted to its Dask path above.
    if "compute_backend" in config and config["compute_backend"] not in ("dask", "mpi"):
        raise ValueError("compute_backend must be 'dask' or 'mpi'.")
    if config.get("mpi_cluster_setup") is not None and (
        mode.imaging_kind == "continuum"
        or config.get("compute_backend", "dask") != "mpi"
    ):
        raise ValueError("mpi_cluster_setup requires the implemented cube MPI backend.")

    # --- Unsupported interface options ---
    # Reject accepted-but-unused options until their implementation is available.
    if config.get("field_name") is not None:
        raise ValueError(
            "field_name selection is not implemented; select a field when preparing the input Processing Set."
        )
    if config.get("vizualize_graph", False) and (
        mode.imaging_kind == "continuum" or config.get("compute_backend") == "mpi"
    ):
        raise ValueError(
            "vizualize_graph is not implemented for this imaging mode/backend."
        )
    if mode.imaging_kind == "continuum" and config.get(
        "write_visibility_model_to_ps", False
    ):
        raise ValueError("Continuum write_visibility_model_to_ps is not implemented.")

    # --- Continuum cache policies ---
    # Only these storage policies have node-task implementations.
    if mode.imaging_kind == "continuum":
        policies = {
            "weight_memory_mode": ("in_memory", "in_place"),
            "visibility_memory_mode": ("in_memory", "in_place", "recompute"),
            "wideband_primary_beam_memory_mode": ("in_memory", "in_place", "recompute"),
        }
        for name, allowed in policies.items():
            if name in config and config[name] not in allowed:
                raise ValueError(f"{name} must be one of {allowed}.")
        if config.get("weight_memory_mode", "in_place") == "in_place" and config.get(
            "skunk_works", False
        ):
            raise ValueError(
                "weight_memory_mode='in_place' is not supported with skunk_works=True."
            )
        if (
            config.get("write_imaging_weights_to_ps", False)
            and config.get("weight_memory_mode", "in_place") != "in_place"
        ):
            raise ValueError(
                "write_imaging_weights_to_ps requires weight_memory_mode='in_place'."
            )
        if (
            config.get("image_chunking") is not None
            or config.get("image_sharding") is not None
        ):
            raise ValueError(
                "Continuum image_chunking and image_sharding are not implemented."
            )

    # --- Cube write-back policies ---
    # The implemented in-memory cube path does not write model/weights to input.
    if mode.imaging_kind == "cube":
        for name in ("write_visibility_model_to_ps", "write_imaging_weights_to_ps"):
            if config.get(name, False):
                raise ValueError(
                    f"{name} is not implemented for the in-memory cube workflow."
                )

    # --- Concurrent sharded output ---
    # Partial shards need the direct writer; the standard writer cannot share them.
    if config.get("image_sharding") and not config.get("skunk_works", False):
        raise ValueError("image_sharding requires skunk_works=True.")

    # --- Output format and layout ---
    # FITS requires the direct cube writer and cannot use Zarr chunk/shard options.
    output_format = config.get("output_image_format", "zarr")
    if output_format not in ("zarr", "fits"):
        raise ValueError("output_image_format must be 'zarr' or 'fits'.")
    if output_format == "fits":
        if mode.imaging_kind != "cube" or not config.get("skunk_works", False):
            raise ValueError("FITS output requires cube imaging with skunk_works=True.")
        if config.get("image_chunking") or config.get("image_sharding"):
            raise ValueError(
                "FITS output does not support image_chunking or image_sharding."
            )

    # --- Output representation ---
    # Check known coordinates, variable roles and codecs before creating files.
    from astroviper.processing_functions.imaging.config_check._output import (
        check_fits_coordinates,
        check_kept_variables,
        check_output_codec,
        check_output_layout,
    )

    image_params = require_mapping("image_params", config.get("image_params", {}))
    if output_format == "fits":
        check_fits_coordinates(image_params)
    check_kept_variables(config, fits=output_format == "fits")
    check_output_layout(config)
    check_output_codec(config)

    # --- Measurement Set opening options ---
    # These keys are owned by the reader; an override would be ineffective.
    if config.get("msv2_open_options") is not None:
        options = require_mapping("msv2_open_options", config["msv2_open_options"])
        reserved = set(options) & {"array_backend", "scan_intents"}
        if reserved:
            raise ValueError(f"msv2_open_options may not set {sorted(reserved)}.")

    # --- Graph partition count ---
    # Frequency is the only implemented distributed imaging dimension.
    if config.get("n_mapping_parallelism") is not None:
        parallelism = require_mapping(
            "n_mapping_parallelism", config["n_mapping_parallelism"]
        )
        if set(parallelism) != {"frequency"}:
            raise ValueError("n_mapping_parallelism must contain only 'frequency'.")
        if parallelism["frequency"] is not None:
            require_integer(
                "n_mapping_parallelism['frequency']",
                parallelism["frequency"],
                minimum=1,
            )

    # --- Threading, reduction, and resource limits ---
    # Reject invalid scalar limits before a graph or worker is started.
    if "processing_function_threads" in config:
        require_integer(
            "processing_function_threads",
            config["processing_function_threads"],
            minimum=1,
        )
    if "reduce_mode" in config and config["reduce_mode"] not in (
        "tree",
        "single_node",
        "tree_n",
    ):
        raise ValueError("reduce_mode must be 'tree', 'single_node', or 'tree_n'.")
    if "reduce_n_batch" in config:
        require_integer("reduce_n_batch", config["reduce_n_batch"], minimum=2)
    for name in ("task_time_kill_switch_seconds", "monitor_resources_seconds"):
        if config.get(name) is not None:
            value = require_finite_real(name, config[name], scalar=True)
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
    if (
        "disk_chunk_sizes" in config
        and isinstance(config["disk_chunk_sizes"], str)
        and config["disk_chunk_sizes"] != "Auto"
    ):
        raise ValueError("disk_chunk_sizes must be 'Auto', a mapping, or None.")

    # --- Shared primary-beam limit ---
    # A per-cycle copy must not silently override the public application limit.
    controls = config.get("cycle_control_params", {})
    if "primary_beam_limit" in config and "primary_beam_limit" in controls:
        if config["primary_beam_limit"] != controls["primary_beam_limit"]:
            raise ValueError(
                "cycle_control_params['primary_beam_limit'] conflicts with the application primary_beam_limit."
            )


def check_channel_partitions(mapping, processing_set, *, image_params):
    """Require exclusive and complete channel coverage before graph construction.

    Parameters
    ----------
    mapping : mapping
        GraphVIPER task mapping containing each child's positional selection.
    processing_set : mapping
        Selected Measurement Set datasets/DataTrees supplying frequency axes.
    image_params : mapping
        The common image frequency axis and, for continuum, Taylor basis.
    """

    # --- Intended selection ---
    # Image frequencies define which original channel indices must appear once.
    image_frequencies = np.asarray(image_params["frequency_coords"])
    expected = {}
    counts = {}
    for name, child in processing_set.items():
        frequencies = np.asarray(child.frequency.values)
        selected = selected_frequency_mask(frequencies, image_frequencies)
        expected[name] = selected
        counts[name] = np.zeros(len(frequencies), dtype=int)

    # --- Exclusive task coverage ---
    # Repeated physical frequencies in different MSes are distinct samples;
    # repeating the same original channel index in two tasks is an error.
    for task_id, task in mapping.items():
        if "image_params" in task:
            check_configuration_consistency(
                image_params,
                task["image_params"],
                names=("n_taylor_terms", "reference_frequency"),
            )
        for name, selection in task.get("data_selection", {}).items():
            if selection is None:
                continue
            if name not in counts:
                raise ValueError(
                    f"Channel partition {task_id} refers to unknown Measurement Set {name!r}."
                )
            indices = np.arange(len(counts[name]))[
                selection.get("frequency", slice(None))
            ]
            np.add.at(counts[name], np.atleast_1d(indices), 1)

    # --- Complete coverage ---
    # Neither missing channels, repeated channels, nor extra channels are allowed.
    for name in counts:
        if not np.array_equal(counts[name], expected[name].astype(int)):
            missing = np.flatnonzero(expected[name] & (counts[name] == 0)).tolist()
            repeated = np.flatnonzero(counts[name] > 1).tolist()
            extra = np.flatnonzero(~expected[name] & (counts[name] > 0)).tolist()
            raise ValueError(
                f"Channel partition preflight failed for {name!r}: missing={missing}, repeated={repeated}, extra={extra}."
            )


def check_configuration_consistency(reference, candidate, *, names=None):
    """Reject changes to shared configuration across workflow stages.

    Parameters
    ----------
    reference, candidate : mapping
        Configurations at two stage boundaries. Inputs are not modified.
    names : iterable of str, optional
        Keys to compare; otherwise compare the shared imaging configuration.
    """

    if names is None:
        names = (
            "image_params",
            "imaging_weights_params",
            "cycle_control_params",
            "deconvolver",
            "single_precision_image",
            "continuum_mode",
        )
    for name in names:
        if (name in reference) != (name in candidate):
            raise ValueError(f"Workflow configuration changed presence of {name!r}.")
        if name not in reference:
            continue
        first, second = reference[name], candidate[name]
        if isinstance(first, dict) and isinstance(second, dict):
            check_configuration_consistency(
                first, second, names=set(first) | set(second)
            )
        elif not np.array_equal(first, second):
            raise ValueError(f"Workflow configuration changed {name!r} between stages.")
