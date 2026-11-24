"""XRADIO-owned staged imaging weights (the proposed #661 region-I/O API)."""

import numpy as np
import zarr


def prepare_imaging_weights(ps_xdt, store, data_group_name, task_mapping=None):
    """Preflight every child, then preallocate unregistered task-write targets.

    Existing products and registered roles are protected. Failed preparation
    removes only staging variables belonging to this run.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing Set metadata with full coordinates for every selected child.
    store : str
        Processing Set store to receive the staged arrays.
    data_group_name : str
        Input data group supplying the statistical weights.
    task_mapping : dict or None, optional
        Restrict targets to children selected by at least one graph task.
        None prepares every supplied child.

    Returns
    -------
    dict
        Small XRADIO transaction descriptors, keyed by Processing Set child.
    """
    from xradio import create_data_variable

    affected = (
        None
        if task_mapping is None
        else {
            name
            for task in task_mapping.values()
            for name, selection in task.get("data_selection", {}).items()
            if selection is not None
        }
    )
    specifications = {}
    for name, child in ps_xdt.items():
        if affected is not None and name not in affected:
            continue
        ds = child.ds
        group = ds.attrs["data_groups"][data_group_name]
        disk = zarr.open_group(store, path=name, mode="r", use_consolidated=False)
        if "WEIGHT_IMAGING" in disk or "weight_imaging" in group:
            raise FileExistsError(
                f"Imaging weights already exist in {name!r}; retain or remove them explicitly before this run"
            )
        source = disk[group["weight"]]
        dims = (
            tuple(source.metadata.dimension_names)
            if source.metadata.zarr_format == 3
            else tuple(source.attrs["_ARRAY_DIMENSIONS"])
        )
        chunks = list(source.chunks)
        chunks[dims.index("frequency")] = 1
        specifications[name] = dict(
            group_path=name,
            variable_name="WEIGHT_IMAGING",
            dimensions=dims,
            coordinates={dim: ds[dim].values for dim in dims},
            shape=source.shape,
            dtype="<f8",
            chunks=chunks,
            register=False,
            attributes=dict(source.attrs),
        )
    transactions = {}
    try:
        for name, spec in specifications.items():
            transactions[name] = create_data_variable(store, **spec)
    except BaseException:
        discard_imaging_weights(transactions)
        raise
    return transactions


def discard_imaging_weights(transactions):
    """Clean up only unpublished targets owned by this imaging run.

    Parameters
    ----------
    transactions : dict
        Descriptors returned by prepare_imaging_weights. Published products
        remain valid; repeated cleanup is harmless.
    """
    if not transactions:
        return
    from xradio import delete_data_variable

    for transaction in transactions.values():
        delete_data_variable(
            transaction["store"],
            group_path=transaction["group_path"],
            variable_name=transaction["variable_name"],
            transaction=transaction,
            require_unregistered=True,
        )


def publish_imaging_weights(transactions, data_group_name, provenance):
    """Register completed weights after all distributed task writes succeeded.

    Parameters
    ----------
    transactions : dict
        XRADIO staging descriptors belonging to the completed imaging graphs.
    data_group_name : str
        Group receiving the weight_imaging role.
    provenance : dict
        Imaging application and weighting parameters retained by XRADIO.
    """
    if not transactions:
        return
    from xradio import update_data_group

    for transaction in transactions.values():
        update_data_group(
            transaction["store"],
            group_path=transaction["group_path"],
            data_group_name=data_group_name,
            roles={"weight_imaging": "WEIGHT_IMAGING"},
            description="AstroVIPER continuum imaging weights retained by request.",
            transaction=transaction,
            complete=True,
            provenance=provenance,
        )


def weight_region_selection(dataset, transaction):
    """Resolve selected labels to indices without reading the full weight array.

    Parameters
    ----------
    dataset : xarray.Dataset
        Task-local visibility partition, with selected coordinate labels.
    transaction : dict
        XRADIO descriptor identifying the target child and variable dimensions.

    Returns
    -------
    dict
        Orthogonal integer indexers for XRADIO region reads and writes.
    """
    disk = zarr.open_group(
        transaction["store"],
        path=transaction["group_path"],
        mode="r",
        use_consolidated=False,
    )
    selection = {}
    for dim in transaction["dimensions"]:
        full = np.asarray(disk[dim][:])
        selected = np.asarray(dataset[dim].values)
        # Preserve arbitrary coordinate order, including correlation subsets.
        indices = {value: index for index, value in enumerate(full.tolist())}
        if len(indices) != full.size:
            raise ValueError(f"Persisted coordinate {dim!r} must be unique")
        try:
            selection[dim] = np.array(
                [indices[value] for value in selected.tolist()], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError(
                f"Selected {dim!r} coordinate is absent from persisted weights"
            ) from exc
    return selection


def imaging_weight_provenance(weighting, image_geometry, task_mapping, processing_set):
    """Describe the actual weighting configuration and selected visibility regions.

    Parameters
    ----------
    weighting : dict
        Resolved imaging-weighting options.
    image_geometry : dict
        Image geometry used for density gridding and channel mapping.
    task_mapping : dict
        Exact graph selections, indexed into the recorded source dimensions.
    processing_set : xarray.DataTree
        Selected Processing Set child metadata.

    Returns
    -------
    dict
        JSON-compatible provenance, including slice bounds and array indexers.
    """

    def encode(value):
        if isinstance(value, slice):
            return {"start": value.start, "stop": value.stop, "step": value.step}
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [encode(item) for item in value]
        if isinstance(value, np.ndarray):
            return encode(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value

    return encode(
        {
            "application": "image_continuum_single_field",
            "imaging_weights_params": weighting,
            "image_geometry": {
                key: image_geometry[key]
                for key in (
                    "image_size",
                    "cell_size",
                    "fft_padding",
                    "frequency_coords",
                    "reference_frequency",
                    "polarization_coords",
                )
                if key in image_geometry
            },
            "source_dimensions": {
                name: dict(child.ds.sizes) for name, child in processing_set.items()
            },
            "data_selection": {
                str(task_id): task.get("data_selection", {})
                for task_id, task in task_mapping.items()
            },
        }
    )
