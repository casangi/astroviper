"""Distributed single-field continuum imaging application.

The notebook-facing setup remains outside, while graph construction and execution
live in :func:`image_continuum_single_field`.
"""

import copy
import os
from typing import Any

import toolviper.utils.parameter
import zarr
from zarr.codecs import BloscCodec

import astroviper.node_tasks as node_tasks
from astroviper.distributed_applications.imaging.continuum_timing import (
    accumulate_graph_timings as _accumulate_graph_timings,
)
from astroviper.distributed_applications.imaging.continuum_timing import (
    combine_graph_timing_frames as _combine_graph_timing_frames,
)
from astroviper.distributed_applications.imaging.continuum_timing import (
    graph_timing_record as _graph_timing_record,
)
from astroviper.node_tasks.imaging.image_continuum_single_field import (
    _prepare_continuum_result_for_transfer,
)
from astroviper.utils.continuum_cache import (
    CONTINUUM_WEIGHT_CACHE_GROUP as _CONTINUUM_WEIGHT_CACHE_GROUP,
)
from astroviper.utils.continuum_cache import (
    CONTINUUM_WEIGHT_CACHE_VARIABLE as _CONTINUUM_WEIGHT_CACHE_VARIABLE,
)
from astroviper.utils.continuum_cache import (
    MFS_VISIBILITY_GRID_CACHE_GROUP as _MFS_VISIBILITY_GRID_CACHE_GROUP,
)
from astroviper.utils.continuum_cache import (
    MVC_VISIBILITY_GRID_CACHE_GROUP as _MVC_VISIBILITY_GRID_CACHE_GROUP,
)
from astroviper.utils.continuum_cache import (
    WIDEBAND_PRIMARY_BEAM_CACHE_GROUP as _WIDEBAND_PRIMARY_BEAM_CACHE_GROUP,
)
from astroviper.utils.continuum_cache import (
    WIDEBAND_PRIMARY_BEAM_CACHE_VARIABLE as _WIDEBAND_PRIMARY_BEAM_CACHE_VARIABLE,
)
from astroviper.utils.param_docs import shares_param_docs

_PARAM_CONFIG_DIR = os.path.dirname(__file__)

DISTRIBUTED_APPLICATION_TIMING_PHASES = [
    (
        "DRIVER SETUP",
        None,
        [
            ("create empty image xds", "T_make_empty_image_xds"),
            (
                "determine chunks + parallel coords",
                "T_determine_chunks_and_parallel_coords",
            ),
            ("open processing set", "T_open_processing_set"),
            ("interpolate data coords", "T_interpolate_data_coords"),
            ("write empty image", "T_write_empty_image"),
        ],
    ),
    (
        "GLOBAL WEIGHTING GRAPHS",
        None,
        [
            ("create density graph", "T_create_imaging_weight_graph"),
            ("generate density graph", "T_generate_imaging_weight_dask_graph"),
            ("compute density graph", "T_compute_imaging_weight_graph"),
            ("calculate Briggs factors", "T_calculate_global_briggs_factors"),
            ("create weight-degrid graph", "T_create_imaging_weight_degrid_graph"),
            (
                "generate weight-degrid graph",
                "T_generate_imaging_weight_degrid_dask_graph",
            ),
            ("compute weight-degrid graph", "T_compute_imaging_weight_degrid_graph"),
        ],
    ),
    (
        "CACHE MANAGEMENT",
        None,
        [
            ("create weight cache", "T_create_in_place_weight_cache"),
            ("activate imaging weights", "T_persist_imaging_weights"),
            ("create visibility cache", "T_create_in_place_visibility_cache"),
            ("create wideband-PB cache", "T_create_in_place_widebandpb_cache"),
            ("remove weight cache", "T_remove_in_place_weight_cache"),
        ],
    ),
    (
        "CONTINUUM IMAGING GRAPHS",
        None,
        [
            (
                "create map/reduce/append graphs",
                "T_create_map_reduce_append_graph",
            ),
            ("generate dask graph", "T_generate_dask_graph"),
            ("compute graph", "T_compute_dask_graph"),
        ],
    ),
    (
        "FINAL OUTPUT",
        None,
        [
            ("write final continuum image", "T_write_final_image"),
            ("consolidate metadata", "T_consolidate_metadata"),
        ],
    ),
]

DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY = "T_total"


def _make_continuum_cycle_controller(cycle_control_params):
    """Build the shared controller with every continuum cycle control option."""
    from astroviper.processing_functions.imaging.utils import (
        make_continuum_cycle_controller,
    )

    return make_continuum_cycle_controller(cycle_control_params)


def _create_wideband_primary_beam_cache_store(
    image_store,
    img_xds,
    frequency_values,
    correlations,
    single_precision_image,
    compressor,
):
    """Create the temporary MVC channel-primary-beam cache in the image store."""
    import numpy as np

    root = zarr.open_group(image_store, mode="r+", use_consolidated=False)
    if _WIDEBAND_PRIMARY_BEAM_CACHE_GROUP in root:
        del root[_WIDEBAND_PRIMARY_BEAM_CACHE_GROUP]
    cache = root.create_group(_WIDEBAND_PRIMARY_BEAM_CACHE_GROUP)

    frequencies = np.asarray(frequency_values, dtype=np.float64)
    polarizations = tuple(correlations)
    dimensions = ("time", "frequency", "polarization", "l", "m")
    shape = (
        int(img_xds.sizes["time"]),
        int(frequencies.size),
        len(polarizations),
        int(img_xds.sizes["l"]),
        int(img_xds.sizes["m"]),
    )
    chunks = (
        1,
        1,
        1,
        min(512, shape[-2]),
        min(512, shape[-1]),
    )
    zarr_format = root.metadata.zarr_format
    array_options = {
        "shape": shape,
        "dtype": np.dtype("<f4" if single_precision_image else "<f8"),
        "chunks": chunks,
        "fill_value": np.nan,
        "attributes": {
            "description": "Temporary task-partitioned MVC primary-beam cache.",
            "type": "primary_beam",
            "method": "airy_disk",
            "polarization_values": list(polarizations),
        },
    }
    frequency_options = {
        "data": frequencies,
        "chunks": (min(1024, max(1, frequencies.size)),),
    }
    if zarr_format == 3:
        array_options["dimension_names"] = dimensions
        frequency_options["dimension_names"] = ("frequency",)
    else:
        array_options["compressor"] = compressor

    primary_beam = cache.create_array(
        _WIDEBAND_PRIMARY_BEAM_CACHE_VARIABLE,
        **array_options,
    )
    frequency = cache.create_array("frequency", **frequency_options)
    if zarr_format == 2:
        primary_beam.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)
        frequency.attrs["_ARRAY_DIMENSIONS"] = ["frequency"]

    zarr.consolidate_metadata(image_store)


def _remove_wideband_primary_beam_cache(image_store):
    """Remove the temporary MVC primary-beam cache if it exists."""
    root = zarr.open_group(image_store, mode="r+", use_consolidated=False)
    if _WIDEBAND_PRIMARY_BEAM_CACHE_GROUP in root:
        del root[_WIDEBAND_PRIMARY_BEAM_CACHE_GROUP]


def _create_mfs_visibility_grid_cache_store(
    image_store,
    img_xds,
    image_params,
    correlations,
    single_precision_image,
    compressor,
):
    """Create the temporary globally reduced MFS observed-grid cache."""
    import numpy as np

    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )

    root = zarr.open_group(image_store, mode="r+", use_consolidated=False)
    if _MFS_VISIBILITY_GRID_CACHE_GROUP in root:
        del root[_MFS_VISIBILITY_GRID_CACHE_GROUP]
    cache = root.create_group(_MFS_VISIBILITY_GRID_CACHE_GROUP)

    n_taylor_terms = int(image_params["n_taylor_terms"])
    polarizations = np.asarray(tuple(correlations))
    n_u, n_v = padded_grid_size(
        (img_xds.sizes["l"], img_xds.sizes["m"]),
        image_params["fft_padding"],
    )
    visibility_dims = ("time", "taylor_term", "polarization", "u", "v")
    normalization_dims = ("time", "taylor_term", "polarization")
    visibility_shape = (
        int(img_xds.sizes["time"]),
        n_taylor_terms,
        int(polarizations.size),
        int(n_u),
        int(n_v),
    )
    normalization_shape = visibility_shape[:3]
    zarr_format = root.metadata.zarr_format

    visibility_options = {
        "shape": visibility_shape,
        "dtype": np.dtype("<c8" if single_precision_image else "<c16"),
        "chunks": (
            1,
            1,
            1,
            min(256, int(n_u)),
            min(256, int(n_v)),
        ),
        "fill_value": np.nan + 1j * np.nan,
        "attributes": {"description": "Temporary MFS observed visibility grid."},
    }
    normalization_options = {
        "shape": normalization_shape,
        "dtype": np.dtype("<f8"),
        "chunks": (1, 1, 1),
        "fill_value": np.nan,
        "attributes": {"description": "Temporary MFS visibility normalization."},
    }
    if zarr_format == 3:
        visibility_options["dimension_names"] = visibility_dims
        normalization_options["dimension_names"] = normalization_dims
    else:
        visibility_options["compressor"] = compressor
        normalization_options["compressor"] = compressor

    visibility = cache.create_array("VISIBILITY", **visibility_options)
    normalization = cache.create_array(
        "VISIBILITY_NORMALIZATION", **normalization_options
    )

    coordinate_values = {
        "time": np.asarray(img_xds.coords["time"].values),
        "taylor_term": np.arange(n_taylor_terms, dtype=np.int64),
        "polarization": polarizations,
    }
    for name, values in coordinate_values.items():
        options = {
            "data": values,
            "chunks": (max(1, values.size),),
            # Coordinate values such as time=0 and Taylor order zero are valid
            # data, not missing-value sentinels for xarray's Zarr decoder.
            "fill_value": None,
        }
        if zarr_format == 3:
            options["dimension_names"] = (name,)
        coordinate = cache.create_array(name, **options)
        if zarr_format == 2:
            coordinate.attrs["_ARRAY_DIMENSIONS"] = [name]

    if zarr_format == 2:
        visibility.attrs["_ARRAY_DIMENSIONS"] = list(visibility_dims)
        normalization.attrs["_ARRAY_DIMENSIONS"] = list(normalization_dims)

    cache.attrs["data_groups"] = {
        "residual": {
            "visibility": "VISIBILITY",
            "visibility_normalization": "VISIBILITY_NORMALIZATION",
        }
    }
    cache.attrs["visibility_grid_source"] = "observed_data"
    zarr.consolidate_metadata(image_store)


def _remove_mfs_visibility_grid_cache(image_store):
    """Remove the temporary disk-backed MFS observed-grid cache if present."""
    root = zarr.open_group(image_store, mode="r+", use_consolidated=False)
    if _MFS_VISIBILITY_GRID_CACHE_GROUP in root:
        del root[_MFS_VISIBILITY_GRID_CACHE_GROUP]


def _create_mvc_visibility_grid_cache_store(
    image_store,
    img_xds,
    image_params,
    frequency_values,
    correlations,
    single_precision_image,
    compressor,
):
    """Create the temporary frequency-resolved MVC observed-grid cache."""
    import numpy as np

    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )

    root = zarr.open_group(image_store, mode="r+", use_consolidated=False)
    if _MVC_VISIBILITY_GRID_CACHE_GROUP in root:
        del root[_MVC_VISIBILITY_GRID_CACHE_GROUP]
    cache = root.create_group(_MVC_VISIBILITY_GRID_CACHE_GROUP)

    frequencies = np.asarray(frequency_values, dtype=np.float64)
    polarizations = tuple(correlations)
    n_u, n_v = padded_grid_size(
        (img_xds.sizes["l"], img_xds.sizes["m"]),
        image_params["fft_padding"],
    )
    visibility_dims = ("time", "frequency", "polarization", "u", "v")
    normalization_dims = ("time", "frequency", "polarization")
    visibility_shape = (
        int(img_xds.sizes["time"]),
        int(frequencies.size),
        len(polarizations),
        int(n_u),
        int(n_v),
    )
    normalization_shape = visibility_shape[:3]
    zarr_format = root.metadata.zarr_format

    visibility_options = {
        "shape": visibility_shape,
        "dtype": np.dtype("<c8" if single_precision_image else "<c16"),
        "chunks": (
            1,
            1,
            1,
            min(256, int(n_u)),
            min(256, int(n_v)),
        ),
        "fill_value": np.nan + 1j * np.nan,
        "attributes": {"description": "Temporary MVC observed visibility grid."},
    }
    normalization_options = {
        "shape": normalization_shape,
        "dtype": np.dtype("<f8"),
        "chunks": (1, 1, 1),
        "fill_value": np.nan,
        "attributes": {"description": "Temporary MVC visibility normalization."},
    }
    frequency_options = {
        "data": frequencies,
        "chunks": (min(1024, max(1, frequencies.size)),),
    }
    if zarr_format == 3:
        visibility_options["dimension_names"] = visibility_dims
        normalization_options["dimension_names"] = normalization_dims
        frequency_options["dimension_names"] = ("frequency",)
    else:
        visibility_options["compressor"] = compressor
        normalization_options["compressor"] = compressor

    visibility = cache.create_array("VISIBILITY", **visibility_options)
    normalization = cache.create_array(
        "VISIBILITY_NORMALIZATION", **normalization_options
    )
    frequency = cache.create_array("frequency", **frequency_options)
    if zarr_format == 2:
        visibility.attrs["_ARRAY_DIMENSIONS"] = list(visibility_dims)
        normalization.attrs["_ARRAY_DIMENSIONS"] = list(normalization_dims)
        frequency.attrs["_ARRAY_DIMENSIONS"] = ["frequency"]

    zarr.consolidate_metadata(image_store)


def _remove_mvc_visibility_grid_cache(image_store):
    """Remove the temporary disk-backed MVC observed-grid cache if present."""
    root = zarr.open_group(image_store, mode="r+", use_consolidated=False)
    if _MVC_VISIBILITY_GRID_CACHE_GROUP in root:
        del root[_MVC_VISIBILITY_GRID_CACHE_GROUP]


def _mapping_with_task_primary_beams(node_task_data_mapping, pb_cache_mapping):
    """Attach only each task's own in-memory MVC primary beam to its mapping."""
    expected = {int(task_id) for task_id in node_task_data_mapping}
    actual = {int(task_id) for task_id in pb_cache_mapping}
    if actual != expected:
        raise ValueError(
            "The MVC primary-beam cache does not match the continuum tasks: "
            f"expected={sorted(expected)}, received={sorted(actual)}."
        )

    task_mapping = {}
    for task_id, mapping in node_task_data_mapping.items():
        task_id = int(task_id)
        task_mapping[task_id] = dict(mapping)
        task_mapping[task_id]["primary_beam_xds"] = pb_cache_mapping[task_id]
    return task_mapping


def _mapping_with_task_weights(node_task_data_mapping, weight_cache_mapping):
    """Attach only each task's own in-memory imaging weights to its mapping."""
    expected = {int(task_id) for task_id in node_task_data_mapping}
    actual = {int(task_id) for task_id in weight_cache_mapping}
    if actual != expected:
        raise ValueError(
            "The imaging-weight cache does not match the continuum tasks: "
            f"expected={sorted(expected)}, received={sorted(actual)}."
        )

    task_mapping = {}
    for task_id, mapping in node_task_data_mapping.items():
        task_id = int(task_id)
        task_mapping[task_id] = dict(mapping)
        task_mapping[task_id]["weight_datasets"] = weight_cache_mapping[task_id]
    return task_mapping


def _mapping_with_task_observed_grids(
    node_task_data_mapping,
    observed_visibility_grid_mapping,
):
    """Attach only each task's own in-memory MVC observed UV-grid cache."""
    expected = {int(task_id) for task_id in node_task_data_mapping}
    actual = {int(task_id) for task_id in observed_visibility_grid_mapping}
    if actual != expected:
        raise ValueError(
            "The MVC visibility-grid cache does not match the continuum tasks: "
            f"expected={sorted(expected)}, received={sorted(actual)}."
        )

    task_mapping = {}
    for task_id, mapping in node_task_data_mapping.items():
        task_id = int(task_id)
        task_mapping[task_id] = dict(mapping)
        task_mapping[task_id]["observed_visibility_grid_xds"] = (
            observed_visibility_grid_mapping[task_id]
        )
    return task_mapping


def _create_continuum_weight_cache_store(
    ps_xdt,
    ps_store,
    image_store,
    processing_set_data_group_name,
    compressor,
):
    """Create a temporary disk-backed weight cache beside the output image."""
    import numpy as np

    weight_sources = []
    ps_root = zarr.open_group(ps_store, mode="r", use_consolidated=False)
    image_root = zarr.open_group(image_store, mode="r+", use_consolidated=False)

    for ms_name, ms_xdt in ps_xdt.items():
        data_groups = ms_xdt.ds.attrs.get("data_groups", {})
        if processing_set_data_group_name not in data_groups:
            raise KeyError(
                f"Processing-set child {ms_name!r} does not contain data group "
                f"{processing_set_data_group_name!r}."
            )

        data_group = data_groups[processing_set_data_group_name]
        source_weight_name = data_group.get("weight")
        if source_weight_name is None:
            raise KeyError(
                f"Data group {processing_set_data_group_name!r} in child "
                f"{ms_name!r} does not register an input weight."
            )

        # Open the child directly: disabling root consolidation does not
        # disable a child's own consolidated metadata.
        ms_group = zarr.open_group(
            ps_root.store, path=ms_name, mode="r", use_consolidated=False
        )
        if source_weight_name not in ms_group:
            raise KeyError(
                f"Registered input weight {source_weight_name!r} is absent "
                f"from processing-set child {ms_name!r}."
            )
        weight_sources.append((ms_name, ms_xdt, ms_group[source_weight_name]))

    # The output image belongs to this run, so separate image stores provide
    # independent caches without locking or per-run names in the input PS.
    if _CONTINUUM_WEIGHT_CACHE_GROUP in image_root:
        del image_root[_CONTINUUM_WEIGHT_CACHE_GROUP]
    cache_root = image_root.create_group(_CONTINUUM_WEIGHT_CACHE_GROUP)
    output_zarr_format = image_root.metadata.zarr_format

    for ms_name, ms_xdt, source in weight_sources:
        ms_cache = cache_root.create_group(ms_name)
        zarr_format = source.metadata.zarr_format
        if zarr_format == 3:
            dimensions = tuple(source.metadata.dimension_names)
        else:
            dimensions = tuple(source.attrs["_ARRAY_DIMENSIONS"])
        chunks = list(source.chunks)
        frequency_axis = dimensions.index("frequency")

        # Separate frequency chunks prevent concurrent tasks that own disjoint
        # channels from performing read/modify/write operations on one
        # compressed Zarr chunk.
        chunks[frequency_axis] = 1
        create_options = {
            "shape": source.shape,
            # Imaging-weight calculations accumulate in float64 even when the
            # input statistical weights are float32.  Preserve that precision
            # so in-place and in-memory execution remain numerically identical.
            "dtype": np.dtype("<f8"),
            "chunks": tuple(chunks),
            "fill_value": np.nan,
            "attributes": dict(source.attrs),
        }
        if output_zarr_format == 3:
            create_options["dimension_names"] = dimensions
        else:
            create_options["compressor"] = compressor

        target = ms_cache.create_array(
            _CONTINUUM_WEIGHT_CACHE_VARIABLE,
            **create_options,
        )
        if output_zarr_format == 2:
            target.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

        for dimension, size in zip(dimensions, source.shape, strict=True):
            if dimension not in ms_xdt.ds.coords:
                raise KeyError(
                    f"Coordinate {dimension!r} required by the imaging weights "
                    f"is missing from processing-set child {ms_name!r}."
                )
            values = np.asarray(ms_xdt.ds.coords[dimension].values)
            if values.shape != (size,):
                raise ValueError(
                    f"Coordinate {dimension!r} in processing-set child "
                    f"{ms_name!r} has shape {values.shape}; expected {(size,)}."
                )
            coordinate_options = {
                "data": values,
                "chunks": (min(1024, max(1, size)),),
            }
            if output_zarr_format == 3:
                coordinate_options["dimension_names"] = (dimension,)
            coordinate = ms_cache.create_array(dimension, **coordinate_options)
            if output_zarr_format == 2:
                coordinate.attrs["_ARRAY_DIMENSIONS"] = [dimension]

    zarr.consolidate_metadata(image_store)


def _remove_continuum_weight_cache(
    image_store,
):
    """Remove the temporary disk-backed imaging-weight cache if present."""
    root = zarr.open_group(image_store, mode="r+", use_consolidated=False)
    if _CONTINUUM_WEIGHT_CACHE_GROUP in root:
        del root[_CONTINUUM_WEIGHT_CACHE_GROUP]


def _persist_continuum_weight_cache_to_processing_set(
    ps_xdt,
    ps_store,
    image_store,
    processing_set_data_group_name,
):
    """Copy a completed external weight cache into the input PS on request."""
    import numpy as np

    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    image_root = zarr.open_group(image_store, mode="r", use_consolidated=False)
    if _CONTINUUM_WEIGHT_CACHE_GROUP not in image_root:
        raise KeyError("The continuum imaging-weight cache is missing.")
    cache_root = image_root[_CONTINUUM_WEIGHT_CACHE_GROUP]
    ps_root = zarr.open_group(ps_store, mode="r+", use_consolidated=False)

    # Validate all children before modifying the input Processing Set.
    for ms_name in ps_xdt:
        if ms_name not in cache_root:
            raise KeyError(f"No cached imaging weights exist for {ms_name!r}.")

    for ms_name, ms_xdt in ps_xdt.items():
        cached = cache_root[ms_name][_CONTINUUM_WEIGHT_CACHE_VARIABLE]
        ms_group = zarr.open_group(
            ps_root.store, path=ms_name, mode="r+", use_consolidated=False
        )
        if _CONTINUUM_WEIGHT_CACHE_VARIABLE in ms_group:
            del ms_group[_CONTINUUM_WEIGHT_CACHE_VARIABLE]

        dimensions = (
            tuple(cached.metadata.dimension_names)
            if cached.metadata.zarr_format == 3
            else tuple(cached.attrs["_ARRAY_DIMENSIONS"])
        )
        options = {
            "shape": cached.shape,
            "dtype": cached.dtype,
            "chunks": cached.chunks,
            "fill_value": np.nan,
            "attributes": dict(cached.attrs),
        }
        if ms_group.metadata.zarr_format == 3:
            options["dimension_names"] = dimensions
        else:
            # The temporary image-store cache can be Zarr v3 while Processing
            # Sets are commonly Zarr v2. A v3 codec pipeline cannot be passed
            # through the deprecated v2 ``compressor`` argument.
            options["compressor"] = (
                cached.compressor if cached.metadata.zarr_format == 2 else None
            )
        destination = ms_group.create_array(
            _CONTINUUM_WEIGHT_CACHE_VARIABLE,
            **options,
        )
        if ms_group.metadata.zarr_format == 2:
            destination.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

        frequency_axis = dimensions.index("frequency")
        for channel in range(cached.shape[frequency_axis]):
            selection = [slice(None)] * len(dimensions)
            selection[frequency_axis] = slice(channel, channel + 1)
            selection = tuple(selection)
            destination[selection] = cached[selection]

        _, data_group_out = create_data_groups_in_and_out(
            ms_xdt.ds,
            data_group_in_name=processing_set_data_group_name,
            data_group_out_name=processing_set_data_group_name,
            data_group_out_modified={
                "weight_imaging": _CONTINUUM_WEIGHT_CACHE_VARIABLE,
            },
            overwrite=True,
        )
        modify_data_groups_xds(
            ms_xdt.ds,
            data_group_out_name=processing_set_data_group_name,
            data_group_out=data_group_out,
            description="AstroVIPER continuum imaging weights retained by request.",
        )
        ms_group.attrs["data_groups"] = copy.deepcopy(ms_xdt.ds.attrs["data_groups"])
        zarr.consolidate_metadata(ps_store, path=ms_name)
    zarr.consolidate_metadata(ps_store)


def _continuum_image_for_disk(img_xds, image_data_variables_keep):
    """Return a lazy view containing the requested finalized continuum products.

    Continuum products carry Taylor-term and PSF-Taylor-order dimensions that
    cannot be represented by the frequency-cube placeholders created before the
    graph runs.  This helper therefore derives the disk schema from the final
    in-memory image itself and only removes products the caller did not request.
    """
    import copy

    from astroviper.utils.io import imaging_data_variables_and_dims_double_precision

    registry = imaging_data_variables_and_dims_double_precision
    requested_names = {registry[key]["name"] for key in image_data_variables_keep}
    output = img_xds.copy(deep=False)
    # The XRADIO deletion accessor mutates its dataset in place; isolate the
    # attributes while retaining zero-copy views of the large arrays.
    output.attrs = copy.deepcopy(img_xds.attrs)
    variables_to_drop = [
        name for name in output.data_vars if name not in requested_names
    ]
    if variables_to_drop:
        output.xr_img.delete_data_variables(variables=variables_to_drop)

    return output


###############################################################################
# The main functions to build and reduce graphviper graphs
###############################################################################


def compute_continuum_graph(
    *,
    ps_xdt,
    node_task_data_mapping,
    cycle_input_params,
    reduce_input_params,
    disk_chunk_sizes,
    processing_set_data_group_name,
    monitor_resources_seconds,
    task_priorities,
    reduce_mode,
    reduce_n_batch,
    append_node=None,
    append_input_params=None,
):
    """The graph performs

    map -> reduce -> append_node

    Parameters
    ----------
    ps_xdt
        Processing set used as the map input.
    node_task_data_mapping
        Mapping between processing-set coordinates and map-task coordinates.
    cycle_input_params : dict
        Parameters forwarded to each residual imaging-cycle map task.
    reduce_input_params : dict
        Parameters forwarded to :func:`combine_continuum_chunks`.
    disk_chunk_sizes : dict or None
        Native disk-level chunk sizes used by the GraphVIPER map stage.
    processing_set_data_group_name : str
        Processing-set data group loaded by each map task.
    monitor_resources_seconds : float or None
        Resource-monitor sampling interval for map tasks.
    task_priorities
        Optional GraphVIPER task priorities.
    reduce_mode : str
        GraphVIPER reduction mode.
    reduce_n_batch : int
        Number of inputs combined per reduction batch.
    append_node : callable, optional
        Global node executed after reduction.
    append_input_params : dict, optional
        Parameters forwarded to ``append_node``.

    Returns
    -------
    tuple
        Graph result and graph-construction/execution timing dictionary.

    Raises
    ------
    ValueError
        If only one of ``append_node`` and ``append_input_params`` is supplied.
    """
    import time

    import dask
    from graphviper.graph_tools import append, generate_dask_workflow, map, reduce

    # Some sanity checks
    if append_node is None and append_input_params is not None:
        raise ValueError("append_input_params was supplied without an append_node.")

    if append_node is not None and append_input_params is None:
        raise ValueError("append_node was supplied without append_input_params.")

    timings = {}

    start = time.time()

    # Mapping stage: Residual update (calculation of taylor order uv grids)
    viper_graph = map(
        input_data=ps_xdt,
        node_task_data_mapping=node_task_data_mapping,
        node_task=node_tasks.imaging.residual_update_continuum_single_field,
        input_params=cycle_input_params,
        in_memory_compute=False,
        data_loading_task=None,
        disk_chunk_sizes=disk_chunk_sizes,
        load_node_input_params={
            "processing_set_data_group_name": processing_set_data_group_name,
        },
        monitor_resources_seconds=monitor_resources_seconds,
        task_priorities=task_priorities,
    )

    # Reduce stage: Combine uv grids
    viper_graph = reduce(
        viper_graph,
        combine_continuum_chunks,
        reduce_input_params,
        mode=reduce_mode,
        n_batch=reduce_n_batch,
    )

    # Append node: Either model update or finalization
    if append_node is not None:
        viper_graph = append(
            viper_graph,
            append_node,
            append_input_params,
        )

    timings["T_create_map_reduce_append_graph"] = time.time() - start

    start = time.time()
    dask_graph = generate_dask_workflow(viper_graph)
    timings["T_generate_dask_graph"] = time.time() - start

    start = time.time()
    timings["compute_start_unixtime"] = start
    graph_result = dask.compute(dask_graph)[0]
    timings["compute_end_unixtime"] = time.time()
    timings["T_compute_dask_graph"] = timings["compute_end_unixtime"] - start

    return graph_result, timings


def prepare_continuum_imaging_weights_global(
    *,
    ps_xdt,
    node_task_data_mapping,
    input_params,
    disk_chunk_sizes,
    processing_set_data_group_name,
    monitor_resources_seconds,
    task_priorities,
    reduce_mode="tree",
    reduce_n_batch=2,
):
    """Prepare global weights with density-reduction and lookup graphs.

    Collapse density frequency planes before reduction, calculate Briggs
    factors, then attach or persist each partition's imaging weights.
    Return the weight mapping and graph timings.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Processing Set used as the input to both weighting graphs.
    node_task_data_mapping : dict
        GraphVIPER mapping from task identifiers to Processing Set selections.
    input_params : dict
        Parameters forwarded to the density-grid and weight-degridding nodes.
    disk_chunk_sizes : dict or None
        Disk-level chunk sizes used when loading each map-task partition.
    processing_set_data_group_name : str
        Processing Set data group loaded by the weighting nodes.
    monitor_resources_seconds : float or None
        Resource-monitor sampling interval for map tasks.
    task_priorities : dict or None
        Optional GraphVIPER task-priority configuration.
    reduce_mode : {"tree", "single_node", "tree_n"}, optional
        GraphVIPER reduction topology.
    reduce_n_batch : int, optional
        Number of inputs combined by each ``"tree_n"`` reduction task.

    Returns
    -------
    weight_result : dict
        Task-indexed imaging-weight cache mapping and graph timing records.
    timings : dict
        Graph construction, execution, and Briggs-factor timings.
    """
    import time

    import dask
    import numpy as np
    import xarray as xr
    from graphviper.graph_tools import generate_dask_workflow, map, reduce

    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        normalize_imaging_weight_params,
    )
    from astroviper.processing_functions.imaging.imaging_weighting.briggs_weighting import (
        calculate_briggs_params,
    )

    timings = {}

    start = time.time()

    weight_density_graph = map(
        input_data=ps_xdt,
        node_task_data_mapping=node_task_data_mapping,
        node_task=(node_tasks.imaging.grid_imaging_weight_density_continuum_node),
        input_params=input_params,
        in_memory_compute=False,
        data_loading_task=None,
        disk_chunk_sizes=disk_chunk_sizes,
        load_node_input_params={
            "processing_set_data_group_name": (processing_set_data_group_name),
        },
        monitor_resources_seconds=monitor_resources_seconds,
        task_priorities=task_priorities,
    )

    weight_density_graph = reduce(
        weight_density_graph,
        combine_continuum_weight_density_chunks,
        {},
        mode=reduce_mode,
        n_batch=reduce_n_batch,
    )

    timings["T_create_imaging_weight_graph"] = time.time() - start

    start = time.time()
    dask_graph = generate_dask_workflow(weight_density_graph)
    timings["T_generate_imaging_weight_dask_graph"] = time.time() - start

    start = time.time()
    weight_density_timings = {"compute_start_unixtime": start}
    weight_density_result = dask.compute(dask_graph)[0]
    weight_density_timings["compute_end_unixtime"] = time.time()
    weight_density_timings["T_compute_dask_graph"] = (
        weight_density_timings["compute_end_unixtime"] - start
    )
    timings["T_compute_imaging_weight_graph"] = weight_density_timings[
        "T_compute_dask_graph"
    ]
    graph_records = []
    record = _graph_timing_record(
        "global weight density",
        weight_density_result,
        weight_density_timings,
        stage_kind="weighting",
    )
    if record is not None:
        graph_records.append(record)

    # =============================================================
    # Compute global Briggs factors
    # =============================================================

    start = time.time()

    if "weight_density" not in weight_density_result:
        raise KeyError(
            "The weight-density graph result does not contain 'weight_density'."
        )

    global_weight_density_xds = weight_density_result["weight_density"]

    if not isinstance(global_weight_density_xds, xr.Dataset):
        raise TypeError(
            "weight_density_result['weight_density'] must be an "
            f"xarray.Dataset; received "
            f"{type(global_weight_density_xds).__name__}."
        )

    required_variables = (
        "WEIGHT_DENSITY_GRID",
        "SUM_WEIGHT",
    )

    missing_variables = [
        variable_name
        for variable_name in required_variables
        if variable_name not in global_weight_density_xds
    ]

    if missing_variables:
        raise KeyError(
            "The globally reduced weight-density dataset is missing "
            f"variables {missing_variables}."
        )

    global_weight_density_da = global_weight_density_xds["WEIGHT_DENSITY_GRID"]
    global_sum_weight_da = global_weight_density_xds["SUM_WEIGHT"]

    expected_density_dims = (
        "frequency",
        "weight_polarization",
        "u",
        "v",
    )
    expected_sum_weight_dims = (
        "frequency",
        "weight_polarization",
    )

    if global_weight_density_da.dims != expected_density_dims:
        raise ValueError(
            "WEIGHT_DENSITY_GRID has dimensions "
            f"{global_weight_density_da.dims}; expected "
            f"{expected_density_dims}."
        )

    if global_sum_weight_da.dims != expected_sum_weight_dims:
        raise ValueError(
            "SUM_WEIGHT has dimensions "
            f"{global_sum_weight_da.dims}; expected "
            f"{expected_sum_weight_dims}."
        )

    if (
        global_weight_density_da.sizes["frequency"]
        != global_sum_weight_da.sizes["frequency"]
    ):
        raise ValueError(
            "WEIGHT_DENSITY_GRID and SUM_WEIGHT have different frequency-axis lengths."
        )

    if (
        global_weight_density_da.sizes["weight_polarization"]
        != global_sum_weight_da.sizes["weight_polarization"]
    ):
        raise ValueError(
            "WEIGHT_DENSITY_GRID and SUM_WEIGHT have different "
            "weight-polarization-axis lengths."
        )

    # Global continuum map tasks accumulate every selected channel directly
    # into one density plane. Keeping that singleton layout through reduction
    # avoids materializing a channel-by-UV-grid intermediate on the driver.
    if not global_weight_density_xds.attrs.get("continuum_frequency_collapsed", False):
        raise ValueError(
            "Global continuum weight-density maps must return a frequency-"
            "collapsed density plane."
        )
    global_weight_density_da = global_weight_density_xds["WEIGHT_DENSITY_GRID"]
    global_sum_weight_da = global_weight_density_xds["SUM_WEIGHT"]

    global_weight_density_grid = np.asarray(
        global_weight_density_da.values,
    )

    global_sum_weight = np.asarray(
        global_sum_weight_da.values,
        dtype=np.float64,
    )

    normalized_weight_params = normalize_imaging_weight_params(
        input_params["imaging_weights_params"]
    )

    if normalized_weight_params["weighting"] != "briggs":
        raise ValueError(
            "Global Briggs-factor calculation requires Briggs or uniform "
            "weighting. After normalization, received weighting="
            f"{normalized_weight_params['weighting']!r}."
        )

    global_briggs_factors = calculate_briggs_params(
        global_weight_density_grid,
        global_sum_weight,
        normalized_weight_params,
    )

    global_briggs_factors = np.asarray(global_briggs_factors)

    expected_factor_shape = (
        2,
        global_weight_density_da.sizes["frequency"],
        global_weight_density_da.sizes["weight_polarization"],
    )

    if global_briggs_factors.shape != expected_factor_shape:
        raise ValueError(
            "calculate_briggs_params returned an unexpected shape: "
            f"{global_briggs_factors.shape}; expected "
            f"{expected_factor_shape}."
        )

    if not np.all(np.isfinite(global_briggs_factors)):
        raise ValueError("The global Briggs factors contain non-finite values.")

    T_calculate_global_briggs_factors = time.time() - start

    timings["T_calculate_global_briggs_factors"] = T_calculate_global_briggs_factors

    # =============================================================
    # Package factors into an xr.DataArray
    # =============================================================

    global_briggs_factors_da = xr.DataArray(
        global_briggs_factors,
        dims=(
            "briggs_parameter",
            "frequency",
            "weight_polarization",
        ),
        coords={
            "briggs_parameter": np.arange(
                global_briggs_factors.shape[0],
                dtype=np.int64,
            ),
            "frequency": global_weight_density_xds.coords["frequency"],
            "weight_polarization": (
                global_weight_density_xds.coords["weight_polarization"]
            ),
        },
        name="BRIGGS_FACTORS",
        attrs={
            "description": (
                "Global Briggs factors calculated from the reduced "
                "continuum weight-density grid."
            ),
            "weighting": normalized_weight_params["weighting"],
            "robust": normalized_weight_params["robust"],
        },
    )

    # =============================================================
    # Add factors to the global dataset
    # =============================================================

    global_weight_density_xds["BRIGGS_FACTORS"] = global_briggs_factors_da

    # =============================================================
    # Degridding
    # =============================================================

    weight_degrid_input_params = dict(input_params)

    weight_degrid_input_params["global_weighting_xds"] = global_weight_density_xds

    (
        weight_result,
        weight_degrid_timings,
    ) = compute_continuum_imaging_weight_degrid_graph(
        ps_xdt=ps_xdt,
        node_task_data_mapping=node_task_data_mapping,
        input_params=weight_degrid_input_params,
        disk_chunk_sizes=disk_chunk_sizes,
        processing_set_data_group_name=(processing_set_data_group_name),
        monitor_resources_seconds=monitor_resources_seconds,
        task_priorities=task_priorities,
        reduce_mode=reduce_mode,
        reduce_n_batch=reduce_n_batch,
    )

    _accumulate_graph_timings(timings, weight_degrid_timings)
    record = _graph_timing_record(
        "global weight degridding",
        weight_result,
        weight_degrid_timings,
        stage_kind="weighting",
    )
    if record is not None:
        graph_records.append(record)

    if "weight_cache_mapping" not in weight_result:
        raise RuntimeError(
            "The imaging-weight preparation graph did not return "
            "'weight_cache_mapping'."
        )

    expected_task_ids = set(range(len(node_task_data_mapping)))
    actual_task_ids = set(weight_result["weight_cache_mapping"])

    if actual_task_ids != expected_task_ids:
        raise RuntimeError(
            "The imaging-weight preparation graph returned an unexpected "
            "set of task identifiers: "
            f"expected={sorted(expected_task_ids)}, "
            f"received={sorted(actual_task_ids)}."
        )

    weight_result["timing_graphs"] = graph_records
    return weight_result, timings


def compute_continuum_imaging_weight_degrid_graph(
    *,
    ps_xdt,
    node_task_data_mapping,
    input_params,
    disk_chunk_sizes,
    processing_set_data_group_name,
    monitor_resources_seconds,
    task_priorities,
    reduce_mode="tree",
    reduce_n_batch=2,
):
    """Compute final per-visibility weights from global weighting products.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Processing Set used as the map input.
    node_task_data_mapping : dict
        GraphVIPER mapping from task identifiers to Processing Set selections.
    input_params : dict
        Parameters forwarded to each weight-degridding node. Must contain
        ``global_weighting_xds``.
    disk_chunk_sizes : dict or None
        Disk-level chunk sizes used when loading each map-task partition.
    processing_set_data_group_name : str
        Processing Set data group loaded by each map task.
    monitor_resources_seconds : float or None
        Resource-monitor sampling interval for map tasks.
    task_priorities : dict or None
        Optional GraphVIPER task-priority configuration.
    reduce_mode : {"tree", "single_node", "tree_n"}, optional
        GraphVIPER reduction topology.
    reduce_n_batch : int, optional
        Number of inputs combined by each ``"tree_n"`` reduction task.

    Returns
    -------
    result : dict
        Task-indexed imaging-weight cache mapping and node timing records.
    timings : dict
        Graph construction and execution timings.
    """
    import time

    import dask
    from graphviper.graph_tools import generate_dask_workflow, map, reduce

    if "global_weighting_xds" not in input_params:
        raise KeyError("Graph 2 input parameters must contain 'global_weighting_xds'.")

    timings = {}

    start = time.time()

    weight_graph = map(
        input_data=ps_xdt,
        node_task_data_mapping=node_task_data_mapping,
        node_task=(node_tasks.imaging.degrid_imaging_weights_continuum_node),
        input_params=input_params,
        in_memory_compute=False,
        data_loading_task=None,
        disk_chunk_sizes=disk_chunk_sizes,
        load_node_input_params={
            "processing_set_data_group_name": (processing_set_data_group_name),
        },
        monitor_resources_seconds=monitor_resources_seconds,
        task_priorities=task_priorities,
    )

    weight_graph = reduce(
        weight_graph,
        combine_continuum_imaging_weight_chunks,
        {},
        mode=reduce_mode,
        n_batch=reduce_n_batch,
    )

    timings["T_create_imaging_weight_degrid_graph"] = time.time() - start

    start = time.time()
    dask_graph = generate_dask_workflow(weight_graph)
    timings["T_generate_imaging_weight_degrid_dask_graph"] = time.time() - start

    start = time.time()
    timings["compute_start_unixtime"] = start
    result = dask.compute(dask_graph)[0]
    timings["compute_end_unixtime"] = time.time()
    timings["T_compute_imaging_weight_degrid_graph"] = (
        timings["compute_end_unixtime"] - start
    )

    # GraphViper does not call the reducer for a one-leaf graph, so normalize
    # that map result to the same task-indexed schema produced by reduction.
    if "weight_cache_mapping" not in result and {
        "task_id",
        "weight_datasets",
    }.issubset(result):
        result = dict(result)
        result["weight_cache_mapping"] = {
            int(result["task_id"]): result["weight_datasets"]
        }

    if "weight_cache_mapping" not in result:
        raise RuntimeError(
            "The imaging-weight degrid graph did not return 'weight_cache_mapping'."
        )

    expected_task_ids = set(range(len(node_task_data_mapping)))
    actual_task_ids = {int(task_id) for task_id in result["weight_cache_mapping"]}

    if actual_task_ids != expected_task_ids:
        raise RuntimeError(
            "The imaging-weight degrid graph returned an unexpected "
            "set of task identifiers: "
            f"expected={sorted(expected_task_ids)}, "
            f"received={sorted(actual_task_ids)}."
        )

    return result, timings


def combine_continuum_chunks(input_data, input_params):
    """Combine frequency-chunk continuum map results.

    This function is intended for use as the ``combine`` function passed to
    :func:`graphviper.graph_tools.reduce`. It is associative: the returned object
    has the same structure as each input element, allowing reductions using
    ``mode="tree"``, ``mode="tree_n"``, or ``mode="single_node"``.

    Each input element is expected to have the form

    .. code-block:: python

        {
            "image": img_xds,
            "timing_node_tasks": timing_df,
            "deconvolution": deconvolution_return_dict,
        }

    where ``img_xds`` contains the continuum imaging products produced by one
    frequency chunk.

    By default, the reducer performs an element-wise sum of the additive image
    products

    * ``VISIBILITY``;
    * ``VISIBILITY_NORMALIZATION``;
    * ``UV_SAMPLING``;
    * ``UV_SAMPLING_NORMALIZATION``.

    With the current continuum implementation these variables have dimensions

    ``VISIBILITY``
        ``(time, taylor_term, polarization, u, v)``

    ``VISIBILITY_NORMALIZATION``
        ``(time, taylor_term, polarization)``

    ``UV_SAMPLING``
        ``(time, psf_taylor_order, polarization, u, v)``

    ``UV_SAMPLING_NORMALIZATION``
        ``(time, psf_taylor_order, polarization)``.

    MVC supplies a mode-specific additive list. Every imaging cycle reduces
    ``MVC_RESIDUAL_TAYLOR_NUMERATOR``, ``MVC_RESIDUAL_WEIGHT_SUM``,
    ``MVC_RESIDUAL_BEAM_SQUARED_TAYLOR_SUM``, ``MVC_RESIDUAL_WEIGHT_TAYLOR_SUM``,
    and ``MVC_RESIDUAL_BEAM_CHANNEL_COUNT``; the
    first cycle additionally reduces ``MVC_PSF_TAYLOR_NUMERATOR``,
    ``MVC_PSF_WEIGHT_SUM``, and ``MVC_PRIMARY_BEAM_WEIGHTED_SUM``. None of
    these products has a frequency dimension.

    MFS also carries ``PRIMARY_BEAM_REFERENCE`` with dimensions
    ``(time, polarization, l, m)`` as a non-additive static product.

    All remaining dataset variables (for example metadata, coordinates, primary
    beam products, and other non-additive quantities) are copied from the first
    input dataset after verifying consistency across all inputs.

    Parameters
    ----------
    input_data : list of dict
        Leaf map-task results or partially reduced results.

    input_params : dict, optional
        Optional reducer configuration. Supported entries are

        ``additive_variables`` : sequence of str, optional
            Dataset variables that are accumulated by element-wise addition.
            Defaults to

            ``("VISIBILITY", "VISIBILITY_NORMALIZATION",
            "UV_SAMPLING", "UV_SAMPLING_NORMALIZATION")``.

        ``strict`` : bool, optional
            If ``True``, every additive variable must exist in every input and
            dimensions, coordinates, and selected metadata must agree exactly.
            Defaults to ``True``.

        ``copy_image_deep`` : bool, optional
            If ``True``, the first image dataset is deep-copied before
            accumulation. This avoids modifying an input object at the cost of
            additional temporary memory. Defaults to ``True``.

    Returns
    -------
    dict
        Dictionary with the same structure as a map-task result.

        ``"image"``
            Dataset containing the accumulated continuum products.

        ``"timing_node_tasks"``
            Concatenated timing dataframe containing one row per original map
            task.

        ``"deconvolution"``
            Combined deconvolution metadata.

    Notes
    -----
    This reducer performs an unnormalized accumulation of continuum products.
    Any final normalization by globally accumulated weights or the effective
    MVC primary beam is performed later, after all chunks have been reduced.

    The reducer assumes that all inputs were produced using identical continuum
    imaging parameters (for example, reference frequency, Taylor expansion, and
    image geometry). These metadata are validated when available before
    accumulation.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.processing_functions.imaging.utils.cycle_control import (
        merge_imaging_dicts,
    )

    if input_params is None:
        input_params = {}

    additive_variables = tuple(
        input_params.get(
            "additive_variables",
            (
                "VISIBILITY",
                "VISIBILITY_NORMALIZATION",
                "UV_SAMPLING",
                "UV_SAMPLING_NORMALIZATION",
            ),
        )
    )
    strict = bool(input_params.get("strict", True))
    copy_image_deep = bool(input_params.get("copy_image_deep", True))
    if not input_data:
        raise ValueError("combine_continuum_chunks received no inputs.")

    # Extract the continuum_mode
    continuum_mode = input_params.get("continuum_mode")

    if continuum_mode is None:
        continuum_mode = (
            input_data[0]["image"]
            .attrs.get(
                "continuum_imaging",
                {},
            )
            .get("continuum_mode", "mfs")
        )

    static_variables = tuple(
        input_params.get(
            "static_variables",
            ("PRIMARY_BEAM_REFERENCE",) if continuum_mode == "mfs" else (),
        )
    )

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    # Kept inline due to their shortness.
    # returns dictionary of metadata
    def _continuum_metadata(dataset):
        metadata = dataset.attrs.get("continuum_imaging", {})

        return {
            "n_taylor_terms": metadata.get(
                "n_taylor_terms",
                dataset.attrs.get("n_taylor_terms"),
            ),
            "reference_frequency_hz": metadata.get(
                "reference_frequency_hz",
                dataset.attrs.get("reference_frequency_hz"),
            ),
            "n_psf_taylor_terms": metadata.get(
                "n_psf_taylor_terms",
                dataset.attrs.get("n_psf_taylor_terms"),
            ),
        }

    # validate metadata
    def _validate_metadata(reference_dataset, candidate_dataset, input_index):
        reference = _continuum_metadata(reference_dataset)
        candidate = _continuum_metadata(candidate_dataset)

        for key in (
            "n_taylor_terms",
            "n_psf_taylor_terms",
        ):
            reference_value = reference[key]
            candidate_value = candidate[key]

            if (
                reference_value is not None
                and candidate_value is not None
                and int(reference_value) != int(candidate_value)
            ):
                raise ValueError(
                    f"Continuum metadata mismatch for {key!r}: "
                    f"reference={reference_value}, "
                    f"input[{input_index}]={candidate_value}."
                )

        reference_frequency = reference["reference_frequency_hz"]
        candidate_frequency = candidate["reference_frequency_hz"]

        if (
            reference_frequency is not None
            and candidate_frequency is not None
            and not np.isclose(
                float(reference_frequency),
                float(candidate_frequency),
                rtol=1.0e-12,
                atol=0.0,
            )
        ):
            raise ValueError(
                "All continuum chunks must use the same reference frequency: "
                f"reference={reference_frequency}, "
                f"input[{input_index}]={candidate_frequency}."
            )

    # validate that additive variables match in dimensions
    def _validate_additive_variable(
        reference_array,
        candidate_array,
        variable_name,
        input_index,
    ):
        if reference_array.dims != candidate_array.dims:
            raise ValueError(
                f"Dimension mismatch for {variable_name!r}: "
                f"reference={reference_array.dims}, "
                f"input[{input_index}]={candidate_array.dims}."
            )

        if reference_array.shape != candidate_array.shape:
            raise ValueError(
                f"Shape mismatch for {variable_name!r}: "
                f"reference={reference_array.shape}, "
                f"input[{input_index}]={candidate_array.shape}."
            )

        # join="exact" catches mismatched Taylor, spatial, polarization, and
        # time coordinates while avoiding silent coordinate reindexing.
        try:
            xr.align(
                reference_array,
                candidate_array,
                join="exact",
                copy=False,
            )
        except ValueError as exc:
            raise ValueError(
                f"Coordinate mismatch for {variable_name!r} in input[{input_index}]."
            ) from exc

    def _merge_weight_mapping(
        destination,
        incoming,
        *,
        input_index,
    ):
        if not isinstance(incoming, dict):
            raise TypeError(
                "Imaging-weight mapping from "
                f"input[{input_index}] must be a "
                f"dictionary; received "
                f"{type(incoming).__name__}."
            )

        for task_id, weight_datasets in incoming.items():
            task_id = int(task_id)

            if task_id in destination:
                raise ValueError(f"Duplicate imaging-weight result for task {task_id}.")

            if not isinstance(weight_datasets, dict):
                raise TypeError(
                    "Imaging weights for task "
                    f"{task_id} must be a dictionary; "
                    f"received "
                    f"{type(weight_datasets).__name__}."
                )

            if not weight_datasets:
                raise ValueError(f"Imaging-weight mapping for task {task_id} is empty.")

            destination[task_id] = weight_datasets

    # ------------------------------------------------------------------
    # Timing and deconvolution metadata
    # ------------------------------------------------------------------

    # Concatenate timing and deconvolution return dictionaries
    combined_timing = pd.DataFrame()
    deconvolution_dicts = []
    # Optional: gather imaging weights
    weight_cache_mapping = {}
    # Optional: gather channelized primary beams
    pb_cache_mapping = {}
    # Optional: gather task-local MVC observed visibility grids
    observed_visibility_grid_mapping = {}

    for input_index, result in enumerate(input_data):
        if not isinstance(result, dict):
            raise TypeError(
                f"input[{input_index}] must be a "
                f"dictionary; received "
                f"{type(result).__name__}."
            )

        if "timing_node_tasks" not in result:
            raise KeyError(
                "Every continuum map/reduce result must contain 'timing_node_tasks'."
            )

        timing = result["timing_node_tasks"]

        # Preserve GraphViper's optional leaf-level resource monitor output in
        # the same way as the existing cube reducer. Partially reduced inputs
        # have already folded this information into their timing dataframe.
        resource_usage = result.get("resource_usage")

        if resource_usage is not None:
            timing = timing.copy()

            for key, value in resource_usage.items():
                timing[key] = [value] if isinstance(value, list) else value

        combined_timing = pd.concat(
            [combined_timing, timing],
            ignore_index=True,
        )

        if "deconvolution" in result:
            deconvolution_dicts.append(result["deconvolution"])

        # Leaf result from a first-imaging-cycle map task.
        #
        # Check for weight_datasets rather than task_id because
        # ordinary map tasks may also carry a task_id.
        if "weight_datasets" in result:
            if "task_id" not in result:
                raise KeyError(
                    f"input[{input_index}] contains "
                    "'weight_datasets' but does not contain "
                    "'task_id'."
                )

            _merge_weight_mapping(
                weight_cache_mapping,
                {int(result["task_id"]): (result["weight_datasets"])},
                input_index=input_index,
            )

        # Partially reduced result from an earlier tree level.
        if "weight_cache_mapping" in result:
            _merge_weight_mapping(
                weight_cache_mapping,
                result["weight_cache_mapping"],
                input_index=input_index,
            )

        # Gather primary beam mapping
        if "pb_xds" in result:
            task_id = int(result["task_id"])

            if task_id in pb_cache_mapping:
                raise ValueError(f"Duplicate MVC PB cache for task {task_id}.")

            pb_cache_mapping[task_id] = result["pb_xds"]

        if "pb_cache_mapping" in result:
            incoming = result["pb_cache_mapping"]

            if not isinstance(incoming, dict):
                raise TypeError("pb_cache_mapping must be a dictionary.")

            for task_id, pb_xds in incoming.items():
                task_id = int(task_id)

                if task_id in pb_cache_mapping:
                    raise ValueError(f"Duplicate MVC PB cache for task {task_id}.")

                pb_cache_mapping[task_id] = pb_xds

        if "observed_visibility_grid_xds" in result:
            task_id = int(result["task_id"])
            if task_id in observed_visibility_grid_mapping:
                raise ValueError(
                    f"Duplicate MVC visibility-grid cache for task {task_id}."
                )
            observed_visibility_grid_mapping[task_id] = result[
                "observed_visibility_grid_xds"
            ]

        if "observed_visibility_grid_mapping" in result:
            incoming = result["observed_visibility_grid_mapping"]
            if not isinstance(incoming, dict):
                raise TypeError(
                    "observed_visibility_grid_mapping must be a dictionary."
                )
            for task_id, observed_grid_xds in incoming.items():
                task_id = int(task_id)
                if task_id in observed_visibility_grid_mapping:
                    raise ValueError(
                        f"Duplicate MVC visibility-grid cache for task {task_id}."
                    )
                observed_visibility_grid_mapping[task_id] = observed_grid_xds

    # ------------------------------------------------------------------
    # Taylor-image reduction
    # ------------------------------------------------------------------

    first_result = input_data[0]

    if "image" not in first_result:
        raise KeyError(
            "Every continuum map/reduce result must contain an 'image' dataset."
        )

    first_image = first_result["image"]

    if not isinstance(first_image, xr.Dataset):
        raise TypeError(
            "result['image'] must be an xarray.Dataset; received "
            f"{type(first_image).__name__}."
        )

    # Make a copy for the combined image
    # In this way, dimensions, shapes, metadata and coordinates are going to be correct
    combined_image = first_image.copy(deep=copy_image_deep)

    for variable_name in additive_variables:
        if variable_name not in combined_image:
            if strict:
                raise KeyError(
                    f"Additive variable {variable_name!r} is absent from the "
                    "first continuum image."
                )
            continue

        # A deep Dataset copy already owns this array. For a shallow copy, or
        # an unusual non-NumPy/read-only backend, materialize one writable
        # accumulator now so later inputs can be added without temporaries.
        accumulator_data = combined_image[variable_name].data
        if (
            not copy_image_deep
            or not isinstance(accumulator_data, np.ndarray)
            or not accumulator_data.flags.writeable
        ):
            combined_image[variable_name] = combined_image[variable_name].copy(
                data=np.array(accumulator_data, copy=True)
            )

    # Main loop: Loop over input data to combine
    for input_index, result in enumerate(input_data[1:], start=1):
        if "image" not in result:
            raise KeyError(f"input[{input_index}] does not contain an 'image' dataset.")

        candidate_image = result["image"]

        # Sanity checks
        if not isinstance(candidate_image, xr.Dataset):
            raise TypeError(
                f"input[{input_index}]['image'] must be an xarray.Dataset; "
                f"received {type(candidate_image).__name__}."
            )

        _validate_metadata(
            combined_image,
            candidate_image,
            input_index,
        )

        for variable_name in static_variables:
            reference_has_static = variable_name in combined_image
            candidate_has_static = variable_name in candidate_image
            if not reference_has_static and not candidate_has_static:
                continue
            if not reference_has_static:
                combined_image[variable_name] = candidate_image[variable_name].copy(
                    deep=True
                )
                continue
            if not candidate_has_static:
                continue
            reference_static = combined_image[variable_name]
            candidate_static = candidate_image[variable_name]
            _validate_additive_variable(
                reference_static, candidate_static, variable_name, input_index
            )
            if not np.allclose(
                np.asarray(reference_static.values),
                np.asarray(candidate_static.values),
                rtol=1.0e-12,
                atol=0.0,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Static variable {variable_name!r} differs in input[{input_index}]."
                )

        # combine for every additive variable
        for variable_name in additive_variables:
            accumulator_has_variable = variable_name in combined_image
            candidate_has_variable = variable_name in candidate_image

            if not accumulator_has_variable or not candidate_has_variable:
                if strict:
                    raise KeyError(
                        f"Additive variable {variable_name!r} must be present "
                        f"in every input. Present in accumulator: "
                        f"{accumulator_has_variable}; present in "
                        f"input[{input_index}]: {candidate_has_variable}."
                    )
                continue

            accumulator = combined_image[variable_name]
            candidate = candidate_image[variable_name]

            _validate_additive_variable(
                accumulator,
                candidate,
                variable_name,
                input_index,
            )

            np.add(
                accumulator.data,
                np.asarray(candidate.data),
                out=accumulator.data,
            )

    # Record how many original map-task rows contributed to this partial or
    # complete reduction. Because timing rows are concatenated at each tree
    # level, this remains correct for partially reduced inputs.
    combined_image.attrs["n_continuum_chunks_combined"] = int(len(combined_timing))
    combined_image.attrs["continuum_additive_variables"] = list(additive_variables)

    if deconvolution_dicts:
        combined_deconvolution = merge_imaging_dicts(deconvolution_dicts)
    else:
        # Keep the output schema stable even when no input supplied
        # deconvolution metadata.
        from astroviper.processing_functions.imaging.utils.imaging_dict import (
            ImagingDict,
        )

        combined_deconvolution = ImagingDict()

    return_dict = {
        "image": combined_image,
        "timing_node_tasks": combined_timing,
        "deconvolution": combined_deconvolution,
    }

    if weight_cache_mapping:
        return_dict["weight_cache_mapping"] = weight_cache_mapping

    if pb_cache_mapping:
        return_dict["pb_cache_mapping"] = pb_cache_mapping

    if observed_visibility_grid_mapping:
        return_dict["observed_visibility_grid_mapping"] = (
            observed_visibility_grid_mapping
        )

    return _prepare_continuum_result_for_transfer(return_dict)


def combine_continuum_weight_density_chunks(
    input_data,
    input_params,
):
    """Combine partition-local continuum weight-density contributions.

    This function is intended for use as the ``combine`` function passed to
    :func:`graphviper.graph_tools.reduce` for the first distributed weighting
    graph.

    Each leaf input is expected to contain

    .. code-block:: python

        {
            "task_id": task_id,
            "weight_density": weight_density_xds,
            "timing_node_tasks": timing_df,
        }

    where ``weight_density_xds`` contains

    ``WEIGHT_DENSITY_GRID``
        Dimensions ``(frequency, weight_polarization, u, v)``.

    ``SUM_WEIGHT``
        Dimensions ``(frequency, weight_polarization)``.

    Partially reduced inputs have the same structure, except that ``task_id``
    is omitted.

    Every input must carry ``continuum_frequency_collapsed=True`` and contain
    one frequency plane. These planes are added positionally, without aligning
    their representative frequency coordinates or expanding channel grids.
    Channel counts and their representative frequencies are combined across
    tree levels; geometry, weighting metadata and timing remain validated.

    Parameters
    ----------
    input_data : list of dict
        Leaf map-task results or partially reduced results.

    input_params : dict, optional
        Optional reducer configuration. Supported entries are

        ``copy_density_deep`` : bool, optional
            Whether to deep-copy the first density dataset before
            accumulation. Defaults to ``True``.

        ``frequency_rtol`` : float, optional
            Relative tolerance used when comparing numeric weighting metadata.
            Defaults to ``1e-12``.

        ``frequency_atol`` : float, optional
            Absolute tolerance used when comparing numeric weighting metadata.
            Defaults to ``0.0``.

    Returns
    -------
    dict
        Associative partial or complete reduction result containing

        ``"weight_density"``
            Dataset containing the globally accumulated
            ``WEIGHT_DENSITY_GRID`` and ``SUM_WEIGHT``.

        ``"timing_node_tasks"``
            Concatenated timing dataframe containing one row per original map
            task.

    Notes
    -----
    This reducer only sums raw weight-density and sum-of-weight contributions.
    Briggs factors must be calculated from the fully reduced result after this
    graph has completed.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    if input_params is None:
        input_params = {}

    copy_density_deep = bool(input_params.get("copy_density_deep", True))
    frequency_rtol = float(input_params.get("frequency_rtol", 1.0e-12))
    frequency_atol = float(input_params.get("frequency_atol", 0.0))

    if not input_data:
        raise ValueError("combine_continuum_weight_density_chunks received no inputs.")

    required_variables = (
        "WEIGHT_DENSITY_GRID",
        "SUM_WEIGHT",
    )

    # -------------------------------------------------------------
    # Validate one reducer input and return its density dataset.
    # -------------------------------------------------------------
    def _get_density_dataset(result, input_index):
        if not isinstance(result, dict):
            raise TypeError(
                f"input[{input_index}] must be a dictionary; received "
                f"{type(result).__name__}."
            )

        if "weight_density" not in result:
            raise KeyError(f"input[{input_index}] does not contain 'weight_density'.")

        density_xds = result["weight_density"]

        if not isinstance(density_xds, xr.Dataset):
            raise TypeError(
                f"input[{input_index}]['weight_density'] must be an "
                f"xarray.Dataset; received "
                f"{type(density_xds).__name__}."
            )

        missing_variables = [
            name for name in required_variables if name not in density_xds
        ]

        if missing_variables:
            raise KeyError(
                f"input[{input_index}]['weight_density'] is missing "
                f"variables {missing_variables}."
            )

        if "frequency" not in density_xds.coords:
            raise KeyError(
                f"input[{input_index}]['weight_density'] does not "
                "contain a frequency coordinate."
            )

        frequency = np.asarray(
            density_xds.coords["frequency"].values,
            dtype=np.float64,
        )

        if frequency.ndim != 1:
            raise ValueError(
                f"input[{input_index}] frequency coordinate must be "
                f"one-dimensional; received shape {frequency.shape}."
            )

        if frequency.size == 0:
            raise ValueError(f"input[{input_index}] contains no frequency channels.")

        if not np.all(np.isfinite(frequency)):
            raise ValueError(
                f"input[{input_index}] frequency coordinate contains non-finite values."
            )

        if np.unique(frequency).size != frequency.size:
            raise ValueError(
                f"input[{input_index}] frequency coordinate contains duplicate values."
            )

        density = density_xds["WEIGHT_DENSITY_GRID"]
        sum_weight = density_xds["SUM_WEIGHT"]

        expected_density_dims = (
            "frequency",
            "weight_polarization",
            "u",
            "v",
        )
        expected_sum_weight_dims = (
            "frequency",
            "weight_polarization",
        )

        if density.dims != expected_density_dims:
            raise ValueError(
                f"input[{input_index}] WEIGHT_DENSITY_GRID has "
                f"dimensions {density.dims}; expected "
                f"{expected_density_dims}."
            )

        if sum_weight.dims != expected_sum_weight_dims:
            raise ValueError(
                f"input[{input_index}] SUM_WEIGHT has dimensions "
                f"{sum_weight.dims}; expected "
                f"{expected_sum_weight_dims}."
            )

        if not density_xds.attrs.get("continuum_frequency_collapsed", False):
            raise ValueError(
                f"input[{input_index}] must contain a frequency-collapsed "
                "continuum weight-density grid."
            )
        if frequency.size != 1:
            raise ValueError(
                f"input[{input_index}] must contain exactly one frequency plane."
            )

        return density_xds

    # -------------------------------------------------------------
    # Validate non-frequency geometry and weighting metadata.
    # -------------------------------------------------------------
    def _validate_compatible_layout(
        reference_xds,
        candidate_xds,
        input_index,
    ):
        for dimension in (
            "weight_polarization",
            "u",
            "v",
        ):
            reference_size = reference_xds.sizes.get(dimension)
            candidate_size = candidate_xds.sizes.get(dimension)

            if reference_size != candidate_size:
                raise ValueError(
                    f"Dimension {dimension!r} differs for "
                    f"input[{input_index}]: reference={reference_size}, "
                    f"candidate={candidate_size}."
                )

            if dimension in reference_xds.coords and dimension in candidate_xds.coords:
                try:
                    xr.align(
                        reference_xds.coords[dimension],
                        candidate_xds.coords[dimension],
                        join="exact",
                        copy=False,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Coordinate {dimension!r} differs for input[{input_index}]."
                    ) from exc

        metadata_keys = (
            "weighting",
            "robust",
            "casa_weighting_implementation",
            "cell_size_l",
            "cell_size_m",
        )

        for key in metadata_keys:
            reference_value = reference_xds.attrs.get(key)
            candidate_value = candidate_xds.attrs.get(key)

            if reference_value is None or candidate_value is None:
                continue

            if isinstance(reference_value, float | np.floating) or isinstance(
                candidate_value, float | np.floating
            ):
                if not np.isclose(
                    float(reference_value),
                    float(candidate_value),
                    rtol=frequency_rtol,
                    atol=frequency_atol,
                ):
                    raise ValueError(
                        f"Weight-density metadata {key!r} differs for "
                        f"input[{input_index}]: "
                        f"reference={reference_value}, "
                        f"candidate={candidate_value}."
                    )
            elif reference_value != candidate_value:
                raise ValueError(
                    f"Weight-density metadata {key!r} differs for "
                    f"input[{input_index}]: "
                    f"reference={reference_value!r}, "
                    f"candidate={candidate_value!r}."
                )

    # -------------------------------------------------------------
    # Initialize accumulation.
    # -------------------------------------------------------------
    density_inputs = [
        _get_density_dataset(result, input_index)
        for input_index, result in enumerate(input_data)
    ]
    combined_xds = density_inputs[0].copy(deep=copy_density_deep)

    # Ensure the two numerical accumulators own writable arrays.
    for variable_name in required_variables:
        combined_xds[variable_name] = combined_xds[variable_name].copy(deep=True)

    combined_timing = pd.DataFrame()

    # -------------------------------------------------------------
    # Collect timing from every input, including the first.
    # -------------------------------------------------------------
    for input_index, result in enumerate(input_data):
        if "timing_node_tasks" not in result:
            raise KeyError(
                f"input[{input_index}] does not contain 'timing_node_tasks'."
            )

        timing_df = result["timing_node_tasks"]

        if not isinstance(timing_df, pd.DataFrame):
            raise TypeError(
                f"input[{input_index}]['timing_node_tasks'] must be a "
                f"pandas.DataFrame; received "
                f"{type(timing_df).__name__}."
            )

        resource_usage = result.get("resource_usage")

        if resource_usage is not None:
            timing_df = timing_df.copy()

            for key, value in resource_usage.items():
                timing_df[key] = [value] if isinstance(value, list) else value

        combined_timing = pd.concat(
            [combined_timing, timing_df],
            ignore_index=True,
        )

    # -------------------------------------------------------------
    # Accumulate remaining density datasets.
    # -------------------------------------------------------------
    for input_index, candidate_xds in enumerate(density_inputs[1:], start=1):
        _validate_compatible_layout(combined_xds, candidate_xds, input_index)
        for variable_name in required_variables:
            combined_xds[variable_name].data[...] += np.asarray(
                candidate_xds[variable_name].values
            )

    input_frequency_counts = np.asarray(
        [
            int(dataset.attrs.get("n_input_frequency_channels", 1))
            for dataset in density_inputs
        ],
        dtype=np.int64,
    )
    representative_frequencies = np.asarray(
        [float(dataset.frequency.values[0]) for dataset in density_inputs],
        dtype=np.float64,
    )
    total_input_frequency_channels = int(input_frequency_counts.sum())
    combined_xds = combined_xds.assign_coords(
        frequency=[
            float(
                np.average(
                    representative_frequencies,
                    weights=input_frequency_counts,
                )
            )
        ]
    )
    combined_xds.attrs["n_input_frequency_channels"] = total_input_frequency_channels

    combined_xds.attrs["n_weight_density_chunks_combined"] = int(len(combined_timing))

    # This count is additive across tree-reduction levels because the input
    # datasets carry the number of original MS datasets represented.
    combined_xds.attrs["n_processing_set_datasets_gridded"] = sum(
        int(dataset.attrs.get("n_processing_set_datasets_gridded", 0))
        for dataset in density_inputs
    )

    return _prepare_continuum_result_for_transfer(
        {
            "weight_density": combined_xds,
            "timing_node_tasks": combined_timing,
        }
    )


def combine_continuum_imaging_weight_chunks(
    input_data,
    input_params,
):
    """Collect per-task imaging-weight preparation results.

    The function is associative and can therefore be used by GraphViper's
    tree, tree_n, and single-node reduction modes.

    Parameters
    ----------
    input_data : list of dict
        Leaf or partially reduced imaging-weight results.
    input_params : dict
        Reserved reducer configuration; currently unused.

    Returns
    -------
    dict
        Combined ``weight_cache_mapping`` and concatenated node-task timings.
    """
    import pandas as pd

    del input_params

    if not input_data:
        raise ValueError("combine_continuum_imaging_weight_chunks received no inputs.")

    weight_cache_mapping = {}
    combined_timing = pd.DataFrame()

    for input_index, result in enumerate(input_data):
        if not isinstance(result, dict):
            raise TypeError(
                f"Weight-preparation input[{input_index}] must be a "
                f"dictionary; received {type(result).__name__}."
            )

        # Leaf result from the preparation map.
        if "task_id" in result:
            task_id = int(result["task_id"])

            if task_id in weight_cache_mapping:
                raise ValueError(f"Duplicate imaging-weight result for task {task_id}.")

            if "weight_datasets" not in result:
                raise KeyError(
                    f"Weight-preparation result for task {task_id} does not "
                    "contain 'weight_datasets'."
                )

            weight_cache_mapping[task_id] = result["weight_datasets"]

        # Partially reduced result.
        elif "weight_cache_mapping" in result:
            for task_id, weight_datasets in result["weight_cache_mapping"].items():
                task_id = int(task_id)

                if task_id in weight_cache_mapping:
                    raise ValueError(
                        f"Duplicate imaging-weight result for task {task_id}."
                    )

                weight_cache_mapping[task_id] = weight_datasets

        else:
            raise KeyError(
                f"Weight-preparation input[{input_index}] contains neither "
                "'task_id' nor 'weight_cache_mapping'."
            )

        timing_df = result.get("timing_node_tasks")

        resource_usage = result.get("resource_usage")
        if resource_usage is not None and timing_df is not None:
            timing_df = timing_df.copy()
            for key, value in resource_usage.items():
                timing_df[key] = [value] if isinstance(value, list) else value

        if timing_df is not None:
            combined_timing = pd.concat(
                [combined_timing, timing_df],
                ignore_index=True,
            )

    return _prepare_continuum_result_for_transfer(
        {
            "weight_cache_mapping": weight_cache_mapping,
            "timing_node_tasks": combined_timing,
        }
    )


###############################################################################
# Generic Helper Functions
###############################################################################


def _apply_exact_frequency_selection_to_continuum_mapping(
    node_task_data_mapping,
    ps_xdt,
    *,
    frequency_rtol=1.0e-12,
    frequency_atol=0.0,
):
    """Replace interpolated continuum frequency selections with exact indexers.

    GraphVIPER's generic coordinate interpolation converts chunk edges to
    positional slices with ``searchsorted``.  CASA channel selections can leave a
    Processing Set child with a non-monotonic frequency coordinate, for which
    those interpolated slices are not valid.  This continuum-specific helper
    retains the mapping and task coordinates created by GraphVIPER, but rebuilds
    each child's frequency ``isel`` indexer from physical channel matches.

    Consecutive ascending indices remain lazy slices.  Reordered or otherwise
    irregular indices use an integer array, which also makes the loaded child
    frequency order agree with the task image frequency order.

    Parameters
    ----------
    node_task_data_mapping : dict
        Mapping returned by
        :func:`graphviper.graph_tools.coordinate_utils.interpolate_data_coords_onto_parallel_coords`.
        It is updated in place and returned.
    ps_xdt : xarray.DataTree
        Open Processing Set whose children supply the on-disk frequency axes.
    frequency_rtol, frequency_atol : float
        Relative and absolute tolerances used only to accommodate floating-point
        representations of the same physical channel coordinate.

    Returns
    -------
    dict
        ``node_task_data_mapping`` with exact per-child frequency indexers.

    Raises
    ------
    ValueError
        If a frequency coordinate is not one-dimensional, a match is ambiguous,
        or any Processing Set channel is not assigned exactly once.
    """
    import numpy as np

    def _slice_or_array(indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 1:
            start = int(indices[0])
            return slice(start, start + 1)

        steps = np.diff(indices)
        if np.all(steps == 1):
            return slice(int(indices[0]), int(indices[-1]) + 1)

        return indices

    child_frequencies = {}
    child_selection_counts = {}
    for child_name, child_xdt in ps_xdt.items():
        if "frequency" not in child_xdt.coords:
            raise ValueError(
                f"Processing Set child {child_name!r} has no frequency coordinate."
            )

        frequencies = np.asarray(child_xdt.coords["frequency"].values, dtype=float)
        if frequencies.ndim != 1:
            raise ValueError(
                f"Processing Set child {child_name!r} frequency coordinate must "
                f"be one-dimensional; received shape {frequencies.shape}."
            )

        child_frequencies[child_name] = frequencies
        child_selection_counts[child_name] = np.zeros(frequencies.size, dtype=np.int64)

    for task_id, task_mapping in node_task_data_mapping.items():
        task_coords = task_mapping.get("task_coords", {})
        if "frequency" not in task_coords:
            raise ValueError(f"Continuum task {task_id} has no frequency coordinate.")

        task_frequencies = np.asarray(
            task_coords["frequency"]["data"],
            dtype=float,
        )
        if task_frequencies.ndim != 1:
            raise ValueError(
                f"Continuum task {task_id} frequency coordinate must be "
                f"one-dimensional; received shape {task_frequencies.shape}."
            )

        previous_selection = task_mapping.get("data_selection", {})
        exact_selection = {}
        task_match_counts = np.zeros(task_frequencies.size, dtype=np.int64)

        for child_name, frequencies in child_frequencies.items():
            matches = np.isclose(
                task_frequencies[:, np.newaxis],
                frequencies[np.newaxis, :],
                rtol=frequency_rtol,
                atol=frequency_atol,
            )
            matches_per_task_channel = matches.sum(axis=1)
            matches_per_child_channel = matches.sum(axis=0)

            if np.any(matches_per_task_channel > 1):
                raise ValueError(
                    f"Continuum task {task_id} has a frequency that matches more "
                    f"than one channel in Processing Set child {child_name!r}."
                )
            if np.any(matches_per_child_channel > 1):
                raise ValueError(
                    f"Processing Set child {child_name!r} has a frequency that "
                    f"matches more than one channel in continuum task {task_id}."
                )

            task_indices, child_indices = np.nonzero(matches)
            if child_indices.size == 0:
                continue

            order = np.argsort(task_indices, kind="stable")
            child_indices = child_indices[order]
            task_indices = task_indices[order]

            child_selection = dict(previous_selection.get(child_name, {}))
            child_selection["frequency"] = _slice_or_array(child_indices)
            exact_selection[child_name] = child_selection

            task_match_counts[task_indices] += 1
            child_selection_counts[child_name][child_indices] += 1

        if np.any(task_match_counts == 0):
            missing = task_frequencies[task_match_counts == 0]
            raise ValueError(
                f"Continuum task {task_id} contains frequencies that are absent "
                f"from every Processing Set child: {missing}."
            )

        task_mapping["data_selection"] = exact_selection

    for child_name, selection_counts in child_selection_counts.items():
        if np.any(selection_counts != 1):
            frequencies = child_frequencies[child_name][selection_counts != 1]
            counts = selection_counts[selection_counts != 1]
            raise ValueError(
                f"Processing Set child {child_name!r} channels were not assigned "
                "exactly once across continuum tasks. "
                f"Frequencies={frequencies}; assignment counts={counts}."
            )

    return node_task_data_mapping


def calculate_number_of_chunks_for_continuum_imaging(
    img_xds,
    single_precision_image,
    n_frequency_chunks,
    thread_info,
    n_taylor_terms=1,
    n_correlations=2,
    fft_padding=1.2,
):
    """Determine the number of frequency chunks for continuum imaging.

    Computes the memory required per single-frequency chunk and delegates to
    :func:`calculate_data_chunking` to find a chunk count that satisfies both
    memory and parallelism constraints. If ``n_frequency_chunks`` is already
    provided it is returned unchanged.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Empty image dataset whose ``sizes`` attribute provides the grid dimensions.
    single_precision_image : bool
        If ``True``, use single-precision (complex64 / float32) memory estimates
        for the image-domain arrays; otherwise double-precision
        (complex128 / float64).
    n_frequency_chunks : int or None
        If not ``None``, this value is returned directly without any computation.
    thread_info : dict or None
        Thread information as returned by :func:`get_thread_info`.
        If ``None``, thread information is queried automatically.
    n_taylor_terms : int, optional
        Number of continuum Taylor terms held by every map task.
    n_correlations : int, optional
        Number of instrument-basis correlation planes gridded by each task.
    fft_padding : float, optional
        FFT padding used to size the task-local UV grids.

    Returns
    -------
    int
        Number of frequency chunks to use for the parallel imaging graph.
    """
    import toolviper.utils.logger as logger

    if n_frequency_chunks is None:
        from astroviper.processing_functions.imaging.utils.fft_sizing import (
            padded_grid_size,
        )
        from astroviper.utils.data_partitioning import bytes_in_dtype

        float_bytes, complex_bytes = (
            (bytes_in_dtype["float32"], bytes_in_dtype["complex64"])
            if single_precision_image
            else (bytes_in_dtype["float64"], bytes_in_dtype["complex128"])
        )
        n_image_pixels = img_xds.sizes["l"] * img_xds.sizes["m"] * img_xds.sizes["time"]
        n_u, n_v = padded_grid_size(
            (img_xds.sizes["l"], img_xds.sizes["m"]), fft_padding
        )
        n_uv_pixels = int(n_u) * int(n_v) * img_xds.sizes["time"]
        fudge_factor = 1.2
        # Selected-channel scratch grows with the number of channels in a task.
        memory_per_frequency = fudge_factor * (
            n_image_pixels
            * n_correlations
            * (3 * complex_bytes + 3 * float_bytes)
            / (1024**3)
        )
        # Taylor grids are allocated once per task, not once per input channel.
        # The residual has n_taylor_terms UV grids and the PSF has 2*n_taylor_terms-1, all in
        # the correlation basis. Taylor residual/model work arrays are in the
        # requested Stokes basis and are constant per task as well.
        n_taylor_uv_grids = n_taylor_terms + (2 * n_taylor_terms - 1)
        constant_memory = (
            fudge_factor
            * (
                n_uv_pixels * n_correlations * n_taylor_uv_grids * complex_bytes
                + n_image_pixels
                * img_xds.sizes["polarization"]
                * 3
                * n_taylor_terms
                * float_bytes
            )
            / (1024**3)
        )

        logger.info(
            "Continuum task memory model: "
            f"{memory_per_frequency} GiB per frequency channel + "
            f"{constant_memory} GiB task-local Taylor grids"
        )

        from astroviper.utils.data_partitioning import calculate_frequency_chunking

        n_frequency_chunks = calculate_frequency_chunking(
            img_xds.sizes["frequency"],
            memory_per_frequency,
            constant_memory,
            n_chunks=None,
            thread_info=thread_info,
            tasks_per_thread=4,
        )
        logger.info(
            "Number of frequency chunks: "
            + str(n_frequency_chunks)
            + " frequency channels: "
            + str(img_xds.sizes["frequency"])
        )
    return n_frequency_chunks


def _load_continuum_clean_mask(clean_mask, image_size):
    """Load and validate an optional two-dimensional continuum CLEAN mask."""
    if clean_mask is None:
        return None

    import numpy as np

    clean_mask_array = np.squeeze(np.asarray(np.load(clean_mask)))
    expected_shape = tuple(int(value) for value in image_size)
    if clean_mask_array.ndim != 2 or clean_mask_array.shape != expected_shape:
        raise ValueError(
            "clean_mask must contain one 2-D image plane with shape "
            f"{expected_shape}; received {clean_mask_array.shape}."
        )
    if not np.all(np.isfinite(clean_mask_array)):
        raise ValueError("clean_mask contains non-finite values.")

    clean_mask_array = np.ascontiguousarray(clean_mask_array > 0.5)
    if not np.any(clean_mask_array):
        raise ValueError("clean_mask does not select any pixels.")
    return clean_mask_array


def _resolve_continuum_deconvolver(deconvolver, image_params):
    """Resolve user-facing continuum deconvolver choices before graph creation."""
    import operator
    import warnings

    deconvolver = str(deconvolver).lower()
    if deconvolver not in ("hogbom", "mtmfs"):
        raise ValueError(
            "deconvolver must be 'hogbom' or 'mtmfs' for continuum imaging; "
            f"received {deconvolver!r}."
        )
    if deconvolver != "mtmfs":
        return deconvolver

    if "n_taylor_terms" not in image_params:
        raise KeyError(
            "image_params must contain 'n_taylor_terms' when deconvolver='mtmfs'."
        )
    try:
        n_taylor_terms = operator.index(image_params["n_taylor_terms"])
    except TypeError as exc:
        raise TypeError(
            "image_params['n_taylor_terms'] must be an integer when deconvolver='mtmfs'."
        ) from exc

    if n_taylor_terms == 1:
        warnings.warn(
            "deconvolver='mtmfs' was requested with n_taylor_terms=1; falling back "
            "to deconvolver='hogbom'.",
            UserWarning,
            stacklevel=3,
        )
        return "hogbom"
    if not 2 <= n_taylor_terms <= 4:
        raise ValueError(
            "deconvolver='mtmfs' requires image_params['n_taylor_terms'] between "
            f"2 and 4; received {n_taylor_terms}."
        )
    return deconvolver


def _resolve_continuum_taylor_expansion(image_params):
    """Resolve the Taylor expansion of the sky model before graph creation.

    The node tasks and processing functions read only
    ``image_params["n_taylor_terms"]`` and ``image_params["reference_frequency"]``.
    The deprecated alias ``image_params["reference_frequency_hz"]`` is
    resolved here, once, so that every stage uses the same frequency.

    Parameters
    ----------
    image_params : dict
        Application-local copy of the image parameters, with ``n_taylor_terms``
        present and at least 1 (checked by the schema). ``n_taylor_terms`` may be any
        integer type, and the reference frequency any real scalar (Python,
        NumPy or xarray). Modified in place: on return ``n_taylor_terms`` is an int,
        ``reference_frequency`` holds the reference frequency in Hz as a float,
        and ``reference_frequency_hz`` is removed.

    Raises
    ------
    TypeError
        If ``n_taylor_terms`` is not an integer.
    KeyError
        If neither ``reference_frequency`` nor ``reference_frequency_hz`` is
        given.
    ValueError
        If both are given with different values, or the reference frequency is
        not a finite and positive scalar.
    """
    import operator
    import warnings

    from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
        _get_reference_frequency_hz,
    )

    try:
        image_params["n_taylor_terms"] = operator.index(image_params["n_taylor_terms"])
    except TypeError as exc:
        raise TypeError("image_params['n_taylor_terms'] must be an integer.") from exc

    if "reference_frequency_hz" in image_params:
        warnings.warn(
            "image_params['reference_frequency_hz'] is deprecated; use "
            "image_params['reference_frequency'] (in Hz) instead.",
            FutureWarning,
            stacklevel=3,
        )
        alias = image_params.pop("reference_frequency_hz")
        if "reference_frequency" not in image_params:
            image_params["reference_frequency"] = alias
        elif _get_reference_frequency_hz(image_params) != _get_reference_frequency_hz(
            {"reference_frequency": alias}
        ):
            raise ValueError(
                "image_params gives two different reference frequencies: "
                f"reference_frequency={image_params['reference_frequency']} and "
                f"reference_frequency_hz={alias}. Give only reference_frequency."
            )

    image_params["reference_frequency"] = _get_reference_frequency_hz(image_params)


###############################################################################
# Main distributed layer level function call
###############################################################################


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def image_continuum_single_field(
    ps_store: str,
    image_store: str,
    image_params: dict[str, Any],
    imaging_weights_params: dict[str, Any],
    cycle_control_params: dict[str, Any],
    gridder: str = "anti_alias",
    deconvolver: str = "hogbom",
    primary_beam_correction: bool = False,
    primary_beam_limit: float = 0.2,
    continuum_mode: str = "mfs",
    instrument_polarization_basis: str = "linear",
    scan_intents: list[str] = ["OBSERVE_TARGET#ON_SOURCE"],  # noqa: B006 - param.json requires list/str (not nullable); never mutated
    field_name: str | None = None,
    clean_mask: str | None = None,
    image_data_variables_keep: list[str] = [  # noqa: B006 - param.json requires a list (not nullable); never mutated
        "sky_model",
        "sky_residual",
        "point_spread_function",
        "primary_beam",
        "beam_fit_params_point_spread_function",
    ],
    compressor=None,
    processing_set_data_group_name: str = "corrected",
    single_precision_image: bool = True,
    thread_info: dict | None = None,
    processing_function_threads: int = 1,
    n_mapping_parallelism: dict[str, int | None] | None = None,
    overwrite: bool = False,
    memory_mode: str = "in_memory",
    weight_memory_mode: str = "in_place",
    visibility_memory_mode: str = "recompute",
    wideband_primary_beam_memory_mode: str = "recompute",
    cache_directory: str | None = None,
    write_visibility_model_to_ps: bool = False,
    write_imaging_weights_to_ps: bool = False,
    clear_cache: bool = True,
    vizualize_graph: bool = False,
    disk_chunk_sizes: dict[str, int] | str | None = None,
    fft_backend: str = "pyfftw",
    restore: bool = False,
    gridding_kernel_params: dict[str, Any] | None = None,
    skunk_works: bool = False,
    compute_backend: str = "dask",
    mpi_cluster_setup: dict[str, Any] | None = None,
    reduce_mode: str = "tree",
    reduce_n_batch: int = 2,
    image_sharding: dict[str, int] | None = None,
    image_chunking: dict[str, int] | None = None,
    task_time_kill_switch_seconds: float | None = None,
    monitor_resources_seconds: float | None = None,
) -> dict:
    """Run distributed MFS or MVC continuum imaging.

    MFS reduces Taylor UV grids before the global inverse FFT. MVC performs
    channel FFTs and PB correction in map tasks, then reduces Taylor
    contributions. Native MT-MFS updates the complete Taylor model; Högbom
    updates Taylor term zero. Both verify refreshed residuals before
    finalization. With one Taylor term, an MT-MFS request emits a warning and
    falls back to Högbom. Restoration is optional.

    The imaging cycle loop (``cycle_control_params``) does not run separately
    for every frequency channel as in cube imaging. Continuum performs one
    model update after reduction across frequency partitions: Högbom updates
    Taylor term zero, while MT-MFS jointly updates all Taylor terms. Higher
    terms never run independent imaging-cycle loops.

    ``imaging_weights_params`` also accepts ``weighting_scope``, the uv
    weight density that uniform and Briggs weights are made from:
    ``"global"`` grids one density from all channels, the density of the
    continuum image, and is the default for ``"uniform"`` and ``"briggs"``;
    ``"local"`` grids the density of each channel on its own, as cube
    imaging does, and is the default for ``"briggs_abs"``, which only the
    local scope computes (``"global"`` refuses it). Natural weights use no
    density, so the scope does not change them.

    When ``deconvolver="mtmfs"``, ``cycle_control_params`` also accepts
    ``scales``, ``small_scale_bias``, and ``mask_threshold``. These names are
    passed unchanged through the Python and native layers.

    With ``deconvolver="mtmfs"`` the divergence test of a model update
    (``max_iter_divergence``) is made on the Taylor-zero residual with the
    RMS limit of Asp and the peak limit of Hogbom: a plane is stopped as
    diverged when its RMS has been above ``(1 + gain)`` times the lowest RMS
    of the model update for ``max_iter_divergence`` consecutive iterations,
    or at once when its peak exceeds ``(1 + gain)`` times its peak at the
    start of the model update.

    Parameters
    ----------
    ps_store : str
        Path or URL of the input Processing Set.
    image_store : str
        Path/URL of the on-disk Zarr image cube.
    primary_beam_correction : bool
        If ``True`` write the primary beam corrected restored sky to the
        ``sky_restored_primary_beam_corrected``
        (``SKY_RESTORED_PRIMARY_BEAM_CORRECTED``) variable: the model divided
        by the (power) primary beam and convolved with the clean beam, plus
        the residual divided by the primary beam; pixels below the primary
        beam cutoff (``primary_beam_limit``) are blanked with NaN.  Requires
        ``restore``.
    restore : bool
        If ``True`` produce a restored image after deconvolution: the model
        convolved with the clean beam (the Gaussian fit to the PSF) plus the
        residual, written to the ``sky_restored`` (``SKY_RESTORED``) variable.
    scan_intents : list of str or str, optional
        Scan intent or intents passed to
        :func:`xradio.measurement_set.open_processing_set` when opening the
        input Processing Set.
    field_name : str or None, optional
        Reserved field-selection interface. It is currently ignored; select a
        field when preparing the input Processing Set.
    memory_mode : str, optional
        General processing-set memory policy. Continuum imaging currently
        supports only ``"in_memory"``; the weight, visibility-grid, and
        wideband-primary-beam policies are controlled separately below.
    cache_directory : str or None, optional
        Reserved general processing-set cache location. It is currently ignored
        by continuum imaging.
    clear_cache : bool, optional
        Reserved general processing-set cache cleanup switch. It is currently
        ignored by continuum imaging; private continuum caches are always
        removed when their lifetime ends.
    write_visibility_model_to_ps : bool
        Reserved processing-set model-persistence switch. Continuum imaging does
        not currently write predicted model visibilities to the input Processing
        Set.
    write_imaging_weights_to_ps : bool
        Copy the completed disk-backed imaging weights into the input Processing
        Set; requires ``weight_memory_mode="in_place"``. Later runs do not use
        them: every run calculates its own imaging weights.
    compute_backend : str, optional
        Reserved backend selector. The continuum driver currently builds and
        computes Dask graphs regardless of this value.
    mpi_cluster_setup : dict or None, optional
        Reserved MPI cluster configuration. It is currently ignored by the
        continuum driver.
    reduce_mode : {"tree", "single_node", "tree_n"}, optional
        GraphVIPER reduction topology used for continuum and global-weighting
        graphs.
    reduce_n_batch : int, optional
        Number of inputs combined by each ``"tree_n"`` reduction task. Must be
        at least 2; ignored by the other reduction modes.
    task_time_kill_switch_seconds : float, optional
        Abort a continuum map task when its wall-clock runtime exceeds this
        many seconds. ``None`` disables the limit.
    monitor_resources_seconds : float, optional
        Sampling interval in seconds for per-task CPU, memory, and I/O resource
        monitoring. ``None`` disables sampling.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
        Continuum imaging also requires ``frequency_coords``,
        ``n_taylor_terms`` (CASA ``nterms``), and ``reference_frequency`` in Hz;
        ``reference_frequency_hz`` is a deprecated alias. ``spectral_reference``
        sets the spectral frame (default ``"lsrk"``). ``primary_beam_model``
        selects ``auto``, ``airy`` or ``casa_airy``; ``list_dish_diameters`` and
        ``list_blockage_diameters`` are arrays in metres,
        ``primary_beam_max_radius_1ghz`` is a radius in radians at 1 GHz, and
        ``primary_beam_ipower`` is the exponent applied to the voltage pattern
        (default 2, the power beam). These primary-beam keys override the inferred
        beam prescription.
    imaging_weights_params : dict
        Weighting configuration: ``weighting`` is ``"natural"``, ``"uniform"``,
        ``"briggs"`` or ``"briggs_abs"``; ``robust`` controls the Briggs
        trade-off. ``weighting_scope`` is ``"local"`` for a density per
        frequency channel or ``"global"`` for one density across all selected
        channels in distributed continuum imaging. Natural weighting is
        independent of this scope.
    continuum_mode : {"mfs", "mvc"}
        Continuum construction mode. ``"mfs"`` grids Taylor-weighted
        visibilities directly. Its ``SKY_MODEL`` Taylor coefficients describe
        the apparent sky, ``A_nu * I_nu``; consequently ``tt1 / tt0`` contains
        both the intrinsic source spectral index and the primary-beam spectral
        index. Its reference-frequency primary beam corrects the Taylor-zero
        restored product, but not the higher Taylor coefficients. ``"mvc"`` forms
        channel images and common-beam Taylor equations, and predicts
        ``M_nu * A_nu / Abar``. Its ``SKY_MODEL`` coefficients therefore describe
        the intrinsic spectrum under the common-beam convention, so
        ``tt1 / tt0`` excludes the primary-beam spectral index. These conventions
        match CASA's standard-gridder MT-MFS and MVC behavior. MVC images the two
        parallel hands only; ``["I", "Q", "U", "V"]`` is refused.
    deconvolver : str
        Deconvolution algorithm for the model update. ``"hogbom"`` (C++,
        threaded across planes) cleans Taylor term zero only. ``"mtmfs"``
        (C++, multi-term multi-frequency synthesis) cleans the Taylor terms
        together and needs ``image_params["n_taylor_terms"]`` from 2 to 4; with
        ``n_taylor_terms=1`` it falls back to ``"hogbom"`` with a warning. Other values
        are refused before anything is written.
    primary_beam_limit : float
        Primary-beam cutoff in ``[0, 1]``. This single setting controls CLEAN
        support, MVC channel correction and spectral fitting, and final
        model-first PB-corrected restoration. Pixels below this limit are
        excluded from cleaning and blanked with NaN in the corrected product.
        This corresponds to CASA ``pblimit`` / ``pbmask``.
    instrument_polarization_basis : str
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` or ``"circular"``. The residual update grids and degrids the
        correlations of this basis and the model update deconvolves in the
        Stokes basis, in which the image is written. The Stokes planes requested
        in ``image_params["polarization_coords"]`` fix the correlations that are
        loaded and gridded: the two parallel hands give ``I, Q`` (linear) or
        ``I, V`` (circular), all four correlations give ``I, Q, U, V``. A sample
        is used only if none of its loaded correlations is flagged.
    image_data_variables_keep : list of str
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    compressor : object, optional
        Output Zarr compressor.
    processing_set_data_group_name : str
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    single_precision_image : bool
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the model
        update runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    thread_info : dict, optional
        Resource description used for partition planning.
    processing_function_threads : int
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    n_mapping_parallelism : dict, optional
        Mapping parallelism as ``{"frequency": count}``. ``count=None`` or a
        missing dictionary auto-determines the number of frequency chunks.
    overwrite : bool
        Allow replacement of the output image store.
    vizualize_graph : bool
        Reserved graph-visualization switch. It is currently ignored.
    disk_chunk_sizes : dict or str, optional
        Input chunk sizes; ``"Auto"`` infers native storage chunks.
    fft_backend : str
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    skunk_works : bool
        Enable the experimental storage path; required for sharded output.
    cycle_control_params : dict
        CLEAN cycle controls. An **imaging cycle** (below simply a cycle)
        is one **residual update** (degrid the model, form residual
        visibilities, grid and inverse FFT them into the residual image)
        followed by one **model update** (deconvolve the residual image into
        the sky model). Every limit and threshold is applied independently to
        each image plane: a plane stops when it meets its own criterion. Cube
        imaging runs the cycle loop separately for every frequency channel
        (one channel per node task); continuum imaging has one controller for
        the Taylor stack after reduction across frequency partitions. **Cycle
        control** is the stop test between the residual update and the model
        update, which
        decides whether a plane gets another cycle and sets the limits of its
        next model update. It is made after every residual update, the first
        included, on the residual that update has just made, never on the
        residual a model update ends with, and the residual and restored
        images written are those of the last residual update. **Iteration
        control** is the stop test inside the model update, made by the
        deconvolver at every iteration against those limits. The CASA
        ``tclean`` equivalent is given in brackets.
        Keys:

        - ``max_iter`` [CASA ``niter``] : Maximum number of deconvolution
          iterations (flux components) per plane, summed over all cycles. A
          plane stops once it has spent this budget. ``max_iter = 0`` makes
          only the dirty image (no deconvolution). *Differs from CASA*: CASA's
          ``niter`` is one budget for the whole image; here every plane gets
          the full value, and no budget is shared or split between planes.
        - ``max_cycles`` [CASA ``nmajor``] : Maximum number of cycles.
          ``max_cycles = N`` performs at most ``N`` model updates and
          ``N + 1`` residual updates: the first makes the dirty image, and
          every model update is followed by a residual update, whose residual
          the stop test checks; the last one's residual is written.
          ``max_cycles = 0`` makes only the dirty image; ``max_cycles = -1``
          removes the limit. Cube imaging counts this per frequency channel;
          continuum imaging counts it once for the reduced Taylor stack.
          ``max_cycles = "auto"`` predicts the limit of an imaging unit from
          its dirty image: every plane takes ``P0``, its peak
          residual at the start of the first cycle (the ``peak_residual``
          below), ``f``, its PSF fraction (below), and ``L``, its threshold or,
          when that is 0, its theoretical point source sensitivity, worked out
          from the imaging weights as the ``SENSITIVITY`` of
          ``point_spread_function_cube_single_field``;
          ``n = ceil(ln(P0 / L) / ln(1 / f))`` cycles (at least 1) bring the
          peak down to ``L``, and the plane's limit is ``n + 3`` against a
          threshold and ``3 * n`` against the sensitivity (``EXTRA_CYCLES``
          and ``NOISY_CYCLE_SAFETY`` of
          :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).
          The sensitivity is the thermal noise of the plane when the weights
          are absolute (``WEIGHT = 1 / sigma**2``). With unit weights, as a
          simulation without noise writes them, it is not a noise level but
          about 1 over the square root of the number of visibilities (natural
          weighting; 1e-3 for a million) in the units of the data; the limit
          grows only with ``ln(P0 / L)``, and on the same prediction ``3 * n``
          cycles bring the peak down to ``P0 * (L / P0)**3``. The imaging unit's
          ``max_cycles`` is the largest limit of its planes (at least 1; a
          plane without a pixel in the clean mask, with a peak of 0 or without
          data is left out) and then counts down like a number. Every plane
          records its limit as ``max_cycles_auto`` in the returned convergence
          record. Needs ``max_psf_fraction < 1``.
        - ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, as a
          float in Jy, or a list of one float per Stokes plane in the order of
          ``polarization_coords`` (the floors of the Stokes planes of noise free
          data scale with their fluxes). The distributed application and the
          node task also take a nested list of shape (n_frequency,
          n_polarization), one row per image channel in the order of the image
          frequency coordinate; each channel gets its row. A plane stops when the
          peak of its residual inside the clean mask, as the latest residual
          update made it, falls to or below its threshold times
          ``1 + threshold_tolerance``; the threshold is
          also a hard floor on ``threshold_per_cycle``. ``0`` disables the
          absolute stop. *Differs from CASA*: floats in Jy only, no ``'1mJy'``
          strings.
        - ``threshold_tolerance`` : Relative tolerance of the threshold stop,
          a finite number ``>= 0``: a plane whose threshold is above 0 stops
          with imaging stop code 2, and gets no iterations while other planes of
          its channel go on, once the peak of its residual inside the clean mask
          is at most ``threshold * (1 + threshold_tolerance)``; the model update
          of a plane that goes on still cleans down to ``threshold_per_cycle``,
          floored at the threshold itself. Default 0.02 (``THRESHOLD_TOLERANCE`` of
          :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).
          No CASA equivalent.
        - ``threshold_sigma`` [CASA ``nsigma``] : Noise based stopping threshold
          per plane, as a multiple of the plane's robust residual rms
          (``1.4826 * MAD``). The effective threshold of a plane is
          ``max(threshold, threshold_sigma * rms)`` and it floors
          ``threshold_per_cycle`` in the same way. ``0`` disables it. Reserved:
          accepted but not yet implemented.
        - ``primary_beam_limit`` [CASA ``pblimit`` / ``pbmask``] : Primary beam
          mask cutoff as a fraction of the peak primary beam, in ``[0, 1]``.
          Pixels where the primary beam is below this fraction are excluded
          from cleaning. A masking cutoff, distinct from ``threshold``. Cube
          imaging reads this key (default 0). Continuum imaging instead takes
          the top-level ``primary_beam_limit`` (default 0.2) and injects it into
          cycle control so cleaning and PB correction use one cutoff.
        - ``gain`` [CASA ``gain``] : CLEAN loop gain, the fraction of the
          selected peak flux subtracted from the residual image at each
          deconvolution iteration (``0 < gain <= 1``).
        - ``psf_sidelobe_factor`` [CASA ``cyclefactor``] : Multiplier applied to
          the measured peak PSF sidelobe level (``max_psf_sidelobe``) when
          setting how deep one model update cleans (see
          ``threshold_per_cycle``). Larger values trigger the next residual
          update sooner; smaller values clean deeper before each residual
          update.
        - ``max_iter_per_cycle`` [CASA ``cycleniter``] : Maximum number of
          deconvolution iterations a plane may run in one cycle's model update
          before the next residual update is triggered. ``max_iter_per_cycle =
          -1`` lets the adaptive ``threshold_per_cycle`` govern the depth
          instead; otherwise the count is clamped to never exceed the plane's
          remaining ``max_iter``.
        - ``min_psf_fraction`` [CASA ``minpsffraction``] : Lower clamp on the PSF
          fraction defined below. Raising it limits how deep a single model
          update cleans.
        - ``max_psf_fraction`` [CASA ``maxpsffraction``] : Upper clamp on the
          same PSF fraction; it guarantees a minimum amount of cleaning per
          model update even when the PSF sidelobe level is high.
        - ``max_iter_divergence`` : Divergence test of the model update.
          Number of consecutive deconvolution iterations the RMS of
          a plane's residual, taken over the clean mask, may be above
          ``(1 + gain / 10)`` times the lowest RMS it has reached in the model
          update before that model update is stopped as diverged. For Hogbom
          a peak above ``(1 + gain)`` times the peak at the start of the
          model update stops it at once. Asp, whose least-squares components
          can raise the peak of a healthy residual, has no peak limit and
          allows ``(1 + gain)`` times the lowest RMS instead. An infinite peak
          always stops the model update. The next residual update then
          recomputes the true residual, the stop test is made on it, and the
          cycles go on (see ``max_cycles_divergence``). Default 1, the first
          such iteration; ``-1`` disables the two limits.
        - ``max_cycles_divergence`` : Divergence stop of the cycle control. A
          plane whose last ``max_cycles_divergence`` model updates all ended
          diverged (see ``max_iter_divergence``) gets no more iterations and
          is stopped with imaging stop code 5 by the stop test after the next
          residual update (unless a stop that takes precedence applies there:
          an empty mask, ``max_iter`` spent, the threshold or the entropy
          stop, codes 7, 1, 2 and 10), while the other planes of its channel
          carry on.
          Default ``-1``, off.
        - ``entropy_stop`` : If ``True``, a plane stops once the entropy of
          its residual has passed its maximum. The entropy (Homan, Roth and
          Pushkarev 2024, AJ 167, 11) measures how much the residual looks
          like noise everywhere. It rises while the clean removes emission
          and falls once the clean fits noise. It is worked out after every
          residual update, and the plane stops when it is lower than in an
          earlier cycle. The fall is noticed one model update after the
          maximum and the model of that model update is kept, so a small
          ``max_iter_per_cycle`` makes the stop sharper. Default ``False``.
          No CASA equivalent.
        - ``entropy_max_snr`` : The entropy of a plane is followed once the
          peak of its residual inside the clean mask is at most this many
          times the RMS of the residual. Above it the residual is dominated
          by the pattern of the point spread function. Default 6.
        - ``entropy_spatial_bins`` : Number of spatial bins along each of the
          two image axes used for the entropy. Default 7.
        - ``entropy_flux_bins`` : Number of flux bins per unit of RMS used for
          the entropy. Default 10.
        - ``record_iteration_history`` : If ``True``, every model update
          records, for every plane and every deconvolution iteration, the peak
          residual the iteration cleaned (signed, in the units of the residual
          image) and the RMS of the residual inside the clean mask after its
          subtraction, to follow the deconvolution in detail. The returned
          convergence record holds them as ``iteration_peak_residual`` and
          ``iteration_residual_rms``: per plane one array per cycle, with one
          value per iteration of that cycle's model update (empty for a cycle
          without one), so the record grows by two float64 values per
          iteration done. A debugging and tuning tool: the images are the same
          to the bit with and without it, and the RMS is the sum of squares
          the peak search of the next iteration makes anyway, so it costs no
          extra pass over the image per iteration. Hogbom deconvolvers only:
          ``deconvolver="asp"`` raises ``ValueError``. Default ``False``.
          No CASA equivalent.
        - ``exact_psf_subtraction`` : If ``True``, a Hogbom model update
          subtracts the PSF of every component it finds from the whole residual
          image, with a PSF of twice the image size made once per channel from
          the same imaging weights, cell size and gridding kernel. With
          ``False`` it subtracts the image-sized PSF, which reaches about
          ``n/2`` pixels either side of the component along each axis
          (``n`` the image size in pixels), so farther pixels keep the
          sidelobes of the flux cleaned until the next residual update. The
          beam fit, ``max_psf_sidelobe`` and the PSF written use the
          image-sized PSF either way. Costs a second PSF per channel (four
          times the pixels, kept through the channel's cycles) and a
          subtraction over the whole image at every iteration. Hogbom
          deconvolvers only: ``deconvolver="asp"`` raises ``ValueError``.
          Default ``False``. No CASA equivalent.
        - ``asp_stop_after_hogbom_run`` : If ``True``, an Asp model update ends
          after each of its Hogbom runs (51 or 510 point components) unless
          the RMS of the residual has fallen, by less than a factor 1.5, since
          the model update started, as in CASA and LibRA. Their residual drifts
          from the true one during a model update; the Asp residual of
          AstroVIPER is the exact one, so with ``False`` the model update goes
          on to its ``threshold_per_cycle``, its iteration limit or the
          divergence test, which takes fewer imaging cycles. Asp only.
          Default ``False``.
        - ``asp_residual_components`` : If ``True``, each Asp model update
          starts with components shaped by the residual (Ma, Zhang & Zhang
          2026, A&A 713, A164) down to ``threshold_per_cycle``, and the Asp
          iterations go on from there. They remove the ring residuals Asp
          leaves on resolved emission in far fewer iterations, and take no
          residual below ``threshold_per_cycle``. Further keys
          (``asp_rc_*``) are described in
          :func:`~astroviper.processing_functions.imaging.deconvolution.asp_clean`.
          Asp only. Default ``True``. No CASA equivalent.
        - ``asp_scale_significance`` : If ``True``, Asp measures the noise of
          the residual smoothed with each of its initial scales and, once the
          peak is below ``threshold_per_cycle``, goes on cleaning the most
          significant smoothed residual while it is as significant as the
          threshold is at the smallest scale. The noise is measured on the
          pixels outside the clean mask, without the outermost row and column
          of the image, sampling one in every ``max(1, npix // 400000)`` of
          the ``npix`` pixels of the image (one in 2 at 1024 x 1024, one in 10
          at 2048 x 2048). With fewer than 100 sampled pixels (a clean mask
          that covers the image, for example), or a noise of 0 there (a
          residual that is mostly 0 outside the mask), nothing is cleaned
          below ``threshold_per_cycle``. Asp only. Default ``True``. No CASA
          equivalent.

        A plane whose model updates do no iteration any more (two in a row)
        is stopped with the no progress stop code at the next stop test, so
        an all-zero plane cannot keep its channel cycling.

        Derived per plane before each model update (not set by the caller):
        ``psf_fraction = clamp(max_psf_sidelobe * psf_sidelobe_factor,
        min_psf_fraction, max_psf_fraction)`` is the fraction of the current
        peak residual down to which one model update cleans, and
        ``threshold_per_cycle = max(psf_fraction * peak_residual, threshold)``
        is the stopping threshold of that model update, where
        ``peak_residual`` is the plane's peak residual inside the clean mask
        after the residual update of the same cycle (the peak of the stop
        test), and
        ``max_iter_per_cycle_updated = min(max_iter_per_cycle,
        max_iter_remaining)`` is the iteration limit of that model update (0
        for a plane the entropy or divergence stop has ended), where
        ``max_iter_remaining`` is what is left of ``max_iter``.
    gridder : str, optional
        Currently ``"anti_alias"``: convolutional gridding with the
        anti-aliasing kernel of ``gridding_kernel_params`` (by default the
        ``prolate_spheroidal_casa`` kernel of support 7 and oversampling 100).
        MFS and MVC dispatch visibility, PSF, and prediction work to the shared
        C++ grid/degrid kernels.
    gridding_kernel_params : dict, optional
        The gridding convolution kernel the gridder and degridder use:
        ``type`` (``"prolate_spheroidal_casa"``, Schwab's rational
        approximation of the ``m = 6``, ``alpha = 1`` function with its
        analytic correction, the kernel CASA uses; ``"prolate_spheroidal"``,
        the generalised prolate spheroidal function of Schwab 1984 with its
        bandwidth and weight exponent chosen for the image region the
        ``fft_padding`` keeps; ``"exponential_semicircle"``, the modified
        exponential of a semicircle of Barnett et al. 2019 and Arras et al.
        2021 shaped for that region; or
        ``"least_misfit"``, the table that minimises the map error over that
        region for the lookup used, after Ye et al. 2020), ``support`` (the
        odd number of grid points the kernel spans along one axis; 7 for the
        ``prolate_spheroidal_casa`` kernel, any odd number from 3 for the others),
        ``oversampling`` (samples of the kernel table per uv cell) and
        ``interpolation`` (how the gridders read the table at a stencil
        point: ``"nearest"`` takes the nearest sample, so the lookup error,
        and with it the floor the imaging cycle converges to, falls as 1 over
        the oversampling; ``"linear"`` interpolates the two neighbouring
        samples and the error falls as 1 over its square). Missing keys take
        the defaults ``{"type": "prolate_spheroidal_casa", "support": 7,
        "oversampling": 100, "interpolation": "nearest"}``; ``None`` is all
        defaults. A ``prolate_spheroidal``, ``exponential_semicircle`` or
        ``least_misfit`` kernel whose image plane correction falls below 1e-6
        of its centre value at the edge of the image kept (a support too large
        for the ``fft_padding``, e.g. 17 at 1.05) is refused. The kernel of the
        former ``gridder="prolate_spheroidal"`` is ``"prolate_spheroidal_casa"``,
        not ``"prolate_spheroidal"``.
    clean_mask : str or None, optional
        Path to a NumPy ``.npy`` file containing one two-dimensional CLEAN mask.
        Its shape must equal ``image_params["image_size"]``. Finite values greater
        than 0.5 select pixels in every continuum residual plane.
    image_sharding : dict, optional
        Reserved for a future node-task finalization path. Continuum imaging
        currently raises ``NotImplementedError`` when this is supplied rather
        than silently discarding the requested layout at finalization.
    image_chunking : dict, optional
        Reserved for a future node-task finalization path. Continuum imaging
        currently raises ``NotImplementedError`` when this is supplied rather
        than silently discarding the requested layout at finalization.
    weight_memory_mode : {"in_memory", "in_place"}, optional
        Storage policy for calculated continuum imaging weights.
        ``"in_memory"``
        returns task-local weights to the driver and embeds them in subsequent
        graphs. ``"in_place"`` writes each task's weights into a temporary group
        in the output image store and reloads only the required partition during
        each imaging cycle. It is the default because it bounds scheduler and
        worker memory without modifying the input Processing Set. The temporary
        group is removed when the run ends, also after an error. A completed cache
        is copied into the input only when ``write_imaging_weights_to_ps=True``.
        This option remains separate from ``memory_mode`` for now; the policies
        may be unified later.

    visibility_memory_mode : {"in_memory", "in_place", "recompute"}, optional
        Continuum residual-update storage policy for observed visibility grids.
        ``"in_memory"`` retains the first-cycle grid in driver memory, while
        ``"in_place"`` stores it temporarily in the image Zarr store. Later
        cycles grid only the predicted-model contribution and subtract it from
        the cached observed grid. ``"recompute"`` instead reloads the original
        observed visibilities and grids their visibility-domain residual every
        cycle. MFS caches the globally reduced Taylor UV grid; MVC caches each
        map task's exclusively owned frequency-resolved UV planes.

    wideband_primary_beam_memory_mode : {"in_memory", "in_place", "recompute"}, optional
        MVC-only storage policy for the frequency-dependent primary beam.
        ``"in_place"`` stores it temporarily in the image Zarr store and reads
        only task-local channels. The default ``"recompute"`` regenerates the
        analytic beam inside every later map, and ``"in_memory"`` retains it at
        the driver but passes each map only its local beam. The setting is ignored
        for MFS.

    Notes
    -----
    The sky model is a Taylor expansion in frequency with
    ``image_params["n_taylor_terms"]`` terms (an integer of at least 1) about
    ``image_params["reference_frequency"]``, one frequency in Hz shared by all
    frequency chunks. Both are required.
    ``image_params["reference_frequency_hz"]`` is a deprecated alias of
    ``reference_frequency``: it is accepted with a ``FutureWarning``, and two
    different values are refused. ``image_params["frequency_coords"]`` must
    hold exactly the channels of the selected Processing Set. These parameters
    are checked before anything is written.

    The continuum Högbom adapter uses float32 for its temporary model-update
    arrays, independently of the gridding/image precision setting.

    XRADIO may normalize the output suffix to ``.img.zarr``. All subsequent
    allocation, cache access, and output use the path returned by its writer.

    Continuum primary beams are selected once from antenna metadata before
    frequency partitioning. ``image_params["primary_beam_model"]`` accepts
    ``"auto"`` (default), ``"airy"`` (physical aperture), or ``"casa_airy"``.
    Auto selects the CASA ALMA/ACA effective aperture and legacy VLA/NVSS
    prescriptions. EVLA polynomial beams are not implemented; EVLA and other
    telescopes retain the physical Airy fallback. Explicit dish/blockage lists
    and ``primary_beam_max_radius_1ghz`` override the inferred values. MFS and
    MVC select the VLA band at the Taylor reference frequency and retain that
    selection across all partitions, making the beam independent of channel
    order. CASA ``makePBImage`` instead selects the MVC beam from the first
    image channel. These choices affect only continuum imaging; cube imaging
    keeps its existing beam path.

    Returns
    -------
    dict
        Final continuum result. Stable entries are:

        ``image``
            Final in-memory image dataset. Requested products are also written
            to ``image_store``.
        ``static_xds``
            Dataset containing reusable PSF, primary-beam, and beam-fit state.
        ``model_xds`` and ``model_uv_xds``
            Accumulated image-domain model and, for MFS, its Fourier-domain
            prediction state. ``model_uv_xds`` is ``None`` for MVC.
        ``n_cycles``
            Number of model-update cycles performed.
        ``deconvolution`` and ``residual_statistics``
            Accumulated deconvolution and final residual statistics.
        ``controller``, ``stopcode``, ``stopdesc`` and ``residual_converged``
            Final cycle-controller state and convergence result.
        ``timing_node_tasks``, ``timing_model_update``, ``timing_psf_fit``,
        ``timing_restore``, ``timing_graphs`` and
        ``timing_distributed_application``
            Node, model-update, PSF-fit, restoration, graph, and driver timing
            records. Optional cache mappings may also be present for the
            selected memory policies.
    """
    import time

    import toolviper.utils.logger as logger
    from graphviper.graph_tools.coordinate_utils import (
        get_disk_chunk_sizes,
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )
    from xradio.image import make_empty_sky_image, write_image
    from xradio.measurement_set import open_processing_set

    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        _validate_n_mapping_parallelism,
    )
    from astroviper.processing_functions.imaging.check_imaging_parameters import (
        check_imaging_weights_params,
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        resolve_gridding_kernel_params,
    )
    from astroviper.processing_functions.imaging.utils import (
        CONTINUUM_IMAGING_TIMING_PHASES,
        CONTINUUM_MODEL_UPDATE_TIMING_PHASES,
        IMAGING_TIMING_TOTAL_KEY,
    )
    from astroviper.processing_functions.imaging.utils.imaging_polarization import (
        correlation_selection,
        correlations_for_stokes,
        is_four_correlation_basis,
    )
    from astroviper.utils.data_group_tools import modify_data_groups_xds
    from astroviper.utils.io import (
        image_data_groups_for_kept_variables,
        write_zarr_image_store,
    )
    from astroviper.utils.measurement_set_v2 import is_measurement_set_v2
    from astroviper.utils.timing import format_timing_summary

    if compressor is None:
        compressor = BloscCodec(cname="lz4", clevel=5)

    if n_mapping_parallelism is not None:
        _validate_n_mapping_parallelism(n_mapping_parallelism)

    if primary_beam_correction and not restore:
        raise ValueError("primary_beam_correction requires restore=True.")
    if not 0.0 <= primary_beam_limit <= 1.0:
        raise ValueError("primary_beam_limit must be between 0 and 1.")

    assert memory_mode == "in_memory", (
        "Currently only in_memory is supported for memory_mode is implemented."
    )
    if image_sharding is not None or image_chunking is not None:
        raise NotImplementedError(
            "Continuum image_sharding and image_chunking are not supported until "
            "final Taylor products are written by node tasks."
        )
    if weight_memory_mode not in ("in_memory", "in_place"):
        raise ValueError(
            "weight_memory_mode must be 'in_memory' or 'in_place'; received "
            f"{weight_memory_mode!r}."
        )
    if visibility_memory_mode not in ("in_memory", "in_place", "recompute"):
        raise ValueError(
            "visibility_memory_mode must be 'in_memory', 'in_place', or "
            f"'recompute'; received {visibility_memory_mode!r}."
        )
    if wideband_primary_beam_memory_mode not in ("in_memory", "in_place", "recompute"):
        raise ValueError(
            "wideband_primary_beam_memory_mode must be 'in_memory', 'in_place', or "
            f"'recompute'; received {wideband_primary_beam_memory_mode!r}."
        )
    if weight_memory_mode == "in_place" and skunk_works:
        raise NotImplementedError(
            "weight_memory_mode='in_place' is not yet supported with skunk_works=True."
        )
    if write_imaging_weights_to_ps and weight_memory_mode != "in_place":
        raise ValueError(
            "write_imaging_weights_to_ps=True keeps the in-place imaging weights "
            "and requires weight_memory_mode='in_place'."
        )

    # Validate continuum_mode
    continuum_mode = str(continuum_mode).lower()
    if continuum_mode not in ("mfs", "mvc"):
        raise ValueError(
            f"continuum_mode must be either 'mfs' or 'mvc'; received {continuum_mode!r}."
        )
    # Work with an application-local copy: continuum setup may augment the
    # image parameters with metadata derived from the Processing Set.
    image_params = dict(image_params)
    deconvolver = _resolve_continuum_deconvolver(deconvolver, image_params)
    _resolve_continuum_taylor_expansion(image_params)
    # Checked here, before anything is written, as cube imaging checks them
    # when it makes its kernel. Whether the padding can hold a kernel designed
    # for it is checked where the kernel is made.
    gridding_kernel_params = resolve_gridding_kernel_params(gridding_kernel_params)

    clean_mask_array = _load_continuum_clean_mask(
        clean_mask,
        image_params["image_size"],
    )

    # Validate once at the application boundary so every graph sees the same
    # normalized scope. A continuum image grids all channels onto one uv grid,
    # so Briggs and uniform weights default to the density of all channels
    # (global). The checker defaults the remaining schemes to the density of
    # each channel (local), the only scope that computes "briggs_abs", and maps
    # a requested global natural-weight calculation to its equivalent local path.
    imaging_weights_params = dict(imaging_weights_params)
    if imaging_weights_params.get("weighting") in ("briggs", "uniform"):
        imaging_weights_params.setdefault("weighting_scope", "global")
    if not check_imaging_weights_params(imaging_weights_params):
        raise ValueError("Invalid imaging_weights_params.")
    weighting = imaging_weights_params["weighting"].lower()
    weighting_scope = imaging_weights_params["weighting_scope"].lower()
    # Refused here, before the image store or the in-place weight cache exists.
    if weighting_scope == "global" and weighting not in ("briggs", "uniform"):
        raise ValueError(
            "Global weighting scope currently supports only Briggs "
            "and uniform weighting; received "
            f"{imaging_weights_params['weighting']!r}."
        )

    # When restoring, the restored sky must be created on disk and written, so
    # ensure it is in the keep list (without mutating the caller's list).
    if restore and "sky_restored" not in image_data_variables_keep:
        image_data_variables_keep = list(image_data_variables_keep) + ["sky_restored"]
    if (
        primary_beam_correction
        and "sky_restored_primary_beam_corrected" not in image_data_variables_keep
    ):
        image_data_variables_keep = list(image_data_variables_keep) + [
            "sky_restored_primary_beam_corrected"
        ]

    # One public primary-beam limit controls CLEAN support, MVC correction,
    # and restored-image blanking throughout the continuum stack.
    cycle_control_params = dict(cycle_control_params)
    cycle_control_params["primary_beam_limit"] = float(primary_beam_limit)

    # Every driver step is timed into ``timing_distributed_application``; the
    # individual per-step timing log messages are replaced by the formatted
    # summary logged just before returning.
    timing_distributed_application = {}
    application_start = time.time()

    if is_measurement_set_v2(ps_store):
        raise ValueError(
            "Continuum imaging requires an XRADIO Processing Set, not an MSv2 "
            "directory. Convert it first with "
            "xradio.measurement_set.convert_msv2_to_processing_set."
        )

    # The residual update grids in the instrument basis and the model update
    # deconvolves in Stokes, so the requested Stokes planes fix the correlations
    # that are loaded and gridded. Checked here, before anything is created.
    correlation_coords = correlations_for_stokes(
        image_params["polarization_coords"], instrument_polarization_basis
    )
    # MVC makes its Taylor terms real, which drops the imaginary part of the
    # cross-hand images (V for linear feeds, U for circular feeds).
    if continuum_mode == "mvc" and is_four_correlation_basis(correlation_coords):
        raise ValueError(
            f"continuum_mode='mvc' does not image polarization_coords "
            f"{list(image_params['polarization_coords'])}: request the two "
            "parallel hands or use continuum_mode='mfs'."
        )

    start = time.time()
    ps_xdt = open_processing_set(ps_store, scan_intents=scan_intents)
    timing_distributed_application["T_open_processing_set"] = time.time() - start

    if len(ps_xdt) == 0:
        raise ValueError(
            "The Processing Set is empty after applying scan_intents="
            f"{scan_intents!r}. Check the available scan intents before imaging."
        )

    # Every measurement set must hold the needed correlations; checked here,
    # before anything is written. The index positions of the needed
    # correlations are added to the data selection of every task below.
    correlation_selections = {
        ms_name: correlation_selection(
            ms_xdt.polarization.values, correlation_coords, ms_name
        )
        for ms_name, ms_xdt in ps_xdt.items()
    }

    # Create an empty image on disk with the correct coordinates and dimensions.
    start = time.time()
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=image_params["frequency_coords"],
        pol_coords=image_params["polarization_coords"],
        time_coords=image_params["time_coords"],
        spectral_reference=image_params.get("spectral_reference", "lsrk"),
        do_sky_coords=False,
    )
    timing_distributed_application["T_make_empty_image_xds"] = time.time() - start

    # Register the image data groups for the kept variables so the on-disk
    # store carries the same group layout the node tasks build in memory
    # (make_empty_sky_image only stamps an empty "base" placeholder, which is
    # dropped here in favor of the real groups).
    img_xds.attrs.get("data_groups", {}).pop("base", None)
    for data_group_name, data_group in image_data_groups_for_kept_variables(
        image_data_variables_keep
    ).items():
        modify_data_groups_xds(
            img_xds,
            data_group_out_name=data_group_name,
            data_group_out=data_group,
            description="Created by the image_continuum_single_field driver; "
            "populated by its node tasks.",
        )

    # Determine the mapping parallelism. Continuum imaging maps only over
    # frequency, matching the cube driver's public dictionary interface.
    start = time.time()
    n_frequency_chunks = int(
        calculate_number_of_chunks_for_continuum_imaging(
            img_xds,
            single_precision_image,
            None
            if n_mapping_parallelism is None
            else n_mapping_parallelism["frequency"],
            thread_info,
            n_taylor_terms=image_params["n_taylor_terms"],
            n_correlations=len(correlation_coords),
            fft_padding=image_params["fft_padding"],
        )
    )

    # Make Parallel Coords
    parallel_coords = {}
    parallel_coords["frequency"] = make_parallel_coord(
        coord=img_xds.frequency, n_chunks=n_frequency_chunks
    )
    logger.info(
        "Number of frequency chunks ... : "
        + str(len(parallel_coords["frequency"]["data_chunks"]))
    )
    timing_distributed_application["T_determine_chunks_and_parallel_coords"] = (
        time.time() - start
    )

    # The image channels (frequency_coords) must be exactly the Processing Set
    # channels; checked here, before the image store is written.
    start = time.time()
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps_xdt
    )
    node_task_data_mapping = _apply_exact_frequency_selection_to_continuum_mapping(
        node_task_data_mapping,
        ps_xdt,
    )
    timing_distributed_application["T_interpolate_data_coords"] = time.time() - start

    start = time.time()
    image_store = write_zarr_image_store(img_xds, image_store, overwrite=overwrite)
    timing_distributed_application["T_write_empty_image"] = time.time() - start

    zarr_meta = {}

    input_params = {}
    input_params["image_params"] = image_params
    input_params["imaging_weights_params"] = imaging_weights_params
    input_params["zarr_meta"] = zarr_meta
    input_params["to_disk"] = True
    input_params["polarization"] = img_xds.polarization.data
    input_params["time"] = [0]
    input_params["compressor"] = compressor
    input_params["image_store"] = image_store
    input_params["input_data_store"] = ps_store
    input_params["processing_set_data_group_name"] = processing_set_data_group_name
    input_params["image_data_variables_keep"] = image_data_variables_keep
    input_params["memory_mode"] = memory_mode
    input_params["weight_memory_mode"] = weight_memory_mode
    input_params["visibility_memory_mode"] = visibility_memory_mode
    input_params["wideband_primary_beam_memory_mode"] = (
        wideband_primary_beam_memory_mode
    )
    input_params["cache_directory"] = cache_directory
    input_params["write_visibility_model_to_ps"] = write_visibility_model_to_ps
    input_params["write_imaging_weights_to_ps"] = write_imaging_weights_to_ps
    input_params["clear_cache"] = clear_cache
    input_params["processing_function_threads"] = processing_function_threads
    input_params["cycle_control_params"] = cycle_control_params
    input_params["gridder"] = gridder
    input_params["gridding_kernel_params"] = gridding_kernel_params
    input_params["is_first_cycle"] = True
    input_params["deconvolver"] = deconvolver
    input_params["primary_beam_correction"] = bool(primary_beam_correction)
    input_params["primary_beam_limit"] = float(primary_beam_limit)
    input_params["continuum_mode"] = continuum_mode
    input_params["instrument_polarization_basis"] = instrument_polarization_basis
    input_params["single_precision_image"] = single_precision_image
    input_params["fft_backend"] = fft_backend
    input_params["restore"] = restore
    input_params["skunk_works"] = skunk_works
    input_params["task_time_kill_switch_seconds"] = task_time_kill_switch_seconds

    controller = _make_continuum_cycle_controller(cycle_control_params)

    from astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam import (
        resolve_continuum_primary_beam,
    )

    # Resolve telescope/band once before task partitioning. Every MVC chunk
    # must use the prescription selected from the full image frequency axis.
    image_params = resolve_continuum_primary_beam(
        image_params,
        ps_xdt.xr_ps.get_combined_antenna_xds(),
        continuum_mode=continuum_mode,
    )
    input_params["image_params"] = image_params

    # Node-task loaders need the resolved role->variable mapping. In addition to
    # supporting the direct-Zarr path, this lets cached-grid MFS cycles omit the
    # observed correlated-data variable from ordinary eager loads.
    first_ms = next(iter(ps_xdt.values()))
    input_params["data_group"] = first_ms.ds.attrs["data_groups"][
        processing_set_data_group_name
    ]

    # Load only the correlations that are needed: their index positions are
    # added to the selection of every measurement set that holds more (or
    # differently ordered) correlations, for the weight and the imaging
    # graphs alike. A set that holds exactly the needed ones keeps its
    # selection unchanged.
    for ms_name, selection in correlation_selections.items():
        if selection is None:
            continue
        for task in node_task_data_mapping.values():
            if task["data_selection"].get(ms_name) is not None:
                task["data_selection"][ms_name]["polarization"] = selection

    if continuum_mode == "mvc" and wideband_primary_beam_memory_mode == "in_place":
        start = time.time()
        _create_wideband_primary_beam_cache_store(
            image_store,
            img_xds,
            image_params["frequency_coords"],
            correlation_coords,
            single_precision_image,
            compressor,
        )
        timing_distributed_application["T_create_in_place_widebandpb_cache"] = (
            time.time() - start
        )

    if continuum_mode == "mfs" and visibility_memory_mode == "in_place":
        start = time.time()
        _create_mfs_visibility_grid_cache_store(
            image_store,
            img_xds,
            image_params,
            correlation_coords,
            single_precision_image,
            compressor,
        )
        timing_distributed_application["T_create_in_place_visibility_cache"] = (
            time.time() - start
        )

    if continuum_mode == "mvc" and visibility_memory_mode == "in_place":
        start = time.time()
        _create_mvc_visibility_grid_cache_store(
            image_store,
            img_xds,
            image_params,
            image_params["frequency_coords"],
            correlation_coords,
            single_precision_image,
            compressor,
        )
        timing_distributed_application["T_create_in_place_visibility_cache"] = (
            time.time() - start
        )

    # Imaging weights already registered in the data group (kept or left
    # behind by an earlier run, or written by another tool) are not used.
    registered_weight_names = set()
    for ms_xdt in ps_xdt.values():
        data_group = ms_xdt.ds.attrs["data_groups"].get(
            processing_set_data_group_name, {}
        )
        if "weight_imaging" in data_group:
            registered_weight_names.add(data_group["weight_imaging"])
    if registered_weight_names:
        logger.warning(
            f"Data group {processing_set_data_group_name!r} registers the imaging "
            f"weights {sorted(registered_weight_names)}. They are not used: this "
            "run calculates its own imaging weights from imaging_weights_params."
        )

    weight_cache_is_active = False
    if weight_memory_mode == "in_place":
        start = time.time()
        _create_continuum_weight_cache_store(
            ps_xdt,
            ps_store,
            image_store,
            processing_set_data_group_name,
            compressor,
        )
        timing_distributed_application["T_create_in_place_weight_cache"] = (
            time.time() - start
        )

    # The temporary disk-backed weights are removed however the run ends.
    # The input Processing Set is touched only after a successful run and only
    # when write_imaging_weights_to_ps=True.
    try:
        # Auto-detect native on-disk chunk sizes if not supplied by the caller.
        if disk_chunk_sizes == "Auto":
            disk_chunk_sizes = get_disk_chunk_sizes(ps_xdt, parallel_coords)
            logger.info("Auto-detected disk chunk sizes: " + str(disk_chunk_sizes))
        elif isinstance(disk_chunk_sizes, str):
            # If disk_chunk_sizes is a string but not "Auto", treat as None
            disk_chunk_sizes = None

        # frequency_coords is not used by node tasks (they use task_coords["frequency"]["data"])
        # so remove it to avoid embedding the full frequency axis in every task in the graph.
        input_params["image_params"] = {
            k: v for k, v in image_params.items() if k != "frequency_coords"
        }

        task_priorities = None

        # =============================================================
        # Distributed imaging-cycle loop
        # =============================================================

        # These timing entries accumulate over all imaging cycles
        timing_distributed_application["T_create_map_reduce_append_graph"] = 0.0

        timing_distributed_application["T_generate_dask_graph"] = 0.0

        timing_distributed_application["T_compute_dask_graph"] = 0.0

        # State carried between independently computed distributed graphs.
        is_first_cycle = True
        static_xds = None
        model_xds = None
        model_uv_xds = None
        observed_visibility_grid_xds = None
        observed_visibility_grid_mapping = None
        last_cycle_return_dict = None
        n_cycles = 0
        timing_graphs = []

        # =============================================================
        # Prepare imaging weights once before entering the imaging-cycle loop.
        # =============================================================

        weight_preparation_input_params = {
            "image_params": input_params["image_params"],
            "imaging_weights_params": imaging_weights_params,
            "input_data_store": ps_store,
            "processing_set_data_group_name": (processing_set_data_group_name),
            "instrument_polarization_basis": (instrument_polarization_basis),
            "processing_function_threads": (processing_function_threads),
            "weight_memory_mode": weight_memory_mode,
            "image_store": image_store,
            "data_group": input_params["data_group"],
        }

        if skunk_works:
            # The preparation node shown above currently uses the standard loader.
            logger.warning(
                "Imaging-weight preparation currently uses the standard "
                "processing-set loader even though skunk_works=True."
            )

        if weighting in ("natural", "briggs", "briggs_abs", "uniform"):
            if weighting_scope == "global":
                (
                    weight_return_dict,
                    weight_graph_timings,
                ) = prepare_continuum_imaging_weights_global(
                    ps_xdt=ps_xdt,
                    node_task_data_mapping=node_task_data_mapping,
                    input_params=weight_preparation_input_params,
                    disk_chunk_sizes=disk_chunk_sizes,
                    processing_set_data_group_name=(processing_set_data_group_name),
                    monitor_resources_seconds=monitor_resources_seconds,
                    task_priorities=task_priorities,
                    reduce_mode=reduce_mode,
                    reduce_n_batch=reduce_n_batch,
                )
                if weight_memory_mode == "in_place":
                    weight_return_dict["weight_cache_mapping"] = None
                    weight_cache_is_active = True
            else:
                # Local weights are calculated inside every first-imaging-cycle map
                # task, returned as a cache, and reattached in later cycles.
                weight_return_dict = {"weight_cache_mapping": None}
                weight_graph_timings = {}

        else:
            raise ValueError(
                "Unsupported imaging weighting scheme "
                f"{imaging_weights_params['weighting']!r}."
            )

        _accumulate_graph_timings(
            timing_distributed_application,
            weight_graph_timings,
        )
        timing_graphs.extend(weight_return_dict.pop("timing_graphs", []))

        weight_cache_mapping = weight_return_dict["weight_cache_mapping"]
        pb_cache_mapping = None

        # ---------------------------------------------------------
        # Main loop
        # ---------------------------------------------------------

        # Model-update stopping is provisional. The next residual graph verifies
        # it and its append either continues cleaning or finalizes that residual.
        while True:
            n_cycles += 1

            logger.debug(f"Starting continuum imaging cycle {n_cycles}.")

            # ---------------------------------------------------------
            # Configure the distributed residual/imaging-cycle map tasks.
            # ---------------------------------------------------------
            cycle_input_params = dict(input_params)

            cycle_input_params["is_first_cycle"] = is_first_cycle
            cycle_input_params["restore"] = False

            # Prepared once before the imaging-cycle loop. In-memory weights
            # are attached task by task below so GraphViper does not deep-copy
            # the complete all-task cache into every map-node parameter dict.
            cycle_input_params["weight_cache_is_active"] = weight_cache_is_active

            if not is_first_cycle:
                if model_xds is None:
                    raise RuntimeError(
                        "No accumulated continuum model is available for "
                        f"imaging cycle {n_cycles}."
                    )

                if continuum_mode == "mfs" and model_uv_xds is None:
                    raise RuntimeError("No Fourier-domain MFS model is available.")

                if static_xds is None:
                    raise RuntimeError(
                        "No static continuum products are available for "
                        f"imaging cycle {n_cycles}."
                    )

                cycle_input_params["model_uv_xds"] = model_uv_xds

                if continuum_mode == "mvc":
                    cycle_input_params["model_xds"] = model_xds

            # During the first imaging cycle the PSF and residual Taylor products
            # are reduced. Later cycles only produce new residual products; the
            # static PSF/PB products are supplied by
            # continuum_cycle_append_node.
            if continuum_mode == "mfs":
                if is_first_cycle:
                    reduce_input_params = {"continuum_mode": "mfs"}
                else:
                    reduce_input_params = {
                        "continuum_mode": "mfs",
                        "additive_variables": (
                            "VISIBILITY",
                            "VISIBILITY_NORMALIZATION",
                        ),
                    }
            else:
                if is_first_cycle:
                    reduce_input_params = {
                        "continuum_mode": "mvc",
                        "additive_variables": (
                            "MVC_RESIDUAL_TAYLOR_NUMERATOR",
                            "MVC_RESIDUAL_WEIGHT_SUM",
                            "MVC_RESIDUAL_BEAM_SQUARED_TAYLOR_SUM",
                            "MVC_RESIDUAL_WEIGHT_TAYLOR_SUM",
                            "MVC_RESIDUAL_BEAM_CHANNEL_COUNT",
                            "MVC_PSF_TAYLOR_NUMERATOR",
                            "MVC_PSF_WEIGHT_SUM",
                            "MVC_PRIMARY_BEAM_WEIGHTED_SUM",
                        ),
                    }
                else:
                    reduce_input_params = {
                        "continuum_mode": "mvc",
                        "additive_variables": (
                            "MVC_RESIDUAL_TAYLOR_NUMERATOR",
                            "MVC_RESIDUAL_WEIGHT_SUM",
                            "MVC_RESIDUAL_BEAM_SQUARED_TAYLOR_SUM",
                            "MVC_RESIDUAL_WEIGHT_TAYLOR_SUM",
                            "MVC_RESIDUAL_BEAM_CHANNEL_COUNT",
                        ),
                    }

            # ---------------------------------------------------------
            # Configure the global continuum model-update append node.
            # ---------------------------------------------------------
            append_input_params = {
                "cycle_control_params": cycle_control_params,
                "deconvolver": deconvolver,
                "processing_function_threads": processing_function_threads,
                "is_first_cycle": is_first_cycle,
                "controller": controller,
                "image_data_group_in_name": "residual",
                "image_data_group_out_name": "model",
                "image_params": image_params,
                "gridding_kernel_params": gridding_kernel_params,
                "image_data_variables_keep": image_data_variables_keep,
                "fft_backend": fft_backend,
                "single_precision_image": single_precision_image,
                "instrument_polarization_basis": instrument_polarization_basis,
                "continuum_mode": continuum_mode,
                "visibility_memory_mode": visibility_memory_mode,
                "image_store": image_store,
                "primary_beam_limit": primary_beam_limit,
                "clean_mask": clean_mask_array,
                "restore": restore,
                "primary_beam_correction": primary_beam_correction,
            }

            # Static products from the first cycle are required in later cycles.
            # This holds static quantities such as PSF and PB
            if not is_first_cycle:
                append_input_params["static_xds"] = static_xds
                append_input_params["model_xds"] = model_xds
                append_input_params["model_uv_xds"] = model_uv_xds
                append_input_params["deconvolution"] = last_cycle_return_dict[
                    "deconvolution"
                ]
                if continuum_mode == "mfs" and visibility_memory_mode == "in_memory":
                    if observed_visibility_grid_xds is None:
                        raise RuntimeError(
                            "No cached observed-data MFS grid is available for "
                            f"imaging cycle {n_cycles}."
                        )
                    append_input_params["observed_visibility_grid_xds"] = (
                        observed_visibility_grid_xds
                    )

            # ---------------------------------------------------------
            # Execute one imaging cycle followed by one model update.
            # ---------------------------------------------------------

            # The append node completes this imaging cycle after the reduction.
            cycle_node_task_data_mapping = node_task_data_mapping
            if weight_cache_mapping is not None:
                cycle_node_task_data_mapping = _mapping_with_task_weights(
                    cycle_node_task_data_mapping,
                    weight_cache_mapping,
                )
            if (
                continuum_mode == "mvc"
                and not is_first_cycle
                and wideband_primary_beam_memory_mode == "in_memory"
            ):
                cycle_node_task_data_mapping = _mapping_with_task_primary_beams(
                    cycle_node_task_data_mapping,
                    pb_cache_mapping,
                )
            if (
                continuum_mode == "mvc"
                and not is_first_cycle
                and visibility_memory_mode == "in_memory"
            ):
                cycle_node_task_data_mapping = _mapping_with_task_observed_grids(
                    cycle_node_task_data_mapping,
                    observed_visibility_grid_mapping,
                )

            cycle_return_dict, graph_timings = compute_continuum_graph(
                ps_xdt=ps_xdt,
                node_task_data_mapping=cycle_node_task_data_mapping,
                cycle_input_params=cycle_input_params,
                reduce_input_params=reduce_input_params,
                disk_chunk_sizes=disk_chunk_sizes,
                processing_set_data_group_name=processing_set_data_group_name,
                monitor_resources_seconds=monitor_resources_seconds,
                task_priorities=task_priorities,
                reduce_mode=reduce_mode,
                reduce_n_batch=reduce_n_batch,
                append_node=node_tasks.imaging.continuum_cycle_append_node,
                append_input_params=append_input_params,
            )

            # Gather timing information while retaining this graph's task stream.
            _accumulate_graph_timings(
                timing_distributed_application,
                graph_timings,
            )
            is_final_graph = bool(cycle_return_dict["residual_converged"])
            if is_final_graph:
                graph_stage = (
                    "final residual + restoration" if restore else "final residual"
                )
            else:
                graph_stage = f"imaging cycle {n_cycles} (residual + model update)"
            record = _graph_timing_record(
                graph_stage,
                cycle_return_dict,
                graph_timings,
                stage_kind="final" if is_final_graph else "imaging_cycle",
                includes_restoration=is_final_graph and restore,
            )
            if record is not None:
                timing_graphs.append(record)

            # Get current status for bookkeeping
            last_cycle_return_dict = cycle_return_dict
            controller = cycle_return_dict["controller"]

            # ---------------------------------------------------------
            # Capture append-prepared state for the next cycle.
            # ---------------------------------------------------------
            if is_first_cycle:
                if "static_xds" not in cycle_return_dict:
                    raise KeyError(
                        "The first continuum append node did not return 'static_xds'."
                    )

                # PSF, PB and PSF sidelobe level are computed only in cycle one.
                static_xds = cycle_return_dict["static_xds"]

                if continuum_mode == "mfs" and visibility_memory_mode == "in_memory":
                    observed_visibility_grid_xds = cycle_return_dict.get(
                        "observed_visibility_grid_xds"
                    )
                    if observed_visibility_grid_xds is None:
                        raise RuntimeError(
                            "The first MFS cycle did not return its globally reduced "
                            "observed-data visibility-grid cache."
                        )

                if continuum_mode == "mvc" and visibility_memory_mode == "in_memory":
                    observed_visibility_grid_mapping = cycle_return_dict.get(
                        "observed_visibility_grid_mapping"
                    )
                    if observed_visibility_grid_mapping is None:
                        raise RuntimeError(
                            "The first MVC cycle did not return its task-local "
                            "observed visibility-grid caches."
                        )
                    expected_task_ids = {
                        int(task_id) for task_id in node_task_data_mapping
                    }
                    actual_task_ids = {
                        int(task_id) for task_id in observed_visibility_grid_mapping
                    }
                    if actual_task_ids != expected_task_ids:
                        raise RuntimeError(
                            "The first MVC cycle returned an incomplete observed "
                            "visibility-grid cache: "
                            f"expected={sorted(expected_task_ids)}, "
                            f"received={sorted(actual_task_ids)}."
                        )

            if "model_xds" not in cycle_return_dict:
                raise KeyError(
                    "The continuum append node did not return accumulated "
                    "'model_xds' state."
                )
            if "model_uv_xds" not in cycle_return_dict:
                raise KeyError(
                    "The continuum append node did not return 'model_uv_xds' state."
                )

            model_xds = cycle_return_dict["model_xds"]
            model_uv_xds = cycle_return_dict["model_uv_xds"]

            if continuum_mode == "mfs" and model_uv_xds is None:
                raise RuntimeError(
                    "The MFS append node did not prepare the Fourier-domain model."
                )
            if continuum_mode == "mvc" and model_uv_xds is not None:
                raise RuntimeError(
                    "The MVC append node returned unexpected MFS UV state."
                )

            # Locally calculated weights either return to the driver or become
            # visible through the Processing Set after the first graph completes.
            if (
                is_first_cycle
                and weight_memory_mode == "in_memory"
                and weight_cache_mapping is None
            ):
                weight_cache_mapping = cycle_return_dict.get("weight_cache_mapping")

                if weight_cache_mapping is None:
                    raise RuntimeError(
                        "The first imaging cycle did not return the locally calculated "
                        "imaging-weight cache."
                    )

                expected_task_ids = {int(task_id) for task_id in node_task_data_mapping}
                actual_task_ids = {int(task_id) for task_id in weight_cache_mapping}

                if actual_task_ids != expected_task_ids:
                    raise RuntimeError(
                        "The first imaging cycle returned an incomplete imaging-weight "
                        "cache: "
                        f"expected={sorted(expected_task_ids)}, "
                        f"received={sorted(actual_task_ids)}."
                    )

            if (
                is_first_cycle
                and weight_memory_mode == "in_place"
                and not weight_cache_is_active
            ):
                weight_cache_is_active = True

            # Store frequency-dependent primary beam in the cache if running as mvc
            if (
                continuum_mode == "mvc"
                and is_first_cycle
                and wideband_primary_beam_memory_mode == "in_memory"
            ):
                pb_cache_mapping = cycle_return_dict.get("pb_cache_mapping")
                if pb_cache_mapping is None:
                    raise RuntimeError(
                        "The first MVC imaging cycle did not return "
                        "the frequency-dependent PB cache."
                    )
                expected_task_ids = {int(task_id) for task_id in node_task_data_mapping}
                actual_task_ids = {int(task_id) for task_id in pb_cache_mapping}
                if actual_task_ids != expected_task_ids:
                    raise RuntimeError(
                        "The first MVC imaging cycle returned an incomplete primary-beam "
                        "cache: "
                        f"expected={sorted(expected_task_ids)}, "
                        f"received={sorted(actual_task_ids)}."
                    )

            is_first_cycle = False
            if cycle_return_dict["residual_converged"]:
                # Verification does not constitute a model update or consume a cycle.
                n_cycles -= 1
                final_return_dict = cycle_return_dict
                logger.debug(
                    "Continuum stopped after refreshed residual verification: "
                    f"{cycle_return_dict['stopdesc']}"
                )
                break

        # =============================================================
        # Assemble the final application result
        # =============================================================

        return_dict = final_return_dict

        # The final imaging-cycle graph computes the final residual/restored image,
        # while the accumulated model comes from all preceding model updates.
        return_dict["image"]["SKY_MODEL"] = model_xds["SKY_MODEL"].copy(deep=True)

        # Final append retains cumulative deconvolution history and reports the
        # stop code from the refreshed residual, without a fictitious model update.

        return_dict["static_xds"] = static_xds
        return_dict["n_cycles"] = n_cycles

        # The initial on-disk arrays are cube-shaped NaN placeholders.  Continuum
        # finalization instead produces Taylor-term products, so replace the store
        # with the finalized dataset and let its real dimensions define the schema.
        start = time.time()
        output_image = _continuum_image_for_disk(
            return_dict["image"],
            image_data_variables_keep,
        )
        if write_imaging_weights_to_ps:
            persist_start = time.time()
            _persist_continuum_weight_cache_to_processing_set(
                ps_xdt,
                ps_store,
                image_store,
                processing_set_data_group_name,
            )
            timing_distributed_application["T_persist_imaging_weights"] = (
                time.time() - persist_start
            )
        if continuum_mode == "mvc" and wideband_primary_beam_memory_mode == "in_place":
            _remove_wideband_primary_beam_cache(image_store)
        if continuum_mode == "mvc" and visibility_memory_mode == "in_place":
            _remove_mvc_visibility_grid_cache(image_store)
        if continuum_mode == "mfs" and visibility_memory_mode == "in_place":
            _remove_mfs_visibility_grid_cache(image_store)
        write_image(
            output_image,
            imagename=image_store,
            out_format="zarr",
            overwrite=True,
            compressor=compressor,
        )
        timing_distributed_application["T_write_final_image"] = time.time() - start

        # Consolidate metadata
        start = time.time()
        zarr.consolidate_metadata(image_store)
        timing_distributed_application["T_consolidate_metadata"] = time.time() - start
    finally:
        if weight_memory_mode == "in_place":
            start = time.time()
            _remove_continuum_weight_cache(image_store)
            timing_distributed_application["T_remove_in_place_weight_cache"] = (
                time.time() - start
            )

    timing_distributed_application["T_total"] = time.time() - application_start

    # The reduce already produced ``{"timing_node_tasks", "deconvolution"}``; add
    # the driver-level timing so the full return dict carries timing for both the
    # distributed application (this driver) and the per-chunk node tasks.
    return_dict["timing_distributed_application"] = timing_distributed_application
    return_dict["timing_graphs"] = timing_graphs

    # Driver-level ("distributed application") timing breakdown.
    logger.info(
        format_timing_summary(
            timing_distributed_application,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            total_key=DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
            title="AstroVIPER distributed-application timing (driver, seconds)",
            total_label="TOTAL (driver wall time)",
        )
    )

    # Summarize every map task from every graph, not just the final residual
    # graph retained under the historical ``timing_node_tasks`` return key.
    timing_node_tasks = _combine_graph_timing_frames(
        timing_graphs,
        "timing_node_tasks",
    )
    if timing_node_tasks is None:
        timing_node_tasks = return_dict["timing_node_tasks"]
    logger.info(
        format_timing_summary(
            timing_node_tasks.mean(numeric_only=True).to_dict(),
            CONTINUUM_IMAGING_TIMING_PHASES,
            total_key=IMAGING_TIMING_TOTAL_KEY,
            title="AstroVIPER continuum task timing: MEAN over all graph tasks (seconds)",
        )
    )
    logger.info(
        format_timing_summary(
            timing_node_tasks.max(numeric_only=True).to_dict(),
            CONTINUUM_IMAGING_TIMING_PHASES,
            total_key=IMAGING_TIMING_TOTAL_KEY,
            title="AstroVIPER continuum task timing: MAX over all graph tasks (seconds)",
        )
    )

    timing_model_updates = _combine_graph_timing_frames(
        timing_graphs,
        "timing_model_update",
    )
    if timing_model_updates is not None:
        logger.info(
            format_timing_summary(
                timing_model_updates.sum(numeric_only=True).to_dict(),
                CONTINUUM_MODEL_UPDATE_TIMING_PHASES,
                total_key="T_model_update_node_task",
                title="AstroVIPER continuum model-update timing (all cycles, seconds)",
            )
        )

    return return_dict
