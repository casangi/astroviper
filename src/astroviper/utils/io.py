full_dims_lm = ["time", "frequency", "polarization", "l", "m"]
full_dims_uv = ["time", "frequency", "polarization", "u", "v"]
norm_dims = ["time", "frequency", "polarization"]

imaging_data_variables_and_dims_double_precision = {
    "aperture": {"dims": full_dims_uv, "dtype": "<c16", "name": "APERTURE"},
    "aperture_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "APERTURE_NORMALIZATION",
    },
    "primary_beam": {"dims": full_dims_lm, "dtype": "<f8", "name": "PRIMARY_BEAM"},
    "uv_sampling": {"dims": full_dims_uv, "dtype": "<c16", "name": "UV_SAMPLING"},
    "uv_sampling_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "UV_SAMPLING_NORMALIZATION",
    },
    "point_spread_function": {
        "dims": full_dims_lm,
        "dtype": "<f8",
        "name": "POINT_SPREAD_FUNCTION",
    },
    "visibility": {"dims": full_dims_uv, "dtype": "<c16", "name": "VISIBILITY"},
    "visibility_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "VISIBILITY_NORMALIZATION",
    },
    "sky_deconvolved": {
        "dims": full_dims_lm,
        "dtype": "<f8",
        "name": "SKY_DECONVOLVED",
    },
    "sky_dirty": {"dims": full_dims_lm, "dtype": "<f8", "name": "SKY_DIRTY"},
    "sky_model": {"dims": full_dims_lm, "dtype": "<f8", "name": "SKY_MODEL"},
    "sky_residual": {"dims": full_dims_lm, "dtype": "<f8", "name": "SKY_RESIDUAL"},
    "mask": {"dims": full_dims_lm, "dtype": "<i8", "name": "MASK"},
}

imaging_data_variables_and_dims_single_precision = {
    "aperture": {"dims": full_dims_uv, "dtype": "<c8", "name": "APERTURE"},
    "aperture_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "APERTURE_NORMALIZATION",
    },
    "primary_beam": {"dims": full_dims_lm, "dtype": "<f4", "name": "PRIMARY_BEAM"},
    "uv_sampling": {"dims": full_dims_uv, "dtype": "<c8", "name": "UV_SAMPLING"},
    "uv_sampling_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "UV_SAMPLING_NORMALIZATION",
    },
    "point_spread_function": {
        "dims": full_dims_lm,
        "dtype": "<f4",
        "name": "POINT_SPREAD_FUNCTION",
    },
    "visibility": {"dims": full_dims_uv, "dtype": "<c8", "name": "VISIBILITY"},
    "visibility_normalization": {
        "dims": norm_dims,
        "dtype": "<c16",
        "name": "VISIBILITY_NORMALIZATION",
    },
    "sky_deconvolved": {
        "dims": full_dims_lm,
        "dtype": "<f4",
        "name": "SKY_DECONVOLVED",
    },
    "sky_dirty": {"dims": full_dims_lm, "dtype": "<f4", "name": "SKY_DIRTY"},
    "sky_model": {"dims": full_dims_lm, "dtype": "<f4", "name": "SKY_MODEL"},
    "sky_residual": {"dims": full_dims_lm, "dtype": "<f4", "name": "SKY_RESIDUAL"},
    "mask": {"dims": full_dims_lm, "dtype": "<i4", "name": "MASK"},
}


def create_empty_data_variables_on_disk(
    zarr_store,
    data_variables,
    shape_dict,
    parallel_coords,
    compressor,
    double_precision,
    data_variable_definitions,
):
    """Create multiple empty data variables on disk.

    No data is allocated in memory; only zarr metadata and array structure are
    written. Unwritten chunks return ``fill_value=nan`` on read until overwritten
    with actual data.

    Parameters
    ----------
    zarr_store : str
        Path to the zarr store on disk.
    data_variables : list of str
        Names of the data variables to create
        (e.g. ``["sky", "point_spread_function"]``).
    shape_dict : dict
        Mapping of dimension name to size for all dimensions used by the
        requested data variables
        (e.g. ``{"time": 1, "frequency": 5, "polarization": 1, "l": 250, "m": 250}``).
    parallel_coords : dict
        Parallel coordinates dictionary as returned by
        :func:`~graphviper.graph_tools.coordinate_utils.make_parallel_coord`.
        Used to determine the chunk size along each parallelized dimension.
    compressor : numcodecs compressor or None
        Compressor applied to every chunk when writing.
        Set to ``None`` for no compression.
    double_precision : bool
        If ``True``, use double precision dtypes; otherwise single precision.
    data_variable_definitions : dict or str
        Dictionary mapping variable names to their definition dicts (with keys
        ``"dims"``, ``"dtype"``, ``"name"``), or the string ``"imaging"`` to
        select the built-in imaging variable definitions.
    """
    import zarr
    import numpy as np

    _ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3
    group = zarr.open_group(zarr_store, mode="r+")

    if isinstance(data_variable_definitions, dict):
        pass
    if data_variable_definitions == "imaging" and double_precision:
        data_variable_definitions = imaging_data_variables_and_dims_double_precision
    elif data_variable_definitions == "imaging" and not double_precision:
        data_variable_definitions = imaging_data_variables_and_dims_single_precision

    for dv in data_variables:
        dv_def = data_variable_definitions[dv]
        dims = dv_def["dims"]

        shape = tuple(shape_dict[dim] for dim in dims)

        chunks = []
        for d in dims:
            if d in parallel_coords:
                chunks.append(len(parallel_coords[d]["data_chunks"][0]))
            else:
                chunks.append(shape_dict[d])

        _kwargs = dict(
            shape=shape,
            chunks=chunks,
            dtype=dv_def["dtype"],
            fill_value=np.nan,
        )

        dv_name = dv_def["name"]
        if _ZARR_V3:
            sky = group.require_array(
                dv_name, **_kwargs, compressors=[compressor] if compressor else []
            )
        else:
            sky = group.require_dataset(dv_name, **_kwargs, compressor=compressor)

        group[dv].attrs["_ARRAY_DIMENSIONS"] = data_variable_definitions[dv]["dims"]

    zarr.consolidate_metadata(zarr_store)


