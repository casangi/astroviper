full_dims_lm = ["time", "frequency", "polarization", "l", "m"]
full_dims_uv = ["time", "frequency", "polarization", "u", "v"]
norm_dims = ["time", "frequency", "polarization"]
beam_params_dims = ["time", "frequency", "polarization", "beam_params_label"]

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
    "sky_restored": {"dims": full_dims_lm, "dtype": "<f8", "name": "SKY_RESTORED"},
    "sky": {"dims": full_dims_lm, "dtype": "<f8", "name": "SKY"},
    "mask": {
        "dims": full_dims_lm,
        "dtype": "|i1",
        "name": "MASK",
        "attrs": {"dtype": "bool"},
    },
    "beam_fit_params_point_spread_function": {
        "dims": beam_params_dims,
        "dtype": "<f8",
        "name": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
    },
    "beam_fit_params_sky_residual": {
        "dims": beam_params_dims,
        "dtype": "<f8",
        "name": "BEAM_FIT_PARAMS_SKY_RESIDUAL",
    },
    "beam_fit_params_sky_dirty": {
        "dims": beam_params_dims,
        "dtype": "<f8",
        "name": "BEAM_FIT_PARAMS_SKY_DIRTY",
    },
    "beam_fit_params_sky_deconvolved": {
        "dims": beam_params_dims,
        "dtype": "<f8",
        "name": "BEAM_FIT_PARAMS_SKY_DECONVOLVED",
    },
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
    "sky_restored": {"dims": full_dims_lm, "dtype": "<f4", "name": "SKY_RESTORED"},
    "sky": {"dims": full_dims_lm, "dtype": "<f4", "name": "SKY"},
    "mask": {
        "dims": full_dims_lm,
        "dtype": "|i1",
        "name": "MASK",
        "attrs": {"dtype": "bool"},
    },
    "beam_fit_params_point_spread_function": {
        "dims": beam_params_dims,
        "dtype": "<f4",
        "name": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
    },
    "beam_fit_params_sky_residual": {
        "dims": beam_params_dims,
        "dtype": "<f4",
        "name": "BEAM_FIT_PARAMS_SKY_RESIDUAL",
    },
    "beam_fit_params_sky_dirty": {
        "dims": beam_params_dims,
        "dtype": "<f4",
        "name": "BEAM_FIT_PARAMS_SKY_DIRTY",
    },
    "beam_fit_params_sky_deconvolved": {
        "dims": beam_params_dims,
        "dtype": "<f4",
        "name": "BEAM_FIT_PARAMS_SKY_DECONVOLVED",
    },
}


def write_result_chunk_to_disk_using_zarr(
    image_store, image_data_variables_keep, task_coords, img_xds
):
    import zarr
    import time

    for dv in image_data_variables_keep:
        dv = dv.upper()
        # size_dict = img_xds.sizes
        idx = []
        for dim in img_xds[dv].dims:
            if dim in task_coords:
                idx.append(task_coords[dim]["slice"])
            else:
                idx.append(slice(None))
        idx = tuple(idx)

        group = zarr.open_group(image_store, mode="r+")
        sky = group[dv]
        sky[idx] = img_xds[dv].values


def _to_zarr_v3_codec(compressor):
    """Translate a numcodecs compressor to a Zarr v3 ``BytesBytesCodec``.

    Zarr v3's ``create_array`` rejects numcodecs compressor instances and
    requires codecs from ``zarr.codecs``. Pass-through if ``compressor`` is
    already a v3 codec.
    """
    from zarr.abc.codec import BytesBytesCodec

    if isinstance(compressor, BytesBytesCodec):
        return compressor

    cfg = compressor.get_config()
    codec_id = cfg.get("id")

    if codec_id == "blosc":
        from zarr.codecs import BloscCodec

        shuffle_map = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle", -1: None}
        return BloscCodec(
            cname=cfg["cname"],
            clevel=cfg["clevel"],
            shuffle=shuffle_map.get(cfg.get("shuffle", -1)),
            blocksize=cfg.get("blocksize", 0),
        )
    if codec_id == "zstd":
        from zarr.codecs import ZstdCodec

        return ZstdCodec(level=cfg.get("level", 0), checksum=cfg.get("checksum", False))
    if codec_id == "gzip":
        from zarr.codecs import GzipCodec

        return GzipCodec(level=cfg.get("level", 5))

    raise TypeError(
        f"Cannot translate numcodecs compressor {compressor!r} to a Zarr v3 codec."
    )


def create_empty_data_variables_on_disk(
    zarr_store,
    data_variables,
    shape_dict,
    parallel_coords,
    compressor,
    double_precision,
    data_variable_definitions,
    shard_channels=None,
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
    shard_channels : int, optional
        If set (and Zarr v3), create each array as a Zarr v3 **sharded** array:
        the per-task chunk (from ``parallel_coords``) becomes the inner chunk, and
        ``shard_channels`` inner chunks along each parallel dimension are packed
        into one shard file (index CRC disabled so concurrent single-chunk writers
        can each set their own index entry). The shard files are pre-created sparse
        with an empty index. This replaces one-file-per-chunk with far fewer files
        (metadata-server relief) and is written by
        :func:`astroviper.node_tasks.imaging.utils.write_result_chunk_to_disk_sharded_skunk_works`.
        ``None`` (default) keeps the original one-file-per-chunk layout.
    """
    import os

    import zarr
    import numpy as np

    _ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3
    group = zarr.open_group(zarr_store, mode="r+")

    if shard_channels and not _ZARR_V3:
        raise ValueError(
            "shard_channels (output sharding) requires Zarr v3; the installed "
            "zarr is v2."
        )
    sharded = bool(shard_channels) and _ZARR_V3

    if _ZARR_V3 and compressor is not None:
        compressor = _to_zarr_v3_codec(compressor)

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

        dtype = np.dtype(dv_def["dtype"])
        extra_attrs = dv_def.get("attrs", {})
        if extra_attrs.get("dtype") == "bool":
            fill_value = None
        elif dtype.kind in ("f", "c"):
            fill_value = np.nan
        else:
            fill_value = 0

        dv_name = dv_def["name"]
        if sharded:
            # Sharded array: `chunks` becomes the INNER chunk; pack `shard_channels`
            # inner chunks along each parallel dim into one shard. Index CRC is
            # disabled so independent single-chunk writers can set their own index
            # entries (see write_result_chunk_to_disk_sharded_skunk_works).
            from zarr.codecs import ShardingCodec, BytesCodec

            inner = list(chunks)
            shard = list(chunks)
            for ax, d in enumerate(dims):
                if d in parallel_coords:
                    step = inner[ax]  # inner chunk size on this parallel dim
                    want = min(int(shard_channels), shape[ax])
                    shard[ax] = max(step, (want // step) * step)  # multiple of inner
            inner_codecs = [BytesCodec()] + ([compressor] if compressor else [])
            sky = group.require_array(
                dv_name,
                shape=shape,
                chunks=tuple(shard),
                dtype=dtype,
                fill_value=fill_value,
                serializer=ShardingCodec(
                    chunk_shape=tuple(inner),
                    codecs=inner_codecs,
                    index_codecs=[BytesCodec()],  # no crc32c -> concurrent-writer safe
                    index_location="end",
                ),
                compressors=[],
                dimension_names=list(dv_def["dims"]),
            )
        elif _ZARR_V3:
            sky = group.require_array(
                dv_name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                fill_value=fill_value,
                compressors=[compressor] if compressor else [],
                dimension_names=list(dv_def["dims"]),
            )
        else:
            sky = group.require_dataset(
                dv_name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                fill_value=fill_value,
                compressor=compressor,
            )
            sky.attrs["_ARRAY_DIMENSIONS"] = dv_def["dims"]

        for k, v in extra_attrs.items():
            sky.attrs[k] = v

    zarr.consolidate_metadata(zarr_store)

    if sharded:
        # Pre-create every shard file (sparse, empty index) ONCE here in the driver
        # so the concurrent write phase creates no files (metadata-server relief).
        from astroviper.node_tasks.imaging.utils import precreate_sharded_files

        for dv in data_variables:
            precreate_sharded_files(
                os.path.join(zarr_store, data_variable_definitions[dv]["name"])
            )
