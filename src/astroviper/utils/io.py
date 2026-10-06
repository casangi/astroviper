full_dims_lm = ["time", "frequency", "polarization", "l", "m"]
full_dims_uv = ["time", "frequency", "polarization", "u", "v"]
norm_dims = ["time", "frequency", "polarization"]
beam_params_dims = ["time", "frequency", "polarization", "beam_params_label"]

# Measurement Set v4 main-dataset variables written chunk-wise by the simulator
# (see ``astroviper.utils.measurement_set_tools`` and the ``simulation`` subdomain).
visibility_dims = ["time", "baseline_id", "frequency", "polarization"]
uvw_dims = ["time", "baseline_id", "uvw_label"]

visibility_data_variables_and_dims_double_precision = {
    "visibility": {
        "dims": visibility_dims,
        "dtype": "<c16",
        "name": "VISIBILITY",
        "attrs": {"type": "quantity", "units": "Jy"},
    },
    "uvw": {
        "dims": uvw_dims,
        "dtype": "<f8",
        "name": "UVW",
        "attrs": {"type": "uvw", "units": "m", "frame": "icrs"},
    },
    "weight": {
        "dims": visibility_dims,
        "dtype": "<f8",
        "name": "WEIGHT",
        "attrs": {"type": "quantity", "units": "1/Jy^2"},
    },
    "flag": {"dims": visibility_dims, "dtype": "|b1", "name": "FLAG"},
}

visibility_data_variables_and_dims_single_precision = {
    **visibility_data_variables_and_dims_double_precision,
    "visibility": {
        **visibility_data_variables_and_dims_double_precision["visibility"],
        "dtype": "<c8",
    },
    "weight": {
        **visibility_data_variables_and_dims_double_precision["weight"],
        "dtype": "<f4",
    },
}

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
    "sky_restored_primary_beam_corrected": {
        "dims": full_dims_lm,
        "dtype": "<f8",
        "name": "SKY_RESTORED_PRIMARY_BEAM_CORRECTED",
    },
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
    "sky_restored_primary_beam_corrected": {
        "dims": full_dims_lm,
        "dtype": "<f4",
        "name": "SKY_RESTORED_PRIMARY_BEAM_CORRECTED",
    },
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


# Data-group membership of the image variables: variable key (as used in
# ``image_data_variables_keep``) -> (data group name, role). Mirrors the
# groups the processing functions register on the in-memory per-chunk image
# dataset (``residual`` / ``model`` / ``restored``) so the driver can stamp
# the same layout onto the on-disk store. Variables not listed here do not
# belong to a data group.
imaging_data_variable_data_group_roles = {
    "sky_residual": ("residual", "sky"),
    "point_spread_function": ("residual", "point_spread_function"),
    "primary_beam": ("residual", "primary_beam"),
    "mask": ("residual", "mask"),
    "uv_sampling": ("residual", "uv_sampling"),
    "uv_sampling_normalization": ("residual", "uv_sampling_normalization"),
    "visibility": ("residual", "visibility"),
    "visibility_normalization": ("residual", "visibility_normalization"),
    "beam_fit_params_point_spread_function": (
        "residual",
        "beam_fit_params_point_spread_function",
    ),
    "sky_model": ("model", "sky"),
    "sky_restored": ("restored", "sky"),
    "sky_restored_primary_beam_corrected": (
        "restored",
        "sky_primary_beam_corrected",
    ),
}


def image_data_groups_for_kept_variables(image_data_variables_keep):
    """Build the image data groups for the variables kept on disk.

    Parameters
    ----------
    image_data_variables_keep : list of str
        Keys into ``imaging_data_variables_and_dims_*`` selecting which image
        variables are written to the output Zarr store.

    Returns
    -------
    dict
        ``{data_group_name: {role: DATA_VARIABLE_NAME}}`` restricted to the
        kept variables. The group layout mirrors what the processing
        functions register on the in-memory per-chunk image dataset
        (``residual`` / ``model`` / ``restored``); kept variables with no
        data-group membership are skipped.
    """
    registry = imaging_data_variables_and_dims_double_precision
    data_groups = {}
    for key in image_data_variables_keep:
        membership = imaging_data_variable_data_group_roles.get(key)
        if membership is None:
            continue
        group_name, role = membership
        data_groups.setdefault(group_name, {})[role] = registry[key]["name"]
    return data_groups


def write_result_chunk_to_disk_using_zarr(
    image_store, image_data_variables_keep, task_coords, img_xds
):
    import zarr

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


def _task_extent(dim, shape_dict, parallel_coords):
    """Extent of one node task along ``dim``: the first task's chunk on a
    parallelized dimension, the whole axis otherwise."""
    if parallel_coords and dim in parallel_coords:
        return len(parallel_coords[dim]["data_chunks"][0])
    return int(shape_dict[dim])


def _task_chunk_lengths(dim, parallel_coords):
    """Per-task chunk lengths along the parallelized dimension ``dim``."""
    data_chunks = parallel_coords[dim]["data_chunks"]
    chunks = data_chunks.values() if isinstance(data_chunks, dict) else data_chunks
    return [len(chunk) for chunk in chunks]


def validate_image_chunking_and_sharding(
    image_chunking, image_sharding, shape_dict, parallel_coords, extra_dims=("u", "v")
):
    """Validate the on-disk image chunking and sharding against the image and
    the mapping parallelism.

    Rules, per dimension:

    * every key of ``image_chunking`` / ``image_sharding`` is an image dimension
      (a key of ``shape_dict``, or one of ``extra_dims`` -- the uv-domain
      ``u``/``v`` -- which are accepted but not checked further) and every value
      a positive int;
    * ``image_chunking[dim]`` is at most the node task's extent on ``dim`` (the
      per-task chunk of a parallelized dimension, the whole axis otherwise) and,
      on a parallelized dimension, divides every node task's chunk except the
      last (the array-edge partial chunk): node tasks write whole on-disk
      chunks, so a chunk straddling two tasks would be written -- and
      clobbered -- by both;
    * ``image_sharding[dim]`` is a multiple of the on-disk chunk on ``dim`` (Zarr
      requires a shard to hold whole chunks); the chunk defaults to the task
      extent when ``dim`` is not in ``image_chunking``. Shards may span several
      node tasks (the fixed-slot sharded writer is concurrency-safe), and a
      shard larger than the axis is clipped to it on creation, so neither is an
      error.

    Parameters
    ----------
    image_chunking : dict or None
        ``{dimension_name: chunk_size}`` requested by the caller.
    image_sharding : dict or None
        ``{dimension_name: shard_size}`` requested by the caller.
    shape_dict : dict
        ``{dimension_name: size}`` of the image being written.
    parallel_coords : dict or None
        Parallel coordinates of the mapping (for cube imaging keyed by
        ``frequency``), providing the per-task chunk lengths.
    extra_dims : tuple of str, optional
        Dimension names accepted as keys although absent from ``shape_dict``.

    Raises
    ------
    ValueError
        On an unknown dimension key, a non-positive / non-integer size, a chunk
        larger than the task extent or not dividing the task chunks, or a shard
        that is not a multiple of its chunk.
    """
    chunking = dict(image_chunking or {})
    sharding = dict(image_sharding or {})
    valid_dims = set(shape_dict) | set(extra_dims)
    for label, mapping in (("image_chunking", chunking), ("image_sharding", sharding)):
        for dim, size in mapping.items():
            if dim not in valid_dims:
                raise ValueError(
                    f"{label} key {dim!r} is not an image dimension; expected one "
                    f"of {sorted(valid_dims)}."
                )
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(
                    f"{label}[{dim!r}] must be a positive int, got {size!r}."
                )
    for dim, size in chunking.items():
        if dim not in shape_dict:
            continue
        extent = _task_extent(dim, shape_dict, parallel_coords)
        if size > extent:
            raise ValueError(
                f"image_chunking[{dim!r}]={size} exceeds the {dim} extent of one "
                f"node task ({extent}); an on-disk chunk cannot be larger than "
                "what one node task writes."
            )
        if parallel_coords and dim in parallel_coords:
            for task_length in _task_chunk_lengths(dim, parallel_coords)[:-1]:
                if task_length % size:
                    raise ValueError(
                        f"image_chunking[{dim!r}]={size} must divide every node "
                        f"task's {dim} chunk (found a task chunk of length "
                        f"{task_length}); otherwise an on-disk chunk would "
                        "straddle two node tasks and be written by both."
                    )
    for dim, size in sharding.items():
        if dim not in shape_dict:
            continue
        chunk = chunking.get(dim, _task_extent(dim, shape_dict, parallel_coords))
        if size < int(shape_dict[dim]) and size % chunk:
            raise ValueError(
                f"image_sharding[{dim!r}]={size} must be a multiple of the on-disk "
                f"chunk on {dim} ({chunk}); set image_chunking[{dim!r}] to a "
                "divisor of the shard size."
            )


def image_chunk_and_shard_shapes(
    dims, shape_dict, parallel_coords, image_chunking=None, image_sharding=None
):
    """On-disk (inner) chunk shape and shard shape of a variable with ``dims``.

    The chunk on a dimension is ``image_chunking[dim]`` when given, else the
    node task's extent (its per-task chunk on a parallelized dimension, the
    whole axis otherwise). The shard is ``image_sharding[dim]`` when given,
    clipped to the axis and rounded up to a multiple of the chunk, else the
    task extent rounded up to a multiple of the chunk (one shard per task
    chunk on a parallelized dimension, the whole axis otherwise). Assumes
    :func:`validate_image_chunking_and_sharding` passed.

    Returns
    -------
    chunks, shards : list of int
        Per-axis chunk and shard sizes, in ``dims`` order. ``shards`` is only
        meaningful for a sharded array.
    """
    chunking = image_chunking or {}
    sharding = image_sharding or {}
    chunks, shards = [], []
    for dim in dims:
        extent = _task_extent(dim, shape_dict, parallel_coords)
        chunk = int(chunking.get(dim, extent))
        want = min(int(sharding.get(dim, extent)), int(shape_dict[dim]))
        chunks.append(chunk)
        shards.append(max(chunk, -(-want // chunk) * chunk))
    return chunks, shards


def write_zarr_image_store(img_xds, image_store, overwrite=False):
    """Write an image dataset to a Zarr store and return the path of the store.

    The store is written with :func:`xradio.image.write_image`. XRADIO
    versions that give image Zarr stores the ``.img.zarr`` extension keep a
    name that ends in ``.img.zarr``, replace a bare ``.zarr`` extension
    (``out.zarr`` gives ``out.img.zarr``), append ``.img.zarr`` to any other
    name and return the paths written; earlier versions write ``image_store``
    as given and return ``None``. Code that opens the store, adds data
    variables to it or passes it on must use the returned path; a name that
    ends in ``.img.zarr`` is kept by every XRADIO version.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset to write (coordinates, attributes and any data
        variables).
    image_store : str
        Requested path of the Zarr store.
    overwrite : bool, default False
        If ``True``, replace an existing store.

    Returns
    -------
    str
        Path of the Zarr store written.

    Raises
    ------
    FileExistsError
        If the store to be written exists and ``overwrite`` is ``False``;
        XRADIO raises it before anything is written.
    """
    from xradio.image import write_image

    written_paths = write_image(
        img_xds, imagename=image_store, out_format="zarr", overwrite=overwrite
    )
    if written_paths:
        return written_paths[0]
    return image_store


def create_empty_data_variables_on_disk(
    zarr_store,
    data_variables,
    shape_dict,
    parallel_coords,
    compressor,
    double_precision,
    data_variable_definitions,
    image_chunking=None,
    image_sharding=None,
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
    image_chunking : dict, optional
        On-disk chunk shape as ``{dimension_name: chunk_size}`` (e.g. ``{"l":
        1024, "m": 1024}`` to chunk the sky plane, ``{"frequency": 1}`` for
        one-channel chunks). A dimension not listed defaults to the node task's
        extent: its per-task chunk (from ``parallel_coords``) on a parallelized
        dimension, the whole axis otherwise. A chunk may not exceed that extent
        and must divide the per-task chunk of a parallelized dimension (see
        :func:`validate_image_chunking_and_sharding`). For a sharded array this
        is the *inner* chunk shape. ``None`` (default) uses the defaults on
        every dimension.
    image_sharding : dict, optional
        If set (and Zarr v3), create each array as a Zarr v3 **sharded** array
        with shard shape ``{dimension_name: shard_size}``: a shard must be a
        multiple of the chunk on its dimension (a shard larger than the axis is
        clipped to it), and a dimension not listed gets one shard per node task
        chunk on a parallelized dimension and the whole axis otherwise. The
        index CRC is disabled so concurrent single-chunk writers can each set
        their own index entry, and the shard files are pre-created sparse with
        an empty index. This replaces one-file-per-chunk with far fewer files
        (metadata-server relief) and is written by
        :func:`astroviper.node_tasks.imaging.utils.write_result_chunk_to_disk_sharded_skunk_works`.
        ``None`` (default) keeps the original one-file-per-chunk layout.
    """
    import os

    import numpy as np
    import zarr

    _ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3
    group = zarr.open_group(zarr_store, mode="r+")

    validate_image_chunking_and_sharding(
        image_chunking, image_sharding, shape_dict, parallel_coords
    )
    if image_sharding and not _ZARR_V3:
        raise ValueError(
            "image_sharding (output sharding) requires Zarr v3; the installed "
            "zarr is v2."
        )
    sharded = bool(image_sharding) and _ZARR_V3

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

        chunks, shard = image_chunk_and_shard_shapes(
            dims, shape_dict, parallel_coords, image_chunking, image_sharding
        )

        dtype = np.dtype(dv_def["dtype"])
        extra_attrs = dv_def.get("attrs", {})
        if extra_attrs.get("dtype") == "bool":
            fill_value = None
        elif dtype.kind in ("f", "c"):
            fill_value = np.nan
        elif dtype.kind == "b":
            fill_value = False
        else:
            fill_value = 0

        dv_name = dv_def["name"]
        if sharded:
            # Sharded array: `chunks` is the INNER chunk shape and `shard` the
            # shard shape (see image_chunk_and_shard_shapes). The index CRC is
            # disabled so independent single-chunk writers can set their own
            # index entries (see write_result_chunk_to_disk_sharded_skunk_works).
            from zarr.codecs import BytesCodec, ShardingCodec

            inner_codecs = [BytesCodec()] + ([compressor] if compressor else [])
            sky = group.require_array(
                dv_name,
                shape=shape,
                chunks=tuple(shard),
                dtype=dtype,
                fill_value=fill_value,
                serializer=ShardingCodec(
                    chunk_shape=tuple(chunks),
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
                chunks=tuple(chunks),
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
        from astroviper.node_tasks.imaging.utils import (
            precreate_sharded_files,
            record_shard_ost_map,
        )

        for dv in data_variables:
            precreate_sharded_files(
                os.path.join(zarr_store, data_variable_definitions[dv]["name"])
            )

        # Record which OST each freshly created shard file landed on (Lustre's
        # allocator gives no distinctness guarantee); saved as a sibling
        # feather of the store. Best-effort: must never fail the imaging run.
        try:
            record_shard_ost_map(
                zarr_store,
                [data_variable_definitions[dv]["name"] for dv in data_variables],
            )
        except Exception as exc:
            import toolviper.utils.logger as logger

            logger.warning(f"record_shard_ost_map failed ({exc}); continuing.")
