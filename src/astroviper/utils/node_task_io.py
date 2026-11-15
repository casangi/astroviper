"""Reads of the node tasks for both I/O engines (``io_engine``).

``"chunk"`` reads Zarr chunk blobs directly (:mod:`astroviper.utils.chunk_io`);
``"xradio"`` uses XRADIO's loaders and xarray. A Measurement Set v2 and a CASA
or FITS image are read through XRADIO by both engines. Writes go through
:func:`astroviper.utils.io.write_result_chunk`.
"""

import json
import os

IO_ENGINES = ("chunk", "xradio")


def check_io_engine(io_engine):
    """Raise ``ValueError`` unless ``io_engine`` is ``"chunk"`` or ``"xradio"``."""
    if io_engine not in IO_ENGINES:
        raise ValueError(f"io_engine must be 'chunk' or 'xradio', not {io_engine!r}.")


def _is_zarr_store(path):
    return os.path.isfile(os.path.join(path, "zarr.json")) or os.path.isfile(
        os.path.join(path, ".zgroup")
    )


def _stored_data_group(input_data_store, ms_name, processing_set_data_group_name):
    """The role->variable mapping of a data group, from the MSv4's attributes."""
    ms_path = os.path.join(input_data_store, ms_name)
    v3 = os.path.join(ms_path, "zarr.json")
    if os.path.isfile(v3):
        with open(v3) as fh:
            attrs = json.load(fh).get("attributes", {})
    else:
        with open(os.path.join(ms_path, ".zattrs")) as fh:
            attrs = json.load(fh)
    return attrs["data_groups"][processing_set_data_group_name]


def load_processing_set(
    io_engine,
    input_data_store,
    data_selection,
    processing_set_data_group_name,
    *,
    frequency_coords,
    instrument_polarization_basis="linear",
    data_group=None,
    load_correlated_data=True,
    processing_function_threads=1,
    lazy_input_data=None,
):
    """Load a node task's selection of a processing set.

    Parameters
    ----------
    io_engine : {"chunk", "xradio"}
        ``"chunk"``: :func:`astroviper.utils.chunk_io.load_processing_set`
        (the data group's variables only, coordinates reconstructed);
        ``"xradio"``: XRADIO's ``load_processing_set`` without sub-datasets.
    input_data_store : str
        Path of the processing-set Zarr store.
    data_selection : dict
        ``{ms_name: {dimension: selector}}`` of the task.
    processing_set_data_group_name : str
        Data group to load.
    frequency_coords : array-like
        Frequencies of the task's image channels (chunk engine).
    instrument_polarization_basis : str, optional
        ``"linear"`` or ``"circular"`` (chunk engine).
    data_group : dict, optional
        Role->variable mapping of the data group; read from the first MSv4 of
        ``data_selection`` when ``None`` and needed.
    load_correlated_data : bool, optional
        If false, the correlated data are not read.
    processing_function_threads : int, optional
        Threads of the chunk loader.
    lazy_input_data : dict, optional
        Lazily indexed selection of a Measurement Set v2
        (:func:`~astroviper.utils.measurement_set_v2_input.add_lazy_input_data`);
        when given it is read with
        :func:`~astroviper.utils.measurement_set_v2_input.load_processing_set_msv2`
        by both engines.

    Returns
    -------
    xarray.DataTree
    """
    check_io_engine(io_engine)
    if lazy_input_data is not None:
        from astroviper.utils.measurement_set_v2_input import (
            load_processing_set_msv2,
        )

        return load_processing_set_msv2(lazy_input_data)

    if data_group is None and (io_engine == "chunk" or not load_correlated_data):
        data_group = _stored_data_group(
            input_data_store,
            next(iter(data_selection)),
            processing_set_data_group_name,
        )

    if io_engine == "chunk":
        from astroviper.processing_functions.imaging.utils.frequency_mapping import (
            map_visibility_frequencies_to_image,
        )
        from astroviper.utils import chunk_io

        return chunk_io.load_processing_set(
            input_data_store,
            sel_parms=data_selection,
            data_group=data_group,
            processing_set_data_group_name=processing_set_data_group_name,
            frequency_coords=frequency_coords,
            instrument_polarization_basis=instrument_polarization_basis,
            processing_function_threads=processing_function_threads,
            load_correlated_data=load_correlated_data,
            map_frequencies=map_visibility_frequencies_to_image,
        )

    from xradio.measurement_set.load_processing_set import (
        load_processing_set as xradio_load_processing_set,
    )

    return xradio_load_processing_set(
        input_data_store,
        sel_parms=data_selection,
        data_group_name=processing_set_data_group_name,
        drop_variables=(
            None if load_correlated_data else [data_group["correlated_data"]]
        ),
        load_sub_datasets=False,
    )


def open_image(io_engine, image_store):
    """Open an image lazily, without dask.

    Parameters
    ----------
    io_engine : {"chunk", "xradio"}
        For a Zarr store, ``"chunk"`` reads the data variables by byte offset
        (:func:`astroviper.utils.chunk_io.open_image`) and ``"xradio"`` with
        ``xarray.open_zarr(chunks=None)``. Other formats open with XRADIO's
        ``open_image``.
    image_store : str
        Path of the image.

    Returns
    -------
    xarray.Dataset
    """
    check_io_engine(io_engine)
    if not _is_zarr_store(image_store):
        from xradio.image import open_image as xradio_open_image

        return xradio_open_image(image_store)
    if io_engine == "chunk":
        from astroviper.utils import chunk_io

        return chunk_io.open_image(image_store)
    import xarray as xr

    return xr.open_zarr(image_store, chunks=None)


def load_image(io_engine, image_store, selection=None):
    """Select part of an image lazily, with XRADIO's image conventions.

    Parameters
    ----------
    io_engine : {"chunk", "xradio"}
        For a Zarr store, ``"chunk"`` uses
        :func:`astroviper.utils.chunk_io.load_image`; otherwise XRADIO's
        ``load_image``.
    image_store : str
        Path of the image.
    selection : dict, optional
        ``{dimension: int, slice or integer array}``.

    Returns
    -------
    xarray.Dataset
    """
    check_io_engine(io_engine)
    if io_engine == "chunk" and _is_zarr_store(image_store):
        from astroviper.utils import chunk_io

        return chunk_io.load_image(image_store, selection)
    from xradio.image import load_image as xradio_load_image

    return xradio_load_image(image_store, block_des=selection or {})
