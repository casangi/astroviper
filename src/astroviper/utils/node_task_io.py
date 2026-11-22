"""Reads of the node tasks for both I/O engines (``io_engine``).

``"chunk"`` reads Zarr chunk blobs directly (:mod:`astroviper.utils.chunk_io`);
``"xradio"`` uses XRADIO's loaders and xarray. A Measurement Set v2 and a CASA
or FITS image are read through XRADIO by both engines. Writes go through
:func:`astroviper.utils.io.write_result_chunk`.
"""

import json
import os

from astroviper.utils.param_docs import shares_param_docs

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


@shares_param_docs
def load_processing_set(
    io_engine,
    input_data_store,
    data_selection,
    processing_set_data_group_name,
    *,
    frequency_coords,
    full_image_frequencies=None,
    frequency_boundaries=None,
    observation_frequency_boundaries=None,
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
        I/O engine of the node tasks. ``"chunk"`` reads and writes Zarr chunk
        blobs (and FITS data blocks) directly by byte offset
        (:mod:`astroviper.utils.chunk_io`); ``"xradio"`` reads with XRADIO's
        loaders and xarray and writes with zarr-python. A Measurement Set v2
        and CASA or FITS images are read through XRADIO by both. Default
        ``"chunk"``.
    input_data_store : str
        Path of the processing-set Zarr store.
    data_selection : dict
        ``{ms_name: {dimension: selector}}`` of the task.
    processing_set_data_group_name : str
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    frequency_coords : array-like
        Frequencies of the task's image channels (chunk engine).
    frequency_boundaries : dict, optional
        Precomputed image-channel acceptance intervals (Hz), keyed by channel
        centre. Each pair gives the lowest and highest accepted float64
        frequency, both inclusive. Only task-local channel records are needed.
        When omitted, use the supplied image axis to infer channel widths.
    observation_frequency_boundaries : dict, optional
        Single-channel image boundaries keyed by observation name, computed
        from each observation's full frequency axis before selection.
    full_image_frequencies : array-like, optional
        Original, unpartitioned image-frequency axis (Hz), retained for legacy
        callers. Mapping and boundary checks use this axis, with observations
        required to belong to the local image-frequency subset. Precomputed
        ``frequency_boundaries`` take precedence when supplied.
    instrument_polarization_basis : str, optional
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` or ``"circular"``. The residual update grids and degrids the
        correlations of this basis and the model update deconvolves in the
        Stokes basis, in which the image is written. The Stokes planes requested
        in ``image_params["polarization_coords"]`` fix the correlations that are
        loaded and gridded: the two parallel hands give ``I, Q`` (linear) or
        ``I, V`` (circular), all four correlations give ``I, Q, U, V``. A sample
        is used only if none of its loaded correlations is flagged.
    data_group : dict, optional
        Role->variable mapping of the data group; read from the first MSv4 of
        ``data_selection`` when ``None`` and needed.
    load_correlated_data : bool, optional
        If false, the correlated data are not read.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels. A distributed application resolves ``0`` (auto) once, before
        its checks, to the CPUs per Dask worker thread of the host with the
        fewest, at least 1 (1 where these are not known, as under MPI), and
        warns when a host's worker threads times the count exceed its CPUs
        (see
        :func:`~astroviper.utils.data_partitioning.resolve_processing_function_threads`).
        Node tasks and processing functions do not resolve ``0``; those that
        check the count refuse it.
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
    from astroviper.processing_functions.imaging.utils.frequency_mapping import (
        attach_observation_frequency_boundaries,
    )

    def with_boundaries(ps_xdt):
        return attach_observation_frequency_boundaries(
            ps_xdt, observation_frequency_boundaries
        )

    check_io_engine(io_engine)
    if lazy_input_data is not None:
        from astroviper.utils.measurement_set_v2_input import (
            load_processing_set_msv2,
        )

        return with_boundaries(load_processing_set_msv2(lazy_input_data))

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

        return with_boundaries(
            chunk_io.load_processing_set(
                input_data_store,
                sel_parms=data_selection,
                data_group=data_group,
                processing_set_data_group_name=processing_set_data_group_name,
                frequency_coords=frequency_coords,
                full_image_frequencies=full_image_frequencies,
                frequency_boundaries=frequency_boundaries,
                observation_frequency_boundaries=observation_frequency_boundaries,
                instrument_polarization_basis=instrument_polarization_basis,
                processing_function_threads=processing_function_threads,
                load_correlated_data=load_correlated_data,
                map_frequencies=map_visibility_frequencies_to_image,
            )
        )

    from xradio.measurement_set.load_processing_set import (
        load_processing_set as xradio_load_processing_set,
    )

    return with_boundaries(
        xradio_load_processing_set(
            input_data_store,
            sel_parms=data_selection,
            data_group_name=processing_set_data_group_name,
            drop_variables=(
                None if load_correlated_data else [data_group["correlated_data"]]
            ),
            load_sub_datasets=False,
        )
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
