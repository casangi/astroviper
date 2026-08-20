"""Build empty Measurement Set v4 processing sets that node tasks fill chunk-wise.

The simulator (``astroviper.distributed_applications.simulation``) creates the
on-disk output before any graph node runs: the processing-set root, one MSv4
group with all coordinates/attributes, its ``antenna_xds`` and
``field_and_source_base_xds`` children, and the (empty) ``VISIBILITY``, ``UVW``,
``WEIGHT`` and ``FLAG`` arrays chunked along the parallel coordinates.  Node tasks
then region-write their ``(time, frequency)`` block.

Everything here follows ``xradio.measurement_set.schema`` (MSv4 schema version
``4.0.0``) and is validated in the unit tests with
``xradio.schema.check.check_datatree``.  These helpers are candidates to move
to XRADIO once it grows an MSv4 factory.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import modify_data_groups_xds

MSV4_SCHEMA_VERSION = "4.0.0"

# casacore Stokes enumeration (Stokes.h) used by the legacy SIRIUS API.
POLARIZATION_CODE_TO_NAME = {
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
}
POLARIZATION_NAME_TO_CODE = {v: k for k, v in POLARIZATION_CODE_TO_NAME.items()}
CIRCULAR_POLARIZATIONS = ("RR", "RL", "LR", "LL")
LINEAR_POLARIZATIONS = ("XX", "XY", "YX", "YY")


def normalize_polarization(polarization: Sequence[str] | Sequence[int]) -> list[str]:
    """Return polarization labels as MSv4 strings.

    Accepts either MSv4 strings (``["RR", "LL"]``) or casacore Stokes codes
    (``[5, 8]``).  The result must be a subset of one instrumental basis
    (``RR, RL, LR, LL`` or ``XX, XY, YX, YY``).
    """
    names = []
    for p in polarization:
        if isinstance(p, str):
            names.append(p.upper())
        else:
            code = int(p)
            if code not in POLARIZATION_CODE_TO_NAME:
                raise ValueError(f"Unknown polarization code {code}.")
            names.append(POLARIZATION_CODE_TO_NAME[code])
    if not (
        set(names) <= set(CIRCULAR_POLARIZATIONS)
        or set(names) <= set(LINEAR_POLARIZATIONS)
    ):
        raise ValueError(
            f"polarization {names} must be a subset of {CIRCULAR_POLARIZATIONS} or "
            f"{LINEAR_POLARIZATIONS}."
        )
    return names


def polarization_index(polarization: Sequence[str]) -> np.ndarray:
    """Index (0..3, row-major 2x2 correlation order) of each polarization label."""
    names = normalize_polarization(polarization)
    basis = (
        CIRCULAR_POLARIZATIONS
        if names[0] in CIRCULAR_POLARIZATIONS
        else LINEAR_POLARIZATIONS
    )
    return np.array([basis.index(n) for n in names], dtype=np.int64)


def polarization_basis(polarization: Sequence[str]) -> tuple[str, str]:
    """Receptor basis ``("R", "L")`` or ``("X", "Y")`` of a polarization list."""
    names = normalize_polarization(polarization)
    return ("R", "L") if names[0] in CIRCULAR_POLARIZATIONS else ("X", "Y")


def baseline_antenna_pairs(
    n_antenna: int, auto_correlations: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Antenna index pairs ``(antenna1, antenna2)`` of all baselines.

    Ordering is antenna1-major (``antenna1 <= antenna2``), matching SIRIUS and
    the MSv4 baseline ordering produced by XRADIO.

    Returns
    -------
    antenna1, antenna2 : np.ndarray, [n_baseline] int
    """
    offset = 0 if auto_correlations else 1
    antenna1, antenna2 = np.triu_indices(n_antenna, k=offset)
    return antenna1.astype(np.int64), antenna2.astype(np.int64)


def number_of_baselines(n_antenna: int, auto_correlations: bool = False) -> int:
    """Number of baselines for ``n_antenna`` antennas."""
    return (
        (n_antenna * (n_antenna + 1)) // 2
        if auto_correlations
        else (n_antenna * (n_antenna - 1)) // 2
    )


def make_time_coordinate(time_params: dict) -> dict:
    """Time coordinate (measures dictionary) of the simulated observation.

    Parameters
    ----------
    time_params : dict
        ``time_start`` (str, ``YYYY-MM-DDTHH:MM:SS.SSS`` UTC), ``time_delta``
        (float, s; also the integration time) and ``n_samples`` (int).

    Returns
    -------
    dict
        ``{"dims": "time", "data": unix seconds (UTC), "attrs": {...}}`` with the
        MSv4 ``time`` coordinate attributes.
    """
    from graphviper.graph_tools.coordinate_utils import make_time_coord

    coord = make_time_coord(
        time_start=time_params["time_start"],
        time_delta=time_params["time_delta"],
        n_samples=time_params["n_samples"],
        time_scale="utc",
    )
    coord["attrs"] = {
        "type": "time",
        "units": "s",
        "scale": "utc",
        "format": "unix",
        "integration_time": {
            "attrs": {"type": "quantity", "units": "s"},
            "data": float(time_params["time_delta"]),
            "dims": [],
        },
    }
    return coord


def make_frequency_coordinate(frequency_params: dict) -> dict:
    """Frequency coordinate (measures dictionary) of the simulated spectral window.

    Parameters
    ----------
    frequency_params : dict
        ``freq_start`` (Hz), ``freq_delta`` (Hz, channel spacing), ``n_channels``,
        optional ``channel_width`` (Hz, defaults to ``freq_delta``),
        ``spectral_window_name`` (default ``"spw_0"``), ``observer`` (spectral
        frame label, default ``"lsrk"``, see Notes) and ``spectral_window_intents``.

    Returns
    -------
    dict
        ``{"dims": "frequency", "data": ..., "attrs": {...}}`` with the MSv4
        ``frequency`` coordinate attributes.

    Notes
    -----
    The simulator evaluates visibilities at the given channel frequencies without
    any Doppler tracking, so the spectral frame defaults to ``"lsrk"`` purely as a
    label (the value XRADIO/AstroVIPER imaging expects).
    """
    n_channels = int(frequency_params["n_channels"])
    freq_start = float(frequency_params["freq_start"])
    freq_delta = float(frequency_params["freq_delta"])
    data = freq_start + freq_delta * np.arange(n_channels, dtype=np.float64)
    channel_width = float(frequency_params.get("channel_width", abs(freq_delta)))
    observer = frequency_params.get("observer", "lsrk")
    return {
        "dims": "frequency",
        "data": data,
        "attrs": {
            "type": "spectral_coord",
            "units": "Hz",
            "observer": observer,
            "spectral_window_name": frequency_params.get(
                "spectral_window_name", "spw_0"
            ),
            "spectral_window_intents": list(
                frequency_params.get("spectral_window_intents", ["UNSPECIFIED"])
            ),
            "reference_frequency": {
                "attrs": {
                    "type": "spectral_coord",
                    "units": "Hz",
                    "observer": observer,
                },
                "data": float(data[n_channels // 2]),
                "dims": [],
            },
            "channel_width": {
                "attrs": {"type": "quantity", "units": "Hz"},
                "data": channel_width,
                "dims": [],
            },
        },
    }


def make_field_and_source_xds(
    field_name: Sequence[str],
    phase_center_ra_dec: np.ndarray,
    frame: str = "fk5",
    source_name: Sequence[str] | None = None,
) -> xr.Dataset:
    """Build an MSv4 ``field_and_source_xds`` for one or more fields.

    Parameters
    ----------
    field_name : sequence of str, [n_field]
        Unique field names.
    phase_center_ra_dec : np.ndarray, [n_field, 2], radians
        Phase centre of each field.
    frame : str
        Sky frame of the directions (``"fk5"`` or ``"icrs"``).
    source_name : sequence of str, optional
        Defaults to ``field_name``.

    Returns
    -------
    xr.Dataset
        Dataset following ``xradio.measurement_set.schema.FieldSourceXds``.
    """
    field_name = np.asarray(field_name, dtype=str)
    phase_center_ra_dec = np.asarray(phase_center_ra_dec, dtype=np.float64).reshape(
        len(field_name), 2
    )
    if source_name is None:
        source_name = field_name
    sky_attrs = {"type": "sky_coord", "units": "rad", "frame": frame}
    xds = xr.Dataset(
        coords={
            "field_name": ("field_name", field_name),
            "source_name": ("field_name", np.asarray(source_name, dtype=str)),
            "sky_dir_label": ("sky_dir_label", ["ra", "dec"]),
        }
    )
    xds["FIELD_PHASE_CENTER_DIRECTION"] = xr.DataArray(
        phase_center_ra_dec, dims=("field_name", "sky_dir_label"), attrs=dict(sky_attrs)
    )
    xds["FIELD_REFERENCE_CENTER_DIRECTION"] = xr.DataArray(
        phase_center_ra_dec.copy(),
        dims=("field_name", "sky_dir_label"),
        attrs=dict(sky_attrs),
    )
    xds["SOURCE_DIRECTION"] = xr.DataArray(
        phase_center_ra_dec.copy(),
        dims=("field_name", "sky_dir_label"),
        attrs=dict(sky_attrs),
    )
    xds.attrs["type"] = "field_and_source"
    return xds


def resolve_fields(
    phase_center_ra_dec: np.ndarray,
    field_name: Sequence[str] | str | None,
    n_time: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand (possibly singleton) per-time phase centres and field names.

    Parameters
    ----------
    phase_center_ra_dec : np.ndarray, [n_time | 1, 2], radians
    field_name : sequence of str [n_time | 1], str or None
        ``None`` names fields ``field_0``, ``field_1``, ... in order of first
        appearance of distinct phase centres.
    n_time : int

    Returns
    -------
    field_name_per_time : np.ndarray, [n_time] str
    unique_field_name : np.ndarray, [n_field] str
        In order of first appearance.
    unique_phase_center : np.ndarray, [n_field, 2]
    """
    pc = np.asarray(phase_center_ra_dec, dtype=np.float64)
    if pc.ndim != 2 or pc.shape[1] != 2 or pc.shape[0] not in (1, n_time):
        raise ValueError(
            f"phase_center_ra_dec must have shape [n_time | 1, 2]; got {pc.shape} for n_time={n_time}."
        )
    pc_per_time = np.broadcast_to(pc, (n_time, 2))

    if field_name is None:
        # distinct phase centres, in order of first appearance
        _, first_index, inverse = np.unique(
            pc_per_time, axis=0, return_index=True, return_inverse=True
        )
        order = np.argsort(first_index)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        field_index = rank[np.ravel(inverse)]
        names_per_time = np.array([f"field_{i}" for i in field_index], dtype=str)
    else:
        names = np.atleast_1d(np.asarray(field_name, dtype=str))
        if names.shape[0] not in (1, n_time):
            raise ValueError(
                f"field_name must have length 1 or n_time={n_time}; got {names.shape[0]}."
            )
        names_per_time = np.broadcast_to(names, (n_time,)).copy()

    unique_names, first = np.unique(names_per_time, return_index=True)
    order = np.argsort(first)
    unique_names = unique_names[order]
    unique_pc = np.stack(
        [pc_per_time[first[order][i]] for i in range(len(unique_names))]
    )
    # a field name must map to a single phase centre
    for name, centre in zip(unique_names, unique_pc, strict=True):
        mask = names_per_time == name
        if not np.allclose(pc_per_time[mask], centre):
            raise ValueError(f"Field '{name}' is used with more than one phase centre.")
    return names_per_time, unique_names, unique_pc


def make_empty_visibility_xds(
    time_coord: dict,
    frequency_coord: dict,
    polarization: Sequence[str],
    antenna_xds: xr.Dataset,
    field_name_per_time: np.ndarray,
    auto_correlations: bool = False,
    scan_name: str = "scan_1",
    scan_intents: Sequence[str] = ("OBSERVE_TARGET#ON_SOURCE",),
    observation_info: dict | None = None,
    processor_info: dict | None = None,
    description: str = "Simulated visibilities (AstroVIPER simulation).",
) -> xr.Dataset:
    """Build the coordinates and attributes of an MSv4 main dataset (no data variables).

    Parameters
    ----------
    time_coord, frequency_coord : dict
        From :func:`make_time_coordinate` / :func:`make_frequency_coordinate`.
    polarization : sequence of str
        MSv4 polarization labels (or casacore codes), see :func:`normalize_polarization`.
    antenna_xds : xr.Dataset
        MSv4 antenna dataset (:func:`astroviper.utils.telescope_layout.read_telescope_layout`).
    field_name_per_time : np.ndarray, [n_time] str
    auto_correlations : bool
    scan_name : str
    scan_intents : sequence of str
    observation_info, processor_info : dict, optional
        MSv4 info dictionaries; sensible simulation defaults are used when ``None``.
    description : str
        Description stored in the ``base`` data group.

    Returns
    -------
    xr.Dataset
        Main dataset with all required coordinates and attributes; the data
        variables (``VISIBILITY``, ``UVW``, ``WEIGHT``, ``FLAG``) are created on
        disk by :func:`create_empty_measurement_set_v4_on_disk`.
    """
    from importlib import metadata

    try:
        version = metadata.version("astroviper")
    except metadata.PackageNotFoundError:
        version = "0.0.0"

    polarization = normalize_polarization(polarization)
    antenna_name = np.asarray(antenna_xds.antenna_name.values, dtype=str)
    antenna1, antenna2 = baseline_antenna_pairs(len(antenna_name), auto_correlations)
    n_time = len(time_coord["data"])
    field_name_per_time = np.asarray(field_name_per_time, dtype=str)
    if field_name_per_time.shape != (n_time,):
        raise ValueError("field_name_per_time must have shape [n_time].")

    xds = xr.Dataset(
        coords={
            "time": (
                "time",
                np.asarray(time_coord["data"], dtype=np.float64),
                dict(time_coord["attrs"]),
            ),
            "baseline_id": ("baseline_id", np.arange(len(antenna1), dtype=np.int64)),
            "frequency": (
                "frequency",
                np.asarray(frequency_coord["data"], dtype=np.float64),
                dict(frequency_coord["attrs"]),
            ),
            "polarization": ("polarization", np.array(polarization, dtype=str)),
            "uvw_label": ("uvw_label", ["u", "v", "w"]),
            "baseline_antenna1_name": ("baseline_id", antenna_name[antenna1]),
            "baseline_antenna2_name": ("baseline_id", antenna_name[antenna2]),
            "field_name": ("time", field_name_per_time),
            "scan_name": (
                "time",
                np.array([scan_name] * n_time, dtype=str),
                {"scan_intents": list(scan_intents)},
            ),
        }
    )
    telescope = str(antenna_xds.attrs.get("overall_telescope_name", "unknown"))
    now = datetime.now(UTC).isoformat()
    xds.attrs.update(
        {
            "type": "visibility",
            "schema_version": MSV4_SCHEMA_VERSION,
            "creator": {"software_name": "astroviper", "version": version},
            "creation_date": now,
            "observation_info": observation_info
            or {
                "observer": ["astroviper"],
                "project_UID": "astroviper_simulation",
                "release_date": now,
                "execution_block_UID": telescope + "_simulation",
            },
            "processor_info": processor_info
            or {"type": "CORRELATOR", "sub_type": "SIMULATED"},
            "data_groups": {},
        }
    )
    modify_data_groups_xds(
        xds,
        data_group_out_name="base",
        data_group_out={
            "correlated_data": "VISIBILITY",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
            "field_and_source": "field_and_source_base_xds",
        },
        description=description,
    )
    return xds


def create_empty_measurement_set_v4_on_disk(
    ps_store: str,
    ms_name: str,
    ms_xds: xr.Dataset,
    antenna_xds: xr.Dataset,
    field_and_source_xds: xr.Dataset,
    parallel_coords: dict,
    compressor=None,
    double_precision: bool = True,
    overwrite: bool = False,
) -> str:
    """Write a processing set containing one empty MSv4 ready for chunk-wise writes.

    Creates ``<ps_store>/`` (root group with ``type="processing_set"``),
    ``<ps_store>/<ms_name>`` with the coordinates/attrs of ``ms_xds`` and the
    ``antenna_xds`` / ``field_and_source_base_xds`` children, then pre-allocates
    ``VISIBILITY``, ``UVW``, ``WEIGHT`` and ``FLAG`` (chunked along the
    ``parallel_coords`` dimensions) without touching the data.

    Parameters
    ----------
    ps_store : str
        Output processing-set directory (conventionally ``*.ps.zarr``).
    ms_name : str
        Name of the MSv4 group inside the processing set.
    ms_xds : xr.Dataset
        From :func:`make_empty_visibility_xds`.
    antenna_xds, field_and_source_xds : xr.Dataset
    parallel_coords : dict
        GraphVIPER parallel coordinates (keys ``time`` and/or ``frequency``).
    compressor : numcodecs compressor or None
    double_precision : bool
        ``VISIBILITY`` complex128 / ``WEIGHT`` float64 if True, else complex64 / float32.
    overwrite : bool

    Returns
    -------
    str
        Path of the MSv4 group (``<ps_store>/<ms_name>``).
    """
    import shutil

    from xradio._utils.zarr.config import ZARR_FORMAT

    from astroviper.utils.io import (
        create_empty_data_variables_on_disk,
        visibility_data_variables_and_dims_double_precision,
        visibility_data_variables_and_dims_single_precision,
    )

    if os.path.exists(ps_store):
        if not overwrite:
            raise FileExistsError(
                f"{ps_store} exists; pass overwrite=True to replace it."
            )
        shutil.rmtree(ps_store)

    root = xr.DataTree()
    root.attrs["type"] = "processing_set"
    root.to_zarr(store=ps_store, mode="w", zarr_format=ZARR_FORMAT)

    ms_xdt = xr.DataTree(ms_xds)
    ms_xdt["antenna_xds"] = xr.DataTree(antenna_xds)
    ms_xdt["field_and_source_base_xds"] = xr.DataTree(field_and_source_xds)
    ms_path = os.path.join(ps_store, ms_name)
    ms_xdt.to_zarr(store=ms_path, mode="w", zarr_format=ZARR_FORMAT)

    definitions = (
        visibility_data_variables_and_dims_double_precision
        if double_precision
        else visibility_data_variables_and_dims_single_precision
    )
    create_empty_data_variables_on_disk(
        ms_path,
        list(definitions.keys()),
        shape_dict=dict(ms_xds.sizes),
        parallel_coords=parallel_coords,
        compressor=compressor,
        double_precision=double_precision,
        data_variable_definitions=definitions,
    )
    return ms_path


def write_visibility_chunk_to_disk(
    ms_path: str, task_coords: dict, chunk_xds: xr.Dataset
) -> None:
    """Region-write the data variables of ``chunk_xds`` into an on-disk MSv4.

    Parameters
    ----------
    ms_path : str
        ``<ps_store>/<ms_name>`` created by :func:`create_empty_measurement_set_v4_on_disk`.
    task_coords : dict
        GraphVIPER task coordinates; ``task_coords[dim]["slice"]`` selects the
        region along each parallel dimension.
    chunk_xds : xr.Dataset
        In-memory chunk with (a subset of) ``VISIBILITY``, ``UVW``, ``WEIGHT``,
        ``FLAG`` whose dims match the on-disk arrays.
    """
    import zarr

    group = zarr.open_group(ms_path, mode="r+")
    for name, da in chunk_xds.data_vars.items():
        if name not in group:
            continue
        index = tuple(
            task_coords[dim]["slice"] if dim in task_coords else slice(None)
            for dim in da.dims
        )
        array = group[name]
        array[index] = np.asarray(da.values, dtype=array.dtype)
