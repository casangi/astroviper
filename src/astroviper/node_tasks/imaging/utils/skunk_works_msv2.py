"""Measurement Set v2 input of the skunk-works cube-imaging path (experimental).

A Measurement Set v2 is imaged without converting it, through XRADIO's
``xradio_msv2`` xarray engine (:func:`xradio.measurement_set.open_msv2`):

* the distributed application opens the MSv2 lazily
  (:func:`open_processing_set_skunk_works_msv2`: metadata only, the main data
  variables are lazily indexed arrays) and builds the graph mapping exactly as
  for a Zarr processing set;
* :func:`add_lazy_input_data` then gives every node task the lazily indexed
  selection of its data under the mapping key ``lazy_input_data``: per MSv4,
  the data group's four variables (``correlated_data``, ``flag``, ``weight``,
  ``uvw``) restricted to the task's ``data_selection``, with the coordinates
  the imaging reads (``frequency``, ``polarization`` and the baseline antenna
  names, shipped as small integer codes into a per-MSv4 name table, which
  keeps the graph payload small for arrays with many baselines);
* the node task reads it with :func:`load_processing_set_skunk_works_msv2`,
  which rebuilds ``baseline_antenna1_name`` and ``baseline_antenna2_name``
  exactly.

Every value is read by the engine, so a Measurement Set v2 is imaged as the
processing set that :func:`xradio.measurement_set.convert_msv2_to_processing_set`
(with the same ``partition_scheme``) gives is imaged by the production loader:
the same MSv4 names and data groups, and the same values, frequencies,
correlations and baseline names. AstroVIPER has no MSv2 reader of its own.

The engine is optional: it exists only in XRADIO releases that carry it and
with a casacore backend (python-casacore or casatools).
:func:`msv2_engine_available` tells whether it is there and
:func:`require_msv2_engine` raises an ``ImportError`` that names what is
missing. Module level imports only the standard library and the docstring
marker of :mod:`astroviper.utils.param_docs`, so importing this module never
needs the engine.

python-casacore holds the GIL, so MSv2 reads do not overlap within a process:
run dask with one thread per worker (``threads_per_worker=1``) or the MPI
backend.

The Zarr processing-set sibling of this path is
:mod:`astroviper.node_tasks.imaging.utils.skunk_works`.
"""

from __future__ import annotations

import os

from astroviper.utils.param_docs import shares_param_docs

#: Data-group roles the imaging reads, in the order the variables are kept.
_DATA_GROUP_ROLES = ("correlated_data", "flag", "weight", "uvw")

#: Coordinates the imaging reads; every other coordinate is dropped from the
#: per-task selections (which keeps the graph payload small).
_KEPT_COORDINATES = (
    "frequency",
    "polarization",
    "baseline_antenna1_name",
    "baseline_antenna2_name",
)

#: Baseline antenna-name coordinates -> the integer-code coordinates that
#: replace them in the per-task selections.
_BASELINE_NAME_CODES = {
    "baseline_antenna1_name": "baseline_antenna1_code",
    "baseline_antenna2_name": "baseline_antenna2_code",
}

#: ``attrs`` key of a per-task selection: the antenna names its baseline codes
#: index (the sorted unique names of both coordinates).
_ANTENNA_NAME_TABLE = "baseline_antenna_name_table"

#: Arguments of ``open_msv2`` that the distributed application sets itself:
#: they are refused in ``msv2_open_options`` (name -> what sets it).
_RESERVED_OPEN_OPTIONS = {
    "array_backend": (
        'fixed to "xarray": lazily indexed arrays, so that a node task reads '
        "only its own selection"
    ),
    "scan_intents": "pass the scan_intents parameter instead",
}

#: XRADIO's environment variable for the default partition-cache mode.
_PARTITION_CACHE_ENV = "XRADIO_MSV2_PARTITION_CACHE"


def msv2_engine_available():
    """Whether XRADIO's ``xradio_msv2`` engine can be used.

    The check is a feature probe, not a version comparison:
    :func:`xradio.measurement_set.open_msv2` is exported only by the XRADIO
    releases that carry the engine, and only when a casacore backend
    (python-casacore or casatools) is installed.

    Returns
    -------
    tuple of (bool, str)
        ``(True, "")`` when :func:`xradio.measurement_set.open_msv2` exists,
        otherwise ``(False, reason)`` with a one-line reason (suitable for a
        ``pytest.mark.skipif`` reason).

    See Also
    --------
    require_msv2_engine : the same check, raising an ``ImportError``.
    """
    try:
        import xradio.measurement_set as measurement_set
    except ImportError as error:
        return False, f"xradio.measurement_set cannot be imported ({error})"
    if hasattr(measurement_set, "open_msv2"):
        return True, ""
    return False, (
        "XRADIO has no xradio.measurement_set.open_msv2 (the xradio_msv2 "
        "engine: an XRADIO release that carries it, with python-casacore or "
        "casatools)"
    )


def require_msv2_engine():
    """XRADIO's ``open_msv2``, or an ``ImportError`` that names what is missing.

    Returns
    -------
    callable
        :func:`xradio.measurement_set.open_msv2`.

    Raises
    ------
    ImportError
        If XRADIO has no ``open_msv2``: an XRADIO release without the
        ``xradio_msv2`` engine (1.2.4 and older), or no casacore backend. The
        message names the remedies, including converting the Measurement Set
        to a processing set instead.

    See Also
    --------
    msv2_engine_available : the same check, without raising.
    """
    available, reason = msv2_engine_available()
    if not available:
        raise ImportError(
            "Imaging a Measurement Set v2 directly needs XRADIO's xradio_msv2 "
            f"engine, but {reason}. Install an XRADIO release with the engine "
            "(1.2.5 or later) and python-casacore (pip install "
            "'astroviper[python_casacore]'), or convert the Measurement Set "
            "with xradio.measurement_set.convert_msv2_to_processing_set and "
            "image the processing set."
        )
    from xradio.measurement_set import open_msv2

    return open_msv2


def open_processing_set_skunk_works_msv2(
    ms_path, scan_intents=None, msv2_open_options=None
):
    """Open a Measurement Set v2 lazily as a processing set.

    :func:`xradio.measurement_set.open_msv2` with lazily indexed (not dask)
    arrays: only the metadata is read; a data variable is read from the
    Measurement Set when it is indexed and computed, and then only the rows
    (and, with XRADIO's channel-sliced reads, the channels) of the selection.

    The defaults suit an imaging run, and ``msv2_open_options`` overrides
    them:

    * ``with_pointing=False``: the imaging does not read the ``pointing_xds``.
    * ``partition_cache="read"``, unless the environment variable
      ``XRADIO_MSV2_PARTITION_CACHE`` is set: the partitions stored in the MS
      by an earlier open are used, but the run never writes into its input
      Measurement Set (otherwise the engine's first open stores them there).
    * the engine's ``partition_scheme`` (``[]``, as the converter's), so the
      MSv4 names and data equal those of a default conversion.

    Parameters
    ----------
    ms_path : str or os.PathLike
        Path of the Measurement Set v2 (made absolute, so that the node tasks
        read the same Measurement Set from any working directory).
    scan_intents : str or list of str, optional
        Keep only the MSv4s with one of these scan intents (as
        :func:`xradio.measurement_set.open_processing_set`). ``None``
        (default) keeps every MSv4.
    msv2_open_options : dict, optional
        Options of XRADIO's ``xradio_msv2`` engine (for example
        ``partition_scheme``, ``partition_cache``, ``skip_columns``,
        ``with_pointing``); they override the defaults above. ``None``
        (default): the defaults.

    Returns
    -------
    xarray.DataTree
        The processing set: one child per MSv4, named as the converted
        processing set names it.

    Raises
    ------
    ValueError
        If ``msv2_open_options`` holds ``array_backend`` or ``scan_intents``
        (both are set here), or if no MSv4 is left (none has one of
        ``scan_intents``, or the Measurement Set has no visibilities).
    ImportError
        If XRADIO's ``xradio_msv2`` engine is not available (see
        :func:`require_msv2_engine`).
    """
    options = dict(msv2_open_options or {})
    reserved = sorted(set(options) & set(_RESERVED_OPEN_OPTIONS))
    if reserved:
        raise ValueError(
            f"msv2_open_options may not set {reserved}: "
            + "; ".join(f"{name}: {_RESERVED_OPEN_OPTIONS[name]}" for name in reserved)
            + "."
        )
    open_msv2 = require_msv2_engine()

    defaults = {"with_pointing": False}
    if _PARTITION_CACHE_ENV not in os.environ:
        defaults["partition_cache"] = "read"
    ms_path = os.path.abspath(os.path.expanduser(os.fspath(ms_path)))
    ps_xdt = open_msv2(
        ms_path,
        scan_intents=scan_intents,
        array_backend="xarray",
        **{**defaults, **options},
    )
    if not ps_xdt.children:
        if scan_intents is not None:
            if isinstance(scan_intents, str):
                scan_intents = [scan_intents]
            raise ValueError(
                f"No measurement set of {ms_path} has one of the scan intents "
                f"{list(scan_intents)}. Pass scan_intents=None to image every "
                "scan intent: Measurement Sets from CASA often lack "
                "'OBSERVE_TARGET#ON_SOURCE'."
            )
        raise ValueError(
            f"{ms_path} holds no visibilities that can be imaged (no MSv4 was opened)."
        )
    return ps_xdt


@shares_param_docs
def add_lazy_input_data(ps_xdt, node_task_data_mapping, processing_set_data_group_name):
    """Add every node task's lazily indexed data to the graph mapping, in place.

    Each task of ``node_task_data_mapping`` gets the key ``"lazy_input_data"``:
    ``{ms_name: xarray.Dataset}``, one entry per MSv4 of its
    ``data_selection``. Each dataset holds

    * the four variables of the data group (``correlated_data``, ``flag``,
      ``weight``, ``uvw``), indexed with the task's selection (its
      ``frequency`` slice and, when present, its ``polarization`` index list)
      but not read;
    * only the coordinates the imaging reads: ``frequency``,
      ``polarization``, and the baseline antenna names as integer codes
      (``baseline_antenna1_code`` and ``baseline_antenna2_code``, of the
      smallest unsigned integer type) into the MSv4's antenna-name table;
    * ``attrs``: ``{"data_groups": {processing_set_data_group_name: group},
      "baseline_antenna_name_table": names}``.

    The codes keep the graph payload small: GraphVIPER copies and pickles
    every task's parameters, and the names, repeated per baseline, would
    dominate it for arrays with many antennas.

    GraphVIPER's ``map`` forwards the key to every node task that declares a
    ``lazy_input_data`` parameter; the node task reads it with
    :func:`load_processing_set_skunk_works_msv2`, which rebuilds the names.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Lazily opened processing set, as
        :func:`open_processing_set_skunk_works_msv2` returns it.
    node_task_data_mapping : dict
        GraphVIPER node-task data mapping, with the final ``data_selection``
        of every task (frequency slice and correlation selection). An MSv4
        whose selection is ``None`` is taken whole, as the production loader
        does.
    processing_set_data_group_name : str
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).

    Raises
    ------
    ValueError
        If a selected MSv4 has no such data group, or the group lacks a role
        the imaging reads (a single-dish MSv4 has no ``uvw``), or a variable
        of the group is missing from the MSv4.
    """
    data_group_datasets = {}
    for task in node_task_data_mapping.values():
        lazy_input_data = {}
        for ms_name, selection in task["data_selection"].items():
            if ms_name not in data_group_datasets:
                data_group_datasets[ms_name] = _data_group_dataset(
                    ps_xdt[ms_name], ms_name, processing_set_data_group_name
                )
            lazy_input_data[ms_name] = data_group_datasets[ms_name].isel(
                selection or {}
            )
        task["lazy_input_data"] = lazy_input_data


def _data_group_dataset(ms_xdt, ms_name, data_group_name):
    """The data group's variables of one MSv4, with the kept coordinates only.

    Parameters
    ----------
    ms_xdt : xarray.DataTree
        One MSv4 of the lazily opened processing set.
    ms_name : str
        Its name (for the error messages).
    data_group_name : str
        Data group to image.

    Returns
    -------
    xarray.Dataset
        Lazy dataset: the group's four variables, the kept coordinates (the
        baseline antenna names as codes, see
        :func:`_encode_baseline_antenna_names`) and
        ``attrs["data_groups"] = {data_group_name: group}``.

    Raises
    ------
    ValueError
        If the group, one of its four roles or one of its variables is
        missing.
    """
    data_groups = ms_xdt.attrs.get("data_groups", {})
    if data_group_name not in data_groups:
        raise ValueError(
            f"Measurement set {ms_name} has no data group {data_group_name!r} "
            f"(it has {sorted(data_groups)}); choose another "
            "processing_set_data_group_name."
        )
    data_group = dict(data_groups[data_group_name])
    missing_roles = [role for role in _DATA_GROUP_ROLES if role not in data_group]
    if missing_roles:
        raise ValueError(
            f"Data group {data_group_name!r} of measurement set {ms_name} has no "
            f"{missing_roles}: imaging reads {list(_DATA_GROUP_ROLES)} (a "
            "single-dish measurement set cannot be imaged here)."
        )
    names = [data_group[role] for role in _DATA_GROUP_ROLES]
    dataset = ms_xdt.to_dataset(inherit=False)
    missing_variables = [name for name in names if name not in dataset.data_vars]
    if missing_variables:
        raise ValueError(
            f"Measurement set {ms_name} lacks the variables {missing_variables} "
            f"of its data group {data_group_name!r}."
        )
    dataset = dataset[names]
    dataset = dataset.drop_vars(
        [name for name in dataset.coords if name not in _KEPT_COORDINATES]
    )
    dataset.attrs = {"data_groups": {data_group_name: data_group}}
    return _encode_baseline_antenna_names(dataset)


def _encode_baseline_antenna_names(dataset):
    """Replace the baseline antenna-name coordinates by integer codes.

    Each name coordinate becomes a code coordinate of the smallest unsigned
    integer type into one table of the sorted unique names of both, stored in
    ``attrs["baseline_antenna_name_table"]``. A code coordinate keeps the
    name coordinate's dtype and attributes in its own attributes, so that
    :func:`_decode_baseline_antenna_names` rebuilds it exactly.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset with zero, one or both of the name coordinates.

    Returns
    -------
    xarray.Dataset
        The dataset with codes (unchanged without name coordinates).
    """
    import numpy as np
    import xarray as xr

    names = [name for name in _BASELINE_NAME_CODES if name in dataset.coords]
    if not names:
        return dataset
    values = [dataset[name].values for name in names]
    table, codes = np.unique(
        np.concatenate([value.ravel() for value in values]), return_inverse=True
    )
    codes = codes.astype(np.min_scalar_type(max(table.size - 1, 0)))
    code_coordinates = {}
    start = 0
    for name, value in zip(names, values, strict=True):
        coordinate = dataset[name]
        code_coordinates[_BASELINE_NAME_CODES[name]] = xr.Variable(
            coordinate.dims,
            codes[start : start + value.size].reshape(value.shape),
            attrs={
                "name_dtype": coordinate.dtype.str,
                "name_attrs": dict(coordinate.attrs),
            },
        )
        start += value.size
    dataset = dataset.drop_vars(names).assign_coords(code_coordinates)
    dataset.attrs[_ANTENNA_NAME_TABLE] = table
    return dataset


def _decode_baseline_antenna_names(dataset):
    """Rebuild the baseline antenna-name coordinates from their codes.

    The inverse of :func:`_encode_baseline_antenna_names`: the names, their
    dtype and attributes are those of the coordinates that were encoded.

    Parameters
    ----------
    dataset : xarray.Dataset
        A computed per-task dataset.

    Returns
    -------
    xarray.Dataset
        The dataset with name coordinates and without the codes and the name
        table (unchanged without a name table).
    """
    import numpy as np
    import xarray as xr

    if _ANTENNA_NAME_TABLE not in dataset.attrs:
        return dataset
    table = np.asarray(dataset.attrs[_ANTENNA_NAME_TABLE])
    name_coordinates = {}
    for name, code_name in _BASELINE_NAME_CODES.items():
        if code_name not in dataset.coords:
            continue
        code = dataset[code_name]
        name_coordinates[name] = xr.Variable(
            code.dims,
            table[code.values].astype(code.attrs["name_dtype"]),
            attrs=dict(code.attrs["name_attrs"]),
        )
    dataset = dataset.drop_vars(
        list(_BASELINE_NAME_CODES.values()), errors="ignore"
    ).assign_coords(name_coordinates)
    dataset.attrs = {
        key: value for key, value in dataset.attrs.items() if key != _ANTENNA_NAME_TABLE
    }
    return dataset


def load_processing_set_skunk_works_msv2(lazy_input_data):
    """Read a node task's lazily indexed data into a processing set.

    Each dataset is computed into a new one (the lazy datasets in the task's
    parameters stay unread, so nothing is held for the task's life), and its
    baseline antenna names are rebuilt exactly from their codes. The data
    variables are made NumPy arrays that are C-contiguous and writeable (the
    imaging weights are computed in place in ``WEIGHT``); an array is copied
    only when it is not already so.

    Parameters
    ----------
    lazy_input_data : dict
        ``{ms_name: xarray.Dataset}`` of one task, from
        :func:`add_lazy_input_data`.

    Returns
    -------
    xarray.DataTree
        One child per MSv4, as
        :func:`~astroviper.node_tasks.imaging.utils.load_processing_set_skunk_works`
        returns it: the four data-group variables, the coordinates
        ``frequency``, ``polarization``, ``baseline_antenna1_name`` and
        ``baseline_antenna2_name`` (the MSv4's own values: the converted
        processing set's) and ``attrs["data_groups"]``.

    Raises
    ------
    xradio.measurement_set.MSv2ChangedError
        If the Measurement Set changed after it was opened (see
        :func:`is_fatal_load_error`).
    """
    import numpy as np
    import xarray as xr

    nodes = {}
    for ms_name, lazy_dataset in lazy_input_data.items():
        dataset = _decode_baseline_antenna_names(lazy_dataset.compute())
        for name, variable in dataset.data_vars.items():
            values = variable.values
            if not (values.flags.c_contiguous and values.flags.writeable):
                dataset[name] = variable.copy(data=np.array(values, order="C"))
        nodes[ms_name] = dataset
    return xr.DataTree.from_dict(nodes)


def is_fatal_load_error(error):
    """Whether a node task's data-load error must abort the run.

    A node task skips a chunk whose data cannot be read, but an
    ``xradio.measurement_set.MSv2ChangedError`` means the Measurement Set
    changed after the distributed application opened it: the graph no longer
    describes it, so every remaining task would be inconsistent.

    Parameters
    ----------
    error : BaseException
        The exception raised while loading a task's data.

    Returns
    -------
    bool
        ``True`` for an ``MSv2ChangedError``; ``False`` for any other error,
        and always when XRADIO has no ``xradio_msv2`` engine.
    """
    try:
        from xradio.measurement_set import MSv2ChangedError
    except ImportError:
        return False
    return isinstance(error, MSv2ChangedError)
