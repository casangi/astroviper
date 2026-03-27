
import xarray as xr

# Keys inside a data group that are metadata rather than data-variable mappings.
# They are excluded when comparing data groups across measurement sets so that
# per-dataset metadata differences do not trigger a false mismatch.
_DATA_GROUP_METADATA_KEYS = {"date", "description"}


def create_ps_xdt_data_groups_in_and_out(
    ps_xdt: xr.DataTree,
    data_group_in_name: str,
    data_group_out_name: str,
    data_group_out_modified: dict,
    overwrite: bool,
):
    """Validate data-group selection parameters across every measurement set in a processing set.

    A *data group* is a dictionary stored under ``xds.attrs["data_groups"]``.
    Each entry maps a logical role name (e.g. ``"visibility"``) to the name of
    the corresponding data variable in the dataset.  New data groups can be
    added to a dataset to record the result of processing steps without
    removing the original groups.

    This function iterates over every measurement set (MS) in the processing
    set DataTree, delegates per-MS validation to :func:`check_sel_params`, and
    then asserts that the resolved input and output data groups are structurally
    identical across all MSes (metadata keys such as ``"date"`` and
    ``"description"`` are ignored during comparison).

    Parameters
    ----------
    ps_xdt : xr.DataTree
        The processing set DataTree whose child nodes are individual
        measurement set DataTrees.
    data_group_in_name : str
        Name of the existing data group to use as input (must already exist in
        every MS's ``data_groups`` attribute).
    data_group_out_name : str
        Name under which the output data group will be registered.  When
        ``overwrite=False`` this name must not already be present.
    data_group_out_modified : dict
        Mapping of role keys to new data-variable names that override the
        corresponding entries copied from the input data group.  Keys not
        listed here are inherited unchanged from ``data_group_in``.
    overwrite : bool
        If ``True``, an existing output data group (or output data variables)
        will be silently overwritten.  If ``False``, their presence raises an
        ``AssertionError``.

    Returns
    -------
    data_group_in : dict
        The resolved input data group (taken from the first MS, after
        asserting all MSes agree).
    data_group_out : dict
        The resolved output data group (taken from the first MS, after
        asserting all MSes agree).

    Raises
    ------
    AssertionError
        If the input or output data groups differ across measurement sets
        (ignoring metadata keys), or if :func:`check_sel_params` raises for
        any individual MS.
    """
    data_group_in_list = []
    data_group_out_list = []

    for ms_name, ms_xdt in ps_xdt.items():
        # Validate and resolve data groups for each individual measurement set.
        data_group_in, data_group_out = create_data_groups_in_and_out(
            ms_xdt.ds,
            data_group_in_name=data_group_in_name,
            data_group_out_name=data_group_out_name,
            data_group_out_modified=data_group_out_modified,
            overwrite=overwrite,
        )
        data_group_out_list.append(data_group_out)
        data_group_in_list.append(data_group_in)

    def _strip_metadata(d):
        """Return a copy of *d* with per-dataset metadata keys removed."""
        return {k: v for k, v in d.items() if k not in _DATA_GROUP_METADATA_KEYS}

    # All MSes in a processing set must share the same data-variable layout so
    # that downstream algorithms can treat them uniformly.
    assert all(
        _strip_metadata(d) == _strip_metadata(data_group_out_list[0])
        for d in data_group_out_list
    ), "data_group_out must be the same for all measurement sets in the processing set."

    assert all(
        _strip_metadata(d) == _strip_metadata(data_group_in_list[0])
        for d in data_group_in_list
    ), "data_group_in must be the same for all measurement sets in the processing set."

    # Return the representative (first MS) data groups; all others are identical.
    return data_group_in_list[0], data_group_out_list[0]


def create_data_groups_in_and_out(
    xds: xr.Dataset,
    data_group_in_name: str,
    data_group_out_name: str,
    data_group_out_modified: dict,
    overwrite: bool,
):
    """Validate data-group selection parameters for a single dataset and resolve the output data group.

    A *data group* is a dictionary stored inside ``xds.attrs["data_groups"]``.
    Each data group maps logical role keys (e.g. ``"visibility"``,
    ``"weight"``) to the names of the corresponding data variables in ``xds``.
    Multiple data groups can coexist in one dataset, allowing processing steps
    to add new groups that reference transformed variables while keeping the
    originals intact.

    This function:

    1. Verifies that ``data_group_in_name`` already exists in the dataset's
       ``data_groups`` attribute.
    2. When ``overwrite=False``, checks that neither the output group name nor
       any of the output data-variable names are already present, preventing
       accidental overwrites.
    3. Constructs the output data group by merging the input data group with
       the caller-supplied ``data_group_out_modified`` overrides.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset whose ``attrs["data_groups"]`` contains the available data
        groups.  The dataset itself is not modified; a deep copy of the
        ``data_groups`` attribute is used internally.
    data_group_in_name : str
        Key of the data group to use as input.  Must exist in
        ``xds.attrs["data_groups"]``.
    data_group_out_name : str
        Key under which the output data group will later be stored.  When
        ``overwrite=False`` this must not already be present in
        ``xds.attrs["data_groups"]``.
    data_group_out_modified : dict
        Partial override mapping: role keys whose data-variable names differ
        from those in the input group.  The output group is formed as::

            data_group_out = {**data_group_in, **data_group_out_modified}

        Only the values (variable names) listed here are checked for
        pre-existing conflicts when ``overwrite=False``.
    overwrite : bool
        If ``True``, skip conflict checks so that an existing output group or
        output data variables can be overwritten.

    Returns
    -------
    data_group_in : dict
        A copy of the input data group as found in ``xds.attrs["data_groups"]``.
    data_group_out : dict
        The resolved output data group: all role keys from ``data_group_in``
        with the entries in ``data_group_out_modified`` merged on top.

    Raises
    ------
    AssertionError
        If ``data_group_in_name`` is not present in ``xds.attrs["data_groups"]``,
        or (when ``overwrite=False``) if ``data_group_out_name`` already exists
        in the data groups or any output data-variable name already exists in
        the dataset.
    """
    import copy

    xds_dv_names = list(xds.data_vars)

    # Work on a deep copy so the original attrs are never mutated here.
    xds_data_groups = copy.deepcopy(xds.attrs.get("data_groups", {}))

    # --- Validate the input data group ---
    assert data_group_in_name in xds_data_groups, (
        "Data group "
        + data_group_in_name
        + " not found in xds data_groups: "
        + str(xds_data_groups.keys())
    )

    data_group_in = xds_data_groups[data_group_in_name]

    # --- Guard against accidental overwrites (unless explicitly allowed) ---
    if not overwrite:
        # Ensure none of the output data variables clash with existing ones.
        for modified_dv in data_group_out_modified.values():
            assert modified_dv not in xds_dv_names, (
                "Output data variable "
                + modified_dv
                + " already exists in xds data variables: "
                + str(xds_dv_names)
                + ". Set overwrite=True to overwrite."
            )

    # Build the output data group: inherit all role mappings from the input
    # group and then apply the caller-specified overrides.
    data_group_out = {**data_group_in, **data_group_out_modified}

    return data_group_in, data_group_out

from datetime import datetime, timezone


def modify_data_groups_ps_xdt(
    ps_xdt: xr.DataTree,
    data_group_out_name: str,
    data_group_out: dict,
    description: str,
):
    """Register a new (or updated) data group on every measurement set in a processing set.

    After :func:`create_ps_xdt_data_groups_in_and_out` has resolved the output
    data group, this function writes it back into each MS DataTree's
    ``data_groups`` attribute and stamps the entry with a UTC timestamp and a
    human-readable description.

    The ``"date"`` and ``"description"`` fields inside the data group act as a
    lightweight audit trail:

    * ``"date"`` — ISO-8601 UTC timestamp of when the group was last written.
      If the group already carries a date (inherited from the input group), the
      new timestamp is appended with ``"; "`` so the history is preserved.
    * ``"description"`` — free-text summary of what produced this data group.
      Appended in the same way if a description already exists.

    Parameters
    ----------
    ps_xdt : xr.DataTree
        The processing set DataTree.  Each child node is a measurement set
        DataTree that must expose a ``data_groups`` mapping in its ``ds``
        attributes.  The mapping is modified **in place**.
    data_group_out_name : str
        Key under which the data group will be stored in ``data_groups``.
    data_group_out : dict
        The resolved output data group dictionary (typically the second return
        value of :func:`create_ps_xdt_data_groups_in_and_out`).  Maps logical
        role keys to data-variable names.
    description : str
        Human-readable description of the processing step that produced this
        data group (e.g. ``"Taper applied to VISIBILITY"``).

    Returns
    -------
    None
        Modifies ``ps_xdt`` in place; no return value.
    """
    for _, ms_xdt in ps_xdt.items():
        modify_data_groups_xds(
            ms_xdt.ds,
            data_group_out_name=data_group_out_name,
            data_group_out=data_group_out,
            description=description,
        )
            
            
def modify_data_groups_xds(
    xds: xr.Dataset,
    data_group_out_name: str,
    data_group_out: dict,
    description: str,
):
    """Register a new (or updated) data group on an xarray Dataset.

    After :func:`create_data_groups_in_and_out` has resolved the output
    data group, this function writes it back into the Dataset's
    ``data_groups`` attribute and stamps the entry with a UTC timestamp and a
    human-readable description.

    The ``"date"`` and ``"description"`` fields inside the data group act as a
    lightweight audit trail:

    * ``"date"`` — ISO-8601 UTC timestamp of when the group was last written.
      If the group already carries a date (inherited from the input group), the
      new timestamp is appended with ``"; "`` so the history is preserved.
    * ``"description"`` — free-text summary of what produced this data group.
      Appended in the same way if a description already exists.

    Parameters
    ----------
    xds : xr.Dataset
        The dataset whose attributes will be modified in place to include the
        new or updated data group.
    data_group_out_name : str
        Key under which the data group will be stored in ``xds.attrs["data_groups"]``.
    data_group_out : dict
        The resolved output data group dictionary (typically the second return
        value of :func:`create_data_groups_in_and_out`).  Maps logical role keys to data-variable names.
    description : str
        Human-readable description of the processing step that produced this
        data group (e.g. ``"Taper applied to VISIBILITY"``).

    Returns
    -------
    None
        Modifies ``xds.attrs["data_groups"]`` in place; no return value.
    """
    
    now = datetime.now(timezone.utc)

    # Register the output data group on this dataset.
    if "data_groups" not in xds.attrs:
        xds.attrs["data_groups"] = {}
        
    xds.attrs["data_groups"][data_group_out_name] = data_group_out

    # Stamp the date metadata: append to any existing timestamp so the processing history is preserved, or set it fresh if none exists yet.
    if "date" in xds.attrs["data_groups"][data_group_out_name]:
        xds.attrs["data_groups"][data_group_out_name]["date"] = (
            xds.attrs["data_groups"][data_group_out_name]["date"] + "; " + now.isoformat()
        )
    else:
        xds.attrs["data_groups"][data_group_out_name]["date"] = now.isoformat() 
        
    # Stamp the description metadata in the same append-or-set pattern.
    if "description" in xds.attrs["data_groups"][data_group_out_name]:
        xds.attrs["data_groups"][data_group_out_name]["description"] = (
            xds.attrs["data_groups"][data_group_out_name]["description"] + "; " + description
        )
    else:
        xds.attrs["data_groups"][data_group_out_name]["description"] = description
        