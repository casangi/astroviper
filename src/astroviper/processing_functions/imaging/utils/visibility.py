"""Shared visibility-side helpers for the imaging processing functions."""


def drop_auto_correlations(ms_xdt):
    """Drop auto-correlation baselines from a measurement set in place.

    ``where(mask, drop=True)`` indexes ``baseline_id`` with an integer array,
    which leaves the (numpy-backed) data variables as transposed,
    non-C-contiguous views.  Downstream C++ kernels (e.g. the degridder in
    :func:`~astroviper.processing_functions.imaging.get_visibility_grid.get_visibility_grid_single_field`,
    reused across major cycles) require C-contiguous input, so C-order is
    restored before storing the dataset back.

    Idempotent: re-running on already-masked data keeps every baseline and is a
    no-op, which is why the setup and each residual cycle can both call it.

    Parameters
    ----------
    ms_xdt : xarray.DataTree
        Measurement-set node whose ``.ds`` is filtered in place.
    """
    import numpy as np

    # Keep only cross-correlations (antenna1 != antenna2).
    mask = ms_xdt["baseline_antenna1_name"] != ms_xdt["baseline_antenna2_name"]
    masked_ds = ms_xdt.ds.where(mask, drop=True)
    for var_name, var in masked_ds.data_vars.items():
        data = var.data
        if isinstance(data, np.ndarray) and not data.flags["C_CONTIGUOUS"]:
            masked_ds[var_name].data = np.ascontiguousarray(data)
    ms_xdt.ds = masked_ds
