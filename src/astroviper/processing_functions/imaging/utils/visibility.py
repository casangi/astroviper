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
    TRUE no-op -- it returns before touching the dataset. This matters for
    memory, not just speed: ``where(..., drop=True)`` fancy-indexes EVERY data
    variable into a fresh array, so on a multi-cycle run (which calls this once
    per residual cycle) the old spelling copied the entire measurement set --
    observed + model + residual visibilities, weights, uvw -- every cycle,
    transiently doubling the measurement set's residency (part of the
    2026-08-16 multi-cycle OOM).

    Parameters
    ----------
    ms_xdt : xarray.DataTree
        Measurement-set node whose ``.ds`` is filtered in place.
    """
    import numpy as np

    # Keep only cross-correlations (antenna1 != antenna2).
    mask = ms_xdt["baseline_antenna1_name"] != ms_xdt["baseline_antenna2_name"]
    if bool(mask.values.all()):
        return  # nothing to drop (already masked, or no auto-correlations)
    masked_ds = ms_xdt.ds.where(mask, drop=True)
    for var_name, var in masked_ds.data_vars.items():
        data = var.data
        if isinstance(data, np.ndarray) and not data.flags["C_CONTIGUOUS"]:
            masked_ds[var_name].data = np.ascontiguousarray(data)
    ms_xdt.ds = masked_ds
