"""Timing-bookkeeping helpers for the imaging cycle (non-science utilities)."""


def accumulate_timing(timing, return_df, phase=None):
    """Accumulate the columns of a one-row timing frame into ``timing`` in place.

    Each processing-function timing is summed across every call (setup plus all
    residual and model-update cycles) so the final per-chunk timing frame reports
    the total wall time spent in each processing function.

    Parameters
    ----------
    timing : dict
        Accumulator mapping timing-column name to a running total (seconds).
    return_df : pandas.DataFrame or None
        One-row timing frame whose columns are added into ``timing``.  ``None``
        is ignored.
    phase : str, optional
        When given, every ``T_<name>`` column is namespaced to
        ``T_<phase>_<name>`` before being accumulated.  This keeps timings that
        share a name across phases (e.g. ``T_transform_pol`` runs in both the
        once-only preparation and every residual cycle) from being summed into a
        single, indistinguishable number.  Non-``T_`` columns are unchanged.
    """
    if return_df is None:
        return
    for column in return_df.columns:
        name = column
        if phase is not None and column.startswith("T_"):
            name = "T_" + phase + "_" + column[2:]
        timing[name] = timing.get(name, 0.0) + float(return_df[column].iloc[0])


# Phase -> (phase-total key, [(label, leaf key), ...]) layout of the per-chunk
# imaging timing dict produced by
# ``astroviper.processing_functions.imaging.image_cube_single_field``.  Only the
# *leaf* timings are listed (intermediate subtotals such as
# ``T_prep_make_point_spread_function`` and ``T_grid`` contain these leaves and
# are therefore excluded so the leaves can be summed without double counting).
_TIMING_PHASES = [
    (
        "PREPARATION (once per chunk)",
        "T_prep",
        [
            ("imaging weights", "T_prep_weights"),
            ("gcf kernel", "T_prep_gcf"),
            ("vis mask", "T_prep_vis_mask"),
            ("uv-sampling grid (PSF)", "T_prep_uv_sampling_grid"),
            ("fft normalize (PSF)", "T_prep_fft_norm"),
            ("primary beam", "T_prep_primary_beam"),
            ("pol transform", "T_prep_transform_pol"),
            ("psf gaussian fit", "T_prep_psf_fit"),
        ],
    ),
    (
        "RESIDUAL UPDATE (summed over major cycles)",
        "T_residual_cycle",
        [
            ("gcf kernel", "T_gcf"),
            ("pol transform", "T_transform_pol"),
            ("degrid (model->vis)", "T_degrid"),
            ("fft (degrid)", "T_fft_degrid"),
            ("residual vis (obs-model)", "T_residual_vis"),
            ("vis mask", "T_vis_mask"),
            ("uv-sampling grid", "T_uv_sampling_grid"),
            ("vis grid (vis->image)", "T_vis_grid"),
            ("fft (grid)", "T_fft_grid"),
        ],
    ),
    (
        "MODEL UPDATE (summed over major cycles)",
        "T_model_update_cycle",
        [
            ("iteration control", "T_iteration_control"),
            ("make mask", "T_make_mask"),
            ("deconvolve", "T_deconvolve"),
            ("convergence / merge", "T_convergence"),
        ],
    ),
    ("RESTORE", "T_restore", []),
    (
        "NODE I/O",
        None,
        [
            ("make empty image", "T_make_empty_image"),
            ("load visibilities", "T_load"),
            ("write image", "T_write"),
        ],
    ),
]

_TIMING_TOTAL_KEY = "T_image_cube_task"


def format_timing_summary(timing, total_key=_TIMING_TOTAL_KEY):
    """Return a phase-grouped, human-readable summary of an imaging timing dict.

    Groups the per-chunk ``T_*`` timings into the four pipeline phases
    (preparation, residual update, model update, restore) plus node I/O, showing
    each phase's total, its leaf sub-timings, and the time within the phase not
    attributed to any leaf (``unaccounted``).  A trailing line reports the grand
    total and the share of it covered by the listed phases.

    Parameters
    ----------
    timing : dict or pandas.DataFrame
        The imaging timing dict (or a one-row timing frame) produced by
        ``image_cube_single_field``.
    total_key : str, optional
        Key holding the grand-total wall time (default ``"T_image_cube_task"``).

    Returns
    -------
    str
        The formatted multi-line summary.
    """
    # Accept either a dict or a one-row DataFrame.
    if hasattr(timing, "columns") and hasattr(timing, "iloc"):
        timing = {c: float(timing[c].iloc[0]) for c in timing.columns}

    def get(key):
        v = timing.get(key)
        return None if v is None else float(v)

    total = get(total_key)
    lines = []
    lines.append("=" * 64)
    lines.append(" AstroVIPER timing breakdown by phase (seconds)")
    lines.append("=" * 64)

    phase_sum = 0.0
    for title, total_k, leaves in _TIMING_PHASES:
        present = [(lbl, k) for lbl, k in leaves if get(k) is not None]
        phase_total = get(total_k) if total_k else None
        leaf_total = sum(get(k) for _, k in present)
        if phase_total is None and not present:
            continue

        header_total = phase_total if phase_total is not None else leaf_total
        phase_sum += header_total if header_total is not None else 0.0
        pct = (
            f"  ({100.0 * header_total / total:.1f}%)"
            if total and header_total is not None
            else ""
        )
        lines.append("")
        lines.append(f"{title}: {header_total:.3f}{pct}")
        for lbl, k in present:
            v = get(k)
            share = f"  ({100.0 * v / header_total:.0f}%)" if header_total else ""
            lines.append(f"    {lbl:<26}{v:>9.3f}{share}")
        if phase_total is not None and present:
            unacc = phase_total - leaf_total
            lines.append(f"    {'(unaccounted)':<26}{unacc:>9.3f}")

    lines.append("")
    lines.append("-" * 64)
    if total is not None:
        covered = f"  ({100.0 * phase_sum / total:.1f}% of total accounted)"
        lines.append(f" {'TOTAL (T_image_cube_task)':<26}{total:>9.3f}")
        lines.append(f" {'sum of phases above':<26}{phase_sum:>9.3f}{covered}")
    lines.append("=" * 64)
    return "\n".join(lines)
