"""General timing-summary formatting helpers (non-science utilities).

A small, layout-driven pretty-printer for the per-step ``T_*`` timing
dictionaries produced across AstroVIPER.  The phase layout is supplied by the
caller (see the ``phases`` argument), so the same formatter renders any timing
dict -- imaging, calibration, or otherwise.
"""

__all__ = ["format_timing_summary", "print_timing_summary"]


def format_timing_summary(
    timing,
    phases,
    total_key=None,
    title="AstroVIPER timing breakdown by phase (seconds)",
    total_label=None,
):
    """Return a phase-grouped, human-readable summary of a timing dict.

    Groups the per-step ``T_*`` timings into the phases described by ``phases``,
    showing each phase's total, its leaf sub-timings, and the time within the
    phase not attributed to any leaf (``unaccounted``).  When ``total_key`` is
    given, a trailing block reports the grand total and the share of it covered
    by the listed phases.  Missing keys are simply skipped, so a partial timing
    dict still renders.

    Parameters
    ----------
    timing : dict or pandas.DataFrame
        The timing dict (or a one-row timing frame) to summarize.  Keys/columns
        are timing names; values are seconds.
    phases : list of tuple
        Phase layout describing how to group the timings::

            [(phase_title, phase_total_key, [(leaf_label, leaf_key), ...]), ...]

        ``phase_total_key`` may be ``None`` for a phase whose total is just the
        sum of its leaves (e.g. node I/O).  Only *leaf* keys should be listed;
        intermediate subtotals would double-count.
    total_key : str, optional
        Key holding the grand-total wall time.  When ``None`` (default) the
        per-phase percentages and the grand-total line are omitted (the divider
        rule is still drawn).
    title : str, optional
        Heading shown in the banner.
    total_label : str, optional
        Label for the grand-total line.  Defaults to ``"TOTAL (<total_key>)"``.

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

    total = get(total_key) if total_key else None
    lines = []
    lines.append("=" * 64)
    lines.append(" " + title)
    lines.append("=" * 64)

    phase_sum = 0.0
    for phase_title, total_k, leaves in phases:
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
        lines.append(f"{phase_title}: {header_total:.3f}{pct}")
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
        label = total_label or f"TOTAL ({total_key})"
        covered = f"  ({100.0 * phase_sum / total:.1f}% of total accounted)"
        lines.append(f" {label:<26}{total:>9.3f}")
        lines.append(f" {'sum of phases above':<26}{phase_sum:>9.3f}{covered}")
    lines.append("=" * 64)
    return "\n".join(lines)


def print_timing_summary(timing, phases, printer=print, **kwargs):
    """Pretty-print a timing summary.

    Thin wrapper over :func:`format_timing_summary`; all keyword arguments are
    forwarded to it.

    Parameters
    ----------
    timing : dict or pandas.DataFrame
        The timing dict (or one-row timing frame) to summarize.
    phases : list of tuple
        Phase layout; see :func:`format_timing_summary`.
    printer : callable, optional
        Sink for the formatted string (default the builtin :func:`print`).  Pass
        e.g. ``logger.debug`` to route the summary through a logger instead.
    **kwargs
        Forwarded to :func:`format_timing_summary` (``total_key``, ``title``,
        ``total_label``).
    """
    printer(format_timing_summary(timing, phases, **kwargs))
