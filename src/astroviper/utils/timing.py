"""General timing-summary formatting helpers (non-science utilities).

A small, layout-driven pretty-printer for the per-step ``T_*`` timing
dictionaries produced across AstroVIPER.  The phase layout is supplied by the
caller (see the ``phases`` argument), so the same formatter renders any timing
dict -- imaging, calibration, or otherwise. :func:`node_task_timing_table`
gives the statistics over the node tasks of a distributed application, which
the applications log and their reports show.
"""

__all__ = [
    "format_timing_summary",
    "node_task_phase_columns",
    "node_task_phase_totals",
    "node_task_timing_table",
    "print_timing_summary",
]


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
    # Accept either a dict or a one-row DataFrame.  A timing frame may carry
    # non-numeric metadata columns (e.g. ``hostname``); skip anything that
    # doesn't coerce to a float rather than blowing up on it.
    if hasattr(timing, "columns") and hasattr(timing, "iloc"):
        numeric = {}
        for c in timing.columns:
            try:
                numeric[c] = float(timing[c].iloc[0])
            except (TypeError, ValueError):
                continue
        timing = numeric

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


def _numeric_timing_columns(frame):
    """The numeric ``T_*`` columns of a node-task timing frame."""
    import pandas as pd

    return [
        column
        for column in frame.columns
        if str(column).startswith("T_") and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _over_tasks(values):
    """The mean, median, maximum and total over the tasks (NaN ignored)."""
    return {
        "mean": values.mean(),
        "median": values.median(),
        "max": values.max(),
        "total": values.sum(),
    }


def node_task_phase_columns(timing_node_tasks, phases):
    """The timing columns of every phase of ``phases`` that the node tasks recorded.

    Parameters
    ----------
    timing_node_tasks : pandas.DataFrame
        The ``"timing_node_tasks"`` frame of a distributed application, one
        row per node task.
    phases : list of tuple
        The phase layout of the node task's timings (as for
        :func:`format_timing_summary`, e.g. ``IMAGING_TIMING_PHASES``).

    Returns
    -------
    dict
        ``{phase title: column}`` in layout order, for the phases with a
        recorded total or step: the phase's total column when recorded, else
        the tuple of its recorded step columns.
    """
    columns = _numeric_timing_columns(timing_node_tasks)
    phase_columns = {}
    for title, total_key, steps in phases:
        keys = tuple(key for _, key in steps if key in columns)
        if total_key in columns:
            phase_columns[title] = total_key
        elif keys:
            phase_columns[title] = keys
    return phase_columns


def node_task_phase_totals(timing_node_tasks, phases):
    """The time of every node task in each phase of ``phases``.

    Parameters
    ----------
    timing_node_tasks : pandas.DataFrame
        The ``"timing_node_tasks"`` frame of a distributed application.
    phases : list of tuple
        The phase layout of the node task's timings.

    Returns
    -------
    pandas.DataFrame
        One column per phase with a recorded total or step, named by the
        phase title, in layout order, on the index of ``timing_node_tasks``:
        the phase's total column, or the sum of its recorded steps when it
        has none (seconds); see :func:`node_task_phase_columns`.
    """
    import pandas as pd

    totals = {
        title: (
            timing_node_tasks[column]
            if isinstance(column, str)
            else timing_node_tasks[list(column)].sum(axis=1)
        )
        for title, column in node_task_phase_columns(timing_node_tasks, phases).items()
    }
    return pd.DataFrame(totals, index=timing_node_tasks.index)


def node_task_timing_table(timing_node_tasks, phases=None, total_key=None):
    """The mean, median, maximum and total over the node tasks of their ``T_*`` timings.

    Parameters
    ----------
    timing_node_tasks : pandas.DataFrame
        The ``"timing_node_tasks"`` frame of a distributed application, one
        row per node task.
    phases : list of tuple, optional
        The phase layout of the node task's timings (as for
        :func:`format_timing_summary`, e.g. ``IMAGING_TIMING_PHASES``).
        ``None`` (default): one row per ``T_*`` column.
    total_key : str, optional
        The column of the node task's total time; with ``phases`` it adds the
        task total and the task time outside every phase.

    Returns
    -------
    pandas.DataFrame
        The columns ``mean``, ``median``, ``max`` and ``total`` (seconds), each
        a statistic over the tasks of the per-task values, NaN ignored.
        Without ``phases`` one row per numeric ``T_*`` column, indexed by
        ``"node task step"``. With ``phases`` the index has the levels
        ``"phase"`` and ``"step"``: for every phase with a recorded total or
        step, in layout order, ``"phase total"`` (see
        :func:`node_task_phase_totals`), one row per recorded step and, for a
        phase with both a total column and steps, ``"(unaccounted)"``, the
        phase total minus its steps; then, when ``total_key`` is a column, the
        phase ``"TASK TOTAL"`` with ``total_key`` and ``"(outside the
        phases)"``, the task total minus the phase totals.

    Raises
    ------
    ValueError
        For ``total_key`` without ``phases``.
    """
    import pandas as pd

    if phases is None:
        if total_key is not None:
            raise ValueError(
                "total_key needs phases: without them every T_* column, the "
                "total included, is a row."
            )
        table = pd.DataFrame(
            _over_tasks(timing_node_tasks[_numeric_timing_columns(timing_node_tasks)])
        )
        table.index.name = "node task step"
        return table

    columns = _numeric_timing_columns(timing_node_tasks)
    totals = node_task_phase_totals(timing_node_tasks, phases)
    index, rows = [], []
    for title, phase_total_key, steps in phases:
        if title not in totals:
            continue
        index.append((title, "phase total"))
        rows.append(_over_tasks(totals[title]))
        recorded = [(label, key) for label, key in steps if key in columns]
        for label, key in recorded:
            index.append((title, label))
            rows.append(_over_tasks(timing_node_tasks[key]))
        if phase_total_key in columns and recorded:
            index.append((title, "(unaccounted)"))
            steps_sum = timing_node_tasks[[key for _, key in recorded]].sum(axis=1)
            rows.append(_over_tasks(timing_node_tasks[phase_total_key] - steps_sum))
    if total_key in columns:
        task_total = timing_node_tasks[total_key]
        index.append(("TASK TOTAL", total_key))
        rows.append(_over_tasks(task_total))
        index.append(("TASK TOTAL", "(outside the phases)"))
        rows.append(_over_tasks(task_total - totals.sum(axis=1)))
    if not index:
        return pd.DataFrame(
            columns=["mean", "median", "max", "total"],
            index=pd.MultiIndex.from_arrays([[], []], names=["phase", "step"]),
        )
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["phase", "step"]),
        columns=["mean", "median", "max", "total"],
    )


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
