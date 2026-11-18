"""Report of a :func:`~astroviper.distributed_applications.imaging.find_continuum_ultra.find_continuum_ultra` run."""

from __future__ import annotations

# The node task timing columns of the task stream, for the frequency and the
# merge tasks; a column a task does not have counts as 0.
FIND_CONTINUUM_ULTRA_TASK_STREAM_COLUMNS = {
    "task": "T_find_continuum_task",
    "load": ["T_read_wait", "T_read_partials"],
    "write": ["T_write_partials", "T_write_maps"],
}
# The columns of the rounds table, of those a run has.
_ROUND_COLUMNS = (
    "round_index",
    "round_kind",
    "threshold_sigma",
    "threshold_sigma_mode",
    "final_threshold_sigma",
    "continuum_ranges",
    "maps_ranges",
    "moment_difference_snr",
    "result",
    "next_round_kind",
)


def find_continuum_ultra_report(return_dict, report_store, report="html"):
    """Make the report of a ``find_continuum_ultra`` run.

    The processing section is the one of every distributed application
    (:func:`~astroviper.utils.report.add_processing_section`): the task stream
    has the frequency and merge tasks of every pass over the image store, and
    the driver's work between the passes shows as the gaps between them. The
    results section holds the continuum ranges and flags, the spectrum of the
    last computed round with the continuum ranges
    (:func:`~astroviper.utils.report_plots.plot_continuum_selection`), one row
    per round and one row per pass.

    Parameters
    ----------
    return_dict : dict
        The return dict of the run.
    report_store : str
        The report is ``<report_store>_report.html`` (or ``.pdf``).
    report : str, optional
        ``"html"`` (default), ``"pdf"`` or ``"notebook"``.

    Returns
    -------
    str or None
        The path of the report written; ``None`` for ``"notebook"``.
    """
    import datetime

    import pandas as pd

    from astroviper.distributed_applications.imaging.find_continuum_ultra import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
    )
    from astroviper.utils.report import Report, add_processing_section, report_path
    from astroviper.utils.report_plots import plot_continuum_selection

    tasks = return_dict["timing_node_tasks"]
    load, write = (
        [
            column
            for column in FIND_CONTINUUM_ULTRA_TASK_STREAM_COLUMNS[key]
            if column in tasks
        ]
        for key in ("load", "write")
    )
    # The task stream draws one write column.
    processing = dict(
        return_dict,
        timing_node_tasks=tasks.assign(write_seconds=tasks[write].sum(axis=1)),
    )
    path = report_path(report_store, report)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with Report(
        "find_continuum_ultra report",
        report,
        path,
        subtitle=f"Run directory {report_store}, report made {stamp}.",
    ) as document:
        add_processing_section(
            document,
            processing,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            task_columns={
                "task": FIND_CONTINUUM_ULTRA_TASK_STREAM_COLUMNS["task"],
                "load": load,
                "write": "write_seconds",
            },
            task_name="frequency chunk or merge tile of one pass",
        )
        document.section("Results")
        start, stop = return_dict["channel_range"]
        snr = return_dict["moment_difference_snr"]
        document.table(
            pd.DataFrame(
                {
                    "value": [
                        f"[{start}, {stop})",
                        _ranges(return_dict["continuum_channel_ranges"]),
                        ", ".join(
                            f"{low / 1e9:.6f}-{high / 1e9:.6f}"
                            for low, high in return_dict["continuum_frequency_ranges"]
                        )
                        or "none",
                        f"{return_dict['aggregate_bandwidth'] / 1e9:.6g} GHz",
                        str(return_dict["all_continuum"]),
                        str(return_dict["low_bandwidth"]),
                        str(return_dict["low_spread"]),
                        "none" if snr is None else f"{snr:.4g}",
                        str(return_dict["constant_cube"]),
                        str(return_dict["joint_mask_kind"]),
                        str(return_dict["n_passes"]),
                        ", ".join(
                            f"{variable} {count}"
                            for variable, count in return_dict["cube_reads"].items()
                        ),
                        f"{return_dict['first_round']['source']}, moments "
                        f"{return_dict['first_round']['moments']}",
                    ]
                },
                index=[
                    "channels of the cube (spectral window)",
                    "continuum channel ranges",
                    f"continuum frequency ranges (GHz, {return_dict['frequency_frame']})",
                    "aggregate bandwidth",
                    "all continuum",
                    "low bandwidth",
                    "low spread",
                    "moment difference SNR",
                    "constant cube",
                    "joint mask",
                    "passes over the image store",
                    "passes reading each data variable",
                    "first round",
                ],
            ),
            caption="The results; channel ranges are [start, stop) in channels of "
            "the spectral window.",
        )
        diagnostics = return_dict["diagnostics"]
        spectrum = diagnostics["spectrum"]
        if spectrum is not None:
            document.figure(
                plot_continuum_selection(
                    spectrum.frequency.values,
                    spectrum.values,
                    return_dict["continuum_channel_ranges"],
                    diagnostics["positive_threshold"],
                    diagnostics["negative_threshold"],
                    diagnostics["corrected_median"],
                ),
                caption="The spectrum of the last computed round, as its channel "
                "selection read it, with its thresholds and corrected median, and "
                "the continuum channel ranges of the result.",
            )
        rounds = diagnostics["rounds"]
        if len(rounds):
            columns = [
                column
                for column in rounds
                if column in _ROUND_COLUMNS or column.endswith("_decision")
            ]
            document.table(
                rounds[columns].map(_cell).set_index("round_index"),
                caption=f"findContinuum's rounds, path code {diagnostics['path_code']!r}"
                f", change code {diagnostics['change_code']!r}"
                + (
                    ", reverted to round original's ranges"
                    if diagnostics["reverted"]
                    else ""
                )
                + (
                    ", low-spread fix applied"
                    if diagnostics["low_spread_fix_applied"]
                    else ""
                )
                + ".",
            )
        passes = return_dict["passes"]
        if len(passes):
            document.table(
                passes.drop(
                    columns=[
                        column for column in passes if column.endswith("_unixtime")
                    ]
                ).set_index("pass_index"),
                caption="The passes over the image store, in order: the data "
                "variables, tasks, channels, planes, chunks and bytes read, the planes "
                "per batch, the task memory planned (bytes), the graph and compute "
                "times of the pass and of its merge (seconds), the bytes of the "
                "moment partials and maps, and the driver's work inside the pass.",
            )
    return path


def _ranges(ranges):
    """``[start, stop)`` pairs as text."""
    return ", ".join(f"[{start}, {stop})" for start, stop in ranges) or "none"


def _cell(value):
    """A table cell: channel ranges as text."""
    return _ranges(value) if isinstance(value, list) else value
