"""Report of a :func:`~astroviper.distributed_applications.imaging.find_continuum.find_continuum.find_continuum` or ``find_continuum_fused`` run."""

from __future__ import annotations

# The node task timing columns of the task stream.
FIND_CONTINUUM_TASK_STREAM_COLUMNS = {
    "task": "T_find_continuum_task",
    "load": ["T_load"],
    "write": None,
}


def find_continuum_report(
    return_dict, report_store, report="html", application="find_continuum"
):
    """Make the report of a ``find_continuum`` or ``find_continuum_fused`` run.

    The processing section is the one of every distributed application
    (:func:`~astroviper.utils.report.add_processing_section`); the task stream
    has the node tasks of every pass over the image store, and the driver's
    work between the passes shows as the gaps between them. The results
    section holds the channel selection and findContinuum's other results,
    its summary plot, and one row per pass with the reductions it made.

    Parameters
    ----------
    return_dict : dict
        The return dict of the run.
    report_store : str
        The report is ``<report_store>_report.html`` (or ``.pdf``).
    report : str, optional
        ``"html"`` (default), ``"pdf"`` or ``"notebook"``.
    application : str, optional
        ``"find_continuum"`` (default) or ``"find_continuum_fused"``.

    Returns
    -------
    str or None
        The path of the report written; ``None`` for ``"notebook"``.
    """
    import datetime
    import os

    import matplotlib.image
    import pandas as pd
    from matplotlib.figure import Figure

    from astroviper.distributed_applications.imaging.find_continuum.find_continuum import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
    )
    from astroviper.utils.report import Report, add_processing_section, report_path

    path = report_path(report_store, report)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with Report(
        f"{application} report",
        report,
        path,
        subtitle=f"Work directory {os.path.dirname(report_store)}, report made {stamp}.",
    ) as document:
        add_processing_section(
            document,
            return_dict,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            task_columns=FIND_CONTINUUM_TASK_STREAM_COLUMNS,
            task_name="frequency chunk of one pass",
        )
        document.section("Results")
        warnings = [warning for warning in return_dict["warnings"] if warning]
        document.table(
            pd.DataFrame(
                {
                    "value": [
                        return_dict["channel_selection"],
                        f"{return_dict['aggregate_bandwidth']:.4g} GHz",
                        str(return_dict["all_continuum"]),
                        f"{return_dict['moment_difference_snr']:.4g}",
                        "; ".join(warnings) or "none",
                        str(return_dict["n_cube_reductions"]),
                        ", ".join(
                            f"{variable} {count}"
                            for variable, count in return_dict["cube_reads"].items()
                        ),
                    ]
                },
                index=[
                    "channel selection",
                    "aggregate bandwidth",
                    "all continuum",
                    "moment difference SNR",
                    "warnings",
                    "passes over the image store",
                    "passes reading each data variable",
                ],
            ),
            caption="findContinuum's results.",
        )
        if return_dict["plot"] and os.path.isfile(return_dict["plot"]):
            image = matplotlib.image.imread(return_dict["plot"])
            height, width = image.shape[:2]
            figure = Figure(figsize=(width / 100, height / 100))
            axes = figure.add_axes((0, 0, 1, 1))
            axes.imshow(image)
            axes.set_axis_off()
            document.figure(
                figure,
                caption="findContinuum's summary plot: the mean spectrum in the "
                "joint mask and the continuum ranges selected.",
            )
        passes = return_dict["passes"]
        if len(passes):
            document.table(
                passes.drop(columns=["start_unixtime"]).set_index("pass"),
                caption="The passes over the image store, in order, with the "
                "reductions each made and its graph and compute times (seconds).",
            )
    return path
