"""Report of a :func:`~astroviper.distributed_applications.image_analysis.moments.moments` run."""

from __future__ import annotations

# The node task timing columns of the task stream; the streamed read is drawn as load.
MOMENTS_TASK_STREAM_COLUMNS = {
    "task": "T_moments_task",
    "load": ("T_load", "T_stream_read"),
    "write": "T_write",
}
# The phases of a node task: T_moments = T_stream_read + T_accumulate on the
# streamed path; the slab path (median) records no split (NaN), its T_load
# reads the slab.
STREAMED_TASK_PHASE_COLUMNS = ("T_load", "T_stream_read", "T_accumulate", "T_write")
SLAB_TASK_PHASE_COLUMNS = ("T_load", "T_moments", "T_write")
# Largest side of a moment map drawn; larger maps are drawn every k-th pixel.
MAX_MOMENT_IMAGE_SIZE = 2048


def moments_report(return_dict, report="html"):
    """Make the report of a ``moments`` run.

    The processing section is the one of every distributed application
    (:func:`~astroviper.utils.report.add_processing_section`), with the node
    tasks named after their chunks of the parallel axes, followed by the task
    maps: per task on the grid of two parallel axes, the task time and, for a
    monitored run, the peak resident memory, the bytes read and the mean CPU
    (:func:`~astroviper.utils.resource_plots.plot_task_maps`; a note for a
    run over another number of axes). The results section holds every node
    task's phases, the tasks sorted by their time
    (:func:`~astroviper.utils.resource_plots.plot_task_phase_times`: the read,
    the accumulation and the write of the streamed path, the load, the moments
    and the write of the slab path), the distribution of the tasks' read rate
    when the monitor recorded the I/O counters (Linux), and the moment maps
    per Stokes plane, read from the store written, when they are ``(l, m)``
    images: the store holds one frequency and more than one ``l`` and ``m`` (a
    moment over ``frequency``); otherwise a note. Maps with a side larger
    than :data:`MAX_MOMENT_IMAGE_SIZE` are drawn every k-th pixel along ``l``
    and ``m``, and a store with several times is drawn at its first time.

    Parameters
    ----------
    return_dict : dict
        The return dict of ``moments``.
    report : str, optional
        ``"html"`` (default), ``"pdf"`` or ``"notebook"``.

    Returns
    -------
    str or None
        The path of the report written (``<name>_report.html`` or ``.pdf``
        next to ``return_dict["moments_image_store"]``); ``None`` for
        ``"notebook"``.
    """
    import datetime

    from astroviper.distributed_applications.image_analysis.moments import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
    )
    from astroviper.utils.report import Report, add_processing_section, report_path

    store = return_dict["moments_image_store"]
    path = report_path(store, report)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with Report(
        "moments report",
        report,
        path,
        subtitle=f"Moment maps {store}, report made {stamp}.",
    ) as document:
        add_processing_section(
            document,
            return_dict,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            task_columns=MOMENTS_TASK_STREAM_COLUMNS,
            task_name=_task_name(return_dict["n_mapping_parallelism"]),
        )
        _add_task_maps(document, return_dict)
        document.section("Results")
        _add_task_phase_times(document, return_dict)
        _add_read_rate(document, return_dict)
        _add_moment_maps(document, store)
    return path


def _task_name(n_mapping_parallelism):
    """What a node task works on: ``"m chunk"`` or ``"(l, m) chunk"``."""
    axes = list(n_mapping_parallelism)
    if len(axes) == 1:
        return f"{axes[0]} chunk"
    return f"({', '.join(axes)}) chunk"


def _task_phase_columns(timing_node_tasks):
    """The phase columns of the node tasks: :data:`STREAMED_TASK_PHASE_COLUMNS`
    when every task recorded a finite ``T_stream_read`` and ``T_accumulate``,
    else :data:`SLAB_TASK_PHASE_COLUMNS`."""
    import numpy as np

    if all(
        column in timing_node_tasks.columns
        and np.isfinite(timing_node_tasks[column].to_numpy(dtype=float)).all()
        for column in ("T_stream_read", "T_accumulate")
    ):
        return STREAMED_TASK_PHASE_COLUMNS
    return SLAB_TASK_PHASE_COLUMNS


def _add_task_maps(document, return_dict):
    from astroviper.utils.report import _captured
    from astroviper.utils.resource_plots import plot_task_maps

    document.subsection("Task maps")
    figure, printed = _captured(
        plot_task_maps, return_dict, task_columns=MOMENTS_TASK_STREAM_COLUMNS
    )
    if figure is None:
        document.note("Task maps: " + (printed or "not recorded."))
        return
    (axis_0, _), (axis_1, n_1) = return_dict["n_mapping_parallelism"].items()
    document.figure(
        figure,
        caption=f"Per task on the grid of its chunk indices ({axis_0} up the "
        f"rows, {axis_1} across; task_id = i_{axis_0} x {n_1} + i_{axis_1}): the "
        "task time and, for a monitored run, the peak resident memory, the bytes "
        "read and the mean CPU (%).",
    )


def _add_task_phase_times(document, return_dict):
    from astroviper.utils.report import _captured
    from astroviper.utils.resource_plots import plot_task_phase_times

    document.subsection("Task phase times")
    columns = _task_phase_columns(return_dict["timing_node_tasks"])
    figure, printed = _captured(
        plot_task_phase_times,
        return_dict,
        columns={column: column for column in columns},
        layout="per_task",
        task_columns=MOMENTS_TASK_STREAM_COLUMNS,
    )
    if figure is None:
        document.note("Task phase times: " + (printed or "not recorded."))
        return
    document.figure(
        figure,
        caption="The phases of every node task, the tasks sorted by their time "
        "(T_moments_task), with the median task time: "
        + (
            "T_load (open the store and plan the read), T_stream_read (the "
            "reads and their decompression), T_accumulate (the accumulation of "
            "the moments) and T_write."
            if columns is STREAMED_TASK_PHASE_COLUMNS
            else "T_load (the load of the chunk's slab), T_moments and T_write."
        ),
    )


def _add_read_rate(document, return_dict):
    from astroviper.utils.report import _captured
    from astroviper.utils.resource_plots import plot_task_time_distribution

    document.subsection("Read rate")
    timing_node_tasks = return_dict["timing_node_tasks"]
    if "read_chars" not in timing_node_tasks.columns:
        document.note(
            "The read rate was not recorded: it needs monitor_resources_seconds "
            "and the I/O counters of the monitor, which exist on Linux only."
        )
        return
    # The read is in T_moments on the streamed path, in T_load on the slab path.
    streamed = _task_phase_columns(timing_node_tasks) is STREAMED_TASK_PHASE_COLUMNS
    read_time_column = "T_moments" if streamed else "T_load"
    figure, printed = _captured(
        plot_task_time_distribution,
        return_dict,
        quantity="read_rate",
        kind="histogram",
        bins=40,
        read_time_column=read_time_column,
        task_columns=MOMENTS_TASK_STREAM_COLUMNS,
    )
    if figure is None:
        document.note("Read rate: " + (printed or "not recorded."))
        return
    document.figure(
        figure,
        caption="The read rate of every node task: the bytes its worker process "
        f"read during the task over {read_time_column} (MiB/s). The counter "
        "(read_chars) is the worker process's, at the system call: reads served "
        "from the page cache count, and so do those of other tasks the worker "
        "runs at the same time.",
    )


def _add_moment_maps(document, store):
    import math

    import xarray as xr

    from astroviper.utils.report_plots import plot_stokes_images

    document.subsection("Moment maps")
    moment_xds = xr.open_zarr(store)
    n_frequency, n_l, n_m = (moment_xds.sizes[dim] for dim in ("frequency", "l", "m"))
    if n_frequency != 1 or n_l < 2 or n_m < 2:
        document.note(
            f"The moment maps are not (l, m) images (frequency {n_frequency}, "
            f"l {n_l}, m {n_m}); not drawn."
        )
        return
    step = max(1, math.ceil(max(n_l, n_m) / MAX_MOMENT_IMAGE_SIZE))
    n_time = moment_xds.sizes["time"]
    names = [
        group["sky"]
        for name, group in moment_xds.attrs["data_groups"].items()
        if name.startswith("moment_")
    ]
    for name in names:
        planes = (
            moment_xds[name]
            .isel(
                time=0,
                frequency=0,
                l=slice(None, None, step),
                m=slice(None, None, step),
            )
            .load()
        )
        document.figure(
            plot_stokes_images(
                planes, title=name, colorbar_label=planes.attrs.get("units")
            ),
            caption=f"The moment map {name} of every Stokes plane ({store})"
            + (f", one pixel in {step} along l and m" if step > 1 else "")
            + (f", at the first of its {n_time} times" if n_time > 1 else "")
            + ".",
        )
