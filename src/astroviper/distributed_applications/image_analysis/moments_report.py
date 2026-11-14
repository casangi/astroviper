"""Report of a :func:`~astroviper.distributed_applications.image_analysis.moments.moments` run."""

from __future__ import annotations

# The node task timing columns of the task stream; the streamed read is drawn as load.
MOMENTS_TASK_STREAM_COLUMNS = {
    "task": "T_moments_task",
    "load": ("T_load", "T_stream_read"),
    "write": "T_write",
}
# Largest side of a moment map drawn; larger maps are drawn every k-th pixel.
MAX_MOMENT_IMAGE_SIZE = 2048


def moments_report(return_dict, report="html"):
    """Make the report of a ``moments`` run.

    The processing section is the one of every distributed application
    (:func:`~astroviper.utils.report.add_processing_section`), with the node
    tasks named after their chunks of the parallel axes. The results section
    holds the moment maps per Stokes plane, read from the store written, when
    they are ``(l, m)`` images: the store holds one frequency and more than
    one ``l`` and ``m`` (a moment over ``frequency``); otherwise a note. Maps
    with a side larger than :data:`MAX_MOMENT_IMAGE_SIZE` are drawn every
    k-th pixel along ``l`` and ``m``, and a store with several times is drawn
    at its first time.

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
        document.section("Results")
        _add_moment_maps(document, store)
    return path


def _task_name(n_mapping_parallelism):
    """What a node task works on: ``"m chunk"`` or ``"(l, m) chunk"``."""
    axes = list(n_mapping_parallelism)
    if len(axes) == 1:
        return f"{axes[0]} chunk"
    return f"({', '.join(axes)}) chunk"


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
