"""Report of a :func:`~astroviper.distributed_applications.visibility_manipulation.uv_continuum_subtraction.uv_continuum_subtraction` run."""

from __future__ import annotations

# At most this many MSv4s get a figure of their mean spectra.
MAX_SPECTRUM_FIGURES = 8


def uv_continuum_subtraction_report(return_dict, report_store, report="html", fit=None):
    """Make the report of a ``uv_continuum_subtraction`` run.

    The processing section is the one of every distributed application
    (:func:`~astroviper.utils.report.add_processing_section`), with one node
    task per block of times of an MSv4. The results section holds the fit, the
    continuum channels and the fit statistics of every MSv4, and the mean
    spectra before and after the subtraction
    (:func:`~astroviper.utils.report_plots.plot_continuum_subtraction`).

    Parameters
    ----------
    return_dict : dict
        The return dict of the run.
    report_store : str
        The report is ``<name>_report.html`` (or ``.pdf``) next to
        ``report_store``, ``<name>`` being ``report_store`` without a store
        suffix (:func:`~astroviper.utils.report.report_path`); the application
        passes the processing set without its suffix, followed by
        ``_<spectral_window_name>``.
    report : str, optional
        ``"html"`` (default), ``"pdf"`` or ``"notebook"``.
    fit : dict, optional
        The run's ``spectral_window_name``, ``continuum_frequency_ranges``,
        ``continuum_frequency_frame``, ``fit_order``,
        ``ms_data_group_in_name`` and ``ms_data_group_out_name``.

    Returns
    -------
    str or None
        The path of the report written; ``None`` for ``"notebook"``.
    """
    import datetime

    import pandas as pd

    from astroviper.distributed_applications.visibility_manipulation.uv_continuum_subtraction import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
        NODE_TASK_TIMING_PHASES,
        TASK_STREAM_COLUMNS,
    )
    from astroviper.utils.report import Report, add_processing_section, report_path
    from astroviper.utils.report_plots import plot_continuum_subtraction

    path = report_path(report_store, report)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with Report(
        "uv_continuum_subtraction report",
        report,
        path,
        subtitle=f"Processing set {return_dict['ps_store']}"
        + (f", spectral window {fit['spectral_window_name']}" if fit else "")
        + f", report made {stamp}.",
    ) as document:
        add_processing_section(
            document,
            return_dict,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            task_columns=TASK_STREAM_COLUMNS,
            task_name="block of times of one MSv4",
            node_task_timing_phases=NODE_TASK_TIMING_PHASES,
        )
        document.section("Results")
        if fit is not None:
            document.table(
                pd.DataFrame(
                    {
                        "value": [
                            fit["spectral_window_name"],
                            ", ".join(
                                f"{low / 1e9:.6f}-{high / 1e9:.6f}"
                                for low, high in fit["continuum_frequency_ranges"]
                            ),
                            fit["continuum_frequency_frame"],
                            str(fit["fit_order"]),
                            f"{fit['ms_data_group_in_name']} -> "
                            f"{fit['ms_data_group_out_name']}",
                        ]
                    },
                    index=[
                        "spectral window",
                        "continuum frequency ranges (GHz)",
                        "frame of the ranges",
                        "fit order",
                        "data groups",
                    ],
                ),
                caption="The continuum fit.",
            )
        document.table(
            return_dict["continuum_channels"].set_index("ms_name"),
            caption="The continuum channels of every MSv4: the fewest and most of a "
            "time, the channels that are continuum channels at every time and at "
            "some time (the selection CASA's pipeline fits at all times), and the "
            "fraction of the channels of all times that are.",
        )
        if len(return_dict["fit_statistics"]):
            document.table(
                return_dict["fit_statistics"].set_index(["ms_name", "polarization"]),
                caption="The fits of every MSv4 and polarization: the spectra, those "
                "fitted, fitted with a lower order, without a valid continuum channel "
                "and, of those, flagged by the subtraction; the chi-squared of the "
                "fits of the real and imaginary parts (weighted sums of the squared "
                "residuals of the continuum channels).",
            )
        for name in list(return_dict["mean_spectra"])[:MAX_SPECTRUM_FIGURES]:
            spectra = return_dict["mean_spectra"][name]
            document.figure(
                plot_continuum_subtraction(
                    spectra.frequency.values,
                    spectra["VISIBILITY_MEAN_INPUT"].values,
                    spectra["VISIBILITY_MEAN_OUTPUT"].values,
                    spectra["CONTINUUM_TIME_FRACTION"].values,
                    [str(label) for label in spectra.polarization.values],
                    title=name,
                ),
                caption=f"{name}: the amplitude of the weighted mean of the "
                "valid visibilities over the times and baselines, before and "
                "after the continuum subtraction, with the continuum channels "
                "shaded.",
            )
    return path
