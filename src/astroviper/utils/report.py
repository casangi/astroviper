"""Reports of the distributed applications.

A distributed application called with ``report="html"``, ``"pdf"`` or
``"notebook"`` makes a compact report of its run once its output is written.
Every report has a processing section common to all distributed applications
(:func:`add_processing_section`: the timing of the application's steps and of
its node tasks, the task stream and, for a monitored run, the cluster CPU,
memory and I/O) and a results section of the application's own, assembled
from text, tables and figures with :class:`Report`.

The report of an output store ``cube.img.zarr`` is ``cube_report.html`` (or
``.pdf``) next to it, and the products a report makes with other distributed
applications are kept in ``cube_report_files`` (:func:`report_path`,
:func:`report_files_directory`); a ``"notebook"`` report is displayed in the
calling Jupyter notebook and writes no file.
"""

from __future__ import annotations

import base64
import html
import io
import os

__all__ = [
    "REPORT_FORMATS",
    "Report",
    "add_processing_section",
    "make_report",
    "node_task_timing_table",
    "report_files_directory",
    "report_path",
]

REPORT_FORMATS = ("html", "pdf", "notebook")

# Suffixes of the output stores, longest first, stripped to name the report.
_STORE_SUFFIXES = (".img.zarr", ".ps.zarr", ".img.fits", ".zarr", ".fits")

_CSS = """
.astroviper-report {font-family: -apple-system, "Segoe UI", Helvetica, Arial,
  sans-serif; font-size: 14px; line-height: 1.4; color: #1f2328;
  background: #ffffff; max-width: 1100px; margin: 0 auto; padding: 16px;}
.astroviper-report h1 {font-size: 22px; margin: 0 0 4px 0;}
.astroviper-report h2 {font-size: 18px; margin: 24px 0 8px 0;
  border-bottom: 1px solid #d0d7de; padding-bottom: 4px;}
.astroviper-report h3 {font-size: 15px; margin: 16px 0 6px 0;}
.astroviper-report p {margin: 4px 0 8px 0;}
.astroviper-report .note {color: #57606a; font-style: italic;}
.astroviper-report .caption {color: #57606a; font-size: 12px;}
.astroviper-report pre {font-size: 12px; background: #f6f8fa; padding: 8px;
  overflow-x: auto;}
.astroviper-report table {border-collapse: collapse; font-size: 12px;
  margin: 4px 0;}
.astroviper-report th, .astroviper-report td {border: 1px solid #d0d7de;
  padding: 2px 8px; text-align: right;}
.astroviper-report th {background: #f6f8fa;}
.astroviper-report figure {margin: 8px 0 16px 0;}
.astroviper-report img {max-width: 100%; height: auto;}
.astroviper-report figcaption {color: #57606a; font-size: 12px;}
"""


def _report_name(store):
    """The path of ``store`` without its store suffix (``.img.zarr``, ...)."""
    path = os.path.normpath(str(store))
    for suffix in _STORE_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def report_path(store, report):
    """The path of the report of an output store.

    Parameters
    ----------
    store : str
        The output store of the distributed application (an image store or a
        processing set).
    report : str
        ``"html"``, ``"pdf"`` or ``"notebook"``.

    Returns
    -------
    str or None
        ``<name>_report.html`` or ``<name>_report.pdf`` next to ``store``,
        where ``<name>`` is ``store`` without its suffix (``cube.img.zarr``
        gives ``cube_report.html``); ``None`` for ``"notebook"``.
    """
    if report not in REPORT_FORMATS:
        raise ValueError(f"report must be one of {REPORT_FORMATS}; got {report!r}.")
    if report == "notebook":
        return None
    return f"{_report_name(store)}_report.{report}"


def report_files_directory(store):
    """The directory of the products the report of ``store`` makes,
    ``<name>_report_files`` next to ``store`` (see :func:`report_path`); on the
    file system of ``store``, so that remote workers can write into it."""
    return f"{_report_name(store)}_report_files"


def make_report(report_function, *args, **kwargs):
    """Make the report of a run, logging a failure instead of raising.

    A distributed application calls this once its output is written, so that a
    report that cannot be made never costs the run its results.

    Parameters
    ----------
    report_function : callable
        The application's report function, e.g.
        :func:`~astroviper.distributed_applications.imaging.image_cube_single_field_report.image_cube_single_field_report`.
    *args, **kwargs
        Passed to ``report_function``.

    Returns
    -------
    str or None
        What ``report_function`` returns (the path of the report written),
        ``None`` when it failed.
    """
    import traceback

    import toolviper.utils.logger as logger

    try:
        path = report_function(*args, **kwargs)
    except Exception:
        logger.error(
            f"The report could not be made ({report_function.__name__}); the "
            "results of the run are not affected.\n" + traceback.format_exc()
        )
        return None
    if path is not None:
        logger.info(f"Report written to {path}")
    return path


def _close_figure(figure):
    """Release ``figure``, also from pyplot when it was made there."""
    import matplotlib.pyplot as plt

    plt.close(figure)


class Report:
    """A compact report of headings, text, tables and figures.

    Items are rendered as they are added: a figure is drawn into the report
    and closed at once, so a report never holds more than one figure. The
    report is finished by :meth:`close` (or by leaving its ``with`` block):
    ``"html"`` writes one self contained file (figures embedded as PNG),
    ``"pdf"`` a PDF of letter pages in which text, tables and figures flow
    as in the HTML, and ``"notebook"`` displays the HTML in the calling
    Jupyter notebook.

    Parameters
    ----------
    title : str
        Title of the report.
    report : str
        ``"html"``, ``"pdf"`` or ``"notebook"``.
    path : str, optional
        File written by ``"html"`` and ``"pdf"`` (required for them, ignored
        for ``"notebook"``).
    subtitle : str, optional
        Line shown under the title.
    dpi : int, optional
        Resolution of the embedded figures. Default 100.
    """

    def __init__(self, title, report, path=None, subtitle=None, dpi=100):
        if report not in REPORT_FORMATS:
            raise ValueError(f"report must be one of {REPORT_FORMATS}; got {report!r}.")
        if report != "notebook" and path is None:
            raise ValueError(f'A "{report}" report needs a path.')
        self.report = report
        self.path = path
        self.dpi = dpi
        self._html = []
        self._pdf = _PdfFlow(path) if report == "pdf" else None
        self._closed = False
        self.heading(title, level=1)
        if subtitle:
            self.text(subtitle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False

    def heading(self, title, level=2):
        """Add a heading; ``level`` 1 is the title, 2 a section, 3 a subsection."""
        self._html.append(f"<h{level}>{html.escape(title)}</h{level}>")
        if self._pdf is not None:
            self._pdf.heading(title, size={1: 16, 2: 13, 3: 11}[level])

    def section(self, title):
        """Start a section."""
        self.heading(title, level=2)

    def subsection(self, title):
        """Start a subsection."""
        self.heading(title, level=3)

    def text(self, text):
        """Add a paragraph."""
        self._html.append(f"<p>{html.escape(text)}</p>")
        if self._pdf is not None:
            self._pdf.text(text, size=9)

    def note(self, text):
        """Add a note, e.g. on content the run did not record."""
        self._html.append(f'<p class="note">{html.escape(text)}</p>')
        if self._pdf is not None:
            self._pdf.text(text, size=9, style="italic", color="#57606a")

    def preformatted(self, text):
        """Add a block of monospace text (e.g. a timing summary)."""
        self._html.append(f"<pre>{html.escape(text)}</pre>")
        if self._pdf is not None:
            self._pdf.text(text, size=7, family="monospace", wrap=False)

    def table(self, frame, caption=None, float_format="{:.4g}"):
        """Add a :class:`pandas.DataFrame` as a table, its index as the first column."""
        formatter = float_format.format
        self._html.append(
            frame.to_html(border=0, float_format=formatter, na_rep="", escape=True)
        )
        if self._pdf is not None:
            self._pdf.text(
                frame.to_string(float_format=formatter, na_rep=""),
                size=7,
                family="monospace",
                wrap=False,
            )
        if caption:
            self._caption(caption)

    def figure(self, figure, caption=None):
        """Add a matplotlib figure (rendered now, then closed)."""
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=self.dpi, bbox_inches="tight")
        finally:
            _close_figure(figure)
        png = buffer.getvalue()
        alt = html.escape(caption or "figure", quote=True)
        self._html.append(
            '<figure><img alt="{}" src="data:image/png;base64,{}">{}</figure>'.format(
                alt,
                base64.b64encode(png).decode("ascii"),
                f"<figcaption>{html.escape(caption)}</figcaption>" if caption else "",
            )
        )
        if self._pdf is not None:
            self._pdf.image(png, self.dpi)
            if caption:
                self._pdf.text(caption, size=8, color="#57606a")

    def _caption(self, caption):
        self._html.append(f'<p class="caption">{html.escape(caption)}</p>')
        if self._pdf is not None:
            self._pdf.text(caption, size=8, color="#57606a")

    def html(self):
        """The report as an HTML fragment (a ``<div>`` with its own style)."""
        return (
            f"<style>{_CSS}</style>"
            '<div class="astroviper-report">' + "\n".join(self._html) + "</div>"
        )

    def close(self):
        """Finish the report: write the file or display it.

        Returns
        -------
        str or None
            The path of the file written; ``None`` for ``"notebook"``.
        """
        if self._closed:
            return self.path if self.report != "notebook" else None
        self._closed = True
        if self.report == "pdf":
            self._pdf.close()
            return self.path
        if self.report == "html":
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as file:
                file.write(
                    '<!doctype html>\n<html><head><meta charset="utf-8">'
                    f"<title>{html.escape(_first_heading(self._html))}</title>"
                    '<meta name="viewport" content="width=device-width, '
                    'initial-scale=1"></head><body style="margin: 0; '
                    f'background: #ffffff;">{self.html()}</body></html>\n'
                )
            return self.path
        from IPython.display import HTML, display

        display(HTML(self.html()))
        return None

    def abort(self):
        """Discard an unfinished report (a partial PDF is removed)."""
        if self._closed:
            return
        self._closed = True
        if self._pdf is not None:
            self._pdf.close()
            if os.path.exists(self.path):
                os.remove(self.path)


def _first_heading(fragments):
    """The text of the first heading of the HTML fragments (the title)."""
    import re

    for fragment in fragments:
        match = re.match(r"<h\d>(.*)</h\d>", fragment)
        if match:
            return html.unescape(match.group(1))
    return "AstroVIPER report"


class _PdfFlow:
    """Letter pages on which text lines and images flow from top to bottom."""

    PAGE_SIZE = (8.5, 11.0)  # inches
    MARGIN = 0.6  # inches
    GAP = 0.08  # inches between items

    def __init__(self, path):
        from matplotlib.backends.backend_pdf import PdfPages

        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._pages = PdfPages(path)
        self._figure = None
        self._headings = []
        self._new_page()

    @property
    def _width(self):
        return self.PAGE_SIZE[0] - 2 * self.MARGIN

    def _new_page(self):
        from matplotlib.figure import Figure

        self._figure = Figure(figsize=self.PAGE_SIZE)
        self._top = self.PAGE_SIZE[1] - self.MARGIN
        self._empty = True

    def _finish_page(self):
        if not self._empty:
            self._pages.savefig(self._figure)
        self._figure = None

    @property
    def _page_height(self):
        return self.PAGE_SIZE[1] - 2 * self.MARGIN

    def _room(self, height):
        """Start a new page unless ``height`` inches fit on this one."""
        if not self._empty and self._top - height < self.MARGIN:
            self._finish_page()
            self._new_page()

    def heading(self, text, size):
        """Add a heading, held back until the item after it, so that the two
        start on the same page."""
        self._headings.append(self._layout(text, size, "sans-serif", wrap=True))

    def _layout(self, text, size, family, wrap):
        """The lines, font size and line height of ``text``: wrapped to the
        page width, or for ``wrap=False`` (tables, monospace blocks) shrunk
        until its longest line fits."""
        import textwrap

        lines = text.splitlines() or [""]
        # Average glyph width: 0.6 em for monospace, about 0.5 em otherwise.
        em = 0.6 if family == "monospace" else 0.5
        if wrap:
            width = max(20, int(self._width * 72 / (em * size)))
            lines = [
                wrapped
                for line in lines
                for wrapped in (textwrap.wrap(line, width) or [""])
            ]
        else:
            longest = max(len(line) for line in lines)
            fit = self._width * 72 / (em * max(longest, 1))
            size = max(4.0, min(size, fit))
        line_height = 1.25 * size / 72
        return {
            "lines": lines,
            "size": size,
            "family": family,
            "line_height": line_height,
            "height": len(lines) * line_height + self.GAP,
        }

    def _place(self, height):
        """Make room for ``height`` inches of an item and the held headings
        before it, then draw the headings."""
        headings, self._headings = self._headings, []
        self._room(sum(heading["height"] for heading in headings) + height)
        for heading in headings:
            self._draw_lines(heading, weight="bold")

    def _draw_lines(self, block, weight="normal", style="normal", color="#1f2328"):
        for line in block["lines"]:
            self._room(block["line_height"])
            self._figure.text(
                self.MARGIN / self.PAGE_SIZE[0],
                self._top / self.PAGE_SIZE[1],
                line,
                fontsize=block["size"],
                family=block["family"],
                weight=weight,
                style=style,
                color=color,
                va="top",
                ha="left",
            )
            self._top -= block["line_height"]
            self._empty = False
        self._top -= self.GAP

    def text(
        self,
        text,
        size,
        family="sans-serif",
        style="normal",
        color="#1f2328",
        wrap=True,
    ):
        """Add ``text`` (see :meth:`_layout`); a block that fits on one page
        is not split, a longer one flows on."""
        block = self._layout(text, size, family, wrap)
        self._place(
            block["height"]
            if block["height"] < self._page_height
            else block["line_height"]
        )
        self._draw_lines(block, style=style, color=color)

    def image(self, png, dpi):
        """Add a PNG image at its size (``dpi``), shrunk to fit the page."""
        import matplotlib.image as mpimg

        image = mpimg.imread(io.BytesIO(png), format="png")
        height_px, width_px = image.shape[:2]
        width = min(self._width, width_px / dpi)
        height = width * height_px / width_px
        # Room for the headings held before the image on the same page.
        limit = self._page_height - sum(heading["height"] for heading in self._headings)
        if height > limit:
            height = limit
            width = height * width_px / height_px
        self._place(height)
        axes = self._figure.add_axes(
            [
                self.MARGIN / self.PAGE_SIZE[0],
                (self._top - height) / self.PAGE_SIZE[1],
                width / self.PAGE_SIZE[0],
                height / self.PAGE_SIZE[1],
            ]
        )
        axes.imshow(image, interpolation="antialiased")
        axes.set_axis_off()
        self._top -= height + self.GAP
        self._empty = False

    def close(self):
        self._place(0.0)
        if self._figure is not None:
            self._finish_page()
        self._pages.close()


def node_task_timing_table(timing_node_tasks):
    """The mean, maximum and total over the node tasks of every ``T_*`` timing.

    Parameters
    ----------
    timing_node_tasks : pandas.DataFrame
        The ``"timing_node_tasks"`` frame of a distributed application, one
        row per node task.

    Returns
    -------
    pandas.DataFrame
        One row per numeric ``T_*`` column (seconds), with the columns
        ``mean``, ``max`` and ``total``.
    """
    import pandas as pd

    columns = [
        column
        for column in timing_node_tasks.columns
        if str(column).startswith("T_")
        and pd.api.types.is_numeric_dtype(timing_node_tasks[column])
    ]
    timing = timing_node_tasks[columns]
    table = pd.DataFrame(
        {"mean": timing.mean(), "max": timing.max(), "total": timing.sum()}
    )
    table.index.name = "node task step"
    return table


def add_processing_section(
    document, return_dict, timing_phases, task_columns=None, task_name="node task"
):
    """Add the processing section common to all distributed applications.

    The timing of the application's own steps
    (:func:`~astroviper.utils.timing.format_timing_summary`), the timing of its
    node tasks (:func:`node_task_timing_table`), the task stream
    (:func:`~astroviper.utils.resource_plots.plot_task_stream`) and, for a run
    with ``monitor_resources_seconds``, the cluster CPU, memory and I/O
    (:func:`~astroviper.utils.resource_plots.plot_cluster_resource_usage`);
    a note says what the run did not record.

    Parameters
    ----------
    document : Report
        The report.
    return_dict : dict
        The application's return dict (``"timing_distributed_application"``,
        ``"timing_node_tasks"`` and, when recorded, ``"timing_reduce_nodes"``).
    timing_phases : list of tuple
        The phase layout of ``"timing_distributed_application"``, the
        application's ``DISTRIBUTED_APPLICATION_TIMING_PHASES``.
    task_columns : dict, optional
        The node task columns of the task stream (``task_columns`` of
        :func:`~astroviper.utils.resource_plots.plot_task_stream`); ``None``
        for the imager's.
    task_name : str, optional
        What a node task does, for the captions (e.g. ``"frequency chunk"``).
    """
    import contextlib

    import pandas as pd

    from astroviper.utils.resource_plots import (
        plot_cluster_resource_usage,
        plot_task_stream,
    )
    from astroviper.utils.timing import format_timing_summary

    document.section("Processing")
    timing = return_dict.get("timing_distributed_application") or {}
    if timing:
        document.preformatted(
            format_timing_summary(
                timing,
                timing_phases,
                total_key="T_total",
                title="Distributed application steps (seconds)",
                total_label="TOTAL (wall time)",
            )
        )
    frame = return_dict.get("timing_node_tasks")
    if isinstance(frame, pd.DataFrame) and len(frame):
        document.table(
            node_task_timing_table(frame),
            caption=f"Node task timing (seconds) over the {len(frame)} tasks, one "
            f"per {task_name}.",
        )

    # The plot functions print their notes and summaries; the figures carry them.
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        figure = plot_task_stream(return_dict, task_columns=task_columns)
    if figure is None:
        document.note("Task stream: " + (printed.getvalue().strip() or "not recorded."))
    else:
        document.figure(
            figure,
            caption="Task stream: one lane per worker, every task as its load "
            "(blue), science (green) and write (orange) time, above it the "
            "number of running tasks and the steps of the application itself.",
        )

    with contextlib.redirect_stdout(io.StringIO()):
        axes = plot_cluster_resource_usage(return_dict)
    if axes is None:
        document.note(
            "Cluster CPU, memory and I/O were not recorded: run with "
            "monitor_resources_seconds (e.g. 0.5) to record them."
        )
        return
    for key, caption in (
        ("cpu", "Cluster CPU: busy cores summed over the running tasks."),
        ("memory", "Cluster memory: resident memory of the task-running workers."),
        ("io", "Cluster I/O: aggregate read and write rate (syscall level)."),
    ):
        if axes.get(key) is not None:
            document.figure(axes[key].figure, caption=caption)
    if axes.get("io") is None:
        document.note(
            "Cluster I/O was not recorded: the I/O counters of the monitor "
            "exist on Linux only."
        )
