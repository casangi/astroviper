"""Pytest plugin for test sessions that run with ``ASTROVIPER_KERNELS`` set.

Usage, from the repository root::

    ASTROVIPER_KERNELS=hogbom=ref python -m pytest -p tests.kernels_plugin <paths>

A switched session passes only if the selected implementations really ran,
not just the reference behind them. The plugin:

* requires explicit ``<kernel>=<impl>`` tokens (a bare ``<impl>`` default is
  refused, so the session states exactly which kernels it is about) and
  validates them with the switch's own grammar after collection (so that the
  session imports its modules in the same order as an unswitched one);
* sets ``ASTROVIPER_KERNELS_TRACE`` to a fresh directory unless it is set, so
  that every hooked call, including those in Dask workers started by the
  tests, writes a trace line, and for ``hogbom=gpu``
  ``ASTROVIPER_KERNELS_GPU_MIN_PIXELS`` to 0 unless it is set, so that the
  GPU runs the tests' planes, which are all below its default floor;
* attributes each trace record, by its start time, to the test file that was
  running (setup, call or teardown), and prints the calls per file, kernel,
  entry, impl, path and error;
* fails the session when a collected file of ``MANIFEST`` made no call that
  executed the selected implementation of an entry it lists for a selected
  kernel (a call executes it on the paths of ``EXECUTED_PATHS``; a file that
  declines it by design, ``DECLINED_BY_DESIGN``, needs one call that reached
  it without raising, on any path), or when any record ran an implementation
  other than the session's selection (another selection leaked into the
  trace);
* provides the ``kernel_trace`` fixture for tests that assert the path taken
  by their own calls.
"""

import bisect
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import pytest

ENV = "ASTROVIPER_KERNELS"
TRACE_ENV = "ASTROVIPER_KERNELS_TRACE"
GPU_FLOOR_ENV = "ASTROVIPER_KERNELS_GPU_MIN_PIXELS"

_DA = "tests/unit/distributed_applications"
_NT = "tests/unit/node_tasks"
_PF = "tests/unit/processing_functions"
_DFT = {"dft": ("dft_predict",)}
_ASP_CUBE = {"asp": ("clean_cube",)}
_HOGBOM_CUBE = {"hogbom": ("clean_cube",)}
_HOGBOM_BOTH = {"hogbom": ("clean_cube", "clean_cube_many_threads")}

#: ``{test file: {kernel: entries}}``: the files whose calls (in process or in
#: Dask workers) execute these hooked entries, measured on the sharded
#: ``tests/unit`` run with every kernel switched to ``ref`` (and the component
#: file that the DFT's verification switches). A switched session
#: fails when a collected file listed here executes none of the selected
#: implementation of one of its entries for an explicitly selected kernel (or,
#: for the files of ``DECLINED_BY_DESIGN``, reaches none). Other files are
#: counted and printed, not required.
MANIFEST = {
    f"{_DA}/imaging/test_continuum.py": _HOGBOM_CUBE,
    f"{_DA}/imaging/test_continuum_cube_agreement.py": _ASP_CUBE | _HOGBOM_CUBE,
    f"{_DA}/imaging/test_continuum_dft.py": _DFT | _HOGBOM_CUBE,
    f"{_DA}/imaging/test_continuum_gridding_kernel.py": _HOGBOM_CUBE,
    f"{_DA}/imaging/test_continuum_polarization.py": _HOGBOM_CUBE,
    f"{_DA}/imaging/test_image_cube_cycle_control_record.py": _HOGBOM_CUBE,
    f"{_DA}/imaging/test_image_cube_single_field.py": _HOGBOM_CUBE,
    f"{_DA}/imaging/test_image_cube_single_field_delphi.py": _HOGBOM_CUBE,
    f"{_DA}/imaging/test_image_cube_single_field_dft.py": _DFT | _HOGBOM_CUBE,
    f"{_DA}/imaging/test_image_cube_single_field_report.py": _HOGBOM_CUBE,
    f"{_DA}/simulation/test_simulate_processing_set.py": _DFT,
    f"{_NT}/imaging/test_continuum.py": _ASP_CUBE,
    f"{_NT}/imaging/test_node_task_science.py": _HOGBOM_CUBE,
    f"{_PF}/imaging/gridders/test_direct_fourier_transform_cpp.py": _DFT,
    f"{_PF}/imaging/test_aspclean.py": {"asp": ("clean", "clean_cube")},
    f"{_PF}/imaging/test_continuum_dft.py": _DFT,
    f"{_PF}/imaging/test_continuum_variants.py": _ASP_CUBE | _HOGBOM_CUBE,
    f"{_PF}/imaging/test_deconvolution.py": _ASP_CUBE | _HOGBOM_BOTH,
    f"{_PF}/imaging/test_direct_fourier_transform.py": _DFT,
    f"{_PF}/imaging/test_exact_psf_subtraction.py": _HOGBOM_BOTH,
    f"{_PF}/imaging/test_hogbom_clean.py": _HOGBOM_BOTH,
    f"{_PF}/imaging/test_hogbom_many_threads.py": _HOGBOM_BOTH,
    f"{_PF}/imaging/test_hogbom_stop_tests.py": _HOGBOM_BOTH,
    f"{_PF}/imaging/test_iteration_history.py": _ASP_CUBE | _HOGBOM_BOTH,
    f"{_PF}/imaging/test_max_cycles_auto.py": _ASP_CUBE | _HOGBOM_CUBE,
    f"{_PF}/simulation/test_sky_image_visibilities.py": _DFT,
    "tests/component/test_single_channel_imaging_consistency.py": _DFT | _HOGBOM_CUBE,
}

_W_TERM = "its calls have w != 0, declined until path B (WP6)"

#: ``{(kernel, impl): {test file: reason}}``: files of ``MANIFEST`` whose calls
#: all decline that implementation by design (measured under ``dft=fast``):
#: their calls are parity evidence, so instead of an executed call they need
#: one call of the implementation that did not raise (on any path, declined
#: included), which shows that their calls go through the hook. The others
#: must execute it.
DECLINED_BY_DESIGN = {
    ("dft", "fast"): {
        f"{_DA}/imaging/test_continuum_dft.py": _W_TERM + " (simulated with w)",
        f"{_DA}/imaging/test_image_cube_single_field_dft.py": _W_TERM
        + " (simulated with w)",
        f"{_DA}/simulation/test_simulate_processing_set.py": _W_TERM
        + " (the simulator's default w_term=True)",
        f"{_PF}/imaging/gridders/test_direct_fourier_transform_cpp.py": _W_TERM
        + " (w_term='large')",
    },
}


def _explicit_selection(value):
    """``{kernel: impl}`` of the tokens of ``value``; refuses bare defaults."""
    selection = {}
    for raw in value.split(","):
        kernel, _, name = "".join(raw.split()).lower().rpartition("=")
        if kernel in ("", "*"):
            raise pytest.UsageError(
                f"{ENV}={value!r}: the kernels plugin needs explicit "
                "<kernel>=<impl> tokens, not a default"
            )
        selection[kernel] = name
    return selection


def _records(trace_dir, since):
    """``(records, undecodable)``: the trace records of every process that
    started at or after ``since``, and the number of lines that are not a
    record (such as a line cut short by a killed process)."""
    records, undecodable = [], 0
    for path in sorted(Path(trace_dir).glob("kernels-*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    recent = record["t_start"] >= since
                except (ValueError, KeyError, TypeError):
                    undecodable += 1
                    continue
                if recent:
                    records.append(record)
    return records, undecodable


#: The paths that execute an implementation which reports its path per call
#: (``astroviper._kernels._switch.report_path``); any other implementation
#: executes on the path named after it. A ``dft=fast`` call that ``declined``
#: ran the reference behind the fast entry's checks: parity evidence, not an
#: execution of the fast kernel (path B joins ``"S"`` here when it lands). A
#: ``hogbom=gpu`` call that was ``routed`` or ``declined`` ran on the CPU.
EXECUTED_PATHS = {("dft", "fast"): ("S",), ("hogbom", "gpu"): ("gpu",)}


def _reached(record, selection):
    """Whether a traced call of the implementation that ``selection``
    (``{kernel: impl}``) selects for its kernel returned without raising, on
    any of its paths."""
    return record.get("error") is None and record["impl"] == selection.get(
        record["kernel"]
    )


def _executed(record, selection):
    """Whether a traced call ran, without raising, the implementation that
    ``selection`` (``{kernel: impl}``) selects for its kernel."""
    name = selection.get(record["kernel"])
    paths = EXECUTED_PATHS.get((record["kernel"], name), (name,))
    return _reached(record, selection) and record["path"] in paths


def _attribute(records, intervals):
    """Test file running at the start of each record (``None`` between tests).

    ``intervals`` are the ``(start, end, file)`` of the tests in run order.
    """
    starts = [start for start, _, _ in intervals]
    files = []
    for record in records:
        i = bisect.bisect_right(starts, record["t_start"]) - 1
        inside = i >= 0 and record["t_start"] <= intervals[i][1]
        files.append(intervals[i][2] if inside else None)
    return files


def summarize(records, explicit, effective, intervals, collected):
    """The session's verdict on its trace records.

    Parameters
    ----------
    records : list of dict
        Trace records of the session.
    explicit, effective : dict
        ``{kernel: impl}`` the session names, and the full selection it makes.
    intervals : list of tuple
        ``(start, end, file)`` of each test run, in run order.
    collected : set of str
        Collected test files, relative to the root directory.

    Returns
    -------
    counts : Counter
        Calls per ``(file, kernel, entry, impl, path, error)``.
    missing : list of str
        ``"<file>: <kernel>.<entry>=<impl>"`` for each manifest entry of a
        collected file, for an explicitly selected kernel, that no call
        executed (that no call reached, for a file of
        ``DECLINED_BY_DESIGN``).
    foreign : Counter
        Calls per ``(kernel, impl)`` of an implementation the session does
        not select.
    """
    files = _attribute(records, intervals)
    counts = Counter(
        (f, r["kernel"], r["entry"], r["impl"], r["path"], r.get("error"))
        for f, r in zip(files, records, strict=True)
    )
    executed, reached = Counter(), Counter()
    for f, r in zip(files, records, strict=True):
        executed[f, r["kernel"], r["entry"]] += _executed(r, effective)
        reached[f, r["kernel"], r["entry"]] += _reached(r, effective)

    def satisfied(path, kernel, entry):
        by_design = path in DECLINED_BY_DESIGN.get((kernel, explicit[kernel]), ())
        return (reached if by_design else executed)[path, kernel, entry] > 0

    missing = [
        f"{path}: {kernel}.{entry}={explicit[kernel]}"
        for path in sorted(collected & MANIFEST.keys())
        for kernel, entries in MANIFEST[path].items()
        if kernel in explicit
        for entry in entries
        if not satisfied(path, kernel, entry)
    ]
    foreign = Counter(
        (r["kernel"], r["impl"])
        for r in records
        if r["impl"] != effective.get(r["kernel"])
    )
    return counts, missing, foreign


class KernelsPlugin:
    def __init__(self, selection, trace_dir):
        self.selection = selection
        self.effective = None
        self.trace_dir = trace_dir
        self.started = time.time()
        self.root = None
        self.collected = set()
        self.intervals = []
        self.counts = Counter()
        self.missing = []
        self.foreign = Counter()
        self.undecodable = 0

    def _relative(self, path):
        try:
            return Path(path).relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def pytest_collection_finish(self, session):
        from astroviper._kernels._switch import parse

        try:
            self.effective = dict(parse(os.environ[ENV]))
        except ValueError as e:
            raise pytest.UsageError(f"kernels plugin: {e}") from None
        self.root = session.config.rootpath
        self.collected = {self._relative(item.path) for item in session.items}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        start = time.time()
        yield
        self.intervals.append((start, time.time(), self._relative(item.path)))

    def pytest_sessionfinish(self, session):
        if self.effective is None:  # the selection is invalid
            return
        records, self.undecodable = _records(self.trace_dir, self.started)
        self.counts, self.missing, self.foreign = summarize(
            records, self.selection, self.effective, self.intervals, self.collected
        )
        failed = self.missing or self.foreign
        if failed and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter):
        write = terminalreporter.write_line
        terminalreporter.section("ASTROVIPER_KERNELS")
        write(f"{ENV}={os.environ[ENV]!r}, traces in {self.trace_dir}")
        for key, n in sorted(self.counts.items(), key=lambda kv: str(kv[0])):
            path, kernel, entry, name, taken, error = key
            raised = f" error={error}" if error else ""
            write(
                f"  {path or '(between tests)'}: {kernel}.{entry} "
                f"impl={name} path={taken}{raised}: {n} calls"
            )
        if self.undecodable:
            write(f"  {self.undecodable} undecodable trace lines skipped", yellow=True)
        for (kernel, name), n in sorted(self.foreign.items()):
            write(
                f"FAILED: {n} calls ran {kernel}={name}, which this session "
                "does not select",
                red=True,
            )
        for item in self.missing:
            write(f"FAILED: no call executed {item}", red=True)


def pytest_configure(config):
    value = os.environ.get(ENV)
    if not value:
        raise pytest.UsageError(f"the kernels plugin needs {ENV} to be set")
    selection = _explicit_selection(value)
    trace_dir = os.environ.get(TRACE_ENV) or tempfile.mkdtemp(prefix="kernels-trace-")
    os.environ[TRACE_ENV] = trace_dir
    if selection.get("hogbom") == "gpu":
        os.environ.setdefault(GPU_FLOOR_ENV, "0")
    config.pluginmanager.register(
        KernelsPlugin(selection, trace_dir), "kernels_plugin_state"
    )


@pytest.fixture
def kernel_trace():
    """``kernel_trace()`` returns the trace records written since the test began."""
    since = time.time()
    return lambda: _records(os.environ[TRACE_ENV], since)[0]
