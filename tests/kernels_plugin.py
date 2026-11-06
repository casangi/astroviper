"""Pytest plugin for test sessions that run with ``ASTROVIPER_KERNELS`` set.

Usage, from the repository root::

    ASTROVIPER_KERNELS=hogbom=ref python -m pytest -p tests.kernels_plugin <paths>

A switched session passes only if the selected implementations really ran,
not just the reference behind them. The plugin:

* requires explicit ``<kernel>=<impl>`` tokens (a bare ``<impl>`` default is
  refused, so the session states exactly which kernels it is about);
* sets ``ASTROVIPER_KERNELS_TRACE`` to a fresh directory unless it is set, so
  that every hooked call, including those in Dask workers started by the
  tests, writes a trace line;
* at the end counts, over every trace file, the calls that executed the
  selected implementation, prints them per kernel, entry and path, and fails
  the session when a selected kernel that the collected test modules name
  (see ``USES``) has none;
* provides the ``kernel_trace`` fixture for tests that assert the path taken
  by their own calls.
"""

import json
import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path

import pytest

ENV = "ASTROVIPER_KERNELS"
TRACE_ENV = "ASTROVIPER_KERNELS_TRACE"

#: A collected test module uses a kernel when its source matches the pattern.
USES = {
    "dft": re.compile(r"\bdft_predict\b"),
    "hogbom": re.compile(r"\bhogbom\b"),
    "asp": re.compile(r"\baspclean\b"),
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
    """Trace records of every process written at or after ``since``."""
    records = []
    for path in sorted(Path(trace_dir).glob("kernels-*.jsonl")):
        with open(path, encoding="utf-8") as f:
            records += [r for r in map(json.loads, f) if r["t_start"] >= since]
    return records


def _executed(record):
    """Whether a traced call ran its selected implementation."""
    return record["path"] == record["impl"]


class KernelsPlugin:
    def __init__(self, selection, trace_dir):
        self.selection = selection
        self.trace_dir = trace_dir
        self.started = time.time()
        self.used = set()
        self.counts = Counter()
        self.missing = []

    def pytest_collection_finish(self, session):
        for path in {item.path for item in session.items}:
            source = Path(path).read_text(encoding="utf-8")
            self.used |= {k for k, p in USES.items() if p.search(source)}

    def pytest_sessionfinish(self, session):
        records = _records(self.trace_dir, self.started)
        self.counts = Counter(
            (r["kernel"], r["entry"], r["impl"], r["path"]) for r in records
        )
        executed = Counter(r["kernel"] for r in records if _executed(r))
        self.missing = [
            f"{kernel}={name}"
            for kernel, name in self.selection.items()
            if kernel in self.used and executed[kernel] == 0
        ]
        if self.missing and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter):
        write = terminalreporter.write_line
        terminalreporter.section("ASTROVIPER_KERNELS")
        write(f"{ENV}={os.environ[ENV]!r}, traces in {self.trace_dir}")
        for (kernel, entry, name, path), n in sorted(self.counts.items()):
            write(f"  {kernel}.{entry} impl={name} path={path}: {n} calls")
        for pair in self.missing:
            write(
                f"FAILED: {pair} is selected and used by the collected tests, "
                "but no call executed it",
                red=True,
            )


def pytest_configure(config):
    value = os.environ.get(ENV)
    if not value:
        raise pytest.UsageError(f"the kernels plugin needs {ENV} to be set")
    selection = _explicit_selection(value)
    trace_dir = os.environ.get(TRACE_ENV) or tempfile.mkdtemp(prefix="kernels-trace-")
    os.environ[TRACE_ENV] = trace_dir
    config.pluginmanager.register(
        KernelsPlugin(selection, trace_dir), "kernels_plugin_state"
    )


@pytest.fixture
def kernel_trace():
    """``kernel_trace()`` returns the trace records written since the test began."""
    since = time.time()
    return lambda: _records(os.environ[TRACE_ENV], since)
