"""Tests of the kernels pytest plugin's verdict, on synthetic trace records."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import kernels_plugin
from tests.kernels_plugin import (
    DECLINED_BY_DESIGN,
    EXECUTED_PATHS,
    MANIFEST,
    KernelsPlugin,
    _records,
    summarize,
)

LISTED = "tests/unit/node_tasks/imaging/test_node_task_science.py"  # hogbom.clean_cube
OTHER = "tests/unit/processing_functions/imaging/test_other.py"
EFFECTIVE = {"dft": "ref", "hogbom": "fast", "asp": "ref"}


def _record(t, impl="fast", path=None, error=None, entry="clean_cube"):
    return {
        "kernel": "hogbom",
        "entry": entry,
        "impl": impl,
        "path": impl if path is None else path,
        "error": error,
        "t_start": t,
    }


INTERVALS = [(10.0, 20.0, LISTED), (30.0, 40.0, OTHER)]


def _verdict(*records, collected=(LISTED, OTHER)):
    return summarize(
        list(records), {"hogbom": "fast"}, EFFECTIVE, INTERVALS, set(collected)
    )


def test_the_manifest_lists_existing_files_and_hooked_entries():
    from astroviper._kernels._switch import _ENTRIES, _REGISTRY

    for path, kernels in MANIFEST.items():
        assert (Path(__file__).parents[3] / path).is_file(), path
        for kernel, entries in kernels.items():
            assert set(entries) <= set(_ENTRIES[kernel]), (path, kernel)
    for kernel, name in DECLINED_BY_DESIGN.keys() | EXECUTED_PATHS.keys():
        assert name in _REGISTRY[kernel]
    for (kernel, _), files in DECLINED_BY_DESIGN.items():
        assert all(kernel in MANIFEST[path] for path in files)


DFT_FILE = "tests/unit/processing_functions/imaging/test_direct_fourier_transform.py"
BY_DESIGN = (
    "tests/unit/processing_functions/imaging/gridders/"
    "test_direct_fourier_transform_cpp.py"
)


@pytest.mark.parametrize(
    "path, executed", [("S", True), ("declined", False), ("fast", False)]
)
def test_dft_fast_executes_on_path_s_only(path, executed):
    record = {"kernel": "dft", "entry": "dft_predict", "impl": "fast", "path": path}
    record |= {"error": None, "t_start": 15.0}
    selection = {"dft": "fast"}
    assert kernels_plugin._executed(record, selection) is executed
    _, missing, foreign = summarize(
        [record], selection, selection, [(10.0, 20.0, DFT_FILE)], {DFT_FILE}
    )
    assert (missing == [], foreign) == (executed, {})


@pytest.mark.parametrize(
    "path, executed", [("gpu", True), ("routed", False), ("declined", False)]
)
def test_hogbom_gpu_executes_on_path_gpu_only(path, executed):
    """A routed or declined hogbom=gpu call ran on the CPU."""
    record = _record(15.0, impl="gpu", path=path)
    selection = {"hogbom": "gpu"}
    assert kernels_plugin._executed(record, selection) is executed
    _, missing, _ = summarize([record], selection, selection, INTERVALS, {LISTED})
    assert (missing == []) is executed


def _dft(path, error=None):
    """A ``dft=fast`` record of the file running at t = 15."""
    record = {"kernel": "dft", "entry": "dft_predict", "impl": "fast", "path": path}
    return record | {"error": error, "t_start": 15.0}


@pytest.mark.parametrize(
    "records, missing",
    [
        ([], [BY_DESIGN, DFT_FILE]),
        ([_dft("declined", "RuntimeError")], [BY_DESIGN, DFT_FILE]),
        ([_dft("declined")], [DFT_FILE]),
        ([_dft("S")], [DFT_FILE]),
    ],
    ids=["no-call", "raised", "declined", "S"],
)
def test_files_declined_by_design_need_a_call_that_reached_fast(records, missing):
    """Such a file needs no executed call but one ``dft=fast`` call that did not
    raise, on any path (its calls go through the hook); under ``dft=ref`` it
    needs an executed call, like every listed file."""
    assert BY_DESIGN in DECLINED_BY_DESIGN["dft", "fast"]
    intervals = [(10.0, 20.0, BY_DESIGN), (30.0, 40.0, DFT_FILE)]
    collected = {BY_DESIGN, DFT_FILE}
    fast, ref = {"dft": "fast"}, {"dft": "ref"}
    verdict = summarize(records, fast, fast, intervals, collected)[1]
    assert verdict == [f"{p}: dft.dft_predict=fast" for p in missing]
    verdict = summarize([], ref, ref, intervals, collected)[1]
    assert verdict == [f"{p}: dft.dft_predict=ref" for p in (BY_DESIGN, DFT_FILE)]


def test_an_executed_call_in_the_listed_file_passes():
    counts, missing, foreign = _verdict(_record(15.0))
    assert (missing, foreign) == ([], {})
    assert counts == {(LISTED, "hogbom", "clean_cube", "fast", "fast", None): 1}


@pytest.mark.parametrize(
    "record",
    [
        _record(35.0),  # in another file
        _record(25.0),  # between tests
        _record(15.0, error="RuntimeError"),  # raised
        _record(15.0, path="ref"),  # declined to the reference
        _record(15.0, entry="clean_cube_many_threads"),  # another entry
    ],
    ids=["other-file", "between-tests", "raised", "declined", "other-entry"],
)
def test_calls_that_do_not_count(record):
    _, missing, foreign = _verdict(record)
    assert missing == [f"{LISTED}: hogbom.clean_cube=fast"]
    assert foreign == {}


def test_a_reference_record_does_not_satisfy_fast_and_is_foreign():
    _, missing, foreign = _verdict(_record(15.0, impl="ref"))
    assert missing == [f"{LISTED}: hogbom.clean_cube=fast"]
    assert foreign == {("hogbom", "ref"): 1}


def test_files_not_collected_or_not_listed_are_not_required():
    assert _verdict(collected=(OTHER,))[1] == []
    assert summarize([], {"dft": "ref"}, EFFECTIVE, INTERVALS, {LISTED})[1] == []


def test_records_skip_undecodable_and_old_lines(tmp_path):
    lines = [
        json.dumps(_record(5.0)),
        json.dumps(_record(15.0)),
        '{"kernel": "hogbom", "t_sta',
        "",
        "[]",
    ]
    (tmp_path / "kernels-host-1.jsonl").write_text("\n".join(lines))
    (tmp_path / "unrelated.jsonl").write_text("garbage\n")
    records, undecodable = _records(tmp_path, since=10.0)
    assert records == [_record(15.0)]
    assert undecodable == 3


def _session(exitstatus=pytest.ExitCode.OK):
    return SimpleNamespace(exitstatus=exitstatus)


def test_session_fails_on_a_missing_or_foreign_call(tmp_path):
    plugin = KernelsPlugin({"hogbom": "ref"}, str(tmp_path))
    plugin.effective = {"dft": "ref", "hogbom": "ref", "asp": "ref"}
    plugin.collected = {LISTED}
    t = plugin.started + 1
    plugin.intervals = [(t, t + 10, LISTED)]
    trace = tmp_path / "kernels-host-1.jsonl"

    trace.write_text(json.dumps(_record(t + 1, impl="ref")) + "\n")
    session = _session()
    plugin.pytest_sessionfinish(session)
    assert session.exitstatus == pytest.ExitCode.OK

    trace.write_text(json.dumps(_record(t + 1, impl="fast")) + "\n")
    session = _session()
    plugin.pytest_sessionfinish(session)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert plugin.foreign == {("hogbom", "fast"): 1}
    assert plugin.missing == [f"{LISTED}: hogbom.clean_cube=ref"]


@pytest.mark.parametrize("value", ["hogbom=fsat", "fft=ref", "hogbom=ref,dft=gpu"])
def test_collection_rejects_an_invalid_selection(monkeypatch, tmp_path, value):
    monkeypatch.setenv(kernels_plugin.ENV, value)
    plugin = KernelsPlugin(kernels_plugin._explicit_selection(value), str(tmp_path))
    with pytest.raises(pytest.UsageError, match="kernels plugin: "):
        plugin.pytest_collection_finish(SimpleNamespace())


@pytest.mark.parametrize("value", ["ref", "*=ref", "hogbom=ref,fast", " "])
def test_bare_defaults_are_refused(value):
    with pytest.raises(pytest.UsageError, match="explicit <kernel>=<impl>"):
        kernels_plugin._explicit_selection(value)


@pytest.mark.parametrize(
    "value, floor, expected",
    [
        ("hogbom=gpu", None, "0"),
        ("hogbom=gpu", "4096", "4096"),
        ("hogbom=fast", None, None),
    ],
)
def test_a_hogbom_gpu_session_runs_the_tests_planes_on_the_gpu(
    monkeypatch, tmp_path, value, floor, expected
):
    """Its pixel floor is 0 unless set: the tests' planes are all below the
    default floor, which would route every call to the CPU."""
    name = kernels_plugin.GPU_FLOOR_ENV
    monkeypatch.setenv(kernels_plugin.ENV, value)
    monkeypatch.setenv(kernels_plugin.TRACE_ENV, str(tmp_path))
    monkeypatch.setenv(name, "unset below")  # restored after the test
    if floor is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, floor)
    manager = SimpleNamespace(register=lambda plugin, plugin_name: None)
    kernels_plugin.pytest_configure(SimpleNamespace(pluginmanager=manager))
    assert os.environ.get(name) == expected
