"""Tests of the kernels pytest plugin's verdict, on synthetic trace records."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import kernels_plugin
from tests.kernels_plugin import MANIFEST, KernelsPlugin, _records, summarize

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
    from astroviper._kernels._switch import _ENTRIES

    for path, kernels in MANIFEST.items():
        assert (Path(__file__).parents[3] / path).is_file(), path
        for kernel, entries in kernels.items():
            assert set(entries) <= set(_ENTRIES[kernel]), (path, kernel)


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
