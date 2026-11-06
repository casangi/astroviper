"""Tests of the ``ASTROVIPER_KERNELS`` grammar, registry, dispatch and trace.

The in-process tests drive ``astroviper._kernels._switch`` directly; whatever
the shims of this session look like, they wrap the reference entries
themselves. A fake ``hogbom=fast`` implementation (this module's
``clean_cube``) shows the grammar and the call-time switching with a second
implementation. Import-time behaviour runs in subprocesses.
"""

import importlib
import json
import logging
import os
import re
import socket
import sys
import textwrap
import threading
from importlib import import_module

import numpy as np
import pytest

from astroviper._kernels import _switch
from astroviper._kernels.__main__ import main as check_main
from astroviper._kernels._switch import ENV, RESERVED, TRACE_ENV, dispatch, impl, parse
from astroviper.processing_functions.imaging.deconvolvers.hogbom import _hogbom_ext
from tests.unit.kernels.conftest import ALL_REF, SHIMS

ALL_REFERENCE = {"dft": "ref", "hogbom": "ref", "asp": "ref"}


def clean_cube(*args, **kwargs):
    """The fake ``hogbom=fast`` entry."""
    return "fast"


@pytest.fixture
def fake_fast(monkeypatch):
    """Register this module as ``hogbom=fast`` for one test."""
    monkeypatch.setitem(_switch._REGISTRY["hogbom"], "fast", __name__)
    monkeypatch.setattr(_switch, "_resolved", {})
    parse.cache_clear()
    yield
    parse.cache_clear()


@pytest.fixture
def fake_module(monkeypatch, tmp_path):
    """``fake_module(name, source=None)`` registers module ``name`` as
    ``hogbom=fast``, written from ``source`` (``None``: no such module)."""
    names = []
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_switch, "_resolved", {})
    monkeypatch.setattr(_switch, "_logged", set())

    def register(name, source=None):
        if source is not None:
            (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
            importlib.invalidate_caches()
        monkeypatch.setitem(_switch._REGISTRY["hogbom"], "fast", name)
        parse.cache_clear()
        names.append(name)
        return name

    yield register
    for name in names:
        sys.modules.pop(name, None)
    parse.cache_clear()


def _hogbom_inputs():
    residual = np.zeros((1, 1, 1, 8, 8))
    residual[..., 4, 4] = 1.0
    return {
        "residual_cube": residual,
        "psf_cube": residual.copy(),
        "model_cube": np.zeros_like(residual),
        "max_iter_per_cycle_updated": np.full((1, 1, 1), 5, np.int32),
        "gain": 0.5,
        "threshold_per_cycle": np.zeros((1, 1, 1)),
    }


@pytest.mark.parametrize(
    "value",
    ["", "ref", " Ref ", "*=ref", "dft=ref, HOGBOM = ref,asp=ref", "ref,dft=ref"],
)
def test_reference_selections(value):
    assert parse(value) == ALL_REFERENCE


@pytest.mark.parametrize(
    "value, dft, hogbom",
    [
        ("fast", "fast", "fast"),
        ("*=FAST", "fast", "fast"),
        ("hogbom=fast", "ref", "fast"),
        ("fast,hogbom=ref", "fast", "ref"),
        ("hogbom=ref,fast", "fast", "fast"),
        ("ref,hogbom=fast,dft=ref", "ref", "fast"),
    ],
)
def test_defaults_overrides_and_later_tokens_win(fake_fast, value, dft, hogbom):
    # asp has fast as well, and no token names it: it follows the default, as dft
    assert parse(value) == {**ALL_REFERENCE, "dft": dft, "hogbom": hogbom, "asp": dft}


@pytest.mark.parametrize(
    "value, message",
    [
        ("dtf=ref", "unknown kernel 'dtf'; kernels: asp, dft, hogbom"),
        (
            "dft=fsat",
            "unknown implementation 'fsat'; implementations: fast, gpu, ref, ref1",
        ),
        (
            "asp=gpu",
            "'asp' has no implementation 'gpu'; its implementations: fast, ref",
        ),
        ("hogbom=fast", "'hogbom' has no implementation 'fast'"),
        ("gpu", "no kernel implements 'gpu' yet"),
        ("dft=ref:avx2", "variants (':...') are not supported"),
        ("ref:x", "variants (':...') are not supported"),
        *[(f"{k}=ref", f"kernel {k!r} is reserved") for k in RESERVED],
        *[
            (v, "expected '<impl>' or '<kernel>=<impl>'")
            for v in (
                "dft=",
                "=ref",
                "dft==ref",
                "dft=ref=ref",
                "ref,",
                ",",
                " ",
                "ref,,dft=ref",
            )
        ],
    ],
)
def test_invalid_selections_raise(value, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        parse(value)


def test_selection_is_read_only_and_cached():
    selection = parse("ref")
    with pytest.raises(TypeError):
        selection["dft"] = "fast"
    assert parse("ref") is selection


def test_registry_holds_the_reference_pairs():
    assert {k: set(v) for k, v in _switch._REGISTRY.items()} == {
        "dft": {"ref", "fast"},
        "hogbom": {"ref"},
        "asp": {"ref", "fast"},
    }
    for kernel, (shim, ext, entries) in SHIMS.items():
        assert _switch._ENTRIES[kernel] == entries
        for entry in entries:
            assert impl(kernel, entry, "ref") is getattr(
                import_module(f"{shim}.{ext}"), entry
            )


def test_impl_rejects_unhooked_entries_and_unregistered_pairs():
    with pytest.raises(ValueError, match="not a hooked entry"):
        impl("hogbom", "clean", "ref")
    with pytest.raises(ValueError, match="has no implementation 'gpu'"):
        impl("asp", "clean", "gpu")


def test_dispatch_wrapper_carries_the_shim_names(monkeypatch):
    monkeypatch.setenv(ENV, ALL_REF)
    for kernel, (shim, ext, entries) in SHIMS.items():
        for entry in entries:
            function = getattr(import_module(f"{shim}.{ext}"), entry)
            wrapper = dispatch(kernel, function)
            assert wrapper.__wrapped__ is function
            assert wrapper.__module__ == shim
            assert wrapper.__name__ == wrapper.__qualname__ == entry
            assert wrapper.__doc__ == function.__doc__


def test_dispatch_validates_at_install(monkeypatch):
    monkeypatch.setenv(ENV, "hogbom=fsat")
    with pytest.raises(ValueError, match="unknown implementation 'fsat'"):
        dispatch("hogbom", _hogbom_ext.clean_cube)
    monkeypatch.setenv(ENV, "ref")
    with pytest.raises(ValueError, match="'clean' is not a hooked entry"):
        dispatch("hogbom", _hogbom_ext.clean)


def test_ref_runs_the_reference_bitwise(monkeypatch):
    monkeypatch.setenv(ENV, "hogbom=ref")
    wrapper = dispatch("hogbom", _hogbom_ext.clean_cube)
    direct, switched = _hogbom_inputs(), _hogbom_inputs()
    expected = _hogbom_ext.clean_cube(**direct)
    result = wrapper(**switched)
    assert result.keys() == expected.keys()
    for key, value in expected.items():
        assert np.array_equal(result[key], value)
    for key in ("residual_cube", "model_cube"):
        assert direct[key].tobytes() == switched[key].tobytes()


def test_selection_is_read_at_each_call(monkeypatch, fake_fast):
    monkeypatch.setenv(ENV, "ref")
    wrapper = dispatch("hogbom", _hogbom_ext.clean_cube)
    assert isinstance(wrapper(**_hogbom_inputs()), dict)
    monkeypatch.setenv(ENV, "hogbom=fast")
    assert wrapper(**_hogbom_inputs()) == "fast"
    monkeypatch.delenv(ENV)
    assert isinstance(wrapper(**_hogbom_inputs()), dict)
    monkeypatch.setenv(ENV, "hogbom=fsat")
    with pytest.raises(ValueError, match="unknown implementation 'fsat'"):
        wrapper(**_hogbom_inputs())


def test_trace_lines(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "hogbom=ref")
    monkeypatch.setenv(TRACE_ENV, str(tmp_path))
    monkeypatch.setattr(_switch, "_traced", set())
    wrapper = dispatch("hogbom", _hogbom_ext.clean_cube)
    inputs = _hogbom_inputs()
    wrapper(inputs.pop("residual_cube"), **inputs)
    wrapper(**_hogbom_inputs())

    (path,) = tmp_path.iterdir()
    assert re.fullmatch(
        rf"kernels-{re.escape(socket.gethostname())}-\d+\.jsonl", path.name
    )
    first, second = map(json.loads, path.read_text().splitlines())
    assert {k: first[k] for k in ("kernel", "entry", "impl", "path", "error")} == {
        "kernel": "hogbom",
        "entry": "clean_cube",
        "impl": "ref",
        "path": "ref",
        "error": None,
    }
    assert (first["first_call"], second["first_call"]) == (True, False)
    assert first["t_start"] <= first["t_end"] <= second["t_start"]
    assert first["t_end"] == first["t_start"] + first["wall_s"]
    assert first["wall_s"] >= 0 and first["rusage_cpu_s"] >= 0
    assert first["cpu_scope"] == "process"
    assert (first["host"], first["pid"]) == (socket.gethostname(), os.getpid())
    assert first["thread"] == threading.get_ident()
    assert first["shapes"]["0"] == second["shapes"]["residual_cube"] == [1, 1, 1, 8, 8]
    assert first["shapes"]["max_iter_per_cycle_updated"] == [1, 1, 1]
    assert first["dtypes"]["max_iter_per_cycle_updated"] == "int32"
    assert "gain" not in first["shapes"]


def test_trace_records_the_reported_path(fake_module, monkeypatch, tmp_path):
    """A path an implementation reports names its own traced call only, not a
    call of another thread, an enclosing call or a call of another kernel."""
    fake_module(
        "_kernels_fake_paths",
        """
        from astroviper._kernels._switch import report_path

        def clean_cube(path=None, inner=None, fail=False, kernel="hogbom"):
            if path:
                report_path(kernel, path)
            if inner:
                inner(path="inner")
            if fail:
                raise RuntimeError(path)

        clean_cube_many_threads = clean_cube
        """,
    )
    monkeypatch.setenv(ENV, "hogbom=fast")
    monkeypatch.setenv(TRACE_ENV, str(tmp_path / "trace"))
    wrapper = dispatch("hogbom", _hogbom_ext.clean_cube)
    wrapper(path="S")
    impl("hogbom", "clean_cube", "fast")(path="untraced")
    worker = threading.Thread(target=lambda: _switch.report_path("hogbom", "thread"))
    worker.start()
    worker.join()
    wrapper()
    with pytest.raises(RuntimeError):
        wrapper(path="declined", fail=True)
    wrapper(path="outer", inner=wrapper)
    wrapper(path="S", kernel="dft")  # an untraced dft call's report inside it
    (path,) = (tmp_path / "trace").iterdir()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["path"], r["error"]) for r in records] == [
        ("S", None),
        ("fast", None),
        ("declined", "RuntimeError"),
        ("inner", None),
        ("outer", None),
        ("fast", None),
    ]


def test_trace_records_a_raising_call(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "hogbom=ref")
    monkeypatch.setenv(TRACE_ENV, str(tmp_path))
    wrapper = dispatch("hogbom", _hogbom_ext.clean_cube)
    with pytest.raises(TypeError):
        wrapper(residual_cube=np.zeros((2, 2)))
    (path,) = tmp_path.iterdir()
    (record,) = map(json.loads, path.read_text().splitlines())
    assert (record["impl"], record["path"], record["error"]) == (
        "ref",
        "ref",
        "TypeError",
    )


def test_unwritable_trace_never_changes_the_outcome(monkeypatch, tmp_path, caplog):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setenv(ENV, "hogbom=ref")
    monkeypatch.setenv(TRACE_ENV, str(blocker))
    monkeypatch.setattr(_switch, "_trace_failed", set())
    wrapper = dispatch("hogbom", _hogbom_ext.clean_cube)
    direct, switched = _hogbom_inputs(), _hogbom_inputs()
    expected = _hogbom_ext.clean_cube(**direct)
    with caplog.at_level(logging.WARNING, logger="astroviper.kernels"):
        result = wrapper(**switched)
        with pytest.raises(TypeError):  # the call's own exception, not OSError
            wrapper(residual_cube=np.zeros((2, 2)))
    assert result.keys() == expected.keys()
    for key, value in expected.items():
        assert np.asarray(result[key]).tobytes() == np.asarray(value).tobytes()
    for key in ("residual_cube", "model_cube"):
        assert direct[key].tobytes() == switched[key].tobytes()
    (warning,) = caplog.records
    assert "cannot write trace lines" in warning.getMessage()


def test_impl_imports_without_holding_the_lock(fake_module):
    """An implementation module may resolve the reference while it imports."""
    fake_module(
        "_kernels_fake_reentrant",
        """
        from astroviper._kernels._switch import impl

        reference = impl("hogbom", "clean_cube", "ref")
        clean_cube = clean_cube_many_threads = reference
        """,
    )
    found = []
    worker = threading.Thread(
        target=lambda: found.append(impl("hogbom", "clean_cube", "fast")), daemon=True
    )
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "impl() deadlocked on a re-entrant import"
    assert found == [_hogbom_ext.clean_cube]


def test_missing_module_raises_import_error(fake_module, monkeypatch, capsys):
    module = fake_module("_kernels_fake_not_built")
    message = f"hogbom=fast: module {module} is not available"
    with pytest.raises(ImportError, match=re.escape(message)):
        impl("hogbom", "clean_cube", "fast")
    monkeypatch.setenv(ENV, "hogbom=fast")
    wrapper = dispatch("hogbom", _hogbom_ext.clean_cube)
    with pytest.raises(ImportError, match=re.escape(message)):
        wrapper(**_hogbom_inputs())
    assert check_main(["--check"]) == 1
    assert f"ASTROVIPER_KERNELS check failed: {message}" in capsys.readouterr().err


def test_import_error_inside_a_module_is_not_relabelled(fake_module):
    fake_module("_kernels_fake_broken", "import _kernels_fake_missing_dependency\n")
    with pytest.raises(ModuleNotFoundError) as caught:
        impl("hogbom", "clean_cube", "fast")
    assert caught.value.name == "_kernels_fake_missing_dependency"
    assert "is not available" not in str(caught.value)


def test_info_logged_once_per_kernel_and_impl_with_isa(fake_module, caplog):
    module = fake_module(
        "_kernels_fake_isa",
        """
        ISA = "avx2"

        def clean_cube(*args, **kwargs):
            return "fast"

        clean_cube_many_threads = clean_cube
        """,
    )
    with caplog.at_level(logging.INFO, logger="astroviper.kernels"):
        for entry in ("clean_cube", "clean_cube_many_threads", "clean_cube"):
            impl("hogbom", entry, "fast")
    assert [r.getMessage() for r in caplog.records] == [
        f"kernel hogbom=fast from {module} (ISA avx2)"
    ]


def test_check_command_checks_the_trace_dir(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setenv(ENV, "hogbom=ref")
    monkeypatch.setenv(TRACE_ENV, str(blocker))
    assert check_main(["--check"]) == 1
    assert "ASTROVIPER_KERNELS check failed: " in capsys.readouterr().err
    monkeypatch.setenv(TRACE_ENV, str(tmp_path / "trace"))
    assert check_main(["--check"]) == 0
    assert list((tmp_path / "trace").iterdir()) == []
    monkeypatch.delenv(ENV)
    assert check_main(["--check"]) == 0
    assert "is set but ASTROVIPER_KERNELS is not" in capsys.readouterr().err


def test_typo_fails_at_shim_import(run_python):
    shim = SHIMS["hogbom"][0]
    done = run_python(
        "-c", f"import {shim}", check=False, ASTROVIPER_KERNELS="hogbom=fsat"
    )
    assert done.returncode != 0
    expected = (
        "ValueError: ASTROVIPER_KERNELS='hogbom=fsat', token 'hogbom=fsat': unknown"
    )
    assert expected in done.stderr


def test_check_command(run_python):
    out = run_python("-m", "astroviper._kernels", "--check", ASTROVIPER_KERNELS=ALL_REF)
    last = out.strip().splitlines()[-1]
    assert (
        last == f"ASTROVIPER_KERNELS={ALL_REF!r} (hooked): dft=ref, hogbom=ref, asp=ref"
    )
    done = run_python(
        "-m",
        "astroviper._kernels",
        "--check",
        check=False,
        ASTROVIPER_KERNELS="asp=gpu",
    )
    assert done.returncode == 1
    assert "ASTROVIPER_KERNELS check failed: " in done.stderr
    assert "'asp' has no implementation 'gpu'" in done.stderr


def test_hooked_entries_pickle_by_reference(run_python):
    names = [f"{shim}.{e}" for shim, _, entries in SHIMS.values() for e in entries]
    code = "\n".join(
        ["import pickle", *(f"import {shim}" for shim, _, _ in SHIMS.values())]
        + [
            f"assert {n}.__wrapped__ and pickle.loads(pickle.dumps({n})) is {n}"
            for n in names
        ]
    )
    run_python("-c", code, ASTROVIPER_KERNELS=ALL_REF)
