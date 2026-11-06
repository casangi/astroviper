"""Tests of the ``ASTROVIPER_KERNELS`` grammar, registry, dispatch and trace.

The in-process tests drive ``astroviper._kernels._switch`` directly; whatever
the shims of this session look like, they wrap the reference entries
themselves. A fake ``hogbom=fast`` implementation (this module's
``clean_cube``) shows the grammar and the call-time switching with a second
implementation. Import-time behaviour runs in subprocesses.
"""

import json
import re
import socket
from importlib import import_module

import numpy as np
import pytest

from astroviper._kernels import _switch
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
    "value, hogbom",
    [
        ("fast", "fast"),
        ("*=FAST", "fast"),
        ("hogbom=fast", "fast"),
        ("fast,hogbom=ref", "ref"),
        ("hogbom=ref,fast", "fast"),
        ("ref,hogbom=fast,dft=ref", "fast"),
    ],
)
def test_defaults_overrides_and_later_tokens_win(fake_fast, value, hogbom):
    assert parse(value) == {**ALL_REFERENCE, "hogbom": hogbom}


@pytest.mark.parametrize(
    "value, message",
    [
        ("dtf=ref", "unknown kernel 'dtf'; kernels: asp, dft, hogbom"),
        (
            "dft=fsat",
            "unknown implementation 'fsat'; implementations: fast, gpu, ref, ref1",
        ),
        ("asp=gpu", "'asp' has no implementation 'gpu'; its implementations: ref"),
        ("hogbom=fast", "'hogbom' has no implementation 'fast'"),
        ("fast", "no kernel implements 'fast' yet"),
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
        "dft": {"ref"},
        "hogbom": {"ref"},
        "asp": {"ref"},
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
    assert {k: first[k] for k in ("kernel", "entry", "impl", "path")} == {
        "kernel": "hogbom",
        "entry": "clean_cube",
        "impl": "ref",
        "path": "ref",
    }
    assert (first["first_call"], second["first_call"]) == (True, False)
    assert first["t_start"] <= first["t_end"] <= second["t_start"]
    assert first["wall_s"] >= 0 and first["rusage_cpu_s"] >= 0
    assert first["shapes"]["0"] == second["shapes"]["residual_cube"] == [1, 1, 1, 8, 8]
    assert first["shapes"]["max_iter_per_cycle_updated"] == [1, 1, 1]
    assert first["dtypes"]["max_iter_per_cycle_updated"] == "int32"
    assert "gain" not in first["shapes"]


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
