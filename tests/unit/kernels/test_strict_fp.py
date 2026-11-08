"""The floating-point guard of the fast modules (``fast/common/strict_fp.hpp``).

The fast kernels need every float and double operation rounded once, as
written. The header refuses (``#error``) the macros by which the compiler
announces options that allow anything else, and every translation unit of a
fast module and every fast module's compile probe includes it, so that such
options skip the fast modules at configure time
(``cmake/AstroviperKernels.cmake``) instead of building inexact code. A tiny
unit that includes the header is compiled here with each such option,
expecting the header's error, and with options that keep the rounding,
expecting none (skipped when no C++ compiler is found). Clang announces some
of these options by no macro (see the header), whose cases skip with Clang:
there the fast modules are compiled with ``-fno-fast-math`` after the flags
they inherit, and a unit whose code each such option changes must compile to
the code built without it. Every fast unit and probe must include the header.
"""

import functools
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

FAST = Path(__file__).parents[3] / "src/astroviper/_kernels/fast"

#: The C++ compiler, if any.
CXX = (
    os.environ.get("CXX")
    or shutil.which("c++")
    or shutil.which("g++")
    or shutil.which("clang++")
)

#: Options that let the compiler round otherwise than as written, and the
#: macro by which the header refuses each.
UNSAFE = {
    "fast-math": (["-ffast-math"], "__FAST_MATH__"),
    "Ofast": (["-Ofast"], "__FAST_MATH__"),
    "unsafe-math-optimizations": (
        ["-funsafe-math-optimizations"],
        "__ASSOCIATIVE_MATH__",
    ),
    "associative-math": (
        ["-fassociative-math", "-fno-signed-zeros", "-fno-trapping-math"],
        "__ASSOCIATIVE_MATH__",
    ),
    "reciprocal-math": (["-freciprocal-math"], "__RECIPROCAL_MATH__"),
    "no-signed-zeros": (["-fno-signed-zeros"], "__NO_SIGNED_ZEROS__"),
    "finite-math-only": (["-ffinite-math-only"], "__FINITE_MATH_ONLY__"),
    "fast-math-not-finite": (
        ["-ffast-math", "-fno-finite-math-only"],
        "__ASSOCIATIVE_MATH__",
    ),
    "x87": (["-mfpmath=387"], "FLT_EVAL_METHOD"),
}
#: The macros of ``UNSAFE`` that Clang does not define.
NOT_IN_CLANG = {"__ASSOCIATIVE_MATH__", "__RECIPROCAL_MATH__", "__NO_SIGNED_ZEROS__"}

#: The options with which the fast modules are compiled after the flags they
#: inherit, with Clang (cmake/AstroviperKernels.cmake).
CLANG_FP_OPTIONS = ["-fno-fast-math"]
#: Options that Clang announces by no macro: those of ``UNSAFE`` whose
#: refusal it cannot make, and options that GCC does not have.
CLANG_UNANNOUNCED = {
    "unsafe-math-optimizations": ["-funsafe-math-optimizations"],
    "associative-math": [
        "-fassociative-math",
        "-fno-signed-zeros",
        "-fno-trapping-math",
    ],
    "reciprocal-math": ["-freciprocal-math"],
    "no-signed-zeros": ["-fno-signed-zeros"],
    "fast-math-not-finite": ["-ffast-math", "-fno-finite-math-only"],
    "approx-func": ["-fapprox-func"],
    "no-honor-nans": ["-fno-honor-nans"],
    "no-honor-infinities": ["-fno-honor-infinities"],
    "fp-model-fast": ["-ffp-model=fast"],
}
#: Functions whose code such options change: reassociation folds (a + b) - b
#: to a, reciprocals turn a / 3 into a product, ignored signed zeros fold
#: a + 0 to a, ignored NaNs and infinities fold the tests for them, and
#: approximate functions turn pow(a, -0.5) into 1 / sqrt(a).
CHANGED = """
double reassociated(double a, double b) { return (a + b) - b; }
double reciprocal(double a) { return a / 3.0; }
double signed_zero(double a) { return a + 0.0; }
bool not_a_number(double a) { return a != a || __builtin_isnan(a); }
bool infinite(double a) { return a == __builtin_inf() || __builtin_isinf(a); }
double approximate(double a) { return __builtin_pow(a, -0.5); }
"""

#: Options that keep every operation rounded as written: none, the modules'
#: Release flags, options that only drop errno or trap guarantees, and
#: -ffast-math turned off again.
SAFE = {
    "none": [],
    "release": ["-O3", "-DNDEBUG", "-ffp-contract=off"],
    "no-math-errno": ["-fno-math-errno"],
    "no-trapping-math": ["-fno-trapping-math"],
    "fast-math-undone": ["-ffast-math", "-fno-fast-math"],
}

needs_compiler = pytest.mark.skipif(CXX is None, reason="no C++ compiler")


def _compile(flags, tmp_path):
    """Compile a unit that includes the header, with the build's standard."""
    unit = tmp_path / "unit.cpp"
    unit.write_text('#include "strict_fp.hpp"\nint main() { return 0; }\n')
    command = [CXX, "-std=c++23", *flags, f"-I{FAST / 'common'}", "-fsyntax-only"]
    return subprocess.run([*command, str(unit)], capture_output=True, text=True)


@functools.cache
def _clang():
    macros = subprocess.run(
        [CXX, "-dM", "-E", "-x", "c++", os.devnull],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return "__clang__" in macros.split()


@needs_compiler
@pytest.mark.parametrize("name", list(UNSAFE))
def test_options_that_round_otherwise_are_refused(name, tmp_path):
    flags, macro = UNSAFE[name]
    if macro == "FLT_EVAL_METHOD":
        if platform.machine().lower() not in ("x86_64", "amd64", "i386", "i686"):
            pytest.skip("-mfpmath=387 is an x86 option")
        if _clang():
            pytest.skip("Clang refuses -mfpmath=387 itself where SSE is on")
    if _clang() and macro in NOT_IN_CLANG:
        pytest.skip(f"Clang defines no {macro}: strict_fp.hpp cannot see {flags}")
    done = _compile(flags, tmp_path)
    assert done.returncode != 0
    assert f"AstroVIPER fast kernels: {macro} " in done.stderr, done.stderr


@needs_compiler
@pytest.mark.parametrize("name", list(SAFE))
def test_options_that_keep_the_rounding_pass(name, tmp_path):
    done = _compile(SAFE[name], tmp_path)
    assert done.returncode == 0, done.stderr


def _code(flags, tmp_path):
    """Assembly of ``CHANGED`` built with ``flags``, its math functions
    without errno so that they may be approximated."""
    unit = tmp_path / "changed.cpp"
    unit.write_text(CHANGED)
    command = [CXX, "-std=c++23", "-O2", "-fno-math-errno", *flags, "-ffp-contract=off"]
    command += ["-S", "-o", "-", str(unit)]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


@needs_compiler
@pytest.mark.parametrize("name", list(CLANG_UNANNOUNCED))
def test_clang_options_it_does_not_announce_are_undone(name, tmp_path):
    """An option that Clang announces by no macro, followed by the options
    the fast modules get after the flags they inherit, gives the code built
    without it."""
    if not _clang():
        pytest.skip(
            "Clang only: GCC announces these options, which skips the fast modules"
        )
    flags = CLANG_UNANNOUNCED[name]
    if _code(flags, tmp_path) == _code([], tmp_path):
        pytest.skip(f"{flags} change nothing in the code of CHANGED with this Clang")
    assert _code([*flags, *CLANG_FP_OPTIONS], tmp_path) == _code(
        CLANG_FP_OPTIONS, tmp_path
    )


def test_every_fast_unit_and_probe_includes_the_header():
    """Every translation unit of the fast modules and every compile probe."""
    units = sorted(FAST.rglob("*.cpp"))
    assert any(unit.name == "probe.cpp" for unit in units)
    assert [
        unit.relative_to(FAST).as_posix()
        for unit in units
        if '#include "strict_fp.hpp"' not in unit.read_text()
    ] == []
