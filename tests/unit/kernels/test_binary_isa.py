"""Instruction-set hygiene of the installed C++ extension modules.

pybind11 strips Release modules (``nm`` reports no symbols), so these rules
apply to whole modules, read from the disassembly of the installed files:

* no fused multiply-add in any module: the build passes ``-ffp-contract=off``
  so that results are bit-identical across compilers and platforms;
* no 256- or 512-bit vector register in a reference module: the reference
  extensions are built for the baseline x86-64 instruction set.

The patterns are checked on canned lines of both objdump syntaxes, each module
must disassemble to real code, and the list of reference modules must equal
the installed extensions outside ``_kernels``, so a new reference extension
cannot escape the rules. Skipped off x86-64 and when ``objdump`` (binutils) is
not installed.

A fast module built for the baseline instruction set only
(``BASELINE_FAST_MODULES``) follows the rules of the reference modules. The two
lists of fast modules must be the modules that the tree declares
(``_kernels/fast/*/CMakeLists.txt``), split by whether they have AVX2 units,
so a fast module cannot escape the rules either.

The fast modules (``FAST_MODULES``) have no fused multiply-add either. Their
AVX2 code lives in ``*_avx2.cpp`` units built without LTO, which must define
no weak or unique symbol (``nm`` types W, V, u): the linker may otherwise keep
an AVX2 unit's copy of an inline or template function for every caller,
generic code included. Their object files are read from the build directory,
``ASTROVIPER_BUILD_DIR`` or else ``build/*`` of the repository (the package's
default), only where the module linked there is the installed one, byte for
byte; the test is skipped when there is none. Where a C++ compiler is on the
path, each AVX2 unit is also compiled at every optimisation level a build
type gives it (Debug: -Og, see cmake/AstroviperKernels.cmake) and checked the
same way.

A fast module that is not installed (a build with
``ASTROVIPER_BUILD_FAST_KERNELS=OFF``, or one whose configure-time checks
skipped it) skips its tests, unless ``ASTROVIPER_EXPECT_FAST_KERNELS`` is set (not empty
or ``0``), as a Linux x86-64 CI job should set it: then it fails them, so that
a module silently lost from such a build is caught. A reference module must be
installed.
"""

import importlib.machinery
import importlib.util
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import pytest

#: The reference extension modules, relative to the ``astroviper`` package.
REFERENCE_MODULES = [
    "processing_functions/image_analysis/psf_gaussian_fit_cpp/_psf_gaussian_fit_ext",
    "processing_functions/imaging/deconvolvers/aspclean/_aspclean_ext",
    "processing_functions/imaging/deconvolvers/hogbom/_hogbom_ext",
    "processing_functions/imaging/deconvolvers/mtmfs/_mtmfs_ext",
    "processing_functions/imaging/gridders/anti_alias_grid_cpp/_anti_alias_grid_ext",
    "processing_functions/imaging/gridders/direct_fourier_transform_cpp/"
    "_direct_fourier_transform_ext",
    "processing_functions/imaging/imaging_weighting/grid_imaging_weights_cpp/"
    "_grid_imaging_weights_ext",
    "processing_functions/simulation/visibility_kernel_cpp/_visibility_kernel_ext",
]

#: The fast-kernel modules, relative to the ``astroviper`` package.
FAST_MODULES = [
    "_kernels/fast/dft/_dft_fast_ext",
    "_kernels/fast/hogbom/_hogbom_fast_ext",
]
#: The fast-kernel modules with no AVX2 unit (no AVX2 code at all).
BASELINE_FAST_MODULES = ["_kernels/fast/aspclean/_aspclean_fast_ext"]

#: FMA3 (vfmadd231pd, vfnmsub132ss, vfmaddsub213ps, ...) and FMA4 mnemonics.
FMA = re.compile(r"\bvfn?m(?:add|sub|addsub|subadd)\w*")
#: Register names wider than xmm, in AT&T (``%ymm0``) or Intel (``ymm0``) syntax.
WIDE_REGISTER = re.compile(r"(?<![\w%])%?[yz]mm\d+\b")
#: SSE register names, which every x86-64 module with floating point uses.
XMM_REGISTER = re.compile(r"(?<![\w%])%?xmm\d+\b")
#: One disassembled instruction: ``  <address>:<tab><instruction>``.
INSTRUCTION = re.compile(r"^\s*[0-9a-f]+:\t", re.MULTILINE)

OBJDUMP = shutil.which("objdump")
NM = shutil.which("nm")

pytestmark = [
    pytest.mark.skipif(
        platform.machine().lower() not in ("x86_64", "amd64"),
        reason="x86-64 instruction rules",
    ),
    pytest.mark.skipif(OBJDUMP is None, reason="objdump (binutils) is not installed"),
]


def _package():
    return Path(importlib.util.find_spec("astroviper").origin).parent


EXPECT = "ASTROVIPER_EXPECT_FAST_KERNELS"
#: Whether every fast module must be installed (see the module docstring).
EXPECT_FAST_KERNELS = os.environ.get(EXPECT, "") not in ("", "0")


def _module_file(relative):
    """Installed file of an extension module, found without importing it."""
    package = _package()
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = package / (relative + suffix)
        if path.is_file():
            return path
    message = f"extension module astroviper/{relative} is not installed"
    if relative.startswith("_kernels/") and not EXPECT_FAST_KERNELS:
        pytest.skip(f"{message} (fast kernels not built)")
    pytest.fail(message)


def _assert_baseline_code(relative):
    """No fused multiply-add and no register wider than xmm in the module."""
    text = _instructions(_module_file(relative))
    assert len(INSTRUCTION.findall(text)) > 1000, "no code was disassembled"
    assert XMM_REGISTER.search(text), "the register pattern matches nothing"
    assert sorted(set(FMA.findall(text))) == []
    assert sorted(set(WIDE_REGISTER.findall(text))) == []


def _instructions(path):
    """Disassembled instruction text of the executable sections of ``path``."""
    return subprocess.run(
        [OBJDUMP, "--disassemble", "--no-show-raw-insn", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.mark.parametrize(
    "line, fma, wide",
    [
        ("  1a2b:\tvfmadd231pd %ymm2,%ymm1,%ymm0", True, True),
        ("  1a2b:\tvfmadd231pd ymm0,ymm1,ymm2", True, True),
        ("  1a2b:\tvfnmsub132ss %xmm2,%xmm1,%xmm0", True, False),
        ("  1a2b:\tvaddpd zmm0,zmm1,zmm2", False, True),
        ("  1a2b:\tmovq   xmm0,rax", False, False),
        ("  1a2b:\tmovapd %xmm1,%xmm0", False, False),
        ("  1a2b:\tcall   1c0 <mymm1@plt>", False, False),
    ],
)
def test_patterns(line, fma, wide):
    """The rules match in both objdump syntaxes, and only register names."""
    assert bool(FMA.search(line)) is fma
    assert bool(WIDE_REGISTER.search(line)) is wide
    assert INSTRUCTION.search(line)


def test_reference_module_list_is_complete():
    """Every installed extension outside ``_kernels`` is a listed reference module."""
    package = _package()
    suffixes = sorted(importlib.machinery.EXTENSION_SUFFIXES, key=len, reverse=True)
    found = set()
    for path in package.rglob("*"):
        relative = path.relative_to(package).as_posix()
        suffix = next((s for s in suffixes if relative.endswith(s)), None)
        if suffix and path.is_file() and not relative.startswith("_kernels/"):
            found.add(relative.removesuffix(suffix))
    assert sorted(found) == sorted(REFERENCE_MODULES)


def _declared_fast_modules():
    """``{module: whether it has AVX2 units}`` for the fast modules of the
    subdirectories that ``_kernels/fast/CMakeLists.txt`` adds, read from
    their ``astroviper_add_fast_kernel_module`` calls."""
    fast = Path(__file__).parents[3] / "src/astroviper/_kernels/fast"

    def code(path):  # without CMake comments
        return re.sub(r"#.*", "", path.read_text())

    declared = {}
    for name in re.findall(
        r"add_subdirectory\(\s*(\w+)\s*\)", code(fast / "CMakeLists.txt")
    ):
        calls = re.findall(
            r"astroviper_add_fast_kernel_module\(([^)]*)\)",
            code(fast / name / "CMakeLists.txt"),
        )
        for arguments in calls:
            module = re.search(r"\bNAME\s+(\w+)", arguments).group(1)
            declared[f"_kernels/fast/{name}/{module}"] = (
                "AVX2_SOURCES" in arguments.split()
            )
    return declared


def test_fast_module_lists_are_complete():
    """``FAST_MODULES`` are the declared fast modules with AVX2 units and
    ``BASELINE_FAST_MODULES`` those without, and they are the packages of the
    switch's fast implementations."""
    from astroviper._kernels._switch import _REGISTRY

    declared = _declared_fast_modules()
    assert sorted(FAST_MODULES) == sorted(m for m, avx2 in declared.items() if avx2)
    assert sorted(BASELINE_FAST_MODULES) == sorted(
        m for m, avx2 in declared.items() if not avx2
    )
    packages = {impls["fast"] for impls in _REGISTRY.values() if "fast" in impls}
    assert packages == {
        "astroviper." + module.rpartition("/")[0].replace("/", ".")
        for module in declared
    }


@pytest.mark.parametrize("relative", REFERENCE_MODULES, ids=lambda r: r.split("/")[-1])
def test_reference_module_has_no_fma_and_no_wide_registers(relative):
    _assert_baseline_code(relative)


@pytest.mark.parametrize(
    "relative", BASELINE_FAST_MODULES, ids=lambda r: r.split("/")[-1]
)
def test_baseline_fast_module_has_no_fma_and_no_wide_registers(relative):
    _assert_baseline_code(relative)


@pytest.mark.parametrize("relative", FAST_MODULES, ids=lambda r: r.split("/")[-1])
def test_fast_module_has_no_fma(relative):
    text = _instructions(_module_file(relative))
    assert len(INSTRUCTION.findall(text)) > 1000, "no code was disassembled"
    assert XMM_REGISTER.search(text), "the register pattern matches nothing"
    assert sorted(set(FMA.findall(text))) == []


def _avx2_objects(relative):
    """The ``*_avx2.cpp`` object files of a fast module, from the build
    directories whose linked module is the installed one (not a stale tree or
    another wheel tag's)."""
    configured = os.environ.get("ASTROVIPER_BUILD_DIR")
    roots = (
        [Path(configured)] if configured else Path(__file__).parents[3].glob("build/*")
    )
    target = relative.rpartition("/")[2]
    installed = _module_file(relative).read_bytes()
    objects = []
    for root in roots:
        linked = [
            path
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            for path in root.glob(f"**/{target}{suffix}")
        ]
        if any(path.read_bytes() == installed for path in linked):
            objects += root.glob(f"**/{target}.dir/**/*_avx2.cpp.o")
    return sorted(objects)


def _check_avx2_object(path):
    listing = subprocess.run(
        [NM, str(path)], check=True, capture_output=True, text=True
    )
    symbols = [
        line.split()[-2:] for line in listing.stdout.splitlines() if line.strip()
    ]
    assert [s for s in symbols if s[0] in ("W", "V", "u")] == [], path.name
    assert any(kind == "T" for kind, _ in symbols), f"{path.name} exports nothing"
    assert WIDE_REGISTER.search(_instructions(path)), f"{path.name} has no AVX code"


@pytest.mark.skipif(NM is None, reason="nm (binutils) is not installed")
@pytest.mark.parametrize("relative", FAST_MODULES, ids=lambda r: r.split("/")[-1])
def test_avx2_units_define_no_weak_symbols(relative):
    objects = _avx2_objects(relative)
    if not objects:
        pytest.skip(
            "no build directory with the installed module's objects "
            "(ASTROVIPER_BUILD_DIR)"
        )
    for path in objects:
        _check_avx2_object(path)


#: The C++ compiler for fresh compiles of the AVX2 units, if any.
CXX = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")


@pytest.mark.skipif(NM is None or CXX is None, reason="no nm or no C++ compiler")
@pytest.mark.parametrize("level", ["-O3", "-O2", "-Os", "-Og", "-O1"])
@pytest.mark.parametrize("relative", FAST_MODULES, ids=lambda r: r.split("/")[-1])
def test_avx2_units_define_no_weak_symbols_at_every_level(relative, level, tmp_path):
    """The rule above on fresh compiles of the AVX2 units with the module's
    flags, at the optimisation levels of the build types (Release -O3,
    RelWithDebInfo -O2, MinSizeRel -Os, Debug -Og for these units) and -O1."""
    module = Path(__file__).parents[3] / "src/astroviper" / relative.rpartition("/")[0]
    units = sorted(module.glob("src/*_avx2.cpp"))
    assert units, f"no AVX2 unit in {module}"
    flags = ["-std=c++23", "-fPIC", "-ffp-contract=off", "-mavx2", "-fno-lto", level]
    common = module.parent / "common"  # on every fast module's include path
    flags += ["-DAV_HAVE_AVX2=1", f"-I{module / 'include'}", f"-I{common}"]
    for unit in units:
        target = tmp_path / f"{unit.stem}.o"
        subprocess.run(
            [CXX, *flags, "-c", str(unit), "-o", str(target)],
            check=True,
            capture_output=True,
        )
        _check_avx2_object(target)
