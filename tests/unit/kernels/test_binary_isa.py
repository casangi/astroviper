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
cannot escape the rules. Each fast-kernel module adds its own rules when it
lands. Skipped off x86-64
and when ``objdump`` (binutils) is not installed.
"""

import importlib.machinery
import importlib.util
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

#: FMA3 (vfmadd231pd, vfnmsub132ss, vfmaddsub213ps, ...) and FMA4 mnemonics.
FMA = re.compile(r"\bvfn?m(?:add|sub|addsub|subadd)\w*")
#: Register names wider than xmm, in AT&T (``%ymm0``) or Intel (``ymm0``) syntax.
WIDE_REGISTER = re.compile(r"(?<![\w%])%?[yz]mm\d+\b")
#: SSE register names, which every x86-64 module with floating point uses.
XMM_REGISTER = re.compile(r"(?<![\w%])%?xmm\d+\b")
#: One disassembled instruction: ``  <address>:<tab><instruction>``.
INSTRUCTION = re.compile(r"^\s*[0-9a-f]+:\t", re.MULTILINE)

OBJDUMP = shutil.which("objdump")

pytestmark = [
    pytest.mark.skipif(
        platform.machine().lower() not in ("x86_64", "amd64"),
        reason="x86-64 instruction rules",
    ),
    pytest.mark.skipif(OBJDUMP is None, reason="objdump (binutils) is not installed"),
]


def _package():
    return Path(importlib.util.find_spec("astroviper").origin).parent


def _module_file(relative):
    """Installed file of an extension module, found without importing it."""
    package = _package()
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = package / (relative + suffix)
        if path.is_file():
            return path
    pytest.fail(f"extension module astroviper/{relative} is not installed")


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


@pytest.mark.parametrize("relative", REFERENCE_MODULES, ids=lambda r: r.split("/")[-1])
def test_reference_module_has_no_fma_and_no_wide_registers(relative):
    text = _instructions(_module_file(relative))
    assert len(INSTRUCTION.findall(text)) > 1000, "no code was disassembled"
    assert XMM_REGISTER.search(text), "the register pattern matches nothing"
    assert sorted(set(FMA.findall(text))) == []
    assert sorted(set(WIDE_REGISTER.findall(text))) == []
