"""Instruction-set hygiene of the installed C++ extension modules.

pybind11 strips Release modules (``nm`` reports no symbols), so these rules
apply to whole modules, read from the disassembly of the installed files:

* no fused multiply-add in any module: the build passes ``-ffp-contract=off``
  so that results are bit-identical across compilers and platforms;
* no 256- or 512-bit vector register in a reference module: the reference
  extensions are built for the baseline x86-64 instruction set.

Each fast-kernel module adds its own rules when it lands. Skipped off x86-64
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
#: AT&T register names wider than xmm.
WIDE_REGISTER = re.compile(r"%[yz]mm\d+")

OBJDUMP = shutil.which("objdump")

pytestmark = [
    pytest.mark.skipif(
        platform.machine().lower() not in ("x86_64", "amd64"),
        reason="x86-64 instruction rules",
    ),
    pytest.mark.skipif(OBJDUMP is None, reason="objdump (binutils) is not installed"),
]


def _module_file(relative):
    """Installed file of an extension module, found without importing it."""
    package = Path(importlib.util.find_spec("astroviper").origin).parent
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


@pytest.mark.parametrize("relative", REFERENCE_MODULES, ids=lambda r: r.split("/")[-1])
def test_reference_module_has_no_fma_and_no_wide_registers(relative):
    text = _instructions(_module_file(relative))
    assert sorted(set(FMA.findall(text))) == []
    assert sorted(set(WIDE_REGISTER.findall(text))) == []
