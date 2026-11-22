"""Compile and run the C++ test of fft.hpp, the FFT wrapper of the deconvolvers."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

POCKETFFT_CONFIGURATION = ["-pthread", "-DPOCKETFFT_CACHE_SIZE=16"]


def _compile(arguments, timeout):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("FFT header tests require a C++ compiler")
    directory = Path(__file__).resolve().parent
    repo = directory.parents[2]
    return subprocess.run(
        [
            *compiler,
            "-std=c++23",
            "-O2",
            "-ffp-contract=off",
            *arguments,
            "-I",
            str(repo / "src/astroviper/processing_functions/include"),
            str(directory / "include/test_fft.cpp"),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def fft_test_binary(tmp_path_factory):
    executable = tmp_path_factory.mktemp("fft") / "test_fft"
    result = _compile([*POCKETFFT_CONFIGURATION, "-o", str(executable)], 300)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize(
    "scenario", ["double_64x64", "double_101x97", "float_128x128", "float_101x97"]
)
def test_fft_matches_direct_summation_for_every_thread_count(fft_test_binary, scenario):
    result = subprocess.run(
        [str(fft_test_binary), scenario],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "macros",
    [
        [],
        ["-DPOCKETFFT_CACHE_SIZE=8"],
        ["-DPOCKETFFT_CACHE_SIZE=16", "-DPOCKETFFT_NO_MULTITHREADING"],
    ],
)
def test_fft_refuses_another_pocketfft_configuration(macros):
    result = _compile(["-fsyntax-only", *macros], 120)
    assert result.returncode != 0
    assert "pocketfft is configured in cmake/AstroviperPybind.cmake only" in (
        result.stderr
    )
