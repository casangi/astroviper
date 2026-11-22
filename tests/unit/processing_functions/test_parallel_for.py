"""Compile and run the C++ test of parallel_for.hpp, the parallel loops of the kernels."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


def test_parallel_for_threads_chunks_and_counts_below_one(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("parallel_for header tests require a C++ compiler")
    directory = Path(__file__).resolve().parent
    repo = directory.parents[2]
    executable = tmp_path / "test_parallel_for"
    result = subprocess.run(
        [
            *compiler,
            "-std=c++23",
            "-O2",
            "-pthread",
            "-I",
            str(repo / "src/astroviper/processing_functions/include"),
            str(directory / "include/test_parallel_for.cpp"),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
