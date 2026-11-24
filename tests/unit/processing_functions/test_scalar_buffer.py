"""Compare the C++ scalar proxy directly with built-in compound assignment."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def scalar_buffer_test_binary(tmp_path_factory):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("ScalarBuffer header tests require a C++ compiler")
    directory = Path(__file__).resolve().parent
    repo = directory.parents[2]
    executable = tmp_path_factory.mktemp("scalar_buffer") / "test_scalar_buffer"
    result = subprocess.run(
        [
            *compiler,
            "-std=c++23",
            "-O2",
            "-ffp-contract=off",
            "-I",
            str(repo / "src/astroviper/processing_functions/include"),
            str(directory / "include/test_scalar_buffer.cpp"),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize(
    "scenario", ["cancellation", "rounding", "fractional_integer", "same_type"]
)
def test_scalar_buffer_add_assign_matches_builtin(scalar_buffer_test_binary, scenario):
    result = subprocess.run(
        [str(scalar_buffer_test_binary), scenario],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
