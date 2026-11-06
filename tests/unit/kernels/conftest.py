"""Shared helpers of the kernel-switch tests.

The switch is decided when a shim is imported, and a default test session
imports the shims with ``ASTROVIPER_KERNELS`` unset during collection, so
every test that needs the variable at import runs Python in a subprocess.
"""

import os
import subprocess
import sys

import pytest

_IMAGING = "astroviper.processing_functions.imaging"

#: kernel -> (shim, its extension module, hooked entries)
SHIMS = {
    "dft": (
        f"{_IMAGING}.gridders.direct_fourier_transform_cpp",
        "_direct_fourier_transform_ext",
        ("dft_predict",),
    ),
    "hogbom": (
        f"{_IMAGING}.deconvolvers.hogbom",
        "_hogbom_ext",
        ("clean_cube", "clean_cube_many_threads"),
    ),
    "asp": (
        f"{_IMAGING}.deconvolvers.aspclean",
        "_aspclean_ext",
        ("clean", "clean_cube"),
    ),
}

#: First line of every shim hook.
HOOK = 'if __import__("os").environ.get("ASTROVIPER_KERNELS"):'

#: Selection that installs every hook and runs the reference everywhere.
ALL_REF = "dft=ref,hogbom=ref,asp=ref"

#: Source of ``tiny_hogbom()``: one small ``hogbom.clean_cube`` call through the
#: shim, for subprocesses and Dask workers.
TINY_HOGBOM = """
def tiny_hogbom():
    import numpy as np

    from astroviper.processing_functions.imaging.deconvolvers import hogbom

    residual = np.zeros((1, 1, 1, 8, 8))
    residual[..., 4, 4] = 1.0
    return hogbom.clean_cube(
        residual_cube=residual,
        psf_cube=residual.copy(),
        model_cube=np.zeros_like(residual),
        max_iter_per_cycle_updated=np.full((1, 1, 1), 5, np.int32),
        gain=0.5,
        threshold_per_cycle=np.zeros((1, 1, 1)),
    )
"""


@pytest.fixture
def run_python():
    """Run a fresh interpreter: ``run_python(*args, check=True, timeout=300, **env)``.

    ``args`` are the interpreter's arguments (``"-c", code`` or ``"-m", ...``).
    The ``ASTROVIPER_KERNELS*`` variables are removed from the environment and
    ``env`` is added. Returns the standard output; a non-zero exit fails the
    test with the standard error. With ``check=False`` the
    ``CompletedProcess`` is returned instead.
    """

    def run(*args, check=True, timeout=300, **env):
        environ = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("ASTROVIPER_KERNELS")
        }
        environ.update(env)
        done = subprocess.run(
            [sys.executable, *args],
            env=environ,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if not check:
            return done
        if done.returncode != 0:
            pytest.fail(f"subprocess exited with {done.returncode}:\n{done.stderr}")
        return done.stdout

    return run
