"""The visibility-manipulation functions call ``ms.xr_ms`` and must register it."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["ms_spectral_frame_conversion", "phase_shift"])
def test_import_registers_xr_ms_accessor(module):
    # A fresh interpreter: other tests import xradio.measurement_set themselves.
    code = (
        "import xarray as xr\n"
        f"import astroviper.processing_functions.visibility_manipulation.{module}\n"
        "assert 'xr_ms' in dir(xr.DataTree), 'xr_ms accessor not registered'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
