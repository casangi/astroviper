"""Unit tests for
:func:`astroviper.processing_functions.image_analysis.transform_polarization_basis.transform_polarization_basis`.

Focus: the ``overwrite`` behaviour. ``overwrite=False`` used to raise
``KeyError`` because it built an empty ``xr.Dataset`` (coordinates only) and
then indexed data variables that were never copied into it; it must instead
return an independent, fully-populated, transformed copy while leaving the
input untouched.
"""

import numpy as np
import xarray as xr

from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
    transform_polarization_basis,
)


def _make_linear_image(xx=12.0, yy=8.0):
    """Small (XX, YY) image whose two correlations are spatially uniform."""
    data = np.empty((2, 3, 3), dtype=float)
    data[0] = xx
    data[1] = yy
    return xr.Dataset(
        {"SKY": (("polarization", "l", "m"), data)},
        coords={"polarization": ["XX", "YY"], "l": [0, 1, 2], "m": [0, 1, 2]},
    )


def test_overwrite_false_returns_transformed_copy_and_leaves_input_unchanged():
    """Regression: overwrite=False must not raise and must not mutate the input."""
    xds = _make_linear_image(xx=12.0, yy=8.0)

    out = transform_polarization_basis(xds, "stokes", overwrite=False)

    # A distinct object, transformed to Stokes I/Q.
    assert out is not xds
    assert list(out.polarization.values) == ["I", "Q"]
    # Symmetric convention: I = (XX + YY) / 2 = 10, Q = (XX - YY) / 2 = 2.
    np.testing.assert_allclose(out["SKY"].isel(polarization=0).values, 10.0)
    np.testing.assert_allclose(out["SKY"].isel(polarization=1).values, 2.0)

    # Input is left completely untouched.
    assert list(xds.polarization.values) == ["XX", "YY"]
    np.testing.assert_allclose(xds["SKY"].isel(polarization=0).values, 12.0)
    np.testing.assert_allclose(xds["SKY"].isel(polarization=1).values, 8.0)


def test_overwrite_false_preserves_skipped_passthrough_variables():
    """Variables the transform skips (e.g. a PSF) survive in the returned copy.

    With the empty-Dataset bug they would have been dropped (or triggered the
    KeyError); the deep copy keeps every data variable present.
    """
    xds = _make_linear_image()
    psf = xr.DataArray(
        np.ones((2, 3, 3)),
        dims=("polarization", "l", "m"),
        coords={"polarization": ["XX", "YY"], "l": [0, 1, 2], "m": [0, 1, 2]},
    )
    psf.attrs["type"] = "point_spread_function"
    xds["POINT_SPREAD_FUNCTION"] = psf

    out = transform_polarization_basis(xds, "stokes", overwrite=False)

    assert "POINT_SPREAD_FUNCTION" in out.data_vars
    # Skipped -> passed through unchanged.
    np.testing.assert_allclose(out["POINT_SPREAD_FUNCTION"].values, np.ones((2, 3, 3)))


def test_overwrite_true_mutates_in_place():
    xds = _make_linear_image()
    out = transform_polarization_basis(xds, "stokes", overwrite=True)
    # True in-place: the same object is returned and the input is left fully
    # consistent -- both the data and the polarization labels are updated.
    assert out is xds
    assert list(xds.polarization.values) == ["I", "Q"]
    np.testing.assert_allclose(xds["SKY"].isel(polarization=0).values, 10.0)
    np.testing.assert_allclose(xds["SKY"].isel(polarization=1).values, 2.0)
