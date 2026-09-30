"""Unit tests for
:func:`astroviper.processing_functions.image_analysis.transform_polarization_basis.transform_polarization_basis`.

Focus: the ``overwrite`` behaviour. ``overwrite=False`` used to raise
``KeyError`` because it built an empty ``xr.Dataset`` (coordinates only) and
then indexed data variables that were never copied into it; it must instead
return an independent, fully-populated, transformed copy while leaving the
input untouched.
"""

import numpy as np
import pytest
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


# --------------------------------------------------------------------------- #
# Four correlations: complex in the instrument basis, real in Stokes
# --------------------------------------------------------------------------- #
STOKES = (2.0, 0.3, -0.4, 0.25)  # I, Q, U, V
LABELS = {
    "linear": ["XX", "XY", "YX", "YY"],
    "circular": ["RR", "RL", "LR", "LL"],
}


def _correlations(stokes, basis):
    i, q, u, v = stokes
    if basis == "linear":  # XX = I + Q, XY = U + iV, YX = U - iV, YY = I - Q
        return [i + q, u + 1j * v, u - 1j * v, i - q]
    return [i + v, q + 1j * u, q - 1j * u, i - v]  # RR, RL = Q + iU, LR, LL


def _make_four_correlation_image(planes, basis, dtype=np.complex128):
    data = np.stack([np.full((3, 3), plane, dtype=dtype) for plane in planes])
    return xr.Dataset(
        {
            "SKY": (("polarization", "l", "m"), data),
            "MASK": (("polarization", "l", "m"), np.ones((4, 3, 3), dtype=bool)),
        },
        coords={"polarization": LABELS[basis], "l": [0, 1, 2], "m": [0, 1, 2]},
    )


@pytest.mark.parametrize("basis", ["linear", "circular"])
@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_four_correlations_to_stokes_and_back(basis, dtype):
    real_dtype = np.float64 if dtype == np.complex128 else np.float32
    rtol = 1e-14 if dtype == np.complex128 else 1e-6
    xds = _make_four_correlation_image(_correlations(STOKES, basis), basis, dtype)

    out = transform_polarization_basis(xds, "stokes")

    assert out is xds  # in place: the dataset is the same object
    assert list(out.polarization.values) == ["I", "Q", "U", "V"]
    assert out["SKY"].dtype == real_dtype  # Stokes images are real
    assert out["SKY"].values.flags["C_CONTIGUOUS"]
    for plane, value in enumerate(STOKES):
        np.testing.assert_allclose(out["SKY"].values[plane], value, rtol=rtol)
    # masks belong to their plane and are left alone
    assert out["MASK"].dtype == bool and out["MASK"].values.all()

    back = transform_polarization_basis(out, basis)

    assert list(back.polarization.values) == LABELS[basis]
    assert back["SKY"].dtype == dtype  # the cross hands are complex again
    for plane, value in enumerate(_correlations(STOKES, basis)):
        np.testing.assert_allclose(back["SKY"].values[plane], value, rtol=rtol)
    assert back["MASK"].dtype == bool and back["MASK"].values.all()


@pytest.mark.parametrize("basis", ["linear", "circular"])
def test_half_plane_images_give_the_stokes_parameters(basis):
    """The gridder holds every sample once, so the image of one correlation is
    complex and only the combination of the conjugate pair is the sky: with any
    junk ``J``, ``P_XY = (U + iV) + J`` and ``P_YX = (U - iV) - conj(J)`` (and any
    imaginary part on the parallel hands) must give the Stokes parameters."""
    rng = np.random.default_rng(3)
    planes = _correlations(STOKES, basis)
    junk = complex(*rng.normal(size=2))
    planes[0] = planes[0] + 1j * rng.normal()
    planes[3] = planes[3] + 1j * rng.normal()
    planes[1] = planes[1] + junk
    planes[2] = planes[2] - np.conj(junk)
    xds = _make_four_correlation_image(planes, basis)

    out = transform_polarization_basis(xds, "stokes")

    assert out["SKY"].dtype == np.float64
    for plane, value in enumerate(STOKES):
        np.testing.assert_allclose(out["SKY"].values[plane], value, rtol=1e-14)


def test_two_hands_are_transformed_in_place_and_stay_real():
    """The two-hand conversion is real: same buffer, same data type, masks included."""
    xds = _make_linear_image(xx=12.0, yy=8.0)
    xds["SKY"] = xds["SKY"].astype(np.float32)
    buffer = xds["SKY"].values

    out = transform_polarization_basis(xds, "stokes")

    assert np.shares_memory(out["SKY"].values, buffer)
    assert out["SKY"].dtype == np.float32
    np.testing.assert_allclose(out["SKY"].values[:, 0, 0], [10.0, 2.0])
