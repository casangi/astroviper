"""Restoration layouts and numerical boundary regression tests."""

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    convert_mvc_cubes_to_taylor_normal_equations,
    restore_image,
)
from tests.utils.continuum_images import make_continuum_image


@pytest.mark.parametrize(
    "psf_axis", [None, "frequency", "taylor_term", "psf_taylor_order"]
)
@pytest.mark.parametrize(
    "beam_axis", [None, "frequency", "taylor_term", "psf_taylor_order"]
)
@pytest.mark.parametrize("legacy", [False, True])
def test_restoration_layouts_match_actual_cube_backend(psf_axis, beam_axis, legacy):
    image = make_continuum_image(psf_axis, beam_axis, legacy)
    snapshot = image.copy(deep=True)
    reference, _ = restore_image(make_continuum_image())
    result, _ = restore_image(image)
    xr.testing.assert_allclose(result.SKY_RESTORED, reference.SKY_RESTORED)
    xr.testing.assert_identical(result.SKY_MODEL, snapshot.SKY_MODEL)
    xr.testing.assert_identical(result.SKY_RESIDUAL, snapshot.SKY_RESIDUAL)
    assert np.isfinite(result.SKY_RESTORED.isel(taylor_term=0)).all()
    assert float(result.SKY_RESTORED.isel(taylor_term=0).max()) == pytest.approx(2.25)


@pytest.mark.parametrize("image_shape", [(8, 8), (7, 7), (7, 10), (10, 7), (8, 11)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_restoration_supports_odd_even_and_rectangular_images(image_shape, dtype):
    image = make_continuum_image(image_shape=image_shape)
    for name in ("SKY_MODEL", "SKY_RESIDUAL", "POINT_SPREAD_FUNCTION"):
        image[name] = image[name].astype(dtype)
    snapshot = image.copy(deep=True)

    result, timing = restore_image(image, primary_beam_correction=True)

    assert result.SKY_RESTORED.sizes["l"] == image_shape[0]
    assert result.SKY_RESTORED.sizes["m"] == image_shape[1]
    assert result.SKY_RESTORED_PBCOR.sizes["l"] == image_shape[0]
    assert result.SKY_RESTORED_PBCOR.sizes["m"] == image_shape[1]
    assert result.SKY_RESTORED.dtype == dtype
    assert result.SKY_RESTORED_PBCOR.dtype == dtype
    restored_taylor_zero = result.SKY_RESTORED.isel(taylor_term=0)
    assert np.isfinite(restored_taylor_zero).all()
    assert result.SKY_RESTORED.isel(taylor_term=slice(1, None)).isnull().all()
    assert np.isfinite(result.SKY_RESTORED_PBCOR).all()
    assert float(restored_taylor_zero.max()) == pytest.approx(2.25, rel=1e-6)
    assert float(result.SKY_RESTORED_PBCOR.max()) == pytest.approx(4.5, rel=1e-6)
    xr.testing.assert_identical(result.SKY_MODEL, snapshot.SKY_MODEL)
    xr.testing.assert_identical(result.SKY_RESIDUAL, snapshot.SKY_RESIDUAL)
    assert "T_restore" in timing
    assert "T_correct_sky_by_primary_beam" in timing


def test_restoration_reports_incomplete_backend_result(monkeypatch):
    module = importlib.import_module("astroviper.processing_functions.imaging.restore")

    def broken_restore(image, **kwargs):
        image.attrs["data_groups"]["restored"] = {"sky": "ABSENT"}
        return image, pd.DataFrame()

    monkeypatch.setattr(module, "restore_image", broken_restore)
    with pytest.raises(RuntimeError, match="did not create"):
        restore_image(make_continuum_image())


@pytest.mark.parametrize(
    "fault,message",
    [("empty_psf", "no Taylor orders"), ("beam_dimensions", "Missing dimensions")],
)
def test_restoration_rejects_remaining_invalid_layouts(fault, message):
    image = make_continuum_image()
    if fault == "empty_psf":
        image = image.isel(psf_taylor_order=slice(0, 0))
    else:
        image["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] = (
            image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.isel(polarization=0, drop=True)
        )
    with pytest.raises(ValueError, match=message):
        restore_image(image)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pb_correction_cutoff_includes_exact_boundary(dtype):
    limit = dtype(0.25)
    values = np.array(
        [np.nextafter(limit, dtype(0)), limit, np.nextafter(limit, dtype(1)), np.nan],
        dtype=dtype,
    )
    image = make_continuum_image()
    for name in ("SKY_MODEL", "SKY_RESIDUAL", "POINT_SPREAD_FUNCTION"):
        image[name] = image[name].astype(dtype)
    primary_beam = np.tile(values, 16).reshape(1, 1, 8, 8)
    image["PRIMARY_BEAM"] = xr.DataArray(
        primary_beam,
        dims=("time", "polarization", "l", "m"),
        coords={key: image.coords[key] for key in ("time", "polarization", "l", "m")},
    )

    result, _ = restore_image(
        image,
        primary_beam_correction=True,
        primary_beam_limit=float(limit),
    )
    finite = np.isfinite(result.SKY_RESTORED_PBCOR.values[0, 0])
    np.testing.assert_array_equal(finite[:, 0::4], False)
    np.testing.assert_array_equal(finite[:, 1::4], True)
    np.testing.assert_array_equal(finite[:, 2::4], True)
    np.testing.assert_array_equal(finite[:, 3::4], False)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("weights", [[1.0], [0.0, 1.0, 0.0], [1e-20, 1.0, 1e20]])
def test_mvc_single_term_extreme_and_zero_weights_match_scalar_oracle(dtype, weights):
    weights = np.array(weights)
    n = len(weights)
    dims = ("time", "frequency", "polarization", "l", "m")
    coords = dict(
        time=[0.0], frequency=np.arange(n) + 100.0, polarization=["I"], l=[0.0], m=[0.0]
    )
    residual = xr.DataArray(
        np.arange(1, n + 1, dtype=dtype).reshape(1, n, 1, 1, 1),
        dims=dims,
        coords=coords,
    )
    norm = xr.DataArray(
        weights.reshape(1, n, 1), dims=dims[:3], coords={k: coords[k] for k in dims[:3]}
    )
    actual, psf, beam = convert_mvc_cubes_to_taylor_normal_equations(
        residual,
        xr.ones_like(residual),
        xr.ones_like(residual),
        norm,
        norm,
        nterms=1,
        reference_frequency=100.0,
    )
    expected = sum(float(w) * float(i + 1) for i, w in enumerate(weights)) / sum(
        float(w) for w in weights
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(psf, 1.0)
    np.testing.assert_allclose(beam, 1.0)
    assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mvc_channel_cutoff_excludes_exact_boundary_without_renormalizing(dtype):
    limit = dtype(0.25)
    dims = ("time", "frequency", "polarization", "l", "m")
    coords = dict(
        time=[0.0],
        frequency=[100.0, 110.0, 120.0, 130.0],
        polarization=["I"],
        l=[0.0],
        m=[0.0],
    )
    residual = xr.DataArray(
        np.array([3.0, 5.0, 7.0, 11.0], dtype=dtype).reshape(1, 4, 1, 1, 1),
        dims=dims,
        coords=coords,
    )
    pb = residual.copy(
        data=np.array(
            [np.nextafter(limit, dtype(0)), limit, np.nextafter(limit, dtype(1)), 1.0],
            dtype=dtype,
        ).reshape(residual.shape)
    )
    weights = xr.ones_like(residual.isel(l=0, m=0, drop=True))
    actual, psf, beam = convert_mvc_cubes_to_taylor_normal_equations(
        residual,
        xr.ones_like(residual),
        pb,
        weights,
        weights,
        nterms=1,
        reference_frequency=100.0,
        pblimit=float(limit),
    )
    np.testing.assert_allclose(actual, (7.0 + 11.0) / 4.0)
    np.testing.assert_allclose(beam, (float(pb.values.ravel()[2]) + 1.0) / 4.0)
    np.testing.assert_allclose(psf, 1.0)
