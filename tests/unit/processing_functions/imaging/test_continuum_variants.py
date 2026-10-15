"""Restoration layouts and numerical boundary regression tests."""

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.deconvolvers import mtmfs
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    accumulate_continuum_model,
    convert_mvc_cubes_to_taylor_normal_equations,
    model_update_mtmfs_single_field,
    restore_image,
)
from astroviper.processing_functions.imaging.utils import ImagingDict
from tests.utils.continuum_images import make_continuum_image


def _native_mtmfs_continuum_image(nterms, shape, dtype, support_kind):
    """Build coupled Taylor point sources for the real continuum adapter."""
    ntime, npol = 2, 2
    nl, nm = shape
    frequencies = np.linspace(-0.75, 0.75, nterms + 4)
    moments = np.asarray(
        [(frequencies**order).mean() for order in range(2 * nterms - 1)],
        dtype=dtype,
    )
    hessian = moments[np.add.outer(np.arange(nterms), np.arange(nterms))]
    psf = np.zeros((ntime, 2 * nterms - 1, npol, nl, nm), dtype=dtype)
    residual = np.zeros((ntime, nterms, npol, nl, nm), dtype=dtype)
    expected_model = np.zeros_like(residual)
    source = (nl // 2, nm // 2)
    for tt in range(ntime):
        for pp in range(npol):
            coefficients = np.asarray(
                [
                    (1.0 + 0.25 * tt + 0.125 * pp) * ((-0.4) ** order)
                    for order in range(nterms)
                ],
                dtype=dtype,
            )
            psf[tt, :, pp, source[0], source[1]] = moments
            residual[tt, :, pp, source[0], source[1]] = hessian @ coefficients
            expected_model[tt, :, pp, source[0], source[1]] = coefficients

    coords = {
        "time": np.arange(ntime, dtype=float),
        "taylor_term": np.arange(nterms),
        "psf_taylor_order": np.arange(2 * nterms - 1),
        "polarization": ["I", "Q"],
        "l": np.arange(nl),
        "m": np.arange(nm),
    }
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                residual,
            ),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                psf,
            ),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "polarization"),
                np.zeros((ntime, npol), dtype=dtype),
            ),
        },
        coords=coords,
        attrs={
            "reference_frequency": 1.0e9,
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                }
            },
        },
    )
    support = np.zeros(shape, dtype=dtype)
    support[source] = 1.0
    if support_kind == "mask":
        image["CLEAN_MASK"] = (("l", "m"), support)
        image.attrs["data_groups"]["residual"]["mask"] = "CLEAN_MASK"
    else:
        image["PRIMARY_BEAM"] = (("l", "m"), support)
        image.attrs["data_groups"]["residual"]["primary_beam"] = "PRIMARY_BEAM"
    return image, expected_model


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_mtmfs_receives_taylor_stacks_without_reshaping(monkeypatch, dtype):
    image = make_continuum_image()
    psf_zero = image.POINT_SPREAD_FUNCTION.isel(psf_taylor_order=0, drop=True)
    image = image.drop_dims("psf_taylor_order")
    image["POINT_SPREAD_FUNCTION"] = xr.concat(
        [psf_zero, psf_zero, psf_zero],
        dim=xr.IndexVariable("psf_taylor_order", [0, 1, 2]),
    ).transpose("time", "psf_taylor_order", "polarization", "l", "m")
    for name in ("SKY_MODEL", "SKY_RESIDUAL", "POINT_SPREAD_FUNCTION"):
        image[name] = image[name].astype(dtype)
    image["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"] = (
        ("time", "polarization"),
        np.asarray([[0.2]], dtype=dtype),
    )
    residual_snapshot = image.SKY_RESIDUAL.copy(deep=True)
    calls = []

    def native_clean(residual, psf, model, **kwargs):
        calls.append((residual.shape, psf.shape, model.shape, kwargs))
        assert residual.ndim == psf.ndim == model.ndim == 3
        assert residual.shape == model.shape == (2, 8, 8)
        assert psf.shape == (3, 8, 8)
        assert residual.flags.c_contiguous
        assert psf.flags.c_contiguous
        assert model.flags.c_contiguous
        assert np.shares_memory(residual, image.SKY_RESIDUAL.values)
        assert residual.dtype == psf.dtype == model.dtype == dtype
        np.testing.assert_array_equal(model, 0.0)
        residual *= dtype(0.5)
        model[0] = dtype(1.0)
        model[1] = dtype(2.0)
        return {
            "iter_done": 3,
            "peakres": 0.125,
            "model_flux": float(model[0].sum()),
            "converged": False,
            "stop_code": mtmfs.MODEL_UPDATE_MAX_ITER_PER_CYCLE,
            "scales": np.asarray([0.0]),
            "hessian_condition_number": np.asarray([2.0]),
        }

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.deconvolvers.mtmfs.clean",
        native_clean,
    )
    statistics, timing = model_update_mtmfs_single_field(
        image,
        "mtmfs",
        {
            "max_iter": 20,
            "max_iter_per_cycle": np.asarray([[[3]]]),
            "threshold_per_cycle": np.asarray([[[0.01]]], dtype=dtype),
            "gain": 0.2,
            "scales": [0.0],
        },
    )

    assert len(calls) == 1
    assert calls[0][3]["max_iter_per_cycle"] == 3
    assert calls[0][3]["threshold_per_cycle"] == pytest.approx(0.01)
    assert "psf_fraction" not in calls[0][3]
    np.testing.assert_array_equal(image.SKY_RESIDUAL, residual_snapshot * dtype(0.5))
    np.testing.assert_array_equal(image.SKY_MODEL.isel(taylor_term=0), 1.0)
    np.testing.assert_array_equal(image.SKY_MODEL.isel(taylor_term=1), 2.0)
    previous = xr.Dataset({"SKY_MODEL": xr.zeros_like(image.SKY_MODEL)})
    previous.SKY_MODEL.loc[{"taylor_term": 0}] = dtype(10.0)
    previous.SKY_MODEL.loc[{"taylor_term": 1}] = dtype(20.0)
    accumulated = accumulate_continuum_model(image, previous)
    np.testing.assert_array_equal(
        accumulated.SKY_MODEL.isel(taylor_term=0), dtype(11.0)
    )
    np.testing.assert_array_equal(
        accumulated.SKY_MODEL.isel(taylor_term=1), dtype(22.0)
    )
    fields = next(iter(statistics.data.values()))
    assert fields["iter_done"] == [3]
    assert fields["stop_code"].model_update == mtmfs.MODEL_UPDATE_MAX_ITER_PER_CYCLE
    assert fields["stop_description"] == "Reached max_iter_per_cycle"
    assert fields["peakres"] == [pytest.approx(0.125)]
    assert fields["model_flux"] == [pytest.approx(64.0)]
    assert fields["masksum"] == [64]
    assert fields["max_psf_sidelobe"] == pytest.approx(0.2)
    np.testing.assert_array_equal(fields["scales"], [0.0])
    np.testing.assert_array_equal(fields["hessian_condition_number"], [2.0])
    assert "T_mtmfs" in timing


@pytest.mark.parametrize(
    "nterms,image_shape",
    [(2, (8, 8)), (3, (7, 10)), (4, (10, 7))],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("support_kind", ["mask", "primary_beam"])
def test_native_mtmfs_adapter_covers_all_planes_terms_and_supports(
    nterms, image_shape, dtype, support_kind
):
    """The real adapter preserves every plane and Taylor model coefficient."""
    image, expected_model = _native_mtmfs_continuum_image(
        nterms, image_shape, dtype, support_kind
    )
    residual_storage = image.SKY_RESIDUAL.values
    residual_snapshot = image.SKY_RESIDUAL.copy(deep=True)
    iterations = np.ones((2, 1, 2), dtype=np.int64)
    thresholds = np.zeros((2, 1, 2), dtype=dtype)

    statistics, timing = model_update_mtmfs_single_field(
        image,
        "mtmfs",
        {
            "max_iter": 1,
            "max_iter_per_cycle": iterations,
            "threshold_per_cycle": thresholds,
            "gain": 1.0,
            "scales": [0.0],
            "primary_beam_limit": 0.5,
        },
    )

    assert image.SKY_RESIDUAL.values is residual_storage
    np.testing.assert_allclose(
        image.SKY_RESIDUAL,
        np.zeros_like(residual_snapshot),
        rtol=2e-4,
        atol=2e-5,
    )
    np.testing.assert_allclose(image.SKY_MODEL, expected_model, rtol=2e-4, atol=2e-5)
    assert image.SKY_MODEL.dtype == dtype
    assert len(statistics.data) == 4
    for key, fields in statistics.data.items():
        assert fields["iter_done"] == [1]
        assert fields["masksum"] == [1]
        assert fields["stop_code"].model_update in {
            mtmfs.MODEL_UPDATE_MAX_ITER_PER_CYCLE,
            mtmfs.MODEL_UPDATE_THRESHOLD_PER_CYCLE,
        }
        assert fields["model_flux"] == [
            pytest.approx(expected_model[key.time, 0, key.pol].sum(), rel=2e-4)
        ]
        np.testing.assert_array_equal(fields["scales"], [0.0])
        assert np.isfinite(fields["hessian_condition_number"]).all()
    assert timing.loc[0, "T_mtmfs"] >= 0.0


def test_single_term_mtmfs_must_be_resolved_by_public_api(monkeypatch):
    image = make_continuum_image().isel(
        taylor_term=slice(0, 1), psf_taylor_order=slice(0, 1)
    )
    image["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"] = (
        ("time", "polarization"),
        [[0.2]],
    )
    called = []

    def hogbom_backend(image_arg, deconvolver, params, **kwargs):
        called.append(deconvolver)
        assert image_arg.SKY_RESIDUAL.dims == (
            "time",
            "frequency",
            "polarization",
            "l",
            "m",
        )
        return ImagingDict(), pd.DataFrame({"T_deconvolve": [0.0]})

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.model_update."
        "model_update_cube_single_field",
        hogbom_backend,
    )
    with pytest.raises(ValueError, match="requires 2 to 4 Taylor terms"):
        model_update_mtmfs_single_field(
            image,
            "mtmfs",
            {"max_iter": 1, "max_iter_per_cycle": 1, "gain": 0.1},
        )

    assert called == []


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
