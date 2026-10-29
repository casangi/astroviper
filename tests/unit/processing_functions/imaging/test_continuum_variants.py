"""Restoration layouts and numerical boundary regression tests."""

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.deconvolvers import mtmfs
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    _convert_mvc_cubes_to_taylor_normal_equations,
    accumulate_continuum_model,
    build_continuum_residual_imaging_dict,
    model_update_mtmfs_single_field,
    restore_continuum_image,
)
from astroviper.processing_functions.imaging.utils import ImagingDict
from tests.utils.continuum_images import (
    make_continuum_image,
    make_runaway_taylor_image,
    make_single_term_continuum_image,
    make_single_term_faint_source_image,
    make_single_term_runaway_image,
)


def _native_mtmfs_continuum_image(n_taylor_terms, shape, dtype, support_kind, npol=2):
    """Build coupled Taylor point sources for the real continuum adapter."""
    ntime = 2
    nl, nm = shape
    frequencies = np.linspace(-0.75, 0.75, n_taylor_terms + 4)
    moments = np.asarray(
        [(frequencies**order).mean() for order in range(2 * n_taylor_terms - 1)],
        dtype=dtype,
    )
    hessian = moments[
        np.add.outer(np.arange(n_taylor_terms), np.arange(n_taylor_terms))
    ]
    psf = np.zeros((ntime, 2 * n_taylor_terms - 1, npol, nl, nm), dtype=dtype)
    residual = np.zeros((ntime, n_taylor_terms, npol, nl, nm), dtype=dtype)
    expected_model = np.zeros_like(residual)
    source = (nl // 2, nm // 2)
    for tt in range(ntime):
        for pp in range(npol):
            coefficients = np.asarray(
                [
                    (1.0 + 0.25 * tt + 0.125 * pp) * ((-0.4) ** order)
                    for order in range(n_taylor_terms)
                ],
                dtype=dtype,
            )
            psf[tt, :, pp, source[0], source[1]] = moments
            residual[tt, :, pp, source[0], source[1]] = hessian @ coefficients
            expected_model[tt, :, pp, source[0], source[1]] = coefficients

    coords = {
        "time": np.arange(ntime, dtype=float),
        "taylor_term": np.arange(n_taylor_terms),
        "psf_taylor_order": np.arange(2 * n_taylor_terms - 1),
        "polarization": ["I", "Q", "U", "V"][:npol],
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
        image["MASK"] = (("l", "m"), support)
        image.attrs["data_groups"]["residual"]["mask"] = "MASK"
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
    reference, _ = restore_continuum_image(make_continuum_image())
    result, _ = restore_continuum_image(image)
    xr.testing.assert_allclose(result.SKY_RESTORED, reference.SKY_RESTORED)
    xr.testing.assert_identical(result.SKY_MODEL, snapshot.SKY_MODEL)
    xr.testing.assert_identical(result.SKY_RESIDUAL, snapshot.SKY_RESIDUAL)
    assert result.SKY_RESTORED.dims == ("time", "polarization", "l", "m")
    assert np.isfinite(result.SKY_RESTORED).all()
    assert float(result.SKY_RESTORED.max()) == pytest.approx(2.25)


@pytest.mark.parametrize("image_shape", [(8, 8), (7, 7), (7, 10), (10, 7), (8, 11)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_restoration_supports_odd_even_and_rectangular_images(image_shape, dtype):
    image = make_continuum_image(image_shape=image_shape)
    for name in ("SKY_MODEL", "SKY_RESIDUAL", "POINT_SPREAD_FUNCTION"):
        image[name] = image[name].astype(dtype)
    snapshot = image.copy(deep=True)

    result, timing = restore_continuum_image(image, primary_beam_correction=True)

    assert result.SKY_RESTORED.sizes["l"] == image_shape[0]
    assert result.SKY_RESTORED.sizes["m"] == image_shape[1]
    assert result.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.sizes["l"] == image_shape[0]
    assert result.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.sizes["m"] == image_shape[1]
    assert result.SKY_RESTORED.dtype == dtype
    assert result.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.dtype == dtype
    assert result.SKY_RESTORED.dims == ("time", "polarization", "l", "m")
    assert result.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.dims == (
        "time",
        "polarization",
        "l",
        "m",
    )
    assert np.isfinite(result.SKY_RESTORED).all()
    assert np.isfinite(result.SKY_RESTORED_PRIMARY_BEAM_CORRECTED).all()
    assert float(result.SKY_RESTORED.max()) == pytest.approx(2.25, rel=1e-6)
    assert float(result.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.max()) == pytest.approx(
        4.5, rel=1e-6
    )
    xr.testing.assert_identical(result.SKY_MODEL, snapshot.SKY_MODEL)
    xr.testing.assert_identical(result.SKY_RESIDUAL, snapshot.SKY_RESIDUAL)
    assert "T_restore" in timing
    assert "T_correct_sky_by_primary_beam" in timing


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mtmfs_restoration_uses_principal_residual_copy(dtype):
    """MT-MFS restores the principal tt0 solution without changing residuals."""
    image = make_continuum_image()
    nl, nm = image.sizes["l"], image.sizes["m"]
    moments = np.asarray([1.0, 0.25, 0.5], dtype=dtype)
    hessian = moments[np.add.outer(np.arange(2), np.arange(2))]
    coefficients = np.asarray([0.25, -0.1], dtype=dtype)
    dirty_terms = hessian @ coefficients

    residual = np.empty((1, 2, 1, nl, nm), dtype=dtype)
    residual[:, 0] = dirty_terms[0]
    residual[:, 1] = dirty_terms[1]
    image["SKY_RESIDUAL"] = (image.SKY_RESIDUAL.dims, residual)
    image["SKY_MODEL"] = xr.zeros_like(image.SKY_RESIDUAL)

    psf = np.zeros((1, 3, 1, nl, nm), dtype=dtype)
    psf[0, :, 0, nl // 2, nm // 2] = moments
    image = image.drop_dims("psf_taylor_order")
    image["POINT_SPREAD_FUNCTION"] = (
        ("time", "psf_taylor_order", "polarization", "l", "m"),
        psf,
    )
    residual_snapshot = image.SKY_RESIDUAL.copy(deep=True)

    result, _ = restore_continuum_image(image, deconvolver="mtmfs")

    np.testing.assert_allclose(result.SKY_RESTORED, coefficients[0], rtol=1e-6)
    xr.testing.assert_identical(result.SKY_RESIDUAL, residual_snapshot)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_mtmfs_receives_taylor_stacks_without_reshaping(monkeypatch, dtype):
    image = make_continuum_image()
    image.attrs.pop("reference_frequency", None)
    image.attrs["continuum_imaging"] = {"reference_frequency_hz": 1.5e9}
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
        calls.append((residual, psf, model, kwargs))
        assert residual.ndim == psf.ndim == model.ndim == 3
        assert residual.shape == model.shape == (2, 8, 8)
        assert psf.shape == (3, 8, 8)
        assert residual.flags.c_contiguous
        assert psf.flags.c_contiguous
        assert model.flags.c_contiguous
        assert np.shares_memory(residual, image.SKY_RESIDUAL.values)
        assert np.shares_memory(psf, image.POINT_SPREAD_FUNCTION.values)
        assert residual.dtype == psf.dtype == model.dtype == dtype
        np.testing.assert_array_equal(model, 0.0)
        residual *= dtype(0.5)
        model[0] = dtype(1.0)
        model[1] = dtype(2.0)
        return {
            "iterations_performed": 3,
            "peak_residual": 0.125,
            "model_flux": float(model[0].sum()),
            "converged": False,
            "stop_code": mtmfs.STOP_MAX_ITER,
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
            "max_iter_per_cycle_updated": np.asarray([[[3]]]),
            "threshold_per_cycle": np.asarray([[[0.01]]], dtype=dtype),
            "gain": 0.2,
            "scales": [0.0],
        },
    )

    assert len(calls) == 1
    assert np.shares_memory(calls[0][2], image.SKY_MODEL.values)
    assert calls[0][3]["max_iter_per_cycle_updated"] == 3
    assert calls[0][3]["threshold_per_cycle"] == pytest.approx(0.01)
    assert "psf_fraction" not in calls[0][3]
    # the divergence test: Asp's RMS limit and the peak limit, 1 + gain each
    assert calls[0][3]["max_iter_divergence"] == 1
    assert calls[0][3]["divergence_rms_factor"] == pytest.approx(1.2)
    assert calls[0][3]["divergence_peak_factor"] == pytest.approx(1.2)
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
    assert fields["frequency"] == 1.5e9
    assert fields["iter_done"] == [3]
    assert fields["stop_code"].model_update == mtmfs.STOP_MAX_ITER
    assert fields["stop_description"] == "Reached max_iter_per_cycle_updated"
    assert fields["peakres"] == [pytest.approx(0.125)]
    assert fields["model_flux"] == [pytest.approx(64.0)]
    assert fields["masksum"] == [64]
    assert fields["max_psf_sidelobe"] == pytest.approx(0.2)
    np.testing.assert_array_equal(fields["scales"], [0.0])
    np.testing.assert_array_equal(fields["hessian_condition_number"], [2.0])
    assert "T_mtmfs" in timing


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("read_only", [False, True])
def test_native_mtmfs_copies_and_writes_back_mvc_four_polarization_layout(
    monkeypatch, dtype, read_only
):
    """MVC Taylor-last and read-only residuals use a writable native copy."""
    image, _ = _native_mtmfs_continuum_image(2, (8, 8), dtype, "mask", npol=4)
    canonical = image.SKY_RESIDUAL.values
    storage = np.ascontiguousarray(canonical.transpose(0, 2, 3, 4, 1))
    if read_only:
        storage.flags.writeable = False
    image["SKY_RESIDUAL"] = xr.DataArray(
        storage,
        dims=("time", "polarization", "l", "m", "taylor_term"),
        coords={dim: image.coords[dim] for dim in image.SKY_RESIDUAL.dims},
    )
    residual_snapshot = image.SKY_RESIDUAL.copy(deep=True)
    calls = []

    def native_clean(residual, psf, model, **kwargs):
        calls.append(residual)
        assert residual.flags.c_contiguous
        assert residual.flags.writeable
        assert not np.shares_memory(residual, storage)
        residual *= dtype(0.5)
        model[0] = dtype(1.0)
        return {
            "iterations_performed": 1,
            "peak_residual": float(np.max(np.abs(residual[0]))),
            "model_flux": float(model[0].sum()),
            "stop_code": mtmfs.STOP_MAX_ITER,
            "scales": np.asarray([0.0]),
            "hessian_condition_number": np.asarray([1.0]),
        }

    monkeypatch.setattr(mtmfs, "clean", native_clean)
    model_update_mtmfs_single_field(
        image,
        "mtmfs",
        {"max_iter": 1, "gain": 0.1, "scales": [0.0]},
    )

    assert len(calls) == 8
    np.testing.assert_array_equal(image.SKY_RESIDUAL, residual_snapshot * dtype(0.5))
    np.testing.assert_array_equal(image.SKY_MODEL.isel(taylor_term=0), dtype(1.0))


@pytest.mark.parametrize(
    "params,max_iter,threshold",
    [
        ({"max_iter_per_cycle": 2, "threshold": 0.05}, 2, 0.05),
        ({"max_iter_per_cycle": 10, "threshold": 0.05}, 5, 0.05),
        ({"max_iter_per_cycle": -1}, 5, 0.0),
        ({}, 5, 0.0),
        ({"max_iter_per_cycle": 2, "threshold": None}, 2, 0.0),
        (
            {
                "max_iter_per_cycle": 2,
                "threshold": 0.2,
                "max_iter_per_cycle_updated": None,
                "threshold_per_cycle": None,
            },
            2,
            0.2,
        ),
    ],
)
def test_native_mtmfs_default_per_cycle_limits_follow_deconvolve(
    monkeypatch, params, max_iter, threshold
):
    """Without the per-cycle limits: min(max_iter_per_cycle, max_iter), a
    negative cap meaning no cap, and threshold_per_cycle = threshold."""
    image, _ = _native_mtmfs_continuum_image(2, (8, 8), np.float64, "mask")
    calls = []

    def native_clean(residual, psf, model, **kwargs):
        calls.append(kwargs)
        return {
            "iterations_performed": 0,
            "peak_residual": 0.0,
            "model_flux": 0.0,
            "stop_code": mtmfs.STOP_THRESHOLD,
            "scales": np.asarray([0.0]),
            "hessian_condition_number": np.asarray([1.0]),
        }

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.deconvolvers.mtmfs.clean",
        native_clean,
    )
    model_update_mtmfs_single_field(
        image,
        "mtmfs",
        {"max_iter": 5, "gain": 0.1, "scales": [0.0], **params},
    )

    assert len(calls) == 4
    for kwargs in calls:
        assert kwargs["max_iter_per_cycle_updated"] == max_iter
        assert kwargs["threshold_per_cycle"] == pytest.approx(threshold)


@pytest.mark.parametrize(
    "params,match",
    [
        ({"gain": 0.0}, "gain must be between 0 and 1"),
        ({"gain": -0.5}, "gain must be between 0 and 1"),
        ({"gain": 1.5}, "gain must be between 0 and 1"),
        ({"max_iter_divergence": 0}, "max_iter_divergence must be"),
        ({"max_iter_divergence": True}, "max_iter_divergence must be"),
        ({"max_iter_divergence": 1.0}, "max_iter_divergence must be"),
    ],
)
def test_native_mtmfs_refuses_gain_and_divergence_values_deconvolve_refuses(
    monkeypatch, params, match
):
    """The divergence limits come from the gain, so no adaptive gain."""
    image, _ = _native_mtmfs_continuum_image(2, (8, 8), np.float64, "mask")
    residual_snapshot = image.SKY_RESIDUAL.copy(deep=True)
    calls = []
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.deconvolvers.mtmfs.clean",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    with pytest.raises(ValueError, match=match):
        model_update_mtmfs_single_field(
            image,
            "mtmfs",
            {"max_iter": 5, "gain": 0.1, "scales": [0.0], **params},
        )

    assert calls == []
    xr.testing.assert_identical(image.SKY_RESIDUAL, residual_snapshot)


@pytest.mark.parametrize(
    "n_taylor_terms,image_shape",
    [(2, (8, 8)), (3, (7, 10)), (4, (10, 7))],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("support_kind", ["mask", "primary_beam"])
def test_native_mtmfs_adapter_covers_all_planes_terms_and_supports(
    n_taylor_terms, image_shape, dtype, support_kind
):
    """The real adapter preserves every plane and Taylor model coefficient."""
    image, expected_model = _native_mtmfs_continuum_image(
        n_taylor_terms, image_shape, dtype, support_kind
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
            "max_iter_per_cycle_updated": iterations,
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
            mtmfs.STOP_MAX_ITER,
            mtmfs.STOP_THRESHOLD,
        }
        assert fields["model_flux"] == [
            pytest.approx(expected_model[key.time, 0, key.pol].sum(), rel=2e-4)
        ]
        np.testing.assert_array_equal(fields["scales"], [0.0])
        assert np.isfinite(fields["hessian_condition_number"]).all()
    assert timing.loc[0, "T_mtmfs"] >= 0.0


@pytest.mark.parametrize(
    "params,limit",
    [
        ({}, 0.0),
        ({"primary_beam_limit": 0.0}, 0.0),
        ({"primary_beam_limit": 0.2}, 0.2),
        ({"primary_beam_limit": 0.5}, 0.5),
    ],
)
def test_native_mtmfs_cleans_the_support_the_stop_test_measures(
    monkeypatch, params, limit
):
    """Without a mask, MT-MFS cleans the pixels where the primary beam is at
    least primary_beam_limit (default 0) times its peak and not 0, exactly
    those the stop test measures."""
    image, _ = _native_mtmfs_continuum_image(2, (8, 8), np.float64, "primary_beam")
    primary_beam = np.linspace(1.0, 0.0, 64).reshape(8, 8)  # ends on 0
    primary_beam[7, 6] = np.nan
    image["PRIMARY_BEAM"] = (("l", "m"), primary_beam)
    with np.errstate(invalid="ignore"):
        expected = (primary_beam >= limit) & (primary_beam != 0)
    masks = []

    def native_clean(residual, psf, model, **kwargs):
        masks.append(kwargs["mask"].astype(bool))
        return {
            "iterations_performed": 0,
            "peak_residual": 0.0,
            "model_flux": 0.0,
            "stop_code": mtmfs.STOP_THRESHOLD,
            "scales": np.asarray([0.0]),
            "hessian_condition_number": np.asarray([1.0]),
        }

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.deconvolvers.mtmfs.clean",
        native_clean,
    )
    stop_test = build_continuum_residual_imaging_dict(image, "residual", params)
    model_update, _ = model_update_mtmfs_single_field(
        image, "mtmfs", {"max_iter": 5, "gain": 0.1, "scales": [0.0], **params}
    )

    assert expected.sum() == {0.0: 62, 0.2: 51, 0.5: 32}[limit]
    assert len(masks) == 4
    for mask in masks:
        np.testing.assert_array_equal(mask, expected)
    for fields in (*stop_test.data.values(), *model_update.data.values()):
        assert fields["masksum"] == [expected.sum()]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "max_iter_divergence,iterations,stop_code",
    [(1, 7, mtmfs.STOP_DIVERGED), (-1, 50, mtmfs.STOP_MAX_ITER)],
)
def test_native_mtmfs_divergence_test_stops_a_runaway(
    dtype, max_iter_divergence, iterations, stop_code
):
    """The real adapter runs the divergence test, with both limits 1.1 at
    gain 0.1 (the peak limit stops this runaway); -1 turns it off."""
    image = make_runaway_taylor_image(dtype)

    statistics, _ = model_update_mtmfs_single_field(
        image,
        "mtmfs",
        {
            "max_iter": 50,
            "max_iter_per_cycle_updated": 50,
            "gain": 0.1,
            "scales": [0.0],
            "max_iter_divergence": max_iter_divergence,
        },
    )

    (fields,) = statistics.data.values()
    assert fields["iter_done"] == [iterations]
    assert fields["stop_code"].model_update == stop_code
    assert fields["peakres"][0] > fields["start_peakres"][0]


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
            {"max_iter": 1, "max_iter_per_cycle_updated": 1, "gain": 0.1},
        )

    assert called == []


def test_hogbom_model_update_does_not_leak_temporary_mask_metadata(monkeypatch):
    """Repeated Högbom updates isolate their mask and register the real model."""
    image = make_continuum_image()
    image["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"] = (
        ("time", "polarization"),
        [[0.2]],
    )
    del image.attrs["data_groups"]["model"]
    calls = []

    def hogbom_backend(hogbom_image, *args, **kwargs):
        calls.append(hogbom_image)
        hogbom_image["MASK"] = xr.ones_like(hogbom_image.SKY_RESIDUAL)
        hogbom_image.attrs["data_groups"]["residual"]["mask"] = "MASK"
        return ImagingDict(), pd.DataFrame({"T_deconvolve": [0.0]})

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.model_update."
        "model_update_cube_single_field",
        hogbom_backend,
    )

    for _ in range(2):
        model_update_mtmfs_single_field(
            image,
            "hogbom",
            {"max_iter": 0, "primary_beam_limit": 0.2},
        )

    assert len(calls) == 2
    assert all(
        call.attrs["data_groups"]["residual"]["mask"] == "MASK" for call in calls
    )
    assert "mask" not in image.attrs["data_groups"]["residual"]
    assert "MASK" not in image
    assert image.attrs["data_groups"]["model"]["sky"] == "SKY_MODEL"
    assert "SKY_MODEL" in image


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hogbom_model_update_preserves_image_precision(monkeypatch, dtype):
    """Högbom receives and returns the configured image precision unchanged."""
    image = make_continuum_image()
    for name in ("SKY_MODEL", "SKY_RESIDUAL"):
        image[name] = image[name].astype(dtype)
    psf_dtype = np.float64 if dtype == np.float32 else np.float32
    image["POINT_SPREAD_FUNCTION"] = image.POINT_SPREAD_FUNCTION.astype(psf_dtype)
    image["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"] = (
        ("time", "polarization"),
        np.asarray([[0.2]], dtype=dtype),
    )
    value = dtype(0.12345678901234568)
    received_dtypes = []

    def hogbom_backend(hogbom_image, *args, **kwargs):
        received_dtypes.append(
            (
                hogbom_image.SKY_RESIDUAL.dtype,
                hogbom_image.POINT_SPREAD_FUNCTION.dtype,
                hogbom_image.SKY_MODEL.dtype,
            )
        )
        hogbom_image.SKY_MODEL.data[...] = value
        return ImagingDict(), pd.DataFrame({"T_deconvolve": [0.0]})

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.model_update."
        "model_update_cube_single_field",
        hogbom_backend,
    )

    model_update_mtmfs_single_field(image, "hogbom", {"max_iter": 1})

    assert received_dtypes == [(dtype, dtype, dtype)]
    assert image.SKY_MODEL.dtype == dtype
    np.testing.assert_array_equal(image.SKY_MODEL.isel(taylor_term=0), value)


def test_hogbom_adapter_builds_one_contiguous_boolean_mask_plane(monkeypatch):
    """Continuum applies its >0.5 mask rule before calling cube Högbom."""
    image = make_continuum_image()
    image["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"] = (
        ("time", "polarization"),
        np.zeros((1, 1)),
    )
    mask = np.ones(image.SKY_RESIDUAL.shape, dtype=np.float64)
    mask[..., 0, 0] = 0.3
    mask[..., 0, 1] = np.nan
    # Exercise the layout produced when a spatial mask is broadcast across
    # polarization: polarization is the innermost dimension before the
    # continuum adapter canonicalizes the temporary cube.
    image["MASK"] = (
        ("time", "taylor_term", "l", "m", "polarization"),
        mask.transpose(0, 1, 3, 4, 2),
    )
    image.attrs["data_groups"]["residual"]["mask"] = "MASK"

    def hogbom_backend(hogbom_image, *args, **kwargs):
        received = hogbom_image.MASK.values
        assert received.dtype == np.bool_
        assert received.flags.c_contiguous
        assert hogbom_image.MASK.dims == (
            "time",
            "frequency",
            "polarization",
            "l",
            "m",
        )
        assert not received[..., 0, 0].any()
        assert not received[..., 0, 1].any()
        assert received[..., 1:, :].all()
        return ImagingDict(), pd.DataFrame({"T_deconvolve": [0.0]})

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.model_update."
        "model_update_cube_single_field",
        hogbom_backend,
    )

    model_update_mtmfs_single_field(image, "hogbom", {"max_iter": 1})


def test_hogbom_adapter_canonicalizes_generated_primary_beam_mask(monkeypatch):
    """The adapter also canonicalizes the default primary-beam mask."""
    image = make_continuum_image()
    image["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"] = (
        ("time", "polarization"),
        np.zeros((1, 1)),
    )

    def hogbom_backend(hogbom_image, *args, **kwargs):
        mask_name = hogbom_image.attrs["data_groups"]["residual"]["mask"]
        received = hogbom_image[mask_name]
        assert received.dims == (
            "time",
            "frequency",
            "polarization",
            "l",
            "m",
        )
        assert received.dtype == np.bool_
        assert received.values.flags.c_contiguous
        return ImagingDict(), pd.DataFrame({"T_deconvolve": [0.0]})

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.model_update."
        "model_update_cube_single_field",
        hogbom_backend,
    )

    model_update_mtmfs_single_field(image, "hogbom", {"max_iter": 1})


@pytest.mark.parametrize(
    "asp_options", [True, False], ids=["asp_options_on", "asp_options_off"]
)
def test_asp_hands_taylor_term_zero_to_the_cube_asp_kernel(monkeypatch, asp_options):
    """The adapter runs the cube Asp path on Taylor term zero of every
    (time, polarization) plane, with the per-plane limits, the divergence
    limits of Asp and the Asp options, and copies the model back."""
    import astroviper.processing_functions.imaging.deconvolution as deconvolution

    image, _, _ = make_single_term_continuum_image()
    limits = np.array([[[3, 4]], [[5, 6]]])  # (time, plane, polarization)
    thresholds = np.array([[[0.05, 0.06]], [[0.07, 0.08]]])
    stop_codes = np.array([[[1, 5]], [[4, 2]]])
    calls = []

    def clean_cube(**kwargs):
        calls.append(kwargs)
        kwargs["model"][..., 8, 7] += 0.5
        kwargs["residual"][...] *= 0.5
        return {
            "iterations_performed": limits.copy(),
            "retval": np.zeros_like(limits),
            "stop_code": stop_codes.copy(),
            "converged": stop_codes == 2,
            "diverged": stop_codes == 4,
            "peak_residual": np.zeros(limits.shape),
            "model_flux": np.full(limits.shape, 0.5),
        }

    monkeypatch.setattr(deconvolution.aspclean, "clean_cube", clean_cube)
    statistics, timing = model_update_mtmfs_single_field(
        image,
        "asp",
        {
            "max_iter": 10,
            "max_iter_per_cycle_updated": limits,
            "threshold_per_cycle": thresholds,
            "gain": 0.2,
            "max_iter_divergence": 3,
            "largestscale": 4,
            "asp_residual_components": asp_options,
            "asp_scale_significance": asp_options,
        },
    )

    (kwargs,) = calls
    cube_shape = (2, 1, 2, 25, 21)  # (time, frequency, polarization, l, m)
    assert kwargs["residual"].shape == cube_shape
    assert kwargs["psf"].shape == cube_shape
    assert kwargs["mask"].shape == cube_shape
    np.testing.assert_array_equal(kwargs["max_iter_per_cycle_updated"], limits)
    np.testing.assert_array_equal(kwargs["threshold_per_cycle"], thresholds)
    assert kwargs["gain"] == 0.2
    # The divergence test as Asp takes it: the RMS limit at 1 + gain, no peak limit.
    assert kwargs["max_iter_divergence"] == 3
    assert kwargs["divergence_rms_factor"] == pytest.approx(1.2)
    assert kwargs["divergence_peak_factor"] == np.inf
    assert kwargs["residual_components"] is asp_options
    assert kwargs["scale_significance"] is asp_options
    assert kwargs["largestscale"] == 4

    expected_model = np.zeros(image.SKY_RESIDUAL.shape)
    expected_model[..., 8, 7] = 0.5
    assert image.SKY_MODEL.dims == image.SKY_RESIDUAL.dims
    np.testing.assert_array_equal(image.SKY_MODEL.values, expected_model)
    description = image.attrs["data_groups"]["model"]["description"]
    assert description.endswith("Continuum model increment produced by Asp CLEAN.")
    records = {(key.time, key.pol): fields for key, fields in statistics.data.items()}
    assert sorted(records) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (time, pol), fields in records.items():
        assert fields["iter_done"] == [limits[time, 0, pol]]
        assert fields["stop_code"].model_update == stop_codes[time, 0, pol]
    assert "T_model_update_cycle" in timing


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "asp_options", [True, False], ids=["asp_options_on", "asp_options_off"]
)
def test_asp_cleans_every_plane_of_taylor_term_zero(dtype, asp_options):
    """The real Asp kernel cleans all four (time, polarization) planes. With
    threshold_per_cycle 0 each spends its 200 iterations (stop code 1); the
    residual peak falls to about 2 % of its start (measured: at most 2.1 %)
    and the model holds the flux of the plane (measured: within 3.6 %)."""
    image, sky, scale = make_single_term_continuum_image(dtype)

    statistics, _ = model_update_mtmfs_single_field(
        image,
        "asp",
        {
            "max_iter": 200,
            "max_iter_per_cycle_updated": 200,
            "threshold_per_cycle": 0.0,
            "gain": 0.1,
            "asp_residual_components": asp_options,
            "asp_scale_significance": asp_options,
        },
    )

    assert image.SKY_MODEL.dims == image.SKY_RESIDUAL.dims
    assert image.SKY_MODEL.dtype == dtype
    model_flux = image.SKY_MODEL.values.sum(axis=(-2, -1))[:, 0]  # (time, pol)
    records = {(key.time, key.pol): fields for key, fields in statistics.data.items()}
    assert sorted(records) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (time, pol), fields in records.items():
        assert fields["iter_done"] == [200]
        assert fields["stop_code"].model_update == 1
        assert abs(fields["peakres"][0]) < 0.05 * abs(fields["start_peakres"][0])
        np.testing.assert_allclose(
            model_flux[time, pol], scale[time, pol] * sky.sum(), rtol=0.1
        )


def _assert_continuum_asp_is_cube_asp(continuum, params):
    """Run continuum Asp on ``continuum`` (one Taylor term) and cube Asp on a
    one-channel cube built here from its arrays; the models and residuals
    must agree bit for bit."""
    from astroviper.processing_functions.imaging.model_update import (
        model_update_cube_single_field,
    )

    dims = ("time", "frequency", "polarization", "l", "m")
    names = ["SKY_RESIDUAL", "POINT_SPREAD_FUNCTION", "PRIMARY_BEAM"]
    names += ["CLEAN_MASK"] if "CLEAN_MASK" in continuum else []
    group = dict(continuum.attrs["data_groups"]["residual"])
    group["max_sidelobe_point_spread_function"] = "MAX_SIDELOBE_POINT_SPREAD_FUNCTION"
    cube = xr.Dataset(
        {name: (dims, continuum[name].values.copy()) for name in names},
        coords={name: continuum.coords[name] for name in dims},
        attrs={"data_groups": {"residual": group}},
    )
    cube["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"] = (
        dims[:3],
        continuum.MAX_SIDELOBE_POINT_SPREAD_FUNCTION.values[:, None, :],
    )

    model_update_mtmfs_single_field(continuum, "asp", dict(params))
    model_update_cube_single_field(cube, "asp", dict(params), model_exists=False)

    assert continuum.SKY_MODEL.values.any()
    np.testing.assert_array_equal(continuum.SKY_MODEL.values, cube.SKY_MODEL.values)
    np.testing.assert_array_equal(
        continuum.SKY_RESIDUAL.values, cube.SKY_RESIDUAL.values
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "asp_options", [True, False], ids=["asp_options_on", "asp_options_off"]
)
def test_continuum_asp_with_one_taylor_term_is_cube_asp_on_one_channel(
    dtype, asp_options
):
    """Continuum Asp with one Taylor term and cube Asp on one channel give
    the same model and residual. The PSF and the primary beam, and with
    primary_beam_limit the mask, differ from plane to plane, so a plane taken
    for another shows."""
    continuum, _, _ = make_single_term_continuum_image(dtype)
    nl, nm = continuum.sizes["l"], continuum.sizes["m"]
    yy, xx = np.indices((nl, nm))
    r2 = (yy - nl // 2) ** 2 + (xx - nm // 2) ** 2
    # (time, plane, polarization, l, m)
    width = (1.2 + 0.3 * np.arange(2)[:, None] + 0.2 * np.arange(2))[
        :, None, :, None, None
    ]
    continuum.POINT_SPREAD_FUNCTION.values[:] = np.exp(-r2 / (2 * width**2))
    continuum.PRIMARY_BEAM.values[:] = np.exp(-r2 / (2 * (4 * width) ** 2))
    _assert_continuum_asp_is_cube_asp(
        continuum,
        {
            "max_iter": 100,
            "max_iter_per_cycle_updated": 100,
            "threshold_per_cycle": 0.0,
            "gain": 0.1,
            "primary_beam_limit": 0.5,
            "asp_residual_components": asp_options,
            "asp_scale_significance": asp_options,
        },
    )


def test_continuum_coral_with_a_clean_mask_is_cube_coral_on_one_channel():
    """The same with the clean mask and threshold_per_cycle 5 of the faint
    source of the Coral test below, where the test of significance per scale
    cleans below the threshold."""
    continuum, _, _ = make_single_term_faint_source_image()
    _assert_continuum_asp_is_cube_asp(
        continuum,
        {
            "max_iter": 3000,
            "max_iter_per_cycle_updated": 3000,
            "threshold_per_cycle": 5.0,
            "gain": 0.1,
            "asp_residual_components": True,
            "asp_scale_significance": True,
        },
    )


@pytest.mark.parametrize(
    "asp_options", [True, False], ids=["asp_options_on", "asp_options_off"]
)
@pytest.mark.parametrize(
    "max_iter_divergence,iterations,stop_code", [(1, 30, 4), (-1, 50, 1)]
)
def test_asp_divergence_test_stops_a_runaway(
    asp_options, max_iter_divergence, iterations, stop_code
):
    """The real adapter runs the divergence test as Asp takes it (gain 0.1:
    the RMS limit at 1.1 times the lowest RMS, no peak limit); -1 turns it
    off. The counts are measured; both option settings give them."""
    image = make_single_term_runaway_image()

    statistics, _ = model_update_mtmfs_single_field(
        image,
        "asp",
        {
            "max_iter": 50,
            "max_iter_per_cycle_updated": 50,
            "gain": 0.1,
            "max_iter_divergence": max_iter_divergence,
            "asp_residual_components": asp_options,
            "asp_scale_significance": asp_options,
        },
    )

    (fields,) = statistics.data.values()
    assert fields["iter_done"] == [iterations]
    assert fields["stop_code"].model_update == stop_code
    assert abs(fields["peakres"][0]) > abs(fields["start_peakres"][0])


def _clean_faint_source(image, residual_components, scale_significance):
    """One model update of the real Asp kernel through the continuum adapter
    with threshold_per_cycle 5 (5 times the noise); return the Taylor term
    zero model and the plane's record."""
    statistics, _ = model_update_mtmfs_single_field(
        image,
        "asp",
        {
            "max_iter": 3000,
            "max_iter_per_cycle_updated": 3000,
            "threshold_per_cycle": 5.0,
            "gain": 0.1,
            "asp_residual_components": residual_components,
            "asp_scale_significance": scale_significance,
        },
    )
    (fields,) = statistics.data.values()
    return image.SKY_MODEL.values[0, 0, 0], fields


def test_coral_cleans_a_faint_source_by_its_significance_with_a_clean_mask():
    """Coral (Asp with asp_residual_components and asp_scale_significance) on
    the real kernel, with an explicit clean mask and noise outside it. The
    dirty peak of the broad source (2) is below threshold_per_cycle (5), so
    Asp alone cleans nothing and the residual-shaped components take nothing.
    The test of significance per scale measures the robust noise of the
    residual smoothed with each initial scale on the pixels outside the mask,
    and cleans the source at the scales where it stands out of that noise
    (measured: 34 % of its flux, the model peak 1 pixel from its centre).
    Replacing the noise outside the mask by white noise of the same standard
    deviation, with every pixel in the mask unchanged, lowers the noise of the
    smoothed residual, and more is cleaned (measured: 116 % of the flux)."""
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        MODEL_UPDATE_THRESHOLD_PER_CYCLE,
    )

    image, sky, mask = make_single_term_faint_source_image()
    dirty = image.SKY_RESIDUAL.values[0, 0, 0].copy()
    assert (~mask).any()
    assert np.abs(dirty[mask]).max() < 5.0

    coral, fields = _clean_faint_source(image, True, True)
    assert fields["stop_code"].model_update == MODEL_UPDATE_THRESHOLD_PER_CYCLE
    assert fields["iter_done"][0] > 1
    assert 0.2 * sky.sum() < coral.sum() < 0.5 * sky.sum()
    peak = np.unravel_index(np.argmax(coral), coral.shape)
    assert np.hypot(peak[0] - 64, peak[1] - 64) <= 3

    # The residual-shaped components take nothing below the threshold.
    image, _, _ = make_single_term_faint_source_image()
    significance_only, _ = _clean_faint_source(image, False, True)
    np.testing.assert_array_equal(significance_only, coral)

    # Without the test of significance nothing is cleaned.
    image, _, _ = make_single_term_faint_source_image()
    no_significance, fields = _clean_faint_source(image, True, False)
    assert fields["stop_code"].model_update == MODEL_UPDATE_THRESHOLD_PER_CYCLE
    assert not no_significance.any()

    # The noise is that of the pixels outside the mask.
    image, _, _ = make_single_term_faint_source_image(white_noise_outside_mask=True)
    np.testing.assert_array_equal(image.SKY_RESIDUAL.values[0, 0, 0][mask], dirty[mask])
    white, fields = _clean_faint_source(image, True, True)
    assert fields["stop_code"].model_update == MODEL_UPDATE_THRESHOLD_PER_CYCLE
    assert white.sum() > 2 * coral.sum()


def test_restoration_reports_incomplete_backend_result(monkeypatch):
    module = importlib.import_module("astroviper.processing_functions.imaging.restore")

    def broken_restore(image, **kwargs):
        image.attrs["data_groups"]["restored"] = {"sky": "ABSENT"}
        return image, pd.DataFrame()

    monkeypatch.setattr(module, "restore_image", broken_restore)
    with pytest.raises(RuntimeError, match="did not create"):
        restore_continuum_image(make_continuum_image())


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
        restore_continuum_image(image)


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

    result, _ = restore_continuum_image(
        image,
        primary_beam_correction=True,
        primary_beam_limit=float(limit),
    )
    finite = np.isfinite(result.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.values[0, 0])
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
    actual, psf, beam = _convert_mvc_cubes_to_taylor_normal_equations(
        residual,
        xr.ones_like(residual),
        xr.ones_like(residual),
        norm,
        norm,
        n_taylor_terms=1,
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
    actual, psf, beam = _convert_mvc_cubes_to_taylor_normal_equations(
        residual,
        xr.ones_like(residual),
        pb,
        weights,
        weights,
        n_taylor_terms=1,
        reference_frequency=100.0,
        primary_beam_limit=float(limit),
    )
    expected_beam = (float(pb.values.ravel()[2]) + 1.0) / 4.0
    # The cutoff applies to the effective beam only: the residual fits every
    # channel with weight pb**2, as the PSF sums every channel.
    channel_pb = pb.values.ravel().astype(float)
    expected = (
        expected_beam
        * np.sum(channel_pb * np.array([3.0, 5.0, 7.0, 11.0]))
        / np.sum(channel_pb**2)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(beam, expected_beam)
    np.testing.assert_allclose(psf, 1.0)
