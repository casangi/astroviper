"""Regression tests for the RelaxedHogbom five-dimensional wrapper."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.deconvolution import (
    hogbom_clean,
    relaxed_hogbom_clean,
)


def _gaussian_psf(size=32, sigma=2.0):
    y, x = np.mgrid[:size, :size]
    psf = np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2.0 * sigma**2))
    return np.ascontiguousarray(psf[None, None, None], dtype=np.float64)


def _bilinear_point(size, x, y, flux=1.0):
    image = np.zeros((size, size), dtype=np.float64)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    dx, dy = x - x0, y - y0
    image[y0, x0] += flux * (1.0 - dx) * (1.0 - dy)
    image[y0, x0 + 1] += flux * dx * (1.0 - dy)
    image[y0 + 1, x0] += flux * (1.0 - dx) * dy
    image[y0 + 1, x0 + 1] += flux * dx * dy
    return image


def _convolve_centered(model, psf):
    convolved = np.fft.ifft2(np.fft.fft2(model) * np.fft.fft2(psf)).real
    return np.roll(
        convolved,
        shift=(-(model.shape[0] // 2), -(model.shape[1] // 2)),
        axis=(0, 1),
    )


def _params(**overrides):
    params = {
        "gain": 1.0,
        "max_iter": 1,
        "threshold": 0.0,
        "max_iter_divergence": -1,
        "relax_adaptive_start": False,
        "relax_constraint_radius": 1.0,
        "relax_min_improvement": 0.0,
    }
    params.update(overrides)
    return params


def test_subpixel_relaxation_reduces_residual_and_preserves_flux():
    psf = _gaussian_psf()
    truth = _bilinear_point(32, 16.30, 15.65)
    dirty = _convolve_centered(truth, psf[0, 0, 0])
    residual = dirty[None, None, None].copy()
    model = np.zeros_like(residual)

    result = relaxed_hogbom_clean(residual, psf, model, _params())

    assert result["relaxation_attempted"][0, 0, 0]
    assert result["relaxation_accepted"][0, 0, 0]
    assert result["relaxed_component_count"][0, 0, 0] == 1
    assert result["final_residual_norm"][0, 0, 0] < (
        1e-8 * result["initial_residual_norm"][0, 0, 0]
    )
    assert model.sum() == pytest.approx(1.0, abs=1e-5)


def test_rejected_trial_is_identical_to_hogbom_result():
    psf = _gaussian_psf()
    truth = _bilinear_point(32, 16.30, 15.65)
    dirty = _convolve_centered(truth, psf[0, 0, 0])
    rng = np.random.default_rng(4)
    dirty += 0.1 * rng.standard_normal(dirty.shape)
    relaxed_residual = dirty[None, None, None].copy()
    hogbom_residual = relaxed_residual.copy()
    relaxed_model = np.zeros_like(relaxed_residual)
    hogbom_model = np.zeros_like(relaxed_residual)

    result = relaxed_hogbom_clean(
        relaxed_residual,
        psf,
        relaxed_model,
        _params(relax_min_improvement=0.999),
    )
    hogbom_clean(hogbom_residual, psf, hogbom_model, _params())

    assert not result["relaxation_accepted"][0, 0, 0]
    np.testing.assert_array_equal(relaxed_model, hogbom_model)
    np.testing.assert_array_equal(relaxed_residual, hogbom_residual)


def test_existing_model_component_is_not_relaxed():
    psf = _gaussian_psf()
    truth = _bilinear_point(32, 16.30, 15.65)
    residual = _convolve_centered(truth, psf[0, 0, 0])[None, None, None]
    residual = np.ascontiguousarray(residual)
    model = np.zeros_like(residual)
    model[0, 0, 0, 4, 5] = 7.0

    relaxed_hogbom_clean(residual, psf, model, _params())

    assert model[0, 0, 0, 4, 5] == 7.0


def test_full_cube_and_psf_polarization_broadcast():
    size = 16
    psf = np.zeros((2, 2, 1, size, size), dtype=np.float32)
    psf[..., size // 2, size // 2] = 1.0
    residual = np.zeros((2, 2, 2, size, size), dtype=np.float32)
    residual[..., 6, 9] = 2.0
    model = np.zeros_like(residual)

    result = relaxed_hogbom_clean(
        residual,
        psf,
        model,
        _params(relax_adaptive_start=False),
        processing_function_threads=3,
    )

    assert result["iterations_performed"].shape == (2, 2, 2)
    np.testing.assert_allclose(residual, 0.0)
    np.testing.assert_allclose(model[..., 6, 9], 2.0)


def test_adaptive_start_skips_mid_progress_model_update():
    size = 16
    psf = np.zeros((1, 1, 1, size, size), dtype=np.float64)
    psf[0, 0, 0, size // 2, size // 2] = 1.0
    residual = np.zeros_like(psf)
    residual[0, 0, 0, 7, 8] = 1.0
    model = np.zeros_like(residual)

    gain = 1.0 - 0.5 ** (1.0 / 100.0)
    result = relaxed_hogbom_clean(
        residual,
        psf,
        model,
        _params(gain=gain, max_iter=100, relax_adaptive_start=True),
    )

    assert result["hogbom_rms_ratio"][0, 0, 0] == pytest.approx(0.5)
    assert not result["relaxation_attempted"][0, 0, 0]
    assert not result["relaxation_accepted"][0, 0, 0]
    assert model[0, 0, 0, 7, 8] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("relax_constraint_radius", -0.1, "non-negative"),
        ("relax_max_components", 0, "positive integer"),
        ("relax_min_improvement", 1.0, r"\[0, 1\)"),
        ("relax_optimizer_max_iter", 0, "positive integer"),
    ],
)
def test_relaxation_parameter_validation(key, value, match):
    psf = _gaussian_psf(16)
    residual = np.zeros_like(psf)
    model = np.zeros_like(psf)
    with pytest.raises(ValueError, match=match):
        relaxed_hogbom_clean(residual, psf, model, {key: value})
