"""Full PSF convolution blocks retain interactions between delta components."""

import json
import warnings

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import CONTINUUM_SINGLE_FIELD
from astroviper.processing_functions.imaging.config_check._spatial_hessian import (
    check_gaussian_spatial_hessian,
    check_psf_spatial_hessian,
    check_spatial_hessian_blocks,
)
from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_model_products,
)
from tests.utils.continuum_images import make_runaway_taylor_image


def coupled_psfs(size=8, coupling=0.98):
    """Unit local Hessian; off-position Taylor coupling through neighbours."""
    psfs = np.zeros((3, size, size))
    centre = size // 2
    psfs[0, centre, centre] = psfs[2, centre, centre] = 1.0
    psfs[1, centre - 1, centre] = psfs[1, centre + 1, centre] = coupling / 2
    return psfs


def test_spatial_diagnostic_detects_coupling_invisible_to_central_pixels():
    psfs = coupled_psfs()
    original = psfs.copy()
    assert np.array_equal(psfs[:, 4, 4], [1, 0, 1])
    with pytest.warns(RuntimeWarning, match="Spatial Taylor-Hessian diagnostic"):
        result = check_psf_spatial_hessian(psfs, precision="float64")
    assert result["max_order_zero_noise_amplification"] == pytest.approx(
        1 / np.sqrt(1 - 0.98**2)
    )
    assert result["excess_noise_mode_count"] > 0
    assert result["rank_deficient_mode_count"] == 0
    assert result["non_positive_mode_count"] == 0
    assert json.loads(json.dumps(result)) == result
    np.testing.assert_array_equal(psfs, original)


def test_fourier_blocks_match_explicit_spatial_delta_component_matrix():
    psfs = coupled_psfs(size=4, coupling=0.5)
    # Construct the full 2*n_pixels square convolution Hessian as an oracle.
    # Only tests construct this matrix; production inspects small Fourier blocks.
    kernels = np.fft.ifftshift(psfs, axes=(1, 2))
    pixels = list(np.ndindex(4, 4))
    dense = np.empty((32, 32))
    for k in range(2):
        for l in range(2):
            for a, first in enumerate(pixels):
                for b, second in enumerate(pixels):
                    separation = tuple(
                        (first[axis] - second[axis]) % 4 for axis in range(2)
                    )
                    dense[k * 16 + a, l * 16 + b] = kernels[k + l][separation]
    eigenvalues = np.linalg.eigvalsh(dense)
    result = check_psf_spatial_hessian(psfs, precision="float64")
    assert result["minimum_relative_eigenvalue"] == pytest.approx(
        eigenvalues.min() / eigenvalues.max()
    )
    # The diagonal of the full inverse averages the mode-by-mode variances.
    # Its largest mode amplification agrees with the spectral norm here.
    expected_max = 1 / np.sqrt(1 - 0.5**2)
    assert result["max_order_zero_noise_amplification"] == pytest.approx(expected_max)


def test_model_product_checker_retains_spatial_diagnostics_and_local_guards():
    image = make_runaway_taylor_image()
    image.POINT_SPREAD_FUNCTION.values[0, :, 0] = coupled_psfs(size=32)
    original = image.copy(deep=True)
    with pytest.warns(RuntimeWarning, match="Spatial Taylor-Hessian diagnostic"):
        result = check_model_products(
            image, {"deconvolver": "mtmfs"}, mode=CONTINUUM_SINGLE_FIELD
        )
    assert result["taylor_hessian_order_zero_noise_amplification"][0][
        "per_scale"
    ] == pytest.approx([1])
    assert (
        result["taylor_hessian_spatial_blocks"][0]["max_order_zero_noise_amplification"]
        > 3
    )
    assert result["taylor_hessian_spatial_blocks"][0]["time_index"] == 0
    import xarray as xr

    xr.testing.assert_identical(image, original)


def test_missing_spatial_modes_do_not_require_a_global_pixel_inverse():
    transfer = np.zeros((3, 4, 3))
    transfer[0, 0, 0] = transfer[2, 0, 0] = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_spatial_hessian_blocks(
            transfer,
            np.arange(4),
            np.arange(3),
            precision="float32",
            frequency_units="cycles_per_pixel",
        )
    assert result["unsupported_mode_count"] == 11
    assert result["assessed_mode_count"] == 1
    assert result["max_order_zero_noise_amplification"] == pytest.approx(1)
    assert not caught


def test_supported_but_taylor_deficient_mode_is_reported_separately():
    transfer = np.array([[[1, 1]], [[0, 1]], [[1, 1]]], dtype=float)
    with pytest.warns(RuntimeWarning, match="Spatial Taylor-Hessian diagnostic"):
        result = check_spatial_hessian_blocks(
            transfer,
            [0],
            [0, 1],
            precision="float64",
            frequency_units="cycles_per_pixel",
        )
    assert result["rank_deficient_mode_count"] == 1
    assert result["unsupported_mode_count"] == 0
    assert result["assessed_mode_count"] == 1


def test_no_support_reports_no_amplification_instead_of_a_fictitious_pass():
    with pytest.warns(RuntimeWarning, match="Spatial Taylor-Hessian diagnostic"):
        result = check_spatial_hessian_blocks(
            np.zeros((3, 2, 2)),
            [0, 1],
            [0, 1],
            precision="float64",
            frequency_units="cycles_per_radian",
        )
    assert result["supported_mode_count"] == 0
    assert result["max_order_zero_noise_amplification"] is None
    assert result["worst_mode"] is None


def test_asymmetric_psf_is_not_silently_symmetrized():
    psfs = coupled_psfs(coupling=0)
    psfs[1, 3, 4] = 0.2
    with pytest.warns(RuntimeWarning, match="Spatial Taylor-Hessian diagnostic"):
        result = check_psf_spatial_hessian(psfs, precision="float64")
    assert result["non_hermitian_mode_count"] > 0


def test_spatial_block_batches_keep_correct_mode_locations():
    transfer = np.zeros((3, 65, 65))
    transfer[0] = transfer[2] = 1
    transfer[1, -1, -1] = 0.98
    with pytest.warns(RuntimeWarning, match="Spatial Taylor-Hessian diagnostic"):
        result = check_spatial_hessian_blocks(
            transfer,
            np.arange(65),
            np.arange(65),
            precision="float64",
            frequency_units="cycles_per_pixel",
        )
    assert result["mode_count"] == 4225
    assert result["excess_noise_mode_count"] == 1
    assert result["worst_mode"]["frequency_l"] == 64
    assert result["worst_mode"]["frequency_m"] == 64


def test_frequency_dependent_gaussian_width_changes_spatial_coupling():
    frequencies = np.array([1e9, 2e9, 3e9])
    kwargs = dict(
        n_taylor_terms=2,
        reference_frequency=2e9,
        image_size=[64, 64],
        cell_size=[-1e-5, 1e-5],
        precision="float64",
    )
    # Bmax proportional to 1/nu gives identical Gaussian widths: the spatial
    # transfer separates from the Taylor matrix and G0 is constant at one.
    constant_width = check_gaussian_spatial_hessian(
        frequencies, np.ones(3), 5000 * 1e9 / frequencies, **kwargs
    )
    with pytest.warns(RuntimeWarning, match="Spatial Taylor-Hessian diagnostic"):
        natural_resolution = check_gaussian_spatial_hessian(
            frequencies, np.ones(3), np.full(3, 5000), **kwargs
        )
    assert constant_width["max_order_zero_noise_amplification"] == pytest.approx(1)
    assert natural_resolution["max_order_zero_noise_amplification"] > 3
    assert natural_resolution["sampling"] == "at_most_33_by_33_half_fourier_grid"


def test_actual_gaussian_psfs_agree_with_analytic_blocks_at_resolved_modes():
    size = 64
    sigma = np.array([3.0, 2.0, 1.5])
    x = np.array([-0.5, 0, 0.5])
    pixel = np.arange(size) - size // 2
    radius_squared = pixel[:, None] ** 2 + pixel[None, :] ** 2
    gaussian_psfs = np.exp(
        -radius_squared[None, :, :] / (2 * sigma[:, None, None] ** 2)
    )
    psfs = np.einsum(
        "ik,ilm->klm", np.polynomial.polynomial.polyvander(x, 2), gaussian_psfs
    )
    actual = np.fft.rfft2(np.fft.ifftshift(psfs, axes=(1, 2)))[:, :8, :8]
    frequency = np.arange(8) / size
    radius_uv = frequency[:, None] ** 2 + frequency[None, :] ** 2
    gaussian_transfer = (
        2
        * np.pi
        * sigma[:, None, None] ** 2
        * np.exp(-2 * np.pi**2 * sigma[:, None, None] ** 2 * radius_uv)
    )
    analytic = np.einsum(
        "ik,ilm->klm", np.polynomial.polynomial.polyvander(x, 2), gaussian_transfer
    )
    kwargs = dict(precision="float64", frequency_units="cycles_per_pixel")
    first = check_spatial_hessian_blocks(actual, frequency, frequency, **kwargs)
    second = check_spatial_hessian_blocks(analytic, frequency, frequency, **kwargs)
    assert first["max_order_zero_noise_amplification"] == pytest.approx(
        second["max_order_zero_noise_amplification"], rel=1e-10
    )


@pytest.mark.parametrize("baseline", [0, -1, np.nan])
def test_gaussian_resolution_requires_a_usable_baseline(baseline):
    with pytest.raises(ValueError, match="positive finite usable projected baselines"):
        check_gaussian_spatial_hessian(
            np.array([1e9]),
            np.ones(1),
            baseline,
            n_taylor_terms=1,
            reference_frequency=1e9,
            image_size=[8, 8],
            cell_size=[1e-5, 1e-5],
            precision="float64",
        )


@pytest.mark.parametrize("fault", ["nan", "even_orders", "axis"])
def test_invalid_spatial_transfers_fail(fault):
    transfer = np.zeros((3, 2, 2))
    if fault == "nan":
        transfer[0, 0, 0] = np.nan
    elif fault == "even_orders":
        transfer = transfer[:2]
    else:
        transfer = transfer[:, :1]
    with pytest.raises(ValueError, match="Spatial Taylor-Hessian transfers"):
        check_spatial_hessian_blocks(
            transfer,
            [0, 1],
            [0, 1],
            precision="float64",
            frequency_units="cycles_per_pixel",
        )
