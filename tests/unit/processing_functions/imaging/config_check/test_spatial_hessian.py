"""Full PSF convolution blocks retain interactions between delta components."""

import json

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    _spatial_hessian,
)
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


def test_spatial_diagnostic_detects_coupling_invisible_to_central_pixels(
    diagnostic_warnings,
):
    psfs = coupled_psfs()
    original = psfs.copy()
    assert np.array_equal(psfs[:, 4, 4], [1, 0, 1])
    with diagnostic_warnings(match="Spatial Taylor-Hessian diagnostic"):
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
    # The diagnostic removes the unpaired leading lag on even axes. Build
    # the independent oracle for exactly that retained periodic domain.
    kernels = np.fft.ifftshift(psfs[:, 1:, 1:], axes=(1, 2))
    pixels = list(np.ndindex(3, 3))
    dense = np.empty((18, 18))
    for k in range(2):
        for l in range(2):
            for a, first in enumerate(pixels):
                for b, second in enumerate(pixels):
                    separation = tuple(
                        (first[axis] - second[axis]) % 3 for axis in range(2)
                    )
                    dense[k * 9 + a, l * 9 + b] = kernels[k + l][separation]
    eigenvalues = np.linalg.eigvalsh(dense)
    result = check_psf_spatial_hessian(psfs, precision="float64")
    assert result["minimum_relative_eigenvalue"] == pytest.approx(
        eigenvalues.min() / eigenvalues.max()
    )
    # The diagonal of the full inverse averages the mode-by-mode variances.
    # Its largest mode amplification agrees with the spectral norm here.
    expected_max = 1 / np.sqrt(1 - 0.5**2)
    assert result["max_order_zero_noise_amplification"] == pytest.approx(expected_max)


def test_model_product_checker_retains_spatial_diagnostics_and_local_guards(
    diagnostic_warnings,
):
    image = make_runaway_taylor_image()
    image.POINT_SPREAD_FUNCTION.values[0, :, 0] = coupled_psfs(size=32)
    original = image.copy(deep=True)
    with diagnostic_warnings(match="Spatial Taylor-Hessian diagnostic"):
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


def test_missing_spatial_modes_do_not_require_a_global_pixel_inverse(
    diagnostic_warnings,
):
    transfer = np.zeros((3, 4, 3))
    transfer[0, 0, 0] = transfer[2, 0, 0] = 1
    with diagnostic_warnings() as caught:
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


def test_supported_but_taylor_deficient_mode_is_reported_separately(
    diagnostic_warnings,
):
    transfer = np.array([[[1, 1]], [[0, 1]], [[1, 1]]], dtype=float)
    with diagnostic_warnings(match="Spatial Taylor-Hessian diagnostic"):
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


def test_no_support_reports_no_amplification_instead_of_a_fictitious_pass(
    diagnostic_warnings,
):
    with diagnostic_warnings(match="Spatial Taylor-Hessian diagnostic"):
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


def test_asymmetric_psf_is_not_silently_symmetrized(
    diagnostic_warnings,
):
    psfs = coupled_psfs(coupling=0)
    psfs[1, 3, 4] = 0.2
    with diagnostic_warnings(match="Spatial Taylor-Hessian diagnostic"):
        result = check_psf_spatial_hessian(psfs, precision="float64")
    assert result["non_hermitian_mode_count"] > 0


@pytest.mark.parametrize(
    "shape", [(64, 64), (192, 192), (192, 128), (64, 65), (65, 64), (65, 65)]
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_self_adjoint_psf_retains_only_paired_lags(
    diagnostic_warnings, shape, precision
):
    # A tilted cosine is physically even under simultaneous lag inversion,
    # but its unmatched negative-edge lines need not be even individually.
    l = np.arange(shape[0]) - shape[0] // 2
    m = np.arange(shape[1]) - shape[1] // 2
    plane = np.cos(0.17 * l[:, None] + 0.31 * m[None, :]).astype(precision)
    psfs = np.stack([plane, np.zeros_like(plane), 0.25 * plane])
    before = psfs.copy()
    with diagnostic_warnings():
        result = check_psf_spatial_hessian(psfs, precision=precision)
    retained = [n - int(n % 2 == 0) for n in shape]
    assert result["input_image_shape"] == list(shape)
    assert result["diagnostic_image_shape"] == retained
    assert result["mode_count"] == retained[0] * (retained[1] // 2 + 1)
    assert result["non_hermitian_mode_count"] == 0
    np.testing.assert_array_equal(psfs, before)


@pytest.mark.parametrize("size", [32, 192])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("error_factor", [0.25, 4.0])
def test_fft_stage_tolerance_accepts_roundoff_but_reports_asymmetry(
    diagnostic_warnings, size, precision, error_factor
):
    psfs = coupled_psfs(size=size, coupling=0).astype(precision)
    epsilon = np.finfo(precision).eps
    relative = epsilon * np.ceil(np.log2((size - 1) ** 2))
    psfs[0, size // 2 + 1, size // 2] = error_factor * relative
    with diagnostic_warnings():
        result = check_psf_spatial_hessian(psfs, precision=precision)
    assert result["fft_relative_roundoff_tolerance"] == relative
    assert result["hermitian_absolute_tolerance"][0] == pytest.approx(
        relative * np.abs(psfs[0]).sum(dtype=np.float64), rel=1e-12, abs=0
    )
    assert (result["non_hermitian_mode_count"] > 0) == (error_factor > 1)


@pytest.mark.parametrize("tolerance", [[-1, 0, 0], [np.nan, 0, 0], [0, 0]])
def test_invalid_hermitian_allowance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="Hermitian tolerances"):
        check_spatial_hessian_blocks(
            np.ones((3, 1, 1)),
            [0],
            [0],
            precision="float64",
            frequency_units="cycles_per_pixel",
            hermitian_absolute_tolerance=tolerance,
        )


@pytest.mark.parametrize("single_precision", [False, True])
def test_gridded_even_psf_passes_self_adjointness(
    diagnostic_warnings, single_precision
):
    from astroviper.processing_functions.imaging.residual_update import (
        imaging_setup_single_field,
    )
    from tests.unit.processing_functions.imaging.test_exact_psf_subtraction import (
        _empty_image,
    )
    from tests.unit.processing_functions.imaging.test_max_cycles_auto import (
        IMAGE_PARAMS,
        _processing_set,
    )

    params = {**IMAGE_PARAMS, "image_size": [192, 192]}
    image, _ = imaging_setup_single_field(
        _processing_set(),
        _empty_image(192),
        params,
        {"weighting": "natural"},
        processing_set_data_group_name="base",
        single_precision_image=single_precision,
        fft_backend="scipy",
    )
    psf = image.POINT_SPREAD_FUNCTION.values[0, 0, :1]
    before = psf.copy()
    with diagnostic_warnings():
        result = check_psf_spatial_hessian(
            psf, precision="float32" if single_precision else "float64"
        )
    assert result["non_hermitian_mode_count"] == 0
    assert result["assessed_mode_count"] > 0
    np.testing.assert_array_equal(psf, before)


def test_spatial_block_batches_keep_correct_mode_locations(
    diagnostic_warnings,
):
    transfer = np.zeros((3, 65, 65))
    transfer[0] = transfer[2] = 1
    transfer[1, -1, -1] = 0.98
    with diagnostic_warnings(match="Spatial Taylor-Hessian diagnostic"):
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


def test_frequency_dependent_gaussian_width_changes_spatial_coupling(
    diagnostic_warnings,
):
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
    with diagnostic_warnings(match="Spatial Taylor-Hessian diagnostic"):
        natural_resolution = check_gaussian_spatial_hessian(
            frequencies, np.ones(3), np.full(3, 5000), **kwargs
        )
    assert constant_width["max_order_zero_noise_amplification"] == pytest.approx(1)
    assert natural_resolution["max_order_zero_noise_amplification"] > 1
    assert natural_resolution["excess_noise_mode_count"] == 0
    assert natural_resolution["rank_deficient_mode_count"] > 0
    assert natural_resolution["sampling"] == "at_most_33_by_33_half_fourier_grid"


@pytest.mark.parametrize("channel_count", [3, 1025])
@pytest.mark.parametrize("scalar_baseline", [False, True])
def test_gaussian_channel_support_matches_explicit_moments(
    diagnostic_warnings, monkeypatch, channel_count, scalar_baseline
):
    frequencies = 299792458.0 * np.resize([1.0, 2.0, 3.0], channel_count)
    weights = np.resize([1.0, 2.0, 0.5], channel_count)
    baselines = 1.0 if scalar_baseline else np.resize([1.0, 0.5, 1.0], channel_count)
    original = (frequencies.copy(), weights.copy(), np.copy(baselines))
    captured = {}

    def inspect(transfer, first, second, **kwargs):
        captured.update(transfer=transfer.copy(), first=first, second=second)
        return check_spatial_hessian_blocks(transfer, first, second, **kwargs)

    monkeypatch.setattr(_spatial_hessian, "check_spatial_hessian_blocks", inspect)
    with diagnostic_warnings():
        result = check_gaussian_spatial_hessian(
            frequencies,
            weights,
            baselines,
            n_taylor_terms=2,
            reference_frequency=2 * 299792458.0,
            image_size=[8, 8],
            cell_size=[-0.125, 0.125],
            precision="float64",
        )
    # Independent scalar moment sums, including the per-channel hard cutoff.
    # The grid contains radius 1 and 3 exactly, testing inclusive boundaries.
    cutoff = np.broadcast_to(baselines, frequencies.shape) * frequencies / 299792458.0
    sigma = (1 / cutoff) / np.sqrt(8 * np.log(2))
    x = frequencies / (2 * 299792458.0) - 1
    expected = np.zeros_like(captured["transfer"])
    for i, u in enumerate(captured["first"]):
        for j, v in enumerate(captured["second"]):
            radius_squared = u**2 + v**2
            for channel in range(channel_count):
                if radius_squared <= cutoff[channel] ** 2:
                    response = (
                        weights[channel]
                        / weights.max()
                        * (sigma[channel] / sigma.max()) ** 2
                        * np.exp(-2 * np.pi**2 * sigma[channel] ** 2 * radius_squared)
                    )
                    for order in range(3):
                        expected[order, i, j] += response * x[channel] ** order
    np.testing.assert_allclose(captured["transfer"], expected, rtol=1e-12, atol=1e-15)
    radius_squared = captured["first"][:, None] ** 2 + captured["second"][None, :] ** 2
    outside = radius_squared > cutoff.max() ** 2
    assert result["outside_baseline_support_mode_count"] == int(outside.sum())
    assert result["unsupported_mode_count"] == int(outside.sum())
    assert (
        result["rank_deficient_mode_count"] > 0
    )  # Only the high-frequency channel remains.
    assert result["maximum_spatial_frequency_cycles_per_radian"] == [
        float(cutoff.min()),
        float(cutoff.max()),
    ]
    for examples in result["examples"].values():
        for mode in examples:
            assert np.hypot(mode["frequency_l"], mode["frequency_m"]) <= cutoff.max()
    np.testing.assert_array_equal(frequencies, original[0])
    np.testing.assert_array_equal(weights, original[1])
    np.testing.assert_array_equal(baselines, original[2])


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_gaussian_tails_outside_all_baselines_do_not_warn(
    diagnostic_warnings, precision
):
    with diagnostic_warnings() as caught:
        result = check_gaussian_spatial_hessian(
            299792458.0 * np.array([1.0, 2.0, 3.0]),
            np.ones(3),
            0.1,
            n_taylor_terms=2,
            reference_frequency=2 * 299792458.0,
            image_size=[8, 8],
            cell_size=[-0.125, 0.125],
            precision=precision,
        )
    assert result["supported_mode_count"] == result["assessed_mode_count"] == 1
    assert result["unsupported_mode_count"] == result["mode_count"] - 1
    assert (
        result["outside_baseline_support_mode_count"]
        == result["unsupported_mode_count"]
    )
    assert not caught


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


@pytest.mark.parametrize(
    "fault", ["dimensions", "even_orders", "empty", "complex", "nonfinite"]
)
def test_actual_psf_layout_rejected_before_fourier_transform(fault):
    psf = coupled_psfs()
    if fault == "dimensions":
        psf = psf[0]
    elif fault == "even_orders":
        psf = psf[:2]
    elif fault == "empty":
        psf = psf[:, :0]
    elif fault == "complex":
        psf = psf.astype(complex)
    else:
        psf[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite real stack"):
        check_psf_spatial_hessian(psf, precision="float64")


@pytest.mark.parametrize("cells", [[0, 1], [1], [np.inf, 1]])
def test_actual_psf_requires_finite_nonzero_two_axis_cells(cells):
    with pytest.raises(ValueError, match="cell sizes"):
        check_psf_spatial_hessian(coupled_psfs(), precision="float64", cell_size=cells)


def test_spatial_blocks_reject_an_unsupported_working_precision():
    with pytest.raises(ValueError, match="precision must"):
        check_spatial_hessian_blocks(
            np.ones((1, 1, 1)),
            [0],
            [0],
            precision="float16",
            frequency_units="cycles_per_pixel",
        )


@pytest.mark.parametrize(
    "frequencies,weights",
    [([[1e9]], [1]), ([1e9], [1, 1]), ([np.nan], [1]), ([1e9], [-1]), ([0], [1])],
)
def test_gaussian_requires_matching_positive_finite_spectral_samples(
    frequencies, weights
):
    with pytest.raises(ValueError, match="matching positive finite"):
        check_gaussian_spatial_hessian(
            frequencies,
            weights,
            1000,
            n_taylor_terms=1,
            reference_frequency=1e9,
            image_size=[8, 8],
            cell_size=[1e-5, 1e-5],
            precision="float64",
        )


@pytest.mark.parametrize(
    "size,cells",
    [
        ([1, 8], [1e-5, 1e-5]),
        ([8.0, 8.0], [1e-5, 1e-5]),
        ([8, 8], [0, 1e-5]),
        ([8, 8], [np.nan, 1e-5]),
    ],
)
def test_gaussian_requires_valid_integer_extents_and_angular_cells(size, cells):
    with pytest.raises(ValueError, match="two image sizes"):
        check_gaussian_spatial_hessian(
            np.array([1e9]),
            np.ones(1),
            1000,
            n_taylor_terms=1,
            reference_frequency=1e9,
            image_size=size,
            cell_size=cells,
            precision="float64",
        )
