"""Antenna beam models: analytic responses, Jones images, sampling, Mueller matrices."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.simulation.antenna_beams import (
    MAP_MUELLER_TO_JONES,
    airy_disk_response,
    apply_mueller,
    beam_model_kind,
    bilinear_interpolate,
    casa_airy_disk_response,
    dish_diameters_of_beam_models,
    evaluate_beam_models,
    make_airy_jones_beam,
    make_mueller_matrix,
    make_polynomial_jones_beam,
    make_zernike_jones_beam,
    pack_beam_models,
    polynomial_beam_response,
    resolve_beam_params,
    sample_jones,
)
from astroviper.processing_functions.simulation.zernike_polynomials import (
    N_ZERNIKE_TERMS,
    zernike_basis,
    zernike_surface,
)
from astroviper.utils.beam_models import (
    AIRY_DISK_MODELS,
    airy_disk_model,
    list_aperture_polynomial_coefficient_models,
    list_beam_polynomial_coefficient_models,
    normalize_beam_model_dict,
    normalize_beam_model_xds,
    read_aperture_polynomial_coefficients,
    read_beam_polynomial_coefficients,
)
from astroviper.utils.coordinate_transforms import directional_cosines
from tests.unit.processing_functions.simulation.legacy_fixtures import load_legacy


# ----------------------------------------------------------------------------
# shipped models
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,dish,blockage,max_rad",
    [
        ("vla", 24.5, 0.0, 0.014946999714079439),
        ("aca", 6.25, 0.75, 0.06227334771115768),
        ("alma", 10.7, 0.75, 0.03113667385557884),
        ("ngvla", 18.0, 0.0, 1.5 * np.pi / 180),
    ],
)
def test_airy_disk_models(name, dish, blockage, max_rad):
    """Legacy SIRIUS test_airy_disk values."""
    model = airy_disk_model(name)
    assert model["func"] == "casa_airy"
    assert model["dish_diameter"] == dish
    assert model["blockage_diameter"] == blockage
    assert np.isclose(model["max_rad_1GHz"], max_rad)
    assert airy_disk_model(name, func="airy")["func"] == "airy"
    assert AIRY_DISK_MODELS[name]["func"] == "casa_airy"  # copies, not mutated


def test_normalize_legacy_names():
    model = normalize_beam_model_dict(
        {"func": "airy", "dish_diam": 25.0, "blockage_diam": 2.0, "max_rad_1GHz": 0.01}
    )
    assert model == {
        "func": "airy",
        "dish_diameter": 25.0,
        "blockage_diameter": 2.0,
        "max_rad_1GHz": 0.01,
    }
    with pytest.raises(ValueError):
        normalize_beam_model_dict({"func": "airy"})
    legacy = xr.Dataset(
        {"J": (("pa", "chan", "pol", "l", "m"), np.zeros((1, 1, 2, 3, 3), complex))},
        coords={
            "pa": [0.0],
            "chan": [1e9],
            "pol": [5, 8],
            "l": np.arange(3.0),
            "m": np.arange(3.0),
        },
        attrs={"dish_diam": 25},
    )
    new = normalize_beam_model_xds(legacy)
    assert set(new.dims) == {"parallactic_angle", "frequency", "polarization", "l", "m"}
    assert list(new.polarization.values) == ["RR", "LL"]
    assert "JONES" in new and new.attrs["dish_diameter"] == 25
    assert beam_model_kind(new) == "jones_image"


def test_read_shipped_coefficient_models():
    assert (
        "EVLA_avg_zcoeffs_SBand_lookup" in list_aperture_polynomial_coefficient_models()
    )
    assert "EVLA_" in list_beam_polynomial_coefficient_models()
    zpc = read_aperture_polynomial_coefficients("EVLA_avg_zcoeffs_SBand_lookup")
    assert zpc.ZPC.dims == ("frequency", "polarization", "coefficient")
    assert zpc.sizes == {"frequency": 16, "polarization": 4, "coefficient": 66}
    assert list(zpc.polarization.values) == ["RR", "RL", "LR", "LL"]
    assert zpc.attrs["dish_diameter"] == 25.0 and zpc.attrs["telescope_name"] == "EVLA"
    # legacy zarr value (SIRIUS EVLA_avg_zcoeffs_SBand_lookup.apc.zarr): first coefficient, 2052 MHz, RR
    assert np.isclose(zpc.ZPC.values[0, 0, 0], 308.13023030 - 0.06968780j)
    assert np.isclose(zpc.ETA.values[0, 0, 0], 1.11)
    meerkat = read_aperture_polynomial_coefficients("MeerKAT_avg_zcoeffs_LBand_lookup")
    assert (
        meerkat.attrs["dish_diameter"] == 13.5
        and list(meerkat.polarization.values)[0] == "XX"
    )
    bpc = read_beam_polynomial_coefficients("EVLA_")
    assert bpc.BPC.dims == ("frequency", "polarization", "coefficient")
    assert bpc.sizes["frequency"] == 235 and bpc.sizes["coefficient"] == 5
    assert bpc.band.values[0] == "P" and bpc.band.values[-1] == "Q"
    np.testing.assert_allclose(
        bpc.BPC.values[0, 0], [1.0, -1.137e-3, 5.19e-7, -1.04e-10, 0.71e-14]
    )
    assert np.all(np.diff(bpc.frequency.values) > 0)
    with pytest.raises(FileNotFoundError):
        read_beam_polynomial_coefficients("nope")
    assert beam_model_kind(zpc) == "aperture_polynomial_coefficients"
    assert beam_model_kind(bpc) == "beam_polynomial_coefficients"
    np.testing.assert_allclose(
        dish_diameters_of_beam_models([zpc, bpc, airy_disk_model("aca")]),
        [25.0, 25.0, 6.25],
    )


# ----------------------------------------------------------------------------
# analytic responses (legacy SIRIUS test_beam_utils values)
# ----------------------------------------------------------------------------
def test_casa_airy_matches_legacy_scalar_values():
    lmn = directional_cosines(np.array([[2.1, 3.2]]))[0]
    val = casa_airy_disk_response(
        lmn[0], lmn[1], 1.2e9, 25.0, 2.0, 0.03113667385557884, 1
    )
    np.testing.assert_allclose(val, -0.00026466, rtol=1e-4)
    # at the pointing centre both responses are one
    assert casa_airy_disk_response(0.0, 0.0, 1.2e9, 25.0, 0.0, 0.03, 1) == 1.0
    assert airy_disk_response(0.0, 0.0, 1.2e9, 25.0, 0.0) == 1.0
    # power = voltage squared
    l, m = 1e-3, 2e-3  # noqa: E741
    v = airy_disk_response(l, m, 3e9, 25.0, 2.0, 1)
    p = airy_disk_response(l, m, 3e9, 25.0, 2.0, 2)
    np.testing.assert_allclose(p, v**2)
    # vectorised broadcasting over frequency
    out = casa_airy_disk_response(
        np.array([[1e-3], [2e-3]]), 0.0, np.array([[1e9, 2e9, 3e9]]), 25.0, 0.0, 0.015
    )
    assert out.shape == (2, 3)
    assert np.all(np.abs(out) <= 1.0)


def test_polynomial_beam_response():
    coef = np.array([1.0, -1.137e-3, 5.19e-7, -1.04e-10, 0.71e-14])
    assert polynomial_beam_response(0.0, 0.0, 1e9, coef, 0.015) == 1.0
    r = np.array([1e-3, 2e-3, 4e-3])
    v = polynomial_beam_response(r, 0.0, 1e9, coef, 0.015, 1, n_sample=None)
    p = polynomial_beam_response(r, 0.0, 1e9, coef, 0.015, 2, n_sample=None)
    np.testing.assert_allclose(p, v**2)
    assert np.all(np.diff(v) < 0)


def test_zernike_basis_and_surface():
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 4), indexing="ij")
    basis = zernike_basis(x, y)
    assert basis.shape == (N_ZERNIKE_TERMS, 5, 4)
    np.testing.assert_array_equal(basis[0], 1.0)
    np.testing.assert_allclose(basis[1], x)
    np.testing.assert_allclose(basis[2], y)
    np.testing.assert_allclose(basis[4], 2 * x**2 + 2 * y**2 - 1)
    c = np.zeros(66)
    c[4] = 2.0
    np.testing.assert_allclose(zernike_surface(c, x, y), 2 * basis[4])
    short = zernike_surface(np.array([1.0, 0.5]), x, y)
    np.testing.assert_allclose(short, 1.0 + 0.5 * x)
    cplx = zernike_surface(np.array([1j, 0.0]), x, y)
    assert cplx.dtype == np.complex128
    with pytest.raises(ValueError):
        zernike_surface(np.zeros(67), x, y)


# ----------------------------------------------------------------------------
# Jones images
# ----------------------------------------------------------------------------
def test_zernike_jones_beam_matches_legacy():
    f = load_legacy("evla_zernike_beam")
    zpc = f["beam_models"][0]
    jones = make_zernike_jones_beam(
        zpc, f["beam_pa"], f["beam_frequency"], f["beam_params"]
    )
    assert jones.JONES.dims == (
        "parallactic_angle",
        "frequency",
        "polarization",
        "l",
        "m",
    )
    assert list(jones.polarization.values) == ["RR", "RL", "LR", "LL"]
    np.testing.assert_allclose(jones.l.values, f["beam_l"])
    np.testing.assert_allclose(jones.m.values, f["beam_m"])
    np.testing.assert_allclose(
        jones.JONES.values[:, :, :, ::25, ::25], f["beam_jones_subsampled"], atol=1e-12
    )
    np.testing.assert_allclose(
        np.abs(jones.JONES.values).max(axis=(3, 4)), f["beam_jones_abs_max"], atol=1e-12
    )
    n = jones.sizes["l"] // 2
    np.testing.assert_allclose(
        jones.JONES.values[:, :, :, n, n], f["beam_jones_center"], atol=1e-12
    )
    # normalisation: mean of the diagonal peak amplitudes is one
    amax = np.abs(jones.JONES.values).max(axis=(3, 4))
    np.testing.assert_allclose((amax[:, :, 0] + amax[:, :, 3]) / 2, 1.0)
    assert (
        jones.attrs["model_type"] == "jones_image"
        and jones.attrs["dish_diameter"] == 25.0
    )


def test_zernike_jones_beam_diagonal_only():
    zpc = read_aperture_polynomial_coefficients("EVLA_avg_zcoeffs_SBand_lookup")
    jones = make_zernike_jones_beam(
        zpc, [0.0, 0.3], [3e9], {"image_size": [64, 64], "mueller_selection": [0, 15]}
    )
    assert jones.JONES.shape == (2, 1, 2, 64, 64)
    assert list(jones.polarization.values) == ["RR", "LL"]
    only_p = make_zernike_jones_beam(
        zpc, [0.0], [3e9], {"image_size": [64, 64], "mueller_selection": [0]}
    )
    assert list(only_p.polarization.values) == ["RR"]
    np.testing.assert_allclose(np.abs(only_p.JONES.values).max(), 1.0)
    with pytest.raises(ValueError):
        resolve_beam_params({"mueller_selection": [5]})


def test_airy_and_polynomial_jones_images():
    j_airy = make_airy_jones_beam(
        airy_disk_model("vla"), [3e9, 3.4e9], {"image_size": [32, 32]}
    )
    assert j_airy.JONES.shape == (1, 2, 2, 32, 32)
    assert list(j_airy.polarization.values) == ["RR", "LL"]
    np.testing.assert_allclose(j_airy.JONES.values[0, :, 0, 16, 16], 1.0)
    assert j_airy.l.values[0] > j_airy.l.values[-1]  # l decreases with pixel index
    bpc = read_beam_polynomial_coefficients("EVLA_")
    j_poly = make_polynomial_jones_beam(
        bpc, [3e9], {"image_size": [32, 32]}, polarization=("XX", "YY")
    )
    assert j_poly.JONES.shape == (1, 1, 2, 32, 32)
    assert list(j_poly.polarization.values) == ["XX", "YY"]
    np.testing.assert_allclose(j_poly.JONES.values[0, 0, 0, 16, 16], 1.0)
    assert np.all(np.abs(j_poly.JONES.values) <= 1.0 + 1e-12)


def test_make_mueller_matrix():
    zpc = read_aperture_polynomial_coefficients("EVLA_avg_zcoeffs_SBand_lookup")
    jones = make_zernike_jones_beam(
        zpc, [0.0], [3e9], {"image_size": [32, 32], "mueller_selection": np.arange(16)}
    )
    mueller = make_mueller_matrix(jones, jones, [0, 5, 10, 15, 1])
    assert mueller.MUELLER.dims == (
        "parallactic_angle",
        "frequency",
        "mueller_element",
        "l",
        "m",
    )
    assert list(mueller.mueller_element.values) == [0, 5, 10, 15, 1]
    # element f -> (a, b) = MAP_MUELLER_TO_JONES[f]: 0->(0,0), 5->(0,3), 10->(3,0), 15->(3,3), 1->(0,1)
    assert list(mueller.polarization_1.values) == ["RR", "RR", "LL", "LL", "RR"]
    assert list(mueller.polarization_2.values) == ["RR", "LL", "RR", "LL", "RL"]
    j = jones.JONES.values[0, 0]
    np.testing.assert_allclose(mueller.MUELLER.values[0, 0, 0], j[0] * np.conj(j[0]))
    np.testing.assert_allclose(mueller.MUELLER.values[0, 0, 4], j[0] * np.conj(j[1]))


# ----------------------------------------------------------------------------
# evaluation / packing / sampling
# ----------------------------------------------------------------------------
def test_evaluate_beam_models_without_zernike_has_zero_parallactic_angle():
    f = load_legacy("vla_airy")
    models, pa = evaluate_beam_models(
        [airy_disk_model("vla")],
        f["time"],
        f["frequency"],
        f["phase_center_ra_dec"],
        f["site_position"],
    )
    np.testing.assert_array_equal(pa, 0.0)
    assert models[0]["func"] == "casa_airy"
    packed = pack_beam_models(models)
    assert packed[0]["kind"] == "analytic"


def test_evaluate_beam_models_with_zernike():
    f = load_legacy("evla_mixed_beams")
    models, pa = evaluate_beam_models(
        f["beam_models"], f["time"], f["frequency"], f["phase_center_ra_dec"], f["site_position"],
        f["beam_params"], direction_frame="fk5",
    )  # fmt: skip
    np.testing.assert_allclose(pa, f["parallactic_angle"], atol=1e-12)
    assert (
        beam_model_kind(models[0]) == "jones_image" and models[1]["func"] == "casa_airy"
    )
    packed = pack_beam_models(models)
    assert packed[0]["kind"] == "jones_image" and packed[0]["jones"].ndim == 5
    assert packed[0]["cell_size_l"] < 0 < packed[0]["cell_size_m"]
    with pytest.raises(ValueError):
        pack_beam_models(f["beam_models"])  # unevaluated Zernike model


def test_bilinear_interpolate_vectorised():
    """The legacy gather was only valid for one sample; the port must be exact for many."""
    image = np.arange(20.0).reshape(4, 5)
    x = np.array([0.0, 1.5, 2.0, 3.0, -2.0, 10.0])
    y = np.array([0.0, 2.5, 4.0, 0.25, 0.0, 4.0])
    out = bilinear_interpolate(image, x, y)
    np.testing.assert_allclose(out[0], image[0, 0])
    np.testing.assert_allclose(
        out[1], 0.25 * (image[1, 2] + image[1, 3] + image[2, 2] + image[2, 3])
    )
    np.testing.assert_allclose(out[2], image[2, 4])
    np.testing.assert_allclose(out[3], 0.75 * image[3, 0] + 0.25 * image[3, 1])
    np.testing.assert_allclose(out[4], image[0, 0])  # clamped
    np.testing.assert_allclose(out[5], image[3, 4])  # clamped
    stacked = bilinear_interpolate(np.stack([image, 2 * image]), x, y)
    np.testing.assert_allclose(stacked[1], 2 * out)


def test_sample_jones_image_reproduces_grid_values():
    """Sampling a Jones image at its own pixel centres returns the pixel values (correct l/m axes)."""
    zpc = read_aperture_polynomial_coefficients("EVLA_avg_zcoeffs_SBand_lookup")
    jones = make_zernike_jones_beam(
        zpc, [0.4], [3e9], {"image_size": [64, 64], "mueller_selection": np.arange(16)}
    )
    packed = pack_beam_models([jones])[0]
    il, im = 20, 40
    lm = np.array([[jones.l.values[il], jones.m.values[im]]])
    sampled = sample_jones(packed, lm, np.array([3e9]), 0.4)[0, 0]
    np.testing.assert_allclose(sampled, jones.JONES.values[0, 0, :, il, im], atol=1e-12)
    # beyond the cut radius the Jones vector is zero
    far = sample_jones(packed, np.array([[0.05, 0.0]]), np.array([3e9]), 0.4)
    np.testing.assert_array_equal(far, 0.0)


def test_sample_jones_analytic_and_polynomial():
    packed = pack_beam_models(
        [airy_disk_model("vla"), read_beam_polynomial_coefficients("EVLA_")]
    )
    lm = np.array([[0.0, 0.0], [1e-3, 0.0], [0.02, 0.0]])
    j = sample_jones(packed[0], lm, np.array([3e9, 3.4e9]), 0.0)
    assert j.shape == (3, 2, 4)
    np.testing.assert_allclose(j[0, :, [0, 3]], 1.0)
    np.testing.assert_array_equal(j[:, :, [1, 2]], 0.0)
    assert np.all(np.abs(j[1, :, 0]) < 1.0)
    np.testing.assert_array_equal(j[2], 0.0)  # outside max_rad at 3 GHz
    jp = sample_jones(packed[1], lm, np.array([3e9]), 0.0)
    np.testing.assert_allclose(jp[0, 0, [0, 3]], 1.0)
    assert 0 < np.abs(jp[1, 0, 0]) < 1.0
    with pytest.raises(ValueError):
        sample_jones({"kind": "bogus", "max_rad_1GHz": 1.0}, lm, np.array([3e9]), 0.0)


def test_apply_mueller():
    j1 = np.array([1.0, 0.1j, 0.0, 0.5])
    j2 = np.array([0.8, 0.0, 0.2, 1.0])
    flux = np.array([1.0, 0.3, 0.3, 2.0])
    full = apply_mueller(j1, j2, flux, np.arange(16))
    mueller = np.zeros((4, 4), complex)
    for f in range(16):
        a, b = MAP_MUELLER_TO_JONES[f]
        mueller[f // 4, f % 4] = j1[a] * np.conj(j2[b])
    np.testing.assert_allclose(full, mueller @ flux)
    diag = apply_mueller(j1, j2, flux, [0, 5, 10, 15])
    np.testing.assert_allclose(diag, np.diag(np.diag(mueller)) @ flux)
    # broadcasting over leading axes
    many = apply_mueller(np.tile(j1, (3, 2, 1)), j2, flux, [0, 15])
    assert many.shape == (3, 2, 4)
