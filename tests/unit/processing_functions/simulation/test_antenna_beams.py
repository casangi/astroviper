"""Antenna beam models: analytic responses, Jones images, sampling, Mueller matrices."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.simulation.antenna_beams import (
    MAP_MUELLER_TO_JONES,
    SPEED_OF_LIGHT,
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
    symmetric_beam_response,
)
from astroviper.processing_functions.simulation.zernike_polynomials import (
    N_ZERNIKE_TERMS,
    zernike_basis,
    zernike_surface,
)
from astroviper.utils.beam_models import (
    AIRY_DISK_MODELS,
    airy_disk_model,
    beam_polynomial_coefficient_index,
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
        ("vla_nvss", 24.5, 0.0, 0.014946999714079439),
        ("aca", 6.25, 0.75, 0.06227334771115768),
        ("alma", 10.7, 0.75, 0.03113667385557884),
        ("ngvla", 18.0, 0.0, 1.5 * np.pi / 180),
    ],
)
def test_airy_disk_models(name, dish, blockage, max_rad):
    """Legacy SIRIUS test_airy_disk values, those of CASA's Airy disk; by
    default the textbook Airy disk of the same diameters, without a cut."""
    model = airy_disk_model(name)
    assert model["func"] == "airy"
    assert model["dish_diameter"] == dish
    assert model["blockage_diameter"] == blockage
    assert model["max_rad_1GHz"] is None
    casa = airy_disk_model(name, func="casa_airy")
    assert casa == {**model, "func": "casa_airy", "max_rad_1GHz": casa["max_rad_1GHz"]}
    assert np.isclose(casa["max_rad_1GHz"], max_rad)
    assert airy_disk_model(name, func="none")["max_rad_1GHz"] == casa["max_rad_1GHz"]
    assert AIRY_DISK_MODELS[name]["func"] == "airy"  # copies, not mutated


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
    for func in ("casa_airy", "none"):
        with pytest.raises(ValueError, match="needs max_rad_1GHz"):
            normalize_beam_model_dict(
                {"func": func, "dish_diameter": 25.0, "max_rad_1GHz": None}
            )
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
    assert [b[0] for b in bpc.attrs["band_definitions"]] == list("PLSCXUKAQ")
    assert bpc.attrs["band_definitions"][0] == ["P", 224e6, 480e6]
    np.testing.assert_allclose(
        bpc.BPC.values[0, 0], [1.0, -1.137e-3, 5.19e-7, -1.04e-10, 0.71e-14]
    )
    assert np.all(np.diff(bpc.frequency.values) > 0)
    assert "n_sample" not in bpc.attrs
    casa_lookup = read_beam_polynomial_coefficients("EVLA_", n_sample=10000)
    assert casa_lookup.attrs["n_sample"] == 10000
    for n_sample in (0, 1, -2):
        with pytest.raises(ValueError, match="n_sample"):
            read_beam_polynomial_coefficients("EVLA_", n_sample=n_sample)
        with pytest.raises(ValueError, match="n_sample"):
            pack_beam_models([casa_lookup.assign_attrs(n_sample=n_sample)])
    with pytest.raises(FileNotFoundError):
        read_beam_polynomial_coefficients("nope")
    assert beam_model_kind(zpc) == "aperture_polynomial_coefficients"
    assert beam_model_kind(bpc) == "beam_polynomial_coefficients"
    np.testing.assert_allclose(
        dish_diameters_of_beam_models([zpc, bpc, airy_disk_model("aca")]),
        [25.0, 25.0, 6.25],
    )


@pytest.mark.parametrize(
    "frequency, table_frequency_mhz",
    [
        (33e9, 34052),  # a tie goes to the higher entry
        (3e9, 3052),
        (6e9, 6052),
        (2.01e9, 2052),  # S band, clamped to its first entry
        (4000.5e6, 3948),  # S band (listed before C), clamped to its last entry
        (45e9, 43948),  # Q band, clamped
        (60e9, 43948),  # no band
        (1.5e9, 1488),
        (100e6, 232),
    ],
)
def test_beam_polynomial_coefficient_index_is_casa_pick(frequency, table_frequency_mhz):
    bpc = read_beam_polynomial_coefficients("EVLA_")
    index = beam_polynomial_coefficient_index(bpc, frequency)
    assert bpc.frequency.values[index] == table_frequency_mhz * 1e6


def test_beam_polynomial_default_radius_is_that_of_casa_evla():
    """The EVLA beam polynomial has CASA's 58 arcmin table at 1 GHz
    (PBMath1DEVLA); the EVLA Zernike models keep 0.8564 deg."""
    assert read_beam_polynomial_coefficients("EVLA_").attrs[
        "max_rad_1GHz"
    ] == pytest.approx(np.radians(58.0 / 60.0), rel=1e-15)
    assert read_aperture_polynomial_coefficients("EVLA_avg_zcoeffs_SBand_lookup").attrs[
        "max_rad_1GHz"
    ] == pytest.approx(np.radians(0.8564), rel=1e-15)


def test_beam_polynomial_coefficient_index_of_the_table():
    bpc = read_beam_polynomial_coefficients("EVLA_")
    np.testing.assert_array_equal(
        beam_polynomial_coefficient_index(bpc, bpc.frequency.values),
        np.arange(bpc.sizes["frequency"]),
    )
    # without band definitions nothing is clamped
    bpc.attrs.pop("band_definitions")
    index = beam_polynomial_coefficient_index(bpc, [2.01e9, 3e9])
    np.testing.assert_array_equal(bpc.frequency.values[index], [2000e6, 3052e6])


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
        airy_disk_model("vla_nvss"), [3e9, 3.4e9], {"image_size": [32, 32]}
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


def test_polynomial_jones_image_uses_the_casa_coefficients():
    """At 33 GHz, a tie of 31948 and 34052 MHz, the image uses 34052 MHz."""
    bpc = read_beam_polynomial_coefficients("EVLA_")
    j_poly = make_polynomial_jones_beam(bpc, [33e9], {"image_size": [32, 32]})
    l, m = j_poly.l.values[20], j_poly.m.values[16]  # noqa: E741

    def voltage(table_frequency, n_sample=None):
        coefficients = bpc.BPC.sel(frequency=table_frequency).values[0]
        return polynomial_beam_response(
            l, m, 33e9, coefficients, bpc.attrs["max_rad_1GHz"], 1, n_sample=n_sample
        )

    sampled = j_poly.JONES.values[0, 0, 0, 20, 16]
    np.testing.assert_allclose(sampled, voltage(34052e6), rtol=1e-12)
    assert not np.isclose(sampled, voltage(31948e6), rtol=1e-6)
    # with an n_sample attr, on CASA's lookup table as the samplers
    j_lookup = make_polynomial_jones_beam(
        read_beam_polynomial_coefficients("EVLA_", n_sample=10000),
        [33e9],
        {"image_size": [32, 32]},
    )
    sampled = j_lookup.JONES.values[0, 0, 0, 20, 16]
    np.testing.assert_allclose(sampled, voltage(34052e6, 10000), rtol=1e-12)
    assert not np.isclose(sampled, voltage(34052e6), rtol=1e-9)


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
        [airy_disk_model("vla_nvss")],
        f["time"],
        f["frequency"],
        f["phase_center_ra_dec"],
        f["site_position"],
    )
    np.testing.assert_array_equal(pa, 0.0)
    assert models[0]["func"] == "airy"
    packed = pack_beam_models(models)
    assert packed[0]["kind"] == "analytic"


def test_evaluate_beam_models_resolves_beam_polynomials_to_the_channels():
    f = load_legacy("vla_airy")
    bpc = read_beam_polynomial_coefficients("EVLA_")
    frequency = np.array([33e9, 2.01e9, 3e9])
    models, _ = evaluate_beam_models(
        [bpc], f["time"], frequency, f["phase_center_ra_dec"], f["site_position"]
    )
    np.testing.assert_array_equal(models[0].frequency.values, frequency)
    np.testing.assert_array_equal(
        models[0].BPC.values,
        bpc.BPC.sel(frequency=[34052e6, 2052e6, 3052e6]).values,
    )
    assert models[0].attrs == bpc.attrs
    packed = pack_beam_models(models)[0]
    np.testing.assert_array_equal(packed["frequency"], frequency)


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
        [
            airy_disk_model("vla_nvss", func="casa_airy"),
            read_beam_polynomial_coefficients("EVLA_"),
        ]
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


def test_symmetric_beam_response_is_the_sampled_voltage_pattern():
    packed = pack_beam_models(
        [
            airy_disk_model("vla_nvss", func="casa_airy"),
            read_beam_polynomial_coefficients("EVLA_"),
        ]
    )
    lm = np.array([[0.0, 0.0], [1e-3, 0.0], [0.02, 0.0]])
    frequency = np.array([3e9, 3.4e9])
    for model in packed:
        voltage = symmetric_beam_response(model, lm[:, :1], lm[:, 1:], frequency)
        np.testing.assert_array_equal(
            sample_jones(model, lm, frequency, 0.0)[:, :, 0], voltage
        )
        power = symmetric_beam_response(model, lm[:, :1], lm[:, 1:], frequency, 2)
        np.testing.assert_allclose(power, voltage**2, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(power[2], 0.0)  # beyond max_rad at 3 GHz
    with pytest.raises(ValueError, match="not azimuthally symmetric"):
        symmetric_beam_response({"kind": "jones_image"}, 0.0, 0.0, 3e9)


def test_symmetric_beam_cut_is_at_sqrt_of_l2_plus_m2():
    """The beam is zero where sqrt(l * l + m * m), the radius the C++ kernel
    computes, reaches the cut, and not one ulp inside it."""
    packed = pack_beam_models([airy_disk_model("alma", func="casa_airy")])[0]
    frequency = 100e9
    cut = packed["max_rad_1GHz"] / (frequency / 1e9)
    angle = np.linspace(0.0, 2.0 * np.pi, 1000)
    # sqrt(fl(x * x)) == |x| in binary64, so the last two points lie on and
    # one ulp inside the cut on every platform.
    l = np.append(cut * np.cos(angle), [cut, np.nextafter(cut, 0.0)])  # noqa: E741
    m = np.append(cut * np.sin(angle), [0.0, 0.0])
    outside = np.sqrt(l * l + m * m) >= cut
    np.testing.assert_array_equal(outside[-2:], [True, False])
    voltage = symmetric_beam_response(packed, l, m, frequency)
    np.testing.assert_array_equal(voltage == 0.0, outside)
    np.testing.assert_array_equal(
        sample_jones(packed, np.stack([l, m], axis=-1), np.array([frequency]), 0.0)[
            :, 0, 0
        ],
        voltage,
    )


@pytest.mark.parametrize("name", ["vla_nvss", "aca", "alma", "ngvla"])
def test_default_airy_disk_has_no_cut(name):
    """The textbook Airy disk is sampled exactly, also its sidelobes beyond
    CASA's table radius."""
    model = airy_disk_model(name)
    packed = pack_beam_models([model])[0]
    assert packed["func"] == "airy" and packed["max_rad_1GHz"] == np.inf
    frequency = np.array([3e9, 100e9])
    cut = airy_disk_model(name, func="casa_airy")["max_rad_1GHz"] / (
        frequency.min() / 1e9
    )
    lm = np.array([[0.0, 0.0], [0.3 * cut, 0.0], [0.0, 1.5 * cut], [3 * cut, cut]])
    voltage = airy_disk_response(
        lm[:, :1],
        lm[:, 1:],
        frequency,
        model["dish_diameter"],
        model["blockage_diameter"],
    )
    np.testing.assert_array_equal(
        symmetric_beam_response(packed, lm[:, :1], lm[:, 1:], frequency), voltage
    )
    np.testing.assert_array_equal(
        sample_jones(packed, lm, frequency, 0.0)[:, :, [0, 3]],
        np.stack([voltage, voltage], axis=-1),
    )
    assert np.all(voltage[2:] != 0.0)


def test_beam_polynomial_is_exact_unless_it_has_n_sample():
    """The EVLA polynomial at the exact radius, or on CASA's lookup table with
    n_sample, zero beyond its 58 arcmin table either way."""
    lm = np.array([[1e-3, 2e-3], [3e-3, 0.0], [0.0, 0.007]])
    frequency = np.array([3e9])
    l, m = lm[:, :1], lm[:, 1:]  # noqa: E741
    for n_sample in (None, 10000):
        bpc = read_beam_polynomial_coefficients("EVLA_", n_sample=n_sample)
        packed = pack_beam_models(
            evaluate_beam_models(
                [bpc], np.zeros(1), frequency, np.zeros((1, 2)), np.zeros(3)
            )[0]
        )[0]
        assert packed["n_sample"] == n_sample
        expected = polynomial_beam_response(
            l,
            m,
            frequency,
            packed["coefficients"][0],
            packed["max_rad_1GHz"],
            1,
            n_sample,
        )
        expected[2] = 0.0  # beyond 58 arcmin at 3 GHz
        np.testing.assert_array_equal(
            symmetric_beam_response(packed, l, m, frequency), expected
        )
        np.testing.assert_array_equal(
            sample_jones(packed, lm, frequency, 0.0)[:, :, 0], expected
        )
    assert not np.isclose(
        symmetric_beam_response(packed, l[:2], m[:2], frequency),
        polynomial_beam_response(
            l[:2],
            m[:2],
            frequency,
            packed["coefficients"][0],
            packed["max_rad_1GHz"],
            1,
            None,
        ),
        rtol=1e-9,
    ).any()


def test_uncut_airy_jones_image_is_sized_on_the_unblocked_first_null():
    model = airy_disk_model("alma")
    jones = make_airy_jones_beam(model, [100e9], {"image_size": [32, 32]})
    first_null = 3.8317059702075125 / (
        np.pi * 100e9 / SPEED_OF_LIGHT * model["dish_diameter"]
    )
    np.testing.assert_allclose(np.diff(jones.m.values), first_null / 32 * 4.0)
    np.testing.assert_array_equal(
        jones.JONES.values[0, 0, 0],
        airy_disk_response(
            jones.l.values[:, None], jones.m.values[None, :], 100e9, 10.7, 0.75
        ),
    )
    # zero, not the edge of the image, beyond the circle inscribed in it
    edge = jones.m.values[-1]
    np.testing.assert_allclose(jones.attrs["max_rad_1GHz"], edge * 100.0, rtol=1e-15)
    packed = pack_beam_models([jones])[0]
    lm = np.array([[0.0, 0.9 * edge], [0.0, 1.01 * edge], [-20 * edge, 0.0]])
    voltage = sample_jones(packed, lm, np.array([100e9, 110e9]), 0.0)[:, :, 0]
    assert np.all(voltage[0] != 0.0) and np.all(voltage[1:] == 0.0)


@pytest.mark.parametrize("casa", [False, True])
@pytest.mark.parametrize(
    "name, frequency",
    [
        ("alma", (90e9, 230e9)),
        ("aca", (90e9, 230e9)),
        ("ngvla", (30e9, 93e9)),
        ("vla_nvss", (1.5e9, 3e9)),
        ("EVLA_", (2.01e9, 3e9)),
    ],
)
def test_jones_image_is_the_simulated_beam(name, frequency, casa):
    """The Jones image of a named model, or of its CASA variant, is
    symmetric_beam_response at its pixel coordinates, zero beyond the cut."""
    frequency = np.array(frequency)
    if name == "EVLA_":
        model = read_beam_polynomial_coefficients(
            name, n_sample=10000 if casa else None
        )
        jones = make_polynomial_jones_beam(model, frequency, {"image_size": [64, 64]})
    else:
        model = airy_disk_model(name, func="casa_airy" if casa else None)
        jones = make_airy_jones_beam(model, frequency, {"image_size": [64, 64]})
    packed = pack_beam_models(
        evaluate_beam_models(
            [model], np.zeros(1), frequency, np.zeros((1, 2)), np.zeros(3)
        )[0]
    )[0]
    for i, f in enumerate(frequency):
        expected = symmetric_beam_response(
            packed, jones.l.values[:, None], jones.m.values[None, :], f
        )
        assert (expected == 0.0).any() == (casa or name == "EVLA_")
        for p in range(jones.sizes["polarization"]):
            np.testing.assert_array_equal(jones.JONES.values[0, i, p], expected)


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
