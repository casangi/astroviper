"""Unit tests for the single-field primary-beam processing function.

Exercises
:func:`astroviper.processing_functions.imaging.primary_beam.make_primary_beam.make_primary_beam_single_field`
and the underlying azimuthally-symmetric Airy-disk models. The tests build a
minimal in-memory image dataset (no visibility data or downloaded files are
required) and check the contract of the produced ``PRIMARY_BEAM`` variable.
"""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric import (
    airy_disk_rorder,
    airy_disk_rorder_v2,
)
from astroviper.processing_functions.imaging.primary_beam.make_primary_beam import (
    make_primary_beam_single_field,
    resolve_primary_beam_params,
)
from astroviper.utils.beam_models import airy_disk_model

ARCSEC = np.pi / (180 * 3600)


def _make_empty_residual_image(
    image_size=(64, 64),
    cell_arcsec=2.0,
    frequency_coords=(1.0e11, 1.1e11),
    pol_coords=("I",),
):
    """Build a minimal empty image dataset with an empty ``residual`` data group.

    Mirrors what the node task / setup do before the primary beam is created.
    """
    from xradio.image import make_empty_sky_image

    cell = np.array([-cell_arcsec, cell_arcsec]) * ARCSEC
    img_xds = make_empty_sky_image(
        phase_center=np.array([0.0, 0.5]),
        image_size=list(image_size),
        cell_size=cell,
        frequency_coords=np.array(frequency_coords, dtype=float),
        pol_coords=list(pol_coords),
        time_coords=[0],
        do_sky_coords=False,
    )
    img_xds.attrs["type"] = "image_dataset"
    img_xds = img_xds.xr_img.add_data_group(
        new_data_group_name="residual",
        new_data_group={"description": "test", "date": "2026"},
    )
    image_params = {"image_size": list(image_size), "cell_size": cell}
    return img_xds, image_params


def test_make_primary_beam_single_field_basic():
    """PRIMARY_BEAM is created with the right shape, dims, attrs and data group."""
    img_xds, image_params = _make_empty_residual_image()

    img_xds, return_df = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
    )

    assert "PRIMARY_BEAM" in img_xds.data_vars
    pb = img_xds["PRIMARY_BEAM"]
    assert pb.dims == ("time", "frequency", "polarization", "l", "m")
    assert pb.shape == (1, 2, 1, 64, 64)
    assert pb.attrs["type"] == "primary_beam"
    assert pb.attrs["method"] == "airy_disk"

    # Registered in the data group.
    assert img_xds.attrs["data_groups"]["residual"]["primary_beam"] == "PRIMARY_BEAM"

    data = pb.values
    assert np.all(np.isfinite(data))
    # The (power) beam peaks at the image centre with value 1.0.
    for f in range(pb.sizes["frequency"]):
        plane = data[0, f, 0]
        assert plane[32, 32] == pytest.approx(1.0, abs=1e-6)
        assert np.argmax(plane) == 32 * plane.shape[1] + 32

    assert "T_primary_beam" in return_df.columns
    assert float(return_df["T_primary_beam"].iloc[0]) >= 0.0


def test_make_primary_beam_single_field_dtype():
    """float_dtype controls the output precision."""
    img_xds, image_params = _make_empty_residual_image()
    img_xds, _ = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
        float_dtype=np.float32,
    )
    assert img_xds["PRIMARY_BEAM"].dtype == np.float32


def test_make_primary_beam_single_field_noop_when_present():
    """A second call is a no-op when the data group already has a primary beam."""
    img_xds, image_params = _make_empty_residual_image()
    img_xds, _ = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
    )
    first = img_xds["PRIMARY_BEAM"].values.copy()

    img_xds, return_df = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
    )
    assert float(return_df["T_primary_beam"].iloc[0]) == 0.0
    assert np.array_equal(img_xds["PRIMARY_BEAM"].values, first)


def test_make_primary_beam_single_field_symmetry():
    """The beam is symmetric about the image centre."""
    img_xds, image_params = _make_empty_residual_image(
        image_size=(65, 65), frequency_coords=(1.0e11,)
    )
    img_xds, _ = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
    )
    plane = img_xds["PRIMARY_BEAM"].values[0, 0, 0]
    center = 32
    np.testing.assert_allclose(
        plane[center, :center], plane[center, center + 1 :][::-1], atol=1e-6
    )
    np.testing.assert_allclose(
        plane[:center, center], plane[center + 1 :, center][::-1], atol=1e-6
    )


def test_make_primary_beam_single_field_frequency_scaling():
    """Lower frequency gives a wider beam (higher value away from centre)."""
    img_xds, image_params = _make_empty_residual_image(
        frequency_coords=(1.0e11, 1.2e11)
    )
    img_xds, _ = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
    )
    pb = img_xds["PRIMARY_BEAM"].values
    # An off-centre pixel: the lower-frequency (wider) beam is larger there.
    assert pb[0, 0, 0, 32, 20] > pb[0, 1, 0, 32, 20]


def test_primary_beam_is_casa_power_pattern():
    """PRIMARY_BEAM follows the CASA definition: for the CASA variant, the
    shared CASA-compatible Airy voltage pattern squared (power), identical to
    the simulation code."""
    from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
        casa_airy_disk_response,
    )
    from astroviper.processing_functions.simulation.antenna_beams import (
        casa_airy_disk_response as simulation_casa_airy_disk_response,
    )

    assert simulation_casa_airy_disk_response is casa_airy_disk_response

    img_xds, image_params = _make_empty_residual_image()
    img_xds, _ = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
        beam_model=airy_disk_model("alma", func="casa_airy"),
    )
    voltage = casa_airy_disk_response(
        img_xds.l.values[:, None],
        img_xds.m.values[None, :],
        img_xds.frequency.values[0],
        10.7,
        0.75,
        0.03113667385557884,
        ipower=1,
    )
    outside = np.hypot(
        img_xds.l.values[:, None], img_xds.m.values[None, :]
    ) >= 0.03113667385557884 / (img_xds.frequency.values[0] / 1e9)
    np.testing.assert_array_equal(
        img_xds["PRIMARY_BEAM"].values[0, 0, 0], np.where(outside, 0.0, voltage**2)
    )


def test_primary_beam_is_the_airy_disk_inside_max_rad_and_zero_beyond():
    """CASA's ALMA beam is its Airy power pattern, as before the cut, bit for bit
    inside max_rad_1GHz / (frequency / 1 GHz) and zero beyond it, as CASA's .pb."""
    from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
        casa_airy_disk_response,
    )

    casa = airy_disk_model("alma", func="casa_airy")
    max_rad = casa["max_rad_1GHz"]
    img_xds, image_params = _make_empty_residual_image(
        frequency_coords=(1.0e11, 2.3e11, 9.5e11)
    )
    img_xds, _ = make_primary_beam_single_field(img_xds, image_params, beam_model=casa)
    l_axis = img_xds.l.values[None, :, None]
    m_axis = img_xds.m.values[None, None, :]
    frequency = img_xds.frequency.values[:, None, None]
    uncut = casa_airy_disk_response(
        l_axis, m_axis, frequency, 10.7, 0.75, max_rad, ipower=2
    )
    outside = np.hypot(l_axis, m_axis) >= max_rad / (frequency / 1e9)
    assert outside.any(axis=(1, 2)).all() and not outside.all(axis=(1, 2)).any()
    np.testing.assert_array_equal(
        img_xds["PRIMARY_BEAM"].values[0, :, 0], np.where(outside, 0.0, uncut)
    )


@pytest.mark.parametrize(
    "telescope, frequency_coords, cell_arcsec",
    [
        ("alma", (1.0e11, 2.3e11), 2.0),
        ("aca", (1.0e11, 2.3e11), 4.0),
        ("ngvla", (3.0e10, 9.3e10), 4.0),
    ],
)
def test_default_primary_beam_is_the_uncut_textbook_airy_disk(
    telescope, frequency_coords, cell_arcsec
):
    """The default beam of a telescope is the textbook Airy power pattern, bit
    for bit, with its first sidelobe beyond CASA's table radius."""
    from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
        airy_disk_response,
    )

    img_xds = _primary_beam(
        {"telescope": telescope},
        frequency_coords=frequency_coords,
        cell_arcsec=cell_arcsec,
    )
    model = airy_disk_model(telescope)
    l_axis = img_xds.l.values[None, :, None]
    m_axis = img_xds.m.values[None, None, :]
    frequency = img_xds.frequency.values[:, None, None]
    expected = airy_disk_response(
        l_axis,
        m_axis,
        frequency,
        model["dish_diameter"],
        model["blockage_diameter"],
        ipower=2,
    )
    pb = img_xds["PRIMARY_BEAM"].values[0, :, 0]
    np.testing.assert_array_equal(pb, expected)
    cut = airy_disk_model(telescope, func="casa_airy")["max_rad_1GHz"] / (
        frequency / 1e9
    )
    beyond = np.hypot(l_axis, m_axis) >= cut
    assert beyond.any(axis=(1, 2)).all()
    assert (pb[np.broadcast_to(beyond, pb.shape)] > 1e-3).any()


def test_airy_disk_rorder_versions_agree():
    """The reference and fast Airy-disk implementations produce the same beam."""
    freq_chan = np.array([1.0e11, 1.1e11])
    pol = ["I"]
    pb_params = {
        "list_dish_diameters": np.array([10.7]),
        "list_blockage_diameters": np.array([0.75]),
        "ipower": 1,
    }
    grid_params = {
        "cell_size": np.array([-2.0, 2.0]) * ARCSEC,
        "image_size": (64, 64),
        "image_center": (32, 32),
    }
    pb_v1 = airy_disk_rorder(freq_chan, pol, pb_params, grid_params)
    pb_v2 = airy_disk_rorder_v2(freq_chan, pol, pb_params, grid_params)
    assert pb_v1.shape == pb_v2.shape
    np.testing.assert_allclose(pb_v1, pb_v2, atol=1e-4)


def _primary_beam(
    primary_beam_params, frequency_coords=(1.0e9, 1.5e9), cell_arcsec=80.0
):
    img_xds, image_params = _make_empty_residual_image(
        cell_arcsec=cell_arcsec, frequency_coords=frequency_coords
    )
    img_xds, _ = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
        **resolve_primary_beam_params(primary_beam_params),
    )
    return img_xds


def test_primary_beam_params_none_and_alma_are_the_default_beam():
    """primary_beam_params None and {"telescope": "alma"} give today's beam bit for bit."""
    img_xds, image_params = _make_empty_residual_image(
        cell_arcsec=80.0, frequency_coords=(1.0e9, 1.5e9)
    )
    img_xds, _ = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
    )
    default = img_xds["PRIMARY_BEAM"].values
    assert resolve_primary_beam_params(None) == {}
    for params in (None, {"telescope": "alma"}, {"telescope": "ALMA"}):
        np.testing.assert_array_equal(
            _primary_beam(params)["PRIMARY_BEAM"].values, default
        )


def test_primary_beam_params_vla_beam():
    """{"telescope": "vla"} is the EVLA beam polynomial: the coefficients CASA
    picks (34052 MHz at the 33 GHz tie, 43948 MHz in Q band at 45 GHz) at the
    exact radius, and zero beyond 58 arcmin at 1 GHz; with n_sample, on CASA's
    lookup table."""
    from astroviper.processing_functions.simulation.antenna_beams import (
        polynomial_beam_response,
    )
    from astroviper.utils.beam_models import (
        EVLA_BEAM_POLYNOMIAL_MAX_RAD_1GHZ,
        read_beam_polynomial_coefficients,
    )

    assert EVLA_BEAM_POLYNOMIAL_MAX_RAD_1GHZ == pytest.approx(
        58.0 / 60.0 * np.pi / 180.0, rel=1e-15
    )
    bpc = read_beam_polynomial_coefficients("EVLA_")
    assert bpc.attrs["max_rad_1GHz"] == EVLA_BEAM_POLYNOMIAL_MAX_RAD_1GHZ
    casa = read_beam_polynomial_coefficients("EVLA_", n_sample=10000)
    for n_sample, params in (
        (None, {"telescope": "vla"}),
        (10000, {"beam_model": casa}),
    ):
        img_xds = _primary_beam(params, frequency_coords=(33e9, 45e9), cell_arcsec=4.0)
        pb = img_xds["PRIMARY_BEAM"]
        assert pb.attrs["method"] == "beam_polynomial"
        l_axis, m_axis = img_xds.l.values[:, None], img_xds.m.values[None, :]
        for f, table_frequency in enumerate((34052e6, 43948e6)):
            frequency = img_xds.frequency.values[f]
            expected = polynomial_beam_response(
                l_axis,
                m_axis,
                frequency,
                bpc.BPC.sel(frequency=table_frequency).values[0],
                EVLA_BEAM_POLYNOMIAL_MAX_RAD_1GHZ,
                ipower=2,
                n_sample=n_sample,
            )
            outside = np.hypot(l_axis, m_axis) >= EVLA_BEAM_POLYNOMIAL_MAX_RAD_1GHZ / (
                frequency / 1e9
            )
            assert outside.any() and not outside.all()
            np.testing.assert_array_equal(
                pb.values[0, f, 0], np.where(outside, 0.0, expected)
            )
    resolved = resolve_primary_beam_params({"telescope": "vla", "max_rad_1GHz": 0.02})
    assert resolved["beam_model"].attrs["max_rad_1GHz"] == 0.02


def test_primary_beam_params_numbers_equal_to_a_telescope_give_its_beam():
    """Explicit numbers equal to a telescope's give its beam; numbers override a telescope."""
    ngvla = airy_disk_model("ngvla")
    explicit = {
        "effective_dish_diameter": ngvla["dish_diameter"],
        "blockage_diameter": ngvla["blockage_diameter"],
        "max_rad_1GHz": ngvla["max_rad_1GHz"],
    }
    by_name = _primary_beam({"telescope": "ngvla"})["PRIMARY_BEAM"].values
    np.testing.assert_array_equal(
        _primary_beam(explicit)["PRIMARY_BEAM"].values, by_name
    )
    overridden = _primary_beam({"telescope": "ngvla", "effective_dish_diameter": 25.0})
    np.testing.assert_array_equal(
        overridden["PRIMARY_BEAM"].values,
        _primary_beam({**explicit, "effective_dish_diameter": 25.0})[
            "PRIMARY_BEAM"
        ].values,
    )
    assert not np.array_equal(overridden["PRIMARY_BEAM"].values, by_name)


@pytest.mark.parametrize(
    "primary_beam_params, match",
    [
        ({"telescope": "gbt"}, "telescope"),
        ({"dish_diameter": 12.0}, "Unknown primary_beam_params"),
        ({"effective_dish_diameter": 0.0}, "dish diameter must be strictly positive"),
        ({"blockage_diameter": 11.0}, "matching diameters"),
        ({"max_rad_1GHz": 0.0}, "max_rad_1GHz must be strictly positive"),
        (
            {"telescope": "vla", "effective_dish_diameter": 25.0},
            "do not accept dish/blockage overrides",
        ),
        (
            {"telescope": "VLA", "blockage_diameter": 1.0},
            "do not accept dish/blockage overrides",
        ),
        (
            {"telescope": "vla", "max_rad_1GHz": 0.0},
            "max_rad_1GHz must be strictly positive",
        ),
        (
            {"telescope": "vla", "max_rad_1GHz": None},
            "max_rad_1GHz must be strictly positive",
        ),
    ],
)
def test_primary_beam_params_refused(primary_beam_params, match):
    with pytest.raises(ValueError, match=match):
        resolve_primary_beam_params(primary_beam_params)


def _beam_model(name, casa=False):
    from astroviper.utils.beam_models import read_beam_polynomial_coefficients

    if name == "EVLA_":
        return read_beam_polynomial_coefficients(name, n_sample=10000 if casa else None)
    return airy_disk_model(name, func="casa_airy" if casa else None)


@pytest.mark.parametrize("casa", [False, True])
@pytest.mark.parametrize(
    "name, frequency_coords",
    [
        ("alma", (90e9, 230e9, 345e9)),
        ("aca", (90e9, 230e9)),
        ("ngvla", (30e9, 93e9)),
        ("vla_nvss", (1.5e9, 3e9)),
        ("EVLA_", (2.01e9, 3e9, 4.5e9)),
    ],
)
def test_primary_beam_is_the_simulator_beam(name, frequency_coords, casa):
    """The primary beam of a beam model is |J_pp|**2 of the simulator's
    sample_jones for that model, inside and beyond CASA's max_rad_1GHz."""
    from astroviper.processing_functions.simulation.antenna_beams import (
        evaluate_beam_models,
        pack_beam_models,
        sample_jones,
    )

    model = _beam_model(name, casa)
    max_rad = (_beam_model(name, casa=True) if name != "EVLA_" else model.attrs)[
        "max_rad_1GHz"
    ]
    # the image reaches 1.5 times the cut at the lowest frequency
    cell_arcsec = 1.5 * max_rad / (min(frequency_coords) / 1e9) / 32 / ARCSEC
    img_xds = _primary_beam(
        {"beam_model": model},
        frequency_coords=frequency_coords,
        cell_arcsec=cell_arcsec,
    )
    frequency = img_xds.frequency.values
    l_grid, m_grid = np.meshgrid(img_xds.l.values, img_xds.m.values, indexing="ij")
    outside = np.hypot(l_grid, m_grid) >= max_rad / (frequency[:, None, None] / 1e9)
    assert outside.any(axis=(1, 2)).all() and not outside.all(axis=(1, 2)).any()
    evaluated, parallactic_angle = evaluate_beam_models(
        [model], np.array([0.0]), frequency, np.zeros((1, 2)), np.zeros(3)
    )
    jones = sample_jones(
        pack_beam_models(evaluated)[0],
        np.stack([l_grid.ravel(), m_grid.ravel()], axis=-1),
        frequency,
        parallactic_angle[0],
    )
    simulated = (np.abs(jones[:, :, 0]) ** 2).T.reshape(outside.shape)
    np.testing.assert_allclose(
        img_xds["PRIMARY_BEAM"].values[0, :, 0], simulated, rtol=0, atol=1e-15
    )


@pytest.mark.parametrize("telescope", ["alma", "aca", "ngvla", "vla"])
def test_primary_beam_params_beam_model_is_the_telescope_beam(telescope):
    """A telescope names a beam model: giving that model is the same beam."""
    model = _beam_model("EVLA_" if telescope == "vla" else telescope)
    np.testing.assert_array_equal(
        _primary_beam({"beam_model": model})["PRIMARY_BEAM"].values,
        _primary_beam({"telescope": telescope})["PRIMARY_BEAM"].values,
    )


def test_primary_beam_params_beam_model_is_normalized():
    legacy = {"dish_diam": 10.7, "blockage_diam": 0.75, "max_rad_1GHz": 0.0311}
    assert resolve_primary_beam_params({"beam_model": legacy})["beam_model"] == {
        "func": "casa_airy",
        "dish_diameter": 10.7,
        "blockage_diameter": 0.75,
        "max_rad_1GHz": 0.0311,
    }


def _zernike_model():
    from astroviper.utils.beam_models import read_aperture_polynomial_coefficients

    return read_aperture_polynomial_coefficients("EVLA_avg_zcoeffs_SBand_lookup")


def _jones_image_model():
    from astroviper.processing_functions.simulation.antenna_beams import (
        make_airy_jones_beam,
    )

    return make_airy_jones_beam(airy_disk_model("alma"), [1e11], {"image_size": [8, 8]})


@pytest.mark.parametrize(
    "primary_beam_params, error, match",
    [
        (
            lambda: {"beam_model": airy_disk_model("alma"), "telescope": "alma"},
            ValueError,
            "cannot be combined",
        ),
        (
            lambda: {"beam_model": _beam_model("EVLA_"), "max_rad_1GHz": 0.02},
            ValueError,
            "cannot be combined",
        ),
        (
            lambda: {"beam_model": _zernike_model()},
            ValueError,
            "analytic mapping or beam-polynomial dataset",
        ),
        (
            lambda: {"beam_model": _jones_image_model()},
            ValueError,
            "analytic mapping or beam-polynomial dataset",
        ),
        (lambda: {"beam_model": "alma"}, TypeError, "Unsupported beam model"),
        (
            lambda: {"beam_model": {**airy_disk_model("alma"), "dish_diameter": 0.0}},
            ValueError,
            "dish diameter must be strictly positive",
        ),
        (
            lambda: {
                "beam_model": {**airy_disk_model("alma"), "blockage_diameter": 12.0}
            },
            ValueError,
            "matching diameters",
        ),
        (
            lambda: {"beam_model": {**airy_disk_model("alma"), "func": "gaussian"}},
            ValueError,
            "func must be 'airy', 'casa_airy', or 'none'",
        ),
        (
            lambda: {"beam_model": {**airy_disk_model("alma"), "max_rad_1GHz": 0.0}},
            ValueError,
            "max_rad_1GHz must be strictly positive",
        ),
        (
            lambda: {"beam_model": _beam_model("EVLA_").assign_attrs(max_rad_1GHz=0.0)},
            ValueError,
            "max_rad_1GHz must be strictly positive",
        ),
    ],
)
def test_primary_beam_params_beam_model_refused(primary_beam_params, error, match):
    with pytest.raises(error, match=match):
        resolve_primary_beam_params(primary_beam_params())


def test_primary_beam_of_a_zernike_model_refused():
    img_xds, image_params = _make_empty_residual_image()
    with pytest.raises(ValueError, match="analytic mapping or beam-polynomial dataset"):
        make_primary_beam_single_field(
            img_xds, image_params, beam_model=_zernike_model()
        )


def test_primary_beam_of_func_none_is_uniform():
    """func "none" is 1 out to max_rad_1GHz, not an Airy disk."""
    pb = _primary_beam(
        {"beam_model": airy_disk_model("alma", func="none")},
        frequency_coords=(1.0e11,),
        cell_arcsec=2.0,
    )["PRIMARY_BEAM"]
    assert pb.attrs["method"] == "uniform"
    assert set(np.unique(pb.values)) == {0.0, 1.0}
