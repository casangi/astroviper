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
    """PRIMARY_BEAM follows the CASA definition: the shared CASA-compatible
    Airy voltage pattern squared (power), identical to the simulation code."""
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
    np.testing.assert_array_equal(img_xds["PRIMARY_BEAM"].values[0, 0, 0], voltage**2)


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
    """{"telescope": "vla"} is the 24.5 m beam without blockage, sampled as CASA's VLA table."""
    from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
        casa_airy_disk_response,
    )

    vla = airy_disk_model("vla")
    img_xds = _primary_beam({"telescope": "vla"})
    pb = img_xds["PRIMARY_BEAM"].values[0, :, 0]
    l_axis, m_axis = img_xds.l.values[:, None], img_xds.m.values[None, :]
    for f, frequency in enumerate(img_xds.frequency.values):
        expected = casa_airy_disk_response(
            l_axis, m_axis, frequency, 24.5, 0.0, vla["max_rad_1GHz"], ipower=2
        )
        np.testing.assert_array_equal(pb[f], expected)
        # the VLA table radius, not ALMA's, sets the sampling of the beam
        alma_sampling = casa_airy_disk_response(
            l_axis,
            m_axis,
            frequency,
            24.5,
            0.0,
            airy_disk_model("alma")["max_rad_1GHz"],
            ipower=2,
        )
        assert not np.array_equal(pb[f], alma_sampling)
    alma = _primary_beam(None)["PRIMARY_BEAM"].values[0, :, 0]
    assert not np.allclose(pb, alma)


def test_primary_beam_params_numbers_equal_to_a_telescope_give_its_beam():
    """Explicit numbers equal to a telescope's give its beam; numbers override a telescope."""
    vla = airy_disk_model("vla")
    explicit = {
        "effective_dish_diameter": vla["dish_diameter"],
        "blockage_diameter": vla["blockage_diameter"],
        "max_rad_1GHz": vla["max_rad_1GHz"],
    }
    by_name = _primary_beam({"telescope": "vla"})["PRIMARY_BEAM"].values
    np.testing.assert_array_equal(
        _primary_beam(explicit)["PRIMARY_BEAM"].values, by_name
    )
    overridden = _primary_beam({"telescope": "vla", "effective_dish_diameter": 25.0})
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
        ({"dish_diameter": 12.0}, "unknown primary_beam_params"),
        ({"effective_dish_diameter": 0.0}, "effective_dish_diameter > 0"),
        ({"blockage_diameter": 11.0}, "blockage_diameter"),
        ({"max_rad_1GHz": 0.0}, "max_rad_1GHz > 0"),
    ],
)
def test_primary_beam_params_refused(primary_beam_params, match):
    with pytest.raises(ValueError, match=match):
        resolve_primary_beam_params(primary_beam_params)
