"""ALMA prescription selection and independent CASA image reference tests."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _convert_primary_beam_to_reference_frequency,
)
from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
    casa_airy_disk_response,
    evaluate_primary_beam,
    resolve_continuum_primary_beam,
)
from astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric import (
    airy_disk_rorder_v2,
)
from astroviper.processing_functions.imaging.primary_beam.make_primary_beam import (
    make_primary_beam_single_field,
)


def antennas(telescope, diameter):
    return xr.Dataset(
        {"ANTENNA_DISH_DIAMETER": ("antenna_name", [diameter, diameter])},
        coords={
            "antenna_name": ["a", "b"],
            "telescope_name": ("antenna_name", [telescope, telescope]),
        },
    )


@pytest.mark.parametrize(
    ("telescope", "diameter", "effective", "model"),
    [
        ("ALMA", 12, 10.7, "casa_airy"),
        ("ALMA", 7, 6.25, "casa_airy"),
        ("ACA", 7, 6.25, "casa_airy"),
        ("VLA", 25, 25, "airy"),
        ("EVLA", 25, 25, "airy"),
        ("OTHER", 12, 12, "airy"),
    ],
)
def test_auto_prescription_requires_telescope_identity(
    telescope, diameter, effective, model
):
    source = antennas(telescope, diameter)
    original = source.copy(deep=True)
    params = {}
    actual = resolve_continuum_primary_beam(params, source)
    assert actual["primary_beam_model"] == model
    assert actual["list_dish_diameters"] == [effective]
    assert actual["list_blockage_diameters"] == [0.75]
    assert params == {}
    xr.testing.assert_identical(source, original)


def test_explicit_parameters_take_precedence():
    params = {
        "list_dish_diameters": [11.1],
        "list_blockage_diameters": [0.3],
        "primary_beam_max_radius_1ghz": 0.04,
    }
    actual = resolve_continuum_primary_beam(params, antennas("ALMA", 12))
    for key in params:
        assert actual[key] == params[key]
    assert actual["primary_beam_model"] == "casa_airy"


def test_physical_airy_opt_out_keeps_physical_alma_diameter():
    actual = resolve_continuum_primary_beam(
        {"primary_beam_model": "airy"}, antennas("ALMA", 12)
    )
    assert actual["list_dish_diameters"] == [12]
    assert actual["primary_beam_model"] == "airy"


def test_telescope_attribute_fallback():
    source = antennas("ACA", 7).drop_vars("telescope_name")
    source.attrs["overall_telescope_name"] = "ALMA"
    assert resolve_continuum_primary_beam({}, source)["list_dish_diameters"] == [6.25]


def test_mixed_apertures_require_explicit_selection():
    source = antennas("ALMA", 12)
    source.ANTENNA_DISH_DIAMETER.values[1] = 7
    with pytest.raises(NotImplementedError, match="one common"):
        resolve_continuum_primary_beam({}, source)


def test_casa_reference_samples():
    """6144 samples from actual CASA 6.7.7.6 tclean PB images, not our formula."""
    with np.load(Path(__file__).parent / "data/casa_alma_pb_samples.npz") as reference:
        for diameter in (6.25, 10.7):
            selected = reference["diameter"] == diameter
            actual = casa_airy_disk_response(
                reference["l"][selected],
                reference["m"][selected],
                reference["frequency"][selected],
                diameter,
                0.75,
                reference["max_radius"][selected][0],
                ipower=2,
            )
            np.testing.assert_allclose(
                actual, reference["expected"][selected], rtol=1e-6, atol=1e-7
            )


@pytest.mark.parametrize("diameter", [7, 12])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_channel_and_reference_paths_share_the_casa_prescription(diameter, dtype):
    geometry = {
        "image_size": [32, 32],
        "cell_size": np.array([-2.0, 2.0]) * np.pi / (180 * 3600),
    }
    params = resolve_continuum_primary_beam(geometry, antennas("ALMA", diameter))
    image = xr.Dataset(
        coords={
            "time": [0],
            "frequency": [90e9, 100e9],
            "polarization": ["I", "Q"],
            "l": np.arange(32),
            "m": np.arange(32),
        },
        attrs={"data_groups": {"residual": {}}},
    )
    channel, _ = make_primary_beam_single_field(
        image,
        params,
        list_dish_diameters=params["list_dish_diameters"],
        list_blockage_diameters=params["list_blockage_diameters"],
        ipower=2,
        float_dtype=dtype,
    )
    expected = channel.PRIMARY_BEAM.isel(frequency=1).values.copy()
    reference = _convert_primary_beam_to_reference_frequency(
        channel, image_params=params, reference_frequency_hz=100e9, float_dtype=dtype
    )
    np.testing.assert_array_equal(reference.PRIMARY_BEAM_REFERENCE.values, expected)
    assert reference.PRIMARY_BEAM_REFERENCE.dtype == np.dtype(dtype)


def test_vla_airy_evaluation_unchanged():
    grid = resolve_continuum_primary_beam(
        {"image_size": [32, 32], "image_center": [16, 16], "cell_size": [-1e-4, 1e-4]},
        antennas("VLA", 25),
    )
    pb = {
        "list_dish_diameters": grid["list_dish_diameters"],
        "list_blockage_diameters": grid["list_blockage_diameters"],
        "ipower": 2,
    }
    frequencies = np.array([1e9, 2e9])
    np.testing.assert_array_equal(
        evaluate_primary_beam(frequencies, ["I"], pb, grid),
        airy_disk_rorder_v2(frequencies, ["I"], pb, grid),
    )


def test_casa_lookup_support_and_voltage_power():
    l = np.array([0.0, 1e-5, 1.0])
    voltage = casa_airy_disk_response(
        l, 0.0, 100e9, 10.7, 0.75, np.deg2rad(1.784), ipower=1
    )
    power = casa_airy_disk_response(
        l, 0.0, 100e9, 10.7, 0.75, np.deg2rad(1.784), ipower=2
    )
    assert voltage[0] == power[0] == 1
    assert voltage[-1] == power[-1] == 0
    np.testing.assert_array_equal(power, voltage**2)
