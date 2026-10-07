"""ALMA prescription selection and independent CASA image reference tests."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _convert_primary_beam_to_reference_frequency,
)
from astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam import (
    casa_airy_disk_response,
    evaluate_primary_beam,
    make_continuum_primary_beam_single_field,
    resolve_continuum_primary_beam,
)
from astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric import (
    airy_disk_rorder_v2,
)

# CASA 6.7.7.6 samples: l, m [rad], frequency [Hz], diameter [m], support [rad at 1 GHz], power.
CASA_ALMA_SAMPLES = np.array(
    [
        [
            1.2605155708847548e-05,
            1.2605155708847548e-05,
            40148937499.95,
            10.7,
            0.03113667385557884,
            0.9999157190322876,
        ],
        [
            0.0003592469377021551,
            -3.7815467126542643e-05,
            40148937499.95,
            10.7,
            0.03113667385557884,
            0.5006424188613892,
        ],
        [
            0.00032143147057561245,
            0.0004159701383919691,
            40148937499.95,
            10.7,
            0.03113667385557884,
            0.20062637329101562,
        ],
        [
            -4.217879025653196e-06,
            5.623838700870928e-06,
            95560388101.64,
            10.7,
            0.03113667385557884,
            0.9999343156814575,
        ],
        [
            -9.560525791480578e-05,
            -0.00011810061271828949,
            95560388101.64,
            10.7,
            0.03113667385557884,
            0.4998404383659363,
        ],
        [
            0.00022073566900918394,
            -1.1247677401741856e-05,
            95560388101.64,
            10.7,
            0.03113667385557884,
            0.20006176829338074,
        ],
        [
            5.817764173313849e-06,
            0.0,
            130451615958.6,
            6.25,
            0.06227334771115768,
            0.9999957084655762,
        ],
        [
            9.308422677302159e-05,
            0.00016871516102610163,
            130451615958.6,
            6.25,
            0.06227334771115768,
            0.500546932220459,
        ],
        [
            0.0002734349161457509,
            -5.2359877559824645e-05,
            130451615958.6,
            6.25,
            0.06227334771115768,
            0.20049455761909485,
        ],
        [
            5.817764173312977e-06,
            5.585053606380458e-06,
            131159397083.7,
            10.7,
            0.03113667385557884,
            0.9996080994606018,
        ],
        [
            9.936741208018565e-05,
            4.9101929622761524e-05,
            131159397083.7,
            10.7,
            0.03113667385557884,
            0.49887824058532715,
        ],
        [
            0.0001610357123173032,
            -1.1635528346625953e-06,
            131159397083.7,
            10.7,
            0.03113667385557884,
            0.20006176829338074,
        ],
        [
            -2.036217460660109e-06,
            5.429913228426958e-06,
            150521369319.3,
            10.7,
            0.03113667385557884,
            0.9998141527175903,
        ],
        [
            2.714956614213479e-06,
            9.638095980457852e-05,
            150521369319.3,
            10.7,
            0.03113667385557884,
            0.5001612305641174,
        ],
        [
            0.00013506909155712059,
            -3.800939259898871e-05,
            150521369319.3,
            10.7,
            0.03113667385557884,
            0.20006176829338074,
        ],
        [
            -0.0,
            5.623838700870229e-06,
            337437990700.3,
            6.25,
            0.06227334771115768,
            0.9998328685760498,
        ],
        [
            7.310990311131298e-05,
            1.4059596752175573e-05,
            337437990700.3,
            6.25,
            0.06227334771115768,
            0.501125693321228,
        ],
        [
            7.029798376087786e-05,
            8.154566116261833e-05,
            337437990700.3,
            6.25,
            0.06227334771115768,
            0.20022669434547424,
        ],
    ]
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
    """Independent CASA samples span six images and central/half-power/outer PB."""
    for l, m, frequency, diameter, max_radius, expected in CASA_ALMA_SAMPLES:
        actual = casa_airy_disk_response(
            l, m, frequency, diameter, 0.75, max_radius, ipower=2
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


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
    channel, _ = make_continuum_primary_beam_single_field(
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


def test_cube_beam_does_not_enter_continuum_prescription(monkeypatch):
    import astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam as continuum_beam
    from astroviper.processing_functions.imaging.primary_beam.make_primary_beam import (
        make_primary_beam_single_field,
    )

    image = xr.Dataset(
        coords={
            "time": [0],
            "frequency": [90e9, 100e9],
            "polarization": ["I"],
            "l": np.linspace(-1e-4, 1e-4, 8),
            "m": np.linspace(-1e-4, 1e-4, 8),
        },
        attrs={"data_groups": {"residual": {}}},
    )
    params = {"image_size": [8, 8], "cell_size": [-1e-5, 1e-5]}
    baseline, _ = make_primary_beam_single_field(
        image.copy(deep=True), params, list_dish_diameters=[12.0]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Cube imaging must not call the continuum prescription")

    monkeypatch.setattr(continuum_beam, "casa_airy_disk_response", forbidden)
    monkeypatch.setattr(continuum_beam, "resolve_continuum_primary_beam", forbidden)
    actual, _ = make_primary_beam_single_field(
        image.copy(deep=True),
        {
            **params,
            "primary_beam_model": "casa_airy",
            "primary_beam_max_radius_1ghz": 0.01,
        },
        list_dish_diameters=[12.0],
    )
    np.testing.assert_array_equal(actual.PRIMARY_BEAM, baseline.PRIMARY_BEAM)
