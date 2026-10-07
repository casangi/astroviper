"""Legacy VLA beam selection and independent CASA image regression."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam import (
    casa_airy_disk_response,
    evaluate_primary_beam,
    resolve_continuum_primary_beam,
)

# CASA MFS image samples: l pixel, m pixel, power.
CASA_VLA_MFS_SAMPLES = np.array(
    [
        [100, 100, 1.0],
        [100, 125, 0.9640339016914368],
        [100, 150, 0.8078857064247131],
        [100, 175, 0.5965918302536011],
        [100, 199, 0.38711267709732056],
        [0, 0, 0.11041608452796936],
        [25, 150, 0.4628506600856781],
        [175, 25, 0.33019864559173584],
    ]
)


# CASA MVC samples: l, m [rad], frequency [Hz], clipped power.
CASA_VLA_MVC_SAMPLES = np.array(
    [
        [
            -0.00019392547244381441,
            0.00019392547244381441,
            999988830.2104049,
            0.9987678527832031,
        ],
        [
            -0.003975472185098195,
            -0.00484813681109536,
            999988830.2104049,
            0.5032012462615967,
        ],
        [
            0.00484813681109536,
            -0.00484813681109536,
            999988830.2104049,
            0.4352755546569824,
        ],
        [
            0.00484813681109536,
            -0.00484813681109536,
            999988830.2104049,
            0.4352755546569824,
        ],
        [
            -0.00019392547244381441,
            0.00019392547244381441,
            1499984801.867004,
            0.9972193241119385,
        ],
        [
            0.0029573634547681695,
            -0.0029573634547681695,
            1499984801.867004,
            0.502850353717804,
        ],
        [
            0.00484813681109536,
            -0.00484813681109536,
            1499984801.867004,
            0.12125224620103836,
        ],
        [
            0.00484813681109536,
            -0.00484813681109536,
            1499984801.867004,
            0.12125224620103836,
        ],
        [
            -0.00019392547244381441,
            0.00019392547244381441,
            1949981176.357943,
            0.9953176975250244,
        ],
        [
            -0.0027149566142134016,
            -0.001696847883883376,
            1949981176.357943,
            0.5063600540161133,
        ],
        [
            -0.0033452143996557985,
            -0.004217879025652963,
            1949981176.357943,
            0.10825737565755844,
        ],
        [0.00484813681109536, -0.00484813681109536, 1949981176.357943, 0.0],
    ]
)


def antennas(telescope="VLA"):
    return xr.Dataset(
        {"ANTENNA_DISH_DIAMETER": ("antenna_name", [25.0])},
        coords={"telescope_name": ("antenna_name", [telescope])},
    )


@pytest.mark.parametrize("frequency", [0.074, 0.3, 1.5, 5, 8, 15, 22, 43])
def test_legacy_vla_bands(frequency):
    params = {"reference_frequency": frequency * 1e9}
    resolved = resolve_continuum_primary_beam(params, antennas())
    assert resolved["primary_beam_model"] == "casa_airy"
    assert resolved["list_dish_diameters"] == [25]
    assert resolved["list_blockage_diameters"] == [2.36]
    assert resolved["primary_beam_max_radius_1ghz"] == np.deg2rad(0.8564)
    assert params == {"reference_frequency": frequency * 1e9}


@pytest.mark.parametrize("frequency", [0.1, 0.15, 0.4, 1, 2, 3, 7, 55, 90])
def test_vla_band_gaps_select_nvss(frequency):
    resolved = resolve_continuum_primary_beam(
        {"reference_frequency": frequency * 1e9}, antennas()
    )
    assert resolved["primary_beam_model"] == "casa_airy"
    assert resolved["list_dish_diameters"] == [24.5]
    assert resolved["list_blockage_diameters"] == [0.0]


def test_evla_is_not_legacy_vla():
    resolved = resolve_continuum_primary_beam(
        {"reference_frequency": 1.5e9}, antennas("EVLA")
    )
    assert resolved["primary_beam_model"] == "airy"


def test_frequency_fallbacks_and_overrides():
    for frequency in (
        {"reference_frequency_hz": 1.5e9},
        {"frequency_coords": [1e9, 2e9]},
    ):
        assert (
            resolve_continuum_primary_beam(frequency, antennas())["primary_beam_model"]
            == "casa_airy"
        )
    params = {
        "reference_frequency": 1.5e9,
        "list_dish_diameters": [24.0],
        "list_blockage_diameters": [0.1],
        "primary_beam_max_radius_1ghz": 0.02,
    }
    resolved = resolve_continuum_primary_beam(params, antennas())
    for key, value in params.items():
        assert resolved[key] == value
    params = {"reference_frequency": 1.5e9, "primary_beam_model": "airy"}
    resolved = resolve_continuum_primary_beam(params, antennas())
    assert resolved["primary_beam_model"] == "airy"
    assert resolved["list_blockage_diameters"] == [0.75]


def test_casa_vla_image():
    # Actual tclean output from the off-axis refim fixture, not our formula.
    geometry = {
        "reference_frequency": 1.5e9,
        "image_size": [200, 200],
        "image_center": [100, 100],
        "cell_size": np.deg2rad([-10 / 3600, 10 / 3600]),
    }
    resolved = resolve_continuum_primary_beam(geometry, antennas())
    actual = evaluate_primary_beam([1.5e9], ["I"], {**resolved, "ipower": 2}, resolved)[
        0, 0, 0
    ]
    pixels = CASA_VLA_MFS_SAMPLES[:, :2].astype(int)
    np.testing.assert_allclose(
        actual[pixels[:, 0], pixels[:, 1]],
        CASA_VLA_MFS_SAMPLES[:, 2],
        rtol=0,
        atol=2e-7,
    )


def test_mvc_selects_first_channel_once():
    geometry = {
        "reference_frequency": 1.5e9,
        "frequency_coords": np.arange(20) * 5e7 + 1e9,
    }
    resolved = resolve_continuum_primary_beam(geometry, antennas(), specmode="mvc")
    assert resolved["list_dish_diameters"] == [24.5]
    assert resolved["list_blockage_diameters"] == [0.0]
    l, m, frequency, expected = CASA_VLA_MVC_SAMPLES.T
    actual = casa_airy_disk_response(
        l,
        m,
        frequency,
        resolved["list_dish_diameters"][0],
        resolved["list_blockage_diameters"][0],
        resolved["primary_beam_max_radius_1ghz"],
        ipower=2,
    )
    # CASA normalizes/clips its PB cube before forming Taylor products.
    actual = np.where(actual > 0.1, actual, 0.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
    # Reversing the spectral axis deliberately changes CASA's band selection.
    geometry["frequency_coords"] = geometry["frequency_coords"][::-1]
    resolved = resolve_continuum_primary_beam(geometry, antennas(), specmode="mvc")
    assert resolved["list_dish_diameters"] == [25]
    assert resolved["list_blockage_diameters"] == [2.36]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_partitioned_mvc_reuses_resolved_vla_prescription(dtype):
    from astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam import (
        make_continuum_primary_beam_single_field,
    )

    geometry = {
        "image_size": [20, 20],
        "image_center": [10, 10],
        "cell_size": np.deg2rad([-10 / 3600, 10 / 3600]),
        "frequency_coords": [1e9, 1.5e9],
        "reference_frequency": 1.5e9,
    }
    resolved = resolve_continuum_primary_beam(geometry, antennas(), specmode="mvc")
    assert resolved["list_dish_diameters"] == [24.5]
    # This partition starts inside VLA L-band. It must still use the NVSS
    # choice resolved from the full cube's first channel, without re-selection.
    image = xr.Dataset(
        coords={
            "time": [0],
            "frequency": [1.5e9],
            "polarization": ["I"],
            "l": np.arange(20),
            "m": np.arange(20),
        },
        attrs={"data_groups": {"residual": {}}},
    )
    channel, _ = make_continuum_primary_beam_single_field(
        image,
        resolved,
        list_dish_diameters=resolved["list_dish_diameters"],
        list_blockage_diameters=resolved["list_blockage_diameters"],
        float_dtype=dtype,
    )
    full = evaluate_primary_beam(
        [1e9, 1.5e9], ["I"], {**resolved, "ipower": 2}, resolved, dtype=dtype
    )
    np.testing.assert_array_equal(channel.PRIMARY_BEAM.values[0, 0, 0], full[0, 1, 0])
