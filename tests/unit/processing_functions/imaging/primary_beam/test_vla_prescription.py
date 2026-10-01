"""Legacy VLA beam selection and independent CASA image regression."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam import (
    casa_airy_disk_response,
    evaluate_primary_beam,
    resolve_continuum_primary_beam,
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
    with np.load(Path(__file__).parent / "data/casa_vla_pb.npz") as data:
        reference = data["pb"]
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
    np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-7)


def test_mvc_selects_first_channel_once():
    geometry = {
        "reference_frequency": 1.5e9,
        "frequency_coords": np.arange(20) * 5e7 + 1e9,
    }
    resolved = resolve_continuum_primary_beam(geometry, antennas(), specmode="mvc")
    assert resolved["list_dish_diameters"] == [24.5]
    assert resolved["list_blockage_diameters"] == [0.0]
    with np.load(Path(__file__).parent / "data/casa_vla_pb.npz") as data:
        actual = casa_airy_disk_response(
            data["l"],
            data["m"],
            data["frequency"],
            resolved["list_dish_diameters"][0],
            resolved["list_blockage_diameters"][0],
            resolved["primary_beam_max_radius_1ghz"],
            ipower=2,
        )
        # CASA normalizes/clips its PB cube before forming Taylor products.
        actual = np.where(actual > 0.1, actual, 0.0)
        np.testing.assert_allclose(actual, data["cube_pb"], rtol=0, atol=2e-7)
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
