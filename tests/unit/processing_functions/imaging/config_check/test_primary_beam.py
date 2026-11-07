"""Explicit and inferred primary-beam metadata validation."""

from types import MappingProxyType

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_residual_update,
)
from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
    check_continuum_beam_metadata,
)


@pytest.mark.parametrize(
    "beam",
    [
        {"primary_beam_model": "unknown"},
        {"list_dish_diameters": []},
        {"list_dish_diameters": [0.0]},
        {"list_dish_diameters": [12.0], "list_blockage_diameters": [12.0]},
        {"list_dish_diameters": [12.0], "list_blockage_diameters": [0.0, 0.0]},
        {"list_blockage_diameters": [-1.0]},
        {"list_blockage_diameters": [[0.0]]},
        {"primary_beam_max_radius_1ghz": np.nan},
    ],
)
def test_explicit_continuum_beam_rejected(beam):
    with pytest.raises((ValueError, TypeError)):
        check_residual_update({"image_params": beam}, mode=CONTINUUM_SINGLE_FIELD)


@pytest.mark.parametrize(
    "params",
    [
        [],
        {"typo": 1},
        {"telescope": "unknown"},
        {"effective_dish_diameter": -1.0},
        {"blockage_diameter": np.inf},
        {"max_rad_1GHz": -1.0},
        {"telescope": "vla", "effective_dish_diameter": 25.0},
        {"telescope": "vla", "max_rad_1GHz": None},
        {"beam_model": {}, "telescope": "alma"},
        {"beam_model": {"dish_diameter": 12.0, "max_rad_1GHz": None}},
        {"beam_model": {"func": "bad", "dish_diameter": 12.0, "max_rad_1GHz": 1.0}},
        {
            "beam_model": {
                "dish_diameter": 12.0,
                "blockage_diameter": 13.0,
                "max_rad_1GHz": 1.0,
            }
        },
        {"beam_model": xr.Dataset({"JONES": ("x", [1.0])})},
    ],
)
def test_cube_primary_beam_rejected(params):
    with pytest.raises((ValueError, TypeError)):
        check_residual_update({"primary_beam_params": params}, mode=CUBE_SINGLE_FIELD)


def polynomial_model():
    return xr.Dataset(
        {"BPC": (("frequency", "polarization", "coefficient"), np.ones((1, 1, 2)))},
        coords={"frequency": [1e9], "polarization": ["I"], "coefficient": [0, 1]},
        attrs={"dish_diameter": 25.0, "max_rad_1GHz": 0.01, "n_sample": None},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_sample", 1),
        ("n_sample", True),
        ("max_rad_1GHz", np.nan),
        ("dish_diameter", 0),
    ],
)
def test_polynomial_beam_metadata_rejected(field, value):
    model = polynomial_model().assign_attrs({field: value})
    with pytest.raises((ValueError, TypeError)):
        check_residual_update(
            {"primary_beam_params": {"beam_model": model}}, mode=CUBE_SINGLE_FIELD
        )


def test_read_only_beam_configuration_is_preserved():
    model = polynomial_model()
    snapshot = model.copy(deep=True)
    params = MappingProxyType({"beam_model": model})
    check_residual_update({"primary_beam_params": params}, mode=CUBE_SINGLE_FIELD)
    xr.testing.assert_identical(model, snapshot)
    analytic = MappingProxyType(
        {"func": "airy", "dish_diameter": 12.0, "max_rad_1GHz": None}
    )
    check_residual_update(
        {"primary_beam_params": {"beam_model": analytic}}, mode=CUBE_SINGLE_FIELD
    )
    assert "blockage_diameter" not in analytic


@pytest.mark.parametrize("diameters", [[12.0, 7.0], [0.0], [np.nan]])
def test_inferred_beam_fails_before_psf(diameters):
    antenna = xr.Dataset({"ANTENNA_DISH_DIAMETER": ("antenna_name", diameters)})
    with pytest.raises((ValueError, NotImplementedError)):
        check_continuum_beam_metadata({}, antenna, continuum_mode="mfs")


def test_inferred_alma_aperture_and_explicit_override():
    antenna = xr.Dataset(
        {"ANTENNA_DISH_DIAMETER": ("antenna_name", [15.0])},
        attrs={"overall_telescope_name": "ALMA"},
    )
    with pytest.raises(ValueError, match="12-m or 7-m"):
        check_continuum_beam_metadata({}, antenna, continuum_mode="mvc")
    params = {"list_dish_diameters": [10.7]}
    resolved = check_continuum_beam_metadata(params, antenna, continuum_mode="mvc")
    assert resolved["list_dish_diameters"] == [10.7]
    assert resolved["list_blockage_diameters"] == [0.75]
    assert resolved["primary_beam_model"] == "casa_airy"
    assert params == {"list_dish_diameters": [10.7]}


@pytest.mark.parametrize(
    "fault,message",
    [
        ("dimensions", "BPC needs"),
        ("empty", "nonempty"),
        ("frequency_dimensions", "one-dimensional"),
        ("dish_diameter", "requires dish_diameter"),
        ("max_rad_1GHz", "requires max_rad_1GHz"),
    ],
)
def test_polynomial_model_structure_is_validated_without_evaluating_pixels(
    fault, message
):
    from astroviper.processing_functions.imaging.config_check._primary_beam import (
        _check_explicit_beam_model,
    )

    model = polynomial_model()
    if fault == "dimensions":
        model["BPC"] = model.BPC.transpose("polarization", "frequency", "coefficient")
    elif fault == "empty":
        model = model.isel(frequency=slice(0, 0))
    elif fault == "frequency_dimensions":
        model = model.assign_coords(frequency=(("frequency", "extra"), [[1e9]]))
    else:
        del model.attrs[fault]
    original = model.copy(deep=True)
    with pytest.raises(ValueError, match=message):
        _check_explicit_beam_model(model)
    xr.testing.assert_identical(model, original)
