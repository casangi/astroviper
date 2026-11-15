"""Actual mask values are checked even when configuration has been validated."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
)
from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_model_products,
)
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    model_update_mtmfs_single_field,
)
from astroviper.processing_functions.imaging.model_update import (
    model_update_cube_single_field,
)
from astroviper.utils.clean_mask import install_clean_mask


def image_with_mask(mode, value, dtype=np.float64):
    """One-term products keep the test independent of Taylor-Hessian safety."""
    continuum = mode == CONTINUUM_SINGLE_FIELD
    plane = "taylor_term" if continuum else "frequency"
    psf_plane = "psf_taylor_order" if continuum else "frequency"
    dims = ("time", plane, "polarization", "l", "m")
    psf_dims = ("time", psf_plane, "polarization", "l", "m")
    shape = (1, 1, 1, 2, 2)
    return xr.Dataset(
        {
            "SKY_RESIDUAL": (dims, np.ones(shape)),
            "SKY_MODEL": (dims, np.zeros(shape)),
            "POINT_SPREAD_FUNCTION": (psf_dims, np.ones(shape)),
            "MASK": (dims, np.full(shape, value, dtype=dtype)),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (("time", "polarization"), [[0.1]]),
        },
        coords={
            "time": [0],
            plane: [0] if continuum else [1e9],
            psf_plane: [0] if continuum else [1e9],
            "polarization": ["I"],
            "l": [-0.1, 0.1],
            "m": [-0.1, 0.1],
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                    "mask": "MASK",
                },
                "model": {"sky": "SKY_MODEL"},
            }
        },
    )


@pytest.mark.parametrize("mode", [CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_product_masks_reject_before_modifying_images(mode, value):
    image = image_with_mask(mode, value)
    before = image.copy(deep=True)
    with pytest.raises(ValueError, match="Mask contains non-finite"):
        check_model_products(image, {"deconvolver": "hogbom"}, mode=mode)
    xr.testing.assert_identical(image, before)


@pytest.mark.parametrize("mode", [CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD])
@pytest.mark.parametrize("checked", [False, True])
def test_direct_model_updates_always_reject_nonfinite_masks(mode, checked):
    image = image_with_mask(mode, np.nan)
    before = image.copy(deep=True)
    with pytest.raises(ValueError, match="Mask contains non-finite"):
        if mode == CONTINUUM_SINGLE_FIELD:
            model_update_mtmfs_single_field(image, "hogbom", {}, input_checked=checked)
        else:
            model_update_cube_single_field(
                image, "hogbom", {}, False, input_checked=checked
            )
    xr.testing.assert_identical(image, before)


@pytest.mark.parametrize("mode", [CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD])
@pytest.mark.parametrize("dtype", [bool, np.float64])
def test_empty_finite_derived_support_is_valid(mode, dtype):
    image = image_with_mask(mode, 0, dtype)
    before = image.copy(deep=True)
    check_model_products(image, {"deconvolver": "hogbom"}, mode=mode)
    xr.testing.assert_identical(image, before)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_mask_installation_cannot_hide_nonfinite_values_by_casting(value):
    image = image_with_mask(CONTINUUM_SINGLE_FIELD, True, bool)
    before = image.copy(deep=True)
    with pytest.raises(ValueError, match="clean mask contains non-finite"):
        install_clean_mask(image, np.full((2, 2), value))
    xr.testing.assert_identical(image, before)
