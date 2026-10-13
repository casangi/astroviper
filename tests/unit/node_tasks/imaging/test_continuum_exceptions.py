"""Missing continuum state must fail explicitly before further computation."""

import numpy as np
import pytest
import xarray as xr

from astroviper.node_tasks.imaging.image_continuum_single_field import (
    _finalize_reference_primary_beam,
    _install_continuum_clean_mask,
    _prepare_cached_mfs_residual_grid,
    continuum_finalize_node,
    model_update_continuum_single_field,
)


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("mode", ValueError, "visibility_memory_mode"),
        ("specmode", ValueError, "specmode"),
        ("group", KeyError, "data group"),
        ("role", KeyError, "register both"),
        ("variable", KeyError, "missing variables"),
        ("initial_store", KeyError, "image_store"),
        ("later_store", KeyError, "image_store"),
        ("later_cache", KeyError, "cached observed-data"),
    ],
)
def test_cached_mfs_append_rejects_incomplete_state(fault, error, message):
    image = xr.Dataset(
        {"GRID": ("u", [1 + 0j]), "SUM": ((), 1.0)},
        attrs={
            "data_groups": {
                "residual": {"visibility": "GRID", "visibility_normalization": "SUM"}
            }
        },
    )
    params = {"visibility_memory_mode": "in_memory", "specmode": "mfs"}
    if fault == "mode":
        params["visibility_memory_mode"] = "unsupported"
    elif fault == "specmode":
        params["specmode"] = "cube"
    elif fault == "group":
        image.attrs["data_groups"] = {}
    elif fault == "role":
        del image.attrs["data_groups"]["residual"]["visibility_normalization"]
    elif fault == "variable":
        image = image.drop_vars("SUM")
    elif fault in ("initial_store", "later_store"):
        params["visibility_memory_mode"] = "in_place"
        params["is_n_iter_0"] = fault == "initial_store"
    else:
        params["is_n_iter_0"] = False
    snapshot = image.copy(deep=True)
    with pytest.raises(error, match=message):
        _prepare_cached_mfs_residual_grid({"image": image}, params)
    xr.testing.assert_identical(image, snapshot)


@pytest.mark.parametrize(
    "missing,message",
    [
        ("static_xds", "static_xds"),
        ("model_xds", "model_xds"),
        ("SKY_MODEL", "SKY_MODEL"),
    ],
)
def test_finalization_rejects_missing_accumulated_state(missing, message):
    params = {
        "static_xds": xr.Dataset(),
        "model_xds": xr.Dataset(),
        "prepared_continuum_image": True,
    }
    if missing != "SKY_MODEL":
        del params[missing]
    with pytest.raises(KeyError, match=message):
        continuum_finalize_node({"image": xr.Dataset()}, params)


@pytest.mark.parametrize(
    "data,params,error,message",
    [
        ([], {}, ValueError, "exactly one"),
        ([{}, {}], {}, ValueError, "exactly one"),
        (None, {}, TypeError, "dictionary"),
        ({}, {}, KeyError, "image"),
        ({"image": xr.Dataset()}, {}, KeyError, "iteration_control_params"),
    ],
)
def test_model_update_node_rejects_invalid_inputs(data, params, error, message):
    with pytest.raises(error, match=message):
        model_update_continuum_single_field(data, params)


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("residual", KeyError, "SKY_RESIDUAL"),
        ("dimensions", ValueError, "dimensions"),
        ("shape", ValueError, "shape"),
        ("group", KeyError, "data group"),
    ],
)
def test_clean_mask_installation_rejects_incompatible_image(fault, error, message):
    image = xr.Dataset(
        {"SKY_RESIDUAL": (("l", "m"), np.ones((2, 2)))},
        coords={"l": [0, 1], "m": [0, 1]},
        attrs={"data_groups": {"residual": {"sky": "SKY_RESIDUAL"}}},
    )
    mask = np.ones((2, 2), dtype=bool)
    if fault == "residual":
        image = image.drop_vars("SKY_RESIDUAL")
    elif fault == "dimensions":
        image = image.rename({"l": "other"})
    elif fault == "shape":
        mask = np.ones((3, 3), dtype=bool)
    else:
        image.attrs["data_groups"] = {}
    with pytest.raises(error, match=message):
        _install_continuum_clean_mask(image, mask)


@pytest.mark.parametrize("dependent_cube", [False, True])
def test_reference_beam_finalization_rejects_missing_or_unreduced_state(dependent_cube):
    image = xr.Dataset()
    if dependent_cube:
        image = xr.Dataset(
            {
                "PRIMARY_BEAM_REFERENCE": ("l", [1.0]),
                "CHANNEL_DATA": ("frequency", [1.0, 2.0]),
            }
        )
    with pytest.raises(
        ValueError if dependent_cube else KeyError,
        match="still depend|PRIMARY_BEAM_REFERENCE",
    ):
        _finalize_reference_primary_beam(image, reference_frequency=100.0)
