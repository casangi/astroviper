"""Model-update configuration and derived-product refusal contracts."""

from copy import deepcopy

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_model_update,
)


@pytest.mark.parametrize(
    "control,value",
    [
        ("threshold", [1.0, 2.0, 3.0]),
        ("threshold", np.ones((1, 1, 2))),
        ("threshold_per_cycle", np.ones((1, 2))),
        ("threshold_per_cycle", np.ones((1, 2, 2))),
        ("max_iter_per_cycle_updated", np.ones((2, 1, 2), dtype=int)),
        ("max_iter", [1, 2]),
    ],
)
def test_control_shapes_rejected(control, value):
    config = {
        "image_params": {"time_coords": [0.0], "polarization_coords": ["I", "Q"]},
        "cycle_control_params": {control: value},
    }
    with pytest.raises(ValueError, match=control):
        check_model_update(config, mode=CONTINUUM_SINGLE_FIELD)


def test_supported_control_shapes_preserve_arrays():
    values = np.ones((1, 1, 2))
    values.flags.writeable = False
    config = {
        "image_params": {"time_coords": [0.0], "polarization_coords": ["I", "Q"]},
        "cycle_control_params": {
            "threshold": [0.0, 1.0],
            "threshold_per_cycle": values,
        },
    }
    check_model_update(config, mode=CONTINUUM_SINGLE_FIELD)
    assert config["cycle_control_params"]["threshold_per_cycle"] is values


@pytest.mark.parametrize(
    "options",
    [
        {"psf_fitting_method": "bad"},
        {"npix_window": [4, 5]},
        {"npix_window": [3]},
        {"sampling": [1, 2]},
        {"sampling": [2.0, 3.0]},
        {"cutoff": np.nan},
        {"psf_cube": np.zeros((1,)), "psf_normalization": None},
    ],
)
def test_fitting_and_direct_helper_controls(options):
    with pytest.raises((ValueError, TypeError)):
        check_model_update(options, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize(
    "config,message",
    [
        ({"cutoff": -0.1}, "cutoff"),
        ({"image_params": {"n_taylor_terms": 2}}, "n_taylor_terms=1"),
        ({"cycle_control_params": {"min_psf_fraction": -0.1}}, "min_psf_fraction"),
        ({"cycle_control_params": {"max_psf_fraction": 1.1}}, "max_psf_fraction"),
        ({"cycle_control_params": {"mask_threshold": 1.1}}, "mask_threshold"),
        (
            {"deconvolver": "hogbom", "cycle_control_params": {"scales": [0.0]}},
            "scales.*mtmfs",
        ),
        (
            {
                "deconvolver": "hogbom",
                "cycle_control_params": {"small_scale_bias": 0.5},
            },
            "MT-MFS controls",
        ),
        (
            {
                "deconvolver": "asp",
                "cycle_control_params": {"asp_scale_normalization_method": 3},
            },
            "normalization_method",
        ),
    ],
)
def test_model_option_refusals_are_specific_and_nonmutating(config, message):
    original = deepcopy(config)
    with pytest.raises(ValueError, match=message):
        check_model_update(config, mode=CUBE_SINGLE_FIELD)
    assert config == original


@pytest.mark.parametrize("size", [[0, 32], [32], [[32, 32]]])
def test_scale_support_rejects_invalid_image_extents(size):
    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_scales,
    )

    with pytest.raises(ValueError, match="image_size"):
        check_scales([0.0], image_size=size)


def cube_products():
    import xarray as xr

    dims = ("time", "frequency", "polarization", "l", "m")
    image = xr.Dataset(
        {
            name: (dims, np.zeros((1, 1, 1, 8, 8), dtype=np.float64))
            for name in ("SKY_RESIDUAL", "SKY_MODEL", "POINT_SPREAD_FUNCTION")
        },
        coords={
            "time": [0.0],
            "frequency": [1e9],
            "polarization": ["I"],
            "l": np.arange(8),
            "m": np.arange(8),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                },
                "model": {"sky": "SKY_MODEL"},
            }
        },
    )
    image.POINT_SPREAD_FUNCTION.values[0, 0, 0, 4, 4] = 1.0
    return image


@pytest.mark.parametrize(
    "fault,message",
    [
        ("empty", "nonempty dimensions"),
        ("residual_dtype", "float32 or float64"),
        ("configured_dtype", "differs from configured"),
        ("psf_layout", "dimensions.*planes"),
        ("psf_dtype", "working precision"),
        ("model_layout", "residual/model dimensions"),
        ("model_dtype", "working precision"),
        ("image_size", "image geometry differs"),
        ("polarization", "polarization coordinates differ"),
        ("missing_mask", "Mask variable"),
        ("mask_layout", "Mask dimensions"),
        ("control_layout", "per-plane shape"),
        ("negative_threshold", "invalid per-plane limits"),
        ("fractional_iterations", "invalid per-plane limits"),
        ("frequency_nan", "incompatible frequency coordinates"),
        ("spatial_nan", "incompatible l coordinates"),
    ],
)
def test_cube_product_refusals_preserve_actual_arrays(fault, message):
    import xarray as xr

    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_model_products,
    )

    image = cube_products()
    config = {}
    if fault == "empty":
        image = image.isel(l=slice(0, 0))
    elif fault == "residual_dtype":
        image["SKY_RESIDUAL"] = image.SKY_RESIDUAL.astype(np.int64)
    elif fault == "configured_dtype":
        config["single_precision_image"] = True
    elif fault == "psf_layout":
        image["POINT_SPREAD_FUNCTION"] = image.POINT_SPREAD_FUNCTION.isel(
            frequency=0, drop=True
        )
    elif fault == "psf_dtype":
        image["POINT_SPREAD_FUNCTION"] = image.POINT_SPREAD_FUNCTION.astype(np.float32)
    elif fault == "model_layout":
        image["SKY_MODEL"] = image.SKY_MODEL.isel(frequency=0, drop=True)
    elif fault == "model_dtype":
        image["SKY_MODEL"] = image.SKY_MODEL.astype(np.float32)
    elif fault == "image_size":
        config["image_params"] = {"image_size": [9, 8]}
    elif fault == "polarization":
        config["image_params"] = {"polarization_coords": ["Q"]}
    elif fault in ("missing_mask", "mask_layout"):
        image.attrs["data_groups"]["residual"]["mask"] = "MASK"
        if fault == "mask_layout":
            image["MASK"] = xr.DataArray([True], dims="extra")
    elif fault == "frequency_nan":
        image = image.assign_coords(frequency=[np.nan])
    elif fault == "spatial_nan":
        image = image.assign_coords(l=np.full(8, np.nan))
    else:
        name, value = {
            "control_layout": ("threshold_per_cycle", [0, 1]),
            "negative_threshold": ("threshold_per_cycle", -1.0),
            "fractional_iterations": ("max_iter_per_cycle_updated", 1.5),
        }[fault]
        config["cycle_control_params"] = {name: value}
    snapshot = image.copy(deep=True)
    with pytest.raises((ValueError, TypeError, KeyError), match=message):
        check_model_products(image, config, mode=CUBE_SINGLE_FIELD)
    xr.testing.assert_identical(image, snapshot)


@pytest.mark.parametrize("axis", ["taylor_term", "psf_taylor_order"])
def test_taylor_product_coordinate_order_must_match_the_validated_basis(axis):
    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_model_products,
    )
    from tests.utils.continuum_images import make_runaway_taylor_image

    image = make_runaway_taylor_image()
    image = image.assign_coords({axis: np.arange(image.sizes[axis]) + 1})
    with pytest.raises(ValueError, match="coordinates must be exactly"):
        check_model_products(
            image, {"deconvolver": "mtmfs"}, mode=CONTINUUM_SINGLE_FIELD
        )


def test_product_reference_frequency_must_equal_the_checked_reference():
    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_model_products,
    )
    from tests.utils.continuum_images import make_runaway_taylor_image

    image = make_runaway_taylor_image()
    image.attrs["reference_frequency"] = 1e9
    with pytest.raises(ValueError, match="Product reference frequency differs"):
        check_model_products(
            image,
            {"deconvolver": "mtmfs", "image_params": {"reference_frequency": 1.1e9}},
            mode=CONTINUUM_SINGLE_FIELD,
        )


def test_hessian_cache_requires_a_mutable_mapping():
    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_model_products,
    )
    from tests.utils.continuum_images import make_runaway_taylor_image

    with pytest.raises(TypeError, match="hessian_cache must be a mutable mapping"):
        check_model_products(
            make_runaway_taylor_image(),
            {"deconvolver": "mtmfs"},
            mode=CONTINUUM_SINGLE_FIELD,
            hessian_cache=[],
        )


@pytest.mark.parametrize(
    "fault,message",
    [
        ("nonfinite", "finite with positive diagonals"),
        ("diagonal", "positive diagonals"),
        ("indefinite", "not positive definite"),
        ("scales", "changed the checked effective scales"),
        ("condition", "non-finite condition numbers"),
        ("inverse", "inverse order-zero diagonals"),
    ],
)
def test_corrupt_native_hessian_results_fail_without_changing_products(
    monkeypatch, fault, message
):
    import xarray as xr

    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_model_products,
    )
    from astroviper.processing_functions.imaging.deconvolvers import mtmfs
    from tests.utils.continuum_images import make_runaway_taylor_image

    image = make_runaway_taylor_image()
    snapshot = image.copy(deep=True)
    result = mtmfs.hessian(
        np.ascontiguousarray(image.POINT_SPREAD_FUNCTION.values[0, :, 0]), scales=[0.0]
    )
    if fault == "nonfinite":
        result["hessian"][0, 0, 0] = np.nan
    elif fault == "diagonal":
        result["hessian"][0, 0, 0] = 0
    elif fault == "indefinite":
        result["hessian"][0] = [[1, 2], [2, 1]]
    elif fault == "scales":
        result["scales"] = [1.0]
    elif fault == "condition":
        result["hessian_condition_number"] = [np.nan]
    else:
        result["inverse_hessian"][0, 0, 0] = -1
    monkeypatch.setattr(mtmfs, "hessian", lambda *args, **kwargs: deepcopy(result))
    with pytest.raises(ValueError, match=message):
        check_model_products(
            image, {"deconvolver": "mtmfs"}, mode=CONTINUUM_SINGLE_FIELD
        )
    xr.testing.assert_identical(image, snapshot)


@pytest.mark.parametrize(
    "name", ["min_psf_fraction", "max_psf_fraction", "mask_threshold"]
)
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_fraction_policy_keeps_its_inclusive_endpoints(name, value):
    controls = {name: value}
    # Keep the interval ordered at either endpoint when varying one bound.
    if name == "min_psf_fraction":
        controls["max_psf_fraction"] = 1.0
    if name == "max_psf_fraction":
        controls["min_psf_fraction"] = 0.0
    check_model_update({"cycle_control_params": controls}, mode=CUBE_SINGLE_FIELD)
    assert controls[name] == value


def test_zero_fitting_cutoff_is_supported():
    check_model_update({"cutoff": 0.0}, mode=CUBE_SINGLE_FIELD)
