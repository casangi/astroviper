"""Model-update products and derived controls must fail before any image writes."""

import copy

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

import astroviper.processing_functions.imaging.image_continuum_single_field as processing
from tests.utils.continuum_images import make_runaway_taylor_image


@pytest.mark.parametrize("input_checked", [False, True])
@pytest.mark.parametrize(
    "fault, message",
    [
        ("residual_precision", "float32 or float64"),
        ("psf_precision", "working precision"),
        ("psf_count", "planes"),
        ("model_layout", "dimensions"),
        ("control_shape", "threshold_per_cycle"),
        ("fractional_iterations", "max_iter_per_cycle_updated"),
        ("negative_threshold", "threshold_per_cycle"),
        ("nonfinite_threshold", "finite"),
    ],
)
def test_invalid_runtime_products_precede_backend_and_mutation(
    monkeypatch, input_checked, fault, message
):
    image = make_runaway_taylor_image()
    controls = {}
    if fault == "residual_precision":
        image["SKY_RESIDUAL"] = image.SKY_RESIDUAL.astype(np.int64)
    elif fault == "psf_precision":
        image["POINT_SPREAD_FUNCTION"] = image.POINT_SPREAD_FUNCTION.astype(np.float32)
    elif fault == "psf_count":
        image = image.isel(psf_taylor_order=slice(0, 2))
    elif fault == "model_layout":
        image["SKY_MODEL"] = image.SKY_RESIDUAL.isel(taylor_term=0, drop=True)
    elif fault == "control_shape":
        controls["threshold_per_cycle"] = np.zeros((1, 2, 1))
    elif fault == "fractional_iterations":
        controls["max_iter_per_cycle_updated"] = np.array([[[1.5]]])
    elif fault == "negative_threshold":
        controls["threshold_per_cycle"] = -1.0
    else:
        controls["threshold_per_cycle"] = np.nan
    original_image = image.copy(deep=True)
    original_controls = copy.deepcopy(controls)

    def unexpected(*args, **kwargs):
        pytest.fail("backend entered before rejecting invalid runtime products")

    monkeypatch.setattr(processing, "_model_update_native_mtmfs", unexpected)
    with pytest.raises((TypeError, ValueError), match=message):
        processing.model_update_mtmfs_single_field(
            image, "mtmfs", controls, input_checked=input_checked
        )
    xr.testing.assert_identical(image, original_image)
    for name, value in controls.items():
        np.testing.assert_array_equal(value, original_controls[name])


def test_shared_product_validation_does_not_load_pixels():
    image = make_runaway_taylor_image()
    products = []

    @delayed
    def forbidden_read():
        pytest.fail("model-product metadata validation loaded image pixels")

    for product in (
        image.SKY_RESIDUAL,
        image.POINT_SPREAD_FUNCTION,
        image.SKY_RESIDUAL.rename("SKY_MODEL"),
    ):
        products.append(
            product.copy(
                data=da.from_delayed(
                    forbidden_read(), shape=product.shape, dtype=product.dtype
                )
            )
        )
    storage = [product.data for product in products]
    processing.check_continuum_model_products(
        products[0], products[1], deconvolver="mtmfs", model=products[2]
    )
    assert all(
        product.data is original
        for product, original in zip(products, storage, strict=True)
    )


@pytest.mark.parametrize("product", ["psf", "model"])
@pytest.mark.parametrize("fault", ["shape", "coordinates"])
def test_product_alignment_requires_same_sky_grid(product, fault):
    image = make_runaway_taylor_image()
    residual = image.SKY_RESIDUAL
    psf = image.POINT_SPREAD_FUNCTION
    model = residual.rename("SKY_MODEL")
    candidate = psf if product == "psf" else model
    if fault == "shape":
        candidate = candidate.isel(l=slice(0, -1))
    else:
        candidate = candidate.assign_coords(l=candidate.l + 1.0)
    with pytest.raises(ValueError, match="incompatible l"):
        processing.check_continuum_model_products(
            residual,
            candidate if product == "psf" else psf,
            deconvolver="mtmfs",
            model=candidate if product == "model" else model,
        )


@pytest.mark.parametrize("scalar", [False, True])
def test_valid_controls_preserve_scalar_and_plane_arrays(scalar):
    image = make_runaway_taylor_image()
    controls = {
        "threshold_per_cycle": 0.1 if scalar else np.array([[[0.1]]]),
        "max_iter_per_cycle_updated": 2 if scalar else np.array([[[2]]]),
    }
    originals = {name: value for name, value in controls.items()}
    processing.check_continuum_model_controls(controls, image.SKY_RESIDUAL)
    assert all(controls[name] is value for name, value in originals.items())
