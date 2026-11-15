"""Shared runtime Taylor layouts, independent of configuration and pixel values."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    check_continuum_taylor_layout,
)


@pytest.fixture(params=[("taylor_term", 2), ("psf_taylor_order", 3)])
def stack(request):
    dimension, count = request.param
    product = xr.DataArray(
        np.ones((2, count, 2, 3, 4)),
        dims=("time", dimension, "polarization", "l", "m"),
    )
    return product, dimension, count


def test_accepts_reordered_axes_without_changing_storage(stack):
    product, dimension, count = stack
    product = product.transpose("m", "polarization", dimension, "time", "l")
    snapshot = product.copy(deep=True)
    original_data = product.data
    check_continuum_taylor_layout(
        product, name="test product", term_dimension=dimension, expected_count=count
    )
    xr.testing.assert_identical(product, snapshot)
    assert product.data is original_data


@pytest.mark.parametrize(
    "fault", ["missing_term", "missing_time", "extra", "empty", "count"]
)
def test_rejects_inconsistent_layout(stack, fault):
    product, dimension, count = stack
    if fault == "missing_term":
        product = product.isel({dimension: 0}, drop=True)
    elif fault == "missing_time":
        product = product.isel(time=0, drop=True)
    elif fault == "extra":
        product = product.expand_dims(extra=[0])
    elif fault == "empty":
        product = product.isel({dimension: slice(0, 0)})
    elif fault == "count":
        count += 1
    with pytest.raises(ValueError, match="test product.*(dimensions|planes)"):
        check_continuum_taylor_layout(
            product, name="test product", term_dimension=dimension, expected_count=count
        )


def test_checks_metadata_without_loading_pixels(stack):
    product, dimension, count = stack

    @delayed
    def forbidden_read():
        pytest.fail("Taylor layout validation loaded image pixels")

    product = product.copy(
        data=da.from_delayed(forbidden_read(), shape=product.shape, dtype=float)
    )
    original_data = product.data
    check_continuum_taylor_layout(
        product, name="lazy product", term_dimension=dimension, expected_count=count
    )
    assert product.data is original_data


def test_model_product_validation_uses_shared_residual_and_psf_checks(monkeypatch):
    import astroviper.processing_functions.imaging.image_continuum_single_field as processing
    from astroviper.processing_functions.imaging.config_check import (
        CONTINUUM_SINGLE_FIELD,
    )
    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_model_products,
    )
    from tests.utils.continuum_images import make_runaway_taylor_image

    calls = []

    class CheckedBothProducts(Exception):
        pass

    def check(product, **kwargs):
        check_continuum_taylor_layout(product, **kwargs)
        calls.append((kwargs["term_dimension"], kwargs["expected_count"]))
        if len(calls) == 2:
            raise CheckedBothProducts

    monkeypatch.setattr(processing, "check_continuum_taylor_layout", check)
    with pytest.raises(CheckedBothProducts):
        check_model_products(
            make_runaway_taylor_image(),
            {"deconvolver": "mtmfs", "image_params": {"n_taylor_terms": 2}},
            mode=CONTINUUM_SINGLE_FIELD,
        )
    assert calls == [("taylor_term", 2), ("psf_taylor_order", 3)]
