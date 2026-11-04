"""Shared restoration roles preserve operation-specific product requirements."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    restore_continuum_image,
)
from astroviper.processing_functions.imaging.restore import (
    resolve_restoration_product,
    restore_image,
)
from tests.unit.processing_functions.imaging.test_restore import _make_restore_xds
from tests.utils.continuum_images import make_continuum_image


@pytest.mark.parametrize("fault", ["group", "role", "variable"])
def test_role_resolution_rejects_incomplete_metadata_without_mutation(fault):
    image = xr.Dataset(
        {"MODEL": ("pixel", [1.0])}, attrs={"data_groups": {"model": {"sky": "MODEL"}}}
    )
    if fault == "group":
        del image.attrs["data_groups"]["model"]
    elif fault == "role":
        del image.attrs["data_groups"]["model"]["sky"]
    else:
        image = image.drop_vars("MODEL")
    snapshot = image.copy(deep=True)
    with pytest.raises(KeyError, match="missing|does not define"):
        resolve_restoration_product(image, "model", "sky")
    xr.testing.assert_identical(image, snapshot)


def test_explicit_legacy_fallback_does_not_insert_registration():
    image = xr.Dataset(
        {"BEAM": ("parameter", [1.0])}, attrs={"data_groups": {"residual": {}}}
    )
    snapshot = image.copy(deep=True)
    assert (
        resolve_restoration_product(image, "residual", "beam", fallback="BEAM")
        == "BEAM"
    )
    xr.testing.assert_identical(image, snapshot)
    image.attrs["data_groups"]["residual"]["beam"] = "MISSING_BEAM"
    with pytest.raises(KeyError, match="MISSING_BEAM"):
        resolve_restoration_product(image, "residual", "beam", fallback="BEAM")


def test_role_resolution_does_not_load_pixels():
    @delayed
    def forbidden_read():
        pytest.fail("role resolution loaded image pixels")

    pixels = da.from_delayed(forbidden_read(), shape=(2,), dtype=float)
    image = xr.Dataset(
        {"MODEL": ("pixel", pixels)}, attrs={"data_groups": {"model": {"sky": "MODEL"}}}
    )
    storage = image.MODEL.data
    assert resolve_restoration_product(image, "model", "sky") == "MODEL"
    assert image.MODEL.data is storage


@pytest.mark.parametrize("kind", ["cube", "mfs", "mvc"])
@pytest.mark.parametrize(
    "group,role",
    [
        ("residual", "sky"),
        ("model", "sky"),
        ("residual", "beam_fit_params_point_spread_function"),
        ("residual", "primary_beam"),
    ],
)
def test_bad_registered_products_fail_before_restoration_writes(kind, group, role):
    if kind == "cube":
        image = _make_restore_xds(ny=8, nx=8)
        image["PRIMARY_BEAM"] = xr.ones_like(image.SKY_RESIDUAL)
        image.attrs["data_groups"]["residual"]["primary_beam"] = "PRIMARY_BEAM"
    else:
        image = make_continuum_image(continuum_mode=kind)
    if kind == "mfs" and role == "primary_beam":
        # MFS corrects the Taylor polynomial; MVC and cube use the common beam.
        role = "primary_beam_taylor"
    # Keep the conventional variable present: an invalid registration must not
    # be hidden by continuum's compatibility fallback.
    image.attrs["data_groups"][group][role] = "MISSING_PRODUCT"
    snapshot = image.copy(deep=True)
    with pytest.raises(KeyError, match="MISSING_PRODUCT"):
        if kind == "cube":
            restore_image(image, primary_beam_correction=True, consume_model=True)
        else:
            restore_continuum_image(
                image, restoration_output="both", input_checked=True
            )
    xr.testing.assert_identical(image, snapshot)


@pytest.mark.parametrize("kind", ["cube", "mfs", "mvc"])
def test_primary_beam_role_is_required_only_for_correction(kind):
    if kind == "cube":
        image = _make_restore_xds(ny=8, nx=8)
        assert "POINT_SPREAD_FUNCTION" not in image
        restore = restore_image
    else:
        image = make_continuum_image(continuum_mode=kind)
        restore = restore_continuum_image
    image.attrs["data_groups"]["residual"]["primary_beam"] = "MISSING_UNUSED_BEAM"
    if kind == "mfs":
        image.attrs["data_groups"]["residual"]["primary_beam_taylor"] = (
            "MISSING_UNUSED_BEAM"
        )
    restored, _ = (
        restore(image, primary_beam_correction=False)
        if kind == "cube"
        else restore(image, restoration_output="apparent")
    )
    restored_group = restored.attrs["data_groups"]["restored"]
    assert np.all(np.isfinite(restored[restored_group["sky"]]))
    assert "sky_primary_beam_corrected" not in restored_group
    assert "sky_flat_sky" not in restored_group
