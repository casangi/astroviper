"""Unit tests for astroviper.processing_functions.image_analysis.make_mask.make_mask."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.make_mask import make_mask


def _build_img_xds(shape=(1, 1, 1, 64, 64), pb_var="PRIMARY_BEAM", mask_var=None):
    """Build a minimal image Dataset with a Gaussian primary beam, an
    optional pre-existing boolean mask, and a ``data_groups`` attribute
    referencing them under the ``"image"`` group."""
    l = np.linspace(-0.05, 0.05, shape[4])
    m = np.linspace(-0.05, 0.05, shape[3])
    L, M = np.meshgrid(l, m, indexing="ij")
    sigma = 0.02
    pb_plane = np.exp(-((L**2 + M**2) / (2 * sigma**2)))
    pb_data = np.zeros(shape)
    pb_data[0, 0, 0, :, :] = pb_plane

    coords = {
        "time": np.arange(shape[0]),
        "frequency": np.arange(shape[1]),
        "polarization": np.arange(shape[2]),
        "l": l,
        "m": m,
    }
    dims = ["time", "frequency", "polarization", "l", "m"]

    xds = xr.Dataset()
    xds[pb_var] = xr.DataArray(pb_data, coords=coords, dims=dims)

    data_group = {"primary_beam": pb_var}
    if mask_var is not None:
        # Pre-populate a boolean mask so combine_mask tests have something
        # to OR with.  A small rectangle is set True; everything else False.
        mask_data = np.zeros(shape, dtype=bool)
        mask_data[0, 0, 0, 20:40, 30:50] = True
        xds[mask_var] = xr.DataArray(mask_data, coords=coords, dims=dims)
        data_group["mask"] = mask_var

    xds.attrs["data_groups"] = {"image": data_group}
    return xds


def test_make_mask_threshold_only():
    """combine_mask=False produces (primary_beam >= threshold * peak)."""
    img_xds = _build_img_xds()
    threshold = 0.1
    pb = img_xds["PRIMARY_BEAM"].values
    expected = (pb >= threshold * np.nanmax(pb)).astype(bool)

    result = make_mask(img_xds, threshold=threshold)

    # Function mutates img_xds in place and returns None.
    assert result is None
    assert img_xds["MASK"].dtype == bool
    np.testing.assert_array_equal(img_xds["MASK"].values, expected)


def test_make_mask_registers_output_data_group():
    """The output data group is registered with mask role, date and description."""
    img_xds = _build_img_xds()
    threshold = 0.2

    make_mask(img_xds, threshold=threshold)

    groups = img_xds.attrs["data_groups"]
    assert "image" in groups
    image_group = groups["image"]
    assert image_group["primary_beam"] == "PRIMARY_BEAM"
    assert image_group["mask"] == "MASK"
    assert "date" in image_group
    assert f"threshold {threshold}" in image_group["description"]


def test_make_mask_custom_output_mask_name():
    """The mask data variable picks up the name from data_group_out_modified."""
    img_xds = _build_img_xds()

    make_mask(
        img_xds,
        threshold=0.5,
        image_data_group_out_modified={"mask": "MY_MASK"},
    )

    assert "MY_MASK" in img_xds.data_vars
    assert "MASK" not in img_xds.data_vars
    assert img_xds.attrs["data_groups"]["image"]["mask"] == "MY_MASK"


def test_make_mask_separate_output_data_group():
    """Output stored under a different group key leaves the input group intact."""
    img_xds = _build_img_xds()

    make_mask(
        img_xds,
        threshold=0.3,
        image_data_group_out_name="image_masked",
    )

    groups = img_xds.attrs["data_groups"]
    assert "image" in groups and "image_masked" in groups
    # Input data group untouched (no mask role added).
    assert "mask" not in groups["image"]
    # Output data group inherits primary_beam and adds mask.
    assert groups["image_masked"]["primary_beam"] == "PRIMARY_BEAM"
    assert groups["image_masked"]["mask"] == "MASK"


def test_make_mask_combine_with_existing_mask():
    """combine_mask=True yields logical OR of the input mask and the threshold mask."""
    # Distinct names for input vs output mask so overwrite=False is fine.
    img_xds = _build_img_xds(mask_var="MASK_IN")
    threshold = 0.9
    pb = img_xds["PRIMARY_BEAM"].values
    existing = img_xds["MASK_IN"].values
    expected = np.logical_or(existing, pb >= threshold * np.nanmax(pb))

    make_mask(img_xds, threshold=threshold, combine_mask=True)

    np.testing.assert_array_equal(img_xds["MASK"].values, expected)
    # Input mask is preserved as a separate data variable.
    np.testing.assert_array_equal(img_xds["MASK_IN"].values, existing)


def test_make_mask_overwrite_false_blocks_existing_mask_var():
    """When the output mask data variable already exists, overwrite=False raises."""
    img_xds = _build_img_xds(mask_var="MASK")
    with pytest.raises(AssertionError):
        make_mask(img_xds, threshold=0.1)


def test_make_mask_overwrite_true_replaces_existing_mask_var():
    """overwrite=True allows the pre-existing mask data variable to be replaced."""
    img_xds = _build_img_xds(mask_var="MASK")
    threshold = 0.1
    pb = img_xds["PRIMARY_BEAM"].values
    expected = (pb >= threshold * np.nanmax(pb)).astype(bool)

    make_mask(img_xds, threshold=threshold, overwrite=True)

    np.testing.assert_array_equal(img_xds["MASK"].values, expected)


def test_make_mask_missing_input_data_group_raises():
    """An unknown input data group name raises an AssertionError."""
    img_xds = _build_img_xds()
    with pytest.raises(AssertionError):
        make_mask(img_xds, threshold=0.1, image_data_group_in_name="does_not_exist")


def test_make_mask_handles_nan_in_primary_beam():
    """NaNs in the primary beam do not break the threshold (nanmax is used)."""
    img_xds = _build_img_xds()
    # Inject a NaN into the primary beam — np.nanmax must ignore it.
    img_xds["PRIMARY_BEAM"].values[0, 0, 0, 0, 0] = np.nan
    threshold = 0.1
    pb = img_xds["PRIMARY_BEAM"].values
    expected = (pb >= threshold * np.nanmax(pb)).astype(bool)

    make_mask(img_xds, threshold=threshold)

    np.testing.assert_array_equal(img_xds["MASK"].values, expected)


def test_make_mask_description_appends_on_repeat_call():
    """A second call with overwrite=True appends to the existing audit-trail description."""
    img_xds = _build_img_xds()

    make_mask(img_xds, threshold=0.1)
    first_description = img_xds.attrs["data_groups"]["image"]["description"]

    make_mask(img_xds, threshold=0.5, overwrite=True)
    second_description = img_xds.attrs["data_groups"]["image"]["description"]

    # modify_data_groups_xds appends with "; " when a description already exists.
    assert second_description.startswith(first_description + "; ")
    assert "threshold 0.5" in second_description
