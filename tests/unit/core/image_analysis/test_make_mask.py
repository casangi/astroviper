"""
Unit tests for make_mask

"""

import pytest
import numpy as np
import xarray as xr
from astroviper.core.image_analysis.make_mask import make_mask

debug = False


# generate primary beam xarray.Dataset image using Gaussian function
# with data variable 'SKY' for now. In future once xradio image schema
# is updated it should be changed to use PRIMARY_BEAM data variable
def make_pbimage(shape=(1, 1, 1, 128, 128)):
    l = np.linspace(-0.05, 0.05, shape[4])
    m = np.linspace(-0.05, 0.05, shape[3])
    L, M = np.meshgrid(l, m, indexing="ij")
    sigma = 0.02
    gaussian_data = np.exp(-((L**2 + M**2) / (2 * sigma**2)))
    pb_data = np.zeros(shape)
    pb_data[0, 0, 0, :, :] = gaussian_data
    pb_image = xr.DataArray(
        pb_data,
        coords={
            "time": np.arange(shape[0]),
            "frequency": np.arange(shape[1]),
            "polarization": np.arange(shape[2]),
            "l": l,
            "m": m,
        },
        dims=["time", "frequency", "polarization", "l", "m"],
        name="SKY",
    )
    pb_xds = xr.Dataset()
    pb_xds["SKY"] = pb_image
    return pb_xds


# generate image with random sources with data variable 'SKY'.
def make_image(shape=(1, 1, 1, 128, 128)):
    l = np.linspace(-0.05, 0.05, shape[4])
    m = np.linspace(-0.05, 0.05, shape[3])
    image_data = np.random.rand(*shape)
    image = xr.DataArray(
        image_data,
        coords={
            "time": np.arange(shape[0]),
            "frequency": np.arange(shape[1]),
            "polarization": np.arange(shape[2]),
            "l": l,
            "m": m,
        },
        dims=["time", "frequency", "polarization", "l", "m"],
        name="SKY",
    )
    image_xds = xr.Dataset()
    image_xds["SKY"] = image
    return image_xds


# plotting function for debugging
def plot_image(image: xr.DataArray, title: str, xlabel: str, ylabel: str, block: bool):
    import matplotlib.pyplot as plt

    plt.imshow(
        image.values,
        origin="lower",
        extent=[
            image.coords["l"].min(),
            image.coords["l"].max(),
            image.coords["m"].min(),
            image.coords["m"].max(),
        ],
    )
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show(block=block)


# basic improper input tests
def test_make_mask_no_target():
    """Test make_mask with no target_image provided"""
    pb_image = make_pbimage()
    pb_threshold = 0.1
    with pytest.raises(
        ValueError, match="Target image must be provided to check dimensions"
    ):
        mask = make_mask(
            input_image=pb_image,
            threshold=pb_threshold,
            target_image=None,
            combine_mask=False,
        )


def test_make_mask_invalid_pb_threshold():
    """Test make_mask with invalid pb_threshold value"""
    pb_image = make_pbimage()
    target = make_image()
    invalid_thresholds = [-0.1, 1.5]
    for pb_threshold in invalid_thresholds:
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            mask = make_mask(
                input_image=pb_image,
                threshold=pb_threshold,
                target_image=target,
                combine_mask=False,
            )


def test_make_mask_dimension_mismatch():
    """Test make_mask with dimension mismatch between input_image and target_image"""
    pb_image = make_pbimage()
    target = make_image(shape=(1, 1, 1, 64, 64))
    pb_threshold = 0.1
    with pytest.raises(
        ValueError,
        match="Input mask image dimensions do not match target image dimensions",
    ):
        mask = make_mask(
            input_image=pb_image,
            threshold=pb_threshold,
            target_image=target,
            combine_mask=False,
        )


def test_make_mask_frequency_mismatch():
    """Test make_mask with frequency dimension mismatch between input_image and target_image"""
    pb_image = make_pbimage(shape=(1, 1, 1, 128, 128))
    target = make_image(shape=(1, 3, 1, 128, 128))
    pb_threshold = 0.1
    with pytest.raises(
        ValueError,
        match="Input mask image frequency dimension does not match target image frequency dimension",
    ):
        mask = make_mask(
            input_image=pb_image,
            threshold=pb_threshold,
            target_image=target,
            combine_mask=False,
        )


def test_make_mask_time_mismatch():
    """Test make_mask with time dimension mismatch between input_image and target_image"""
    pb_image = make_pbimage(shape=(1, 1, 1, 128, 128))
    target = make_image(shape=(2, 1, 1, 128, 128))
    pb_threshold = 0.1
    with pytest.raises(
        ValueError,
        match="Input mask image time dimension does not match target image time dimension",
    ):
        mask = make_mask(
            input_image=pb_image,
            threshold=pb_threshold,
            target_image=target,
            combine_mask=False,
        )


def test_make_mask_polarization_mismatch():
    """Test make_mask with polarization dimension mismatch between input_image and target_image"""
    pb_image = make_pbimage(shape=(1, 1, 1, 128, 128))
    target = make_image(shape=(1, 1, 4, 128, 128))
    pb_threshold = 0.1
    with pytest.raises(
        ValueError,
        match="Input mask image polarization dimension does not match target image polarization dimension",
    ):
        mask = make_mask(
            input_image=pb_image,
            threshold=pb_threshold,
            target_image=target,
            combine_mask=False,
        )


# test most basic combine_mask = False case
def test_make_mask_pb_thresh_only():
    """Test combine_mask = False case"""
    pb_image = make_pbimage()

    if debug:
        plot_image(
            pb_image["SKY"],
            "Primary Beam Image",
            "l (radians)",
            "m (radians)",
            block=debug,
        )

    target = make_image()

    pb_threshold = 0.1
    mask = make_mask(
        input_image=pb_image,
        threshold=pb_threshold,
        target_image=target,
        combine_mask=False,
    )
    expected_mask_data = (pb_image["SKY"].values >= pb_threshold).astype(bool)
    np.testing.assert_array_equal(mask["MASK"].values, expected_mask_data)

    if debug:
        plot_image(
            mask["MASK"],
            "Generated Mask Image",
            "l (radians)",
            "m (radians)",
            block=debug,
        )


def test_make_mask_combine():
    """Test combine_mask = True case without active_mask attribute"""
    # first make target image and add a rectangular mask in 'MASK' data variable
    target = make_image()
    target["MASK"] = xr.DataArray(
        np.zeros_like(target["SKY"].values, dtype=bool),
        coords=target["SKY"].coords,
        dims=target["SKY"].dims,
    )
    if debug:
        plot_image(
            target["MASK"],
            "Initial Mask in Target Image",
            "l (radians)",
            "m (radians)",
            block=debug,
        )
    # set a rectangular region to True
    target["MASK"].isel(l=slice(60, 80), m=slice(40, 80)).values[:] = True
    if debug:
        plot_image(
            target["MASK"],
            "Modified Mask in Target Image",
            "l (radians)",
            "m (radians)",
            block=debug,
        )
    # make target data xarray.Dataset
    target_xds = xr.Dataset()
    target_xds["SKY"] = target["SKY"]
    target_xds["MASK"] = target["MASK"]

    pb_image = make_pbimage()
    pb_threshold = 0.9
    mask = make_mask(
        input_image=pb_image,
        threshold=pb_threshold,
        target_image=target,
        combine_mask=True,
    )
    expected_mask_data = (pb_image["SKY"].values >= pb_threshold) & (
        target["MASK"].values
    )

    if debug:
        plot_image(
            target["MASK"],
            "Existing Mask in Target Image",
            "l (radians)",
            "m (radians)",
            block=debug,
        )
        plot_image(
            mask["MASK"],
            "Generated Combined Mask Image",
            "l (radians)",
            "m (radians)",
            block=debug,
        )

    np.testing.assert_array_equal(mask["MASK"].values, expected_mask_data)


def test_make_mask_apply_on_target():
    """Test make_mask with apply_on_target = True"""
    pb_image = make_pbimage()
    target = make_image()
    pb_threshold = 0.2
    mask = make_mask(
        input_image=pb_image,
        threshold=pb_threshold,
        target_image=target,
        apply_on_target=True,
        combine_mask=False,
    )
    expected_mask_data = (pb_image["SKY"].values >= pb_threshold).astype(bool)

    if debug:
        plot_image(
            mask["MASK"],
            "Generated Mask Image Applied on Target",
            "l (radians)",
            "m (radians)",
            block=debug,
        )

    np.testing.assert_array_equal(mask["MASK"].values, expected_mask_data)


def test_make_mask_apply_on_target_existing_mask():
    """Test make_mask with apply_on_target = True and existing mask in target image"""
    # first make target image and add a rectangular mask in 'MASK' data variable
    target = make_image()
    target["MASK"] = xr.DataArray(
        np.zeros_like(target["SKY"].values, dtype=bool),
        coords=target["SKY"].coords,
        dims=target["SKY"].dims,
    )
    # set a rectangular region to True
    target["MASK"].isel(l=slice(60, 80), m=slice(40, 80)).values[:] = True
    # make target data xarray.Dataset
    target_xds = xr.Dataset()
    target_xds["SKY"] = target["SKY"]
    target_xds["MASK"] = target["MASK"]

    pb_image = make_pbimage()
    pb_threshold = 0.9
    mask = make_mask(
        input_image=pb_image,
        threshold=pb_threshold,
        target_image=target_xds,
        apply_on_target=True,
        combine_mask=True,
    )
    expected_mask_data = (pb_image["SKY"].values >= pb_threshold) & (
        target["MASK"].values
    )

    np.testing.assert_array_equal(mask["MASK"].values, expected_mask_data)


def test_make_mask_active_mask_attribute():
    """Test make_mask with combine_mask = True and active_mask attribute"""
    # first make target image and add a rectangular mask in 'MASK' data variable
    target = make_image()
    target["MASK"] = xr.DataArray(
        np.zeros_like(target["SKY"].values, dtype=bool),
        coords=target["SKY"].coords,
        dims=target["SKY"].dims,
    )
    # set a rectangular region to True
    target["MASK"].isel(l=slice(60, 80), m=slice(40, 80)).values[:] = True
    # make target data xarray.Dataset
    target_xds = xr.Dataset()
    target_xds["SKY"] = target["SKY"]
    target_xds["MASK"] = target["MASK"]
    # set active_mask attribute
    target_xds["SKY"].attrs["active_mask"] = "MASK"

    pb_image = make_pbimage()
    pb_threshold = 0.9
    mask = make_mask(
        input_image=pb_image,
        threshold=pb_threshold,
        target_image=target_xds,
        combine_mask=True,
    )
    expected_mask_data = (pb_image["SKY"].values >= pb_threshold) & (
        target["MASK"].values
    )

    np.testing.assert_array_equal(mask["MASK"].values, expected_mask_data)


def test_make_mask_active_mask_attribute_casa_mask():
    """
    Test make_mask with combine_mask = True and active_mask attribute pointing to
    CASA mask definition
    """
    # first make target image and add a rectangular mask in 'MASK0' data variable
    target = make_image()
    target["MASK0"] = xr.DataArray(
        np.zeros_like(target["SKY"].values, dtype=bool),
        coords=target["SKY"].coords,
        dims=target["SKY"].dims,
    )
    # set a rectangular region to True
    target["MASK0"].isel(l=slice(60, 80), m=slice(40, 80)).values[:] = True
    # make target data xarray.Dataset
    target_xds = xr.Dataset()
    target_xds["SKY"] = target["SKY"]
    target_xds["MASK0"] = target["MASK0"]
    # set active_mask attribute to 'MASK0' (CASA mask definition)
    target_xds["SKY"].attrs["active_mask"] = "MASK0"
    pb_image = make_pbimage()

    pb_threshold = 0.9
    mask = make_mask(
        input_image=pb_image,
        threshold=pb_threshold,
        target_image=target_xds,
        combine_mask=True,
    )
    # CASA mask definition is reverse of xradio mask definition
    expected_mask_data = (pb_image["SKY"].values >= pb_threshold) & (
        ~target["MASK0"].values
    )

    np.testing.assert_array_equal(mask["MASK"].values, expected_mask_data)


def test_make_mask_active_mask_attribute_missing_dv():
    """Test make_mask with combine_mask = True and active_mask attribute pointing to missing data variable"""
    # first make target image and add a rectangular mask in 'MASK' data variable
    target = make_image()
    target["MASK0"] = xr.DataArray(
        np.zeros_like(target["SKY"].values, dtype=bool),
        coords=target["SKY"].coords,
        dims=target["SKY"].dims,
    )
    # set a rectangular region to True
    target["MASK0"].isel(l=slice(60, 80), m=slice(40, 80)).values[:] = True
    # make target data xarray.Dataset
    target_xds = xr.Dataset()
    target_xds["SKY"] = target["SKY"]
    target_xds["MASK0"] = target["MASK0"]
    # set active_mask attribute to a non-existing data variable
    target_xds["SKY"].attrs["active_mask"] = "NON_EXISTING_MASK"

    pb_image = make_pbimage()
    pb_threshold = 0.9
    with pytest.raises(
        KeyError,
        match="Active mask data variable 'NON_EXISTING_MASK' not found in target image",
    ):
        mask = make_mask(
            input_image=pb_image,
            threshold=pb_threshold,
            target_image=target_xds,
            combine_mask=True,
        )


def test_make_mask_target_no_existing_mask():
    """Test make_mask with combine_mask = True and no existing mask in target image"""
    pb_image = make_pbimage()
    target = make_image()
    pb_threshold = 0.1
    mask = make_mask(
        input_image=pb_image,
        threshold=pb_threshold,
        target_image=target,
        combine_mask=True,
    )
    expected_mask_data = (pb_image["SKY"].values >= pb_threshold).astype(bool)
    np.testing.assert_array_equal(mask["MASK"].values, expected_mask_data)


def test_make_mask_no_active_mask_casa_mask():
    """
    Test make_mask with combine_mask = True and no active_mask attribute
    but target image has CASA mask definition in 'MASK0' data variable
    """
    # first make target image and add a rectangular mask in 'MASK0' data variable
    target = make_image()
    target["MASK0"] = xr.DataArray(
        np.zeros_like(target["SKY"].values, dtype=bool),
        coords=target["SKY"].coords,
        dims=target["SKY"].dims,
    )
    # set a rectangular region to True
    target["MASK0"].isel(l=slice(60, 80), m=slice(40, 80)).values[:] = True
    # make target data xarray.Dataset
    target_xds = xr.Dataset()
    target_xds["SKY"] = target["SKY"]
    target_xds["MASK0"] = target["MASK0"]

    pb_image = make_pbimage()

    pb_threshold = 0.9
    mask = make_mask(
        input_image=pb_image,
        threshold=pb_threshold,
        target_image=target_xds,
        combine_mask=True,
    )
    # CASA mask definition is reverse of xradio mask definition
    expected_mask_data = (pb_image["SKY"].values >= pb_threshold) & (
        ~target["MASK0"].values
    )

    np.testing.assert_array_equal(mask["MASK"].values, expected_mask_data)


def test_make_mask_output_image(tmp_path):
    """
    Test make_mask write to disk in zarr format
    """
    pb_image = make_pbimage()
    target = make_image()
    pb_threshold = 0.3
    output_image_name_str = str(tmp_path / "test_mask_output.zarr")
    mask = make_mask(
        input_image=pb_image,
        threshold=pb_threshold,
        target_image=target,
        output_image_name=output_image_name_str,
        output_format="zarr",
        combine_mask=False,
    )
    expected_mask_data = (pb_image["SKY"].values >= pb_threshold).astype(bool)

    import os

    assert os.path.exists(output_image_name_str)
    # Read back the zarr file to verify contents
    loaded_mask = xr.open_zarr(output_image_name_str)
    np.testing.assert_array_equal(loaded_mask["MASK"].values, expected_mask_data)
