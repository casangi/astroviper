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
    print("pb_image_shape=", pb_image.shape)
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
            pb_threshold=pb_threshold,
            target_image=None,
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
            pb_threshold=pb_threshold,
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
        pb_threshold=pb_threshold,
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
    print("target sum true= ", target["MASK"].values.sum())
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
        pb_threshold=pb_threshold,
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
