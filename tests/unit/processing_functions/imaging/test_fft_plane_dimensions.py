"""Generic spectral planes retain precision and complex correlation signals."""

import numpy as np
import pytest
import xarray as xr
import xradio.image.image_xds  # noqa: F401

from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
    ifft_norm_img_xds,
)


def _image(axis, four):
    labels = ["XX", "XY", "YX", "YY"] if four else ["I"]
    rng = np.random.default_rng(5)
    shape = (1, 2, len(labels), 20, 20)
    dims = ("time", axis, "polarization", "u", "v")
    grid = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return xr.Dataset(
        {"GRID": (dims, grid), "NORM": (dims[:3], np.ones(shape[:3]))},
        coords={"polarization": labels, axis: [0, 1]},
        attrs={
            "type": "image_dataset",
            "data_groups": {
                "input": {"visibility": "GRID", "visibility_normalization": "NORM"}
            },
        },
    )


def _transform(image, dtype, overwrite=True):
    return ifft_norm_img_xds(
        image,
        {"image_size": [16, 16]},
        image_data_group_in_name="input",
        image_data_group_out_name="output",
        image_data_group_out_modified={"sky": "RESULT"},
        image_data_variables_keep=["visibility"],
        fft_backend="scipy",
        complex_dtype=dtype,
        overwrite=overwrite,
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("four", [False, True])
def test_taylor_and_frequency_planes_agree_and_replace_wrong_dtype(dtype, four):
    cube = _transform(_image("frequency", four), dtype)
    taylor = _image("taylor_term", four)
    # Existing output with correct geometry but incompatible precision/type.
    taylor["RESULT"] = (
        ("time", "taylor_term", "polarization", "l", "m"),
        np.zeros((1, 2, 4 if four else 1, 16, 16), dtype=np.int32),
    )
    result = _transform(taylor, dtype)["RESULT"]
    expected = dtype if four else (np.float32 if dtype == np.complex64 else np.float64)
    assert result.dtype == expected
    assert result.dims == ("time", "taylor_term", "polarization", "l", "m")
    np.testing.assert_array_equal(result.values, cube["RESULT"].values)
    if four:
        assert np.max(np.abs(result.values.imag)) > 0


@pytest.mark.parametrize("overwrite", [False, True])
def test_incompatible_output_dimensions_respect_overwrite(overwrite):
    image = _image("taylor_term", False)
    image["RESULT"] = (
        ("time", "old_plane", "polarization", "l", "m"),
        np.zeros((1, 1, 1, 16, 16)),
    )
    if not overwrite:
        with pytest.raises(
            AssertionError, match="Output data variable RESULT already exists"
        ):
            _transform(image, np.complex128, overwrite=False)
    else:
        result = _transform(image, np.complex128)["RESULT"]
        assert result.dims == ("time", "taylor_term", "polarization", "l", "m")
        assert result.shape == (1, 2, 1, 16, 16)
