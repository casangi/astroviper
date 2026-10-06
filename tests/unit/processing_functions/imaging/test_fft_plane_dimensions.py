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


@pytest.mark.parametrize("plane_dim", ["taylor_term", "psf_taylor_order"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("four", [False, True])
def test_taylor_and_frequency_planes_agree_and_replace_wrong_dtype(
    dtype, four, plane_dim
):
    cube = _transform(_image("frequency", four), dtype)
    taylor = _image(plane_dim, four)
    # Existing output with correct geometry but incompatible precision/type.
    taylor["RESULT"] = (
        ("time", plane_dim, "polarization", "l", "m"),
        np.zeros((1, 2, 4 if four else 1, 16, 16), dtype=np.int32),
    )
    result = _transform(taylor, dtype)["RESULT"]
    expected = dtype if four else (np.float32 if dtype == np.complex64 else np.float64)
    assert result.dtype == expected
    assert result.dims == ("time", plane_dim, "polarization", "l", "m")
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


def _sky_image(axis):
    rng = np.random.default_rng(12)
    shape = (2, 2, 4, 8, 8)
    sky = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return xr.Dataset(
        {"MODEL": (("time", axis, "polarization", "l", "m"), sky)},
        coords={
            "time": [0, 1],
            axis: [10, 20],
            "polarization": ["XX", "XY", "YX", "YY"],
        },
        attrs={"type": "image_dataset", "data_groups": {"model": {"sky": "MODEL"}}},
    )


def _forward(image, axis, dtype, **kwargs):
    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        fft_norm_continuum_img_xds,
        fft_norm_img_xds,
    )

    transform = fft_norm_img_xds if axis == "frequency" else fft_norm_continuum_img_xds
    return transform(
        image,
        {"fft_padding": 1.25},
        image_data_group_out_name="uv_model",
        image_data_group_out_modified={"visibility": "OUTPUT"},
        complex_dtype=dtype,
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("backend", ["scipy", "pyfftw"])
def test_forward_frequency_and_taylor_match_independent_fft(dtype, backend):
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_correcting_image_1D,
    )
    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )

    outputs = []
    for axis in ("frequency", "taylor_term"):
        image = _sky_image(axis)
        original = image.MODEL.values.copy()
        # Both wrappers must honor named dimensions, not positional assumptions.
        image["MODEL"] = image.MODEL.transpose("polarization", "m", axis, "time", "l")
        result = _forward(
            image, axis, dtype, fft_backend=backend, image_data_variables_keep=["sky"]
        )
        np.testing.assert_array_equal(
            result.MODEL.transpose("time", axis, "polarization", "l", "m"), original
        )
        uv = result.OUTPUT
        assert uv.dims == ("time", axis, "polarization", "u", "v")
        assert uv.dtype == dtype
        np.testing.assert_array_equal(uv[axis], [10, 20])
        n_uv = padded_grid_size([8, 8], 1.25)
        padded = np.zeros(original.shape[:3] + tuple(n_uv), dtype=dtype)
        starts = [int(n_uv[0]) // 2 - 4, int(n_uv[1]) // 2 - 4]
        padded[..., starts[0] : starts[0] + 8, starts[1] : starts[1] + 8] = original
        kl, km = create_prolate_spheroidal_correcting_image_1D(n_lm_padded=n_uv)
        padded /= kl[:, None]
        padded /= km[None, :]
        expected = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(padded, axes=(-2, -1))), axes=(-2, -1)
        )
        tolerance = 2e-6 if dtype == np.complex64 else 1e-13
        np.testing.assert_allclose(
            uv.values,
            expected,
            rtol=tolerance,
            atol=tolerance * np.max(np.abs(expected)),
        )
        outputs.append(uv.values)
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("axis", ["frequency", "taylor_term"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("existing", ["compatible", "dtype", "shape", "dimensions"])
def test_forward_output_buffer_reuse_and_replacement(axis, dtype, existing):
    image = _sky_image(axis)
    reference = _forward(
        image.copy(deep=True),
        axis,
        dtype,
        fft_backend="scipy",
        image_data_variables_keep=["sky"],
    )
    dims = list(reference.OUTPUT.dims)
    shape = list(reference.OUTPUT.shape)
    old_dtype = dtype
    if existing == "dtype":
        old_dtype = np.complex128 if dtype == np.complex64 else np.complex64
    elif existing == "shape":
        shape[-1] += 2
    elif existing == "dimensions":
        dims[1] = "old_plane"
    buffer = np.full(shape, np.nan, dtype=old_dtype)
    image["OUTPUT"] = (dims, buffer)
    result = _forward(
        image, axis, dtype, fft_backend="scipy", image_data_variables_keep=["sky"]
    )
    assert result is image
    assert np.shares_memory(result.OUTPUT.values, buffer) == (existing == "compatible")
    np.testing.assert_array_equal(result.OUTPUT.values, reference.OUTPUT.values)
    assert result.OUTPUT.dtype == dtype


@pytest.mark.parametrize("axis", ["frequency", "taylor_term"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_forward_overwrite_false_preserves_existing_output(axis, dtype):
    image = _sky_image(axis)
    image["OUTPUT"] = (("old_plane",), np.array([42], dtype=dtype))
    original = image.copy(deep=True)
    with pytest.raises(
        AssertionError, match="Output data variable OUTPUT already exists"
    ):
        _forward(image, axis, dtype, fft_backend="scipy", overwrite=False)
    xr.testing.assert_identical(image, original)


@pytest.mark.parametrize("axis", ["frequency", "taylor_term"])
def test_forward_uses_output_as_workspace_and_drops_unkept_sky(axis, monkeypatch):
    import importlib

    module = importlib.import_module(
        "astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder"
    )
    original_fft = module.fft_lm_to_uv
    image = _sky_image(axis)
    original_sky = image.MODEL.values
    saved_sky = original_sky.copy()
    calls = []

    def tracked_fft(plane, **kwargs):
        assert kwargs["overwrite_input"] is True
        assert np.shares_memory(plane, image.OUTPUT.values)
        assert not np.shares_memory(plane, original_sky)
        calls.append(plane.shape)
        return original_fft(plane, **kwargs)

    monkeypatch.setattr(module, "fft_lm_to_uv", tracked_fft)
    result = _forward(image, axis, np.complex64, fft_backend="scipy")
    assert len(calls) == 16
    assert "MODEL" not in result
    np.testing.assert_array_equal(original_sky, saved_sky)
    assert result.attrs["data_groups"]["uv_model"]["visibility"] == "OUTPUT"
