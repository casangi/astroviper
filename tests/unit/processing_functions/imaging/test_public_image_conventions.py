"""The PDF's public-image identities, independent of the internal solver basis."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.finalize_image import (
    finalize_image_products,
    multiply_taylor_polynomials,
    public_continuum_layout,
    restore_finalized_image,
)


def image_case(mode):
    n = 16
    axis = "frequency" if mode is None else "taylor_term"
    nt = 1 if mode is None else 2
    dims = ("time", axis, "polarization", "l", "m")
    coords = {
        "time": [60000.0],
        axis: [1.0e9] if mode is None else np.arange(nt),
        "polarization": ["I", "V"],
        "l": np.arange(n) * -1.0e-5,
        "m": np.arange(n) * 1.0e-5,
    }
    shape = (1, nt, 2, n, n)
    intrinsic = xr.DataArray(np.zeros(shape), dims=dims, coords=coords)
    intrinsic.values[:, 0, :, 8, 8] = 2
    rho = xr.full_like(intrinsic, 0.01)
    beam = xr.ones_like(intrinsic) * 0.7
    if nt == 2:
        intrinsic.values[:, 1] = intrinsic.values[:, 0] * 0.5
        rho.values[:, 1] *= 0.2
        beam.values[:, 1] = -0.14
    if mode == "mfs":
        model = multiply_taylor_polynomials(intrinsic, beam)
        apparent_rho = multiply_taylor_polynomials(rho, beam)
    elif mode == "mvc":
        model, apparent_rho = intrinsic * 0.8, rho * 0.8
    else:
        model, apparent_rho = intrinsic * beam, rho * beam
    residual = apparent_rho.copy(deep=True)
    image = xr.Dataset({"SKY_MODEL": model, "SKY_RESIDUAL": residual})
    if mode is not None:
        image = image.assign_coords(frequency=[1.0e9])
        psf = np.zeros((1, 3, 2, n, n))
        psf[:, 0, :, 8, 8] = 1
        psf[:, 2, :, 8, 8] = 0.25
        image["POINT_SPREAD_FUNCTION"] = (
            ("time", "psf_taylor_order", "polarization", "l", "m"),
            psf,
        )
        image["SKY_RESIDUAL"].values[:, 1] *= 0.25
        image["PRIMARY_BEAM_TAYLOR"] = beam
    else:
        image["POINT_SPREAD_FUNCTION"] = xr.zeros_like(intrinsic)
        image["POINT_SPREAD_FUNCTION"].values[..., 8, 8] = 1
    primary = (
        beam
        if mode is None
        else beam.isel(taylor_term=0, drop=True).expand_dims(frequency=[1.0e9], axis=1)
    )
    if mode == "mvc":
        primary = xr.full_like(primary, 0.8)
    image["PRIMARY_BEAM"] = primary
    image["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] = (
        ("time", "frequency", "polarization", "beam_params"),
        np.tile([3.0e-5, 2.0e-5, 0.0], (1, 1, 2, 1)),
    )
    image.attrs = {
        "reference_frequency": 1.0e9,
        "continuum_imaging": {"continuum_mode": mode},
        "data_groups": {
            "model": {"sky": "SKY_MODEL"},
            "residual": {
                "sky": "SKY_RESIDUAL",
                "primary_beam": "PRIMARY_BEAM",
                "point_spread_function": "POINT_SPREAD_FUNCTION",
                "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
            },
        },
    }
    return image, intrinsic, rho


@pytest.mark.parametrize("mode", [None, "mfs", "mvc"])
def test_finalization_exposes_flat_sky_model_and_principal_residual(mode):
    image, intrinsic, rho = image_case(mode)
    original = image.copy(deep=True)
    out = finalize_image_products(
        image, continuum_mode=mode, deconvolver="mtmfs" if mode else "hogbom"
    )
    np.testing.assert_allclose(out.SKY_MODEL, intrinsic, atol=1.0e-15)
    np.testing.assert_allclose(out.SKY_RESIDUAL, rho, atol=1.0e-15)
    xr.testing.assert_identical(image, original)
    assert finalize_image_products(out, continuum_mode=mode) is out
    assert not any(name in out for name in ("ALPHA", "ALPHA_ERROR", "BETA"))


@pytest.mark.parametrize("mode", [None, "mfs", "mvc"])
@pytest.mark.parametrize("target", [None, [6.0e-5, 5.0e-5, 0.0]])
def test_restored_flat_sky_identity_uses_the_reported_residual(mode, target):
    from astroviper.processing_functions.imaging.restore import (
        _clean_beam_kernel_ft,
        _convolve_with_clean_beam,
    )

    image, _, _ = image_case(mode)
    out = finalize_image_products(
        image, continuum_mode=mode, deconvolver="mtmfs" if mode else "hogbom"
    )
    out, _ = restore_finalized_image(
        out, restoration_output="both", restoring_beam=target
    )
    beam = target or [3.0e-5, 2.0e-5, 0.0]
    kernel = _clean_beam_kernel_ft(beam, 16, 16, 1.0e-5, np.float64, 1)
    for index in np.ndindex(out.SKY_MODEL.shape[:3]):
        expected = (
            _convolve_with_clean_beam(out.SKY_MODEL.values[index], kernel, 1)
            + out.SKY_RESIDUAL.values[index]
        )
        np.testing.assert_allclose(
            out.SKY_PRIMARY_BEAM_CORRECTED_RESTORED.values[index],
            expected,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
    assert "SKY_RESTORED" in out
    if mode:
        assert out.PRIMARY_BEAM.dims == out.SKY_MODEL.dims
        public = public_continuum_layout(out)
        assert "frequency_polynomial" in public.SKY_MODEL.dims
        assert "frequency_polynomial_moment" in public.POINT_SPREAD_FUNCTION.dims


def test_mvc_flat_noise_restoration_is_apparent_not_common_beam():
    image, _, _ = image_case("mvc")
    finalized = finalize_image_products(
        image, continuum_mode="mvc", deconvolver="mtmfs"
    )
    out, _ = restore_finalized_image(finalized, restoration_output="both")
    np.testing.assert_allclose(
        out.SKY_RESTORED.values[:, 0],
        out.SKY_PRIMARY_BEAM_CORRECTED_RESTORED.values[:, 0] * 0.7,
    )
    np.testing.assert_allclose(
        out.SKY_RESTORED.values[:, 1],
        out.SKY_PRIMARY_BEAM_CORRECTED_RESTORED.values[:, 1] * 0.7
        - out.SKY_PRIMARY_BEAM_CORRECTED_RESTORED.values[:, 0] * 0.14,
        atol=1.0e-14,
    )


@pytest.mark.parametrize("mode", ["mfs", "mvc"])
def test_public_polynomial_schema_zarr_roundtrip(tmp_path, mode):
    from xradio.image import check_image, load_image, make_empty_sky_image, write_image

    image, _, _ = image_case(mode)
    template = make_empty_sky_image(
        [0.0, 0.0],
        [16, 16],
        [-1e-5, 1e-5],
        [1e9],
        ["I", "V"],
        [60000.0],
        do_sky_coords=False,
    )
    original_coords = {
        name: dict(value.attrs) for name, value in template.coords.items()
    }
    template = template.assign_coords(l=image.l, m=image.m)
    for name, value in image.data_vars.items():
        template[name] = value
    template.attrs.update(image.attrs)
    for name in ("SKY_MODEL", "SKY_RESIDUAL"):
        template[name].attrs = {
            "type": "sky",
            "sub_type": "Intensity",
            "units": "Jy/beam",
        }
    template.PRIMARY_BEAM.attrs = {"type": "primary_beam", "units": ""}
    template.PRIMARY_BEAM_TAYLOR.attrs = template.PRIMARY_BEAM.attrs
    template.POINT_SPREAD_FUNCTION.attrs = {
        "type": "point_spread_function",
        "units": "",
    }
    template.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.attrs = {
        "type": "beam_fit_params_point_spread_function",
        "units": "rad",
    }
    for name, attrs in original_coords.items():
        if name in template.coords:
            template[name].attrs = attrs
    template.attrs["continuum_bandwidth"] = 4e8
    result = public_continuum_layout(
        restore_finalized_image(
            finalize_image_products(template, continuum_mode=mode, deconvolver="mtmfs"),
            restoration_output="both",
        )[0]
    )
    assert not check_image(result)
    from astroviper.processing_functions.image_analysis import spectral_products

    analyzed = spectral_products(
        result, image_data_group_in_name="primary_beam_corrected_restored"
    )
    assert not check_image(analyzed), str(check_image(analyzed))
    path = str(tmp_path / "continuum.img.zarr")
    write_image(result, path, out_format="zarr")
    loaded = load_image(path)
    assert not check_image(loaded)
    assert loaded.frequency_polynomial.size == 2
    assert loaded.frequency_polynomial_moment.size == 3
    assert loaded.frequency_polynomial.attrs["channel_width"]["data"] == 4e8
    assert loaded.PRIMARY_BEAM.dims == loaded.SKY_MODEL.dims
    assert "PRIMARY_BEAM_TAYLOR" not in loaded
    if mode == "mvc":
        assert loaded.COMMON_PRIMARY_BEAM.sizes["frequency"] == 1
    assert (
        loaded.attrs["data_groups"]["primary_beam_corrected_restored"]["sky"]
        == "SKY_PRIMARY_BEAM_CORRECTED_RESTORED"
    )


@pytest.mark.parametrize("mode", [None, "mfs", "mvc"])
def test_finalized_images_are_not_corrected_by_the_primary_beam_again(mode):
    from astroviper.processing_functions.imaging import correct_sky_by_primary_beam

    image, _, _ = image_case(mode)
    finalized = finalize_image_products(
        image, continuum_mode=mode, deconvolver="mtmfs" if mode else "hogbom"
    )
    original = finalized.copy(deep=True)
    with pytest.raises(ValueError, match="already flat sky"):
        correct_sky_by_primary_beam(finalized)
    xr.testing.assert_identical(finalized, original)


def test_public_cube_restoration_honors_beam_selection_and_overwrite():
    from astroviper.processing_functions.imaging import restore_image

    image, _, _ = image_case(None)
    alternate = [4e-5, 3e-5, 0.1]
    image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0, 0, 1] = alternate
    finalized = finalize_image_products(image)
    restored, _ = restore_image(
        finalized,
        restoration_output="primary_beam_corrected",
        beam_polarization_index=1,
    )
    np.testing.assert_array_equal(
        restored.BEAM_FIT_PARAMS_SKY_RESTORED.values[0, 0, 0], alternate
    )
    with pytest.raises(FileExistsError, match="already exist"):
        restore_image(restored, overwrite=False)


@pytest.mark.parametrize("mode", [None, "mfs", "mvc"])
@pytest.mark.parametrize("output", ["apparent", "both"])
@pytest.mark.parametrize("target", [None, [6e-5, 5e-5, 0.0]])
def test_flat_noise_uses_unmasked_inputs_and_releases_them(mode, output, target):
    """The PB cutoff must not discard finite apparent sky before convolution."""
    import weakref

    from astroviper.processing_functions.imaging.restore import _restore_image_arrays

    image, intrinsic, rho = image_case(mode)
    # Put a component below the cutoff, close enough to illuminate supported
    # pixels through convolution. The beam varies across the cutoff boundary.
    intrinsic.values[..., 8, 8] = 0
    intrinsic.values[:, 0, :, 2, 8] = 2
    if mode:
        intrinsic.values[:, 1] = intrinsic.values[:, 0] * 0.5
    beam = xr.ones_like(intrinsic)
    beam.values[:, 0] = np.linspace(0.1, 0.7, 16)[:, None]
    if mode:
        beam.values[:, 1] = -0.2 * beam.values[:, 0]
        expected_model = multiply_taylor_polynomials(intrinsic, beam)
        expected_residual = multiply_taylor_polynomials(rho, beam)
        image["PRIMARY_BEAM_TAYLOR"] = beam
        primary = beam.isel(taylor_term=0, drop=True).expand_dims(
            frequency=image.frequency, axis=1
        )
    else:
        primary = beam
        expected_model, expected_residual = intrinsic * beam, rho * beam
    image["PRIMARY_BEAM"] = primary
    if mode == "mvc":
        common = primary.isel(frequency=0, drop=True)
        image["SKY_MODEL"], image["SKY_RESIDUAL"] = intrinsic * common, rho * common
    else:
        image["SKY_MODEL"], image["SKY_RESIDUAL"] = expected_model, expected_residual
    if mode:
        image.SKY_RESIDUAL.values[:, 1] *= 0.25
        # Assignment above may share storage with this expected array.
        expected_residual = multiply_taylor_polynomials(rho, beam)
    original = image.copy(deep=True)
    apparent_inputs = {}
    finalized = finalize_image_products(
        image,
        continuum_mode=mode,
        deconvolver="mtmfs" if mode else "hogbom",
        apparent_inputs=apparent_inputs,
    )
    np.testing.assert_allclose(apparent_inputs["model"], expected_model)
    np.testing.assert_allclose(apparent_inputs["residual"], expected_residual)
    references = [weakref.ref(value) for value in apparent_inputs.values()]
    assert np.isnan(finalized.SKY_RESIDUAL.values[..., :3, :]).all()
    assert not finalized.SKY_MODEL.values[..., :3, :].any()
    restored, _ = restore_finalized_image(
        finalized,
        restoration_output=output,
        restoring_beam=target,
        apparent_inputs=apparent_inputs,
    )
    expected, _ = _restore_image_arrays(
        expected_residual.values,
        expected_model.values,
        np.tile([3e-5, 2e-5, 0.0], (1, expected_model.shape[1], 2, 1)),
        1e-5,
        output="apparent",
        restoring_beam=target,
        processing_function_threads=1,
    )
    np.testing.assert_allclose(restored.SKY_RESTORED, expected["apparent"], atol=1e-14)
    assert np.isfinite(restored.SKY_RESTORED).all()
    assert not apparent_inputs
    assert all(reference() is None for reference in references)
    assert not any("APPARENT" in name for name in restored.data_vars)
    xr.testing.assert_identical(image, original)


@pytest.mark.parametrize("mode", ["mfs", "mvc"])
@pytest.mark.parametrize("output", ["apparent", "both", "primary_beam_corrected"])
def test_continuum_restoration_collects_inputs_only_for_flat_noise(
    monkeypatch, mode, output
):
    import importlib

    module = importlib.import_module(
        "astroviper.processing_functions.imaging.image_continuum_single_field"
    )
    image, _, _ = image_case(mode)
    finalizer = importlib.import_module(
        "astroviper.processing_functions.imaging.finalize_image"
    )
    finalize = finalizer.finalize_image_products
    captured = []

    def record(*args, **kwargs):
        captured.append(kwargs["apparent_inputs"])
        return finalize(*args, **kwargs)

    monkeypatch.setattr(finalizer, "finalize_image_products", record)
    module.restore_continuum_image(
        image,
        continuum_mode=mode,
        deconvolver="mtmfs",
        restoration_output=output,
        input_checked=True,
    )
    if output == "primary_beam_corrected":
        assert captured == [None]
    else:
        assert captured == [{}]  # Restoration consumed every retained input.


@pytest.mark.parametrize("mode", [None, "mfs", "mvc"])
@pytest.mark.parametrize("output", ["apparent", "primary_beam_corrected", "both"])
def test_invalid_target_preserves_public_products_and_temporary_inputs(
    mode, output, monkeypatch
):
    import importlib

    backend = importlib.import_module("astroviper.processing_functions.imaging.restore")

    def unexpected_restore(*args, **kwargs):
        pytest.fail("Target containment must be checked before convolution")

    monkeypatch.setattr(backend, "_restore_image_arrays", unexpected_restore)
    image, _, _ = image_case(mode)
    apparent_inputs = {} if output in ("apparent", "both") else None
    finalized = finalize_image_products(
        image,
        continuum_mode=mode,
        deconvolver="mtmfs" if mode else "hogbom",
        apparent_inputs=apparent_inputs,
    )
    original = finalized.copy(deep=True)
    saved = None if apparent_inputs is None else dict(apparent_inputs)
    with pytest.raises(ValueError, match="cannot be achieved"):
        restore_finalized_image(
            finalized,
            restoration_output=output,
            restoring_beam=[5e-5, 1.5e-5, 0.0],
            apparent_inputs=apparent_inputs,
        )
    xr.testing.assert_identical(finalized, original)
    if apparent_inputs is not None:
        assert all(apparent_inputs[key] is value for key, value in saved.items())


@pytest.mark.parametrize("missing", [0.0, np.nan])
@pytest.mark.parametrize("has_valid_fit", [False, True])
def test_deferred_fallback_preserves_missing_beams_without_changing_pixels(
    missing, has_valid_fit
):
    from astroviper.processing_functions.imaging.restore import NO_FIT_BEAM

    image, _, _ = image_case(None)
    image = xr.concat([image, image.assign_coords(frequency=[1.1e9])], dim="frequency")
    image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[...] = missing
    if has_valid_fit:
        image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[:, 0] = [3e-5, 2e-5, 0.0]
    finalized = finalize_image_products(image)
    original = finalized.copy(deep=True)
    deferred, _ = restore_finalized_image(
        finalized, restoration_output="both", defer_beam_fallback=True
    )
    standalone, _ = restore_finalized_image(finalized, restoration_output="both")
    np.testing.assert_equal(
        deferred.BEAM_FIT_PARAMS_SKY_RESTORED.values,
        image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
    )
    np.testing.assert_array_equal(
        standalone.BEAM_FIT_PARAMS_SKY_RESTORED,
        np.broadcast_to(
            [3e-5, 2e-5, 0.0] if has_valid_fit else NO_FIT_BEAM,
            standalone.BEAM_FIT_PARAMS_SKY_RESTORED.shape,
        ),
    )
    for name in ("SKY_RESTORED", "SKY_PRIMARY_BEAM_CORRECTED_RESTORED"):
        xr.testing.assert_identical(deferred[name], standalone[name])
    xr.testing.assert_identical(finalized, original)
