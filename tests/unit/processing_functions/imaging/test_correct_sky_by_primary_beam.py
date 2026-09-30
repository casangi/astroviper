"""Unit tests for the primary beam correction of the restored sky, in both
conventions (the model corrected before the restore step, the default, and the
restored image divided as CASA pbcor does)."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.correct_sky_by_primary_beam import (
    correct_sky_by_primary_beam,
)

ARCSEC = np.pi / (180 * 3600)


def _make_restored_image(sky_value=1.0, dtype=np.float64):
    """Minimal image dataset with a restored data group (sky + primary beam)."""
    from xradio.image import make_empty_sky_image

    img_xds = make_empty_sky_image(
        phase_center=np.array([0.0, 0.5]),
        image_size=[32, 32],
        cell_size=np.array([-8.0, 8.0]) * ARCSEC,
        frequency_coords=np.array([1.0e11]),
        pol_coords=["I"],
        time_coords=[0],
        do_sky_coords=False,
    )
    img_xds.attrs["type"] = "image_dataset"
    shape = (1, 1, 1, 32, 32)
    # A radially declining fake power beam crossing the 0.2 cutoff.
    radius = np.hypot(*np.meshgrid(np.arange(32) - 16, np.arange(32) - 16))
    primary_beam = np.clip(1.0 - radius / 18.0, 0.0, None)[None, None, None]
    img_xds["SKY_RESTORED"] = xr.DataArray(
        np.full(shape, sky_value, dtype=dtype),
        dims=("time", "frequency", "polarization", "l", "m"),
    )
    img_xds["PRIMARY_BEAM"] = xr.DataArray(
        primary_beam.astype(dtype), dims=("time", "frequency", "polarization", "l", "m")
    )
    img_xds = img_xds.xr_img.add_data_group(
        new_data_group_name="restored",
        new_data_group={
            "description": "test",
            "date": "2026",
            "sky": "SKY_RESTORED",
            "primary_beam": "PRIMARY_BEAM",
        },
    )
    return img_xds


def _add_model_and_residual(img_xds, flux=2.0, source=(20, 8)):
    """Model with one point source, zero residual and a clean beam fit, so that
    the default convention can be applied to the image."""
    shape = img_xds["SKY_RESTORED"].shape
    dims = img_xds["SKY_RESTORED"].dims
    dtype = img_xds["SKY_RESTORED"].dtype
    model = np.zeros(shape, dtype=dtype)
    model[0, 0, 0, source[0], source[1]] = flux
    img_xds["SKY_MODEL"] = xr.DataArray(model, dims=dims)
    img_xds["SKY_RESIDUAL"] = xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims)
    cell = 8.0 * ARCSEC
    img_xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] = xr.DataArray(
        np.array([[[[6.0 * cell, 3.0 * cell, 0.4]]]]),
        dims=("time", "frequency", "polarization", "beam_params"),
    )
    img_xds.attrs["data_groups"]["model"] = {"sky": "SKY_MODEL"}
    img_xds.attrs["data_groups"]["residual"] = {
        "sky": "SKY_RESIDUAL",
        "primary_beam": "PRIMARY_BEAM",
        "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
    }
    return img_xds


def test_correct_then_restore_is_exact_for_a_point_source():
    from astroviper.processing_functions.imaging.restore import restore_image

    img_xds = _add_model_and_residual(_make_restored_image())
    primary_beam = img_xds["PRIMARY_BEAM"].values
    img_xds, return_df = correct_sky_by_primary_beam(img_xds)  # the default convention
    corrected = img_xds["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"].values
    inside = primary_beam >= 0.2
    # the true sky (flux / P at the source) convolved with the clean beam
    reference = _add_model_and_residual(
        _make_restored_image(), flux=2.0 / primary_beam[0, 0, 0, 20, 8]
    )
    reference, _ = restore_image(reference)
    expected = reference["SKY_RESTORED"].values
    np.testing.assert_allclose(
        corrected[inside], expected[inside], rtol=1e-9, atol=1e-12
    )
    assert np.isnan(corrected[~inside]).all()
    assert "T_correct_sky_by_primary_beam" in return_df.columns


def test_unknown_order_is_refused():
    img_xds = _make_restored_image()
    with pytest.raises(ValueError, match="primary_beam_correction_order"):
        correct_sky_by_primary_beam(img_xds, primary_beam_correction_order="pbcor")


def test_correct_sky_by_primary_beam_divides_and_blanks():
    img_xds = _make_restored_image()
    img_xds, return_df = correct_sky_by_primary_beam(
        img_xds, primary_beam_correction_order="restore_then_correct"
    )

    assert "SKY_RESTORED_PRIMARY_BEAM_CORRECTED" in img_xds.data_vars
    corrected = img_xds["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"].values
    primary_beam = img_xds["PRIMARY_BEAM"].values
    inside = primary_beam >= 0.2
    # Inside the cutoff: exact division; outside: blanked with NaN.
    np.testing.assert_array_equal(corrected[inside], (1.0 / primary_beam)[inside])
    assert np.isnan(corrected[~inside]).all()
    assert inside.any() and (~inside).any()

    # Registered on the data group under the corrected-sky role.
    assert (
        img_xds.attrs["data_groups"]["restored"]["sky_primary_beam_corrected"]
        == "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"
    )
    assert "T_correct_sky_by_primary_beam" in return_df.columns


def test_correct_sky_by_primary_beam_limit_and_dtype():
    img_xds = _make_restored_image(sky_value=2.0, dtype=np.float32)
    img_xds, _ = correct_sky_by_primary_beam(
        img_xds,
        primary_beam_limit=0.5,
        primary_beam_correction_order="restore_then_correct",
    )
    corrected = img_xds["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"]
    assert corrected.dtype == np.float32
    primary_beam = img_xds["PRIMARY_BEAM"].values
    assert np.isnan(corrected.values[primary_beam < 0.5]).all()
    # Peak pixel: beam = 1 so the corrected value equals the sky value.
    assert corrected.values[0, 0, 0, 16, 16] == pytest.approx(2.0)


def test_correct_sky_by_primary_beam_requires_primary_beam():
    img_xds = _make_restored_image()
    del img_xds.attrs["data_groups"]["restored"]["primary_beam"]
    with pytest.raises(AssertionError, match="primary_beam"):
        correct_sky_by_primary_beam(
            img_xds, primary_beam_correction_order="restore_then_correct"
        )
