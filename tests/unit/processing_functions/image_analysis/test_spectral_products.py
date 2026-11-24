"""Spectral analysis is explicit and uses the public flat-sky coefficient basis."""

import numpy as np
import pytest

from astroviper.processing_functions.image_analysis import spectral_products
from astroviper.processing_functions.imaging.finalize_image import (
    finalize_image_products,
    public_continuum_layout,
    restore_finalized_image,
)
from tests.unit.processing_functions.imaging.test_public_image_conventions import (
    image_case,
)


def restored_image():
    image, _, _ = image_case("mfs")
    image = finalize_image_products(image, continuum_mode="mfs", deconvolver="mtmfs")
    return public_continuum_layout(restore_finalized_image(image)[0])


def test_spectral_products_are_explicit_and_use_reported_flat_sky_residual():
    image = restored_image()
    assert "ALPHA" not in image
    result = spectral_products(image)
    zero = image.SKY_PRIMARY_BEAM_CORRECTED_RESTORED.isel(frequency_polynomial=0)
    one = image.SKY_PRIMARY_BEAM_CORRECTED_RESTORED.isel(frequency_polynomial=1)
    rzero = image.SKY_RESIDUAL.isel(frequency_polynomial=0)
    rone = image.SKY_RESIDUAL.isel(frequency_polynomial=1)
    expected = one / zero
    np.testing.assert_allclose(result.ALPHA.isel(frequency=0), expected)
    np.testing.assert_allclose(
        result.ALPHA_ERROR.isel(frequency=0),
        np.sqrt(rone**2 + (expected * rzero) ** 2) / abs(zero),
    )
    assert result.ALPHA.dims == ("time", "frequency", "polarization", "l", "m")
    assert result.frequency.values.tolist() == [1e9]
    assert "BETA" not in result
    assert "ALPHA" not in image


def test_only_requested_diagnostic_is_added():
    result = spectral_products(restored_image(), products=("ALPHA_ERROR",))
    assert "ALPHA_ERROR" in result
    assert "ALPHA" not in result


def test_spectral_analysis_refuses_apparent_sky_and_insufficient_terms():
    image = restored_image()
    with pytest.raises(ValueError, match="3 coefficients"):
        spectral_products(image, products=("BETA",))
    image.SKY_PRIMARY_BEAM_CORRECTED_RESTORED.attrs["sky_convention"] = "flat_noise"
    with pytest.raises(ValueError, match="flat-sky"):
        spectral_products(image)
