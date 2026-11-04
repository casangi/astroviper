"""One continuum selector controls the products at every entry boundary."""

import inspect

import numpy as np
import pytest

from astroviper.distributed_applications.imaging import image_continuum_single_field
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    restore_continuum_image,
)
from tests.unit.processing_functions.imaging.config_check.test_full_validation import (
    configuration,
)
from tests.utils.continuum_images import make_continuum_image


@pytest.mark.parametrize("mode", ["mfs", "mvc"])
@pytest.mark.parametrize("output", ["apparent", "primary_beam_corrected", "both"])
def test_selector_alone_controls_restored_products(mode, output):
    image = make_continuum_image(continuum_mode=mode)
    residual = image.SKY_RESIDUAL.copy(deep=True)
    restored, _ = restore_continuum_image(image, restoration_output=output)
    assert ("SKY_RESTORED_FLAT_NOISE" in restored) is (output in {"apparent", "both"})
    assert ("SKY_RESTORED_FLAT_SKY" in restored) is (
        output in {"primary_beam_corrected", "both"}
    )
    np.testing.assert_array_equal(restored.SKY_RESIDUAL, residual)


def test_continuum_interfaces_expose_only_the_output_selector():
    for function in (image_continuum_single_field, restore_continuum_image):
        assert "primary_beam_correction" not in inspect.signature(function).parameters
        assert "restoration_output" in inspect.signature(function).parameters


@pytest.mark.parametrize(
    "output,key",
    [
        ("apparent", "sky_restored_flat_sky"),
        ("primary_beam_corrected", "sky_restored_flat_noise"),
    ],
)
def test_conflicting_keep_requests_fail_before_opening_input(
    monkeypatch, tmp_path, output, key
):
    import xradio.measurement_set

    def unexpected_open(*args, **kwargs):
        pytest.fail("Input opened before checking restoration selection")

    monkeypatch.setattr(xradio.measurement_set, "open_processing_set", unexpected_open)
    config = configuration()
    image_store = tmp_path / "conflicting.img.zarr"
    with pytest.raises(ValueError, match="unavailable with restoration_output"):
        image_continuum_single_field(
            ps_store="unopened.ps.zarr",
            image_store=str(image_store),
            restoration_output=output,
            image_data_variables_keep=[key],
            restore=True,
            **config,
        )
    assert not image_store.exists()


def test_cube_keeps_its_separate_correction_api():
    from astroviper.processing_functions.imaging.config_check import (
        CUBE_SINGLE_FIELD,
        check_driver_workflow,
    )

    check_driver_workflow(
        {"restore": True, "primary_beam_correction": True}, mode=CUBE_SINGLE_FIELD
    )
    with pytest.raises(ValueError, match="continuum-only"):
        check_driver_workflow({"restoration_output": "both"}, mode=CUBE_SINGLE_FIELD)


def test_removed_processing_keyword_has_no_alias():
    with pytest.raises(TypeError, match="primary_beam_correction"):
        restore_continuum_image(None, primary_beam_correction=True)
