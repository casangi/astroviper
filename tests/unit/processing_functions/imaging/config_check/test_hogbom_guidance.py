"""Multi-term single-term deconvolvers receive actionable MT-MFS guidance."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    image_continuum_single_field,
)
from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    check_model_update,
)
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    model_update_mtmfs_single_field,
)
from tests.utils.continuum_images import make_runaway_taylor_image

GUIDANCE = "Use deconvolver='mtmfs' for 2 to 4 Taylor terms"


@pytest.mark.parametrize("deconvolver", ["hogbom", "asp"])
@pytest.mark.parametrize("nterms", [2, 3, 4])
def test_single_term_configuration_errors_explain_the_supported_alternative(
    deconvolver, nterms
):
    with pytest.raises(ValueError, match=GUIDANCE):
        check_model_update(
            {"deconvolver": deconvolver, "image_params": {"n_taylor_terms": nterms}},
            mode=CONTINUUM_SINGLE_FIELD,
        )


@pytest.mark.parametrize("restore", [False, True])
def test_public_hogbom_rejection_directs_to_mtmfs_before_opening_or_writing(
    monkeypatch, tmp_path, restore
):
    open_input = Mock(side_effect=AssertionError("must not open input"))
    monkeypatch.setattr("xradio.measurement_set.open_processing_set", open_input)
    image_params = {
        "n_taylor_terms": 2,
        "reference_frequency": 1.1e9,
        "image_size": [8, 8],
        "cell_size": [-1e-5, 1e-5],
        "phase_direction": [0.0, 0.0],
        "frequency_coords": [1e9, 1.2e9],
        "time_coords": [0.0],
        "polarization_coords": ["I", "Q"],
    }
    before = deepcopy(image_params)
    output = tmp_path / "image.zarr"
    with pytest.raises(ValueError, match=GUIDANCE):
        image_continuum_single_field(
            "missing-input",
            str(output),
            image_params,
            {"weighting": "natural"},
            {},
            deconvolver="hogbom",
            restore=restore,
        )
    open_input.assert_not_called()
    assert not output.exists()
    assert image_params == before


@pytest.mark.parametrize("checked", [False, True])
def test_direct_hogbom_product_errors_also_explain_mtmfs(checked):
    image = make_runaway_taylor_image()
    before = image.copy(deep=True)
    with pytest.raises(ValueError, match=GUIDANCE):
        model_update_mtmfs_single_field(image, "hogbom", {}, input_checked=checked)
    xr.testing.assert_identical(image, before)
