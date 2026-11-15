"""H5: Högbom cannot solve higher continuum coefficients, with or without restore."""

from copy import deepcopy

import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    check_model_update,
)


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("n_taylor_terms", [2, 3, 4])
def test_hogbom_rejects_unsolved_taylor_terms_without_changing_configuration(
    continuum_mode, restore, n_taylor_terms
):
    config = {
        "continuum_mode": continuum_mode,
        "restore": restore,
        "deconvolver": "hogbom",
        "image_params": {"n_taylor_terms": n_taylor_terms},
    }
    before = deepcopy(config)
    with pytest.raises(ValueError, match="Continuum hogbom requires n_taylor_terms=1"):
        check_model_update(config, mode=CONTINUUM_SINGLE_FIELD)
    assert config == before


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
@pytest.mark.parametrize("restore", [False, True])
def test_single_term_hogbom_remains_supported(continuum_mode, restore):
    check_model_update(
        {
            "continuum_mode": continuum_mode,
            "restore": restore,
            "deconvolver": "hogbom",
            "image_params": {"n_taylor_terms": 1},
        },
        mode=CONTINUUM_SINGLE_FIELD,
    )
