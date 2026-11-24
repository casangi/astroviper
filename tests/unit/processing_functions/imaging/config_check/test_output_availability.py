"""Requested output products must be available before any input or graph I/O."""

from copy import deepcopy

import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_driver_workflow,
)
from tests.unit.processing_functions.imaging.config_check.test_full_validation import (
    configuration,
)


@pytest.mark.parametrize(
    "key",
    [
        "alpha",
        "alpha_error",
        "beta",
    ],
)
def test_cube_rejects_automatic_spectral_products(key):
    with pytest.raises(ValueError, match="separate spectral_products"):
        check_driver_workflow(
            {"image_data_variables_keep": [key], "restore": True},
            mode=CUBE_SINGLE_FIELD,
        )


@pytest.mark.parametrize(
    "mode,keys",
    [
        (CUBE_SINGLE_FIELD, ["sky_restored", "sky_primary_beam_corrected_restored"]),
        (
            CONTINUUM_SINGLE_FIELD,
            [
                "sky_restored",
                "sky_primary_beam_corrected_restored",
            ],
        ),
    ],
)
def test_restoration_products_require_restoration(mode, keys):
    for key in keys:
        with pytest.raises(ValueError, match="require restore=True"):
            check_driver_workflow(
                {"image_data_variables_keep": [key], "restore": False}, mode=mode
            )


@pytest.mark.parametrize(
    "key",
    [
        "sky_restored_flat_noise",
        "sky_restored_flat_sky",
        "sky_restored_primary_beam_corrected",
    ],
)
def test_continuum_legacy_keys_report_replacements(key):
    with pytest.raises(ValueError, match="unknown imaging variable"):
        check_driver_workflow(
            {"image_data_variables_keep": [key], "restore": True},
            mode=CONTINUUM_SINGLE_FIELD,
        )


@pytest.mark.parametrize(
    "key,nterms", [("alpha", 1), ("alpha_error", 1), ("beta", 1), ("beta", 2)]
)
def test_spectral_products_require_their_taylor_coefficients(key, nterms):
    with pytest.raises(ValueError, match="separate spectral_products"):
        check_driver_workflow(
            {
                "image_data_variables_keep": [key],
                "restore": True,
                "image_params": {"n_taylor_terms": nterms},
            },
            mode=CONTINUUM_SINGLE_FIELD,
        )


def test_cube_corrected_output_requires_correction():
    with pytest.raises(ValueError, match="unavailable with restoration_output"):
        check_driver_workflow(
            {
                "image_data_variables_keep": ["sky_primary_beam_corrected_restored"],
                "restore": True,
                "restoration_output": "apparent",
            },
            mode=CUBE_SINGLE_FIELD,
        )


@pytest.mark.parametrize(
    "mode,kept,nterms,correct",
    [
        (CUBE_SINGLE_FIELD, ["sky_residual"], 1, False),
        (CUBE_SINGLE_FIELD, ["sky_restored"], 1, False),
        (CUBE_SINGLE_FIELD, ["sky_primary_beam_corrected_restored"], 1, True),
        (
            CONTINUUM_SINGLE_FIELD,
            ["sky_restored", "sky_primary_beam_corrected_restored"],
            1,
            False,
        ),
    ],
)
def test_available_products_preserve_configuration(mode, kept, nterms, correct):
    config = {
        "image_data_variables_keep": kept,
        "image_params": {"n_taylor_terms": nterms},
        "restore": True,
        **(
            {"restoration_output": "both" if correct else "apparent"}
            if mode.imaging_kind == "cube"
            else {"restoration_output": "both"}
        ),
    }
    before = deepcopy(config)
    check_driver_workflow(config, mode=mode)
    assert config == before


@pytest.mark.parametrize(
    "kind,key,restore,nterms",
    [
        ("cube", "alpha", True, 1),
        ("cube", "sky_restored", False, 1),
        ("continuum", "alpha", False, 2),
        ("continuum", "sky_restored", False, 2),
        ("continuum", "sky_primary_beam_corrected_restored", False, 2),
        ("continuum", "beta", True, 2),
        ("continuum", "sky_restored", True, 2),
    ],
)
def test_public_drivers_reject_unavailable_outputs_before_opening_input(
    monkeypatch, tmp_path, kind, key, restore, nterms
):
    import xradio.measurement_set

    from astroviper.distributed_applications.imaging import (
        image_continuum_single_field,
        image_cube_single_field,
    )

    def unexpected_open(*args, **kwargs):
        pytest.fail("Input opened before output-capability validation")

    monkeypatch.setattr(xradio.measurement_set, "open_processing_set", unexpected_open)
    config = configuration()
    config["image_params"]["n_taylor_terms"] = nterms
    config["deconvolver"] = "hogbom" if kind == "cube" else "mtmfs"
    driver = image_cube_single_field if kind == "cube" else image_continuum_single_field
    output = tmp_path / "unavailable.img.zarr"
    with pytest.raises(ValueError):
        driver(
            ps_store="unopened.ps.zarr",
            image_store=str(output),
            image_data_variables_keep=[key],
            restore=restore,
            **config,
        )
    assert not output.exists()


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
def test_distributed_fits_keeps_model_and_residual_with_empty_optional_list(mode):
    from astroviper.processing_functions.imaging.config_check._output import (
        check_kept_variables,
    )

    config = {"image_data_variables_keep": [], "restore": False}
    check_kept_variables(config, mode=mode, fits=True, distributed=True)
    assert config["image_data_variables_keep"] == []
