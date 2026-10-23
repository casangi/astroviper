"""Early validation of configurations previously rejected by imaging helpers."""

from copy import deepcopy
from types import MappingProxyType

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_config,
    check_driver_workflow,
    check_model_update,
    check_residual_update,
)
from astroviper.processing_functions.imaging.config_check._output import (
    check_output_layout,
)
from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
    check_continuum_beam_metadata,
)


@pytest.mark.parametrize("mode", [CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD])
@pytest.mark.parametrize(
    "options,message",
    [
        ({"fft_backend": "missing"}, "fft_backend"),
        ({"gridder": "mosaic"}, "gridder"),
        ({"complex_dtype": "float64"}, "complex_dtype"),
        ({"image_params": {"image_size": [1, 8]}}, "at least 2"),
    ],
)
def test_shared_residual_options_fail_before_gridding(mode, options, message):
    with pytest.raises((ValueError, TypeError), match=message):
        check_residual_update(options, mode=mode)


def test_missing_optional_fft_backend_fails_early(monkeypatch):
    import importlib

    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "pyfftw.interfaces.scipy_fft":
            raise ImportError("pyfftw unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="pyfftw"):
        check_residual_update({"fft_backend": "pyfftw"}, mode=CUBE_SINGLE_FIELD)


def test_mvc_time_restriction_is_mode_specific():
    config = {"continuum_mode": "mvc", "image_params": {"time_coords": [0.0, 1.0]}}
    with pytest.raises(ValueError, match="one image time plane"):
        check_residual_update(config, mode=CONTINUUM_SINGLE_FIELD)
    check_residual_update(config, mode=CUBE_SINGLE_FIELD)
    config["continuum_mode"] = "mfs"
    check_residual_update(config, mode=CONTINUUM_SINGLE_FIELD)


@pytest.mark.parametrize(
    "beam",
    [
        {"primary_beam_model": "unknown"},
        {"list_dish_diameters": []},
        {"list_dish_diameters": [0.0]},
        {"list_dish_diameters": [12.0], "list_blockage_diameters": [12.0]},
        {"list_dish_diameters": [12.0], "list_blockage_diameters": [0.0, 0.0]},
        {"list_blockage_diameters": [-1.0]},
        {"list_blockage_diameters": [[0.0]]},
        {"primary_beam_max_radius_1ghz": np.nan},
    ],
)
def test_explicit_continuum_beam_rejected(beam):
    with pytest.raises((ValueError, TypeError)):
        check_residual_update({"image_params": beam}, mode=CONTINUUM_SINGLE_FIELD)


@pytest.mark.parametrize(
    "params",
    [
        [],
        {"typo": 1},
        {"telescope": "unknown"},
        {"effective_dish_diameter": -1.0},
        {"blockage_diameter": np.inf},
        {"max_rad_1GHz": -1.0},
        {"telescope": "vla", "effective_dish_diameter": 25.0},
        {"telescope": "vla", "max_rad_1GHz": None},
        {"beam_model": {}, "telescope": "alma"},
        {"beam_model": {"dish_diameter": 12.0, "max_rad_1GHz": None}},
        {"beam_model": {"func": "bad", "dish_diameter": 12.0, "max_rad_1GHz": 1.0}},
        {
            "beam_model": {
                "dish_diameter": 12.0,
                "blockage_diameter": 13.0,
                "max_rad_1GHz": 1.0,
            }
        },
        {"beam_model": xr.Dataset({"JONES": ("x", [1.0])})},
    ],
)
def test_cube_primary_beam_rejected(params):
    with pytest.raises((ValueError, TypeError)):
        check_residual_update({"primary_beam_params": params}, mode=CUBE_SINGLE_FIELD)


def polynomial_model():
    return xr.Dataset(
        {"BPC": (("frequency", "polarization", "coefficient"), np.ones((1, 1, 2)))},
        coords={"frequency": [1e9], "polarization": ["I"], "coefficient": [0, 1]},
        attrs={"dish_diameter": 25.0, "max_rad_1GHz": 0.01, "n_sample": None},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_sample", 1),
        ("n_sample", True),
        ("max_rad_1GHz", np.nan),
        ("dish_diameter", 0),
    ],
)
def test_polynomial_beam_metadata_rejected(field, value):
    model = polynomial_model().assign_attrs({field: value})
    with pytest.raises((ValueError, TypeError)):
        check_residual_update(
            {"primary_beam_params": {"beam_model": model}}, mode=CUBE_SINGLE_FIELD
        )


def test_read_only_beam_configuration_is_preserved():
    model = polynomial_model()
    snapshot = model.copy(deep=True)
    params = MappingProxyType({"beam_model": model})
    check_residual_update({"primary_beam_params": params}, mode=CUBE_SINGLE_FIELD)
    xr.testing.assert_identical(model, snapshot)
    analytic = MappingProxyType(
        {"func": "airy", "dish_diameter": 12.0, "max_rad_1GHz": None}
    )
    check_residual_update(
        {"primary_beam_params": {"beam_model": analytic}}, mode=CUBE_SINGLE_FIELD
    )
    assert "blockage_diameter" not in analytic


@pytest.mark.parametrize("diameters", [[12.0, 7.0], [0.0], [np.nan]])
def test_inferred_beam_fails_before_psf(diameters):
    antenna = xr.Dataset({"ANTENNA_DISH_DIAMETER": ("antenna_name", diameters)})
    with pytest.raises((ValueError, NotImplementedError)):
        check_continuum_beam_metadata({}, antenna, continuum_mode="mfs")


def test_inferred_alma_aperture_and_explicit_override():
    antenna = xr.Dataset(
        {"ANTENNA_DISH_DIAMETER": ("antenna_name", [15.0])},
        attrs={"overall_telescope_name": "ALMA"},
    )
    with pytest.raises(ValueError, match="12-m or 7-m"):
        check_continuum_beam_metadata({}, antenna, continuum_mode="mvc")
    params = {"list_dish_diameters": [10.7]}
    check_continuum_beam_metadata(params, antenna, continuum_mode="mvc")
    assert params == {"list_dish_diameters": [10.7]}


@pytest.mark.parametrize(
    "params,padding",
    [
        ([], 1.2),
        ({"unknown": 1}, 1.2),
        ({"type": "bad"}, 1.2),
        ({"support": 2}, 1.2),
        ({"support": True}, 1.2),
        ({"support": 5}, 1.2),
        ({"oversampling": 0}, 1.2),
        ({"oversampling": True}, 1.2),
        ({"interpolation": "cubic"}, 1.2),
        ({"type": "least_misfit"}, None),
        ({"type": "least_misfit"}, 1.01),
    ],
)
def test_kernel_recipe_rejected_before_grid(params, padding):
    config = {"gridding_kernel_params": params, "image_params": {}}
    if padding is not None:
        config["image_params"]["fft_padding"] = padding
    with pytest.raises((ValueError, TypeError)):
        check_residual_update(config, mode=CUBE_SINGLE_FIELD)


def test_kernel_design_reuses_production_cache():
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        _make_gridding_kernel,
        make_gridding_kernel,
    )

    params = {"type": "exponential_semicircle", "support": 5, "oversampling": 12}
    before = deepcopy(params)
    check_residual_update(
        {"gridding_kernel_params": params, "image_params": {"fft_padding": 1.5}},
        mode=CUBE_SINGLE_FIELD,
    )
    hits = _make_gridding_kernel.cache_info().hits
    make_gridding_kernel(params, fft_padding=1.5)
    assert _make_gridding_kernel.cache_info().hits == hits + 1
    assert params == before


def test_kernel_edge_correction_is_rejected_during_parameter_check():
    # This stencil/padding combination is known to fall below the 1e-6 floor.
    with pytest.raises(ValueError, match="below the floor"):
        check_residual_update(
            {
                "gridding_kernel_params": {
                    "type": "exponential_semicircle",
                    "support": 17,
                },
                "image_params": {"fft_padding": 1.05},
            },
            mode=CUBE_SINGLE_FIELD,
        )


@pytest.mark.parametrize(
    "control,value",
    [
        ("threshold", [1.0, 2.0, 3.0]),
        ("threshold", np.ones((1, 1, 2))),
        ("threshold_per_cycle", np.ones((1, 2))),
        ("threshold_per_cycle", np.ones((1, 2, 2))),
        ("max_iter_per_cycle_updated", np.ones((2, 1, 2), dtype=int)),
        ("max_iter", [1, 2]),
    ],
)
def test_control_shapes_rejected(control, value):
    config = {
        "image_params": {"time_coords": [0.0], "polarization_coords": ["I", "Q"]},
        "cycle_control_params": {control: value},
    }
    with pytest.raises(ValueError, match=control):
        check_model_update(config, mode=CONTINUUM_SINGLE_FIELD)


def test_supported_control_shapes_preserve_arrays():
    values = np.ones((1, 1, 2))
    values.flags.writeable = False
    config = {
        "image_params": {"time_coords": [0.0], "polarization_coords": ["I", "Q"]},
        "cycle_control_params": {
            "threshold": [0.0, 1.0],
            "threshold_per_cycle": values,
        },
    }
    check_model_update(config, mode=CONTINUUM_SINGLE_FIELD)
    assert config["cycle_control_params"]["threshold_per_cycle"] is values


@pytest.mark.parametrize(
    "options",
    [
        {"psf_fitting_method": "bad"},
        {"npix_window": [4, 5]},
        {"npix_window": [3]},
        {"sampling": [1, 2]},
        {"sampling": [2.0, 3.0]},
        {"cutoff": np.nan},
        {"psf_cube": np.zeros((1,)), "psf_normalization": None},
    ],
)
def test_fitting_and_direct_helper_controls(options):
    with pytest.raises((ValueError, TypeError)):
        check_model_update(options, mode=CUBE_SINGLE_FIELD)


def fits_config(**image):
    return {"output_image_format": "fits", "skunk_works": True, "image_params": image}


@pytest.mark.parametrize(
    "image",
    [
        {"time_coords": [0.0, 1.0]},
        {"image_size": [1, 4]},
        {"polarization_coords": ["I", "V"]},
        {"polarization_coords": ["bad"]},
        {"frequency_coords": [1e9, 1.3e9, 1.4e9]},
        {"frequency_coords": [1e9, 1e9]},
        {"direction_reference": "GALACTIC"},
    ],
)
def test_fits_coordinates_rejected_without_approximation(image):
    with pytest.raises(ValueError, match="FITS"):
        check_driver_workflow(fits_config(**image), mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize("frequency", [[1e9], [1.2e9, 1.1e9, 1e9]])
def test_fits_single_and_descending_uniform_channels(frequency):
    config = fits_config(
        time_coords=[0.0],
        image_size=[4, 4],
        frequency_coords=frequency,
        polarization_coords=["I", "Q"],
        direction_reference="ICRS",
    )
    config["image_data_variables_keep"] = ["sky_residual"]
    check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize(
    "kept", [["typo"], ["visibility"], ["beam_fit_params_point_spread_function"], []]
)
def test_fits_kept_variables_rejected(kept):
    config = fits_config()
    config["image_data_variables_keep"] = kept
    with pytest.raises(ValueError):
        check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)


def test_restore_effective_output_does_not_mutate_keep_list():
    config = fits_config()
    config.update(restore=True, image_data_variables_keep=[])
    check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)
    assert config["image_data_variables_keep"] == []


@pytest.mark.parametrize(
    "options",
    [
        {"image_chunking": []},
        {"image_chunking": {"bogus": 1}},
        {"image_chunking": {"frequency": True}},
        {"image_chunking": {"frequency": 0}},
        {"msv2_open_options": []},
        {"msv2_open_options": {"array_backend": "numpy"}},
        {"msv2_open_options": {"scan_intents": None}},
        {"compressor": object()},
    ],
)
def test_workflow_schema_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        check_driver_workflow(options, mode=CUBE_SINGLE_FIELD)


def test_layout_uses_task_boundaries_when_available():
    config = {
        "image_params": {"frequency_coords": list(range(8))},
        "image_chunking": {"frequency": 3},
    }
    check_output_layout(config)
    with pytest.raises(ValueError, match="divide every node task"):
        check_output_layout(
            config,
            parallel_coords={
                "frequency": {"data_chunks": {0: np.arange(4), 1: np.arange(4, 8)}}
            },
        )


@pytest.mark.parametrize(
    "sharding",
    [
        {"index_codecs": [{"name": "crc32c"}], "codecs": [{"name": "bytes"}]},
        {"codecs": [{"name": "bytes"}, {"name": "zstd"}, {"name": "blosc"}]},
        {"codecs": [{"name": "bytes"}, {"name": "unknown"}]},
    ],
)
def test_planned_sharded_codec_chain_rejected(sharding):
    config = {"skunk_works": True, "zarr_meta": {"SKY": {"sharding": sharding}}}
    with pytest.raises(ValueError):
        check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)


def test_planned_sharded_codec_chain_is_read_only():
    config = {
        "skunk_works": True,
        "zarr_meta": {
            "SKY": {
                "sharding": {
                    "index_codecs": [{"name": "bytes"}],
                    "codecs": [{"name": "bytes"}, {"name": "blosc"}],
                }
            }
        },
    }
    snapshot = deepcopy(config)
    check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)
    assert config == snapshot


def test_unsupported_codec_rejected_only_for_direct_writer(monkeypatch):
    from zarr.codecs import ZstdCodec

    # Third-party BytesBytesCodec implementations are valid for Zarr, but their
    # serialized names must also be implemented by the direct writer.
    monkeypatch.setattr(
        ZstdCodec, "to_dict", lambda self: {"name": "third_party_codec"}
    )
    config = {"compressor": ZstdCodec()}
    check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)
    with pytest.raises(ValueError, match="direct writer"):
        check_driver_workflow({**config, "skunk_works": True}, mode=CUBE_SINGLE_FIELD)


def test_shared_coordinator_rejects_audit_probe_regressions():
    for mode, config in [
        (CONTINUUM_SINGLE_FIELD, {"fft_backend": "bad"}),
        (
            CONTINUUM_SINGLE_FIELD,
            {"continuum_mode": "mvc", "image_params": {"time_coords": [1.0, 2.0]}},
        ),
        (CONTINUUM_SINGLE_FIELD, {"image_params": {"primary_beam_model": "bad"}}),
        (CUBE_SINGLE_FIELD, {"psf_fitting_method": "bad"}),
        (CUBE_SINGLE_FIELD, {"gridding_kernel_params": {"support": 2}}),
        (CUBE_SINGLE_FIELD, {"primary_beam_params": {"effective_dish_diameter": -1}}),
        (
            CONTINUUM_SINGLE_FIELD,
            {
                "image_params": {"polarization_coords": ["I"]},
                "cycle_control_params": {"threshold": [1.0, 2.0]},
            },
        ),
    ]:
        with pytest.raises((ValueError, TypeError)):
            check_config(config, mode=mode)


def test_direct_gridder_checks_once_then_trusts_checked_input(monkeypatch):
    import astroviper.processing_functions.imaging.config_check as checks
    from astroviper.processing_functions.imaging.add_visibility_grid_continuum import (
        add_visibility_grid_continuum_single_field,
    )

    calls = []

    def shared(config, **kwargs):
        calls.append(config)
        raise ValueError("shared checker reached")

    monkeypatch.setattr(checks, "check_residual_update", shared)
    with pytest.raises(ValueError, match="shared checker reached"):
        add_visibility_grid_continuum_single_field(None, None, None, -1, 1e9)
    assert len(calls) == 1

    class RuntimeInput:
        @property
        def attrs(self):
            raise RuntimeError("runtime input reached")

    with pytest.raises(RuntimeError, match="runtime input reached"):
        add_visibility_grid_continuum_single_field(
            RuntimeInput(), None, None, 1, 1e9, input_checked=True
        )
    assert len(calls) == 1


def test_checked_fitter_still_requires_runtime_data_group(monkeypatch):
    import astroviper.processing_functions.imaging.config_check as checks
    from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
        point_spread_function_gaussian_fit,
    )

    def unexpected_check(*args, **kwargs):
        raise AssertionError("Repeated the shared configuration check")

    monkeypatch.setattr(checks, "check_model_update", unexpected_check)
    import xarray as xr

    with pytest.raises(AssertionError, match="Data group image not found"):
        point_spread_function_gaussian_fit(xr.Dataset(), input_checked=True)


def test_direct_cube_setup_checks_fitter_before_input_access():
    from astroviper.processing_functions.imaging.residual_update import (
        imaging_setup_single_field,
    )

    with pytest.raises(ValueError, match="psf_fitting_method"):
        imaging_setup_single_field(None, None, {}, {}, psf_fitting_method="bad")


def test_frequency_matching_public_spelling_rejected_early():
    with pytest.raises(ValueError, match="frequency_matching"):
        check_residual_update({"frequency_matching": "bad"}, mode=CUBE_SINGLE_FIELD)
