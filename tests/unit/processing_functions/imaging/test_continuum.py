"""Unit tests for continuum processing algorithms and validation."""

import sys
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.add_visibility_grid_continuum import (
    add_visibility_grid_continuum_single_field,
)
from astroviper.processing_functions.imaging.add_visibility_grid_continuum_mvc import (
    add_visibility_grid_mvc_single_field,
)
from astroviper.processing_functions.imaging.degrid_visibility_grid import (
    degrid_visibility_grid_single_field,
)
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    accumulate_continuum_model,
    apply_mvc_primary_beam_convention,
    convert_mvc_cubes_to_taylor_normal_equations,
    finalize_mvc_taylor_normal_equations,
    form_mfs_residual_grid_from_cache,
    make_mvc_taylor_normal_equation_contributions,
    prepare_model_uv_continuum_single_field,
    prepare_model_uv_mvc_single_field,
    primary_beam_correct_restored_continuum,
    residual_update_continuum_single_field,
)
from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _convert_primary_beam_to_reference_frequency,
)
from astroviper.processing_functions.imaging.residual_cycle_continuum_single_field import (
    make_visibility_model_continuum_single_field,
)
from astroviper.processing_functions.imaging.utils.frequency_mapping import (
    map_visibility_frequencies_to_image,
)


def _model_dataset(value, *, include_primary_beam=False):
    model = xr.DataArray(
        np.full((1, 2, 1, 2, 2), value, dtype=np.float64),
        dims=("time", "taylor_term", "polarization", "l", "m"),
        attrs={"description": "persistent model attributes"},
    )
    data_vars = {"SKY_MODEL": model}
    if include_primary_beam:
        data_vars["PRIMARY_BEAM"] = xr.DataArray(
            np.ones((1, 1, 1, 2, 2), dtype=np.float64),
            dims=("time", "frequency", "polarization", "l", "m"),
        )
    return xr.Dataset(
        data_vars,
        attrs={"data_groups": {"model": {"sky": "SKY_MODEL"}}},
    )


def test_accumulate_continuum_model_copies_initial_mvc_state():
    """The initial MVC state retains its Taylor model and effective beam."""
    initial = _model_dataset(2.0, include_primary_beam=True)

    actual = accumulate_continuum_model(initial, specmode="mvc")

    assert set(actual.data_vars) == {"SKY_MODEL", "PRIMARY_BEAM"}
    np.testing.assert_array_equal(actual["SKY_MODEL"], initial["SKY_MODEL"])
    assert actual["SKY_MODEL"].data is not initial["SKY_MODEL"].data


def _mfs_grid_dataset(grid_value, normalization_value=4.0):
    """Construct a registered, globally reduced MFS Taylor UV grid."""
    dims = ("time", "taylor_term", "polarization", "u", "v")
    normalization_dims = ("time", "taylor_term", "polarization")
    return xr.Dataset(
        {
            "VISIBILITY": xr.DataArray(
                np.full((1, 2, 1, 2, 2), grid_value, dtype=np.complex128),
                dims=dims,
            ),
            "VISIBILITY_NORMALIZATION": xr.DataArray(
                np.full((1, 2, 1), normalization_value, dtype=np.float64),
                dims=normalization_dims,
            ),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "visibility": "VISIBILITY",
                    "visibility_normalization": "VISIBILITY_NORMALIZATION",
                }
            }
        },
    )


def test_cached_mfs_grid_subtracts_model_but_preserves_normalization():
    """Cached-grid MFS forms GWVobs-GWDmodel without subtracting weight sums."""
    observed = _mfs_grid_dataset(5.0)
    model = _mfs_grid_dataset(1.5)

    residual = form_mfs_residual_grid_from_cache(observed, model)

    np.testing.assert_array_equal(residual.VISIBILITY, 3.5)
    np.testing.assert_array_equal(residual.VISIBILITY_NORMALIZATION, 4.0)
    np.testing.assert_array_equal(observed.VISIBILITY, 5.0)
    np.testing.assert_array_equal(model.VISIBILITY, 1.5)
    assert residual.attrs["visibility_grid_source"] == "cached_observed_minus_model"


def test_cached_mfs_grid_ignores_model_normalization():
    """Zero predicted samples may reduce its sum; the observed sum must win."""
    residual = form_mfs_residual_grid_from_cache(
        _mfs_grid_dataset(5.0, normalization_value=4.0),
        _mfs_grid_dataset(1.5, normalization_value=3.0),
    )

    np.testing.assert_array_equal(residual.VISIBILITY, 3.5)
    np.testing.assert_array_equal(residual.VISIBILITY_NORMALIZATION, 4.0)


def test_accumulate_continuum_model_adds_later_increment_positionally():
    """A later increment reproduces the former driver-side model addition."""
    previous = _model_dataset(2.0)
    increment = _model_dataset(0.5)

    actual = accumulate_continuum_model(
        increment,
        previous_model_xds=previous,
        specmode="mfs",
    )

    np.testing.assert_array_equal(actual["SKY_MODEL"], 2.5)
    np.testing.assert_array_equal(previous["SKY_MODEL"], 2.0)
    assert actual["SKY_MODEL"].attrs == previous["SKY_MODEL"].attrs


def test_continuum_prediction_calls_shared_degridder_with_frequency_grid(monkeypatch):
    """Taylor grids are evaluated per channel without calling the cube API."""
    frequencies = np.array([0.9e9, 1.1e9])
    ms_xds = xr.Dataset(coords={"frequency": frequencies})
    model_grid = np.empty((1, 2, 1, 2, 2), dtype=np.complex128)
    model_grid[:, 0, ...] = 2.0
    model_grid[:, 1, ...] = 3.0
    model_xds = xr.Dataset(
        {
            "MODEL_UV": (
                ("time", "taylor_term", "polarization", "u", "v"),
                model_grid,
            )
        },
        coords={"taylor_term": [0, 1], "l": [0.0, 1.0], "m": [0.0, 1.0]},
        attrs={"data_groups": {"model": {"visibility": "MODEL_UV"}}},
    )
    captured = {}

    def fake_degrid(ms_arg, kernel, geometry, grid, frequency_map, **kwargs):
        captured.update(
            ms_xds=ms_arg,
            geometry=geometry,
            grid=grid.copy(),
            frequency_map=frequency_map.copy(),
            kwargs=kwargs,
        )

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.degrid_visibility_grid."
        "degrid_visibility_grid_single_field",
        fake_degrid,
    )
    make_visibility_model_continuum_single_field(
        {"partition": ms_xds},
        model_xds,
        np.ones(351),
        nterms=2,
        reference_frequency=1.0e9,
    )
    expected = np.broadcast_to(
        np.array([1.7, 2.3])[:, np.newaxis, np.newaxis], (2, 2, 2)
    )
    np.testing.assert_allclose(captured["grid"][0, :, 0, :, :], expected)
    np.testing.assert_array_equal(captured["frequency_map"], [0, 1])
    assert captured["ms_xds"] is ms_xds
    assert captured["geometry"] is model_xds


def test_primary_beam_correction_returns_restored_taylor_zero_only():
    """The PB-corrected continuum image has no Taylor-stack dimension."""
    image = xr.Dataset(
        {
            "SKY_RESTORED": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.asarray([[[[[4.0, 4.0]]], [[[40.0, 40.0]]]]]),
            ),
            "PRIMARY_BEAM": (
                ("time", "frequency", "polarization", "l", "m"),
                np.asarray([[[[[1.0, 0.1]]]]]),
            ),
        },
        coords={
            "time": [0],
            "taylor_term": [0, 1],
            "frequency": [1.5e9],
            "polarization": ["I"],
            "l": [0.0],
            "m": [0.0, 1.0],
        },
        attrs={
            "type": "image_dataset",
            "data_groups": {"restored": {"sky": "SKY_RESTORED"}},
        },
    )
    result = primary_beam_correct_restored_continuum(image, pblimit=0.2)
    corrected = result["SKY_RESTORED_PBCOR"]
    assert corrected.dims == ("time", "polarization", "l", "m")
    np.testing.assert_allclose(corrected.values, [[[[4.0, np.nan]]]], equal_nan=True)


@pytest.mark.parametrize(
    ("visibility", "image", "expected"),
    [
        ([1.1e9, 1.3e9], [1.0e9, 1.1e9, 1.2e9, 1.3e9], [1, 3]),
        ([1.0e9, 1.1e9], [1.0e9, 1.1e9], [0, 1]),
    ],
)
def test_frequency_mapping_matches_unique_image_channels(visibility, image, expected):
    """Full and subset channel selections map to their unique image planes."""
    np.testing.assert_array_equal(
        map_visibility_frequencies_to_image(visibility, image), expected
    )


@pytest.mark.parametrize(
    ("visibility", "image"),
    [([1.4e9], [1.0e9, 1.1e9]), ([1.0e9], [1.0e9, 1.0e9])],
)
def test_frequency_mapping_rejects_missing_or_ambiguous_channels(visibility, image):
    """Missing and duplicate image frequencies cannot define a unique map."""
    with pytest.raises(ValueError, match="exactly one image frequency"):
        map_visibility_frequencies_to_image(visibility, image)


def test_mvc_grid_accumulates_child_into_full_image_frequency_axis(monkeypatch):
    """MVC passes child channel indices and leaves unowned planes untouched."""
    import xradio.image.image_xds  # noqa: F401

    image_frequencies = np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9])
    child_frequencies = image_frequencies[[1, 3]]
    visibility_shape = (1, 1, child_frequencies.size, 1)
    ms_xds = xr.Dataset(
        {
            "VISIBILITY": (
                ("time", "baseline", "frequency", "polarization"),
                np.ones(visibility_shape, dtype=np.complex128),
            ),
            "WEIGHT_IMAGING": (
                ("time", "baseline", "frequency", "polarization"),
                np.ones(visibility_shape),
            ),
            "UVW": (
                ("time", "baseline", "uvw_label"),
                np.zeros((1, 1, 3)),
            ),
        },
        coords={"frequency": child_frequencies},
        attrs={
            "data_groups": {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "weight_imaging": "WEIGHT_IMAGING",
                    "uvw": "UVW",
                }
            }
        },
    )
    image = xr.Dataset(
        coords={
            "time": [0.0],
            "frequency": image_frequencies,
            "polarization": ["I"],
            "l": np.arange(4) * 1.0e-5,
            "m": np.arange(4) * 1.0e-5,
        },
        attrs={"type": "image_dataset", "data_groups": {"residual": {}}},
    )
    observed_channel_map = None

    def fake_grid(grid, normalization, *args, **kwargs):
        nonlocal observed_channel_map
        observed_channel_map = np.asarray(args[3])
        grid[:, observed_channel_map, ...] = 1.0
        normalization[:, observed_channel_map, ...] = 1.0

    module_name = (
        "astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp"
    )
    extension_module = ModuleType(module_name)
    extension_module.prolate_spheroidal_grid = fake_grid
    monkeypatch.setitem(sys.modules, module_name, extension_module)

    add_visibility_grid_mvc_single_field(ms_xds, np.ones(8), image)

    np.testing.assert_array_equal(observed_channel_map, [1, 3])
    assert image.VISIBILITY.shape[1] == image_frequencies.size
    assert np.all(image.VISIBILITY_NORMALIZATION.values[:, [1, 3], ...] == 1.0)
    assert np.all(image.VISIBILITY_NORMALIZATION.values[:, [0, 2], ...] == 0.0)


def test_mfs_grid_accumulates_normalization_across_processing_children(monkeypatch):
    """Repeated MFS child grids retain every child's normalization sum."""
    import xradio.image.image_xds  # noqa: F401

    def make_child(weight):
        visibility_shape = (1, 1, 1, 1)
        return xr.Dataset(
            {
                "VISIBILITY": (
                    ("time", "baseline", "frequency", "polarization"),
                    np.ones(visibility_shape, dtype=np.complex128),
                ),
                "WEIGHT_IMAGING": (
                    ("time", "baseline", "frequency", "polarization"),
                    np.full(visibility_shape, weight, dtype=np.float64),
                ),
                "UVW": (
                    ("time", "baseline", "uvw_label"),
                    np.zeros((1, 1, 3), dtype=np.float64),
                ),
            },
            coords={"frequency": [1.0e9]},
            attrs={
                "data_groups": {
                    "base": {
                        "correlated_data": "VISIBILITY",
                        "weight_imaging": "WEIGHT_IMAGING",
                        "uvw": "UVW",
                    }
                }
            },
        )

    image = xr.Dataset(
        coords={
            "time": [0.0],
            "polarization": ["XX"],
            "l": np.arange(4) * 1.0e-5,
            "m": np.arange(4) * 1.0e-5,
        },
        attrs={"type": "image_dataset", "data_groups": {"residual": {}}},
    )

    def fake_grid(grid, normalization, *args, **kwargs):
        imaging_weight = args[6]
        weight_sum = imaging_weight.sum(axis=(0, 1, 2))[None, None, :]
        grid += weight_sum[..., None, None]
        normalization += weight_sum

    module_name = (
        "astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp"
    )
    extension_module = ModuleType(module_name)
    extension_module.prolate_spheroidal_grid = fake_grid
    monkeypatch.setitem(sys.modules, module_name, extension_module)

    for child_weight in (2.0, 3.0):
        add_visibility_grid_continuum_single_field(
            make_child(child_weight),
            np.ones(8),
            image,
            nterms=1,
            reference_frequency=1.0e9,
        )

    np.testing.assert_allclose(image.VISIBILITY_NORMALIZATION, 5.0)
    np.testing.assert_allclose(image.VISIBILITY, 5.0)


def _frequency_cube(values, frequency, polarization="I"):
    """Create a scalar-pixel channel cube."""
    return xr.DataArray(
        np.asarray(values, dtype=np.float64)[None, :, None, None, None],
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={
            "time": [0],
            "frequency": frequency,
            "polarization": [polarization],
            "l": [0.0],
            "m": [0.0],
        },
    )


def _frequency_normalization(values, frequency):
    """Create channel normalization values."""
    return xr.DataArray(
        np.asarray(values, dtype=np.float64)[None, :, None],
        dims=("time", "frequency", "polarization"),
        coords={"time": [0], "frequency": frequency, "polarization": ["I"]},
    )


def test_mvc_conversion_forms_weighted_taylor_normal_equations():
    """MVC produces weighted residual, Hessian, and effective-PB terms."""
    frequency = np.array([0.8e9, 1.0e9, 1.2e9])
    x = np.array([-0.2, 0.0, 0.2])
    residual = np.array([2.0, 4.0, 8.0])
    psf = np.array([1.0, 2.0, 4.0])
    pb = np.array([0.5, 1.0, 1.5])
    weight = np.array([1.0, 2.0, 3.0])
    (
        residual_taylor,
        psf_taylor,
        effective_pb,
    ) = convert_mvc_cubes_to_taylor_normal_equations(
        _frequency_cube(residual, frequency),
        _frequency_cube(psf, frequency),
        _frequency_cube(pb, frequency),
        _frequency_normalization(weight, frequency),
        _frequency_normalization(weight, frequency),
        nterms=2,
        reference_frequency=1.0e9,
        pblimit=0.2,
    )
    expected_pb = np.sum(weight * pb) / np.sum(weight)
    np.testing.assert_allclose(effective_pb.squeeze(), expected_pb)
    np.testing.assert_allclose(
        residual_taylor.squeeze(),
        [
            np.sum(weight * residual) / np.sum(weight),
            np.sum(weight * x * residual) / np.sum(weight),
        ],
    )
    np.testing.assert_allclose(
        psf_taylor.squeeze(),
        [
            np.sum(weight * x**order * pb * psf) / (np.sum(weight) * expected_pb)
            for order in range(3)
        ],
    )


def test_mvc_conversion_uses_psf_weights_for_common_response():
    """MVC PSF and effective PB follow CASA's PSF-weight normalization."""
    frequency = np.array([0.9e9, 1.1e9])
    residual_weight = np.array([1.0, 3.0])
    psf_weight = np.array([2.0, 2.0])
    psf = np.array([1.0, 4.0])
    pb = np.array([0.5, 1.5])

    _, psf_taylor, effective_pb = convert_mvc_cubes_to_taylor_normal_equations(
        _frequency_cube([2.0, 6.0], frequency),
        _frequency_cube(psf, frequency),
        _frequency_cube(pb, frequency),
        _frequency_normalization(residual_weight, frequency),
        _frequency_normalization(psf_weight, frequency),
        nterms=1,
        reference_frequency=1.0e9,
        pblimit=0.2,
    )

    np.testing.assert_allclose(
        effective_pb.squeeze(),
        np.sum(psf_weight * pb) / np.sum(psf_weight),
    )
    np.testing.assert_allclose(
        psf_taylor.squeeze(),
        np.sum(psf_weight * pb * psf) / (np.sum(psf_weight) * effective_pb.squeeze()),
    )


def test_mvc_conversion_preserves_single_precision_image_products():
    """Float32 MVC cubes remain float32 despite float64 weight accumulation."""
    frequency = np.array([0.9e9, 1.1e9])
    cube = _frequency_cube([1.0, 2.0], frequency).astype(np.float32)
    normalization = _frequency_normalization([1.0, 1.0], frequency)

    residual, psf, primary_beam = convert_mvc_cubes_to_taylor_normal_equations(
        cube,
        cube,
        cube,
        normalization,
        normalization,
        nterms=2,
        reference_frequency=1.0e9,
    )

    assert residual.dtype == np.dtype(np.float32)
    assert psf.dtype == np.dtype(np.float32)
    assert primary_beam.dtype == np.dtype(np.float32)


def test_mvc_map_contributions_reduce_to_full_cube_result():
    """Summed map Taylor products reproduce the former global cube conversion."""
    frequency = np.array([0.8e9, 0.9e9, 1.1e9, 1.2e9])
    residual_cube = _frequency_cube([2.0, 3.0, 7.0, 11.0], frequency)
    psf_cube = _frequency_cube([1.0, 2.0, 4.0, 8.0], frequency)
    primary_beam_cube = _frequency_cube([0.5, 0.7, 1.1, 1.3], frequency)
    normalization = _frequency_normalization([1.0, 2.0, 3.0, 4.0], frequency)

    expected = convert_mvc_cubes_to_taylor_normal_equations(
        residual_cube,
        psf_cube,
        primary_beam_cube,
        normalization,
        normalization,
        nterms=2,
        reference_frequency=1.0e9,
    )
    chunks = []
    for channel_indices in ([0, 1], [2, 3]):
        selection = {"frequency": channel_indices}
        chunks.append(
            make_mvc_taylor_normal_equation_contributions(
                residual_cube.isel(selection),
                psf_cube.isel(selection),
                primary_beam_cube.isel(selection),
                normalization.isel(selection),
                normalization.isel(selection),
                nterms=2,
                reference_frequency=1.0e9,
            )
        )

    reduced = chunks[0].copy(deep=True)
    for variable_name in reduced.data_vars:
        reduced[variable_name] = reduced[variable_name] + chunks[1][variable_name]
    actual = finalize_mvc_taylor_normal_equations(reduced)

    for actual_array, expected_array in zip(actual, expected, strict=False):
        np.testing.assert_allclose(actual_array, expected_array)


def test_mvc_prediction_applies_channel_over_effective_primary_beam():
    """MVC scales a common-beam model by each channel beam."""
    frequency = np.array([0.9e9, 1.1e9])
    model = _frequency_cube([10.0, 20.0], frequency)
    channel_pb = _frequency_cube([0.5, 1.5], frequency, polarization="XX")
    effective_pb = xr.DataArray(
        np.array([[[[1.0]]]]),
        dims=("time", "polarization", "l", "m"),
        coords={"time": [0], "polarization": ["I"], "l": [0.0], "m": [0.0]},
    )
    actual = apply_mvc_primary_beam_convention(model, channel_pb, effective_pb)
    np.testing.assert_allclose(actual.squeeze(), [5.0, 30.0])


def test_mfs_primary_beam_is_evaluated_at_reference_frequency():
    """MFS replaces channel accumulators with a reference-frequency PB."""
    image_params = {
        "image_size": [8, 8],
        "cell_size": np.asarray([-2.0e-4, 2.0e-4]),
        "list_dish_diameters": np.asarray([10.7]),
        "list_blockage_diameters": np.asarray([0.75]),
        "primary_beam_ipower": 2,
    }
    image = xr.Dataset(
        coords={
            "time": [0],
            "frequency": [1.0e9, 2.0e9],
            "polarization": ["XX"],
            "l": np.arange(8),
            "m": np.arange(8),
        },
        attrs={"data_groups": {"residual": {}}},
    )
    result = _convert_primary_beam_to_reference_frequency(
        image,
        image_params=image_params,
        reference_frequency_hz=1.5e9,
        float_dtype=np.float64,
    )
    assert "PRIMARY_BEAM_REFERENCE" in result
    assert "PRIMARY_BEAM" not in result
    assert result["PRIMARY_BEAM_REFERENCE"].attrs["reference_frequency_hz"] == 1.5e9


def _taylor_model(value=1.0):
    """Create a canonical two-term Stokes model."""
    return xr.Dataset(
        {
            "SKY_MODEL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.full((1, 2, 1, 2, 2), value),
            )
        },
        coords={
            "time": [0],
            "taylor_term": [0, 1],
            "polarization": ["I"],
            "l": [0.0, 1.0],
            "m": [0.0, 1.0],
        },
        attrs={"data_groups": {"model": {"sky": "SKY_MODEL"}}},
    )


def test_prepare_mfs_model_transforms_and_fourier_converts_all_terms(monkeypatch):
    """MFS prediction preparation validates and registers its Taylor UV model."""
    monkeypatch.setattr(
        "astroviper.processing_functions.image_analysis.transform_polarization_basis."
        "transform_polarization_basis",
        lambda dataset, **kwargs: dataset,
    )

    def fake_fft(dataset, **kwargs):
        dataset["VISIBILITY_MODEL"] = dataset.SKY_MODEL.astype(np.complex64)
        dataset.attrs["data_groups"]["model"]["visibility"] = "VISIBILITY_MODEL"
        return dataset

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder."
        "fft_norm_continuum_img_xds",
        fake_fft,
    )
    result = prepare_model_uv_continuum_single_field(
        _taylor_model(), {"nterms": 2}, instrument_polarization_basis="linear"
    )
    assert result.VISIBILITY_MODEL.sizes["taylor_term"] == 2
    assert result.VISIBILITY_MODEL.attrs["nterms"] == 2


def test_prepare_mvc_model_evaluates_taylor_terms_and_channel_beams(monkeypatch):
    """MVC builds a channel cube, applies PB convention, and performs one FFT."""
    model = _taylor_model()
    model["PRIMARY_BEAM"] = xr.DataArray(
        np.ones((1, 1, 1, 2, 2)),
        dims=("time", "frequency", "polarization", "l", "m"),
    )
    channel_pb = xr.Dataset(
        {
            "PRIMARY_BEAM": xr.DataArray(
                np.ones((1, 2, 1, 2, 2)),
                dims=("time", "frequency", "polarization", "l", "m"),
            )
        },
        coords={
            "time": [0],
            "frequency": [0.9e9, 1.1e9],
            "polarization": ["I"],
            "l": [0.0, 1.0],
            "m": [0.0, 1.0],
        },
        attrs={"primary_beam_name": "PRIMARY_BEAM"},
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.image_analysis.transform_polarization_basis."
        "transform_polarization_basis",
        lambda dataset, **kwargs: dataset,
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder."
        "fft_norm_img_xds",
        lambda dataset, **kwargs: dataset,
    )
    result = prepare_model_uv_mvc_single_field(
        model,
        channel_pb,
        [0.9e9, 1.1e9],
        {"nterms": 2, "reference_frequency_hz": 1.0e9},
    )
    assert result.SKY_MODEL_MVC.dims == (
        "time",
        "frequency",
        "polarization",
        "l",
        "m",
    )


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
def test_residual_update_wraps_setup_cycle_and_local_mvc_ifft(monkeypatch, specmode):
    """The processing entry point runs setup once and MVC image conversion locally."""
    import importlib

    import pandas as pd
    import xradio.image.image_xds  # noqa: F401

    module = importlib.import_module(
        "astroviper.processing_functions.imaging.image_continuum_single_field"
    )
    image = xr.Dataset(
        {
            "PRIMARY_BEAM": xr.DataArray([1.0], dims=("frequency",)),
            "VISIBILITY_NORMALIZATION": xr.DataArray([1.0], dims=("frequency",)),
            "UV_SAMPLING_NORMALIZATION": xr.DataArray([1.0], dims=("frequency",)),
        },
        coords={"frequency": [1.0e9]},
        attrs={
            "type": "image_dataset",
            "data_groups": {"residual": {"primary_beam": "PRIMARY_BEAM"}},
        },
    )
    monkeypatch.setattr(
        module,
        "imaging_preparation_continuum_single_field",
        lambda *args, **kwargs: (args[1], pd.DataFrame({"T_setup": [1.0]}), 1.0),
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.residual_cycle_continuum_single_field."
        "residual_cycle_continuum_single_field",
        lambda *args, **kwargs: (args[1], pd.DataFrame({"T_grid": [1.0]})),
    )

    def fake_ifft(dataset, image_data_group_out_modified, **kwargs):
        name = next(iter(image_data_group_out_modified.values()))
        dataset[name] = xr.DataArray([0.0], dims=("frequency",))
        return dataset

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder."
        "ifft_norm_img_xds",
        fake_ifft,
    )
    monkeypatch.setattr(
        module,
        "make_mvc_taylor_normal_equation_contributions",
        lambda *args, **kwargs: xr.Dataset(
            {
                "MVC_RESIDUAL_TAYLOR_NUMERATOR": xr.DataArray([0.0], dims=("x",)),
                "MVC_RESIDUAL_WEIGHT_SUM": xr.DataArray([1.0], dims=("x",)),
            }
        ),
    )
    result, timing = residual_update_continuum_single_field(
        {}, image, {"nterms": 2}, {}, specmode=specmode, is_n_iter_0=True
    )
    assert timing.iloc[0].nterms == 2
    if specmode == "mvc":
        assert "SKY_RESIDUAL_MVC_CUBE" in result
        assert "POINT_SPREAD_FUNCTION_MVC_CUBE" in result


def test_shared_degridder_registers_output_and_calls_cpp_kernel(monkeypatch):
    """The shared primitive validates mappings and invokes the numerical kernel."""
    import xradio.image.image_xds  # noqa: F401

    ms = xr.Dataset(
        {
            "VISIBILITY": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.zeros((1, 1, 2, 1), dtype=np.complex128),
            ),
            "UVW": (("time", "baseline_id", "uvw_label"), np.zeros((1, 1, 3))),
        },
        coords={"frequency": [1.0e9, 1.1e9]},
        attrs={
            "data_groups": {"base": {"correlated_data": "VISIBILITY", "uvw": "UVW"}}
        },
    )
    geometry = xr.Dataset(
        coords={"l": [0.0, 1.0], "m": [0.0, 1.0]}, attrs={"type": "image_dataset"}
    )
    called = {}

    def fake_kernel(grid, vis, *args, **kwargs):
        called["shape"] = grid.shape
        vis[...] = 3.0

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp."
        "prolate_spheroidal_degrid",
        fake_kernel,
    )
    degrid_visibility_grid_single_field(
        ms,
        np.ones(351),
        geometry,
        np.ones((1, 2, 1, 2, 2), dtype=np.complex64),
        [0, 1],
        fft_padding=1.0,
    )
    assert called["shape"] == (1, 2, 1, 2, 2)
    np.testing.assert_allclose(ms.VISIBILITY_MODEL, 3.0)


def test_shared_degridder_allocates_model_from_weights_without_observed_data(
    monkeypatch,
):
    """Cached-grid MFS prediction does not require loading visibility values."""
    import xradio.image.image_xds  # noqa: F401

    visibility_dims = ("time", "baseline_id", "frequency", "polarization")
    ms = xr.Dataset(
        {
            "WEIGHT_IMAGING": (visibility_dims, np.ones((1, 1, 2, 1))),
            "UVW": (("time", "baseline_id", "uvw_label"), np.zeros((1, 1, 3))),
        },
        coords={"frequency": [1.0e9, 1.1e9]},
        attrs={
            "data_groups": {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "uvw": "UVW",
                    "weight_imaging": "WEIGHT_IMAGING",
                }
            }
        },
    )
    geometry = xr.Dataset(
        coords={"l": [0.0, 1.0], "m": [0.0, 1.0]},
        attrs={"type": "image_dataset"},
    )

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp."
        "prolate_spheroidal_degrid",
        lambda grid, vis, *args, **kwargs: vis.fill(2.0),
    )

    degrid_visibility_grid_single_field(
        ms,
        np.ones(351),
        geometry,
        np.ones((1, 2, 1, 2, 2), dtype=np.complex64),
        [0, 1],
        fft_padding=1.0,
    )

    assert "VISIBILITY" not in ms
    assert ms.VISIBILITY_MODEL.dims == visibility_dims
    np.testing.assert_array_equal(ms.VISIBILITY_MODEL, 2.0)
