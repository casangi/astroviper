"""Unit tests for continuum processing algorithms and validation."""

import sys
import warnings
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
    _convert_mvc_cubes_to_taylor_normal_equations,
    accumulate_continuum_model,
    apply_mvc_primary_beam_convention,
    finalize_mvc_taylor_normal_equations,
    form_mfs_residual_grid_from_cache,
    form_residual_grid_from_cache,
    make_mvc_taylor_normal_equation_contributions,
    prepare_model_uv_continuum_single_field,
    prepare_model_uv_mvc_single_field,
    residual_update_continuum_single_field,
    restore_continuum_image,
)
from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _convert_primary_beam_to_reference_frequency,
)
from astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field import (
    add_uv_sampling_grid_continuum_single_field,
)
from astroviper.processing_functions.imaging.residual_cycle_continuum_single_field import (
    make_visibility_model_continuum_single_field,
    residual_cycle_continuum_single_field,
)
from astroviper.processing_functions.imaging.utils import Key
from astroviper.processing_functions.imaging.utils.frequency_mapping import (
    map_visibility_frequencies_to_image,
)
from tests.utils.continuum_images import make_continuum_image


def test_psf_taylor_gridding_uses_output_views_and_reuses_order_zero_weights(
    monkeypatch,
):
    """Each Taylor order is gridded in place without a full temporary plane."""
    weights = np.ones((1, 1, 2, 1), dtype=np.float64, order="C")
    measurement_set = xr.Dataset(
        {
            "WEIGHT_IMAGING": (
                ("time", "baseline", "frequency", "polarization"),
                weights,
            ),
            "UVW": (
                ("time", "baseline", "uvw_label"),
                np.zeros((1, 1, 3), dtype=np.float64),
            ),
        },
        coords={"frequency": [90.0, 110.0]},
        attrs={
            "data_groups": {
                "base": {
                    "weight_imaging": "WEIGHT_IMAGING",
                    "uvw": "UVW",
                }
            }
        },
    )
    image = xr.Dataset(
        coords={"l": np.arange(4) * -1.0e-5, "m": np.arange(4) * 1.0e-5},
        attrs={"type": "image_dataset", "data_groups": {"residual": {}}},
    )
    calls = []

    def fake_grid(grid, normalization, *args, **kwargs):
        order = len(calls)
        calls.append((grid, args[5]))
        grid[...] += order + 1
        normalization[...] = 7.0 + order

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.gridders.anti_alias_grid_cpp."
        "anti_alias_grid_uv_sampling",
        fake_grid,
    )

    add_uv_sampling_grid_continuum_single_field(
        measurement_set,
        np.ones(1),
        image,
        n_psf_taylor_terms=3,
        reference_frequency=100.0,
        fft_padding=1.0,
    )

    output_grid = image["UV_SAMPLING"].data
    assert len(calls) == 3
    assert all(np.shares_memory(grid, output_grid) for grid, _ in calls)
    assert calls[0][1] is weights
    np.testing.assert_array_equal(output_grid[:, 0], 1.0)
    np.testing.assert_array_equal(output_grid[:, 1], 2.0)
    np.testing.assert_array_equal(output_grid[:, 2], 3.0)
    np.testing.assert_array_equal(image["UV_SAMPLING_NORMALIZATION"], 7.0)


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

    actual = accumulate_continuum_model(initial, continuum_mode="mvc")

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


def test_cached_mvc_frequency_grid_subtracts_model_plane_by_plane():
    """MVC cache subtraction preserves every task-local channel and weight sum."""
    dims = ("time", "frequency", "polarization", "u", "v")
    normalization_dims = ("time", "frequency", "polarization")
    attrs = {
        "data_groups": {
            "residual": {
                "visibility": "VISIBILITY",
                "visibility_normalization": "VISIBILITY_NORMALIZATION",
            }
        }
    }
    coordinates = {"frequency": [1.0e9, 1.1e9]}
    observed = xr.Dataset(
        {
            "VISIBILITY": xr.DataArray(
                np.asarray([4.0, 7.0], dtype=np.complex128).reshape(1, 2, 1, 1, 1),
                dims=dims,
                coords=coordinates,
            ),
            "VISIBILITY_NORMALIZATION": xr.DataArray(
                np.asarray([2.0, 3.0]).reshape(1, 2, 1),
                dims=normalization_dims,
                coords=coordinates,
            ),
        },
        attrs=attrs,
    )
    model = observed.copy(deep=True)
    model["VISIBILITY"].data[...] = np.asarray([1.0, 2.5]).reshape(1, 2, 1, 1, 1)
    model["VISIBILITY_NORMALIZATION"].data[...] = 1.0

    residual = form_residual_grid_from_cache(observed, model, grid_kind="MVC")

    np.testing.assert_allclose(
        residual.VISIBILITY.values.ravel(),
        np.asarray([3.0, 4.5]),
    )
    np.testing.assert_array_equal(
        residual.VISIBILITY_NORMALIZATION,
        observed.VISIBILITY_NORMALIZATION,
    )


def test_cached_grid_accepts_distinct_observed_and_model_groups():
    """Each cached-grid input resolves its explicitly named data group."""
    observed = _mfs_grid_dataset(5.0)
    model = _mfs_grid_dataset(1.5)
    observed.attrs["data_groups"]["observed"] = observed.attrs["data_groups"].pop(
        "residual"
    )
    model.attrs["data_groups"]["predicted"] = model.attrs["data_groups"].pop("residual")

    residual = form_residual_grid_from_cache(
        observed,
        model,
        image_data_group_in_observed="observed",
        image_data_group_in_model="predicted",
    )

    np.testing.assert_array_equal(residual.VISIBILITY, 3.5)
    np.testing.assert_array_equal(residual.VISIBILITY_NORMALIZATION, 4.0)


def test_accumulate_continuum_model_adds_later_increment_positionally():
    """A later increment reproduces the former driver-side model addition."""
    previous = _model_dataset(0.0)
    previous.SKY_MODEL.loc[{"taylor_term": 0}] = 2.0
    previous.SKY_MODEL.loc[{"taylor_term": 1}] = 20.0
    increment = _model_dataset(0.0)
    increment.SKY_MODEL.loc[{"taylor_term": 0}] = 0.5
    increment.SKY_MODEL.loc[{"taylor_term": 1}] = 5.0

    actual = accumulate_continuum_model(
        increment,
        previous_model_xds=previous,
        continuum_mode="mfs",
    )

    np.testing.assert_array_equal(actual.SKY_MODEL.isel(taylor_term=0), 2.5)
    np.testing.assert_array_equal(actual.SKY_MODEL.isel(taylor_term=1), 25.0)
    np.testing.assert_array_equal(previous.SKY_MODEL.isel(taylor_term=0), 2.0)
    np.testing.assert_array_equal(previous.SKY_MODEL.isel(taylor_term=1), 20.0)
    assert actual["SKY_MODEL"].attrs == previous["SKY_MODEL"].attrs


def test_continuum_prediction_degrids_and_accumulates_each_taylor_plane(monkeypatch):
    """Taylor planes are degridded separately and combined in visibility space."""
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
    captured = []

    def fake_degrid(ms_arg, kernel, geometry, grid, frequency_map, **kwargs):
        captured.append(
            {
                "ms_xds": ms_arg,
                "geometry": geometry,
                "grid": grid.copy(),
                "frequency_map": frequency_map.copy(),
                "kwargs": kwargs,
            }
        )
        ms_arg["VISIBILITY_MODEL"] = xr.DataArray(
            np.full(frequencies.size, grid[0, 0, 0, 0, 0], dtype=np.complex128),
            dims=("frequency",),
            coords={"frequency": frequencies},
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
        n_taylor_terms=2,
        reference_frequency=1.0e9,
        ms_data_group_in_name="corrected",
    )
    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0]["grid"], model_grid[:, 0:1, ...])
    np.testing.assert_array_equal(captured[1]["grid"], model_grid[:, 1:2, ...])
    for call in captured:
        np.testing.assert_array_equal(call["frequency_map"], [0, 0])
        assert call["ms_xds"] is ms_xds
        assert call["geometry"] is model_xds
        assert call["kwargs"]["ms_data_group_in_name"] == "corrected"
    np.testing.assert_allclose(ms_xds.VISIBILITY_MODEL, [1.7, 2.3])


def test_primary_beam_correction_restores_every_taylor_term():
    """PB correction uses the model-first convention for every Taylor term."""
    from astroviper.processing_functions.imaging.restore import (
        _clean_beam_kernel_ft,
        primary_beam_corrected_plane,
    )

    image = make_continuum_image()
    primary_beam = np.linspace(0.25, 1.0, 64).reshape(1, 1, 8, 8)
    image["PRIMARY_BEAM"].data[...] = primary_beam
    model = image.SKY_MODEL.values.copy()
    residual = image.SKY_RESIDUAL.values.copy()
    beam = image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values

    result, _ = restore_continuum_image(
        image,
        primary_beam_correction=True,
        primary_beam_limit=0.2,
    )
    corrected = result["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"]
    assert corrected.dims == ("time", "taylor_term", "polarization", "l", "m")

    kernel_ft = _clean_beam_kernel_ft(beam[0, 0], 8, 8, 1.0e-5, model.dtype, 1)
    for term in range(image.sizes["taylor_term"]):
        expected = primary_beam_corrected_plane(
            primary_beam[0, 0],
            0.2,
            model[0, term, 0],
            residual[0, term, 0],
            kernel_ft=kernel_ft,
        )
        np.testing.assert_allclose(corrected.values[0, term, 0], expected)
        old_post_restore_division = (
            result.SKY_RESTORED.values[0, term, 0] / primary_beam[0, 0]
        )
        assert not np.allclose(corrected.values[0, term, 0], old_post_restore_division)


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
        map_visibility_frequencies_to_image(visibility, image, matching="exact"),
        expected,
    )


@pytest.mark.parametrize(
    ("visibility", "image"),
    [([1.4e9], [1.0e9, 1.1e9]), ([1.0e9], [1.0e9, 1.0e9])],
)
def test_frequency_mapping_rejects_missing_or_ambiguous_channels(visibility, image):
    """Missing and duplicate image frequencies cannot define a unique map."""
    with pytest.raises(ValueError, match="exactly one image frequency"):
        map_visibility_frequencies_to_image(visibility, image, matching="exact")


@pytest.mark.parametrize("frequency_shift", [0.0, 1.0e6])
def test_mvc_grid_accumulates_child_into_full_image_frequency_axis(
    monkeypatch, frequency_shift
):
    """MVC passes child channel indices and leaves unowned planes untouched."""
    import xradio.image.image_xds  # noqa: F401

    image_frequencies = np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9])
    child_frequencies = image_frequencies[[1, 3]] + frequency_shift
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
    observed_frequency_map = None

    def fake_grid(grid, normalization, *args, **kwargs):
        nonlocal observed_frequency_map
        observed_frequency_map = np.asarray(args[3])
        grid[:, observed_frequency_map, ...] = 1.0
        normalization[:, observed_frequency_map, ...] = 1.0

    module_name = "astroviper.processing_functions.imaging.gridders.anti_alias_grid_cpp"
    extension_module = ModuleType(module_name)
    extension_module.anti_alias_grid = fake_grid
    monkeypatch.setitem(sys.modules, module_name, extension_module)

    add_visibility_grid_mvc_single_field(ms_xds, np.ones(8), image)

    np.testing.assert_array_equal(observed_frequency_map, [1, 3])
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

    module_name = "astroviper.processing_functions.imaging.gridders.anti_alias_grid_cpp"
    extension_module = ModuleType(module_name)
    extension_module.anti_alias_grid = fake_grid
    monkeypatch.setitem(sys.modules, module_name, extension_module)

    for child_weight in (2.0, 3.0):
        add_visibility_grid_continuum_single_field(
            make_child(child_weight),
            np.ones(8),
            image,
            n_taylor_terms=1,
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


@pytest.mark.parametrize("mvc_primary_beam_approach", ["casa", "preconditioned"])
def test_mvc_conversion_matches_casa_memo_taylor_equations(
    mvc_primary_beam_approach,
):
    """Both MVC approaches produce the CASA/Memo-I Taylor equations."""
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
    ) = _convert_mvc_cubes_to_taylor_normal_equations(
        _frequency_cube(residual, frequency),
        _frequency_cube(psf, frequency),
        _frequency_cube(pb, frequency),
        _frequency_normalization(weight, frequency),
        _frequency_normalization(weight, frequency),
        n_taylor_terms=2,
        reference_frequency=1.0e9,
        primary_beam_limit=0.2,
        mvc_primary_beam_approach=mvc_primary_beam_approach,
    )
    expected_pb = np.sum(weight * pb) / np.sum(weight)
    np.testing.assert_allclose(effective_pb.squeeze(), expected_pb)
    np.testing.assert_allclose(
        residual_taylor.squeeze(),
        [np.sum(weight * x**order * residual) / np.sum(weight) for order in range(2)],
    )
    np.testing.assert_allclose(
        psf_taylor.squeeze(),
        [
            np.sum(weight * x**order * pb * psf) / (np.sum(weight) * expected_pb)
            for order in range(3)
        ],
    )
    assert (
        residual_taylor.attrs["mvc_primary_beam_approach"] == mvc_primary_beam_approach
    )
    assert psf_taylor.attrs["mvc_primary_beam_approach"] == mvc_primary_beam_approach
    assert effective_pb.attrs["mvc_primary_beam_approach"] == mvc_primary_beam_approach


def test_mvc_casa_and_preconditioned_approaches_use_distinct_evaluation_order():
    """CASA divides then multiplies while Memo I uses the cancelled residual."""
    frequency = np.array([1.0e9])
    residual = _frequency_cube([14.305966], frequency).astype(np.float32)
    psf = _frequency_cube([1.0], frequency).astype(np.float32)
    primary_beam = _frequency_cube([0.7793717], frequency).astype(np.float32)
    normalization = _frequency_normalization([1.0], frequency)

    products = {}
    for approach in ("casa", "preconditioned"):
        products[approach] = _convert_mvc_cubes_to_taylor_normal_equations(
            residual,
            psf,
            primary_beam,
            normalization,
            normalization,
            n_taylor_terms=1,
            reference_frequency=1.0e9,
            primary_beam_limit=0.2,
            mvc_primary_beam_approach=approach,
        )[0]

    np.testing.assert_allclose(products["casa"], products["preconditioned"], rtol=1e-6)
    assert not np.array_equal(
        products["casa"].values, products["preconditioned"].values
    )


def test_mvc_legacy_approach_restores_spatial_g_inverse_preconditioner():
    """Legacy MVC returns ``Abar C G^-1 u`` and beam-free Taylor PSFs."""
    frequency = np.array([0.8e9, 1.0e9, 1.2e9])
    x = np.array([-0.2, 0.0, 0.2])
    residual = np.array([2.0, 4.0, 8.0])
    psf = np.array([1.0, 2.0, 4.0])
    primary_beam = np.array([0.5, 1.0, 1.5])
    weight = np.array([1.0, 2.0, 3.0])

    residual_taylor, psf_taylor, effective_pb = (
        _convert_mvc_cubes_to_taylor_normal_equations(
            _frequency_cube(residual, frequency),
            _frequency_cube(psf, frequency),
            _frequency_cube(primary_beam, frequency),
            _frequency_normalization(weight, frequency),
            _frequency_normalization(weight, frequency),
            n_taylor_terms=2,
            reference_frequency=1.0e9,
            primary_beam_limit=0.2,
            mvc_primary_beam_approach="legacy",
        )
    )

    basis = x[:, None] ** np.arange(2)
    spectral_moments = basis.T @ (weight[:, None] * basis) / np.sum(weight)
    gram = basis.T @ ((weight * primary_beam**2)[:, None] * basis)
    numerator = basis.T @ (weight * primary_beam * residual)
    expected_pb = np.sum(weight * primary_beam) / np.sum(weight)
    expected_residual = (
        expected_pb * spectral_moments @ np.linalg.solve(gram, numerator)
    )
    expected_psf = [
        np.sum(weight * x**order * psf) / np.sum(weight) for order in range(3)
    ]

    np.testing.assert_allclose(effective_pb.squeeze(), expected_pb)
    np.testing.assert_allclose(residual_taylor.squeeze(), expected_residual)
    np.testing.assert_allclose(psf_taylor.squeeze(), expected_psf)
    assert residual_taylor.attrs["mvc_primary_beam_approach"] == "legacy"
    assert psf_taylor.attrs["mvc_primary_beam_approach"] == "legacy"


def test_mvc_conversion_uses_psf_weights_for_common_response():
    """MVC PSF and effective PB follow CASA's PSF-weight normalization."""
    frequency = np.array([0.9e9, 1.1e9])
    residual_weight = np.array([1.0, 3.0])
    psf_weight = np.array([2.0, 2.0])
    psf = np.array([1.0, 4.0])
    pb = np.array([0.5, 1.5])

    _, psf_taylor, effective_pb = _convert_mvc_cubes_to_taylor_normal_equations(
        _frequency_cube([2.0, 6.0], frequency),
        _frequency_cube(psf, frequency),
        _frequency_cube(pb, frequency),
        _frequency_normalization(residual_weight, frequency),
        _frequency_normalization(psf_weight, frequency),
        n_taylor_terms=1,
        reference_frequency=1.0e9,
        primary_beam_limit=0.2,
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

    residual, psf, primary_beam = _convert_mvc_cubes_to_taylor_normal_equations(
        cube,
        cube,
        cube,
        normalization,
        normalization,
        n_taylor_terms=2,
        reference_frequency=1.0e9,
    )

    assert residual.dtype == np.dtype(np.float32)
    assert psf.dtype == np.dtype(np.float32)
    assert primary_beam.dtype == np.dtype(np.float32)


@pytest.mark.parametrize(
    "mvc_primary_beam_approach", ["casa", "preconditioned", "legacy"]
)
def test_mvc_map_contributions_reduce_to_full_cube_result(
    mvc_primary_beam_approach,
):
    """Summed map Taylor products reproduce the former global cube conversion."""
    frequency = np.array([0.8e9, 0.9e9, 1.1e9, 1.2e9])
    residual_cube = _frequency_cube([2.0, 3.0, 7.0, 11.0], frequency)
    psf_cube = _frequency_cube([1.0, 2.0, 4.0, 8.0], frequency)
    primary_beam_cube = _frequency_cube([0.5, 0.7, 1.1, 1.3], frequency)
    normalization = _frequency_normalization([1.0, 2.0, 3.0, 4.0], frequency)

    expected = _convert_mvc_cubes_to_taylor_normal_equations(
        residual_cube,
        psf_cube,
        primary_beam_cube,
        normalization,
        normalization,
        n_taylor_terms=2,
        reference_frequency=1.0e9,
        mvc_primary_beam_approach=mvc_primary_beam_approach,
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
                n_taylor_terms=2,
                reference_frequency=1.0e9,
                mvc_primary_beam_approach=mvc_primary_beam_approach,
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


def test_mvc_prediction_masks_zero_effective_beam_without_warning():
    """Invalid PB support is zeroed without evaluating a masked division."""
    frequency = np.array([0.9e9, 1.1e9])
    model = _frequency_cube([10.0, 20.0], frequency)
    channel_pb = _frequency_cube([0.5, 1.5], frequency, polarization="XX")
    effective_pb = xr.DataArray(
        np.zeros((1, 1, 1, 1)),
        dims=("time", "polarization", "l", "m"),
        coords={"time": [0], "polarization": ["I"], "l": [0.0], "m": [0.0]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = apply_mvc_primary_beam_convention(model, channel_pb, effective_pb)
    np.testing.assert_array_equal(actual, 0.0)


@pytest.mark.parametrize(
    "channel_pb",
    [
        [1.0, 1.0, 1.0, 1.0],
        [0.61, 0.54, 0.47, 0.40],
        [0.55, 0.42, 0.31, 0.19],
    ],
    ids=["centre", "off_centre", "channel_below_pblimit"],
)
def test_mvc_residual_uses_casa_cancelled_primary_beam_moments(channel_pb):
    """MVC residuals match CASA's cancelled ``A_nu / A_nu`` expression."""
    frequency = np.array([0.85e9, 0.95e9, 1.05e9, 1.15e9])
    x = (frequency - 1.0e9) / 1.0e9
    weight = _frequency_normalization([1.0, 2.0, 3.0, 4.0], frequency)
    model_change = np.array([0.3, -0.2])
    options = dict(n_taylor_terms=2, reference_frequency=1.0e9, primary_beam_limit=0.2)

    def two_pixel_cube(centre, source):
        values = np.stack(
            [np.broadcast_to(centre, 4), np.broadcast_to(source, 4)], axis=-1
        )
        return xr.DataArray(
            values[None, :, None, :, None],
            dims=("time", "frequency", "polarization", "l", "m"),
            coords={
                "time": [0],
                "frequency": frequency,
                "polarization": ["I"],
                "l": [0.0, 1.0],
                "m": [0.0],
            },
        )

    channel_beam = two_pixel_cube(1.0, channel_pb)
    _, _, effective_beam = _convert_mvc_cubes_to_taylor_normal_equations(
        two_pixel_cube(0.0, 0.0),
        two_pixel_cube(1.0, 0.0),
        channel_beam,
        weight,
        weight,
        **options,
    )
    prediction = apply_mvc_primary_beam_convention(
        two_pixel_cube(0.0, model_change[0] + model_change[1] * x),
        channel_beam,
        effective_beam,
    )
    residual, _, _ = finalize_mvc_taylor_normal_equations(
        make_mvc_taylor_normal_equation_contributions(
            -prediction, None, channel_beam, weight, None, **options
        ),
        effective_primary_beam=effective_beam,
    )

    valid = np.asarray(channel_pb) > options["primary_beam_limit"]
    predicted_source = prediction.isel(l=1, m=0).squeeze().values
    expected = np.asarray(
        [
            -np.sum(weight.squeeze().values * x**order * predicted_source * valid)
            / np.sum(weight.squeeze().values)
            for order in range(2)
        ]
    )
    np.testing.assert_allclose(residual.isel(l=1, m=0).squeeze(), expected)


@pytest.mark.parametrize(
    "channel_pb",
    [[0.9, 0.7, 0.5, 1.0e-6], [0.9, 1.0e-6]],
    ids=["three_channels_above_pblimit", "one_channel_above_pblimit"],
)
def test_mvc_channel_near_beam_null_adds_bounded_residual(channel_pb):
    """A channel near a null of its beam does not amplify its residual.

    CASA masks a channel beam below ``primary_beam_limit`` before its residual
    correction. The later multiplication by that beam cancels the correction
    for valid channels, so a near-null channel neither contributes nor causes
    noise amplification.
    """
    nchan = len(channel_pb)
    frequency = np.linspace(0.85e9, 1.15e9, nchan)
    weight = _frequency_normalization(np.ones(nchan), frequency)
    for channel in range(nchan):
        residual, _, effective_beam = _convert_mvc_cubes_to_taylor_normal_equations(
            _frequency_cube(np.eye(nchan)[channel], frequency),
            _frequency_cube(np.ones(nchan), frequency),
            _frequency_cube(channel_pb, frequency),
            weight,
            weight,
            n_taylor_terms=2,
            reference_frequency=1.0e9,
            primary_beam_limit=0.2,
        )
        assert float(effective_beam.squeeze()) > 0.2
        assert np.all(np.isfinite(residual.values))
        assert np.max(np.abs(residual.values)) < 1.0


def _mvc_image_point_sources(sources, threshold):
    """Image point sources with the MVC Taylor products and native MT-MFS.

    Channel residuals are formed in the image domain, without noise or
    gridding error, as ``B_nu (*) (A_nu * I_nu - A_nu * M_nu / Abar)``.  The
    power beam ``A_nu`` is a Gaussian of 40 pixels FWHM and the PSF ``B_nu``
    a Gaussian of 1.5 pixels sigma at the reference frequency, both
    narrowing as ``1 / nu``; 8 channels cover +-12.5 %.  Each model update
    cleans to ``max(0.05 * peak, threshold)``, the controller's per-cycle
    threshold for a PSF without sidelobes, and imaging stops when the
    refreshed Taylor-zero peak in the beam support is below ``threshold``.
    Returns the Taylor model and the effective beam.
    """
    from astroviper.processing_functions.imaging.deconvolvers import mtmfs

    size, centre, n_taylor_terms = 64, 32, 2
    x = (np.arange(8) - 3.5) / 32.0
    frequency = 1.0e11 * (1.0 + x)
    offsets = np.arange(size) - centre
    radius = np.hypot(*np.meshgrid(offsets, offsets, indexing="ij"))
    channel_beam = np.exp(
        -4.0 * np.log(2.0) * (radius * (1.0 + x)[:, None, None] / 40.0) ** 2
    )
    # Channel PSFs on a doubled grid, for convolutions without wrap-around.
    doubled = np.arange(2 * size) - size
    doubled_radius = np.hypot(*np.meshgrid(doubled, doubled, indexing="ij"))
    sigma = 1.5 / (1.0 + x)[:, None, None]
    doubled_psf = np.exp(-0.5 * (doubled_radius / sigma) ** 2)
    psf_ft = np.fft.rfft2(np.fft.ifftshift(doubled_psf, axes=(1, 2)))
    window = slice(size - centre, 2 * size - centre)
    psf = doubled_psf[:, window, window]

    sky = np.zeros((8, size, size))
    for pixel, spectrum in sources:
        sky[(slice(None), *pixel)] += channel_beam[(slice(None), *pixel)] * spectrum

    coords = {
        "time": [0.0],
        "frequency": frequency,
        "polarization": ["I"],
        "l": offsets * 1.0e-5,
        "m": offsets * 1.0e-5,
    }

    def cube(values):
        return xr.DataArray(
            values[None, :, None],
            dims=("time", "frequency", "polarization", "l", "m"),
            coords=coords,
        )

    weight = xr.DataArray(
        np.ones((1, 8, 1)),
        dims=("time", "frequency", "polarization"),
        coords={name: coords[name] for name in ("time", "frequency", "polarization")},
    )
    basis = x[:, None] ** np.arange(n_taylor_terms)
    model = np.zeros((n_taylor_terms, size, size))
    effective_beam = None
    for _ in range(20):
        apparent = sky.copy()
        if effective_beam is not None:
            apparent -= apply_mvc_primary_beam_convention(
                cube(np.einsum("ct,tij->cij", basis, model)),
                cube(channel_beam),
                effective_beam,
            ).values[0, :, 0]
        padded = np.zeros((8, 2 * size, 2 * size))
        padded[:, :size, :size] = apparent
        residual_cube = np.fft.irfft2(np.fft.rfft2(padded) * psf_ft)[:, :size, :size]
        first = effective_beam is None
        residual, psf_taylor, beam = finalize_mvc_taylor_normal_equations(
            make_mvc_taylor_normal_equation_contributions(
                cube(residual_cube),
                cube(psf) if first else None,
                cube(channel_beam),
                weight,
                weight if first else None,
                n_taylor_terms=n_taylor_terms,
                reference_frequency=1.0e11,
                primary_beam_limit=0.2,
            ),
            effective_primary_beam=effective_beam,
        )
        if first:
            effective_beam = beam
            support = (beam.values[0, 0] > 0.2).astype(np.float64)
            hessian_planes = np.ascontiguousarray(
                psf_taylor.transpose(
                    "time", "psf_taylor_order", "polarization", "l", "m"
                ).values[0, :, 0],
                dtype=np.float64,
            )
        residual = np.ascontiguousarray(
            residual.transpose("time", "taylor_term", "polarization", "l", "m").values[
                0, :, 0
            ],
            dtype=np.float64,
        )
        peak = float(np.max(np.abs(residual[0]) * support))
        if peak < threshold:
            return model, effective_beam.values[0, 0]
        increment = np.zeros_like(residual)
        mtmfs.clean(
            residual,
            hessian_planes,
            increment,
            mask=support,
            max_iter_per_cycle_updated=50000,
            gain=0.1,
            threshold_per_cycle=max(0.05 * peak, threshold),
        )
        model += increment
    pytest.fail("MVC imaging did not reach the threshold in 20 imaging cycles.")


@pytest.mark.parametrize(
    ("off_centre_index", "expected_casa_index"),
    [(0.0, -0.27), (-0.7, -1.00)],
)
def test_mvc_characterizes_casa_off_axis_spectral_index_approximation(
    off_centre_index, expected_casa_index
):
    """The exact CASA equations retain CASA's known off-axis index bias.

    One source sits at the pointing centre and one where the beam is 0.5 at
    the reference frequency. Imaging stops at 5 mJy, 2 % of the apparent flux
    of the off-centre source. CASA's common-beam approximation uses the
    pointing-centre Hessian away from the centre; in this controlled setup the
    off-centre ``tt1 / tt0`` therefore reaches about -0.27 for a flat source
    and -1.00 for a source with index -0.7. This test deliberately records
    that approximation while separately checking the on-axis source.
    """
    x = (np.arange(8) - 3.5) / 32.0
    sources = {
        "centre": ((32, 32), 1.0 * (1.0 + x) ** -0.7),
        "off_centre": ((52, 32), 0.5 * (1.0 + x) ** off_centre_index),
    }
    model, effective_beam = _mvc_image_point_sources(sources.values(), 5.0e-3)

    basis = x[:, None] ** np.arange(2)
    centre_pixel, centre_spectrum = sources["centre"]
    centre_expected = np.linalg.lstsq(basis, centre_spectrum, rcond=None)[0]
    row, column = centre_pixel
    centre_tt0, centre_tt1 = model[:, row - 2 : row + 3, column - 2 : column + 3].sum(
        axis=(1, 2)
    )
    np.testing.assert_allclose(
        centre_tt0,
        effective_beam[row, column] * centre_expected[0],
        rtol=0.03,
    )
    np.testing.assert_allclose(
        centre_tt1 / centre_tt0,
        centre_expected[1] / centre_expected[0],
        atol=0.01,
    )

    (row, column), off_centre_spectrum = sources["off_centre"]
    off_centre_expected = np.linalg.lstsq(basis, off_centre_spectrum, rcond=None)[0]
    off_centre_tt0, off_centre_tt1 = model[
        :, row - 2 : row + 3, column - 2 : column + 3
    ].sum(axis=(1, 2))
    np.testing.assert_allclose(
        off_centre_tt0,
        effective_beam[row, column] * off_centre_expected[0],
        rtol=0.03,
    )
    np.testing.assert_allclose(
        off_centre_tt1 / off_centre_tt0,
        expected_casa_index,
        atol=0.02,
    )


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
        image_data_group_in_name="residual",
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
    model = _taylor_model(0.0)
    model.SKY_MODEL.loc[{"taylor_term": 0}] = 2.0
    model.SKY_MODEL.loc[{"taylor_term": 1}] = 3.0
    monkeypatch.setattr(
        "astroviper.processing_functions.image_analysis.transform_polarization_basis."
        "transform_polarization_basis",
        lambda dataset, **kwargs: dataset,
    )

    fft_options = {}

    def fake_fft(dataset, **kwargs):
        fft_options.update(kwargs)
        dataset["VISIBILITY_MODEL"] = dataset.SKY_MODEL.astype(np.complex64)
        dataset.attrs["data_groups"][kwargs["image_data_group_out_name"]] = {
            "visibility": "VISIBILITY_MODEL"
        }
        return dataset

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder."
        "fft_norm_continuum_img_xds",
        fake_fft,
    )
    result = prepare_model_uv_continuum_single_field(
        model,
        {"n_taylor_terms": 2},
        instrument_polarization_basis="linear",
        image_data_group_in_name="model",
        image_data_group_out_name="model_uv",
    )
    assert result.VISIBILITY_MODEL.sizes["taylor_term"] == 2
    assert result.VISIBILITY_MODEL.attrs["n_taylor_terms"] == 2
    assert fft_options["image_data_variables_keep"] == []
    assert fft_options["image_data_group_in_name"] == "model"
    assert fft_options["image_data_group_out_name"] == "model_uv"
    np.testing.assert_array_equal(
        result.VISIBILITY_MODEL.isel(taylor_term=0), 2.0 + 0.0j
    )
    np.testing.assert_array_equal(
        result.VISIBILITY_MODEL.isel(taylor_term=1), 3.0 + 0.0j
    )


def test_prepare_mvc_model_evaluates_taylor_terms_and_channel_beams(monkeypatch):
    """MVC preparation is repeatable and does not rewrite the caller's group."""
    model = _taylor_model(0.0)
    model.SKY_MODEL.loc[{"taylor_term": 0}] = 2.0
    model.SKY_MODEL.loc[{"taylor_term": 1}] = 3.0
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
    fft_options = {}

    def fake_fft(dataset, **kwargs):
        fft_options.update(kwargs)
        return dataset

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder."
        "fft_norm_img_xds",
        fake_fft,
    )
    result = prepare_model_uv_mvc_single_field(
        model,
        channel_pb,
        [0.9e9, 1.1e9],
        {"n_taylor_terms": 2, "reference_frequency": 1.0e9},
        image_data_group_in_name="model",
        image_data_group_out_name="model_uv",
    )
    repeated = prepare_model_uv_mvc_single_field(
        model,
        channel_pb,
        [0.9e9, 1.1e9],
        {"n_taylor_terms": 2, "reference_frequency": 1.0e9},
        image_data_group_in_name="model",
        image_data_group_out_name="model_uv",
    )
    assert model.attrs["data_groups"]["model"] == {"sky": "SKY_MODEL"}
    assert result.attrs["data_groups"] is not model.attrs["data_groups"]
    assert result.attrs["data_groups"]["model_uv"]["sky"] == "SKY_MODEL_MVC"
    assert fft_options["image_data_group_in_name"] == "model_uv"
    assert fft_options["image_data_group_out_name"] == "model_uv"
    assert result.SKY_MODEL_MVC.dims == (
        "time",
        "frequency",
        "polarization",
        "l",
        "m",
    )
    np.testing.assert_allclose(
        result.SKY_MODEL_MVC.isel(time=0, polarization=0, l=0, m=0),
        [1.7, 2.3],
    )
    xr.testing.assert_identical(repeated.SKY_MODEL_MVC, result.SKY_MODEL_MVC)


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_residual_update_wraps_setup_cycle_and_local_mvc_ifft(
    monkeypatch, continuum_mode
):
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
        "astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder."
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
        {},
        image,
        {"n_taylor_terms": 2, "reference_frequency": 1.0e9},
        {},
        continuum_mode=continuum_mode,
        is_first_cycle=True,
    )
    assert timing.iloc[0].n_taylor_terms == 2
    if continuum_mode == "mvc":
        assert "SKY_RESIDUAL_MVC_CUBE" in result
        assert "POINT_SPREAD_FUNCTION_MVC_CUBE" in result
        np.testing.assert_array_equal(result["MVC_RESIDUAL_TAYLOR_NUMERATOR"], [0.0])
        np.testing.assert_array_equal(result["MVC_RESIDUAL_WEIGHT_SUM"], [1.0])


@pytest.mark.parametrize("continuum_mode,degrid_time", [("mfs", 0.3), ("mvc", 0.5)])
@pytest.mark.parametrize("visibility_memory_mode", ["recompute", "in_memory"])
def test_residual_cycle_times_prediction_stages_once(
    monkeypatch, continuum_mode, degrid_time, visibility_memory_mode
):
    """Prediction, residual formation, and gridding have independent timers."""
    import time

    import pandas as pd

    image = xr.Dataset(
        coords={"frequency": [1.0e9]},
        attrs={"data_groups": {"residual": {}}},
    )
    image_params = {
        "n_taylor_terms": 2,
        "reference_frequency": 1.0e9,
        "fft_padding": 1.2,
    }

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.gridding_convolution_functions."
        "gcf_prolate_spheroidal."
        "create_prolate_spheroidal_kernel_1D",
        lambda *args: np.ones(1),
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.residual_cycle_continuum_single_field."
        "make_visibility_model_continuum_single_field",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.image_continuum_single_field."
        "prepare_model_uv_mvc_single_field",
        lambda *args, **kwargs: xr.Dataset(),
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.residual_update."
        "make_visibility_model_single_field",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.residual_update."
        "calculate_residual_visibilities",
        lambda *args, **kwargs: None,
    )

    def fake_grid(*args, **kwargs):
        return args[1], pd.DataFrame()

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.make_undeconvolved_image_continuum."
        "make_undeconvolved_image_continuum_single_field",
        fake_grid,
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.make_undeconvolved_image_continuum_mvc."
        "make_undeconvolved_image_mvc_single_field",
        fake_grid,
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.image_continuum_single_field."
        "form_residual_grid_from_cache",
        lambda observed, current, **kwargs: current,
    )

    times = [0.0, 0.1, 1.0, 1.0 + degrid_time]
    if visibility_memory_mode == "recompute":
        times.extend([2.0, 2.7])
    times.extend([3.0, 3.4])
    clock = iter(times)
    monkeypatch.setattr(time, "time", lambda: next(clock))

    _, timing = residual_cycle_continuum_single_field(
        {},
        image,
        xr.Dataset(),
        image_params,
        is_first_cycle=False,
        continuum_mode=continuum_mode,
        model_xds=xr.Dataset(),
        primary_beam_xds=xr.Dataset(),
        visibility_memory_mode=visibility_memory_mode,
        observed_visibility_grid_xds=(
            xr.Dataset() if continuum_mode == "mvc" else None
        ),
    )

    assert timing.T_degrid.iloc[0] == pytest.approx(degrid_time)
    assert timing.T_residual_vis.iloc[0] == pytest.approx(
        0.7 if visibility_memory_mode == "recompute" else 0.0
    )
    assert timing.T_grid.iloc[0] == pytest.approx(0.4)


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
        "astroviper.processing_functions.imaging.gridders.anti_alias_grid_cpp."
        "anti_alias_degrid",
        fake_kernel,
    )
    degrid_visibility_grid_single_field(
        ms,
        np.ones(400),  # support-7 half kernel table, oversampling 100
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
        "astroviper.processing_functions.imaging.gridders.anti_alias_grid_cpp."
        "anti_alias_degrid",
        lambda grid, vis, *args, **kwargs: vis.fill(2.0),
    )

    degrid_visibility_grid_single_field(
        ms,
        np.ones(400),  # support-7 half kernel table, oversampling 100
        geometry,
        np.ones((1, 2, 1, 2, 2), dtype=np.complex64),
        [0, 1],
        fft_padding=1.0,
    )

    assert "VISIBILITY" not in ms
    assert ms.VISIBILITY_MODEL.dims == visibility_dims
    np.testing.assert_array_equal(ms.VISIBILITY_MODEL, 2.0)


@pytest.mark.parametrize("with_mask", [False, True])
def test_continuum_statistics_use_only_taylor_zero_and_preserve_input(with_mask):
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        build_continuum_residual_imaging_dict,
    )
    from astroviper.processing_functions.imaging.utils import (
        CycleController,
        get_per_cycle_limits,
    )

    residual = np.full((2, 3, 2, 3, 4), 1000.0)
    peaks = np.array([[2.0, 3.0], [4.0, 5.0]])
    residual[:, 0] = peaks[..., None, None]
    roles = {"sky": "RESIDUAL", "max_sidelobe_point_spread_function": "SIDELOBE"}
    image = xr.Dataset(
        {
            "RESIDUAL": (("time", "taylor_term", "polarization", "l", "m"), residual),
            # Deliberately reversed axes: expansion must use dimension names.
            "SIDELOBE": (("polarization", "time"), np.full((2, 2), 0.2)),
        },
        attrs={"data_groups": {"residual": roles}},
    )
    if with_mask:
        mask = np.ones((3, 4), dtype=bool)
        mask[0] = False
        image["MASK"] = (("l", "m"), mask)
        roles["mask"] = "MASK"
    original = image.copy(deep=True)
    params = {"gain": 0.1}
    stats = build_continuum_residual_imaging_dict(image, "residual", params)
    assert len(stats.data) == 4
    for key, values in stats.data.items():
        assert key.chan == 0
        assert values["peakres"] == [peaks[key.time, key.pol]]
        assert values["masksum"] == [8 if with_mask else 12]
        assert values["max_psf_sidelobe"] == 0.2
    controller = CycleController(max_iter=10, max_iter_per_cycle=3)
    limits, thresholds = get_per_cycle_limits(controller, stats)
    assert limits.shape == (2, 1, 2)
    np.testing.assert_array_equal(limits, 3)
    np.testing.assert_allclose(thresholds[:, 0], 0.2 * peaks)
    xr.testing.assert_identical(image, original)


def test_continuum_statistics_record_taylor_zero_entropy_only_when_enabled():
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        build_continuum_residual_imaging_dict,
    )

    rng = np.random.default_rng(20261013)
    residual = rng.standard_normal((1, 2, 2, 48, 40))
    residual[0, 1] = 1000.0  # Higher Taylor terms must not affect entropy.
    residual[0, 0, 1, 12, 15] = 50.0
    image = xr.Dataset(
        {
            "RESIDUAL": (("time", "taylor_term", "polarization", "l", "m"), residual),
            "SIDELOBE": (("time", "polarization"), [[0.2, 0.2]]),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "RESIDUAL",
                    "max_sidelobe_point_spread_function": "SIDELOBE",
                }
            }
        },
    )

    disabled = build_continuum_residual_imaging_dict(image, "residual", {"gain": 0.1})
    for fields in disabled.data.values():
        assert "entropy" not in fields
        assert "residual_snr" not in fields

    enabled = build_continuum_residual_imaging_dict(
        image,
        "residual",
        {
            "gain": 0.1,
            "entropy_stop": True,
            "entropy_max_snr": 6.0,
            "entropy_spatial_bins": 5,
            "entropy_flux_bins": 8,
        },
    )
    quiet = enabled.data[Key(time=0, chan=0, pol=0)]
    bright = enabled.data[Key(time=0, chan=0, pol=1)]
    assert np.isfinite(quiet["entropy"][0])
    assert quiet["residual_snr"][0] < 6.0
    assert np.isnan(bright["entropy"][0])
    assert bright["residual_snr"][0] > 6.0


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_refreshed_statistics_respect_clean_support(explicit_mask):
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        _continuum_clean_support,
        build_continuum_residual_imaging_dict,
    )

    image = xr.Dataset(
        {
            "R": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.array([[[[[0.05, 10.0], [np.nan, 0.02]]]]]),
            ),
            "PB": (("l", "m"), [[1.0, 0.1], [1.0, 1.0]]),
            "S": (("time", "polarization"), [[0.2]]),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "R",
                    "primary_beam": "PB",
                    "max_sidelobe_point_spread_function": "S",
                }
            }
        },
    )
    if explicit_mask:
        image["MASK"] = (("l", "m"), [[True, False], [True, True]])
        image.attrs["data_groups"]["residual"]["mask"] = "MASK"
    support = _continuum_clean_support(
        image,
        "residual",
        image.R.isel(taylor_term=0, drop=True),
        {"primary_beam_limit": 0.2},
    )
    assert support.flags.c_contiguous
    stats = build_continuum_residual_imaging_dict(
        image, "residual", {"primary_beam_limit": 0.2}
    )
    fields = next(iter(stats.data.values()))
    assert fields["peakres"] == [0.05]
    assert fields["peakres_nomask"] == [10.0]
    assert fields["masksum"] == [2]
    assert fields["iter_done"] == [0]


@pytest.mark.parametrize("clipped_pb", [0.1, 0.2, np.nan, np.inf])
def test_mvc_clips_channel_beams_in_casa_taylor_products(clipped_pb):
    """A clipped channel contributes weight but no residual, PSF, or PB signal."""
    frequency = np.array([0.9e9, 1.1e9])
    residual, psf, pb = _convert_mvc_cubes_to_taylor_normal_equations(
        _frequency_cube([5.0, 100.0], frequency),
        _frequency_cube([2.0, 10.0], frequency),
        _frequency_cube([1.0, clipped_pb], frequency),
        _frequency_normalization([1.0, 3.0], frequency),
        _frequency_normalization([1.0, 3.0], frequency),
        n_taylor_terms=1,
        reference_frequency=1e9,
        primary_beam_limit=0.2,
    )
    np.testing.assert_allclose(pb.squeeze(), 0.25)
    np.testing.assert_allclose(psf.squeeze(), 2.0)
    np.testing.assert_allclose(residual.squeeze(), 1.25)
