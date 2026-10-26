"""Unit tests for continuum map, reduce-root, and append node tasks."""

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import astroviper.node_tasks.imaging.image_continuum_single_field as continuum_node
import astroviper.processing_functions.imaging.image_continuum_single_field as continuum_processing
from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    _create_mfs_visibility_grid_cache_store,
)
from astroviper.processing_functions.imaging.utils import ImagingDict
from tests.utils.continuum_images import make_runaway_taylor_image


def _model_dataset(value):
    return xr.Dataset(
        {
            "SKY_MODEL": xr.DataArray(
                np.full((1, 2, 1, 2, 2), value, dtype=np.float64),
                dims=("time", "taylor_term", "polarization", "l", "m"),
            )
        },
        attrs={"data_groups": {"model": {"sky": "SKY_MODEL"}}},
    )


def _mfs_uv_grid(value):
    """Build a registered MFS Taylor grid for append-cache tests."""
    return xr.Dataset(
        {
            "VISIBILITY": xr.DataArray(
                np.full((1, 2, 1, 2, 2), value, dtype=np.complex128),
                dims=("time", "taylor_term", "polarization", "u", "v"),
            ),
            "VISIBILITY_NORMALIZATION": xr.DataArray(
                np.full((1, 2, 1), 3.0),
                dims=("time", "taylor_term", "polarization"),
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


def test_install_continuum_clean_mask_broadcasts_and_registers_data_group():
    """A 2-D user mask is broadcast across all Taylor residual planes."""
    residual = xr.DataArray(
        np.zeros((1, 2, 1, 2, 3)),
        dims=("time", "taylor_term", "polarization", "l", "m"),
        coords={
            "time": [0.0],
            "taylor_term": [0, 1],
            "polarization": ["I"],
            "l": [0, 1],
            "m": [0, 1, 2],
        },
    )
    image = xr.Dataset(
        {"SKY_RESIDUAL": residual},
        attrs={"data_groups": {"residual": {"sky": "SKY_RESIDUAL"}}},
    )
    mask = np.array([[True, False, True], [False, True, False]])

    continuum_node._install_continuum_clean_mask(image, mask)

    assert image["MASK"].dims == residual.dims
    np.testing.assert_array_equal(
        image["MASK"].isel(time=0, taylor_term=0, polarization=0), mask
    )
    np.testing.assert_array_equal(
        image["MASK"].isel(time=0, taylor_term=1, polarization=0), mask
    )
    assert image.attrs["data_groups"]["residual"]["mask"] == "MASK"
    assert (
        "User-supplied continuum deconvolution mask installed"
        in (image.attrs["data_groups"]["residual"]["description"])
    )


def test_install_continuum_clean_mask_rejects_wrong_shape():
    """Node-level validation protects direct callers from mismatched masks."""
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": xr.DataArray(
                np.zeros((1, 1, 1, 2, 3)),
                dims=("time", "taylor_term", "polarization", "l", "m"),
            )
        },
        attrs={"data_groups": {"residual": {"sky": "SKY_RESIDUAL"}}},
    )

    with pytest.raises(ValueError, match="expected"):
        continuum_node._install_continuum_clean_mask(
            image,
            np.ones((3, 2), dtype=bool),
        )


def test_first_cached_mfs_append_captures_an_independent_observed_grid():
    """The first reduced GWVobs grid is copied before its inverse FFT."""
    reduced = _mfs_uv_grid(5.0)

    cached = continuum_node._prepare_cached_mfs_residual_grid(
        {"image": reduced},
        {
            "continuum_mode": "mfs",
            "visibility_memory_mode": "in_memory",
            "is_first_cycle": True,
        },
    )

    reduced.VISIBILITY.data[...] = 0.0
    np.testing.assert_array_equal(cached.VISIBILITY, 5.0)
    assert cached.attrs["visibility_grid_source"] == "observed_data"


def test_later_cached_mfs_append_forms_residual_from_reduced_model_grid():
    """A later append replaces GWDmodel with cached GWVobs-GWDmodel."""
    input_data = {"image": _mfs_uv_grid(1.5)}

    returned_cache = continuum_node._prepare_cached_mfs_residual_grid(
        input_data,
        {
            "continuum_mode": "mfs",
            "visibility_memory_mode": "in_memory",
            "is_first_cycle": False,
            "observed_visibility_grid_xds": _mfs_uv_grid(5.0),
        },
    )

    assert returned_cache is None
    np.testing.assert_array_equal(input_data["image"].VISIBILITY, 3.5)
    np.testing.assert_array_equal(
        input_data["image"].VISIBILITY_NORMALIZATION,
        3.0,
    )


def test_in_place_mfs_append_persists_and_reloads_observed_grid(tmp_path):
    """The disk-backed cache survives the first append and forms later residuals."""
    image_store = tmp_path / "continuum.img.zarr"
    xr.Dataset().to_zarr(image_store, mode="w")
    reduced = _mfs_uv_grid(5.0)
    image_geometry = xr.Dataset(
        coords={
            "time": [0.0],
            "polarization": ["XX"],
            "l": [0.0, 1.0],
            "m": [0.0, 1.0],
        }
    )
    _create_mfs_visibility_grid_cache_store(
        str(image_store),
        image_geometry,
        {"fft_padding": 1.0, "n_taylor_terms": 2},
        ["XX"],
        False,
        None,
    )

    returned_cache = continuum_node._prepare_cached_mfs_residual_grid(
        {"image": reduced},
        {
            "continuum_mode": "mfs",
            "visibility_memory_mode": "in_place",
            "is_first_cycle": True,
            "image_store": str(image_store),
        },
    )

    assert returned_cache is None
    persisted = continuum_node._load_mfs_visibility_grid_in_place(str(image_store))
    np.testing.assert_array_equal(persisted.VISIBILITY, 5.0)
    assert persisted.attrs["visibility_grid_source"] == "observed_data"

    later_input = {"image": _mfs_uv_grid(1.5)}
    continuum_node._prepare_cached_mfs_residual_grid(
        later_input,
        {
            "continuum_mode": "mfs",
            "visibility_memory_mode": "in_place",
            "is_first_cycle": False,
            "image_store": str(image_store),
        },
    )

    np.testing.assert_array_equal(later_input["image"].VISIBILITY, 3.5)
    np.testing.assert_array_equal(
        later_input["image"].VISIBILITY_NORMALIZATION,
        3.0,
    )


def test_in_place_mfs_writer_requires_driver_created_cache(tmp_path):
    """The append node never creates its own temporary cache structure."""
    image_store = tmp_path / "continuum.img.zarr"
    xr.Dataset().to_zarr(image_store, mode="w")

    with pytest.raises(KeyError, match="not initialized"):
        continuum_node._write_mfs_visibility_grid_in_place(
            _mfs_uv_grid(5.0),
            str(image_store),
        )

    assert "_MFS_VISIBILITY_GRID_CACHE" not in zarr.open_group(image_store)


def test_attach_imaging_weights_uses_each_cached_dataset_variable_name():
    """A retained nonstandard weight name is reattached in later cycles."""
    dimensions = ("time", "baseline", "frequency", "polarization")
    shape = (1, 1, 2, 2)
    loaded = {
        "ms": xr.Dataset(
            {"WEIGHT": xr.DataArray(np.ones(shape), dims=dimensions)},
            attrs={"data_groups": {"base": {"weight": "WEIGHT"}}},
        )
    }
    cached = {
        "ms": xr.Dataset(
            {"RETAINED_WEIGHT": xr.DataArray(np.full(shape, 2.0), dims=dimensions)},
            attrs={"weight_imaging_name": "RETAINED_WEIGHT"},
        )
    }

    actual = continuum_node._attach_imaging_weights_continuum(
        loaded,
        cached,
        processing_set_data_group_name="base",
    )

    np.testing.assert_array_equal(actual["ms"].RETAINED_WEIGHT, 2.0)
    assert (
        actual["ms"].attrs["data_groups"]["base"]["weight_imaging"] == "RETAINED_WEIGHT"
    )


def test_recompute_mfs_append_does_not_require_or_modify_a_cache():
    """Recompute mode leaves the freshly reduced visibility residual untouched."""
    reduced = _mfs_uv_grid(2.5)

    returned_cache = continuum_node._prepare_cached_mfs_residual_grid(
        {"image": reduced},
        {
            "continuum_mode": "mfs",
            "visibility_memory_mode": "recompute",
            "is_first_cycle": False,
        },
    )

    assert returned_cache is None
    np.testing.assert_array_equal(reduced.VISIBILITY, 2.5)


def test_extract_mvc_observed_grid_removes_private_map_payload():
    """The MVC map transfers its owned cache buffers without copying them again."""
    frequency = [1.0e9, 1.1e9]
    image = xr.Dataset(
        {
            continuum_node._MVC_OBSERVED_VISIBILITY_CACHE: xr.DataArray(
                np.full((1, 2, 1, 2, 2), 5.0 + 0.0j),
                dims=("time", "frequency", "polarization", "u", "v"),
                coords={"frequency": frequency},
            ),
            continuum_node._MVC_OBSERVED_NORMALIZATION_CACHE: xr.DataArray(
                np.full((1, 2, 1), 4.0),
                dims=("time", "frequency", "polarization"),
                coords={"frequency": frequency},
            ),
            "MVC_RESIDUAL_TAYLOR_NUMERATOR": xr.DataArray(
                np.ones((1, 2, 1, 2, 2)),
                dims=("time", "taylor_term", "polarization", "l", "m"),
            ),
        },
        attrs={"data_groups": {}},
    )
    visibility_buffer = image[continuum_node._MVC_OBSERVED_VISIBILITY_CACHE].data
    normalization_buffer = image[continuum_node._MVC_OBSERVED_NORMALIZATION_CACHE].data

    reduced_image, observed = continuum_node._extract_mvc_observed_visibility_grid(
        image
    )

    assert continuum_node._MVC_OBSERVED_VISIBILITY_CACHE not in reduced_image
    assert continuum_node._MVC_OBSERVED_NORMALIZATION_CACHE not in reduced_image
    np.testing.assert_array_equal(observed.VISIBILITY, 5.0)
    np.testing.assert_array_equal(observed.VISIBILITY_NORMALIZATION, 4.0)
    assert observed.VISIBILITY.data is visibility_buffer
    assert observed.VISIBILITY_NORMALIZATION.data is normalization_buffer
    assert observed.attrs["data_groups"]["residual"] == {
        "visibility": "VISIBILITY",
        "visibility_normalization": "VISIBILITY_NORMALIZATION",
    }


def test_mfs_append_accumulates_before_preparing_fourier_model(monkeypatch):
    """The append FFT consumes the fully accumulated post-update MFS model."""
    previous = _model_dataset(2.0)
    increment = _model_dataset(0.5)
    captured = {}
    expected_uv = xr.Dataset({"VISIBILITY_MODEL": xr.DataArray([7.0])})

    def fake_prepare(model_xds, **kwargs):
        captured["model"] = model_xds.copy(deep=True)
        captured["kwargs"] = kwargs
        return expected_uv

    monkeypatch.setattr(
        continuum_processing,
        "prepare_model_uv_continuum_single_field",
        fake_prepare,
    )

    model_xds, model_uv_xds = continuum_node._prepare_post_update_continuum_model_state(
        increment,
        {
            "is_first_cycle": False,
            "model_xds": previous,
            "continuum_mode": "mfs",
            "image_params": {"n_taylor_terms": 2},
            "instrument_polarization_basis": "circular",
            "single_precision_image": False,
            "processing_function_threads": 3,
            "fft_backend": "scipy",
        },
    )

    np.testing.assert_array_equal(model_xds["SKY_MODEL"], 2.5)
    np.testing.assert_array_equal(captured["model"]["SKY_MODEL"], 2.5)
    assert model_uv_xds is expected_uv
    assert captured["kwargs"]["instrument_polarization_basis"] == "circular"
    assert captured["kwargs"]["single_precision_image"] is False
    assert captured["kwargs"]["processing_function_threads"] == 3
    assert captured["kwargs"]["fft_backend"] == "scipy"


def test_mvc_append_returns_accumulated_image_model_without_global_fft(monkeypatch):
    """MVC accumulation stays in the append node and does not invoke MFS FFT."""
    previous = _model_dataset(2.0)
    increment = _model_dataset(0.5)

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("MVC must not prepare a global Fourier model")

    monkeypatch.setattr(
        continuum_processing,
        "prepare_model_uv_continuum_single_field",
        unexpected_prepare,
    )

    model_xds, model_uv_xds = continuum_node._prepare_post_update_continuum_model_state(
        increment,
        {
            "is_first_cycle": False,
            "model_xds": previous,
            "continuum_mode": "mvc",
        },
    )

    np.testing.assert_array_equal(model_xds["SKY_MODEL"], 2.5)
    assert model_uv_xds is None


def _static_products():
    """Build static PSF, PB, beam-fit, and sidelobe products."""
    return xr.Dataset(
        {
            "POINT_SPREAD_FUNCTION": xr.DataArray(
                np.ones((1, 3, 1, 2, 2)),
                dims=("time", "psf_taylor_order", "polarization", "l", "m"),
            ),
            "PRIMARY_BEAM": xr.DataArray(
                np.ones((1, 1, 1, 2, 2)),
                dims=("time", "frequency", "polarization", "l", "m"),
            ),
            "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION": xr.DataArray(
                np.ones((1, 1, 3)),
                dims=("time", "polarization", "beam_params_label"),
            ),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": xr.DataArray(
                np.ones((1, 1)), dims=("time", "polarization")
            ),
        },
        coords={"frequency": [1.5e9]},
    )


@pytest.mark.parametrize("chunk_frequency", [[1.0e9], [1.0e9, 2.0e9]])
def test_install_static_products_replaces_unused_frequency_coordinate(
    chunk_frequency,
):
    """Reference frequency replaces a chunk coordinate no variable uses."""
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": xr.DataArray(
                np.zeros((1, 2, 1, 2, 2)),
                dims=("time", "taylor_term", "polarization", "l", "m"),
            )
        },
        coords={"frequency": chunk_frequency},
    )
    result = continuum_node._install_static_continuum_products(
        image, _static_products()
    )
    np.testing.assert_array_equal(result.frequency, [1.5e9])


def test_finalize_reference_primary_beam_preserves_reference_frequency():
    """MFS beam construction retains its explicit reference-frequency label."""
    image = xr.Dataset(
        {
            "PRIMARY_BEAM_REFERENCE": xr.DataArray(
                np.ones((1, 1, 2, 2)),
                dims=("time", "polarization", "l", "m"),
            )
        },
        coords={"frequency": [1.0e9]},
    )

    result = continuum_node._finalize_reference_primary_beam(
        image,
        reference_frequency=1.5e9,
    )

    np.testing.assert_array_equal(result.PRIMARY_BEAM.frequency, [1.5e9])
    assert result.PRIMARY_BEAM.attrs["effective_frequency_hz"] == 1.5e9


def test_install_static_products_does_not_alias_the_cache():
    """Per-cycle polarization conversion must not mutate cached static arrays."""
    static = _static_products()
    result = continuum_node._install_static_continuum_products(
        xr.Dataset(),
        static,
    )

    result["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"].data[...] = 0.0

    np.testing.assert_array_equal(
        static["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"],
        1.0,
    )


def test_install_static_products_rejects_frequency_still_in_use():
    """A live channel cube prevents replacement of its frequency coordinate."""
    image = xr.Dataset(
        {"CHANNEL_DATA": (("frequency", "l", "m"), np.zeros((2, 2, 2)))},
        coords={"frequency": [1.0e9, 2.0e9]},
    )
    with pytest.raises(ValueError, match="variables still use it"):
        continuum_node._install_static_continuum_products(image, _static_products())


def test_minor_append_normalizes_single_leaf_cache_state(monkeypatch):
    """A one-leaf result makes every task-local cache into a task mapping."""
    pb_xds = xr.Dataset({"PRIMARY_BEAM": xr.DataArray([1.0], dims=("frequency",))})
    weights = {"ms_0": xr.Dataset({"WEIGHT_IMAGING": xr.DataArray([1.0])})}
    observed_grid = _mfs_uv_grid(5.0)

    def fake_prepare(image, input_params, **kwargs):
        return image, xr.Dataset(), None

    monkeypatch.setattr(continuum_node, "_prepare_continuum_image", fake_prepare)
    monkeypatch.setattr(
        continuum_node,
        "model_update_continuum_single_field",
        lambda input_data, input_params: {"image": input_data["image"]},
    )
    monkeypatch.setattr(
        continuum_node,
        "_prepare_post_update_continuum_model_state",
        lambda image, input_params: (xr.Dataset(), None),
    )
    result = continuum_node.continuum_cycle_append_node(
        {
            "image": xr.Dataset(),
            "task_id": 7,
            "pb_xds": pb_xds,
            "weight_datasets": weights,
            "observed_visibility_grid_xds": observed_grid,
        },
        {"is_first_cycle": True},
    )
    assert result["pb_cache_mapping"] == {7: pb_xds}
    assert result["weight_cache_mapping"] == {7: weights}
    assert result["observed_visibility_grid_mapping"] == {7: observed_grid}


def test_minor_append_preserves_history_when_reduce_result_is_empty(monkeypatch):
    """An empty reduce placeholder must not hide prior deconvolution history."""
    reduced_history = ImagingDict()
    previous_history = ImagingDict()
    previous_history.add(
        {"peakres": 2.5, "max_psf_sidelobe": 0.35},
        time=0,
        pol=0,
        chan=0,
    )
    captured = {}

    monkeypatch.setattr(
        continuum_node,
        "_prepare_continuum_image",
        lambda image, input_params, **kwargs: (image, xr.Dataset(), None),
    )

    def fake_model_update(input_data, input_params):
        captured["history"] = input_data["deconvolution"]
        return {"image": input_data["image"]}

    monkeypatch.setattr(
        continuum_node,
        "model_update_continuum_single_field",
        fake_model_update,
    )
    monkeypatch.setattr(
        continuum_node,
        "_prepare_post_update_continuum_model_state",
        lambda image, input_params: (xr.Dataset(), None),
    )

    continuum_node.continuum_cycle_append_node(
        {"image": xr.Dataset(), "deconvolution": reduced_history},
        {"is_first_cycle": False, "deconvolution": previous_history},
    )

    assert captured["history"] is previous_history


def _empty_weight_image(**kwargs):
    """Return the image geometry expected by weighting node tests."""
    frequency = kwargs["frequency_coords"]
    return xr.Dataset(
        coords={
            "time": kwargs["time_coords"],
            "frequency": frequency,
            "polarization": kwargs["pol_coords"],
            "l": np.arange(2),
            "m": np.arange(2),
        }
    )


def _weight_node_params():
    """Return compact common parameters for weighting nodes."""
    return {
        "image_params": {
            "phase_direction": [0.0, 0.0],
            "image_size": [2, 2],
            "cell_size": [1.0, 1.0],
            "time_coords": [0.0],
            "polarization_coords": ["I", "Q"],
        },
        "imaging_weights_params": {"weighting": "briggs", "robust": 0.5},
        "task_coords": {"frequency": {"data": np.array([1.0e9, 1.1e9])}},
        "data_selection": {},
        "input_data_store": "unused",
        "processing_set_data_group_name": "base",
        "input_data": {"ms": xr.Dataset()},
        "task_id": 4,
    }


def test_first_cycle_does_not_rewrite_loaded_global_weights(monkeypatch):
    """A global in-place weight cache is read once and never written back."""
    monkeypatch.setattr("xradio.image.make_empty_sky_image", _empty_weight_image)
    weight = xr.DataArray(
        np.ones((1, 1, 2, 2)),
        dims=("time", "baseline", "frequency", "polarization"),
    )
    processing_set = {
        "ms": xr.Dataset(
            {"WEIGHT_IMAGING": weight},
            attrs={"data_groups": {"base": {"weight_imaging": "WEIGHT_IMAGING"}}},
        )
    }
    calls = {"load": 0, "write": 0}
    processing_kwargs = {}

    def fake_load_weights(ps_xdt, *args, **kwargs):
        calls["load"] += 1
        return ps_xdt

    monkeypatch.setattr(
        continuum_node, "_load_continuum_weights_in_place", fake_load_weights
    )
    monkeypatch.setattr(
        continuum_node,
        "_write_continuum_weights_in_place",
        lambda *args, **kwargs: calls.__setitem__("write", calls["write"] + 1),
    )

    def fake_residual_update(ps_xdt, img_xds, *args, **kwargs):
        processing_kwargs.update(kwargs)
        return img_xds, pd.DataFrame({"imaging_weights_calculated": [0.0]})

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.residual_update_continuum_single_field",
        fake_residual_update,
    )
    params = _weight_node_params()
    params.update(
        image_store="image.zarr",
        weight_memory_mode="in_place",
        weight_cache_is_active=True,
        continuum_mode="mfs",
        mvc_primary_beam_approach="preconditioned",
        data_group={"correlated_data": "VISIBILITY"},
        input_data=processing_set,
    )

    result = continuum_node.residual_update_continuum_single_field(**params)

    assert calls == {"load": 1, "write": 0}
    assert processing_kwargs["mvc_primary_beam_approach"] == "preconditioned"
    assert "weight_datasets" not in result


def test_weight_density_node_returns_valid_reducer_leaf(monkeypatch):
    """The first weighting node packages density products and timings."""
    monkeypatch.setattr("xradio.image.make_empty_sky_image", _empty_weight_image)
    load_kwargs = {}

    def fake_load(*args, **kwargs):
        load_kwargs.update(kwargs)
        return {"ms": xr.Dataset()}

    load_module = importlib.import_module("xradio.measurement_set.load_processing_set")
    monkeypatch.setattr(load_module, "load_processing_set", fake_load)

    def fake_grid(ps_xdt, image, params, **kwargs):
        return xr.Dataset(
            {
                "WEIGHT_DENSITY_GRID": (
                    ("frequency", "weight_polarization", "u", "v"),
                    np.ones((1, 1, 2, 2)),
                ),
                "SUM_WEIGHT": (
                    ("frequency", "weight_polarization"),
                    np.ones((1, 1)),
                ),
            },
            coords={"frequency": [1.05e9]},
        )

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.calculate_imaging_weights."
        "grid_imaging_weight_density_continuum",
        fake_grid,
    )
    params = _weight_node_params()
    params.update(
        input_data=None,
        data_group={"correlated_data": "VISIBILITY"},
    )
    result = continuum_node.grid_imaging_weight_density_continuum_node(**params)
    assert result["task_id"] == 4
    assert load_kwargs["drop_variables"] == ["VISIBILITY"]
    assert set(result["weight_density"]) == {"WEIGHT_DENSITY_GRID", "SUM_WEIGHT"}
    assert result["timing_node_tasks"].iloc[0].n_frequency_channels == 1
    assert {
        "start_unixtime",
        "hostname",
        "process_pid",
        "thread_native_id",
        "worker_name",
    } <= set(result["timing_node_tasks"])


def test_weight_degrid_node_extracts_only_registered_weight_arrays(monkeypatch):
    """The second weighting node returns lightweight per-child weight caches."""
    monkeypatch.setattr("xradio.image.make_empty_sky_image", _empty_weight_image)
    load_kwargs = {}

    def fake_load(*args, **kwargs):
        load_kwargs.update(kwargs)
        return {"ms": xr.Dataset()}

    load_module = importlib.import_module("xradio.measurement_set.load_processing_set")
    monkeypatch.setattr(load_module, "load_processing_set", fake_load)
    weight = xr.DataArray(
        np.ones((1, 1, 2, 2)),
        dims=("time", "baseline", "frequency", "polarization"),
        attrs={"units": "arbitrary"},
    )
    child = xr.Dataset(
        {"WEIGHT_IMAGING": weight},
        attrs={"data_groups": {"base": {"weight_imaging": "WEIGHT_IMAGING"}}},
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.calculate_imaging_weights."
        "degrid_imaging_weights_continuum",
        lambda *args, **kwargs: {"ms": child},
    )
    params = _weight_node_params()
    params["global_weighting_xds"] = xr.Dataset()
    params["weight_memory_mode"] = "in_memory"
    params["image_store"] = "unused-image"
    params["input_data"] = None
    params["data_group"] = {"correlated_data": "VISIBILITY"}
    result = continuum_node.degrid_imaging_weights_continuum_node(**params)
    cached = result["weight_datasets"]["ms"]
    assert load_kwargs["drop_variables"] == ["VISIBILITY"]
    assert set(cached) == {"WEIGHT_IMAGING"}
    assert cached.WEIGHT_IMAGING.attrs["units"] == "arbitrary"
    assert result["timing_node_tasks"].iloc[0].n_processing_set_children == 1
    assert {
        "start_unixtime",
        "hostname",
        "process_pid",
        "thread_native_id",
        "worker_name",
    } <= set(result["timing_node_tasks"])


def test_weight_degrid_node_writes_in_place_and_returns_no_array_cache(monkeypatch):
    """In-place global weighting writes locally and keeps the graph result small."""
    monkeypatch.setattr("xradio.image.make_empty_sky_image", _empty_weight_image)
    child = xr.Dataset(
        {
            "WEIGHT_IMAGING": xr.DataArray(
                np.ones((1, 1, 2, 2)),
                dims=("time", "baseline", "frequency", "polarization"),
            )
        },
        attrs={"data_groups": {"base": {"weight_imaging": "WEIGHT_IMAGING"}}},
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.calculate_imaging_weights."
        "degrid_imaging_weights_continuum",
        lambda *args, **kwargs: {"ms": child},
    )
    written = {}
    monkeypatch.setattr(
        continuum_node,
        "_write_continuum_weights_in_place",
        lambda ps_xdt, ps_store: written.update(processing_set=ps_xdt, store=ps_store),
    )
    params = _weight_node_params()
    params.update(global_weighting_xds=xr.Dataset(), image_store="image.zarr")

    result = continuum_node.degrid_imaging_weights_continuum_node(**params)

    assert result["weight_datasets"] == {}
    assert written == {"processing_set": {"ms": child}, "store": "image.zarr"}
    assert result["timing_node_tasks"].iloc[0].n_processing_set_children == 1


def test_stored_coordinate_indexer_supports_irregular_subsets():
    """An irregular selected coordinate maps to exact on-disk positions."""
    actual = continuum_node._stored_coordinate_indexer(
        np.array([100.0, 101.0, 102.0, 103.0]),
        np.array([103.0, 100.0, 102.0]),
        "frequency",
    )
    np.testing.assert_array_equal(actual, [3, 0, 2])


def test_stored_coordinate_indexer_rejects_missing_values():
    """An in-place write fails rather than placing an unmatched coordinate."""
    with pytest.raises(ValueError, match="absent from the in-place store"):
        continuum_node._stored_coordinate_indexer(
            np.array([100.0, 101.0]),
            np.array([102.0]),
            "frequency",
        )


@pytest.mark.parametrize("initial", [True, False])
def test_model_update_passes_cube_parameter_names_to_backend(monkeypatch, initial):
    """Node and Taylor adapter preserve the controller's per-plane controls."""
    import astroviper.processing_functions.imaging.model_update as cube_update
    import astroviper.processing_functions.imaging.utils as imaging_utils

    dims = ("time", "taylor_term", "polarization", "l", "m")
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (dims, np.ones((1, 2, 1, 3, 3))),
            "SKY_MODEL": (dims, np.zeros((1, 2, 1, 3, 3))),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                np.ones((1, 3, 1, 3, 3)),
            ),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "polarization"),
                np.zeros((1, 1)),
            ),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                },
                "model": {"sky": "SKY_MODEL"},
            }
        },
    )
    params = {
        "max_iter": 17,
        "max_cycles": 3,
        "max_iter_per_cycle": 5,
        "threshold": 0.01,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    controller = imaging_utils.CycleController(**params)
    iterations = np.array([[[4]]], dtype=np.int64)
    thresholds = np.array([[[0.125]]])

    def controls(controller_arg, residual_imaging_dict):
        assert controller_arg is controller
        # controls use the refreshed residual
        assert residual_imaging_dict is not None
        return iterations, thresholds

    class ReachedBackend(Exception):
        pass

    def backend(image_arg, deconvolver, deconvolve_params, model_exists, **kwargs):
        assert model_exists is (not initial)
        assert deconvolve_params["max_iter"] == 17
        assert deconvolve_params["max_cycles"] == 3
        np.testing.assert_array_equal(
            deconvolve_params["max_iter_per_cycle_updated"], iterations
        )
        np.testing.assert_array_equal(
            deconvolve_params["threshold_per_cycle"], thresholds
        )
        assert not {
            "niter",
            "cycleniter",
            "cyclethreshold",
            "niter_per_plane",
            "cyclethreshold_per_plane",
        }.intersection(deconvolve_params)
        assert image_arg.sizes["frequency"] == 1
        raise ReachedBackend

    monkeypatch.setattr(imaging_utils, "get_per_cycle_limits", controls)
    monkeypatch.setattr(cube_update, "model_update_cube_single_field", backend)
    with pytest.raises(ReachedBackend):
        continuum_node.model_update_continuum_single_field(
            {"image": image},
            {
                "cycle_control_params": params,
                "controller": controller,
                "is_first_cycle": initial,
            },
        )
    assert params["max_iter_per_cycle"] == 5


@pytest.mark.parametrize("deconvolver", ["hogbom", "mtmfs"])
def test_continuum_node_updates_controller_and_cumulative_flux(
    monkeypatch, deconvolver
):
    """Both backends report cumulative flux from the previous Taylor-0 model."""
    from astroviper.processing_functions.imaging.utils import (
        CycleController,
        StopCode,
    )
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_CONTINUE,
        MODEL_UPDATE_CONTINUE,
        MODEL_UPDATE_MAX_ITER_PER_CYCLE,
    )

    dims = ("time", "taylor_term", "polarization", "l", "m")
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (dims, np.ones((1, 2, 2, 4, 4))),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                np.ones((1, 3, 2, 4, 4)),
            ),
            "PRIMARY_BEAM": (("l", "m"), np.ones((4, 4))),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "polarization"),
                [[0.1, 0.2]],
            ),
        },
        coords={
            "time": [0.0],
            "taylor_term": [0, 1],
            "psf_taylor_order": [0, 1, 2],
            "polarization": ["I", "Q"],
            "l": np.arange(4),
            "m": np.arange(4),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                    "primary_beam": "PRIMARY_BEAM",
                }
            }
        },
    )
    params = {
        "max_iter": 10,
        "max_cycles": 3,
        "threshold": 0.0,
        "gain": 0.1,
        "max_iter_per_cycle": 2,
        "psf_sidelobe_factor": 1.0,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "primary_beam_limit": 0.2,
    }
    controller = CycleController(
        **{key: value for key, value in params.items() if key != "primary_beam_limit"}
    )
    previous_model = xr.Dataset({"SKY_MODEL": xr.zeros_like(image.SKY_RESIDUAL)})
    previous_model.SKY_MODEL.loc[{"taylor_term": 0, "polarization": "I"}] = 2.0
    previous_model.SKY_MODEL.loc[{"taylor_term": 0, "polarization": "Q"}] = 3.0

    def native_backend(image_arg, selected_deconvolver, controls, **kwargs):
        assert selected_deconvolver == deconvolver
        assert controls["max_iter_per_cycle_updated"].shape == (1, 1, 2)
        assert controls["threshold_per_cycle"].shape == (1, 1, 2)
        image_arg["SKY_MODEL"] = xr.zeros_like(image_arg.SKY_RESIDUAL)
        image_arg.SKY_MODEL.loc[{"taylor_term": 0}] = 1.0
        image_arg.SKY_MODEL.loc[{"taylor_term": 1}] = -0.25
        result = ImagingDict()
        for pol, iterations in enumerate((1, 2)):
            result.add(
                {
                    "iter_done": iterations,
                    "peakres": 0.5,
                    "masksum": 16,
                    "start_model_flux": 0.0,
                    "model_flux": float(16 * (pol + 1)),
                    "max_psf_sidelobe": 0.1 * (pol + 1),
                    "stop_code": StopCode(
                        imaging=IMAGING_CONTINUE,
                        model_update=MODEL_UPDATE_MAX_ITER_PER_CYCLE,
                    ),
                    "stop_description": "Reached max_iter_per_cycle",
                    "scales": np.asarray([0.0]),
                    "hessian_condition_number": np.asarray([2.0 + pol]),
                },
                time=0,
                pol=pol,
                chan=0,
            )
        return result, pd.DataFrame({"T_mtmfs": [0.25]})

    monkeypatch.setattr(
        continuum_processing, "model_update_mtmfs_single_field", native_backend
    )
    result = continuum_node.model_update_continuum_single_field(
        {"image": image},
        {
            "controller": controller,
            "deconvolver": deconvolver,
            "cycle_control_params": params,
            "is_first_cycle": False,
            "model_xds": previous_model,
        },
    )

    assert not result["residual_converged"]
    # The model update is booked, not tested: the stop code is that of the
    # check on the refreshed residual, the model update codes are per plane.
    assert result["stopcode"] == StopCode(
        imaging=IMAGING_CONTINUE,
        model_update=MODEL_UPDATE_CONTINUE,
    )
    assert result["stopdesc"] == "Continue imaging cycles"
    assert controller.total_iter_done == 3
    assert controller.cycles_done == 1
    assert controller.max_cycles == 2
    np.testing.assert_array_equal(controller.max_iter_remaining[0, 0], [9, 8])
    np.testing.assert_array_equal(
        controller.stop_code_model_update[0, 0],
        [MODEL_UPDATE_MAX_ITER_PER_CYCLE, MODEL_UPDATE_MAX_ITER_PER_CYCLE],
    )
    assert result["timing_model_update"].loc[0, "T_mtmfs"] == pytest.approx(0.25)
    assert len(result["deconvolution"].data) == 2
    for pol, fields in enumerate(result["deconvolution"].data.values()):
        assert fields["stop_code"].model_update == MODEL_UPDATE_MAX_ITER_PER_CYCLE
        assert np.isfinite(fields["hessian_condition_number"]).all()
        previous_flux = 16.0 * (pol + 2)
        assert fields["start_model_flux"] == [previous_flux]
        assert fields["model_flux"] == [previous_flux + 16.0 * (pol + 1)]


@pytest.mark.parametrize(
    "max_iter_divergence,iterations,diverged_model_updates",
    [(1, 7, 1), (-1, 50, 0)],
)
def test_native_mtmfs_node_books_a_diverged_model_update(
    max_iter_divergence, iterations, diverged_model_updates
):
    """A plane the MT-MFS divergence test stops is booked as diverged."""
    from astroviper.processing_functions.imaging.utils import (
        CycleController,
        StopCode,
    )
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_CONTINUE,
        MODEL_UPDATE_CONTINUE,
        MODEL_UPDATE_DIVERGENCE,
        MODEL_UPDATE_MAX_ITER_PER_CYCLE,
    )

    params = {
        "max_iter": 100,
        "max_cycles": 3,
        "threshold": 0.0,
        "gain": 0.1,
        "max_iter_per_cycle": 50,
        "psf_sidelobe_factor": 1.0,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    controller = CycleController(**params)
    result = continuum_node.model_update_continuum_single_field(
        {"image": make_runaway_taylor_image()},
        {
            "controller": controller,
            "deconvolver": "mtmfs",
            "cycle_control_params": {
                **params,
                "max_iter_divergence": max_iter_divergence,
            },
        },
    )

    stop_code = (
        MODEL_UPDATE_DIVERGENCE
        if diverged_model_updates
        else MODEL_UPDATE_MAX_ITER_PER_CYCLE
    )
    (fields,) = result["deconvolution"].data.values()
    assert fields["iter_done"] == [iterations]
    assert fields["stop_code"].model_update == stop_code
    assert result["stopcode"] == StopCode(
        imaging=IMAGING_CONTINUE,
        model_update=MODEL_UPDATE_CONTINUE,
    )
    assert not result["residual_converged"]
    assert controller.total_iter_done == iterations
    assert controller.stop_code_model_update[0, 0, 0] == stop_code
    assert controller.diverged_model_updates[0, 0, 0] == diverged_model_updates


@pytest.mark.parametrize("primary_beam_limit", [None, 0.2])
def test_native_mtmfs_imaging_cycles_clean_what_the_stop_test_measures(
    primary_beam_limit,
):
    """A source where the primary beam is 0.15 is cleaned by default, and the
    imaging cycles end on the threshold; with primary_beam_limit 0.2 neither
    the stop test nor the model update sees it.

    Real node task, adapter and kernel. The residual update is made in the
    image domain, residual_k = dirty_k - sum_j psf_(k+j) * model_j (linear
    convolutions), so a model update without iterations leaves the residual
    as it was. max_cycles is -1; the test stops after 30 imaging cycles,
    where the distributed application would go on.
    """
    from scipy.signal import fftconvolve

    from astroviper.processing_functions.imaging.utils import CycleController
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_THRESHOLD,
    )

    n, centre, outer = 64, 32, (32, 57)

    def gaussian(size, middle, fwhm):
        yy, xx = np.indices((size, size))
        r2 = (yy - middle) ** 2 + (xx - middle) ** 2
        return np.exp(-4 * np.log(2) * r2 / fwhm**2)

    primary_beam = gaussian(n, centre, 30.0)
    psf_orders = (1.0, 0.1, 0.5)
    psf_big = gaussian(2 * n, n, 3.0)

    def convolve(image, order):
        full = fftconvolve(image, psf_orders[order] * psf_big, mode="full")
        return full[n : 2 * n, n : 2 * n]

    sky = np.zeros((n, n))
    sky[centre, centre] = 1.0
    sky[outer] = 0.5  # apparent flux, flat spectrum
    dirty = np.stack([convolve(sky, 0), convolve(sky, 1)])
    dims = ("time", "taylor_term", "polarization", "l", "m")
    psf = np.stack([order * gaussian(n, centre, 3.0) for order in psf_orders])

    def image(residual):
        return xr.Dataset(
            {
                "SKY_RESIDUAL": (dims, residual[None, :, None].copy()),
                "POINT_SPREAD_FUNCTION": (
                    ("time", "psf_taylor_order", "polarization", "l", "m"),
                    psf[None, :, None],
                ),
                "PRIMARY_BEAM": (("l", "m"), primary_beam),
                "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                    ("time", "polarization"),
                    [[0.2]],
                ),
            },
            coords={
                "time": [0.0],
                "taylor_term": [0, 1],
                "psf_taylor_order": [0, 1, 2],
                "polarization": ["I"],
                "l": np.arange(n),
                "m": np.arange(n),
            },
            attrs={
                "data_groups": {
                    "residual": {
                        "sky": "SKY_RESIDUAL",
                        "point_spread_function": "POINT_SPREAD_FUNCTION",
                        "primary_beam": "PRIMARY_BEAM",
                    }
                }
            },
        )

    params = {
        "max_iter": 1000,
        "max_cycles": -1,
        "threshold": 0.05,
        "gain": 0.1,
        "max_iter_per_cycle": -1,
        "psf_sidelobe_factor": 1.0,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    controller = CycleController(**params)
    cycle_control_params = {**params, "scales": [0.0]}
    if primary_beam_limit is not None:
        cycle_control_params["primary_beam_limit"] = primary_beam_limit
    model = np.zeros((2, n, n))
    residual = dirty
    deconvolution = None
    for _ in range(30):
        result = continuum_node.model_update_continuum_single_field(
            {"image": image(residual)},
            {
                "controller": controller,
                "deconvolver": "mtmfs",
                "cycle_control_params": cycle_control_params,
                "deconvolution": deconvolution,
            },
        )
        if result["residual_converged"]:
            break
        deconvolution = result["deconvolution"]
        model += result["image"].SKY_MODEL.transpose(*dims).values[0, :, 0]
        residual = dirty - np.stack(
            [sum(convolve(model[j], k + j) for j in range(2)) for k in range(2)]
        )
    else:
        pytest.fail("The imaging cycles did not end within 30.")

    assert result["stopcode"].imaging == IMAGING_THRESHOLD
    level = params["threshold"] * (1 + controller.threshold_tolerance)
    near_outer = (slice(outer[0] - 2, outer[0] + 3), slice(outer[1] - 2, outer[1] + 3))
    if primary_beam_limit is None:
        assert primary_beam[outer] == pytest.approx(0.146, abs=1e-3)
        assert np.abs(residual[0]).max() <= level
        assert model[0][near_outer].sum() == pytest.approx(0.5, abs=level)
    else:
        assert np.abs(residual[0][primary_beam >= 0.2]).max() <= level
        assert residual[0][outer] == pytest.approx(0.5, abs=0.01)
        np.testing.assert_array_equal(model[:, primary_beam < 0.2], 0.0)
    assert model[0][centre, centre] == pytest.approx(1.0, abs=level)


def test_later_mfs_cycle_predicts_with_all_accumulated_taylor_terms(monkeypatch):
    """Every Taylor increment reaches later-cycle MFS prediction unchanged."""
    previous = _model_dataset(0.0)
    previous.SKY_MODEL.loc[{"taylor_term": 0}] = 1.0
    previous.SKY_MODEL.loc[{"taylor_term": 1}] = 10.0
    increment = _model_dataset(0.0)
    increment.SKY_MODEL.loc[{"taylor_term": 0}] = 2.0
    increment.SKY_MODEL.loc[{"taylor_term": 1}] = 20.0
    predicted = xr.Dataset({"VISIBILITY_MODEL": ("marker", [1.0])})

    def prepare_model(model_xds, **kwargs):
        np.testing.assert_array_equal(model_xds.SKY_MODEL.isel(taylor_term=0), 3.0)
        np.testing.assert_array_equal(model_xds.SKY_MODEL.isel(taylor_term=1), 30.0)
        return predicted

    monkeypatch.setattr(
        continuum_processing,
        "prepare_model_uv_continuum_single_field",
        prepare_model,
    )
    accumulated, model_uv = continuum_node._prepare_post_update_continuum_model_state(
        increment,
        {
            "is_first_cycle": False,
            "model_xds": previous,
            "continuum_mode": "mfs",
            "image_params": {},
        },
    )

    np.testing.assert_array_equal(accumulated.SKY_MODEL.isel(taylor_term=0), 3.0)
    np.testing.assert_array_equal(accumulated.SKY_MODEL.isel(taylor_term=1), 30.0)
    assert model_uv is predicted


def test_zero_max_iter_uses_current_controller_fields():
    from astroviper.processing_functions.imaging.utils import CycleController

    image = _model_dataset(0.0).rename({"SKY_MODEL": "SKY_RESIDUAL"})
    image.attrs["data_groups"]["residual"] = {"sky": "SKY_RESIDUAL"}
    image["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"] = (("time", "polarization"), [[0.1]])
    controller = CycleController(max_iter=0)
    result = continuum_node.model_update_continuum_single_field(
        {"image": image},
        {"cycle_control_params": {"max_iter": 0}, "controller": controller},
    )
    assert result["stopcode"].imaging == 1
    assert result["stopcode"].model_update == 0
    np.testing.assert_array_equal(controller.max_iter_remaining, 0)


def test_refreshed_continuum_residual_can_stop_on_entropy(monkeypatch):
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_ENTROPY,
        CycleController,
    )

    rng = np.random.default_rng(20261013)
    residual = rng.standard_normal((1, 1, 1, 32, 32))
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                residual,
            ),
            "PRIMARY_BEAM": (("l", "m"), np.ones((32, 32))),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "polarization"),
                [[0.1]],
            ),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "primary_beam": "PRIMARY_BEAM",
                    "max_sidelobe_point_spread_function": (
                        "MAX_SIDELOBE_POINT_SPREAD_FUNCTION"
                    ),
                }
            }
        },
    )
    controller = CycleController(max_iter=100, entropy_stop=True)
    controller.ensure_planes(1, 1, 1)
    controller.max_entropy[0, 0, 0] = 1.0e6

    def unexpected_update(*args, **kwargs):
        pytest.fail("An entropy-stopped residual must not run a model update")

    monkeypatch.setattr(
        continuum_processing, "model_update_mtmfs_single_field", unexpected_update
    )
    result = continuum_node.model_update_continuum_single_field(
        {"image": image},
        {
            "controller": controller,
            "cycle_control_params": {
                "max_iter": 100,
                "gain": 0.1,
                "primary_beam_limit": 0.2,
                "entropy_stop": True,
            },
        },
    )

    assert result["residual_converged"]
    assert result["stopcode"].imaging == IMAGING_ENTROPY
    assert controller.entropy_stopped[0, 0, 0]
    fields = next(iter(result["residual_statistics"].data.values()))
    assert np.isfinite(fields["entropy"][0])
    assert np.isfinite(fields["residual_snr"][0])
    assert np.all(result["image"].SKY_MODEL.values == 0)


@pytest.mark.parametrize("legacy_cache", [False, True])
def test_continuum_transfer_preserves_arrays_metadata_and_nested_datasets(legacy_cache):
    import pickle

    image = _model_dataset(2.0)
    nested = _model_dataset(3.0)
    image.attrs["nested_dataset"] = nested
    original = image.copy(deep=True)
    array = image.SKY_MODEL.data
    _ = image.xr_img
    _ = nested.xr_img
    if legacy_cache:
        image._cache = {"xr_img": image.xr_img}
        nested._cache = {"xr_img": nested.xr_img}
    payload = {"image": image, "cache": [{0: (nested, image)}]}
    payload["alias"] = payload

    result = continuum_node._prepare_continuum_result_for_transfer(payload)

    assert result is payload
    assert image.SKY_MODEL.data is array
    assert not getattr(image, "_cache", None)
    assert not getattr(nested, "_cache", None)
    xr.testing.assert_equal(image, original)
    assert image.attrs["data_groups"] == original.attrs["data_groups"]
    assert image.attrs["nested_dataset"] is nested
    xr.testing.assert_identical(nested, original.attrs["nested_dataset"])
    restored = pickle.loads(pickle.dumps(result))
    xr.testing.assert_equal(restored["image"], original)
    assert restored["image"].attrs["data_groups"] == original.attrs["data_groups"]
    xr.testing.assert_identical(
        restored["image"].attrs["nested_dataset"], original.attrs["nested_dataset"]
    )
    assert restored["alias"] is restored
    assert restored["cache"][0][0][1] is restored["image"]
    assert restored["image"].xr_img._xds is restored["image"]


@pytest.mark.parametrize(
    ("peak", "max_iter", "max_cycles", "masked", "reason"),
    [
        (0.05, 100, 3, False, "threshold"),
        (1.0, 0, 3, False, "iterations"),
        (1.0, 100, 0, False, "cycles"),
        (1.0, 100, 3, True, "mask"),
    ],
)
def test_refreshed_stop_skips_model_update_and_preserves_counters(
    monkeypatch, peak, max_iter, max_cycles, masked, reason
):
    """Verified stopping never runs CLEAN or charges a verification as an update."""
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_MAX_CYCLES,
        IMAGING_MAX_ITER,
        IMAGING_THRESHOLD,
        IMAGING_ZERO_MASK,
        CycleController,
    )

    controller = CycleController(
        max_iter=max_iter, max_cycles=max_cycles, threshold=0.1
    )
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.full((1, 1, 1, 2, 2), peak),
            ),
            "PRIMARY_BEAM": (("l", "m"), np.ones((2, 2))),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (("time", "polarization"), [[0.1]]),
        },
        attrs={
            "data_groups": {
                "residual": {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
            }
        },
    )
    if masked:
        image["MASK"] = (("l", "m"), np.zeros((2, 2), dtype=bool))
        image.attrs["data_groups"]["residual"]["mask"] = "MASK"

    def unexpected_update(*args, **kwargs):
        pytest.fail("A confirmed residual stop must not run a model update")

    monkeypatch.setattr(
        continuum_processing, "model_update_mtmfs_single_field", unexpected_update
    )
    result = continuum_node.model_update_continuum_single_field(
        {"image": image},
        {
            "controller": controller,
            "cycle_control_params": {
                "max_iter": max_iter,
                "primary_beam_limit": 0.2,
            },
        },
    )
    expected = {
        "threshold": IMAGING_THRESHOLD,
        "iterations": IMAGING_MAX_ITER,
        "cycles": IMAGING_MAX_CYCLES,
        "mask": IMAGING_ZERO_MASK,
    }[reason]
    assert result["residual_converged"]
    assert result["stopcode"].imaging == expected
    assert controller.total_iter_done == controller.cycles_done == 0
    assert controller.max_cycles == max_cycles
    assert np.all(controller.max_iter_remaining == max_iter)
    assert not result["deconvolution"].data
    assert np.all(result["image"].SKY_MODEL.values == 0)


def test_model_update_constructs_controller_when_omitted():
    """A direct node call builds the documented continuum cycle controller."""
    from astroviper.processing_functions.imaging.utils import CycleController

    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.ones((1, 1, 1, 2, 2)),
            ),
            "PRIMARY_BEAM": (("l", "m"), np.ones((2, 2))),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "polarization"),
                [[0.1]],
            ),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "primary_beam": "PRIMARY_BEAM",
                }
            }
        },
    )
    params = {
        "max_iter": 0,
        "max_cycles": 2,
        "threshold": 0.0,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_iter_per_cycle": 1,
    }

    result = continuum_node.model_update_continuum_single_field(
        {"image": image},
        {"cycle_control_params": params},
    )

    assert isinstance(result["controller"], CycleController)
    assert result["controller"].max_cycles == 2
    assert result["residual_converged"]


def test_finalization_reuses_prepared_residual_without_another_transform(monkeypatch):
    image = _model_dataset(0.25).rename({"SKY_MODEL": "SKY_RESIDUAL"})
    image.attrs["data_groups"] = {"residual": {"sky": "SKY_RESIDUAL"}}
    model = _model_dataset(0.75)
    original_residual = image.SKY_RESIDUAL.data

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("A verified residual must not be transformed a second time")

    monkeypatch.setattr(continuum_node, "_prepare_continuum_image", unexpected_prepare)
    monkeypatch.setattr(
        continuum_node, "_prepare_cached_mfs_residual_grid", unexpected_prepare
    )
    result = continuum_node.continuum_finalize_node(
        {"image": image, "residual_converged": True},
        {
            "prepared_continuum_image": True,
            "static_xds": xr.Dataset(),
            "model_xds": model,
            "restore": False,
        },
    )
    assert result["image"].SKY_RESIDUAL.data is original_residual
    np.testing.assert_array_equal(result["image"].SKY_MODEL, model.SKY_MODEL)
    assert result["residual_converged"]


def test_finalization_passes_mtmfs_to_restoration(monkeypatch):
    """The final append node enables MT-MFS principal-solution restoration."""
    image = _model_dataset(0.25).rename({"SKY_MODEL": "SKY_RESIDUAL"})
    image.attrs["data_groups"] = {"residual": {"sky": "SKY_RESIDUAL"}}
    received = {}

    def fake_restore(image, **kwargs):
        received.update(kwargs)
        return image, pd.DataFrame({"T_restore": [0.0]})

    monkeypatch.setattr(continuum_processing, "restore_continuum_image", fake_restore)
    continuum_node.continuum_finalize_node(
        {"image": image, "residual_converged": True},
        {
            "prepared_continuum_image": True,
            "static_xds": xr.Dataset(),
            "model_xds": _model_dataset(0.75),
            "restore": True,
            "deconvolver": "mtmfs",
        },
    )

    assert received["deconvolver"] == "mtmfs"
