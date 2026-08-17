"""Unit tests for continuum map, reduce-root, and append node tasks."""

import numpy as np
import pytest
import xarray as xr

import astroviper.node_tasks.imaging.image_continuum_single_field as continuum_node
import astroviper.processing_functions.imaging.image_continuum_single_field as continuum_processing


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
            "is_n_iter_0": False,
            "model_xds": previous,
            "specmode": "mfs",
            "image_params": {"nterms": 2},
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
            "is_n_iter_0": False,
            "model_xds": previous,
            "specmode": "mvc",
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


def test_install_static_products_replaces_unused_frequency_coordinate():
    """Reference frequency replaces a chunk coordinate no variable uses."""
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": xr.DataArray(
                np.zeros((1, 2, 1, 2, 2)),
                dims=("time", "taylor_term", "polarization", "l", "m"),
            )
        },
        coords={"frequency": [1.0e9, 2.0e9]},
    )
    result = continuum_node._install_static_continuum_products(
        image, _static_products()
    )
    np.testing.assert_array_equal(result.frequency, [1.5e9])


def test_install_static_products_rejects_frequency_still_in_use():
    """A live channel cube prevents replacement of its frequency coordinate."""
    image = xr.Dataset(
        {"CHANNEL_DATA": (("frequency", "l", "m"), np.zeros((2, 2, 2)))},
        coords={"frequency": [1.0e9, 2.0e9]},
    )
    with pytest.raises(ValueError, match="variables still use it"):
        continuum_node._install_static_continuum_products(image, _static_products())


def test_minor_append_normalizes_single_leaf_cache_state(monkeypatch):
    """A one-leaf result becomes task-keyed PB and weight cache mappings."""
    pb_xds = xr.Dataset({"PRIMARY_BEAM": xr.DataArray([1.0], dims=("frequency",))})
    weights = {"ms_0": xr.Dataset({"WEIGHT_IMAGING": xr.DataArray([1.0])})}
    captured = {}

    def fake_prepare(image, input_params, **kwargs):
        captured.update(kwargs)
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
    result = continuum_node.continuum_minor_cycle_node(
        {
            "image": xr.Dataset(),
            "task_id": 7,
            "pb_xds": pb_xds,
            "weight_datasets": weights,
        },
        {"is_n_iter_0": True},
    )
    assert captured["pb_cache_mapping"] == {7: pb_xds}
    assert result["weight_cache_mapping"] == {7: weights}


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
        },
        "imaging_weights_params": {"weighting": "briggs", "robust": 0.5},
        "task_coords": {"frequency": {"data": np.array([1.0e9, 1.1e9])}},
        "data_selection": {},
        "input_data_store": "unused",
        "processing_set_data_group_name": "base",
        "input_data": {"ms": xr.Dataset()},
        "task_id": 4,
    }


def test_weight_density_node_returns_valid_reducer_leaf(monkeypatch):
    """The first weighting node packages density products and timings."""
    monkeypatch.setattr("xradio.image.make_empty_sky_image", _empty_weight_image)

    def fake_grid(ps_xdt, image, params, **kwargs):
        return xr.Dataset(
            {
                "WEIGHT_DENSITY_GRID": (
                    ("frequency", "weight_polarization", "u", "v"),
                    np.ones((2, 1, 2, 2)),
                ),
                "SUM_WEIGHT": (
                    ("frequency", "weight_polarization"),
                    np.ones((2, 1)),
                ),
            },
            coords={"frequency": [1.0e9, 1.1e9]},
        )

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.calculate_imaging_weights."
        "grid_imaging_weight_density_continuum",
        fake_grid,
    )
    result = continuum_node.grid_imaging_weight_density_continuum_node(
        **_weight_node_params()
    )
    assert result["task_id"] == 4
    assert set(result["weight_density"]) == {"WEIGHT_DENSITY_GRID", "SUM_WEIGHT"}
    assert result["timing_node_tasks"].iloc[0].n_frequency_channels == 2


def test_weight_degrid_node_extracts_only_registered_weight_arrays(monkeypatch):
    """The second weighting node returns lightweight per-child weight caches."""
    monkeypatch.setattr("xradio.image.make_empty_sky_image", _empty_weight_image)
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
    result = continuum_node.degrid_imaging_weights_continuum_node(**params)
    cached = result["weight_datasets"]["ms"]
    assert set(cached) == {"WEIGHT_IMAGING"}
    assert cached.WEIGHT_IMAGING.attrs["units"] == "arbitrary"
    assert result["timing_node_tasks"].iloc[0].n_processing_set_children == 1
