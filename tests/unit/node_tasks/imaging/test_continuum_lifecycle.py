"""Generated temporary caches and finalization lifecycle regressions."""

import importlib

import numpy as np
import pytest
import xarray as xr
import zarr

from tests.utils.continuum_images import make_continuum_image

driver = importlib.import_module(
    "astroviper.distributed_applications.imaging.image_continuum_single_field"
)
node = importlib.import_module(
    "astroviper.node_tasks.imaging.image_continuum_single_field"
)
processing = importlib.import_module(
    "astroviper.processing_functions.imaging.image_continuum_single_field"
)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("restore,pbcor", [(False, False), (True, False), (True, True)])
def test_finalization_prepares_only_when_needed_and_restores_once(
    monkeypatch, prepared, restore, pbcor
):
    image = make_continuum_image()
    model = image[["SKY_MODEL"]].copy(deep=True)
    model_snapshot = model.copy(deep=True)
    image["SKY_MODEL"] = xr.zeros_like(
        image.SKY_MODEL
    )  # Must install accumulated state.
    calls = []

    def prepare(data, params, **kwargs):
        calls.append("prepare")
        assert kwargs["initialize_static_products"] is False
        return data, xr.Dataset(), None

    original_restore = processing.restore_image

    def restore_once(data, **kwargs):
        calls.append("restore")
        return original_restore(data, **kwargs)

    monkeypatch.setattr(node, "_prepare_continuum_image", prepare)
    monkeypatch.setattr(processing, "restore_image", restore_once)
    result = node.continuum_finalize_node(
        {"image": image},
        {
            "static_xds": xr.Dataset(),
            "model_xds": model,
            "prepared_continuum_image": prepared,
            "restore": restore,
            "pbcor": pbcor,
        },
    )
    assert calls == ([] if prepared else ["prepare"]) + (["restore"] if restore else [])
    xr.testing.assert_identical(model, model_snapshot)
    xr.testing.assert_identical(result["image"].SKY_MODEL, model.SKY_MODEL)
    assert not np.shares_memory(result["image"].SKY_MODEL, model.SKY_MODEL)
    if restore:
        expected, _ = original_restore(make_continuum_image())
        xr.testing.assert_allclose(result["image"].SKY_RESTORED, expected.SKY_RESTORED)
    else:
        assert "SKY_RESTORED" not in result["image"]
        assert result["timing_restore"] is None
    if pbcor:
        np.testing.assert_allclose(
            result["image"].SKY_RESTORED_PBCOR,
            result["image"].SKY_RESTORED.isel(taylor_term=0, drop=True) / 0.5,
        )
    else:
        assert "SKY_RESTORED_PBCOR" not in result["image"]


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("kind", ["mfs", "mvc", "pb"])
@pytest.mark.parametrize("single_precision", [False, True])
def test_image_cache_replacement_and_repeated_cleanup(
    tmp_path, zarr_format, kind, single_precision
):
    store = str(tmp_path / "image.zarr")
    root = zarr.open_group(store, mode="w", zarr_format=zarr_format)
    root.create_array("PUBLIC", data=np.array([42.0]))
    coords = dict(
        time=[0.0],
        frequency=[100.0, 110.0],
        polarization=["XX", "YY"],
        l=[0.0, 1.0],
        m=[0.0, 1.0],
    )
    image = xr.Dataset(coords=coords)
    if kind == "mfs":
        cache = xr.Dataset(
            {
                "VISIBILITY": (
                    ("taylor_term", "u"),
                    np.ones(
                        (1, 2),
                        dtype=np.complex64 if single_precision else np.complex128,
                    ),
                )
            },
            coords={"taylor_term": [0]},
        )
        for value in (1.0, 7.0):
            cache.VISIBILITY.data[:] = value
            node._write_mfs_visibility_grid_in_place(cache, store)
            assert zarr.open_group(store).metadata.zarr_format == zarr_format
            np.testing.assert_array_equal(zarr.open_group(store)["PUBLIC"][:], [42.0])
            xr.testing.assert_equal(
                node._load_mfs_visibility_grid_in_place(store), cache
            )
        remove = driver._remove_mfs_visibility_grid_cache
        group = "_MFS_VISIBILITY_GRID_CACHE"
    elif kind == "mvc":
        for value in (1.0, 7.0):
            driver._create_mvc_visibility_grid_cache_store(
                store,
                image,
                {"fft_padding": 1.0},
                image.frequency.values,
                "linear",
                single_precision,
                None,
            )
            blank = node._load_mvc_visibility_grid_in_place(image, store)
            assert np.isnan(blank.VISIBILITY).all()
            blank.VISIBILITY.data[:] = value + 2j
            blank.VISIBILITY_NORMALIZATION.data[:] = value
            node._write_mvc_visibility_grid_in_place(
                blank.isel(frequency=[1, 0]), store
            )
            loaded = node._load_mvc_visibility_grid_in_place(image, store)
            np.testing.assert_array_equal(loaded.VISIBILITY, value + 2j)
            np.testing.assert_array_equal(loaded.VISIBILITY_NORMALIZATION, value)
            assert loaded.VISIBILITY.dtype == (
                np.complex64 if single_precision else np.complex128
            )
        remove = driver._remove_mvc_visibility_grid_cache
        group = "_MVC_VISIBILITY_GRID_CACHE"
    else:
        for value in (1.0, 7.0):
            driver._create_wideband_primary_beam_cache_store(
                store, image, image.frequency.values, "linear", single_precision, None
            )
            blank = node._load_wideband_primary_beam_in_place(image, store)
            assert np.isnan(blank.PRIMARY_BEAM).all()
            blank.PRIMARY_BEAM.data[:] = value
            node._write_wideband_primary_beam_in_place(blank, store)
            np.testing.assert_array_equal(
                node._load_wideband_primary_beam_in_place(image, store).PRIMARY_BEAM,
                value,
            )
        remove = driver._remove_wideband_primary_beam_cache
        group = "_WIDEBAND_PRIMARY_BEAM_CACHE"
    remove(store)
    remove(store)
    root = zarr.open_group(store, mode="r", use_consolidated=False)
    assert group not in root
    np.testing.assert_array_equal(root["PUBLIC"][:], [42.0])


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("interrupted", [False, True])
def test_weight_cache_activation_replacement_and_cleanup(
    tmp_path, zarr_format, interrupted
):
    store = str(tmp_path / "weights.zarr")
    name = "WEIGHT_IMAGING_CONTINUUM_CACHE"
    group = {"weight": "WEIGHT"}
    if interrupted:
        group["weight_imaging"] = name
    child = xr.Dataset(
        {"WEIGHT": ("frequency", np.array([1.0, 2.0], dtype=np.float32))},
        coords={"frequency": [100.0, 110.0]},
        attrs={"data_groups": {"base": group}},
    )
    zarr.open_group(store, mode="w", zarr_format=zarr_format)
    child.to_zarr(
        store, group="child", mode="a", consolidated=False, zarr_format=zarr_format
    )
    ps = xr.DataTree.from_dict({"child": child})
    for value in (3.0, 7.0):
        original = driver._create_continuum_weight_cache_store(ps, store, "base")
        root = zarr.open_group(store, mode="r+", use_consolidated=False)
        assert np.isnan(root["child"][name][:]).all()
        assert root["child"][name].dtype == np.dtype("float64")
        assert root["child"][name].chunks == (1,)
        task_child = child.copy(deep=True)
        task_child["CALCULATED"] = ("frequency", [value, value + 1])
        task_child.attrs["data_groups"]["base"]["weight_imaging"] = "CALCULATED"
        node._write_continuum_weights_in_place(
            xr.DataTree.from_dict({"child": task_child}), store
        )
        driver._activate_continuum_weight_cache(ps, store, "base")
        root = zarr.open_group(store, mode="r", use_consolidated=False)
        assert root["child"].attrs["data_groups"]["base"]["weight_imaging"] == name
        np.testing.assert_array_equal(root["child"][name][:], [value, value + 1])
        for consolidated in (False, True):
            for path in (None, "child"):
                view = zarr.open_group(
                    store, path=path, mode="r", use_consolidated=consolidated
                )
                if path is None:
                    view = view["child"]
                np.testing.assert_array_equal(view[name][:], [value, value + 1])
                assert view.attrs["data_groups"]["base"]["weight_imaging"] == name
    driver._remove_continuum_weight_cache(ps, store, original)
    driver._remove_continuum_weight_cache(ps, store, original)
    root = zarr.open_group(store, mode="r", use_consolidated=False)
    assert name not in root["child"]
    assert "weight_imaging" not in root["child"].attrs["data_groups"]["base"]
    np.testing.assert_array_equal(root["child"]["WEIGHT"][:], [1.0, 2.0])
    for consolidated in (False, True):
        for path in (None, "child"):
            view = zarr.open_group(
                store, path=path, mode="r", use_consolidated=consolidated
            )
            if path is None:
                view = view["child"]
            assert name not in view
            assert "weight_imaging" not in view.attrs["data_groups"]["base"]
            np.testing.assert_array_equal(view["WEIGHT"][:], [1.0, 2.0])
