"""Unit tests for continuum distributed orchestration and output.

The continuum graph reduces frequency chunks into Taylor products in memory.
These tests ensure the disk output uses those finalized dimensions and values,
rather than retaining the frequency-cube NaN placeholders created at startup.
"""

import json
import shutil
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from xradio.image import write_image
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    _accumulate_graph_timings,
    _apply_exact_frequency_selection_to_continuum_mapping,
    _continuum_image_for_disk,
    _create_continuum_weight_cache_store,
    _graph_timing_record,
    _load_continuum_clean_mask,
    _mapping_with_task_primary_beams,
    _mapping_with_task_weights,
    calculate_number_of_chunks_for_continuum_imaging,
    combine_continuum_chunks,
    combine_continuum_imaging_weight_chunks,
    combine_continuum_weight_density_chunks,
    compute_continuum_imaging_weight_degrid_graph,
    image_continuum_single_field,
    prepare_continuum_imaging_weights_global,
)
from astroviper.distributed_applications.simulation import simulate_processing_set
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.data_group_tools import (
    create_ps_xdt_data_groups_in_and_out,
    modify_data_groups_ps_xdt,
)
from astroviper.utils.telescope_layout import read_telescope_layout

TW_HYDRA_STORE_NAME = "twhya_selfcal_lsrk_5chans.ps.zarr"
TW_HYDRA_RELATIVE_TOLERANCE = 1.0e-6


def test_mapping_with_task_weights_attaches_only_each_task_partition():
    """In-memory weights enter each map task without an all-task cache copy."""
    mappings = {
        0: {"data_selection": {"ms": {"frequency": slice(0, 1)}}},
        1: {"data_selection": {"ms": {"frequency": slice(1, 2)}}},
    }
    weights = {
        0: {"ms": xr.Dataset({"WEIGHT_IMAGING": ("frequency", [1.0])})},
        1: {"ms": xr.Dataset({"WEIGHT_IMAGING": ("frequency", [2.0])})},
    }

    actual = _mapping_with_task_weights(mappings, weights)

    assert actual[0]["weight_datasets"] is weights[0]
    assert actual[1]["weight_datasets"] is weights[1]
    assert "weight_datasets" not in mappings[0]
    assert "weight_datasets" not in mappings[1]


def test_mapping_with_task_weights_requires_every_task_once():
    """A partial or foreign task cache is rejected before graph construction."""
    mappings = {0: {}, 1: {}}
    with pytest.raises(ValueError, match="does not match"):
        _mapping_with_task_weights(mappings, {0: {}})


def _frequency_mapping(task_frequency_chunks, child_frequencies):
    """Build a minimal GraphVIPER-shaped mapping and Processing Set."""
    processing_set = xr.DataTree()
    processing_set["child"] = xr.Dataset(
        coords={"frequency": np.asarray(child_frequencies, dtype=float)}
    )
    mapping = {
        task_id: {
            "task_coords": {
                "frequency": {"data": np.asarray(frequencies, dtype=float)}
            },
            "data_selection": {},
        }
        for task_id, frequencies in enumerate(task_frequency_chunks)
    }
    return mapping, processing_set


def test_load_continuum_clean_mask_returns_contiguous_boolean_plane(tmp_path):
    """The public mask file is normalized to a C-contiguous Boolean plane."""
    mask_path = tmp_path / "clean_mask.npy"
    np.save(mask_path, np.array([[0.0, 1.0, 0.5], [0.6, -1.0, 2.0]]))

    actual = _load_continuum_clean_mask(mask_path, [2, 3])

    assert actual.dtype == np.bool_
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(
        actual,
        [[False, True, False], [True, False, True]],
    )


@pytest.mark.parametrize(
    ("mask", "image_size", "message"),
    [
        (np.ones((2, 2)), [2, 3], "shape"),
        (np.array([[1.0, np.nan], [0.0, 0.0]]), [2, 2], "non-finite"),
        (np.zeros((2, 2)), [2, 2], "does not select any pixels"),
    ],
)
def test_load_continuum_clean_mask_rejects_invalid_input(
    tmp_path,
    mask,
    image_size,
    message,
):
    """Invalid mask geometry, values, and empty selections fail early."""
    mask_path = tmp_path / "invalid_clean_mask.npy"
    np.save(mask_path, mask)

    with pytest.raises(ValueError, match=message):
        _load_continuum_clean_mask(mask_path, image_size)


def test_exact_frequency_selection_reorders_a_nonmonotonic_child():
    """Irregular integer indexers put a non-monotonic child in task order."""
    mapping, processing_set = _frequency_mapping(
        [[10.0, 11.0, 20.0], [21.0, 30.0, 31.0]],
        [30.0, 31.0, 20.0, 21.0, 10.0, 11.0],
    )

    result = _apply_exact_frequency_selection_to_continuum_mapping(
        mapping,
        processing_set,
    )

    for task_id in result:
        selector = result[task_id]["data_selection"]["child"]["frequency"]
        assert isinstance(selector, np.ndarray)
        selected = processing_set["child"].isel(frequency=selector).frequency.values
        np.testing.assert_array_equal(
            selected,
            result[task_id]["task_coords"]["frequency"]["data"],
        )


def test_exact_frequency_selection_retains_lazy_contiguous_slices():
    """Contiguous child indices remain slice indexers instead of arrays."""
    mapping, processing_set = _frequency_mapping(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        [30.0, 31.0, 20.0, 21.0, 10.0, 11.0],
    )

    result = _apply_exact_frequency_selection_to_continuum_mapping(
        mapping,
        processing_set,
    )

    selectors = [
        result[task_id]["data_selection"]["child"]["frequency"] for task_id in result
    ]
    assert selectors == [slice(4, 6), slice(2, 4), slice(0, 2)]


def test_exact_frequency_selection_rejects_unassigned_child_channels():
    """The helper never silently drops a Processing Set frequency channel."""
    mapping, processing_set = _frequency_mapping([[10.0]], [10.0, 20.0])

    with pytest.raises(ValueError, match="not assigned exactly once"):
        _apply_exact_frequency_selection_to_continuum_mapping(
            mapping,
            processing_set,
        )


@pytest.fixture(scope="module")
def tw_hydra_store(tmp_path_factory, tw_hydra_archive):
    """Extract a writable copy of the downloaded TW Hydra processing set."""
    with TemporaryDirectory(
        prefix="tw_hydra_continuum_", dir=tmp_path_factory.getbasetemp()
    ) as directory:
        fixture_directory = Path(directory)
        with ZipFile(tw_hydra_archive) as archive:
            archive.extractall(fixture_directory)
        yield fixture_directory / TW_HYDRA_STORE_NAME


@pytest.fixture(scope="module")
def tw_hydra_corrected_store(tmp_path_factory, tw_hydra_store):
    """TW Hydra with a ``corrected`` group whose visibilities copy ``base``."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, corrected = create_ps_xdt_data_groups_in_and_out(
        processing_set,
        data_group_in_name="base",
        data_group_out_name="corrected",
        data_group_out_modified={"correlated_data": "VISIBILITY_CORRECTED"},
        overwrite=False,
    )
    for ms_xdt in processing_set.values():
        ms_xdt["VISIBILITY_CORRECTED"] = ms_xdt["VISIBILITY"]
    modify_data_groups_ps_xdt(
        processing_set,
        data_group_out_name="corrected",
        data_group_out=corrected,
        description="Copy of the base visibilities.",
    )
    store = tmp_path_factory.mktemp("tw_hydra_corrected") / TW_HYDRA_STORE_NAME
    processing_set.to_zarr(store, zarr_format=2)
    return store


def _final_continuum_image():
    """Create a small node-finalization-shaped continuum image."""
    coords = {
        "time": [0.0],
        "taylor_term": [0, 1],
        "psf_taylor_order": [0, 1, 2],
        "frequency": [100.0],
        "polarization": ["I"],
        "l": [-1.0, 1.0],
        "m": [-1.0, 0.0, 1.0],
    }
    sky_dims = ("time", "taylor_term", "polarization", "l", "m")
    psf_dims = ("time", "psf_taylor_order", "polarization", "l", "m")
    pb_dims = ("time", "frequency", "polarization", "l", "m")
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (sky_dims, np.arange(12.0).reshape(1, 2, 1, 2, 3)),
            "SKY_MODEL": (sky_dims, np.ones((1, 2, 1, 2, 3))),
            "POINT_SPREAD_FUNCTION": (psf_dims, np.ones((1, 3, 1, 2, 3))),
            "PRIMARY_BEAM": (pb_dims, np.full((1, 1, 1, 2, 3), 0.8)),
            "SKY_RESTORED_PBCOR": (sky_dims, np.full((1, 2, 1, 2, 3), 2.5)),
        },
        coords=coords,
        attrs={
            "type": "image_dataset",
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                    "primary_beam": "PRIMARY_BEAM",
                },
                "model": {"sky": "SKY_MODEL"},
                "restored_pbcor": {"sky": "SKY_RESTORED_PBCOR"},
            },
        },
    )
    return image


def test_continuum_image_for_disk_keeps_requested_products_without_copying_data():
    """Filtering keeps requested Taylor products and leaves the input untouched."""
    image = _final_continuum_image()

    output = _continuum_image_for_disk(
        image,
        ["sky_residual", "point_spread_function"],
    )

    assert set(output.data_vars) == {"SKY_RESIDUAL", "POINT_SPREAD_FUNCTION"}
    assert output["SKY_RESIDUAL"].dims == (
        "time",
        "taylor_term",
        "polarization",
        "l",
        "m",
    )
    assert np.shares_memory(
        output["SKY_RESIDUAL"].values,
        image["SKY_RESIDUAL"].values,
    )
    assert "SKY_MODEL" in image
    assert image.attrs["data_groups"]["model"]["sky"] == "SKY_MODEL"


def test_final_continuum_image_round_trips_finite_taylor_data(tmp_path):
    """The finalized dataset replaces placeholders with finite Zarr values."""
    output = _continuum_image_for_disk(
        _final_continuum_image(),
        [
            "sky_model",
            "sky_residual",
            "point_spread_function",
            "primary_beam",
        ],
        pbcor=True,
    )
    store = tmp_path / "continuum.img.zarr"

    write_image(output, imagename=str(store), out_format="zarr", overwrite=True)
    persisted = xr.open_zarr(store)

    assert persisted["SKY_RESIDUAL"].dims == (
        "time",
        "taylor_term",
        "polarization",
        "l",
        "m",
    )
    np.testing.assert_array_equal(
        persisted["SKY_RESIDUAL"].values,
        output["SKY_RESIDUAL"].values,
    )
    assert np.isfinite(persisted["SKY_RESIDUAL"].values).all()
    assert np.isfinite(persisted["POINT_SPREAD_FUNCTION"].values).all()
    assert "SKY_RESTORED_PBCOR" in persisted


def _map_result(specmode, task_id, frequency=1.0e9, value=1.0):
    """Construct a minimal MFS or MVC reducer leaf."""
    if specmode == "mvc":
        image = xr.Dataset(
            {
                "SKY_RESIDUAL_MVC_CUBE": xr.DataArray(
                    np.full((1, 1, 1, 2, 2), value),
                    dims=("time", "frequency", "polarization", "l", "m"),
                    coords={"frequency": [frequency]},
                )
            }
        )
    else:
        image = xr.Dataset(
            {
                "PRIMARY_BEAM_REFERENCE": xr.DataArray(
                    np.full((1, 1, 2, 2), value),
                    dims=("time", "polarization", "l", "m"),
                )
            }
        )
    image.attrs["continuum_imaging"] = {
        "specmode": specmode,
        "nterms": 2,
        "reference_frequency_hz": 1.0e9,
    }
    return {
        "image": image,
        "timing_node_tasks": pd.DataFrame({"task_id": [task_id]}),
    }


def test_mvc_reducer_sorts_disjoint_frequency_ownership():
    """MVC concatenates exclusively owned channels into frequency order."""
    result = combine_continuum_chunks(
        [_map_result("mvc", 1, 1.1e9, 1.0), _map_result("mvc", 2, 0.9e9, 2.0)],
        {
            "specmode": "mvc",
            "additive_variables": (),
            "frequency_cube_variables": ("SKY_RESIDUAL_MVC_CUBE",),
        },
    )
    residual = result["image"]["SKY_RESIDUAL_MVC_CUBE"]
    np.testing.assert_array_equal(residual.frequency, [0.9e9, 1.1e9])
    np.testing.assert_allclose(residual.isel(frequency=0), 2.0)


def test_mvc_reducer_rejects_duplicate_frequency_ownership():
    """MVC refuses two leaves that claim the same channel."""
    with pytest.raises(ValueError, match="exclusive frequency ownership"):
        combine_continuum_chunks(
            [_map_result("mvc", 1), _map_result("mvc", 2)],
            {
                "specmode": "mvc",
                "additive_variables": (),
                "frequency_cube_variables": ("SKY_RESIDUAL_MVC_CUBE",),
            },
        )


def test_mfs_reducer_retains_one_identical_static_primary_beam():
    """MFS keeps a static PB once and validates duplicate copies."""
    result = combine_continuum_chunks(
        [_map_result("mfs", 1, value=3.0), _map_result("mfs", 2, value=3.0)],
        {"specmode": "mfs", "additive_variables": ()},
    )
    np.testing.assert_allclose(result["image"]["PRIMARY_BEAM_REFERENCE"], 3.0)


def test_mfs_reducer_rejects_inconsistent_static_primary_beams():
    """MFS rejects numerically different copies of a static PB."""
    with pytest.raises(ValueError, match="Static variable.*differs"):
        combine_continuum_chunks(
            [_map_result("mfs", 1, value=1.0), _map_result("mfs", 2, value=2.0)],
            {"specmode": "mfs", "additive_variables": ()},
        )


def _weight_density(frequencies, value=1.0, task_id=0):
    """Construct one valid global-weighting reducer leaf."""
    frequencies = np.asarray(frequencies, dtype=float)
    density = xr.Dataset(
        {
            "WEIGHT_DENSITY_GRID": (
                ("frequency", "weight_polarization", "u", "v"),
                np.full((frequencies.size, 1, 2, 2), value),
            ),
            "SUM_WEIGHT": (
                ("frequency", "weight_polarization"),
                np.full((frequencies.size, 1), value),
            ),
        },
        coords={
            "frequency": frequencies,
            "weight_polarization": [0],
            "u": [0, 1],
            "v": [0, 1],
        },
        attrs={
            "weighting": "briggs",
            "robust": 0.5,
            "n_processing_set_datasets_gridded": 1,
        },
    )
    return {
        "task_id": task_id,
        "weight_density": density,
        "timing_node_tasks": pd.DataFrame({"task_id": [task_id]}),
    }


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("frequencies", [[1.1e9], [1.0e9, 1.1e9]])
def test_weight_density_reducer_rejects_uncollapsed_inputs(position, frequencies):
    """Even a singleton requires the explicit continuum-collapse contract."""
    valid = _weight_density([1.2e9])
    valid["weight_density"].attrs["continuum_frequency_collapsed"] = True
    leaves = [valid, valid]
    leaves[position] = _weight_density(frequencies)
    with pytest.raises(ValueError, match="frequency-collapsed"):
        combine_continuum_weight_density_chunks(leaves, {})


def test_weight_density_reducer_is_associative_and_preserves_inputs():
    leaves = [
        _weight_density([frequency], value, i)
        for i, (frequency, value) in enumerate(
            [(1.0e9, 2.0), (1.2e9, 3.0), (1.4e9, 7.0)]
        )
    ]
    for leaf, count in zip(leaves, [2, 3, 5], strict=True):
        leaf["weight_density"].attrs.update(
            continuum_frequency_collapsed=True, n_input_frequency_channels=count
        )
    originals = [leaf["weight_density"].copy(deep=True) for leaf in leaves]
    flat = combine_continuum_weight_density_chunks(leaves, {})
    partial = combine_continuum_weight_density_chunks(leaves[:2], {})
    tree = combine_continuum_weight_density_chunks([partial, leaves[2]], {})
    xr.testing.assert_allclose(flat["weight_density"], tree["weight_density"])
    actual = tree["weight_density"]
    np.testing.assert_allclose(actual.WEIGHT_DENSITY_GRID, 12.0)
    np.testing.assert_allclose(actual.SUM_WEIGHT, 12.0)
    np.testing.assert_allclose(
        actual.frequency, [(1.0e9 * 2 + 1.2e9 * 3 + 1.4e9 * 5) / 10]
    )
    assert actual.attrs["n_input_frequency_channels"] == 10
    assert actual.attrs["n_processing_set_datasets_gridded"] == 3
    assert actual.attrs["n_weight_density_chunks_combined"] == 3
    assert len(tree["timing_node_tasks"]) == 3
    for leaf, original in zip(leaves, originals, strict=True):
        xr.testing.assert_identical(leaf["weight_density"], original)


def test_weight_density_reducer_keeps_collapsed_maps_single_plane():
    """Global continuum reduction adds singleton planes without frequency expansion."""
    first = _weight_density([1.0e9], 2.0, 0)
    second = _weight_density([1.2e9], 3.0, 1)
    for leaf, n_channels in ((first, 5), (second, 7)):
        leaf["weight_density"].attrs.update(
            continuum_frequency_collapsed=True,
            n_input_frequency_channels=n_channels,
        )

    result = combine_continuum_weight_density_chunks([first, second], {})
    density = result["weight_density"]

    assert density.sizes["frequency"] == 1
    np.testing.assert_allclose(density.WEIGHT_DENSITY_GRID, 5.0)
    np.testing.assert_allclose(density.SUM_WEIGHT, 5.0)
    np.testing.assert_allclose(density.frequency, (1.0e9 * 5 + 1.2e9 * 7) / 12)
    assert density.attrs["n_input_frequency_channels"] == 12


def test_weight_cache_reducer_combines_leaf_and_partial_results():
    """Weight cache reduction is associative across leaf and partial schemas."""
    result = combine_continuum_imaging_weight_chunks(
        [
            {
                "task_id": 0,
                "weight_datasets": {"a": 1},
                "timing_node_tasks": pd.DataFrame({"task_id": [0]}),
            },
            {
                "weight_cache_mapping": {1: {"b": 2}},
                "timing_node_tasks": pd.DataFrame({"task_id": [1]}),
            },
        ],
        {},
    )
    assert result["weight_cache_mapping"] == {0: {"a": 1}, 1: {"b": 2}}
    assert list(result["timing_node_tasks"].task_id) == [0, 1]


def test_weight_cache_reducer_retains_resource_monitor_series():
    """Weight-degridding graphs preserve GraphVIPER resource samples."""
    result = combine_continuum_imaging_weight_chunks(
        [
            {
                "task_id": 0,
                "weight_datasets": {"a": 1},
                "timing_node_tasks": pd.DataFrame(
                    {"task_id": [0], "T_imaging_weight_degrid_node": [2.0]}
                ),
                "resource_usage": {
                    "hostname": "worker-a",
                    "start_unixtime": 100.0,
                    "cpu_percent": [90.0, 100.0],
                },
            }
        ],
        {},
    )
    timing = result["timing_node_tasks"].iloc[0]
    assert timing.hostname == "worker-a"
    assert timing.start_unixtime == 100.0
    assert timing.cpu_percent == [90.0, 100.0]


def test_graph_timing_record_normalizes_weight_task_phases():
    """Weight graphs expose the common columns used by task-stream plots."""
    result = {
        "timing_node_tasks": pd.DataFrame(
            {
                "task_id": [0],
                "T_load": [1.0],
                "T_weight_density_node": [3.0],
                "hostname": ["worker-a"],
                "start_unixtime": [101.0],
            }
        )
    }
    record = _graph_timing_record(
        "global weight density",
        result,
        {
            "compute_start_unixtime": 100.0,
            "compute_end_unixtime": 104.0,
            "T_compute_dask_graph": 4.0,
        },
    )
    assert record["stage"] == "global weight density"
    assert record["T_compute"] == 4.0
    timing = record["timing_node_tasks"].iloc[0]
    assert timing.T_image_cube_task == 3.0
    assert timing.T_write == 0.0


def test_graph_timing_accumulation_excludes_absolute_anchors():
    """Per-graph wall-clock anchors are not summed into driver durations."""
    destination = {"T_compute_dask_graph": 2.0}
    _accumulate_graph_timings(
        destination,
        {
            "T_compute_dask_graph": 3.0,
            "compute_start_unixtime": 100.0,
            "compute_end_unixtime": 103.0,
        },
    )
    assert destination == {"T_compute_dask_graph": 5.0}


def test_weight_degrid_graph_normalizes_a_single_map_leaf(monkeypatch):
    """A one-partition graph returns the same task-indexed cache as reduction."""
    leaf = {
        "task_id": 0,
        "weight_datasets": {"partition": "weights"},
        "timing_node_tasks": pd.DataFrame({"task_id": [0]}),
    }
    monkeypatch.setattr("graphviper.graph_tools.map", lambda **kwargs: "map")
    monkeypatch.setattr(
        "graphviper.graph_tools.reduce", lambda *args, **kwargs: "reduce"
    )
    monkeypatch.setattr(
        "graphviper.graph_tools.generate_dask_workflow", lambda graph: graph
    )
    monkeypatch.setattr("dask.compute", lambda graph: (leaf,))

    result, timings = compute_continuum_imaging_weight_degrid_graph(
        ps_xdt={},
        node_task_data_mapping=[{}],
        input_params={"global_weighting_xds": xr.Dataset()},
        disk_chunk_sizes={},
        processing_set_data_group_name="base",
        monitor_resources_seconds=1,
        task_priorities={},
    )

    assert result["weight_cache_mapping"] == {0: {"partition": "weights"}}
    assert "T_compute_imaging_weight_degrid_graph" in timings


def test_global_weight_preparation_builds_factors_and_returns_all_tasks(monkeypatch):
    """Two global weighting graphs produce collapsed Briggs factors and caches."""
    import importlib

    module = importlib.import_module(
        "astroviper.distributed_applications.imaging.image_continuum_single_field"
    )
    leaf = _weight_density([1.05e9])
    leaf["weight_density"].attrs.update(
        continuum_frequency_collapsed=True,
        n_input_frequency_channels=2,
    )
    graph_result = {"weight_density": leaf["weight_density"]}
    monkeypatch.setattr("graphviper.graph_tools.map", lambda **kwargs: "map")
    monkeypatch.setattr(
        "graphviper.graph_tools.reduce", lambda *args, **kwargs: "reduce"
    )
    monkeypatch.setattr(
        "graphviper.graph_tools.generate_dask_workflow", lambda graph: graph
    )
    monkeypatch.setattr("dask.compute", lambda graph: (graph_result,))
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.calculate_imaging_weights.normalize_imaging_weight_params",
        lambda params: {"weighting": "briggs", "robust": 0.5},
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.imaging_weighting.briggs_weighting.calculate_briggs_params",
        lambda density, weights, params: np.ones((2, 1, 1)),
    )
    monkeypatch.setattr(
        module,
        "compute_continuum_imaging_weight_degrid_graph",
        lambda **kwargs: ({"weight_cache_mapping": {0: {}, 1: {}}}, {"T_degrid": 1.0}),
    )
    result, timings = prepare_continuum_imaging_weights_global(
        ps_xdt={},
        node_task_data_mapping=[{}, {}],
        input_params={"imaging_weights_params": {"weighting": "briggs"}},
        disk_chunk_sizes={},
        processing_set_data_group_name="base",
        monitor_resources_seconds=1,
        task_priorities={},
    )
    assert set(result["weight_cache_mapping"]) == {0, 1}
    assert timings["T_degrid"] == 1.0


def _tw_hydra_image_params(processing_set, overrides=None):
    """Build a compact continuum image geometry from TW Hydra coordinates."""
    field_xds = processing_set.xr_ps.get_combined_field_and_source_xds()
    phase_direction = field_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=field_xds.attrs["center_field_name"]
    )
    frequency = np.asarray(processing_set.xr_ps.get_freq_axis().values)
    reference_frequency = float(np.mean(frequency))
    image_params = {
        "image_size": [64, 64],
        "cell_size": np.array([-0.2, 0.2]) * np.pi / (180 * 3600),
        "phase_direction": phase_direction.values,
        "frequency_coords": frequency,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
        "nterms": 2,
        "reference_frequency": reference_frequency,
    }
    image_params.update(overrides or {})
    return image_params


def _run_tw_hydra_continuum(
    store,
    output_store,
    processing_set,
    n_chunks,
    specmode,
    weighting,
    *,
    deconvolver="hogbom",
    cycle_control_params=None,
    image_param_overrides=None,
    image_params=None,
    restore=False,
    pbcor=False,
    pblimit=0.2,
    single_precision_image=False,
    skunk_works=False,
    disk_chunk_sizes=None,
    image_sharding=None,
    image_chunking=None,
    compressor=None,
    reduce_mode="tree",
    reduce_n_batch=2,
    cache_directory=None,
    weight_memory_mode="in_memory",
    visibility_memory_mode="recompute",
    widebandpb_memory_mode="recompute",
    write_visibility_model_to_ps=False,
    write_imaging_weights_to_ps=False,
    clear_cache=True,
    scheduler="synchronous",
    processing_set_data_group_name="base",
):
    """Run one public distributed continuum configuration on TW Hydra."""
    if cycle_control_params is None:
        cycle_control_params = {
            "max_iter": 1,
            "max_cycles": 0,
            "threshold": 1.0e30,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": 1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        }
    with dask.config.set(scheduler=scheduler):
        result = image_continuum_single_field(
            ps_store=str(store),
            image_store=str(output_store),
            image_params=(
                _tw_hydra_image_params(processing_set, image_param_overrides)
                if image_params is None
                else image_params
            ),
            imaging_weights_params=weighting,
            cycle_control_params=cycle_control_params,
            gridder="anti_alias",
            deconvolver=deconvolver,
            restore=restore,
            pbcor=pbcor,
            pblimit=pblimit,
            specmode=specmode,
            scan_intents=["OBSERVE_TARGET#ON_SOURCE"],
            image_data_variables_keep=[
                "sky_model",
                "sky_residual",
                "point_spread_function",
                "primary_beam",
                "beam_fit_params_point_spread_function",
            ],
            processing_set_data_group_name=processing_set_data_group_name,
            single_precision_image=single_precision_image,
            processing_function_threads=1,
            n_chunks=n_chunks,
            overwrite=True,
            memory_mode="in_memory",
            **(
                {}
                if weight_memory_mode is None
                else {"weight_memory_mode": weight_memory_mode}
            ),
            visibility_memory_mode=visibility_memory_mode,
            widebandpb_memory_mode=widebandpb_memory_mode,
            cache_directory=cache_directory,
            write_visibility_model_to_ps=write_visibility_model_to_ps,
            write_imaging_weights_to_ps=write_imaging_weights_to_ps,
            clear_cache=clear_cache,
            skunk_works=skunk_works,
            disk_chunk_sizes=disk_chunk_sizes,
            image_sharding=image_sharding,
            image_chunking=image_chunking,
            compressor=compressor,
            vizualize_graph=False,
            compute_backend="dask",
            reduce_mode=reduce_mode,
            reduce_n_batch=reduce_n_batch,
        )
    return result, xr.open_zarr(output_store)


def test_task_primary_beam_mapping_contains_only_each_local_dataset():
    """In-memory MVC graph inputs contain one PB partition, not the full cache."""
    mapping = {
        0: {"task_coords": {"frequency": {"data": np.array([1.0])}}},
        1: {"task_coords": {"frequency": {"data": np.array([2.0])}}},
    }
    beams = {
        0: xr.Dataset({"PRIMARY_BEAM": ("frequency", [10.0])}),
        1: xr.Dataset({"PRIMARY_BEAM": ("frequency", [20.0])}),
    }

    actual = _mapping_with_task_primary_beams(mapping, beams)

    assert actual[0]["primary_beam_xds"] is beams[0]
    assert actual[1]["primary_beam_xds"] is beams[1]
    assert "primary_beam_xds" not in mapping[0]
    assert "pb_cache_mapping" not in actual[0]


def test_task_primary_beam_mapping_rejects_incomplete_cache():
    """A missing task PB fails before constructing an invalid distributed graph."""
    with pytest.raises(ValueError, match="does not match the continuum tasks"):
        _mapping_with_task_primary_beams({0: {}, 1: {}}, {0: xr.Dataset()})


@pytest.mark.parametrize("widebandpb_memory_mode", ["in_place", "in_memory"])
def test_tw_hydra_widebandpb_memory_modes_match_recompute(
    tmp_path, tw_hydra_store, widebandpb_memory_mode
):
    """MVC disk-backed and task-local memory PBs reproduce recomputation."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, reference = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "widebandpb_reference.img.zarr",
        processing_set,
        2,
        "mvc",
        {"weighting": "natural", "weighting_scope": "local"},
    )
    output_store = tmp_path / f"widebandpb_{widebandpb_memory_mode}.img.zarr"
    _, actual = _run_tw_hydra_continuum(
        tw_hydra_store,
        output_store,
        processing_set,
        2,
        "mvc",
        {"weighting": "natural", "weighting_scope": "local"},
        widebandpb_memory_mode=widebandpb_memory_mode,
    )

    for variable in ("SKY_RESIDUAL", "POINT_SPREAD_FUNCTION", "PRIMARY_BEAM"):
        np.testing.assert_allclose(
            actual[variable],
            reference[variable],
            rtol=TW_HYDRA_RELATIVE_TOLERANCE,
            atol=0.0,
            equal_nan=True,
        )
    assert "_WIDEBAND_PRIMARY_BEAM_CACHE" not in zarr.open_group(output_store)


def test_tw_hydra_mfs_ignores_widebandpb_memory_mode(tmp_path, tw_hydra_store):
    """The MVC-specific PB storage keyword leaves MFS products unchanged."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, reference = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "mfs_pb_reference.img.zarr",
        processing_set,
        2,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
    )
    output_store = tmp_path / "mfs_pb_in_place.img.zarr"
    _, actual = _run_tw_hydra_continuum(
        tw_hydra_store,
        output_store,
        processing_set,
        2,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        widebandpb_memory_mode="in_place",
    )

    for variable in ("SKY_RESIDUAL", "POINT_SPREAD_FUNCTION", "PRIMARY_BEAM"):
        np.testing.assert_array_equal(actual[variable], reference[variable])
    assert "_WIDEBAND_PRIMARY_BEAM_CACHE" not in zarr.open_group(output_store)


def _store_snapshot(store, parse_metadata=False):
    """Return every file of a Zarr v2 store, keyed by its relative path.

    With ``parse_metadata`` the metadata files are parsed, and consolidated
    metadata is reduced to the entries it lists: consolidating rewrites them
    in zarr's own JSON layout.
    """
    snapshot = {}
    for path in sorted(Path(store).rglob("*")):
        if not path.is_file():
            continue
        content = path.read_bytes()
        if parse_metadata and path.name in (".zattrs", ".zarray", ".zgroup"):
            content = json.loads(content)
        elif parse_metadata and path.name == ".zmetadata":
            content = sorted(json.loads(content)["metadata"])
        snapshot[str(path.relative_to(store))] = content
    return snapshot


@pytest.mark.parametrize(
    ("specmode", "weighting"),
    [
        pytest.param(
            "mfs",
            {"weighting": "natural", "weighting_scope": "local"},
            id="mfs-local",
        ),
        pytest.param(
            "mfs",
            {
                "weighting": "briggs",
                "robust": 0.5,
                "weighting_scope": "global",
                "casa_weighting_implementation": True,
            },
            id="mfs-global",
        ),
        pytest.param(
            "mvc",
            {"weighting": "natural", "weighting_scope": "local"},
            id="mvc-local",
        ),
        pytest.param(
            "mvc",
            {
                "weighting": "briggs",
                "robust": 0.5,
                "weighting_scope": "global",
                "casa_weighting_implementation": True,
            },
            id="mvc-global",
        ),
    ],
)
def test_tw_hydra_in_place_weights_match_memory_and_are_cleaned(
    tmp_path, tw_hydra_store, specmode, weighting
):
    """Task-local disk weights reproduce the default and are temporary.

    The default uses the output-store cache and leaves the input Processing Set
    byte for byte unchanged.
    """
    cycle_control_params = {
        "max_iter": 20,
        "max_cycles": 2,
        "threshold": 0.001,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "max_iter_per_cycle": 5,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    reference_store = tmp_path / "reference.ps.zarr"
    shutil.copytree(tw_hydra_store, reference_store)
    reference_files = _store_snapshot(reference_store)
    reference_processing_set = open_processing_set(str(reference_store))
    _, reference = _run_tw_hydra_continuum(
        reference_store,
        tmp_path / "reference.img.zarr",
        reference_processing_set,
        2,
        specmode,
        weighting,
        cycle_control_params=cycle_control_params,
        weight_memory_mode="in_memory",
    )
    assert _store_snapshot(reference_store) == reference_files

    in_place_store = tmp_path / "in_place.ps.zarr"
    shutil.copytree(tw_hydra_store, in_place_store)
    in_place_processing_set = open_processing_set(str(in_place_store))
    _, actual = _run_tw_hydra_continuum(
        in_place_store,
        tmp_path / "in_place.img.zarr",
        in_place_processing_set,
        2,
        specmode,
        weighting,
        cycle_control_params=cycle_control_params,
        weight_memory_mode=None,
    )

    assert np.nanmax(np.abs(reference.SKY_MODEL.values)) > 0.0
    for variable in (
        "SKY_MODEL",
        "SKY_RESIDUAL",
        "POINT_SPREAD_FUNCTION",
        "PRIMARY_BEAM",
    ):
        np.testing.assert_array_equal(actual[variable], reference[variable])

    reopened = open_processing_set(str(in_place_store))
    for _, ms_xdt in reopened.items():
        assert "WEIGHT_IMAGING_CONTINUUM_CACHE" not in ms_xdt.ds
        assert "weight_imaging" not in ms_xdt.ds.attrs["data_groups"]["base"]


def test_tw_hydra_in_place_weights_can_be_retained(tmp_path, tw_hydra_store):
    """The persistence flag explicitly copies usable weights into the input PS."""
    in_place_store = tmp_path / "retained.ps.zarr"
    shutil.copytree(tw_hydra_store, in_place_store)
    processing_set = open_processing_set(str(in_place_store))
    _run_tw_hydra_continuum(
        in_place_store,
        tmp_path / "retained.img.zarr",
        processing_set,
        2,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        weight_memory_mode="in_place",
        write_imaging_weights_to_ps=True,
    )

    reopened = open_processing_set(str(in_place_store))
    for _, ms_xdt in reopened.items():
        cache_name = ms_xdt.ds.attrs["data_groups"]["base"]["weight_imaging"]
        assert cache_name == "WEIGHT_IMAGING_CONTINUUM_CACHE"
        cached_weights = ms_xdt.ds[cache_name].values
        # Flagged or absent samples may remain NaN, as they do in memory.  The
        # cache is valid when it contains finite, positive imaging weights.
        assert np.isfinite(cached_weights).any()
        assert np.nanmax(cached_weights) > 0.0

    # A later temporary run ignores and leaves explicitly retained weights.
    _run_tw_hydra_continuum(
        in_place_store,
        tmp_path / "rerun.img.zarr",
        reopened,
        2,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        weight_memory_mode="in_place",
    )
    cleaned = open_processing_set(str(in_place_store))
    for _, ms_xdt in cleaned.items():
        assert "WEIGHT_IMAGING_CONTINUUM_CACHE" in ms_xdt.ds
        assert (
            ms_xdt.ds.attrs["data_groups"]["base"]["weight_imaging"]
            == "WEIGHT_IMAGING_CONTINUUM_CACHE"
        )


@pytest.mark.parametrize(
    "deconvolver", ["hogbom_many_threads", "asp", "aspclean", "asp_clean"]
)
def test_tw_hydra_refused_deconvolvers_write_nothing(
    tmp_path, tw_hydra_store, deconvolver
):
    """A deconvolver the continuum model update does not support is refused
    before the in-place imaging weights or the image store are written."""
    store = tmp_path / "input.ps.zarr"
    shutil.copytree(tw_hydra_store, store)
    files = _store_snapshot(store)
    output_store = tmp_path / "refused.img.zarr"
    with pytest.raises(AssertionError):
        _run_tw_hydra_continuum(
            store,
            output_store,
            open_processing_set(str(store)),
            2,
            "mfs",
            {"weighting": "briggs", "robust": 0.5, "weighting_scope": "global"},
            deconvolver=deconvolver,
            weight_memory_mode="in_place",
        )
    assert _store_snapshot(store) == files
    assert not output_store.exists()


@pytest.mark.parametrize(
    ("weighting", "options", "message"),
    [
        pytest.param(
            {
                "weighting": "briggs_abs",
                "robust": 0.5,
                "briggs_abs_noise": 1.0,
                "weighting_scope": "global",
            },
            {"weight_memory_mode": "in_place"},
            "Global weighting scope",
            id="briggs_abs-global",
        ),
        pytest.param(
            {"weighting": "natural", "weighting_scope": "local"},
            {
                "weight_memory_mode": "in_memory",
                "write_imaging_weights_to_ps": True,
            },
            "requires weight_memory_mode='in_place'",
            id="keep-weights-in-memory",
        ),
    ],
)
def test_tw_hydra_refused_weight_requests_write_nothing(
    tmp_path, tw_hydra_store, weighting, options, message
):
    """A refused request leaves the input Processing Set byte for byte unchanged."""
    store = tmp_path / "input.ps.zarr"
    shutil.copytree(tw_hydra_store, store)
    files = _store_snapshot(store)
    output_store = tmp_path / "refused.img.zarr"
    with pytest.raises(ValueError, match=message):
        _run_tw_hydra_continuum(
            store,
            output_store,
            open_processing_set(str(store)),
            2,
            "mfs",
            weighting,
            **options,
        )
    assert _store_snapshot(store) == files
    assert not output_store.exists()


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "change,error,message",
    [
        pytest.param(
            lambda params: params.update(
                reference_frequency_hz=1.001 * params["reference_frequency"]
            ),
            ValueError,
            "two different reference frequencies",
            id="differing-reference-frequencies",
        ),
        pytest.param(
            lambda params: params.pop("reference_frequency"),
            KeyError,
            "'reference_frequency'",
            id="no-reference-frequency",
        ),
        pytest.param(
            lambda params: params.update(reference_frequency=0.0),
            ValueError,
            "finite and positive",
            id="zero-reference-frequency",
        ),
        pytest.param(
            lambda params: params.pop("nterms"),
            AssertionError,
            None,
            id="no-nterms",
        ),
        pytest.param(
            lambda params: params.update(nterms=0),
            AssertionError,
            None,
            id="zero-nterms",
        ),
        pytest.param(
            lambda params: params.update(nterms=2.0),
            TypeError,
            "must be an integer",
            id="non-integer-nterms",
        ),
        pytest.param(
            lambda params: params.update(
                frequency_coords=params["frequency_coords"][:2]
            ),
            ValueError,
            "not assigned exactly once",
            id="channel-subset",
        ),
    ],
)
def test_tw_hydra_refused_image_params_write_nothing(
    tmp_path, tw_hydra_store, change, error, message
):
    """Invalid image parameters are refused before anything is written."""
    store = tmp_path / "input.ps.zarr"
    shutil.copytree(tw_hydra_store, store)
    files = _store_snapshot(store)
    output_store = tmp_path / "refused.img.zarr"
    processing_set = open_processing_set(str(store))
    image_params = _tw_hydra_image_params(processing_set)
    change(image_params)
    with pytest.raises(error, match=message):
        _run_tw_hydra_continuum(
            store,
            output_store,
            processing_set,
            2,
            "mfs",
            {"weighting": "natural", "weighting_scope": "local"},
            image_params=image_params,
            weight_memory_mode="in_place",
        )
    assert _store_snapshot(store) == files
    assert not output_store.exists()


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
def test_tw_hydra_reference_frequency_alias_gives_the_same_images(
    tmp_path, tw_hydra_store, specmode
):
    """``reference_frequency`` alone, both keys equal and the deprecated
    ``reference_frequency_hz`` alone give the same images."""
    processing_set = open_processing_set(str(tw_hydra_store))
    reference_frequency = _tw_hydra_image_params(processing_set)["reference_frequency"]
    images = {}
    for name, keys in (
        ("reference_frequency", {"reference_frequency": reference_frequency}),
        (
            "both",
            {
                "reference_frequency": reference_frequency,
                "reference_frequency_hz": reference_frequency,
            },
        ),
        ("alias", {"reference_frequency_hz": reference_frequency}),
    ):
        image_params = _tw_hydra_image_params(processing_set)
        del image_params["reference_frequency"]
        image_params.update(keys)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result, images[name] = _run_tw_hydra_continuum(
                tw_hydra_store,
                tmp_path / f"reference_frequency_{specmode}_{name}.img.zarr",
                processing_set,
                2,
                specmode,
                {"weighting": "natural", "weighting_scope": "local"},
                image_params=image_params,
                cycle_control_params={
                    "max_iter": 20,
                    "max_cycles": 2,
                    "threshold": 0.001,
                    "gain": 0.1,
                    "psf_sidelobe_factor": 1.5,
                    "max_iter_per_cycle": 5,
                    "min_psf_fraction": 0.05,
                    "max_psf_fraction": 0.8,
                },
            )
        deprecations = [
            warning
            for warning in caught
            if issubclass(warning.category, FutureWarning)
            and "reference_frequency_hz" in str(warning.message)
        ]
        assert len(deprecations) == ("reference_frequency_hz" in keys)
        assert result["n_cycles"] >= 2
        assert np.nanmax(np.abs(images[name].SKY_MODEL.values)) > 0.0

    for name in ("both", "alias"):
        xr.testing.assert_equal(images[name], images["reference_frequency"])


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
def test_tw_hydra_numpy_and_xarray_scalars_give_the_same_images(
    tmp_path, tw_hydra_store, specmode
):
    """A NumPy integer ``nterms`` and a NumPy or xarray scalar reference
    frequency are accepted and give the same images as Python numbers."""
    processing_set = open_processing_set(str(tw_hydra_store))
    # Rounded to float32 so that every variant holds the same value.
    reference_frequency = float(
        np.float32(_tw_hydra_image_params(processing_set)["reference_frequency"])
    )
    images = {}
    for name, keys in (
        ("python", {"nterms": 2, "reference_frequency": reference_frequency}),
        (
            "numpy",
            {
                "nterms": np.int64(2),
                "reference_frequency": np.float32(reference_frequency),
            },
        ),
        (
            "xarray",
            {
                "nterms": np.int64(2),
                "reference_frequency": xr.DataArray(reference_frequency),
            },
        ),
    ):
        result, images[name] = _run_tw_hydra_continuum(
            tw_hydra_store,
            tmp_path / f"scalars_{specmode}_{name}.img.zarr",
            processing_set,
            2,
            specmode,
            {"weighting": "natural", "weighting_scope": "local"},
            image_param_overrides=keys,
            cycle_control_params={
                "max_iter": 20,
                "max_cycles": 2,
                "threshold": 0.001,
                "gain": 0.1,
                "psf_sidelobe_factor": 1.5,
                "max_iter_per_cycle": 5,
                "min_psf_fraction": 0.05,
                "max_psf_fraction": 0.8,
            },
        )
        assert result["n_cycles"] >= 2
        assert np.nanmax(np.abs(images[name].SKY_MODEL.values)) > 0.0

    for name in ("numpy", "xarray"):
        xr.testing.assert_equal(images[name], images["python"])


@pytest.mark.parametrize(
    "weighting",
    [
        pytest.param({"weighting": "natural", "weighting_scope": "local"}, id="local"),
        pytest.param(
            {"weighting": "briggs", "robust": 0.5, "weighting_scope": "global"},
            id="global",
        ),
    ],
)
def test_tw_hydra_failed_in_place_run_removes_its_weights(
    tmp_path, tw_hydra_store, monkeypatch, weighting
):
    """An error removes the output cache without modifying the input PS."""
    import importlib

    node_module = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    store = tmp_path / "input.ps.zarr"
    shutil.copytree(tw_hydra_store, store)
    contents = _store_snapshot(store, parse_metadata=True)
    write_weights = node_module._write_continuum_weights_in_place

    def write_then_fail(ps_xdt, image_store):
        write_weights(ps_xdt, image_store)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(
        node_module,
        "_write_continuum_weights_in_place",
        write_then_fail,
    )
    image_store = tmp_path / "failed.img.zarr"
    with pytest.raises(RuntimeError, match="injected failure"):
        _run_tw_hydra_continuum(
            store,
            image_store,
            open_processing_set(str(store)),
            2,
            "mfs",
            weighting,
            weight_memory_mode="in_place",
        )
    assert _store_snapshot(store, parse_metadata=True) == contents
    image_root = zarr.open_group(str(image_store), mode="r", use_consolidated=False)
    assert "_CONTINUUM_WEIGHT_CACHE" not in image_root


def test_weight_cache_store_checks_every_child_before_writing(tmp_path, tw_hydra_store):
    """A child without the data group is refused before any cache array exists."""
    store = tmp_path / "input.ps.zarr"
    shutil.copytree(tw_hydra_store, store)
    files = _store_snapshot(store)
    processing_set = dict(open_processing_set(str(store)).items())
    processing_set["without_base"] = xr.DataTree(xr.Dataset(attrs={"data_groups": {}}))
    image_store = tmp_path / "image.zarr"
    zarr.open_group(str(image_store), mode="w")
    with pytest.raises(KeyError, match="does not contain data group"):
        _create_continuum_weight_cache_store(
            processing_set,
            str(store),
            str(image_store),
            "base",
            None,
        )
    assert _store_snapshot(store) == files
    assert "_CONTINUUM_WEIGHT_CACHE" not in zarr.open_group(str(image_store), mode="r")


def test_weight_cache_store_is_external_to_the_input(tmp_path, tw_hydra_store):
    """Creating the disk cache changes only the output image store."""
    store = tmp_path / "input.ps.zarr"
    shutil.copytree(tw_hydra_store, store)
    files = _store_snapshot(store, parse_metadata=True)
    processing_set = open_processing_set(str(store))
    image_store = tmp_path / "image.zarr"
    zarr.open_group(str(image_store), mode="w")
    _create_continuum_weight_cache_store(
        processing_set,
        str(store),
        str(image_store),
        "base",
        None,
    )
    assert _store_snapshot(store, parse_metadata=True) == files
    cache = zarr.open_group(str(image_store), mode="r")["_CONTINUUM_WEIGHT_CACHE"]
    assert set(cache.group_keys()) == set(processing_set.children)
    for ms_name in processing_set.children:
        assert "WEIGHT_IMAGING_CONTINUUM_CACHE" in cache[ms_name]


def _flag_one_hand(store):
    """Flag the first correlation of every fourth time in a Processing Set."""
    root = zarr.open_group(str(store), mode="r+", use_consolidated=False)
    for ms_name in list(root.group_keys()):
        flag = root[ms_name]["FLAG"]
        flag_values = np.asarray(flag[:])
        flag_values[::4, :, :, 0] = True
        flag[:] = flag_values


@pytest.mark.parametrize("weight_memory_mode", ["in_memory", "in_place"])
@pytest.mark.parametrize("earlier_run", ["retained", "foreign"])
def test_tw_hydra_rerun_calculates_its_own_weights(
    tmp_path, tw_hydra_store, monkeypatch, earlier_run, weight_memory_mode
):
    """Imaging weights registered by an earlier run or another tool are not used.

    The earlier run uses Briggs weighting and explicitly retains its weights,
    or another tool registers the input weights. A natural-weighting rerun over
    two imaging cycles equals the same run on a clean copy with the same flags.
    """
    import toolviper.utils.logger

    cycle_control_params = {
        "max_iter": 20,
        "max_cycles": 2,
        "threshold": 0.001,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "max_iter_per_cycle": 5,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    natural = {"weighting": "natural", "weighting_scope": "local"}
    briggs = {"weighting": "briggs", "robust": -2.0, "weighting_scope": "local"}
    registered_name = (
        "WEIGHT_IMAGING_FOREIGN"
        if earlier_run == "foreign"
        else "WEIGHT_IMAGING_CONTINUUM_CACHE"
    )

    reference_store = tmp_path / "reference.ps.zarr"
    shutil.copytree(tw_hydra_store, reference_store)
    if earlier_run == "foreign":
        _flag_one_hand(reference_store)
    _, reference = _run_tw_hydra_continuum(
        reference_store,
        tmp_path / "reference.img.zarr",
        open_processing_set(str(reference_store)),
        2,
        "mfs",
        natural,
        cycle_control_params=cycle_control_params,
        weight_memory_mode=weight_memory_mode,
    )

    store = tmp_path / "input.ps.zarr"
    shutil.copytree(tw_hydra_store, store)
    if earlier_run == "retained":
        _, earlier = _run_tw_hydra_continuum(
            store,
            tmp_path / "earlier.img.zarr",
            open_processing_set(str(store)),
            2,
            "mfs",
            briggs,
            weight_memory_mode="in_place",
            write_imaging_weights_to_ps=True,
        )
        assert not np.array_equal(
            earlier.POINT_SPREAD_FUNCTION.values,
            reference.POINT_SPREAD_FUNCTION.values,
            equal_nan=True,
        )
    else:
        _flag_one_hand(store)
        root = zarr.open_group(str(store), mode="r+", use_consolidated=False)
        for ms_name in list(root.group_keys()):
            ms_group = zarr.open_group(
                root.store, path=ms_name, mode="r+", use_consolidated=False
            )
            ms_group.create_array(
                registered_name,
                data=np.where(ms_group["FLAG"][:], np.nan, ms_group["WEIGHT"][:]),
                attributes=dict(ms_group["WEIGHT"].attrs),
            )
            data_groups = ms_group.attrs["data_groups"]
            ms_group.attrs["data_groups"] = {
                **data_groups,
                "base": {**data_groups["base"], "weight_imaging": registered_name},
            }
            zarr.consolidate_metadata(str(store), path=ms_name)
        zarr.consolidate_metadata(str(store))
    for _, ms_xdt in open_processing_set(str(store)).items():
        data_group = ms_xdt.ds.attrs["data_groups"]["base"]
        assert data_group["weight_imaging"] == registered_name
    files = _store_snapshot(store, parse_metadata=weight_memory_mode == "in_place")

    warnings = []
    monkeypatch.setattr(
        toolviper.utils.logger,
        "warning",
        lambda message, *args, **kwargs: warnings.append(message),
    )
    _, actual = _run_tw_hydra_continuum(
        store,
        tmp_path / "rerun.img.zarr",
        open_processing_set(str(store)),
        2,
        "mfs",
        natural,
        cycle_control_params=cycle_control_params,
        weight_memory_mode=weight_memory_mode,
    )

    assert np.nanmax(np.abs(reference.SKY_MODEL.values)) > 0.0
    for variable in (
        "SKY_MODEL",
        "SKY_RESIDUAL",
        "POINT_SPREAD_FUNCTION",
        "PRIMARY_BEAM",
    ):
        np.testing.assert_array_equal(actual[variable], reference[variable])
    assert any(registered_name in message for message in warnings)
    if weight_memory_mode == "in_memory":
        assert _store_snapshot(store) == files
    elif earlier_run == "foreign":
        # The foreign array and registration are untouched.
        assert _store_snapshot(store, parse_metadata=True) == files
    else:
        assert _store_snapshot(store, parse_metadata=True) == files


@pytest.mark.parametrize(
    ("specmode", "weighting"),
    [
        pytest.param(
            "mfs",
            {"weighting": "natural", "weighting_scope": "local"},
            id="mfs-natural-local",
        ),
        pytest.param(
            "mvc",
            {"weighting": "natural", "weighting_scope": "local"},
            id="mvc-natural-local",
        ),
        pytest.param(
            "mfs",
            {
                "weighting": "briggs",
                "robust": 0.5,
                "weighting_scope": "global",
                "casa_weighting_implementation": True,
            },
            id="mfs-briggs-global-casa",
        ),
        pytest.param(
            "mfs",
            {
                "weighting": "briggs",
                "robust": 0.5,
                "weighting_scope": "global",
                "casa_weighting_implementation": False,
            },
            id="mfs-briggs-global-proper",
        ),
    ],
)
def test_tw_hydra_distributed_continuum_is_partition_invariant(
    tmp_path, tw_hydra_store, specmode, weighting
):
    """Real TW Hydra data give equivalent products with one and five leaves."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, one_chunk = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"tw_hydra_{specmode}_one.img.zarr",
        processing_set,
        1,
        specmode,
        weighting,
    )
    _, five_chunks = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"tw_hydra_{specmode}_five.img.zarr",
        processing_set,
        5,
        specmode,
        weighting,
    )

    assert five_chunks.SKY_RESIDUAL.sizes["taylor_term"] == 2
    assert five_chunks.POINT_SPREAD_FUNCTION.sizes["psf_taylor_order"] == 3
    assert five_chunks.PRIMARY_BEAM.sizes["frequency"] == 1
    for variable in ("SKY_RESIDUAL", "POINT_SPREAD_FUNCTION", "PRIMARY_BEAM"):
        actual = np.asarray(five_chunks[variable].values)
        reference = np.asarray(one_chunk[variable].values)
        assert actual.shape == reference.shape
        assert np.array_equal(np.isfinite(actual), np.isfinite(reference))
        denominator = np.nanmax(np.abs(reference))
        difference = np.nanmax(np.abs(actual - reference))
        relative_difference = difference / denominator if denominator else difference
        assert relative_difference < TW_HYDRA_RELATIVE_TOLERANCE


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
def test_tw_hydra_cleaning_runs_later_imaging_cycles_and_builds_a_model(
    tmp_path, tw_hydra_store, specmode
):
    """A per-cycle cap exercises prediction and cached later-cycle state."""
    processing_set = open_processing_set(str(tw_hydra_store))
    result, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"tw_hydra_{specmode}_clean.img.zarr",
        processing_set,
        2,
        specmode,
        {"weighting": "natural", "weighting_scope": "local"},
        cycle_control_params={
            "max_iter": 20,
            "max_cycles": 2,
            "threshold": 0.001,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": 5,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
    )

    assert result["n_cycles"] >= 2
    assert len(result["timing_graphs"]) == result["n_cycles"] + 1
    assert result["timing_graphs"][0]["stage"].startswith("imaging cycle 1")
    assert result["timing_graphs"][-1]["stage"] == "final residual + restoration"
    assert np.nanmax(np.abs(image.SKY_MODEL.values)) > 0.0
    assert np.isfinite(image.SKY_RESIDUAL.values).all()
    assert np.isfinite(image.SKY_MODEL.values).all()


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
def test_tw_hydra_native_mtmfs_matches_reduction_and_visibility_cache_modes(
    tmp_path, tw_hydra_store, specmode
):
    """Native MT-MFS survives later cycles across distributed execution modes."""
    processing_set = open_processing_set(str(tw_hydra_store))
    cycle_control_params = {
        "max_iter": 2,
        "max_cycles": 2,
        "threshold": 0.0,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "max_iter_per_cycle": 1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "scales": [0.0],
        "small_scale_bias": 0.0,
        "mask_threshold": 0.9,
    }
    common = {
        "deconvolver": "mtmfs",
        "cycle_control_params": cycle_control_params,
    }
    reference_result, reference = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"native_mtmfs_{specmode}_tree_recompute.img.zarr",
        processing_set,
        2,
        specmode,
        {"weighting": "natural", "weighting_scope": "local"},
        reduce_mode="tree",
        visibility_memory_mode="recompute",
        **common,
    )
    cached_result, cached = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"native_mtmfs_{specmode}_single_cached.img.zarr",
        processing_set,
        2,
        specmode,
        {"weighting": "natural", "weighting_scope": "local"},
        reduce_mode="single_node",
        visibility_memory_mode="in_memory",
        **common,
    )

    for result, image in (
        (reference_result, reference),
        (cached_result, cached),
    ):
        assert result["n_cycles"] == 2
        assert result["controller"].cycles_done == 2
        assert result["controller"].total_iter_done > 0
        assert result["deconvolution"].data
        assert image.SKY_MODEL.sizes["taylor_term"] == 2
        assert np.nanmax(np.abs(image.SKY_MODEL.isel(taylor_term=0))) > 0.0
        assert np.nanmax(np.abs(image.SKY_MODEL.isel(taylor_term=1))) > 0.0
        for fields in result["deconvolution"].data.values():
            assert np.isfinite(fields["hessian_condition_number"]).all()
            np.testing.assert_array_equal(fields["scales"], [0.0])

    for variable in (
        "SKY_MODEL",
        "SKY_RESIDUAL",
        "POINT_SPREAD_FUNCTION",
        "PRIMARY_BEAM",
    ):
        np.testing.assert_allclose(
            cached[variable],
            reference[variable],
            rtol=TW_HYDRA_RELATIVE_TOLERANCE,
            atol=1.0e-12,
            equal_nan=True,
        )


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
@pytest.mark.parametrize(
    "weighting",
    [
        pytest.param(
            {"weighting": "natural", "weighting_scope": "local"},
            id="natural-local",
        ),
        pytest.param(
            {
                "weighting": "briggs",
                "robust": 0.5,
                "weighting_scope": "global",
                "casa_weighting_implementation": True,
            },
            id="briggs-global",
        ),
    ],
)
def test_tw_hydra_cached_visibility_grids_match_recomputed_residuals(
    tmp_path,
    tw_hydra_store,
    weighting,
    specmode,
):
    """Cached GWVobs reproduces visibility-domain subtraction through CLEAN."""
    processing_set = open_processing_set(str(tw_hydra_store))
    cycle_control_params = {
        "max_iter": 20,
        "max_cycles": 2,
        "threshold": 0.001,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "max_iter_per_cycle": 5,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    reference_result, reference = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"visibility_{specmode}_{weighting['weighting']}_recompute.img.zarr",
        processing_set,
        2,
        specmode,
        weighting,
        cycle_control_params=cycle_control_params,
        visibility_memory_mode="recompute",
    )
    for cache_mode in ("in_memory", "in_place"):
        output_store = (
            tmp_path
            / f"visibility_{specmode}_{weighting['weighting']}_{cache_mode}.img.zarr"
        )
        cached_result, cached = _run_tw_hydra_continuum(
            tw_hydra_store,
            output_store,
            processing_set,
            2,
            specmode,
            weighting,
            cycle_control_params=cycle_control_params,
            visibility_memory_mode=cache_mode,
        )

        assert cached_result["n_cycles"] == reference_result["n_cycles"]
        for variable in (
            "SKY_MODEL",
            "SKY_RESIDUAL",
            "POINT_SPREAD_FUNCTION",
            "PRIMARY_BEAM",
        ):
            np.testing.assert_allclose(
                cached[variable],
                reference[variable],
                rtol=TW_HYDRA_RELATIVE_TOLERANCE,
                atol=1.0e-12,
                equal_nan=True,
            )

        cache_group = (
            "_MFS_VISIBILITY_GRID_CACHE"
            if specmode == "mfs"
            else "_MVC_VISIBILITY_GRID_CACHE"
        )
        assert cache_group not in zarr.open_group(output_store)


@pytest.mark.parametrize("visibility_memory_mode", ["recompute", "in_memory"])
@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
def test_tw_hydra_corrected_data_group_matches_base(
    tmp_path, tw_hydra_corrected_store, specmode, visibility_memory_mode
):
    """Imaging a copy of ``base`` as ``corrected`` gives bitwise equal products.

    Regression: model prediction degridded against the ``base`` group, which
    the loader drops when another group is imaged, so the first residual
    update after a model update failed with ``Data group base not found``.
    """
    processing_set = open_processing_set(str(tw_hydra_corrected_store))
    images = {}
    for data_group_name in ("base", "corrected"):
        result, images[data_group_name] = _run_tw_hydra_continuum(
            tw_hydra_corrected_store,
            tmp_path / f"{data_group_name}.img.zarr",
            processing_set,
            2,
            specmode,
            {"weighting": "natural", "weighting_scope": "local"},
            cycle_control_params={
                "max_iter": 20,
                "max_cycles": 2,
                "threshold": 0.001,
                "gain": 0.1,
                "psf_sidelobe_factor": 1.5,
                "max_iter_per_cycle": 5,
                "min_psf_fraction": 0.05,
                "max_psf_fraction": 0.8,
            },
            visibility_memory_mode=visibility_memory_mode,
            processing_set_data_group_name=data_group_name,
        )
        assert result["n_cycles"] == 2
        assert np.nanmax(np.abs(images[data_group_name].SKY_MODEL.values)) > 0.0

    for variable in (
        "SKY_MODEL",
        "SKY_RESIDUAL",
        "POINT_SPREAD_FUNCTION",
        "PRIMARY_BEAM",
    ):
        expected = images["base"][variable].values
        actual = images["corrected"][variable].values
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("widebandpb_memory_mode", ["in_place", "recompute"])
def test_tw_hydra_widebandpb_modes_preserve_later_mvc_imaging_cycles(
    tmp_path, tw_hydra_store, widebandpb_memory_mode
):
    """Both low-memory PB policies reproduce MVC prediction after real CLEAN."""
    processing_set = open_processing_set(str(tw_hydra_store))
    cycle_control_params = {
        "max_iter": 20,
        "max_cycles": 2,
        "threshold": 0.001,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "max_iter_per_cycle": -1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    _, reference = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "clean_reference.img.zarr",
        processing_set,
        2,
        "mvc",
        {"weighting": "natural", "weighting_scope": "local"},
        cycle_control_params=cycle_control_params,
    )
    _, actual = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"clean_{widebandpb_memory_mode}.img.zarr",
        processing_set,
        2,
        "mvc",
        {"weighting": "natural", "weighting_scope": "local"},
        cycle_control_params=cycle_control_params,
        widebandpb_memory_mode=widebandpb_memory_mode,
    )

    for variable in ("SKY_MODEL", "SKY_RESIDUAL"):
        np.testing.assert_allclose(
            actual[variable],
            reference[variable],
            rtol=TW_HYDRA_RELATIVE_TOLERANCE,
            atol=1.0e-12,
            equal_nan=True,
        )


def test_tw_hydra_restoration_and_pbcor_apply_the_primary_beam_limit(
    tmp_path, tw_hydra_store
):
    """Restoration is finite inside the PB and PB correction masks its edge."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "tw_hydra_restored_pbcor.img.zarr",
        processing_set,
        2,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        image_param_overrides={
            "image_size": [96, 96],
            "cell_size": np.array([-1.0, 1.0]) * np.pi / (180 * 3600),
        },
        restore=True,
        pbcor=True,
        pblimit=0.2,
    )

    assert "SKY_RESTORED" in image
    assert "SKY_RESTORED_PBCOR" in image
    assert np.isfinite(image.SKY_RESTORED.isel(taylor_term=0).values).all()
    pbcor_finite = np.isfinite(image.SKY_RESTORED_PBCOR.values)
    assert np.any(pbcor_finite)
    assert np.any(~pbcor_finite)
    primary_beam = np.squeeze(np.asarray(image.PRIMARY_BEAM.values), axis=1)
    assert primary_beam.shape == pbcor_finite.shape
    assert not np.any(pbcor_finite[primary_beam < 0.2])


@pytest.mark.parametrize(
    ("specmode", "weighting"),
    [
        pytest.param(
            "mfs",
            {
                "weighting": "briggs",
                "robust": 0.5,
                "weighting_scope": "local",
                "casa_weighting_implementation": True,
            },
            id="mfs-briggs-local",
        ),
        pytest.param(
            "mvc",
            {
                "weighting": "briggs",
                "robust": 0.5,
                "weighting_scope": "global",
                "casa_weighting_implementation": True,
            },
            id="mvc-briggs-global",
        ),
    ],
)
def test_tw_hydra_additional_weighting_paths(
    tmp_path, tw_hydra_store, specmode, weighting
):
    """Briggs-local MFS and Briggs-global MVC produce finite products."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"tw_hydra_{specmode}_{weighting['weighting_scope']}.img.zarr",
        processing_set,
        2,
        specmode,
        weighting,
    )

    assert np.isfinite(image.SKY_RESIDUAL.values).all()
    assert np.isfinite(image.POINT_SPREAD_FUNCTION.values).all()


def test_tw_hydra_auto_chunked_single_precision_skunk_works_path(
    tmp_path, tw_hydra_store
):
    """Automatic chunking and direct loading support float32 continuum imaging."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "tw_hydra_auto_float32_skunk.img.zarr",
        processing_set,
        None,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        single_precision_image=True,
        skunk_works=True,
        disk_chunk_sizes="Auto",
    )

    assert image.SKY_RESIDUAL.dtype == np.dtype(np.float32)
    assert np.isfinite(image.SKY_RESIDUAL.values).all()


def test_tw_hydra_mvc_global_cleaning_restoration_and_pbcor(tmp_path, tw_hydra_store):
    """MVC reuses global weights and PBs through cleaning and restoration."""
    processing_set = open_processing_set(str(tw_hydra_store))
    result, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "tw_hydra_mvc_global_clean_restored.img.zarr",
        processing_set,
        2,
        "mvc",
        {
            "weighting": "briggs",
            "robust": 0.5,
            "weighting_scope": "global",
            "casa_weighting_implementation": True,
        },
        cycle_control_params={
            "max_iter": 20,
            "max_cycles": 2,
            "threshold": 0.001,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": 5,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        image_param_overrides={
            "image_size": [96, 96],
            "cell_size": np.array([-1.0, 1.0]) * np.pi / (180 * 3600),
        },
        restore=True,
        pbcor=True,
        pblimit=0.2,
        single_precision_image=True,
    )

    assert result["n_cycles"] >= 2
    assert len(result["timing_graphs"]) == result["n_cycles"] + 3
    assert [record["stage"] for record in result["timing_graphs"][:2]] == [
        "global weight density",
        "global weight degridding",
    ]
    assert np.nanmax(np.abs(image.SKY_MODEL.values)) > 0.0
    assert image.SKY_RESIDUAL.dtype == np.dtype(np.float32)
    assert np.isfinite(image.SKY_RESTORED.isel(taylor_term=0).values).all()
    assert np.isfinite(image.SKY_RESTORED_PBCOR.values).any()
    assert not np.isfinite(image.SKY_RESTORED_PBCOR.values).all()


@pytest.mark.parametrize(
    ("specmode", "weighting"),
    [
        pytest.param(
            "mvc",
            {
                "weighting": "briggs",
                "robust": 0.5,
                "weighting_scope": "local",
                "casa_weighting_implementation": True,
            },
            id="mvc-briggs-local",
        ),
        pytest.param(
            "mfs",
            {"weighting": "uniform", "weighting_scope": "global"},
            id="mfs-uniform-global",
        ),
        pytest.param(
            "mfs",
            {
                "weighting": "briggs_abs",
                "briggs_abs_noise": 0.1,
                "robust": 0.5,
                "weighting_scope": "local",
            },
            id="mfs-briggs-absolute-local",
        ),
    ],
)
def test_tw_hydra_remaining_weighting_schemes(
    tmp_path, tw_hydra_store, specmode, weighting
):
    """Remaining supported weighting configurations produce finite products."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"tw_hydra_{specmode}_{weighting['weighting']}.img.zarr",
        processing_set,
        2,
        specmode,
        weighting,
    )

    assert np.isfinite(image.SKY_RESIDUAL.values).all()
    assert np.isfinite(image.POINT_SPREAD_FUNCTION.values).all()


@pytest.fixture(scope="module")
def wideband_store(tmp_path_factory):
    """A point source at the phase centre, without noise, seen by the VLA in
    D configuration at six times 30 minutes apart and at four channels from 1
    to 2 GHz. Over this octave the uv tracks of the channels overlap only in
    part."""
    store = tmp_path_factory.mktemp("wideband") / "wideband.ps.zarr"
    antenna_xds = read_telescope_layout("vla.d")
    phase_center = np.array([5.233686, 0.71094])
    with dask.config.set(scheduler="synchronous"):
        simulate_processing_set(
            ps_store=str(store),
            antenna_xds=antenna_xds,
            time_params={
                "time_start": "2019-10-03T19:00:00.000",
                "time_delta": 1800.0,
                "n_samples": 6,
            },
            frequency_params={
                "freq_start": 1.0e9,
                "freq_delta": 1.0e9 / 3.0,
                "n_channels": 4,
                "channel_width": 0.01e9,
            },
            polarization=["XX", "YY"],
            sky_components=[
                {
                    "kind": "point",
                    "flux": np.array([1.0, 0.0, 0.0, 1.0]),
                    "ra_dec": phase_center,
                }
            ],
            phase_center_ra_dec=phase_center[None, :],
            beam_models=[airy_disk_model("vla_nvss")],
            beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
            n_time_chunks=1,
            n_frequency_chunks=1,
            overwrite=True,
        )
    return store


def _run_wideband_scopes(tmp_path, store, specmode, weighting, scopes):
    """Image ``store`` with ``weighting`` and each ``weighting_scope`` of
    ``scopes``; ``"default"`` leaves the scope out."""
    processing_set = open_processing_set(str(store))
    images = {}
    for scope in scopes:
        scoped = dict(weighting)
        if scope != "default":
            scoped["weighting_scope"] = scope
        _, images[scope] = _run_tw_hydra_continuum(
            store,
            tmp_path / f"wideband_{specmode}_{scope}.img.zarr",
            processing_set,
            2,
            specmode,
            scoped,
            image_param_overrides={
                "cell_size": np.array([-8.0, 8.0]) * np.pi / (180 * 3600)
            },
        )
    return images


@pytest.mark.parametrize(
    ("specmode", "weighting", "widening"),
    [
        pytest.param("mfs", {"weighting": "uniform"}, 1.2, id="mfs-uniform"),
        pytest.param("mvc", {"weighting": "uniform"}, 1.2, id="mvc-uniform"),
        pytest.param(
            "mfs", {"weighting": "briggs", "robust": 0.5}, 1.02, id="mfs-briggs"
        ),
    ],
)
def test_wideband_uniform_and_briggs_weights_default_to_the_global_density(
    tmp_path, wideband_store, specmode, weighting, widening
):
    """Uniform and Briggs weights of a continuum image default to the weight
    density of all channels (``weighting_scope="global"``); the density of each
    channel (``"local"``) can still be requested. With the local density the
    weights of a uv cell add up over the channels that sample it, which favours
    the inner uv plane, where the tracks of most channels overlap, so the PSF
    is wider (measured: major axis 1.29 times for uniform, 1.03 times for
    Briggs 0.5)."""
    images = _run_wideband_scopes(
        tmp_path, wideband_store, specmode, weighting, ("default", "global", "local")
    )

    xr.testing.assert_equal(images["default"], images["global"])
    major = {
        scope: float(
            images[scope].BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.isel(
                time=0, polarization=0, beam_params=0
            )
        )
        for scope in ("global", "local")
    }
    assert major["local"] > widening * major["global"]


@pytest.mark.parametrize(
    "weighting",
    [
        pytest.param({"weighting": "natural"}, id="natural"),
        pytest.param(
            {"weighting": "briggs_abs", "robust": 0.5, "briggs_abs_noise": 0.1},
            id="briggs_abs",
        ),
    ],
)
def test_wideband_natural_and_absolute_briggs_weights_default_to_the_local_density(
    tmp_path, wideband_store, weighting
):
    """Natural weights use no density and keep the local scope. Absolute Briggs
    weights keep the density of each channel, the only scope that computes
    them: the global scope refuses them."""
    images = _run_wideband_scopes(
        tmp_path, wideband_store, "mfs", weighting, ("default", "local")
    )

    xr.testing.assert_equal(images["default"], images["local"])


def test_tw_hydra_direct_max_iter_zero_dirty_image(tmp_path, tw_hydra_store):
    """The public continuum application accepts a genuine zero-iteration run."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "tw_hydra_max_iter_zero.img.zarr",
        processing_set,
        2,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        cycle_control_params={
            "max_iter": 0,
            "max_cycles": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
    )

    assert np.isfinite(image.SKY_RESIDUAL.values).all()
    assert np.count_nonzero(image.SKY_MODEL.values) == 0


@pytest.mark.parametrize("nterms", [1, 3])
def test_tw_hydra_supports_alternate_taylor_orders(tmp_path, tw_hydra_store, nterms):
    """Continuum dimensions follow one- and three-term Taylor expansions."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"tw_hydra_nterms_{nterms}.img.zarr",
        processing_set,
        2,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        image_param_overrides={"nterms": nterms},
    )

    assert image.SKY_RESIDUAL.sizes["taylor_term"] == nterms
    assert image.POINT_SPREAD_FUNCTION.sizes["psf_taylor_order"] == 2 * nterms - 1


@pytest.mark.parametrize(
    ("reduce_mode", "reduce_n_batch"),
    [("tree_n", 3), ("single_node", 2)],
)
def test_tw_hydra_alternate_reduction_and_direct_io(
    tmp_path, tw_hydra_store, reduce_mode, reduce_n_batch
):
    """Alternate reducers work with the direct Processing Set loading path."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"tw_hydra_{reduce_mode}_direct.img.zarr",
        processing_set,
        5,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        skunk_works=True,
        disk_chunk_sizes="Auto",
        reduce_mode=reduce_mode,
        reduce_n_batch=reduce_n_batch,
    )

    assert np.isfinite(image.SKY_RESIDUAL.values).all()


def test_chunk_count_respects_override_and_calculates_when_absent(monkeypatch):
    """Automatic chunking separates channel scratch from Taylor-grid memory."""
    image = xr.Dataset(
        coords={
            "time": [0],
            "frequency": [1, 2],
            "polarization": [0],
            "l": range(4),
            "m": range(4),
        }
    )
    assert calculate_number_of_chunks_for_continuum_imaging(image, True, 7, {}) == 7
    call = {}

    def chunking(memory_per_frequency, sizes, thread_info, **kwargs):
        call.update(
            memory_per_frequency=memory_per_frequency,
            sizes=sizes,
            thread_info=thread_info,
            **kwargs,
        )
        return {"frequency": 2}

    monkeypatch.setattr(
        "astroviper.utils.data_partitioning.calculate_data_chunking", chunking
    )
    assert (
        calculate_number_of_chunks_for_continuum_imaging(
            image,
            False,
            None,
            {"n_threads": 1, "memory_per_thread": 8.0},
            nterms=3,
            n_correlations=4,
            fft_padding=1.2,
        )
        == 2
    )
    assert call["sizes"] == {"frequency": 2}
    assert call["memory_per_frequency"] > 0
    assert call["constant_memory"] > 0
    assert call["tasks_per_thread"] == 4


@pytest.mark.parametrize(
    "layout",
    [
        {"image_sharding": {"frequency": 2}},
        {"image_chunking": {"l": 32, "m": 32}},
    ],
)
def test_continuum_rejects_output_layout_options_until_final_node_writes(
    tmp_path, tw_hydra_store, layout
):
    """Continuum fails fast instead of discarding a requested output layout."""
    output = tmp_path / "unsupported_layout.img.zarr"
    processing_set = open_processing_set(str(tw_hydra_store))
    with pytest.raises(NotImplementedError, match="not supported"):
        _run_tw_hydra_continuum(
            tw_hydra_store,
            output,
            processing_set,
            5,
            "mfs",
            {"weighting": "natural", "weighting_scope": "local"},
            skunk_works=True,
            **layout,
        )
    assert not output.exists()


def test_continuum_passes_compressor_to_the_final_image_write(
    monkeypatch, tmp_path, tw_hydra_store
):
    """The requested codec applies to the real Taylor products, not placeholders."""
    import xradio.image

    compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=1)
    write_calls = []
    write_image = xradio.image.write_image

    def capture_write(*args, **kwargs):
        write_calls.append(kwargs.copy())
        return write_image(*args, **kwargs)

    monkeypatch.setattr(xradio.image, "write_image", capture_write)
    processing_set = open_processing_set(str(tw_hydra_store))
    _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "compressed.img.zarr",
        processing_set,
        2,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        compressor=compressor,
    )

    assert write_calls[-1]["compressor"] is compressor


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
def test_tw_hydra_continuum_transfers_results_between_processes(
    tmp_path, tw_hydra_store, specmode
):
    from distributed import Client, LocalCluster

    processing_set = open_processing_set(str(tw_hydra_store))
    params = {
        "max_iter": 20,
        "max_cycles": 2,
        "threshold": 0.001,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "max_iter_per_cycle": 5,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    kwargs = dict(
        cycle_control_params=params,
        visibility_memory_mode="in_memory",
        widebandpb_memory_mode="in_memory",
    )
    _, reference = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "serial.img.zarr",
        processing_set,
        3,
        specmode,
        {
            "weighting": "briggs",
            "weighting_scope": "global",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        **kwargs,
    )
    missing_scheduler = object()
    original_scheduler = dask.config.get("scheduler", missing_scheduler)
    with (
        # Client and graph execution both enter scheduler configuration
        # contexts. Restore the caller's setting after every thread and
        # client has finished, including when execution raises or times out.
        dask.config.set(scheduler=dask.config.get("scheduler", None)),
        LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=True,
            memory_limit="2GB",
            dashboard_address=None,
        ) as cluster,
        Client(cluster) as client,
    ):
        import concurrent.futures

        # A broken transfer can leave Dask waiting instead of raising. Bound
        # this regression so it reports a failure rather than hanging CI.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        pending = pool.submit(
            _run_tw_hydra_continuum,
            tw_hydra_store,
            tmp_path / "distributed.img.zarr",
            processing_set,
            3,
            specmode,
            {
                "weighting": "briggs",
                "weighting_scope": "global",
                "robust": 0.5,
                "casa_weighting_implementation": True,
            },
            scheduler=client,
            **kwargs,
        )
        try:
            result, image = pending.result(timeout=90)
        finally:
            client.close()
            pool.shutdown(wait=True, cancel_futures=True)
    assert dask.config.get("scheduler", missing_scheduler) is original_scheduler
    # Do not override the scheduler here: later tests must be able to run
    # ordinary Dask computations without this test's now-closed client.
    assert dask.delayed(sum)([1, 2, 3]).compute() == 6
    assert result["n_cycles"] == 2
    # The helper reopens the output store lazily; read it locally after the
    # distributed client has closed.
    with dask.config.set(scheduler="synchronous"):
        for name in (
            "SKY_MODEL",
            "SKY_RESIDUAL",
            "POINT_SPREAD_FUNCTION",
            "PRIMARY_BEAM",
        ):
            np.testing.assert_allclose(
                image[name], reference[name], rtol=1e-6, atol=1e-12
            )


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
@pytest.mark.parametrize(
    "visibility_memory_mode", ["recompute", "in_memory", "in_place"]
)
@pytest.mark.parametrize(("max_iter", "expected_updates"), [(100, 3), (1, 1)])
def test_refreshed_residual_resumes_after_provisional_threshold_without_restoring(
    tmp_path,
    tw_hydra_store,
    monkeypatch,
    specmode,
    visibility_memory_mode,
    max_iter,
    expected_updates,
):
    """An optimistic model-update peak cannot terminate the distributed driver."""
    import astroviper.processing_functions.imaging.image_continuum_single_field as processing
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_MAX_CYCLES,
        IMAGING_MAX_ITER,
    )

    original_update = processing.model_update_mtmfs_single_field
    original_restore = processing.restore_continuum_image
    events = []

    def optimistic_update(*args, **kwargs):
        result, timings = original_update(*args, **kwargs)
        events.append("update")
        # Emulate an approximate model-update residual below threshold, while
        # keeping the actual model and visibility-based residual untouched.
        for fields in result.data.values():
            if isinstance(fields["peakres"], list):
                fields["peakres"][-1] = 0.0
            else:
                fields["peakres"] = 0.0
        return result, timings

    def counted_restore(*args, **kwargs):
        events.append("restore")
        return original_restore(*args, **kwargs)

    monkeypatch.setattr(
        processing, "model_update_mtmfs_single_field", optimistic_update
    )
    monkeypatch.setattr(processing, "restore_continuum_image", counted_restore)
    processing_set = open_processing_set(str(tw_hydra_store))
    result, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "verified.img.zarr",
        processing_set,
        2,
        specmode,
        {"weighting": "natural", "weighting_scope": "local"},
        visibility_memory_mode=visibility_memory_mode,
        restore=True,
        image_param_overrides={"polarization_coords": ["I", "Q"], "nterms": 1},
        cycle_control_params={
            "max_iter": max_iter,
            "max_cycles": 3,
            "threshold": 1e-8,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": 1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
    )
    assert events == ["update"] * expected_updates + ["restore"]
    assert result["stopcode"].imaging == (
        IMAGING_MAX_CYCLES if max_iter == 100 else IMAGING_MAX_ITER
    )
    assert result["controller"].cycles_done == result["n_cycles"] == expected_updates
    assert len(result["timing_graphs"]) == expected_updates + 1
    assert result["controller"].total_iter_done > 0
    assert np.nanmax(np.abs(image.SKY_RESIDUAL.values)) > 1e-8
    assert result["residual_statistics"].data
