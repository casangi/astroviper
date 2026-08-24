"""Unit tests for continuum distributed orchestration and output.

The continuum graph reduces frequency chunks into Taylor products in memory.
These tests ensure the disk output uses those finalized dimensions and values,
rather than retaining the frequency-cube NaN placeholders created at startup.
"""

from pathlib import Path
from zipfile import ZipFile

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xradio.image import write_image
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    _apply_exact_frequency_selection_to_continuum_mapping,
    _continuum_image_for_disk,
    calculate_number_of_chunks_for_continuum_imaging,
    combine_continuum_chunks,
    combine_continuum_imaging_weight_chunks,
    combine_continuum_weight_density_chunks,
    compute_continuum_imaging_weight_degrid_graph,
    image_continuum_single_field,
    prepare_continuum_imaging_weights_global,
)

TW_HYDRA_ARCHIVE = Path(__file__).parent / "data" / "tw_hydra_5chan_fixture.zip"
TW_HYDRA_STORE_NAME = "twhya_selfcal_lsrk_5chans.ps.zarr"
TW_HYDRA_RELATIVE_TOLERANCE = 1.0e-6


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
def tw_hydra_store(tmp_path_factory):
    """Extract the repository's five-channel TW Hydra processing-set fixture."""
    fixture_directory = tmp_path_factory.mktemp("tw_hydra_continuum")
    with ZipFile(TW_HYDRA_ARCHIVE) as archive:
        archive.extractall(fixture_directory)
    return fixture_directory / TW_HYDRA_STORE_NAME


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


def test_weight_density_reducer_aligns_adds_and_sorts_channels():
    """Global density reduction adds overlaps and retains disjoint channels."""
    result = combine_continuum_weight_density_chunks(
        [
            _weight_density([1.1e9, 1.2e9], 1.0, 0),
            _weight_density([1.0e9, 1.1e9], 2.0, 1),
        ],
        {},
    )
    density = result["weight_density"]
    np.testing.assert_array_equal(density.frequency, [1.0e9, 1.1e9, 1.2e9])
    np.testing.assert_allclose(density.WEIGHT_DENSITY_GRID[:, 0, 0, 0], [2, 3, 1])
    assert density.attrs["n_processing_set_datasets_gridded"] == 2
    assert len(result["timing_node_tasks"]) == 2


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

    leaf = _weight_density([1.0e9, 1.1e9])
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
        "astroviper.processing_functions.imaging.calculate_imaging_weights.collapse_continuum_weight_density",
        lambda dataset: dataset.isel(frequency=slice(0, 1)),
    )
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
        "reference_frequency_hz": reference_frequency,
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
    iteration_control_params=None,
    image_param_overrides=None,
    restore=False,
    pbcor=False,
    pblimit=0.2,
    single_precision_image=False,
    skunk_works=False,
    disk_chunk_sizes=None,
    output_shard_channels=None,
    reduce_mode="tree",
    reduce_n_batch=2,
    cache_directory=None,
    write_visibility_model_to_ps=False,
    write_imaging_weights_to_ps=False,
    clear_cache=True,
):
    """Run one public distributed continuum configuration on TW Hydra."""
    if iteration_control_params is None:
        iteration_control_params = {
            "niter": 1,
            "nmajor": 0,
            "threshold": 1.0e30,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
        }
    with dask.config.set(scheduler="synchronous"):
        result = image_continuum_single_field(
            ps_store=str(store),
            image_store=str(output_store),
            image_params=_tw_hydra_image_params(processing_set, image_param_overrides),
            imaging_weights_params=weighting,
            iteration_control_params=iteration_control_params,
            gridder="prolate_spheroidal",
            deconvolver="hogbom",
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
            processing_set_data_group_name="base",
            single_precision_image=single_precision_image,
            processing_function_threads=1,
            n_chunks=n_chunks,
            overwrite=True,
            memory_mode="in_memory",
            cache_directory=cache_directory,
            write_visibility_model_to_ps=write_visibility_model_to_ps,
            write_imaging_weights_to_ps=write_imaging_weights_to_ps,
            clear_cache=clear_cache,
            skunk_works=skunk_works,
            disk_chunk_sizes=disk_chunk_sizes,
            output_shard_channels=output_shard_channels,
            vizualize_graph=False,
            compute_backend="dask",
            reduce_mode=reduce_mode,
            reduce_n_batch=reduce_n_batch,
        )
    return result, xr.open_zarr(output_store)


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
            id="mfs-briggs-global",
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
def test_tw_hydra_cleaning_runs_later_major_cycles_and_builds_a_model(
    tmp_path, tw_hydra_store, specmode
):
    """An active threshold exercises prediction and cached later-cycle state."""
    processing_set = open_processing_set(str(tw_hydra_store))
    result, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"tw_hydra_{specmode}_clean.img.zarr",
        processing_set,
        2,
        specmode,
        {"weighting": "natural", "weighting_scope": "local"},
        iteration_control_params={
            "niter": 20,
            "nmajor": 2,
            "threshold": 0.001,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": -1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
        },
    )

    assert result["n_major_cycles"] >= 2
    assert np.nanmax(np.abs(image.SKY_MODEL.values)) > 0.0
    assert np.isfinite(image.SKY_RESIDUAL.values).all()
    assert np.isfinite(image.SKY_MODEL.values).all()


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
        iteration_control_params={
            "niter": 20,
            "nmajor": 2,
            "threshold": 0.001,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 5,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
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

    assert result["n_major_cycles"] >= 2
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


def test_tw_hydra_direct_niter_zero_dirty_image(tmp_path, tw_hydra_store):
    """The public continuum application accepts a genuine zero-iteration run."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / "tw_hydra_niter_zero.img.zarr",
        processing_set,
        2,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        iteration_control_params={
            "niter": 0,
            "nmajor": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": -1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
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
def test_tw_hydra_alternate_reduction_and_sharded_direct_io(
    tmp_path, tw_hydra_store, reduce_mode, reduce_n_batch
):
    """Alternate reducers work with direct loading and channel-sharded output."""
    processing_set = open_processing_set(str(tw_hydra_store))
    _, image = _run_tw_hydra_continuum(
        tw_hydra_store,
        tmp_path / f"tw_hydra_{reduce_mode}_sharded.img.zarr",
        processing_set,
        5,
        "mfs",
        {"weighting": "natural", "weighting_scope": "local"},
        skunk_works=True,
        disk_chunk_sizes="Auto",
        output_shard_channels=2,
        reduce_mode=reduce_mode,
        reduce_n_batch=reduce_n_batch,
    )

    assert np.isfinite(image.SKY_RESIDUAL.values).all()


def test_chunk_count_respects_override_and_calculates_when_absent(monkeypatch):
    """Explicit chunking is retained; automatic chunking uses image memory."""
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
    monkeypatch.setattr(
        "astroviper.utils.data_partitioning.calculate_data_chunking",
        lambda *args, **kwargs: {"frequency": 2},
    )
    assert (
        calculate_number_of_chunks_for_continuum_imaging(
            image, False, None, {"n_threads": 1}
        )
        == 2
    )
