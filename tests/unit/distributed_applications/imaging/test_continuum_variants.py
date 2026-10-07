"""Partition invariants, upstream failures and remaining reducer guards."""

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    combine_continuum_chunks,
    combine_continuum_weight_density_chunks,
)
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    finalize_mvc_taylor_normal_equations,
    make_mvc_taylor_normal_equation_contributions,
)

driver = importlib.import_module(
    "astroviper.distributed_applications.imaging.image_continuum_single_field"
)


@pytest.mark.parametrize("partition", [(7,), (1, 6), (2, 1, 4), (1, 1, 1, 1, 1, 1, 1)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mvc_reduction_invariant_under_partition_order_tree_and_zero_chunk(
    partition, reverse, dtype
):
    rng = np.random.default_rng(420)
    coords = dict(
        time=[0.0],
        frequency=[80.0, 91.0, 97.0, 100.0, 103.0, 119.0, 125.0],
        polarization=["I"],
        l=[0.0, 1.0],
        m=[0.0, 1.0],
    )
    dims = tuple(coords)
    residual = xr.DataArray(
        rng.normal(size=(1, 7, 1, 2, 2)).astype(dtype), dims=dims, coords=coords
    )
    psf = xr.ones_like(residual)
    pb = xr.DataArray(
        rng.uniform(0.1, 1.0, size=residual.shape).astype(dtype),
        dims=dims,
        coords=coords,
    )
    weights = xr.DataArray(
        np.arange(1.0, 8.0).reshape(1, 7, 1),
        dims=dims[:3],
        coords={k: coords[k] for k in dims[:3]},
    )

    def contribution(selection):
        return make_mvc_taylor_normal_equation_contributions(
            residual.isel(frequency=selection),
            psf.isel(frequency=selection),
            pb.isel(frequency=selection),
            weights.isel(frequency=selection),
            weights.isel(frequency=selection),
            nterms=3,
            reference_frequency=100.0,
        )

    whole = contribution(slice(None))
    expected = finalize_mvc_taylor_normal_equations(whole)
    leaves = []
    start = 0
    for index, size in enumerate(partition):
        leaves.append(
            {
                "image": contribution(slice(start, start + size)),
                "timing_node_tasks": pd.DataFrame({"task_id": [index]}),
            }
        )
        start += size
    zero = whole.copy(deep=True)
    for name in zero:
        zero[name].data[:] = 0
    leaves.append(
        {"image": zero, "timing_node_tasks": pd.DataFrame({"task_id": [len(leaves)]})}
    )
    if reverse:
        leaves.reverse()
    params = {"specmode": "mvc", "additive_variables": list(whole.data_vars)}
    flat = combine_continuum_chunks(leaves, params)
    tree = leaves[0]
    for leaf in leaves[1:]:  # Intentionally unbalanced reduction tree.
        tree = combine_continuum_chunks([tree, leaf], params)
    for reduced in (flat, tree):
        actual = finalize_mvc_taylor_normal_equations(reduced["image"])
        for actual_array, expected_array in zip(actual, expected, strict=True):
            xr.testing.assert_allclose(
                actual_array, expected_array, rtol=1e-6, atol=1e-7
            )
        assert reduced["image"].attrs["n_continuum_chunks_combined"] == len(leaves)


def _density():
    return xr.Dataset(
        {
            "WEIGHT_DENSITY_GRID": (
                ("frequency", "weight_polarization", "u", "v"),
                np.ones((1, 1, 2, 2)),
            ),
            "SUM_WEIGHT": (("frequency", "weight_polarization"), np.ones((1, 1))),
        },
        coords={
            "frequency": [100.0],
            "weight_polarization": [0],
            "u": [0, 1],
            "v": [0, 1],
        },
        attrs={"continuum_frequency_collapsed": True},
    )


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("missing_density", KeyError, "weight_density"),
        ("density_type", TypeError, "Dataset"),
        ("missing_array", KeyError, "missing"),
        ("density_dims", ValueError, "dimensions"),
        ("sum_dims", ValueError, "dimensions"),
        ("uncollapsed", ValueError, "collapsed"),
        ("weighting", ValueError, "Briggs"),
        ("factor_shape", ValueError, "unexpected shape"),
        ("factor_nan", ValueError, "non-finite"),
        ("missing_cache", RuntimeError, "weight_cache_mapping"),
        ("task_ids", RuntimeError, "task identifiers"),
    ],
)
def test_global_weight_preparation_reports_corrupt_upstream_results(
    monkeypatch, fault, error, message
):
    density = _density()
    result = {"weight_density": density}
    factors = np.ones((2, 1, 1))
    downstream = {"weight_cache_mapping": {0: {}}}
    params = {"weighting": "briggs", "robust": 0.5}
    if fault == "missing_density":
        result = {}
    elif fault == "density_type":
        result["weight_density"] = []
    elif fault == "missing_array":
        result["weight_density"] = density.drop_vars("SUM_WEIGHT")
    elif fault == "density_dims":
        result["weight_density"] = density.rename({"u": "other"})
    elif fault == "sum_dims":
        density["SUM_WEIGHT"] = density.SUM_WEIGHT.transpose(
            "weight_polarization", "frequency"
        )
    elif fault == "uncollapsed":
        density.attrs["continuum_frequency_collapsed"] = False
    elif fault == "weighting":
        params["weighting"] = "natural"
    elif fault == "factor_shape":
        factors = np.ones((1, 1, 1))
    elif fault == "factor_nan":
        factors[:] = np.nan
    elif fault == "missing_cache":
        downstream = {}
    else:
        downstream = {"weight_cache_mapping": {1: {}}}
    monkeypatch.setattr("graphviper.graph_tools.map", lambda **kwargs: "map")
    monkeypatch.setattr(
        "graphviper.graph_tools.reduce", lambda *args, **kwargs: "reduce"
    )
    monkeypatch.setattr(
        "graphviper.graph_tools.generate_dask_workflow", lambda graph: graph
    )
    monkeypatch.setattr("dask.compute", lambda graph: (result,))
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.calculate_imaging_weights.normalize_imaging_weight_params",
        lambda p: params,
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.imaging_weighting.briggs_weighting.calculate_briggs_params",
        lambda *args: factors,
    )
    monkeypatch.setattr(
        driver,
        "compute_continuum_imaging_weight_degrid_graph",
        lambda **kwargs: (downstream, {}),
    )
    with pytest.raises(error, match=message):
        driver.prepare_continuum_imaging_weights_global(
            ps_xdt={},
            node_task_data_mapping=[{}],
            input_params={"imaging_weights_params": params},
            disk_chunk_sizes={},
            processing_set_data_group_name="base",
            monitor_resources_seconds=None,
            task_priorities={},
        )


@pytest.mark.parametrize(
    "fault,message",
    [
        ("shape", "Dimension"),
        ("coordinates", "Coordinate"),
        ("robust", "metadata"),
        ("weighting", "metadata"),
        ("collapse", "frequency-collapsed"),
        ("planes", "exactly one frequency"),
        ("timing", "timing_node_tasks"),
        ("timing_type", "DataFrame"),
    ],
)
def test_density_reducer_rejects_inconsistent_geometry_and_metadata(fault, message):
    first = _density()
    second = first.copy(deep=True)
    if fault == "shape":
        second = second.isel(u=[0])
    elif fault == "coordinates":
        second = second.assign_coords(u=[2, 3])
    elif fault == "robust":
        first.attrs["robust"], second.attrs["robust"] = 0.5, 1.0
    elif fault == "weighting":
        first.attrs["weighting"], second.attrs["weighting"] = "briggs", "uniform"
    elif fault == "collapse":
        second.attrs["continuum_frequency_collapsed"] = False
    elif fault == "planes":
        second = xr.concat(
            [second, second.assign_coords(frequency=[110.0])], dim="frequency"
        )
    leaves = [
        {"weight_density": image, "timing_node_tasks": pd.DataFrame({"task_id": [i]})}
        for i, image in enumerate((first, second))
    ]
    if fault == "timing":
        del leaves[1]["timing_node_tasks"]
    elif fault == "timing_type":
        leaves[1]["timing_node_tasks"] = []
    error = (
        KeyError
        if fault == "timing"
        else TypeError
        if fault == "timing_type"
        else ValueError
    )
    with pytest.raises(error, match=message):
        combine_continuum_weight_density_chunks(leaves, {})
