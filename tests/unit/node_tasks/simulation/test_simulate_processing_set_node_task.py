"""Node task: chunk slicing, graph_mode=False, and region writes."""

import numpy as np
import pytest
import zarr
from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
    make_parallel_coord,
)
from xradio.measurement_set import load_processing_set, open_processing_set
from xradio.schema.check import check_datatree

from astroviper.node_tasks.simulation import simulate_processing_set as node_task
from astroviper.processing_functions.simulation import (
    simulate_processing_set as simulate_processing_set_pf,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.measurement_set_tools import (
    create_empty_measurement_set_v4_on_disk,
    make_empty_visibility_xds,
    make_field_and_source_xds,
    make_frequency_coordinate,
    make_time_coordinate,
    resolve_fields,
)
from astroviper.utils.telescope_layout import (
    observatory_position,
    read_telescope_layout,
)


@pytest.fixture()
def setup(tmp_path):
    ant = read_telescope_layout("vla.d", antenna_selection=list(range(6)))
    tc = make_time_coordinate(
        {"time_start": "2019-10-03T19:00:00.000", "time_delta": 600.0, "n_samples": 6}
    )
    fc = make_frequency_coordinate(
        {"freq_start": 3e9, "freq_delta": 1e8, "n_channels": 4}
    )
    pc = np.array([[5.2, 0.7]])
    src = np.array([[[5.2 + 1e-3, 0.7 + 5e-4], [5.2 - 2e-3, 0.7]]])
    # time- and frequency-dependent flux so that the slicing is exercised
    flux = (
        np.ones((2, 6, 4, 4)) * np.arange(1, 7)[None, :, None, None]
        + np.arange(4)[None, None, :, None] * 0.1
    )
    names, uniq, upc = resolve_fields(pc, None, 6)
    ms_xds = make_empty_visibility_xds(tc, fc, ["RR", "LL"], ant, names)
    parallel_coords = {
        "time": make_parallel_coord(coord=tc, n_chunks=2),
        "frequency": make_parallel_coord(coord=fc, n_chunks=2),
    }
    ps_store = str(tmp_path / "sim.ps.zarr")
    ms_path = create_empty_measurement_set_v4_on_disk(
        ps_store,
        "vla",
        ms_xds,
        ant,
        make_field_and_source_xds(uniq, upc),
        parallel_coords,
        overwrite=True,
    )
    mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords, {})
    params = dict(
        ms_path=ms_path,
        polarization=["RR", "LL"],
        antenna_position=ant.ANTENNA_POSITION.values,
        site_position=observatory_position("VLA"),
        point_source_flux=flux,
        point_source_ra_dec=src,
        phase_center_ra_dec=pc,
        beam_models=[airy_disk_model("vla")],
        beam_model_map=np.zeros(6, int),
        uvw_params={"auto_correlations": False},
        noise_params={"t_receiver": 50.0, "random_seed": 11},
        channel_width=1e8,
        integration_time=600.0,
    )
    return ps_store, ms_path, tc, fc, mapping, params


def test_graph_mode_false_matches_processing_function(setup):
    ps_store, ms_path, tc, fc, mapping, params = setup
    task = mapping[3]  # second time chunk, second frequency chunk
    xds = node_task(
        **params,
        task_coords=task["task_coords"],
        data_selection={},
        task_id=3,
        graph_mode=False,
    )
    t_sl = task["task_coords"]["time"]["slice"]
    f_sl = task["task_coords"]["frequency"]["slice"]
    ref, _ = simulate_processing_set_pf(
        np.asarray(tc["data"])[t_sl],
        np.asarray(fc["data"])[f_sl],
        ["RR", "LL"],
        params["antenna_position"],
        params["site_position"],
        params["point_source_flux"][:, t_sl, f_sl],
        params["point_source_ra_dec"],
        params["phase_center_ra_dec"],
        params["beam_models"],
        params["beam_model_map"],
        noise_params={"t_receiver": 50.0, "random_seed": 11 + 3},
        channel_width=1e8,
        integration_time=600.0,
    )
    np.testing.assert_allclose(xds.VISIBILITY.values, ref.VISIBILITY.values)
    np.testing.assert_allclose(xds.UVW.values, ref.UVW.values)
    assert xds.sizes == {
        "time": 3,
        "baseline_id": 15,
        "frequency": 2,
        "polarization": 2,
        "uvw_label": 3,
    }


def test_tasks_write_disjoint_regions_and_timing(setup):
    ps_store, ms_path, tc, fc, mapping, params = setup
    frames = [
        node_task(
            **params,
            task_coords=task["task_coords"],
            data_selection={},
            task_id=task_id,
        )
        for task_id, task in mapping.items()
    ]
    for df in frames:
        assert {
            "task_id",
            "T_uvw",
            "T_beams",
            "T_visibilities",
            "T_noise",
            "T_write",
            "T_simulate_task",
        } <= set(df)
    zarr.consolidate_metadata(ps_store)
    # the assembled processing set must pass the XRADIO MSv4 schema checker
    issues = check_datatree(open_processing_set(ps_store))
    assert str(issues) == "No schema issues found", str(issues)
    ms = load_processing_set(ps_store)["vla"].ds
    assert not np.isnan(ms.VISIBILITY.values).any()
    assert not np.isnan(ms.UVW.values).any()
    assert np.all(ms.WEIGHT.values > 0)
    # full reference computed in one go (same per-task seeds -> noise differs, compare UVW and
    # the noise-free part through a second noiseless run)
    params_no_noise = {**params, "noise_params": None}
    for task_id, task in mapping.items():
        node_task(
            **params_no_noise,
            task_coords=task["task_coords"],
            data_selection={},
            task_id=task_id,
        )
    ms = load_processing_set(ps_store)["vla"].ds
    ref, _ = simulate_processing_set_pf(
        np.asarray(tc["data"]), np.asarray(fc["data"]), ["RR", "LL"],
        params["antenna_position"], params["site_position"], params["point_source_flux"],
        params["point_source_ra_dec"], params["phase_center_ra_dec"], params["beam_models"], params["beam_model_map"],
    )  # fmt: skip
    np.testing.assert_allclose(ms.VISIBILITY.values, ref.VISIBILITY.values, atol=1e-12)
    np.testing.assert_allclose(ms.UVW.values, ref.UVW.values, atol=1e-9)
    np.testing.assert_array_equal(ms.WEIGHT.values, 1.0)
    assert not ms.FLAG.values.any()
