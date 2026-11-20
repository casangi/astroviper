"""Tests of the uv_continuum_subtraction node task on one block of times."""

import os

import numpy as np
import pytest
from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
    make_parallel_coord,
)
from numcodecs import Blosc
from xradio.measurement_set import load_processing_set, open_processing_set

from astroviper.distributed_applications.visibility_manipulation.uv_continuum_subtraction import (
    _create_output_variables,
)
from astroviper.node_tasks.visibility_manipulation import uv_continuum_subtraction
from astroviper.processing_functions.visibility_manipulation import (
    uv_continuum_subtraction as uv_continuum_subtraction_pf,
)
from astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction import (
    DEFAULT_MS_DATA_GROUP_OUT_MODIFIED,
)
from astroviper.utils.io import create_empty_data_variables_on_disk

N_ANTENNA, N_CHANNELS = 5, 16
PHASE_CENTER = np.array([1.2, -0.4])


@pytest.fixture
def store(tmp_path):
    from astroviper.distributed_applications.simulation import (
        simulate_processing_set,
    )
    from astroviper.utils.beam_models import airy_disk_model
    from astroviper.utils.telescope_layout import read_telescope_layout

    store = str(tmp_path / "node_task.ps.zarr")
    flux = np.linspace(1.0, 1.5, N_CHANNELS)
    flux[6:9] += 1.0
    simulate_processing_set(
        ps_store=store,
        antenna_xds=read_telescope_layout(
            "alma.cycle8.1", antenna_selection=list(range(N_ANTENNA))
        ),
        time_params={
            "time_start": "2026-03-01T22:12:00.000",
            "time_delta": 60.0,
            "n_samples": 4,
        },
        frequency_params={
            "freq_start": 100e9,
            "freq_delta": 1e6,
            "n_channels": N_CHANNELS,
            "channel_width": 1e6,
            "spectral_window_name": "spw_0",
        },
        polarization=["XX", "YY"],
        sky_components=[
            {
                "kind": "point",
                "flux": np.stack([flux, 0 * flux, 0 * flux, flux], axis=-1),
                "ra_dec": PHASE_CENTER,
            }
        ],
        phase_center_ra_dec=PHASE_CENTER[None, :],
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(N_ANTENNA, int),
        n_time_chunks=1,
        n_frequency_chunks=1,
        overwrite=True,
    )
    return store


def test_node_task_writes_its_block_of_times(store):
    ps_xdt = open_processing_set(store)
    ms_name = next(iter(ps_xdt.children))
    ms_xdt = ps_xdt[ms_name]
    parallel_coords = {"time": make_parallel_coord(coord=ms_xdt.time, n_chunks=2)}
    _create_output_variables(
        os.path.join(store, ms_name),
        ms_xdt,
        "base",
        "continuum_subtracted",
        DEFAULT_MS_DATA_GROUP_OUT_MODIFIED,
        parallel_coords,
        False,
        create_empty_data_variables_on_disk,
        Blosc(cname="lz4", clevel=5),
    )
    mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, {ms_name: ms_xdt.to_dataset()}
    )
    node = mapping[1]
    frequency = ms_xdt.frequency.values
    ranges = [[frequency[0], frequency[5]], [frequency[9], frequency[-1]]]
    result = uv_continuum_subtraction(
        ps_store=store,
        ms_name=ms_name,
        continuum_frequency_ranges=ranges,
        continuum_frequency_frame="lsrk",
        fit_order=1,
        ms_data_group_in_name="base",
        ms_data_group_out_name="continuum_subtracted",
        task_coords=node["task_coords"],
        data_selection=node["data_selection"],
        task_id=1,
    )
    row = result["timing_node_tasks"].iloc[0]
    assert row["task_id"] == 1 and row["ms_name"] == ms_name and row["n_time"] == 2
    assert row["T_uv_continuum_subtraction_task"] >= row["T_write"] >= 0.0
    statistics = result["fit_statistics"]
    assert list(statistics["polarization"]) == ["XX", "YY"]
    assert (statistics["n_fitted"] == 2 * ms_xdt.sizes["baseline_id"]).all()
    sums = result["spectrum_sums"][ms_name]
    line = np.zeros(N_CHANNELS)
    line[6:9] = 1.0
    np.testing.assert_allclose(
        sums["sum_weighted_visibility_out"][:, 0] / sums["sum_weight_out"][:, 0],
        line,
        atol=1e-12,
    )

    # The driver consolidates the processing set's metadata once all tasks ran.
    written = load_processing_set(store, sel_parms={ms_name: {}})[ms_name]
    output = written.VISIBILITY_CONTINUUM_SUBTRACTED.values
    assert np.isnan(output[:2]).all()  # the other task's block
    expected = load_processing_set(
        store,
        sel_parms={ms_name: {"time": slice(2, 4)}},
        data_group_name="base",
        include_variables=["VISIBILITY", "FLAG", "WEIGHT"],
    )[ms_name]
    uv_continuum_subtraction_pf(expected, ranges, "lsrk", 1)
    np.testing.assert_array_equal(
        output[2:], expected.VISIBILITY_CONTINUUM_SUBTRACTED.values
    )
    np.testing.assert_allclose(
        output[2:], np.broadcast_to(line[:, np.newaxis], output[2:].shape), atol=1e-12
    )
    assert not written.FLAG_CONTINUUM_SUBTRACTED.values[2:].any()
