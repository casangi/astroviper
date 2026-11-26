"""Unit tests for the cube-imaging driver's timing/return-dict contract.

These exercise the data-free parts of
``astroviper.distributed_applications.imaging.image_cube_single_field``:

* the reduce step ``combine_return_data_frames`` keys its output on
  ``"timing_node_tasks"`` (not the old ``"timing"``), and
* the driver's pre-return timing summaries -- the per-step
  ``timing_distributed_application`` breakdown and the node-task table over
  the per-node-task timing frame -- format without error.

They do not run the imaging graph (no data download); the end-to-end shape is
validated in ``tests/stakeholder/test_single_field_imaging.py``.
"""

import numpy as np
import pandas as pd

from astroviper.distributed_applications.imaging.image_cube_single_field import (
    DISTRIBUTED_APPLICATION_TIMING_PHASES,
    DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
    combine_return_data_frames,
)
from astroviper.processing_functions.imaging.utils import (
    IMAGING_TIMING_PHASES,
    IMAGING_TIMING_TOTAL_KEY,
    add_to_imaging_dict,
)
from astroviper.utils.timing import format_timing_summary, node_task_timing_table


def _make_chunk_result(task_id, t_load, t_deconvolve, t_total, chan):
    """One node-task-shaped result: a one-row timing frame + an imaging_dict.

    The frame carries the string ``hostname`` column the real node task adds for
    straggler grouping, so the reduce/summary paths are exercised against the
    same mixed-dtype frame production produces (not a numeric-only stand-in).
    """
    timing_df = pd.DataFrame(
        [
            {
                "task_id": task_id,
                "n_channels": 1,
                "n_cycles": 2,
                "T_load": t_load,
                "T_deconvolve": t_deconvolve,
                IMAGING_TIMING_TOTAL_KEY: t_total,
                "hostname": f"node{task_id}",
            }
        ]
    )
    deconvolve = {}
    add_to_imaging_dict(
        deconvolve, {"peakres": 1.0, "iter_done": 5}, time=0, pol=0, chan=chan
    )
    return {"timing_node_tasks": timing_df, "deconvolution": deconvolve}


def test_combine_return_data_frames_uses_timing_node_tasks_key():
    """The reducer reads/writes ``timing_node_tasks`` and concatenates rows."""
    results = [
        _make_chunk_result(0, t_load=1.0, t_deconvolve=2.0, t_total=4.0, chan=0),
        _make_chunk_result(1, t_load=3.0, t_deconvolve=6.0, t_total=10.0, chan=1),
    ]

    combined = combine_return_data_frames(results, input_params={})

    # New key name; the old "timing" key must be gone.
    assert "timing_node_tasks" in combined
    assert "timing" not in combined
    assert "deconvolution" in combined

    combined_timing = combined["timing_node_tasks"]
    assert isinstance(combined_timing, pd.DataFrame)
    # One row per chunk, concatenated.
    assert len(combined_timing) == 2
    assert sorted(combined_timing["task_id"].tolist()) == [0, 1]

    # The merged deconvolution dict covers both chunks' channels.
    merged_chans = {key.chan for key in combined["deconvolution"]}
    assert merged_chans == {0, 1}


def test_combine_return_data_frames_composes_under_tree_reduction():
    """Output shape matches the input shape so tree-mode reduce can re-feed it."""
    leaves = [
        _make_chunk_result(0, 1.0, 2.0, 4.0, chan=0),
        _make_chunk_result(1, 3.0, 6.0, 10.0, chan=1),
    ]
    partial = combine_return_data_frames(leaves[:1], input_params={})
    # A partially-reduced result is itself a valid reducer input.
    final = combine_return_data_frames([partial, leaves[1]], input_params={})
    assert set(final) == {
        "timing_node_tasks",
        "deconvolution",
        "image_statistics",
        "timing_reduce_nodes",
    }
    assert len(final["timing_node_tasks"]) == 2
    # One provenance record per reduce node of the tree: the partial reduce's
    # record is pooled forward and the final call appends its own.
    assert len(final["timing_reduce_nodes"]) == 2


def test_node_task_timing_table_over_the_chunks():
    """The driver logs the node-task table grouped by the imaging layout."""
    combined = combine_return_data_frames(
        [
            _make_chunk_result(0, t_load=1.0, t_deconvolve=2.0, t_total=4.0, chan=0),
            _make_chunk_result(1, t_load=3.0, t_deconvolve=6.0, t_total=10.0, chan=1),
        ],
        input_params={},
    )
    table = node_task_timing_table(
        combined["timing_node_tasks"],
        IMAGING_TIMING_PHASES,
        total_key=IMAGING_TIMING_TOTAL_KEY,
    )
    total = table.loc[("TASK TOTAL", IMAGING_TIMING_TOTAL_KEY)]
    assert (total["mean"], total["median"], total["max"]) == (7.0, 7.0, 10.0)
    deconvolve = ("MODEL UPDATE (summed over imaging cycles)", "deconvolve")
    assert table.loc[deconvolve, "max"] == 6.0
    assert "(outside the phases)" in table.to_string(float_format="{:.3f}".format)


def test_format_timing_summary_skips_string_metadata_columns():
    """A per-node-task timing frame carries a string ``hostname`` column, and the
    node-task print path (``print_timing_summary(timing_df, ...)``) passes that
    raw one-row frame straight into ``format_timing_summary``. The formatter must
    therefore skip non-numeric columns rather than ``float("<hostname>")`` them
    -- the regression that crashed every stakeholder imaging test on any host
    whose name is not a number (``ValueError: could not convert string to float``).

    This guards the DataFrame-coercion branch directly.
    """
    timing_df = pd.DataFrame(
        [
            {
                "task_id": 0,
                "n_channels": 1,
                "n_cycles": 2,
                "T_load": 1.0,
                "T_deconvolve": 2.0,
                IMAGING_TIMING_TOTAL_KEY: 4.0,
                "hostname": "cordierite",  # a string that is not floatable
            }
        ]
    )

    # Must not raise on the string column.
    summary = format_timing_summary(
        timing_df,
        IMAGING_TIMING_PHASES,
        total_key=IMAGING_TIMING_TOTAL_KEY,
        title="node task",
    )

    # Numeric timings still render; the string metadata column is dropped, not
    # coerced into a bogus row.
    assert "4.000" in summary
    assert "deconvolve" in summary
    assert "cordierite" not in summary


def test_distributed_application_timing_summary_formats():
    """The driver-level timing dict renders against its phase layout."""
    timing_distributed_application = {
        "T_make_empty_image_xds": 0.1,
        "T_write_empty_image": 0.2,
        "T_determine_chunks_and_parallel_coords": 0.3,
        "T_create_empty_data_variables": 0.4,
        "T_open_processing_set": 0.5,
        "T_interpolate_data_coords": 0.6,
        "T_add_lazy_input_data": 0.25,
        "T_create_map_reduce_graph": 0.7,
        "T_generate_dask_graph": 0.8,
        "T_compute_dask_graph": 5.0,
        "T_consolidate_metadata": 0.9,
        "T_total": 9.5,
    }

    summary = format_timing_summary(
        timing_distributed_application,
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
        total_key=DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
        title="distributed application",
    )

    assert "DISTRIBUTED APPLICATION (driver)" in summary
    assert "compute dask graph" in summary
    # Grand total line uses T_total.
    assert "9.500" in summary


def test_add_lazy_input_data_is_a_driver_timing_leaf():
    """The Measurement Set v2 step (``add_lazy_input_data``; ``0.0`` for a
    processing set) is a leaf of the driver's timing layout, right after the
    graph mapping, and renders in the summary."""
    (_, _, leaves) = DISTRIBUTED_APPLICATION_TIMING_PHASES[0]
    keys = [key for _, key in leaves]
    assert keys.index("T_add_lazy_input_data") == (keys.index("T_input_preflight") + 1)
    summary = format_timing_summary(
        {"T_add_lazy_input_data": 0.25, "T_total": 1.0},
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
        total_key=DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
        title="distributed application",
    )
    assert "add lazy input data" in summary
    assert "0.250" in summary


def test_combine_return_data_frames_concatenates_the_find_continuum_first_round():
    """The node tasks' first-round rows are concatenated in input order and
    their ``write_failed`` or-ed; a leaf without rows adds nothing, and
    without any there are none."""
    leaves = [
        _make_chunk_result(task_id, 1.0, 2.0, 4.0, chan=task_id) for task_id in range(3)
    ]
    rows = [
        {
            "channels": np.array([4, 5], dtype=np.int32),
            "sky_valid_count": np.array([7, 0]),
            "sky_minimum": np.array([-1.5, np.nan]),
            "sky_maximum": np.array([2.5, np.nan]),
            "primary_beam_valid_count": np.array([9, 9]),
            "primary_beam_minimum": np.array([0.2, 0.25]),
            "tiles": np.array([True, False] * 4).reshape(2, 2, 1, 2),
            "write_failed": False,
        },
        {
            "channels": np.array([0], dtype=np.int32),
            "sky_valid_count": np.array([3]),
            "sky_minimum": np.array([-0.5]),
            "sky_maximum": np.array([0.5]),
            "primary_beam_valid_count": np.array([4]),
            "primary_beam_minimum": np.array([0.3]),
            "tiles": np.ones((1, 2, 1, 2), dtype=bool),
            "write_failed": True,
        },
    ]
    leaves[0]["find_continuum_first_round"] = rows[0]
    leaves[2]["find_continuum_first_round"] = rows[1]
    merged = combine_return_data_frames(leaves, input_params={})[
        "find_continuum_first_round"
    ]
    assert set(merged) == set(rows[0])
    for key in set(rows[0]) - {"write_failed"}:
        np.testing.assert_array_equal(
            merged[key], np.concatenate([row[key] for row in rows])
        )
        assert merged[key].dtype == rows[0][key].dtype
    assert merged["write_failed"] is True

    del leaves[2]["find_continuum_first_round"]
    merged = combine_return_data_frames(leaves, input_params={})[
        "find_continuum_first_round"
    ]
    np.testing.assert_array_equal(merged["channels"], rows[0]["channels"])
    assert merged["write_failed"] is False

    del leaves[0]["find_continuum_first_round"]
    assert "find_continuum_first_round" not in combine_return_data_frames(
        leaves, input_params={}
    )
