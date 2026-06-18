"""Unit tests for the cube-imaging driver's timing/return-dict contract.

These exercise the data-free parts of
``astroviper.distributed_graphs.imaging.image_cube_single_field``:

* the reduce step ``combine_return_data_frames`` keys its output on
  ``"timing_node_tasks"`` (not the old ``"timing"``), and
* the driver's pre-return timing summaries -- the per-step
  ``timing_distributed_application`` breakdown and the mean / max over the
  per-node-task timing frame -- format without error.

They do not run the imaging graph (no data download); the end-to-end shape is
validated in ``tests/stakeholder/test_single_field_imaging.py``.
"""

import pandas as pd

from astroviper.distributed_graphs.imaging.image_cube_single_field import (
    combine_return_data_frames,
    DISTRIBUTED_APPLICATION_TIMING_PHASES,
    DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
)
from astroviper.processing_functions.imaging.utils import (
    ReturnDict,
    IMAGING_TIMING_PHASES,
    IMAGING_TIMING_TOTAL_KEY,
)
from astroviper.utils.timing import format_timing_summary


def _make_chunk_result(task_id, t_load, t_deconvolve, t_total, chan):
    """One node-task-shaped result: a one-row timing frame + a ReturnDict."""
    timing_df = pd.DataFrame(
        [
            {
                "task_id": task_id,
                "n_channels": 1,
                "n_major_cycles": 2,
                "T_load": t_load,
                "T_deconvolve": t_deconvolve,
                IMAGING_TIMING_TOTAL_KEY: t_total,
            }
        ]
    )
    deconvolve = ReturnDict()
    deconvolve.add({"peakres": 1.0, "iter_done": 5}, time=0, pol=0, chan=chan)
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
    merged_chans = {key.chan for key in combined["deconvolution"].data}
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
    assert set(final) == {"timing_node_tasks", "deconvolution"}
    assert len(final["timing_node_tasks"]) == 2


def test_node_task_timing_mean_and_max_summaries_format():
    """The driver's mean/max-over-chunks summaries render without error."""
    combined = combine_return_data_frames(
        [
            _make_chunk_result(0, t_load=1.0, t_deconvolve=2.0, t_total=4.0, chan=0),
            _make_chunk_result(1, t_load=3.0, t_deconvolve=6.0, t_total=10.0, chan=1),
        ],
        input_params={},
    )
    timing_node_tasks = combined["timing_node_tasks"]

    mean_summary = format_timing_summary(
        timing_node_tasks.mean(numeric_only=True).to_dict(),
        IMAGING_TIMING_PHASES,
        total_key=IMAGING_TIMING_TOTAL_KEY,
        title="mean",
    )
    max_summary = format_timing_summary(
        timing_node_tasks.max(numeric_only=True).to_dict(),
        IMAGING_TIMING_PHASES,
        total_key=IMAGING_TIMING_TOTAL_KEY,
        title="max",
    )

    # Mean total is (4 + 10) / 2 = 7; max total is 10. Both show under the
    # IMAGING_TIMING_TOTAL_KEY grand-total line.
    assert "7.000" in mean_summary
    assert "10.000" in max_summary
    # The deconvolve leaf is grouped under the model-update phase.
    assert "deconvolve" in mean_summary


def test_distributed_application_timing_summary_formats():
    """The driver-level timing dict renders against its phase layout."""
    timing_distributed_application = {
        "T_make_empty_image_xds": 0.1,
        "T_write_empty_image": 0.2,
        "T_determine_chunks_and_parallel_coords": 0.3,
        "T_create_empty_data_variables": 0.4,
        "T_open_processing_set": 0.5,
        "T_interpolate_data_coords": 0.6,
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
