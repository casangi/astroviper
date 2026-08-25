"""The imaging node task must skip-and-log a failed read/write instead of
raising -- one bad chunk previously tore down entire multi-node runs (the
2019.1.01463.S 80000-channel benchmark died on a single unreadable shard)."""

from __future__ import annotations

import numpy as np


def _minimal_task_inputs(tmp_path):
    """Just enough to reach (and fail) the load phase: the input store does not
    exist, so the read raises after the empty per-chunk image is built."""
    image_params = {
        "phase_direction": np.array([1.0, 0.5]),
        "image_size": [4, 4],
        "cell_size": np.array([-1.0, 1.0]) * 4.85e-6,
        "time_coords": [0],
        "polarization_coords": ["I"],
        "fft_padding": 1.2,
    }
    task_coords = {"frequency": {"data": np.array([1.0e9, 1.1e9])}}
    data_selection = {"ms_a": {"frequency": slice(5, 7)}}
    return dict(
        image_params=image_params,
        imaging_weights_params={"weighting": "natural"},
        iteration_control_params={"niter": 0},
        task_coords=task_coords,
        data_selection=data_selection,
        image_store=str(tmp_path / "img.zarr"),
        input_data_store=str(tmp_path / "does_not_exist.ps.zarr"),
        skunk_works=True,
        task_id=7,
    )


def test_load_failure_returns_marked_row_instead_of_raising(tmp_path):
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    result = image_cube_single_field(**_minimal_task_inputs(tmp_path))
    df = result["timing_node_tasks"]
    assert list(df["task_failed_phase"]) == ["load"]
    assert df["failed_channel_start"].iloc[0] == 5  # from the frequency slice
    assert df["n_channels"].iloc[0] == 2
    assert df["task_id"].iloc[0] == 7
    assert df["task_error"].iloc[0]  # carries the original exception repr
    assert "T_image_cube_task" in df.columns
    assert result["deconvolution"].data == {}  # empty ReturnDict merges cleanly


def test_failed_rows_merge_through_the_standard_reduce(tmp_path):
    """Failed-task results must flow through the production reducer unchanged
    (the shape a mixed failed/successful run reduces through)."""
    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        combine_return_data_frames,
    )
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    r1 = image_cube_single_field(**_minimal_task_inputs(tmp_path))
    r2 = image_cube_single_field(**_minimal_task_inputs(tmp_path))
    combined = combine_return_data_frames([r1, r2], {})
    assert len(combined["timing_node_tasks"]) == 2
    assert list(combined["timing_node_tasks"]["task_failed_phase"]) == ["load", "load"]
    assert combined["deconvolution"].data == {}


def test_reduce_records_timing_provenance(tmp_path):
    """Every reduce call appends one timing record (start/end/host/n_inputs)
    and pools its children's records, so the final result carries one record
    per reduce node of the whole tree -- what the task-stream analysis draws."""
    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        combine_return_data_frames,
    )
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    leaves = [
        image_cube_single_field(**_minimal_task_inputs(tmp_path)) for _ in range(4)
    ]
    assert all("timing_reduce_nodes" not in r for r in leaves)

    partial_a = combine_return_data_frames(leaves[:2], {})
    partial_b = combine_return_data_frames(leaves[2:], {})
    assert len(partial_a["timing_reduce_nodes"]) == 1
    rec = partial_a["timing_reduce_nodes"][0]
    assert rec["n_inputs"] == 2 and rec["n_rows_out"] == 2
    assert rec["end_unixtime"] >= rec["start_unixtime"]
    assert rec["hostname"]

    final = combine_return_data_frames([partial_a, partial_b], {})
    # 2 child records + this call's own = one record per reduce node.
    assert len(final["timing_reduce_nodes"]) == 3
    assert len(final["timing_node_tasks"]) == 4
