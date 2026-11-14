"""Unit tests for the node-task tables of astroviper.utils.timing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astroviper.utils.timing import (
    node_task_phase_columns,
    node_task_phase_totals,
    node_task_timing_table,
)

#: A node-task layout like IMAGING_TIMING_PHASES: a phase with a total and
#: steps, one without a total, one with a total only and one never recorded.
PHASES = [
    (
        "PREP",
        "T_prep",
        [("weights", "T_prep_weights"), ("gcf", "T_prep_gcf")],
    ),
    ("NODE I/O", None, [("load", "T_load"), ("write", "T_write")]),
    ("RESTORE", "T_restore", []),
    ("NEVER", "T_never", [("never step", "T_never_step")]),
]


def _timing_frame():
    """Two node tasks, with a string column the tables skip."""
    return pd.DataFrame(
        {
            "task_id": [0, 1],
            "T_prep": [10.0, 4.0],
            "T_prep_weights": [9.0, 1.0],
            "T_prep_gcf": [0.5, 2.5],
            "T_load": [1.0, 2.0],
            "T_write": [3.0, 1.0],
            "T_image_cube_task": [15.0, 8.0],
            "hostname": ["node", "node"],
        }
    )


def test_flat_node_task_timing_table():
    frame = pd.DataFrame(
        {
            "T_image_cube_task": [2.0, 2.0, 2.0],
            "T_load": [0.4, 0.4, 1.0],
            "T_write": [0.3, np.nan, 0.3],
            "hostname": ["node"] * 3,
        }
    )
    table = node_task_timing_table(frame)
    assert list(table.columns) == ["mean", "median", "max", "total"]
    assert list(table.index) == ["T_image_cube_task", "T_load", "T_write"]
    assert table.index.name == "node task step"
    assert table.loc["T_load", "median"] == pytest.approx(0.4)
    assert table.loc["T_load", "total"] == pytest.approx(1.8)
    assert table.loc["T_write", "mean"] == pytest.approx(0.3)  # NaN ignored


def test_node_task_timing_table_by_phase():
    table = node_task_timing_table(
        _timing_frame(), PHASES, total_key="T_image_cube_task"
    )
    assert table.index.names == ["phase", "step"]
    assert list(table.index) == [
        ("PREP", "phase total"),
        ("PREP", "weights"),
        ("PREP", "gcf"),
        ("PREP", "(unaccounted)"),
        ("NODE I/O", "phase total"),
        ("NODE I/O", "load"),
        ("NODE I/O", "write"),
        ("TASK TOTAL", "T_image_cube_task"),
        ("TASK TOTAL", "(outside the phases)"),
    ]
    # statistics of the per-task values, not of the column statistics:
    # the column maxima would give 10 - 9 - 2.5 = -1.5 and 2 + 3 = 5
    assert table.loc[("PREP", "(unaccounted)"), "max"] == pytest.approx(0.5)
    assert table.loc[("NODE I/O", "phase total"), "max"] == pytest.approx(4.0)
    outside = table.loc[("TASK TOTAL", "(outside the phases)")]
    assert outside["mean"] == outside["max"] == pytest.approx(1.0)
    assert outside["total"] == pytest.approx(2.0)
    assert table.loc[("PREP", "phase total"), "median"] == pytest.approx(7.0)


def test_node_task_timing_table_needs_phases_for_a_total():
    with pytest.raises(ValueError, match="total_key needs phases"):
        node_task_timing_table(_timing_frame(), total_key="T_image_cube_task")


def test_node_task_timing_table_without_recorded_phases():
    table = node_task_timing_table(_timing_frame(), PHASES[3:], total_key="T_absent")
    assert table.empty
    assert table.index.names == ["phase", "step"]
    assert list(table.columns) == ["mean", "median", "max", "total"]


def test_node_task_phase_columns_and_totals():
    frame = _timing_frame()
    assert node_task_phase_columns(frame, PHASES) == {
        "PREP": "T_prep",
        "NODE I/O": ("T_load", "T_write"),
    }
    frame["T_restore"] = [0.25, 0.75]
    totals = node_task_phase_totals(frame, PHASES)
    assert list(totals.columns) == ["PREP", "NODE I/O", "RESTORE"]
    assert totals.index.equals(frame.index)
    np.testing.assert_allclose(totals["PREP"], [10.0, 4.0])
    np.testing.assert_allclose(totals["NODE I/O"], [4.0, 3.0])
    np.testing.assert_allclose(totals["RESTORE"], [0.25, 0.75])
