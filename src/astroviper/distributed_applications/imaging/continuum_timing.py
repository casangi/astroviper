"""Timing-record helpers shared by continuum distributed graph stages."""


def graph_timing_record(
    stage,
    graph_result,
    graph_timings,
    *,
    stage_kind,
    includes_restoration=False,
    timing_key="timing_node_tasks",
):
    """Package one graph's compact task stream without changing its result."""
    if stage_kind not in {"weighting", "imaging_cycle", "final"}:
        raise ValueError(f"Unknown graph timing stage kind {stage_kind!r}.")
    if timing_key not in {"timing_node_tasks", "timing_finalization"}:
        raise ValueError(f"Unknown graph timing frame {timing_key!r}.")
    timing_frame = graph_result.get(timing_key)
    if timing_frame is None:
        return None

    timing_frame = timing_frame.copy(deep=True)
    if "T_image_cube_task" not in timing_frame:
        for total_column in (
            "T_weight_density_node",
            "T_imaging_weight_degrid_node",
        ):
            if total_column in timing_frame:
                timing_frame["T_image_cube_task"] = timing_frame[total_column]
                break
    if "T_make_empty_image" not in timing_frame:
        timing_frame["T_make_empty_image"] = 0.0
    if "T_load" not in timing_frame:
        timing_frame["T_load"] = 0.0
    if "T_write" not in timing_frame:
        write_columns = [
            column for column in timing_frame if column.startswith("T_write_")
        ]
        timing_frame["T_write"] = (
            timing_frame[write_columns].sum(axis=1) if write_columns else 0.0
        )

    timing_frame = timing_frame.drop(
        columns=[
            "time_seconds",
            "cpu_percent",
            "memory_rss_bytes",
            "read_chars",
            "write_chars",
        ],
        errors="ignore",
    )

    compute_duration = graph_timings.get("T_compute_dask_graph")
    if compute_duration is None:
        for duration_key in (
            "T_compute_imaging_weight_graph",
            "T_compute_imaging_weight_degrid_graph",
        ):
            if duration_key in graph_timings:
                compute_duration = graph_timings[duration_key]
                break

    record = {
        "stage": stage,
        "stage_kind": stage_kind,
        "includes_restoration": bool(includes_restoration),
        timing_key: timing_frame,
        "compute_start_unixtime": graph_timings.get("compute_start_unixtime"),
        "compute_end_unixtime": graph_timings.get("compute_end_unixtime"),
        "T_compute": compute_duration,
    }
    timing_model_update = graph_result.get("timing_model_update")
    if timing_model_update is not None:
        record["timing_model_update"] = timing_model_update.copy(deep=True)
    return record


def combine_graph_timing_frames(timing_graphs, key):
    """Concatenate one timing-frame kind from every recorded graph."""
    import pandas as pd

    frames = [record[key] for record in timing_graphs if record.get(key) is not None]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def accumulate_graph_timings(destination, graph_timings):
    """Add duration values while excluding absolute wall-clock anchors."""
    for key, value in graph_timings.items():
        if key in {"compute_start_unixtime", "compute_end_unixtime"}:
            continue
        destination[key] = destination.get(key, 0.0) + value
