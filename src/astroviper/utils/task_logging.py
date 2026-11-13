"""Shared logging helpers for distributed node tasks."""


def write_task_kill_switch_log(
    timing_df, task_total_time, threshold, image_store, task_id, hostname
):
    """Write an overrunning node task's timing breakdown next to its output.

    Parameters
    ----------
    timing_df : pandas.DataFrame
        Per-stage timing information for the node task.
    task_total_time : float
        Total task wall time in seconds.
    threshold : float
        Configured kill-switch threshold in seconds.
    image_store : str
        Output image-store path; its parent receives the log.
    task_id : int or str
        Graph task identifier.
    hostname : str
        Host on which the task ran.

    Returns
    -------
    str
        Written path, or a diagnostic placeholder if writing failed.
    """
    import os
    import time

    try:
        parent = os.path.dirname(os.path.abspath(image_store)) or "."
        stamp = time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        path = os.path.join(
            parent, f"KILL_SWITCH_task_{task_id}_{hostname}_{pid}_{stamp}.log"
        )
        lines = [
            "TASK TIME KILL SWITCH TRIPPED",
            f"task_id: {task_id}",
            f"hostname: {hostname}",
            f"pid: {pid}",
            f"task wall time: {task_total_time:.2f} s   (threshold: {threshold} s)",
            f"image_store: {image_store}",
            "",
            "per-task timing breakdown:",
            timing_df.to_string(index=False),
        ]
        with open(path, "w") as log_file:
            log_file.write("\n".join(str(line) for line in lines) + "\n")
        return path
    except Exception as exc:
        # Logging must never hide the kill-switch exception that prompted it.
        return f"(failed to write kill-switch log: {exc!r})"
