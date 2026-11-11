"""Dask workers inherit the driver's kernel selection, through the environment.

A driver started with ``ASTROVIPER_KERNELS`` set starts a real ToolVIPER
``local_client``: each worker process sees the driver's exact value and has
hooked shims, a hooked call made on its event loop (``client.run``) and one
pickled to a worker thread as a task (``client.submit`` of the shim entry)
both run the wrapper, and the worker's own trace file shows them. The
documented hazard is shown too: a cluster started before the variable is set
has unhooked workers, which run the reference untraced although the driver is
hooked. One cluster per process, so each case is its own subprocess.
"""

import json
import socket

from tests.unit.kernels.conftest import TINY_HOGBOM

WORKERS = 2

#: A valid value that differs from every other selection the tests use.
VALUE = "hogbom = REF, dft=ref"

DRIVER = f"""
import json, os, sys
from toolviper.dask.client import local_client
{TINY_HOGBOM}

def in_worker():
    import os
    import threading

    from astroviper.processing_functions.imaging.deconvolvers import hogbom

    tiny_hogbom()
    return {{
        "pid": os.getpid(),
        "value": os.environ.get("ASTROVIPER_KERNELS"),
        "hooked": hasattr(hogbom.clean_cube, "__wrapped__"),
        "thread": threading.get_ident(),
    }}


client = local_client(
    cores={WORKERS}, memory_limit="1GiB", dask_local_dir=os.environ["DASK_DIR"],
    dashboard_address=None,
)
try:
    if sys.argv[1] == "late":  # set only after the workers started
        os.environ["ASTROVIPER_KERNELS"] = os.environ["LATE_VALUE"]
    from astroviper.processing_functions.imaging.deconvolvers import hogbom

    import numpy as np

    residual = np.zeros((1, 1, 1, 8, 8))
    residual[..., 4, 4] = 1.0
    task = client.submit(
        hogbom.clean_cube,
        residual_cube=residual,
        psf_cube=residual.copy(),
        model_cube=np.zeros_like(residual),
        max_iter_per_cycle_updated=np.full((1, 1, 1), 5, np.int32),
        gain=0.5,
        threshold_per_cycle=np.zeros((1, 1, 1)),
        pure=False,
    )
    iterations = int(task.result()["iterations_performed"].sum())
    workers = list(client.run(in_worker).values())
    task_worker = next(iter(client.who_has(task)[task.key]))
    task_pid = client.run(os.getpid, workers=[task_worker])[task_worker]
finally:
    cluster = client.cluster
    client.close()
    cluster.close()
print("RESULT", json.dumps({{
    "driver": os.getpid(),
    "driver_hooked": hasattr(hogbom.clean_cube, "__wrapped__"),
    "iterations": iterations,
    "task_pid": task_pid,
    "workers": workers,
}}))
"""


def _run(run_python, tmp_path, mode, **env):
    trace = tmp_path / "trace"
    out = run_python(
        "-c",
        DRIVER,
        mode,
        ASTROVIPER_KERNELS_TRACE=str(trace),
        DASK_DIR=str(tmp_path / "dask"),
        **env,
    )
    return json.loads(out.rpartition("RESULT ")[2]), trace


def _worker_records(trace, pid):
    path = trace / f"kernels-{socket.gethostname()}-{pid}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_workers_run_the_driver_selection(run_python, tmp_path):
    result, trace = _run(run_python, tmp_path, "inherit", ASTROVIPER_KERNELS=VALUE)
    assert result["driver_hooked"] is True
    assert result["iterations"] > 0
    assert len(result["workers"]) == WORKERS
    for worker in result["workers"]:
        assert worker["pid"] != result["driver"]
        assert (worker["value"], worker["hooked"]) == (VALUE, True)
        records = _worker_records(trace, worker["pid"])
        on_loop = [r for r in records if r["thread"] == worker["thread"]]
        assert [(r["kernel"], r["entry"], r["impl"], r["error"]) for r in on_loop] == [
            ("hogbom", "clean_cube", "ref", None)
        ]
    # The submitted wrapper ran in a task thread of the worker that holds it.
    (loop_thread,) = [
        w["thread"] for w in result["workers"] if w["pid"] == result["task_pid"]
    ]
    tasks = [
        r
        for r in _worker_records(trace, result["task_pid"])
        if r["thread"] != loop_thread
    ]
    assert [(r["entry"], r["impl"], r["error"]) for r in tasks] == [
        ("clean_cube", "ref", None)
    ]


def test_cluster_started_before_the_variable_runs_unhooked(run_python, tmp_path):
    result, trace = _run(run_python, tmp_path, "late", LATE_VALUE=VALUE)
    assert result["driver_hooked"] is True
    assert result["iterations"] > 0
    assert len(result["workers"]) == WORKERS
    for worker in result["workers"]:
        assert (worker["value"], worker["hooked"]) == (None, False)
    worker_pids = {w["pid"] for w in result["workers"]}
    assert result["task_pid"] in worker_pids
    traced = {p.name for p in trace.glob("*.jsonl")} if trace.exists() else set()
    host = socket.gethostname()
    assert traced.isdisjoint({f"kernels-{host}-{pid}.jsonl" for pid in worker_pids})
