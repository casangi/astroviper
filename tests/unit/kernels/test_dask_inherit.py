"""Dask workers inherit the driver's kernel selection.

The selection travels only through the environment: a driver started with
``ASTROVIPER_KERNELS`` set starts a real ToolVIPER ``local_client`` whose
worker processes import the shims with it set. Each worker makes one hooked
call, and its own trace file must show the driver's selection.
"""

import json
import socket

from tests.unit.kernels.conftest import ALL_REF, TINY_HOGBOM

WORKERS = 2

DRIVER = f"""
import json, os
from astroviper._kernels import selected
from toolviper.dask.client import local_client
{TINY_HOGBOM}

def in_worker():
    import os

    from astroviper._kernels import selected

    tiny_hogbom()
    return os.getpid(), dict(selected())


client = local_client(
    cores={WORKERS}, memory_limit="1GiB", dask_local_dir=os.environ["DASK_DIR"],
    dashboard_address=None,
)
try:
    workers = list(client.run(in_worker).values())
finally:
    cluster = client.cluster
    client.close()
    cluster.close()
print("RESULT", json.dumps({{"driver": [os.getpid(), dict(selected())], "workers": workers}}))
"""


def test_workers_run_the_driver_selection(run_python, tmp_path):
    trace = tmp_path / "trace"
    out = run_python(
        "-c",
        DRIVER,
        ASTROVIPER_KERNELS=ALL_REF,
        ASTROVIPER_KERNELS_TRACE=str(trace),
        DASK_DIR=str(tmp_path / "dask"),
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    driver_pid, selection = result["driver"]
    assert selection == {"dft": "ref", "hogbom": "ref", "asp": "ref"}
    assert len(result["workers"]) == WORKERS
    for pid, worker_selection in result["workers"]:
        assert pid != driver_pid
        assert worker_selection == selection
        lines = (trace / f"kernels-{socket.gethostname()}-{pid}.jsonl").read_text()
        records = [json.loads(line) for line in lines.splitlines()]
        assert [(r["kernel"], r["entry"], r["impl"]) for r in records] == [
            ("hogbom", "clean_cube", selection["hogbom"])
        ]
