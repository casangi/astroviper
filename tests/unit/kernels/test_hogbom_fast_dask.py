"""The fast Hogbom kernel in Dask workers: a 2-worker ToolVIPER ``local_client``
under ``ASTROVIPER_KERNELS=hogbom=fast`` with the trace, in a subprocess (one
cluster per process).

A ``clean_cube_many_threads`` call through the hooked shim is submitted to each
worker with cubes made in the driver. Dask deserialises the arrays it ships as
views of the frames it received, at arbitrary offsets (misaligned), which the
fast kernel reads through aligned copies; each worker's residual, model and
results are the driver's reference call's, bit for bit, and every Hogbom call
of the workers' trace files takes path ``fast``.
"""

import json
import socket

import pytest

pytest.importorskip(
    "astroviper._kernels.fast.hogbom._hogbom_fast_ext",
    reason="fast Hogbom module not built",
)

DRIVER = """
import json, os
import numpy as np
from toolviper.dask.client import local_client

from astroviper._kernels import impl


def clean(arguments):
    import os
    import numpy as np
    from astroviper.processing_functions.imaging.deconvolvers import hogbom
    aligned = bool(arguments["residual_cube"].flags.aligned)
    result = hogbom.clean_cube_many_threads(**arguments)
    out = [arguments["residual_cube"].tobytes(), arguments["model_cube"].tobytes()]
    out += [np.asarray(value).tobytes() for value in result.values()]
    return out, aligned, os.getpid()


rng = np.random.default_rng(0)
n = 96
y, x = np.mgrid[0:n, 0:n] - n // 2
psf = np.exp(-(x * x + y * y) / 8.0)
residual = np.zeros((1, 1, 2, n, n))
for p in range(2):
    for _ in range(5):
        dy, dx = rng.integers(-30, 30, 2)
        residual[0, 0, p] += rng.uniform(0.5, 2) * np.roll(psf, (dy, dx), (0, 1))
arguments = dict(
    residual_cube=residual,
    psf_cube=np.ascontiguousarray(np.broadcast_to(psf, residual.shape)),
    model_cube=np.zeros_like(residual),
    mask_cube=np.ascontiguousarray(np.broadcast_to(x * x + y * y < 40**2, residual.shape)),
    max_iter_per_cycle_updated=np.full((1, 1, 2), 200, np.int32),
    gain=0.2,
    threshold_per_cycle=np.zeros((1, 1, 2)),
    processing_function_threads=2,
)
reference = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in arguments.items()}
result = impl("hogbom", "clean_cube_many_threads", "ref")(**reference)
expected = [reference["residual_cube"].tobytes(), reference["model_cube"].tobytes()]
expected += [np.asarray(value).tobytes() for value in result.values()]

work = os.environ["WORK_DIR"]
client = local_client(
    cores=2, memory_limit="1GiB", dask_local_dir=os.path.join(work, "dask"),
    dashboard_address=None,
)
try:
    shipped = []
    for address in sorted(client.scheduler_info()["workers"]):
        future = client.submit(
            clean, arguments, workers=[address], allow_other_workers=False, pure=False,
        )
        out, aligned, pid = future.result()
        shipped.append({"pid": pid, "aligned": aligned, "bitwise": out == expected})
finally:
    cluster = client.cluster
    client.close()
    cluster.close()
print("RESULT", json.dumps({"driver": os.getpid(), "shipped": shipped}))
"""


def test_dask_workers_take_the_fast_path(run_python, tmp_path):
    trace = tmp_path / "trace"
    out = run_python(
        "-c",
        DRIVER,
        ASTROVIPER_KERNELS="hogbom=fast",
        ASTROVIPER_KERNELS_TRACE=str(trace),
        WORK_DIR=str(tmp_path),
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    shipped = result["shipped"]
    assert len({s["pid"] for s in shipped}) == len(shipped) == 2
    assert all(s["bitwise"] for s in shipped)
    host = socket.gethostname()
    records = {}  # pid -> Hogbom records of that process
    for path in trace.glob(f"kernels-{host}-*.jsonl"):
        pid = int(path.stem.rpartition("-")[2])
        lines = path.read_text().splitlines()
        records[pid] = [r for r in map(json.loads, lines) if r["kernel"] == "hogbom"]
    workers = {pid: r for pid, r in records.items() if pid != result["driver"]}
    assert {s["pid"] for s in shipped} <= workers.keys()
    calls = [record for worker in workers.values() for record in worker]
    assert len(calls) == len(shipped)
    assert {(r["impl"], r["path"], r["error"]) for r in calls} == {
        ("fast", "fast", None)
    }
