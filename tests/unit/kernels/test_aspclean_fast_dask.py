"""The fast Asp in Dask workers: a 2-worker ``LocalCluster`` (processes)
started under ``ASTROVIPER_KERNELS=asp=fast`` with the trace, in a subprocess
(one cluster per process).

A two-channel 128 x 128 cube made in the driver goes to each worker, which
cleans it with the hooked ``clean_cube``: Dask rebuilds the arrays it ships as
views of the frames it received, at any byte offset, so the PSF and the mask
may arrive misaligned (they are read in place). Each worker's residual, model
and result are the bits of the reference run in the driver, and every Asp
record of the workers' trace files took the fast path.
"""

import json
import socket

import pytest

pytest.importorskip(
    "astroviper._kernels.fast.aspclean._aspclean_fast_ext",
    reason="fast Asp module not built",
)

DRIVER = """
import json, os
import numpy as np
from distributed import Client, LocalCluster

from astroviper._kernels import impl

OPTIONS = {"gain": 0.3, "processing_function_threads": 2, "asp_residual_components": True,
           "asp_scale_significance": True, "record_iteration_history": True}


def in_worker(residual, psf, model, mask, threshold, limit):
    import os
    import numpy as np
    from astroviper.processing_functions.imaging.deconvolvers import aspclean

    aligned = {name: bool(a.flags.aligned) for name, a in
               (("residual", residual), ("psf", psf), ("model", model), ("mask", mask))}
    residual, model = np.array(residual), np.array(model)  # writeable
    result = aspclean.clean_cube(residual=residual, psf=psf, model=model, mask=mask,
                                 threshold_per_cycle=threshold,
                                 max_iter_per_cycle_updated=limit, **OPTIONS)
    return residual, model, result, aligned, os.getpid()


def same(a, b):
    if isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in b)
    if isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype, a.shape) == (b.dtype, b.shape) and a.tobytes() == b.tobytes()


n = 128
yy, xx = np.mgrid[0:n, 0:n]
rng = np.random.default_rng(128)
psf = np.exp(-((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / 8.0)
planes = []
for k in range(2):
    sky = np.exp(-((xx - 60 - 4 * k) ** 2 + (yy - 66) ** 2) / (2 * 9.0**2))
    sky[58, 70 - k] += 2.0
    dirty = np.real(np.fft.ifft2(np.fft.fft2(sky) * np.fft.fft2(np.fft.ifftshift(psf))))
    planes.append(dirty + 1e-3 * rng.standard_normal((n, n)))
residual = np.stack(planes).reshape(1, 2, 1, n, n)
cube_psf = np.ascontiguousarray(np.broadcast_to(psf, (1, 2, 1, n, n)))
mask = np.zeros((1, 2, 1, n, n))
mask[..., 30:100, 30:100] = 1.0
model = np.zeros_like(residual)
threshold = np.full((1, 2, 1), 1e-2)
limit = np.full((1, 2, 1), 60, np.int32)

expected_residual, expected_model = residual.copy(), model.copy()
expected = impl("asp", "clean_cube", "ref")(
    residual=expected_residual, psf=cube_psf, model=expected_model, mask=mask,
    threshold_per_cycle=threshold, max_iter_per_cycle_updated=limit, **OPTIONS)

work = os.environ["WORK_DIR"]
cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True,
                       dashboard_address=None, local_directory=os.path.join(work, "dask"),
                       memory_limit="2GiB")
client = Client(cluster)
try:
    shipped = []
    for address in sorted(client.scheduler_info()["workers"]):
        future = client.submit(in_worker, residual, cube_psf, model, mask, threshold, limit,
                               workers=[address], allow_other_workers=False, pure=False)
        got_residual, got_model, got, aligned, pid = future.result()
        bitwise = (got_residual.tobytes() == expected_residual.tobytes()
                   and got_model.tobytes() == expected_model.tobytes() and same(got, expected))
        shipped.append({"pid": pid, "bitwise": bitwise, "aligned": aligned})
finally:
    client.close()
    cluster.close()
iterations = int(np.asarray(expected["iterations_performed"]).sum())
print("RESULT", json.dumps({"driver": os.getpid(), "shipped": shipped, "iterations": iterations}))
"""


def test_dask_workers_run_the_fast_kernel(run_python, tmp_path):
    trace = tmp_path / "trace"
    out = run_python(
        "-c",
        DRIVER,
        ASTROVIPER_KERNELS="asp=fast",
        ASTROVIPER_KERNELS_TRACE=str(trace),
        WORK_DIR=str(tmp_path),
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    shipped = result["shipped"]
    assert len({s["pid"] for s in shipped}) == len(shipped) == 2
    assert all(s["bitwise"] for s in shipped)
    assert result["iterations"] > 2
    host = socket.gethostname()
    records = {}  # pid -> Asp records of that process
    for path in trace.glob(f"kernels-{host}-*.jsonl"):
        pid = int(path.stem.rpartition("-")[2])
        lines = path.read_text().splitlines()
        records[pid] = [r for r in map(json.loads, lines) if r["kernel"] == "asp"]
    assert result["driver"] not in records
    assert {s["pid"] for s in shipped} == records.keys()
    calls = [record for worker in records.values() for record in worker]
    assert len(calls) == 2
    assert {(r["entry"], r["impl"], r["path"], r["error"]) for r in calls} == {
        ("clean_cube", "fast", "fast", None)
    }
