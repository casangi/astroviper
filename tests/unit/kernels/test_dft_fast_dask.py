"""Path S in Dask workers: a 2-worker ToolVIPER ``local_client`` under
``ASTROVIPER_KERNELS=dft=fast`` with the trace, in a subprocess (one cluster
per process).

* A predict submitted to each worker with a 128 x 128 model made in the
  driver: Dask deserialises the arrays it ships as views of the frames it
  received, at arbitrary offsets (misaligned), which path S reads through
  aligned copies; each worker's result is the driver's, bit for bit.
* A 64 x 64 sky-image simulate with ``uvw_params={"w_term": False}``: its DFT
  calls run in the workers on the image they load.

Every DFT call of the workers' trace files takes path S, none declines.
"""

import json
import socket

import pytest

pytest.importorskip(
    "astroviper._kernels.fast.dft._dft_fast_ext", reason="fast DFT module not built"
)

DRIVER = """
import json, os
import numpy as np
from astropy.coordinates import SkyCoord
from toolviper.dask.client import local_client
from xradio.image import make_empty_sky_image, write_image

import astroviper.distributed_applications as distributed_applications
from astroviper.processing_functions.imaging.gridders import direct_fourier_transform_cpp as shim
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout


def predict(model, uvw, frequency, maps, l, m):
    import os
    import numpy as np
    from astroviper.processing_functions.imaging.gridders import (
        direct_fourier_transform_cpp as shim,
    )
    vis = np.zeros(uvw.shape[:2] + (frequency.size, maps[2].size), np.complex128)
    shim.dft_predict(model, vis, uvw, frequency, *maps, l, m)
    return vis, bool(model.flags.aligned), os.getpid()


rng = np.random.default_rng(0)
n = 128
l = np.array([(i - n // 2) * -2e-5 for i in range(n)])
m = np.array([(j - n // 2) * 2e-5 for j in range(n)])
model = np.ascontiguousarray(np.broadcast_to(rng.uniform(0, 1e-3, (n, n)), (1, 1, 2, n, n)))
uvw = rng.uniform(-1e3, 1e3, (3, 40, 3))
uvw[..., 2] = 0.0
frequency = np.array([1e11])
maps = (np.zeros(1, np.int64), np.zeros(3, np.int64), np.arange(2, dtype=np.int64))
expected = np.zeros((3, 40, 1, 2), np.complex128)
shim.dft_predict(model, expected, uvw, frequency, *maps, l, m)

center = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
phase_center = np.array([[center.ra.rad, center.dec.rad]])
arcsec = np.pi / (180 * 3600)
image = make_empty_sky_image(
    phase_center=phase_center[0], image_size=[64, 64],
    cell_size=[-10 * arcsec, 10 * arcsec], frequency_coords=[3.2e9],
    pol_coords=["I"], time_coords=[0],
)
sky = np.zeros((1, 1, 1, 64, 64))
sky[0, 0, 0, 24:40, 20:44] = rng.uniform(0, 1e-3, (16, 24))
image["SKY"] = (("time", "frequency", "polarization", "l", "m"), sky)
work = os.environ["WORK_DIR"]
write_image(image, os.path.join(work, "model.img.zarr"), out_format="zarr", overwrite=True)

client = local_client(
    cores=2, memory_limit="1GiB", dask_local_dir=os.path.join(work, "dask"),
    dashboard_address=None,
)
try:
    shipped = []
    for address in sorted(client.scheduler_info()["workers"]):
        future = client.submit(
            predict, model, uvw, frequency, maps, l, m,
            workers=[address], allow_other_workers=False, pure=False,
        )
        vis, aligned, pid = future.result()
        shipped.append({"pid": pid, "aligned": aligned,
                        "bitwise": vis.tobytes() == expected.tobytes()})
    distributed_applications.simulation.simulate_processing_set(
        ps_store=os.path.join(work, "sim.ps.zarr"),
        antenna_xds=read_telescope_layout("vla.d", antenna_selection=list(range(8))),
        time_params={"time_start": "2019-10-03T19:00:00.000", "time_delta": 1800.0,
                     "n_samples": 4},
        frequency_params={"freq_start": 3e9, "freq_delta": 0.2e9, "n_channels": 3,
                          "channel_width": 1e7, "spectral_window_name": "SBand"},
        polarization=["RR", "LL"],
        point_source_flux=None,
        point_source_ra_dec=None,
        sky_model_image=os.path.join(work, "model.img.zarr"),
        phase_center_ra_dec=phase_center,
        beam_models=[airy_disk_model("vla_nvss")],
        beam_model_map=np.zeros(8, int),
        n_time_chunks=2,
        n_frequency_chunks=3,
        uvw_params={"w_term": False},
        noise_params=None,
        overwrite=True,
    )
finally:
    cluster = client.cluster
    client.close()
    cluster.close()
print("RESULT", json.dumps({"driver": os.getpid(), "shipped": shipped}))
"""


def test_dask_workers_take_path_s(run_python, tmp_path):
    trace = tmp_path / "trace"
    out = run_python(
        "-c",
        DRIVER,
        ASTROVIPER_KERNELS="dft=fast",
        ASTROVIPER_KERNELS_TRACE=str(trace),
        WORK_DIR=str(tmp_path),
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    shipped = result["shipped"]
    assert len({s["pid"] for s in shipped}) == len(shipped) == 2
    assert all(s["bitwise"] for s in shipped)
    host = socket.gethostname()
    records = {}  # pid -> DFT records of that process
    for path in trace.glob(f"kernels-{host}-*.jsonl"):
        pid = int(path.stem.rpartition("-")[2])
        lines = path.read_text().splitlines()
        records[pid] = [r for r in map(json.loads, lines) if r["kernel"] == "dft"]
    workers = {pid: r for pid, r in records.items() if pid != result["driver"]}
    assert {s["pid"] for s in shipped} <= workers.keys()
    calls = [record for worker in workers.values() for record in worker]
    assert {(r["impl"], r["path"], r["error"]) for r in calls} == {("fast", "S", None)}
    assert len(calls) > len(shipped)  # the simulate's calls ran in the workers too
