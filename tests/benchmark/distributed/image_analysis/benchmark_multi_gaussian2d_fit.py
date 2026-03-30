# benchmark for fit_multi_gaussian2d implementations on a fixed synthetic cube.
# Run with --scheduler=distributed to see Dask overheads and scaling behavior,
# and with --scheduler=synchronous for stable baselines.
# Dave Mehringer

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import time
import numpy as np
import xarray as xr
import dask
import dask.array as da
import matplotlib.pyplot as plt
from astroviper.utils.plotting import generate_plot

from astroviper.distributed.image_analysis.multi_gaussian2d_fit import (
    fit_multi_gaussian2d,
    plot_components,
)
from astroviper.distributed.model.component_models import make_gauss2d

from astroviper.distributed.image_analysis.selection import select_mask
from dask.distributed import Client

from astroviper.utils.plotting import generate_plot

# Reproducibility
rng = np.random.default_rng(1234)


def make_scene_via_component_models(
    nx: int,
    ny: int,
    components: list[dict],
    *,
    offset: float = 0.1,
    noise_std: float = 0.02,
    seed: int | None = None,
    coords: bool = True,
    angle: str = "math",  # "math" | "pa" | "auto" — same semantics as astroviper's model
    x_world: tuple[float, float] = (0.0, 1.0),
    y_world: tuple[float, float] = (0.0, 1.0),
) -> xr.DataArray:
    """
    Build a synthetic image using astroviper.model.component_models.(make_gaussian|make_gauss2d).

    components: list of dicts with keys:
      {"amp"/"amplitude","x0","y0","sigma_x","sigma_y","theta"}.
    """
    rng = np.random.default_rng(seed)

    nchan = 1 if isinstance(components[0], dict) else len(components)

    # coords
    if coords:
        x0, x1 = x_world
        y0, y1 = y_world
        x = np.linspace(x0, x1, nx, dtype=float)
        y = np.linspace(y0, y1, ny, dtype=float)
    else:
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)
    # accumulate model in float64
    if nchan == 1:
        components = [components.copy()]
    z = np.zeros((nx, ny, nchan), dtype=float)
    print("components", components, type(components))
    for i in range(nchan):
        plane_components = components[i]
        for c in plane_components:
            print("plane component", c, type(c))
            amp = float(c.get("amp", c.get("amplitude")))
            x0c = float(c["x0"])
            y0c = float(c["y0"])
            fwhm_x = float(c["fwhm_major"])
            fwhm_y = float(c["fwhm_minor"])
            thc = float(c.get("theta", 0.0))

            data = z[..., i]
            z[:, :, i] = make_gauss2d(
                data=z[..., i],
                a=fwhm_x,
                b=fwhm_y,
                theta=thc,
                x0=x0c,
                y0=y0c,
                peak=amp,
                x_coord="x",
                y_coord="y",
                coords={"x": x, "y": y},
                add=True,
                angle=angle,
            )
    if nchan == 1:
        z = np.squeeze(z, axis=2)

    z += float(offset)
    if noise_std > 0:
        z += rng.normal(scale=noise_std, size=z.shape)

    dims = ("x", "y") if nchan == 1 else ("x", "y", "z")
    print("shape z", z.shape, "dims", dims)
    xda = xr.DataArray(z, dims=dims)
    print("xda shape", xda.shape)
    if coords:
        vals = (
            dict(x=x, y=y)
            if nchan == 1
            else dict(x=x, y=y, z=np.array((2, 4), dtype=float))
        )
        xda = xda.assign_coords(vals)
    return xda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fit_multi_gaussian2d implementations on a fixed synthetic cube."
    )
    parser.add_argument(
        "--impl",
        choices=("current", "parammap"),
        default="current",
        help="Implementation to benchmark. Run the script separately for each implementation.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated fit/compute runs on the same fixed cube.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("synchronous", "distributed"),
        default="synchronous",
        help="Dask execution mode. Use synchronous first for stable baselines.",
    )
    parser.add_argument(
        "--n-planes",
        type=int,
        default=5,
        help="Number of time planes in the synthetic cube.",
    )
    parser.add_argument(
        "--chunk-time",
        type=int,
        default=None,
        help="Explicit time-axis chunk size for the Dask-backed cube. Overrides --planes-per-chunk.",
    )
    parser.add_argument(
        "--planes-per-chunk",
        type=int,
        default=2,
        help="Number of time planes assigned to each Dask chunk/task.",
    )
    parser.add_argument(
        "--client-threads",
        type=int,
        default=1,
        help="Threads per worker when --scheduler=distributed.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of Dask workers when --scheduler=distributed.",
    )
    parser.add_argument(
        "--cube-seed",
        type=int,
        default=20260326,
        help="Seed used to generate the benchmark cube so separate runs see identical input data.",
    )
    parser.add_argument(
        "--scene-seed",
        type=int,
        default=1234,
        help="Seed used to generate the base noisy benchmark image so separate runs start from identical scenes.",
    )
    return parser.parse_args()


def build_fixed_cube(
    base_plane: xr.DataArray,
    *,
    n_planes: int,
    chunk_time: int | None,
    planes_per_chunk: int,
    cube_seed: int,
) -> xr.DataArray:
    """
    Build one fixed Dask-backed cube so repeated runs and separate processes use
    identical inputs.
    """
    cube_rng = np.random.default_rng(cube_seed)
    time_chunk = int(chunk_time) if chunk_time is not None else int(planes_per_chunk)
    arr = base_plane + 0.01 * cube_rng.normal(size=base_plane.shape)
    print("type arr:", type(arr), flush=True)
    print("type arr.data", type(arr.data), flush=True)
    cube_np = np.stack(
        [
            (base_plane + 0.01 * cube_rng.normal(size=base_plane.shape)).values
            for _ in range(n_planes)
        ],
        axis=0,
    )
    return xr.DataArray(
        da.from_array(
            cube_np,
            chunks=(time_chunk, base_plane.shape[0], base_plane.shape[1]),
        ),
        dims=("time", "x", "y"),
        coords={"x": base_plane.coords["x"], "y": base_plane.coords["y"]},
    )


def run_benchmark(
    code,
    cube_da: xr.DataArray,
    init_arr,
    *,
    repeats: int,
    scheduler: str,
) -> None:
    parms = dict(
        data=cube_da, n_components=2, initial_guesses=init_arr, dims=("x", "y")
    )
    total_times: list[float] = []
    for run_idx in range(1, repeats + 1):
        print(f"run {run_idx}", flush=True)
        t0 = time.time()
        ds_dask = code(**parms)
        t1 = time.time()
        print("time to create the DAG", t1 - t0, flush=True)
        with dask.config.set(scheduler=scheduler):
            metric = float(ds_dask["amplitude"].mean().compute())
        t2 = time.time()
        print("time to compute()", t2 - t1, flush=True)
        print("metric", metric, flush=True)
        total_times.append(t2 - t1)
        print(flush=True)
    mean_compute = float(np.mean(total_times))
    std_compute = float(np.std(total_times))
    print(f"compute mean {mean_compute:.6f}s std {std_compute:.6f}s", flush=True)


def main():
    args = parse_args()
    client = None
    print("args.scheduler", args.scheduler, flush=True)
    if args.scheduler == "distributed":
        print("spin up distributed client", flush=True)
        client = Client(
            n_workers=args.n_workers, threads_per_worker=args.client_threads
        )

    nx = 131
    ny = 121
    comp_1g = [
        dict(amp=5.0, x0=40, y0=-20, fwhm_major=20.0, fwhm_minor=10.0, theta=0.4)
    ]
    comp_2g = comp_1g.copy()
    comp_2g.append(
        dict(amp=7.0, x0=-50, y0=10, fwhm_major=8.0, fwhm_minor=6.0, theta=1)
    )

    data_2g_noise = make_scene_via_component_models(
        ny=ny,
        nx=nx,
        components=comp_2g,
        offset=0.0,
        noise_std=0.5,
        seed=args.scene_seed,
        x_world=(-nx + 1, nx - 1),
        y_world=(-ny + 1, ny - 1),
    )

    import astroviper.distributed.image_analysis.multi_gaussian2d_fit_parammap as parammap_fit

    init_arr = [
        [5.5, 41.0, -22.5, 15.0, 8.0, 0.3],
        [8.5, -56.0, 8.5, 6.0, 4.5, 0.9],
    ]
    cube_da = build_fixed_cube(
        data_2g_noise,
        n_planes=args.n_planes,
        chunk_time=args.chunk_time,
        planes_per_chunk=args.planes_per_chunk,
        cube_seed=args.cube_seed,
    )
    print(args.impl, flush=True)
    print("shape", data_2g_noise.shape, flush=True)
    print("cube shape", cube_da.shape, flush=True)
    print("cube chunks", cube_da.chunks, flush=True)

    code = (
        fit_multi_gaussian2d
        if args.impl == "current"
        else parammap_fit.fit_multi_gaussian2d
    )
    run_benchmark(
        code,
        cube_da,
        init_arr,
        repeats=args.repeats,
        scheduler=args.scheduler,
    )
    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
