"""Every application hands on its thread count and resolves 0; six runs keep their bits.

Each distributed application hands its ``processing_function_threads`` to every
C++ kernel and Python FFT that takes a count, and resolves ``0``. Six runs of
the cube, continuum and simulator applications give the same results with 1 and
3 threads. The anti-alias gridders are left out of the identity: they add into
shared grid cells with atomic adds, so their sums depend on the order of the
threads (``test_anti_alias_grid_cpp.py``). The MT-MFS spatial-Hessian check
runs its FFTs on one thread. Small simulated inputs, 64 x 64 images, the
synchronous scheduler.
"""

import importlib
import importlib.machinery
import importlib.util
import json
import os
import pkgutil
import re
import sys
from functools import partial

import dask
import numpy as np
import pytest
import scipy.fft
import toolviper.utils.logger as logger
import xarray as xr
from xradio.image import make_empty_sky_image, write_image
from xradio.measurement_set import load_processing_set, open_processing_set

import astroviper.distributed_applications as distributed_applications
import astroviper.processing_functions
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

PF = "astroviper.processing_functions"
COUNT = "processing_function_threads"
# The attribute each caller reads, and its functions that take a count.
NATIVE = {
    f"{PF}.imaging.gridders.anti_alias.anti_alias_grid_cpp": (
        "anti_alias_grid",
        "anti_alias_grid_uv_sampling",
        "anti_alias_degrid",
    ),
    f"{PF}.imaging.gridders.direct_fourier_transform.direct_fourier_transform_cpp": (
        "dft_image",
        "dft_image_uv_sampling",
        "dft_predict",
    ),
    f"{PF}.imaging.imaging_setup.imaging_weighting.grid_imaging_weights_cpp": (
        "grid_imaging_weights",
        "degrid_imaging_weights",
    ),
    f"{PF}.imaging.model_update.deconvolvers.hogbom": (
        "clean_cube",
        "clean_cube_many_threads",
    ),
    f"{PF}.imaging.model_update.deconvolvers.aspclean": ("clean", "clean_cube"),
    f"{PF}.imaging.model_update.deconvolvers.mtmfs": ("clean",),
    f"{PF}.image_analysis.psf_gaussian_fit_cpp._psf_gaussian_fit_ext": (
        "fit_psf_beam",
    ),
    f"{PF}.simulation.visibility_kernel_cpp": ("calculate_visibilities",),
}
NO_COUNT = {
    f"{PF}.imaging.model_update.deconvolvers.mtmfs": ("hessian",),
    f"{PF}.imaging.model_update.deconvolvers.aspclean": ("convolve_centered",),
}
EXTENSIONS = {
    f"{PF}.imaging.model_update.deconvolvers.hogbom._hogbom_ext",
    f"{PF}.imaging.model_update.deconvolvers.aspclean._aspclean_ext",
    f"{PF}.imaging.model_update.deconvolvers.mtmfs._mtmfs_ext",
    f"{PF}.imaging.gridders.anti_alias.anti_alias_grid_cpp._anti_alias_grid_ext",
    f"{PF}.imaging.gridders.direct_fourier_transform.direct_fourier_transform_cpp."
    "_direct_fourier_transform_ext",
    f"{PF}.imaging.imaging_setup.imaging_weighting.grid_imaging_weights_cpp."
    "_grid_imaging_weights_ext",
    f"{PF}.image_analysis.psf_gaussian_fit_cpp._psf_gaussian_fit_ext",
    f"{PF}.simulation.visibility_kernel_cpp._visibility_kernel_ext",
}
# Callers that hand the FFT module one thread whatever the count.
ONE_THREAD = {
    f"{PF}.imaging.config_check._spatial_hessian.check_psf_spatial_hessian",
}
FFT = f"{PF}.fft"
FFT_FUNCTIONS = ("fft2", "ifft2", "rfft2", "irfft2", "ifft", "irfft", "set_workers")
HESSIAN_CALLERS = {
    f"{PF}.imaging.config_check.param_check_model_update.check_model_products",
    f"{PF}.imaging.image_continuum_single_field."
    "_mtmfs_principal_residual_for_restoration",
}

DFT = "direct_fourier_transform"
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
CELL = 0.7 * np.pi / (180 * 3600)
IMAGE_SIZE = [64, 64]
SOURCES = [
    (np.array([1.2, 0.1 + 0.05j, 0.1 - 0.05j, 0.8]), (0.0, 0.0)),
    (np.array([0.45, 0.1 - 0.02j, 0.1 + 0.02j, 0.55]), (12.0, -8.0)),
]
CONTROLS = {
    "max_iter": 1000,
    "max_cycles": 2,
    "threshold": 0.0,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.5,
    "max_iter_per_cycle": 30,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}
BRIGGS = {"weighting": "briggs", "robust": 0.5}
NATURAL = {"weighting": "natural"}
MASK = {"sidelobe_threshold": 1.0}

AA_GRID = "anti_alias_grid_cpp.anti_alias_grid"
AA_UV_SAMPLING = "anti_alias_grid_cpp.anti_alias_grid_uv_sampling"
AA_DEGRID = "anti_alias_grid_cpp.anti_alias_degrid"
DFT_PREDICT = "direct_fourier_transform_cpp.dft_predict"
DFT_KERNELS = {
    "direct_fourier_transform_cpp.dft_image",
    "direct_fourier_transform_cpp.dft_image_uv_sampling",
    DFT_PREDICT,
}
WEIGHTS = {
    "grid_imaging_weights_cpp.grid_imaging_weights",
    "grid_imaging_weights_cpp.degrid_imaging_weights",
}
FFTW_IFFT2 = "fft.ifft2[pyfftw]"
FFTW_FFT2 = "fft.fft2[pyfftw]"
GRIDDED_IMAGING = {AA_GRID, AA_UV_SAMPLING, AA_DEGRID, FFTW_IFFT2, FFTW_FFT2}
RESTORE = {"fft.rfft2[pyfftw]", "fft.ifft[pyfftw]", "fft.irfft[pyfftw]"}
MASK_FFTS = {"fft.rfft2[pyfftw]", "fft.irfft2[pyfftw]", "fft.set_workers[pyfftw]"}
SPATIAL_HESSIAN = {"fft.rfft2[pyfftw]"}
MTMFS = {"mtmfs.clean", "mtmfs.hessian"}
VISIBILITY_KERNEL = "visibility_kernel_cpp.calculate_visibilities"
# The entry points each run reaches, and no others; R6's report runs the
# point spread function application (uv sampling and ifft2).
REACHED = {
    "R1": GRIDDED_IMAGING | WEIGHTS | RESTORE | MASK_FFTS | {"hogbom.clean_cube"},
    "R2": GRIDDED_IMAGING | WEIGHTS | RESTORE | MASK_FFTS | MTMFS | SPATIAL_HESSIAN,
    "R3": GRIDDED_IMAGING | {"hogbom.clean_cube"},
    "R4": WEIGHTS | {AA_UV_SAMPLING, FFTW_IFFT2, "_psf_gaussian_fit_ext.fit_psf_beam"},
    "R5": WEIGHTS | {AA_UV_SAMPLING, FFTW_IFFT2},
    "R6": {VISIBILITY_KERNEL, AA_DEGRID, FFTW_FFT2, AA_UV_SAMPLING, FFTW_IFFT2},
    "R7": {"fft.fft2[scipy]", "fft.ifft2[scipy]"},
    "I1": DFT_KERNELS | WEIGHTS | RESTORE | MASK_FFTS | {"hogbom.clean_cube"},
    "I2": DFT_KERNELS | {"hogbom.clean_cube_many_threads"},
    "I3": DFT_KERNELS | {"aspclean.clean_cube"},
    "I4": DFT_KERNELS | MTMFS | SPATIAL_HESSIAN,
    "I5": {VISIBILITY_KERNEL, DFT_PREDICT},
    "I6": {VISIBILITY_KERNEL, AA_DEGRID, FFTW_FFT2},
}
IDENTITY_RUNS = ("I1", "I2", "I3", "I4", "I5", "I6")


def _signature_parameters(function):
    """Parameter names of the first signature line of a pybind11 docstring."""
    for line in (function.__doc__ or "").splitlines():
        line = re.sub(r"^\d+\. ", "", line.strip())
        if re.match(r"^[A-Za-z_]\w*\(.*\) -> ", line):
            body = line[line.index("(") + 1 : line.rindex(") -> ")]
            names, depth, current = [], 0, ""
            for character in body:
                if character in "[(":
                    depth += 1
                elif character in "])":
                    depth -= 1
                if character == "," and depth == 0:
                    names.append(current)
                    current = ""
                else:
                    current += character
            return [name.split(":")[0].strip() for name in [*names, current]]
    return []


def _count_position(function):
    """Index of the count in the first signature line, or None."""
    names = _signature_parameters(function)
    return names.index(COUNT) if COUNT in names else None


class _Recorder:
    """Records (entry point, count, caller) of every call of an entry point.

    The callers read each native function at the attribute of ``NATIVE`` and
    ``NO_COUNT``, and the FFT module's functions at ``FFT``, at call time; an
    FFT is labelled with its ``fft_backend``. An FFT called without a count
    records ``scipy.fft.get_workers()``, the count of the module's
    ``set_workers``.
    """

    def __init__(self, monkeypatch):
        self.calls = []
        for module_name, functions in [*NATIVE.items(), *NO_COUNT.items()]:
            module = importlib.import_module(module_name)
            for name in functions:
                real = getattr(module, name)
                position = (
                    _count_position(real)
                    if name in NATIVE.get(module_name, ())
                    else None
                )
                label = f"{module_name.rsplit('.', 1)[-1]}.{name}"
                monkeypatch.setattr(module, name, self._native(label, real, position))
        module = importlib.import_module(FFT)
        for name in FFT_FUNCTIONS:
            monkeypatch.setattr(module, name, self._fft(name, getattr(module, name)))

    @staticmethod
    def _caller():
        frame = sys._getframe(2)
        return f"{frame.f_globals.get('__name__')}.{frame.f_code.co_name}"

    def _native(self, label, real, position):
        def wrapper(*args, **kwargs):
            if position is None:
                count = None
            elif COUNT in kwargs:
                count = kwargs[COUNT]
            else:
                count = args[position] if len(args) > position else "default"
            self.calls.append((label, count, self._caller()))
            return real(*args, **kwargs)

        return wrapper

    def _fft(self, name, real):
        def wrapper(*args, **kwargs):
            if name == "set_workers":
                count = args[0] if args else kwargs[COUNT]
            else:
                count = kwargs.get(COUNT)
            if count is None:
                count = scipy.fft.get_workers()
            label = f"fft.{name}[{kwargs['fft_backend']}]"
            self.calls.append((label, count, self._caller()))
            return real(*args, **kwargs)

        return wrapper

    def astroviper_calls(self):
        return [call for call in self.calls if call[2].startswith("astroviper.")]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """A processing set (6 ALMA antennas, 4 times, 2 channels, XX and YY, two
    point sources), a 2-channel sky model image and two feather images."""
    root = tmp_path_factory.mktemp("processing_function_threads")
    ps_store = str(root / "two_hands.ps.zarr")
    _simulate(ps_store, 1)

    model = make_empty_sky_image(
        phase_center=PHASE_CENTER,
        image_size=IMAGE_SIZE,
        cell_size=[-CELL, CELL],
        frequency_coords=[100e9, 100.5e9],
        pol_coords=["I"],
        time_coords=[0],
    )
    sky = np.zeros((1, 2, 1, *IMAGE_SIZE))
    sky[0, :, 0, 40, 20] = 2.0
    model["SKY"] = (("time", "frequency", "polarization", "l", "m"), sky)
    model_store = str(root / "model.img.zarr")
    write_image(model, model_store, out_format="zarr", overwrite=True)

    feather_stores = {}
    cell = 1.0e-3
    lm = (np.arange(32) - 16) * cell
    ll, mm = np.meshgrid(lm, lm, indexing="ij")
    for name, beam in (("highres", 0.01), ("lowres", 0.04)):
        image = xr.Dataset(
            {
                "SKY": (
                    ["time", "frequency", "polarization", "l", "m"],
                    np.exp(-(ll**2 + mm**2) / (2 * (5 * cell) ** 2))[
                        None, None, None
                    ].astype(np.float32),
                ),
                "BEAM_FIT_PARAMS_SKY": (
                    ["time", "frequency", "polarization", "beam_params_label"],
                    np.array([[[[beam, beam, 0.0]]]]),
                ),
            },
            coords={
                "time": [0.0],
                "frequency": [1.0e9],
                "polarization": ["I"],
                "l": lm,
                "m": lm,
                "beam_params_label": ["major", "minor", "pa"],
            },
        )
        image["BEAM_FIT_PARAMS_SKY"].attrs["units"] = "rad"
        feather_stores[name] = str(root / f"{name}.img.zarr")
        image.to_zarr(feather_stores[name], mode="w")
    return {"ps": ps_store, "model": model_store, **feather_stores}


def _simulate(ps_store, threads, **kwargs):
    antenna_xds = read_telescope_layout(
        "alma.cycle8.1", antenna_selection=list(range(6))
    )
    sky_components = [
        {
            "kind": "point",
            "flux": flux,
            "ra_dec": PHASE_CENTER
            + np.deg2rad(np.array(offset) / 3600.0)
            / np.array([np.cos(PHASE_CENTER[1]), 1.0]),
        }
        for flux, offset in SOURCES
    ]
    with dask.config.set(scheduler="synchronous"):
        return distributed_applications.simulation.simulate_processing_set(
            ps_store=ps_store,
            antenna_xds=antenna_xds,
            time_params={
                "time_start": "2019-10-03T19:00:00.000",
                "time_delta": 600.0,
                "n_samples": 4,
            },
            frequency_params={
                "freq_start": 100e9,
                "freq_delta": 0.5e9,
                "n_channels": 2,
                "channel_width": 2e6,
            },
            polarization=["XX", "YY"],
            sky_components=sky_components,
            phase_center_ra_dec=PHASE_CENTER[None, :],
            beam_models=[airy_disk_model("alma")],
            beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
            n_time_chunks=1,
            n_frequency_chunks=1,
            processing_function_threads=threads,
            overwrite=True,
            **kwargs,
        )


def _image_params(ps_store, n_taylor_terms=None):
    frequency = np.asarray(open_processing_set(ps_store).xr_ps.get_freq_axis().values)
    params = {
        "image_size": IMAGE_SIZE,
        "cell_size": np.array([-CELL, CELL]),
        "phase_direction": PHASE_CENTER,
        "frequency_coords": frequency,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }
    if n_taylor_terms is not None:
        params["n_taylor_terms"] = n_taylor_terms
        params["reference_frequency"] = float(np.mean(frequency))
    return params


def _cube(inputs, threads, directory, weighting=NATURAL, **kwargs):
    store = str(directory / "cube.img.zarr")
    with dask.config.set(scheduler="synchronous"):
        result = distributed_applications.imaging.image_cube_single_field(
            ps_store=inputs["ps"],
            image_store=store,
            image_params=_image_params(inputs["ps"]),
            imaging_weights_params=weighting,
            cycle_control_params={**CONTROLS, "primary_beam_limit": 0.0},
            image_data_variables_keep=[
                "sky_residual",
                "sky_model",
                "point_spread_function",
                "primary_beam",
            ],
            processing_set_data_group_name="base",
            single_precision_image=False,
            n_mapping_parallelism={"frequency": 1},
            processing_function_threads=threads,
            overwrite=True,
            **kwargs,
        )
    return result, store


def _continuum(
    inputs, threads, directory, n_taylor_terms=2, weighting=NATURAL, **kwargs
):
    store = str(directory / "continuum.img.zarr")
    with dask.config.set(scheduler="synchronous"):
        result = distributed_applications.imaging.image_continuum_single_field(
            ps_store=inputs["ps"],
            image_store=store,
            image_params=_image_params(inputs["ps"], n_taylor_terms),
            imaging_weights_params=weighting,
            cycle_control_params=CONTROLS,
            primary_beam_limit=0.0,
            image_data_variables_keep=[
                "sky_residual",
                "sky_model",
                "point_spread_function",
            ],
            processing_set_data_group_name="base",
            single_precision_image=False,
            n_mapping_parallelism={"frequency": 1},
            processing_function_threads=threads,
            overwrite=True,
            **kwargs,
        )
    return result, store


def _psf(inputs, threads, directory, **kwargs):
    store = str(directory / "psf.img.zarr")
    with dask.config.set(scheduler="synchronous"):
        result = (
            distributed_applications.imaging.point_spread_function_cube_single_field(
                ps_store=inputs["ps"],
                image_params=_image_params(inputs["ps"]),
                imaging_weights_params=BRIGGS,
                image_store=store,
                processing_set_data_group_name="base",
                processing_function_threads=threads,
                overwrite=True,
                **kwargs,
            )
        )
    return result, store


def _delphi(inputs, threads, directory):
    parameters_file = str(directory / "delphi.parameters.json")
    with dask.config.set(scheduler="synchronous"):
        result = distributed_applications.imaging.image_cube_single_field_delphi(
            inputs["ps"],
            str(directory / "delphi.img.zarr"),
            processing_set_data_group_name="base",
            n_mapping_parallelism={"frequency": 1},
            calculate_parameters_only=True,
            parameters_file=parameters_file,
            processing_function_threads=threads,
            overwrite=True,
        )
    return result, parameters_file


def _simulator(inputs, threads, directory, **kwargs):
    store = str(directory / "simulated.ps.zarr")
    return _simulate(store, threads, sky_model_image=inputs["model"], **kwargs), store


def _feather(inputs, threads, directory):
    store = str(directory / "feather.img.zarr")
    with dask.config.set(scheduler="synchronous"):
        result = distributed_applications.imaging.feather(
            outim={"name": store, "overwrite": True},
            highres=inputs["highres"],
            lowres=inputs["lowres"],
            processing_function_threads=threads,
            fft_backend="scipy",
        )
    return result, store


def _components(inputs, threads, directory):
    store = str(directory / "simulated.ps.zarr")
    return _simulate(store, threads), store


GRIDDED_PREDICTION = {
    "sky_model_image_predict": {"predict": "gridded"},
    "uvw_params": {"w_term": False},
}
RUNS = {
    "R1": partial(
        _cube,
        weighting=BRIGGS,
        auto_mask_params=MASK,
        restore=True,
        restoration_output="both",
    ),
    "R2": partial(
        _continuum,
        weighting=BRIGGS,
        deconvolver="mtmfs",
        auto_mask_params=MASK,
        restore=True,
    ),
    "R3": partial(
        _continuum, n_taylor_terms=1, continuum_mode="mvc", deconvolver="hogbom"
    ),
    "R4": partial(_psf, psf_fitting_method="casa"),
    "R5": _delphi,
    "R6": partial(_simulator, report="html", **GRIDDED_PREDICTION),
    "R7": _feather,
    "I1": partial(
        _cube,
        weighting=BRIGGS,
        gridder=DFT,
        auto_mask_params=MASK,
        restore=True,
        restoration_output="both",
    ),
    "I2": partial(_cube, gridder=DFT, deconvolver="hogbom_many_threads"),
    "I3": partial(_cube, gridder=DFT, deconvolver="asp"),
    "I4": partial(_continuum, gridder=DFT, deconvolver="mtmfs"),
    "I5": partial(_simulator, sky_model_image_predict={"predict": "dft"}),
    "I6": partial(_simulator, **GRIDDED_PREDICTION),
}
ZERO_RUNS = {
    "cube": partial(_cube, gridder=DFT),
    "continuum": partial(_continuum, gridder=DFT, deconvolver="mtmfs"),
    "point_spread_function": _psf,
    "delphi": _delphi,
    "feather": _feather,
    "simulator": _components,
}


def _recorded_run(run, inputs, threads, directory):
    """``run`` under a recorder: its result, output path, records and error."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        recorder = _Recorder(monkeypatch)
        try:
            result, output = run(inputs, threads, directory)
        except Exception as error:  # reported by the tests
            return {"error": error, "calls": recorder.astroviper_calls()}
    return {
        "error": None,
        "result": result,
        "output": output,
        "calls": recorder.astroviper_calls(),
    }


@pytest.fixture(scope="module")
def three_thread_runs(inputs, tmp_path_factory):
    return {
        name: _recorded_run(run, inputs, 3, tmp_path_factory.mktemp(name))
        for name, run in RUNS.items()
    }


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "O":
        return all(_same(x, y) for x, y in zip(a.ravel(), b.ravel(), strict=True))
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


def _record(result):
    """The model update record without the timings."""
    return {
        (str(key), name): value
        for key, entry in result.get("deconvolution", {}).items()
        for name, value in entry.items()
        if not str(name).startswith("t_")
    }


def test_every_binding_with_a_count_is_listed():
    """The extension modules are the eight of CMakeLists.txt, and their
    functions whose signature names the count are those of NATIVE."""
    found = set()
    for info in pkgutil.walk_packages(
        astroviper.processing_functions.__path__, f"{PF}."
    ):
        spec = importlib.util.find_spec(info.name)
        if spec is not None and (spec.origin or "").endswith(
            tuple(importlib.machinery.EXTENSION_SUFFIXES)
        ):
            found.add(info.name)
    assert found == EXTENSIONS
    with_count = {
        function
        for name in EXTENSIONS
        for function in vars(importlib.import_module(name)).values()
        if callable(function) and _count_position(function) is not None
    }
    listed = {
        getattr(importlib.import_module(module_name), name)
        for module_name, names in NATIVE.items()
        for name in names
    }
    assert with_count == listed


@pytest.mark.parametrize("run", sorted(RUNS))
def test_every_entry_point_gets_the_count(run, three_thread_runs):
    """A run at 3 threads hands 3 to every entry point it reaches, but in the
    calls of ONE_THREAD's callers (at most 1), and reaches exactly those of its
    row of REACHED."""
    outcome = three_thread_runs[run]
    assert outcome["error"] is None, repr(outcome["error"])
    calls = outcome["calls"]
    assert {label for label, _, _ in calls} == REACHED[run]
    counts = {
        (label, count)
        for label, count, caller in calls
        if label != "mtmfs.hessian" and caller not in ONE_THREAD
    }
    assert {count for _, count in counts} == {3}, sorted(counts, key=str)
    assert {count for _, count, caller in calls if caller in ONE_THREAD} <= {1}
    hessian_callers = {caller for label, _, caller in calls if label == "mtmfs.hessian"}
    assert hessian_callers <= HESSIAN_CALLERS
    if run == "R6":
        assert os.path.isfile(outcome["result"]["report"])


def test_every_count_consumer_is_reached(three_thread_runs):
    """Every native function with a count is reached by a run, but Asp's plane
    clean, which no application calls."""
    reached = {
        label
        for outcome in three_thread_runs.values()
        for label, _, _ in outcome["calls"]
    }
    listed = {
        f"{module_name.rsplit('.', 1)[-1]}.{name}"
        for module_name, names in NATIVE.items()
        for name in names
    }
    assert listed - reached <= {"aspclean.clean"}


def test_the_count_is_recorded(three_thread_runs):
    """The cube, continuum and simulator results record 3, and so does
    Delphi's parameters file."""
    for run in ("R1", "R2", "R3", "R6", "I1", "I2", "I3", "I4", "I5", "I6"):
        resources = three_thread_runs[run]["result"]["compute_resources"]
        assert resources[COUNT] == 3, run
    with open(three_thread_runs["R5"]["output"]) as parameters_file:
        parameters = json.load(parameters_file)
    assert parameters["image_cube_single_field"][COUNT] == 3
    assert parameters["provenance"]["delphi_inputs"][COUNT] == 3


@pytest.mark.parametrize("application", sorted(ZERO_RUNS))
def test_zero_is_resolved(application, inputs, tmp_path, monkeypatch):
    """With 3 CPUs and one worker thread, 0 becomes 3 for every entry point, but
    in the calls of ONE_THREAD's callers, and in the records."""
    monkeypatch.setattr("dask.system.CPU_COUNT", 3)
    outcome = _recorded_run(ZERO_RUNS[application], inputs, 0, tmp_path)
    assert outcome["error"] is None, repr(outcome["error"])
    counts = {
        count
        for label, count, caller in outcome["calls"]
        if label != "mtmfs.hessian" and caller not in ONE_THREAD
    }
    assert counts == {3}
    if application in ("cube", "continuum", "simulator"):
        assert outcome["result"]["compute_resources"][COUNT] == 3
    if application == "delphi":
        with open(outcome["output"]) as parameters_file:
            parameters = json.load(parameters_file)
        assert parameters["image_cube_single_field"][COUNT] == 3
        assert parameters["provenance"]["delphi_inputs"][COUNT] == 3


def test_oversubscription_warns_once(inputs, tmp_path, monkeypatch):
    """4 threads on 3 CPUs: one warning for one application call."""
    monkeypatch.setattr("dask.system.CPU_COUNT", 3)
    warnings = []
    real_warning = logger.warning

    def warning(message, *args, **kwargs):
        if str(message).startswith(f"{COUNT}="):
            warnings.append(message)
        return real_warning(message, *args, **kwargs)

    monkeypatch.setattr(logger, "warning", warning)
    _psf(inputs, 4, tmp_path)
    assert len(warnings) == 1


@pytest.mark.parametrize("run", IDENTITY_RUNS)
def test_one_and_three_threads_give_the_same_results(
    run, inputs, tmp_path, three_thread_runs
):
    """1 thread gives the arrays and the model update record of 3 threads.

    The record is compared too: a thread-dependent defect in the residual at
    the peak shows only there, because the next residual update recomputes
    the stored residual.
    """
    three = three_thread_runs[run]
    assert three["error"] is None, repr(three["error"])
    one_result, one_output = RUNS[run](inputs, 1, tmp_path)
    if run in ("I5", "I6"):
        sets = [load_processing_set(path) for path in (one_output, three["output"])]
        assert sorted(sets[0]) == sorted(sets[1])
        for name in sets[0]:
            for variable in ("VISIBILITY", "UVW", "WEIGHT", "FLAG"):
                values = [ps[name].ds[variable].values for ps in sets]
                assert _same(*values), f"{name} {variable}"
        return
    with xr.open_zarr(one_output) as a, xr.open_zarr(three["output"]) as b:
        assert sorted(a.data_vars) == sorted(b.data_vars)
        for name in a.data_vars:
            assert _same(a[name].values, b[name].values), name
    one_record, three_record = _record(one_result), _record(three["result"])
    assert one_record.keys() == three_record.keys() and one_record
    for key, value in one_record.items():
        assert _same(value, three_record[key]), key
