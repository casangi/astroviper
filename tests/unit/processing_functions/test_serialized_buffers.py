"""External-array kernels consume Dask frames in place across real processes."""

import copy
import logging
import os

import dask
import numpy as np
import pytest
from distributed import Client, LocalCluster, WorkerPlugin
from distributed.protocol.serialize import deserialize_bytes, serialize_bytes


def _case(kind):
    """Large enough for pickle protocol-5 frames rather than inline array copies."""
    single = kind.endswith("32")
    real = np.float32 if single else np.float64
    complex_dtype = np.complex64 if single else np.complex128
    frequency = np.array([1.0e9])
    indices = np.zeros(1, dtype=np.int64)
    common = dict(
        uvw=np.zeros((1, 1, 3)),
        frequency_coord=frequency,
        frequency_map=indices,
        time_map=indices,
        pol_map=indices,
        processing_function_threads=2,
    )
    if kind.startswith("degrid") or kind.startswith("grid"):
        kwargs = dict(
            common,
            grid=np.ones((1, 1, 1, 128, 128), dtype=complex_dtype),
            vis_data=np.zeros((1, 1, 1, 1), dtype=np.complex128),
            cgk_1D=np.ones(8),
            n_uv=np.array([128, 128]),
            delta_lm=np.array([1.0e-4, 1.0e-4]),
            support=3,
            oversampling=2,
            interpolation=0,
        )
        if kind.startswith("grid"):
            kwargs.update(
                normalization=np.zeros((1, 1, 1)), weight=np.ones((1, 1, 1, 1))
            )
            kwargs["vis_data"][...] = 1.0 + 0j
        return kwargs
    if kind.startswith("dft"):
        return dict(
            common,
            model=np.ones((1, 1, 1, 128, 128), dtype=real),
            vis_data=np.zeros((1, 1, 1, 1), dtype=complex_dtype),
            l=np.arange(128) * 1.0e-5,
            m=np.arange(128) * 1.0e-5,
        )
    if kind.startswith("mtmfs"):
        residual = np.zeros((1, 128, 128), dtype=real)
        residual[0, 64, 64] = 1.0
        return dict(
            residual=residual,
            psf=residual.copy(),
            model=np.zeros_like(residual),
            max_iter_per_cycle_updated=2,
        )
    if kind == "simulation":
        return dict(
            visibility=np.zeros((1, 1, 16384, 1), dtype=np.complex128),
            uvw=np.zeros((1, 1, 3)),
            antenna1=indices,
            antenna2=indices,
            frequency=np.full(16384, 1.0e9),
            polarization_index=indices,
            point_source_flux=np.ones((1, 1, 1, 4), dtype=np.complex128),
            k_vector=np.zeros((1, 1, 3)),
            inverse_n=np.ones((1, 1)),
            point_source_ra_dec=np.zeros((1, 1, 2)),
            pointing_ra_dec=np.zeros((1, 1, 2)),
            beam_model_map=indices,
            beam_models=[
                dict(
                    kind="analytic",
                    func="none",
                    max_rad_1GHz=1.0,
                    dish_diameter=1.0,
                    blockage_diameter=0.0,
                )
            ],
            parallactic_angle=np.zeros(1),
            mueller_selection=indices,
            processing_function_threads=2,
        )
    raise AssertionError(kind)


def _run(kind, kwargs):
    if kind.startswith("degrid"):
        from astroviper.processing_functions.imaging.gridders.anti_alias.anti_alias_grid_cpp import (
            anti_alias_degrid,
        )

        anti_alias_degrid(**kwargs)
    elif kind.startswith("grid"):
        from astroviper.processing_functions.imaging.gridders.anti_alias.anti_alias_grid_cpp import (
            anti_alias_grid,
        )

        anti_alias_grid(**kwargs)
    elif kind.startswith("dft"):
        from astroviper.processing_functions.imaging.gridders.direct_fourier_transform.direct_fourier_transform_cpp import (
            dft_predict,
        )

        dft_predict(**kwargs)
    elif kind.startswith("mtmfs"):
        from astroviper.processing_functions.imaging.model_update.deconvolvers.mtmfs._mtmfs_ext import (
            clean,
        )

        clean(**kwargs)
    elif kind == "simulation":
        from astroviper.processing_functions.simulation.visibility_kernel_cpp import (
            calculate_visibilities,
        )

        calculate_visibilities(**kwargs)


def _pack_case(kind):
    # Spill files are read as mutable bytearrays. Immutable bytes would trigger
    # NumPy's writable-copy fallback and hide the alignment regression.
    return os.getpid(), bytearray(serialize_bytes(_case(kind)))


def _consume_case(kind, packed):
    sender_pid, frames = packed
    kwargs = deserialize_bytes(frames)
    unaligned = [
        name
        for name, value in kwargs.items()
        if isinstance(value, np.ndarray) and not value.flags.aligned
    ]
    assert unaligned, "The transfer did not exercise an unaligned Dask frame"
    reference = copy.deepcopy(_case(kind))
    _run(kind, reference)
    _run(kind, kwargs)
    for name, value in reference.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(kwargs[name], value)
    return sender_pid, os.getpid(), unaligned


@pytest.fixture(scope="module")
def process_client():
    with dask.config.set(
        {
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.pause": False,
            "distributed.worker.memory.terminate": False,
        }
    ):
        with (
            LocalCluster(
                n_workers=2,
                threads_per_worker=1,
                processes=True,
                dashboard_address=None,
                silence_logs=logging.ERROR,
            ) as cluster,
            Client(cluster) as client,
        ):
            yield client


@pytest.mark.parametrize(
    "kind",
    [
        "grid32",
        "grid64",
        "degrid32",
        "degrid64",
        "dft32",
        "dft64",
        "mtmfs32",
        "mtmfs64",
        "simulation",
    ],
)
def test_kernels_consume_unaligned_frames_transferred_between_processes(
    process_client, kind
):
    workers = sorted(process_client.scheduler_info()["workers"])
    packed = process_client.submit(
        _pack_case, kind, workers=[workers[0]], allow_other_workers=False, pure=False
    )
    future = process_client.submit(
        _consume_case,
        kind,
        packed,
        workers=[workers[1]],
        allow_other_workers=False,
        pure=False,
    )
    sender, receiver, unaligned = future.result(timeout=60)
    assert sender != receiver
    assert receiver != os.getpid()
    assert unaligned
    packed.release()
    future.release()


def _unaligned_arrays(value, seen=None):
    """Inspect only already materialized buffers, without computing lazy data."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return int(not value.flags.aligned)
    if isinstance(value, dict):
        return sum(_unaligned_arrays(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_unaligned_arrays(v, seen) for v in value)
    import xarray as xr

    if isinstance(value, xr.Dataset):
        return sum(_unaligned_arrays(v.data, seen) for v in value.variables.values())
    if isinstance(value, xr.DataArray):
        return _unaligned_arrays(value.data, seen)
    return 0


class _ObserveUnspill(WorkerPlugin):
    def setup(self, worker):
        worker.alignment_restores = 0
        worker.alignment_unaligned = 0
        if not hasattr(worker.data, "slow"):
            return
        slow = worker.data.slow.data
        original = slow.load

        def load(frames):
            value = original(frames)
            worker.alignment_restores += 1
            worker.alignment_unaligned += _unaligned_arrays(value)
            return value

        slow.load = load


def _spill_counts(dask_worker):
    return dask_worker.alignment_restores, dask_worker.alignment_unaligned


@pytest.mark.parametrize("mode", ["mfs", "mvc", "simulation"])
def test_process_workflows_match_after_forced_spilling(tmp_path, mode):
    from xradio.image import load_image
    from xradio.measurement_set import open_processing_set

    from astroviper.distributed_applications.imaging import image_continuum_single_field
    from tests.unit.distributed_applications.imaging.test_image_cube_single_field import (
        PC,
        _simulate_point_source,
    )

    # Generate once; both imaging runs read the identical observation.
    input_store = (
        _simulate_point_source(tmp_path, n_channels=4) if mode != "simulation" else None
    )
    products = []
    for spill in (False, True):
        config = {
            "distributed.worker.memory.target": 1.0e-9 if spill else False,
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.pause": False,
            "distributed.worker.memory.terminate": False,
            "distributed.worker.memory.spill-compression": False,
        }
        run_dir = tmp_path / str(spill)
        run_dir.mkdir()
        with (
            dask.config.set(config),
            LocalCluster(
                n_workers=2,
                threads_per_worker=1,
                processes=True,
                dashboard_address=None,
                memory_limit="1GB",
                silence_logs=logging.ERROR,
            ) as cluster,
            Client(cluster) as client,
        ):
            client.register_plugin(_ObserveUnspill(), name="observe-unaligned-restore")
            if mode == "simulation":
                store = _simulate_point_source(run_dir, n_channels=4)
                ps = open_processing_set(store)
                arrays = {
                    name: child.VISIBILITY.values.copy() for name, child in ps.items()
                }
            else:
                ps = open_processing_set(input_store)
                frequencies = ps.xr_ps.get_freq_axis().values.copy()
                output = str(run_dir / "image.img.zarr")
                image_continuum_single_field(
                    ps_store=input_store,
                    image_store=output,
                    image_params={
                        "image_size": [128, 128],
                        "cell_size": np.deg2rad(np.array([-8.0, 8.0]) / 3600),
                        "phase_direction": PC,
                        "frequency_coords": frequencies,
                        "polarization_coords": ["I", "V"],
                        "time_coords": [0.0],
                        "fft_padding": 1.2,
                        "reference_frequency": float(np.mean(frequencies)),
                        "n_taylor_terms": 2,
                    },
                    imaging_weights_params={
                        "weighting": "briggs",
                        "robust": 0.5,
                        "weighting_scope": "global",
                        "casa_weighting_implementation": True,
                    },
                    cycle_control_params={
                        "max_iter": 3,
                        "max_cycles": 3,
                        "threshold": 0.0,
                        "gain": 0.1,
                        "psf_sidelobe_factor": 1.5,
                        "max_iter_per_cycle": 1,
                        "min_psf_fraction": 0.05,
                        "max_psf_fraction": 0.8,
                    },
                    processing_set_data_group_name="base",
                    instrument_polarization_basis="circular",
                    n_mapping_parallelism={"frequency": 2},
                    continuum_mode=mode,
                    deconvolver="mtmfs",
                    restore=False,
                    single_precision_image=False,
                    processing_function_threads=1,
                    weight_memory_mode="in_memory",
                    visibility_memory_mode="in_memory",
                    wideband_primary_beam_memory_mode="in_memory",
                )
                image = load_image(output)
                arrays = {
                    name: image[name].values.copy()
                    for name in ("SKY_MODEL", "SKY_RESIDUAL", "POINT_SPREAD_FUNCTION")
                }
                assert np.max(np.abs(arrays["SKY_MODEL"])) > 0
            counts = client.run(_spill_counts)
            if spill:
                assert sum(v[0] for v in counts.values()) > 0, (
                    "No task data were restored from spill"
                )
                if mode != "simulation":
                    assert sum(v[1] for v in counts.values()) > 0, (
                        "No restored array was unaligned"
                    )
            assert all(np.isfinite(array).all() for array in arrays.values())
            products.append(arrays)
    assert products[0].keys() == products[1].keys()
    for name in products[0]:
        np.testing.assert_allclose(
            products[1][name], products[0][name], rtol=1e-12, atol=1e-12, err_msg=name
        )


@pytest.mark.parametrize("kind", ["polynomial", "jones_image"])
def test_simulation_nested_beam_arrays_support_unaligned_storage(kind, unaligned_array):
    kwargs = _case("simulation")
    kwargs["frequency"] = np.array([1.0e9, 1.1e9])
    kwargs["visibility"] = np.zeros((1, 1, 2, 1), dtype=np.complex128)
    if kind == "polynomial":
        model = dict(
            kind=kind,
            max_rad_1GHz=1.0,
            frequency=np.array([1.0e9]),
            coefficients=np.array([[1.0, 0.0]]),
            n_sample=0,
        )
    else:
        model = dict(
            kind=kind,
            max_rad_1GHz=1.0,
            jones=np.ones((1, 1, 4, 4, 4), dtype=np.complex128),
            parallactic_angle=np.array([0.0]),
            frequency=np.array([1.0e9]),
            polarization_index=np.arange(4, dtype=np.int64),
            cell_size_l=1.0e-3,
            cell_size_m=1.0e-3,
        )
    kwargs["beam_models"] = [model]
    reference = copy.deepcopy(kwargs)
    _run("simulation", reference)
    for name, value in model.items():
        if not isinstance(value, np.ndarray):
            continue
        for offset in range(1, value.dtype.alignment):
            trial = copy.deepcopy(kwargs)
            trial["beam_models"][0][name] = unaligned_array(value, offset)
            _run("simulation", trial)
            np.testing.assert_array_equal(trial["visibility"], reference["visibility"])
            np.testing.assert_array_equal(trial["beam_models"][0][name], value)
            assert trial["beam_models"][0][name].base[0] == 0xA5
            assert trial["beam_models"][0][name].base[-1] == 0xA5


def test_in_place_outputs_reject_conversion_instead_of_updating_temporaries():
    kwargs = _case("grid64")
    before = kwargs["grid"].copy()
    kwargs["normalization"] = np.zeros((1, 1, 1), dtype=np.float32)
    with pytest.raises((RuntimeError, ValueError), match="normalization.*float64"):
        _run("grid64", kwargs)
    np.testing.assert_array_equal(kwargs["grid"], before)
    from astroviper.processing_functions.image_analysis.psf_gaussian_fit_cpp._psf_gaussian_fit_ext import (
        fit_psf_beam,
    )

    psf = np.zeros((1, 8, 8))
    psf[0, 4, 4] = 1.0
    beams = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="beams.*float64"):
        fit_psf_beam(psf, 1.0e-4, 1.0e-4, 0.35, beams)
    np.testing.assert_array_equal(beams, 0.0)


@pytest.mark.parametrize("kernel", ["hogbom", "aspclean"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cube_deconvolvers_consume_unaligned_storage(kernel, dtype, unaligned_array):
    from astroviper.processing_functions.imaging.model_update.deconvolvers import (
        aspclean,
        hogbom,
    )

    clean = (hogbom if kernel == "hogbom" else aspclean).clean_cube
    residual = np.zeros((1, 2, 1, 32, 32), dtype=dtype)
    residual[..., 16, 16] = 1
    kwargs = dict(
        residual_cube=residual,
        psf_cube=residual.copy(),
        model_cube=np.zeros_like(residual),
        mask_cube=np.ones_like(residual, dtype=bool if kernel == "hogbom" else dtype),
        threshold_per_cycle=np.zeros((1, 2, 1), dtype=dtype),
        max_iter_per_cycle_updated=np.full((1, 2, 1), 2, dtype=np.int32),
        processing_function_threads=2,
    )
    if kernel == "aspclean":
        for old, new in (
            ("residual_cube", "residual"),
            ("psf_cube", "psf"),
            ("model_cube", "model"),
            ("mask_cube", "mask"),
        ):
            kwargs[new] = kwargs.pop(old)
        kwargs.update(asp_psf_width=1.0, asp_largest_scale=0)
        kwargs["threshold_per_cycle"] = np.zeros((1, 2, 1), dtype=np.float64)
    reference = copy.deepcopy(kwargs)
    clean(**reference)
    for name, value in kwargs.items():
        if not isinstance(value, np.ndarray):
            continue
        for offset in range(1, value.dtype.alignment):
            trial = copy.deepcopy(kwargs)
            trial[name] = unaligned_array(value, offset)
            clean(**trial)
            for key, expected in reference.items():
                if isinstance(expected, np.ndarray):
                    np.testing.assert_array_equal(trial[key], expected)
            assert trial[name].base[0] == 0xA5
            assert trial[name].base[-1] == 0xA5


@pytest.mark.parametrize("kind", ["grid32", "grid64"])
def test_unaligned_grid_accumulation_is_thread_safe(kind, unaligned_array):
    kwargs = _case(kind)
    kwargs["uvw"] = np.zeros((1, 128, 3))
    kwargs["vis_data"] = np.ones((1, 128, 1, 1), dtype=np.complex128)
    kwargs["weight"] = np.ones((1, 128, 1, 1))
    kwargs["processing_function_threads"] = 4
    reference = copy.deepcopy(kwargs)
    _run(kind, reference)
    for offset in range(1, kwargs["grid"].dtype.alignment):
        trial = copy.deepcopy(kwargs)
        for name in ("grid", "normalization"):
            trial[name] = unaligned_array(trial[name], offset)
        _run(kind, trial)
        for name in ("grid", "normalization"):
            np.testing.assert_array_equal(trial[name], reference[name])
            assert trial[name].base[0] == 0xA5
            assert trial[name].base[-1] == 0xA5
