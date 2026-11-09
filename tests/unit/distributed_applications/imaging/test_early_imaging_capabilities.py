"""Reject unsupported configurations before I/O and retain supported workflows."""

from copy import deepcopy

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging import (
    image_continuum_single_field,
    image_cube_single_field,
)
from tests.unit.distributed_applications.imaging import test_continuum as helpers
from tests.unit.distributed_applications.imaging.test_output_destination import options


@pytest.mark.parametrize(
    "workflow,case",
    [
        (workflow, case)
        for workflow in ("cube", "mfs", "mvc")
        for case in ("missing_padding", "default_padding", "two_times")
    ]
    + [
        (workflow, key)
        for workflow in ("cube", "mvc")
        for key in ("visibility", "uv_sampling")
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_unsupported_configuration_fails_before_any_io(
    tmp_path, monkeypatch, workflow, case, existing
):
    driver, config = options(workflow)
    if case == "missing_padding":
        config["image_params"].pop("fft_padding")
    elif case == "default_padding":
        config["image_params"]["fft_padding"] = 1.0
    elif case == "two_times":
        config["image_params"]["time_coords"] = [0.0, 1.0]
    else:
        config["image_data_variables_keep"].append(case)
    before = deepcopy(config)
    output = tmp_path / "output.img.zarr"
    if existing:
        output.mkdir()
        (output / "marker").write_bytes(b"keep existing output")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Malformed configuration reached filesystem or input preflight"
        )

    monkeypatch.setattr("astroviper.utils.io.check_image_output_destination", forbidden)
    monkeypatch.setattr("xradio.measurement_set.open_processing_set", forbidden)
    monkeypatch.setattr(
        "astroviper.node_tasks.imaging.config_check.preflight_imaging_inputs", forbidden
    )
    match = (
        "fft_padding"
        if "padding" in case
        else "time plane"
        if case == "two_times"
        else "UV-grid"
    )
    with pytest.raises((KeyError, ValueError), match=match):
        driver(str(tmp_path / "input.ps.zarr"), str(output), **config)
    assert config == before
    if existing:
        assert (output / "marker").read_bytes() == b"keep existing output"
        assert list(output.iterdir()) == [output / "marker"]
    else:
        assert not output.exists()


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return helpers.wideband_store.__wrapped__(tmp_path_factory)


def run(
    store,
    output,
    workflow,
    *,
    threshold=0.0,
    cycles=0,
    chunks=2,
    kernel=None,
    padding=1.2,
    kept=(),
):
    params = helpers._tw_hydra_image_params(open_processing_set(str(store)))
    params.update(n_taylor_terms=1 if workflow == "cube" else 2, fft_padding=padding)
    controls = dict(
        max_iter=2,
        max_cycles=cycles,
        threshold=threshold,
        gain=0.1,
        psf_sidelobe_factor=1.0,
        max_iter_per_cycle=1,
        min_psf_fraction=0.05,
        max_psf_fraction=0.8,
    )
    config = dict(
        image_params=params,
        imaging_weights_params={"weighting": "natural"},
        cycle_control_params=controls,
        processing_set_data_group_name="base",
        n_mapping_parallelism={"frequency": chunks},
        single_precision_image=False,
        processing_function_threads=1,
        restore=False,
        deconvolver="hogbom" if workflow == "cube" else "mtmfs",
        image_data_variables_keep=[
            "sky_residual",
            "point_spread_function",
            "sky_model",
            *kept,
        ],
        gridding_kernel_params=kernel,
    )
    if workflow != "cube":
        config["continuum_mode"] = workflow
    with dask.config.set(scheduler="synchronous"):
        result = (
            image_cube_single_field
            if workflow == "cube"
            else image_continuum_single_field
        )(str(store), str(output), **config)
    assert controls["max_cycles"] == cycles
    assert controls["threshold"] == threshold
    with xr.open_zarr(output) as image:
        return result, image.load()


@pytest.mark.parametrize("workflow", ["cube", "mfs", "mvc"])
def test_casa_kernel_keeps_padding_one_supported(source, tmp_path, workflow):
    _, image = run(
        source,
        tmp_path / "casa.img.zarr",
        workflow,
        kernel={"type": "prolate_spheroidal_casa"},
        padding=1.0,
    )
    assert np.isfinite(image.SKY_RESIDUAL).all()


@pytest.mark.parametrize("key", ["visibility", "uv_sampling"])
def test_mfs_publishes_requested_taylor_grids(source, tmp_path, key):
    _, image = run(source, tmp_path / "grid.img.zarr", "mfs", kept=[key])
    values = image[key.upper()]
    assert {"u", "v"} <= set(values.dims)
    assert np.isfinite(values).all()
    assert np.any(values.values != 0)


@pytest.mark.parametrize("workflow", ["mfs", "mvc"])
@pytest.mark.parametrize("threshold", [1e-3, 0.0])
def test_automatic_limits_resolve_and_are_partition_invariant(
    source, tmp_path, monkeypatch, workflow, threshold
):
    from astroviper.processing_functions.imaging import weight_statistics

    original = weight_statistics.continuum_correlation_weight_sums
    calls = []

    def record(*args, **kwargs):
        assert threshold == 0, (
            "Positive thresholds must not scan weights for sensitivity"
        )
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(weight_statistics, "continuum_correlation_weight_sums", record)
    limits = []
    for chunks in (1, 2):
        result, image = run(
            source,
            tmp_path / f"auto_{chunks}.img.zarr",
            workflow,
            cycles="auto",
            threshold=threshold,
            chunks=chunks,
        )
        controller = result["controller"]
        assert isinstance(controller.max_cycles, int)
        assert controller.max_cycles > 0
        assert any(limit is not None for limit in controller.max_cycles_auto.values())
        limits.append(controller.max_cycles_auto)
        assert np.isfinite(image.SKY_RESIDUAL).all()
    assert limits[0] == limits[1]
    assert len(calls) == (3 if threshold == 0 else 0)


def test_standalone_processing_can_keep_working_grids():
    from astroviper.processing_functions.imaging.config_check import (
        CUBE_SINGLE_FIELD,
        check_driver_workflow,
    )

    check_driver_workflow(
        {"image_data_variables_keep": ["visibility", "uv_sampling"]},
        mode=CUBE_SINGLE_FIELD,
    )
