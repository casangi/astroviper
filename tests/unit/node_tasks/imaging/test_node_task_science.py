"""The imaging node task with the real science function, on a small simulated
field (one visibility channel, one point source):

* an image channel that no visibility channel maps onto is imaged from views
  of the whole chunk, whether it comes before or after a mapped channel, and
  every channel of the task comes out exactly as when it is imaged in a task
  of its own (the loaded chunk is never modified by a channel);
* everything a node task creates dies by reference counting: with the
  garbage collector off, a collection after the task finds no object in a
  reference cycle (xradio AGENT.md: cycle-free death proven by a test).
"""

from __future__ import annotations

import copy
import functools
import gc

import dask
import numpy as np
import pytest
import xarray as xr

from tests.unit.node_tasks.imaging.cycle_test_utils import cyclic_garbage

ARCSEC = np.pi / (180 * 3600)
F0 = 100e9  # the visibility channel
# image channel spacing: the visibility channel maps onto the image channel at
# F0 only (the node task's tolerance is half of this spacing)
SPACING = 0.1e9
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
KEEP = [
    "sky_model",
    "sky_residual",
    "primary_beam",
    "point_spread_function",
    "beam_fit_params_point_spread_function",
    "mask",
]


@pytest.fixture(scope="module")
def node_task_inputs(tmp_path_factory):
    """Keyword arguments of the one node task the distributed application
    runs on a small simulated field (graph mode, its image store created by
    the distributed application)."""
    from xradio.measurement_set import open_processing_set

    import astroviper.distributed_applications as distributed_applications
    import astroviper.node_tasks.imaging as node_tasks_imaging
    from astroviper.distributed_applications.imaging import image_cube_single_field
    from astroviper.utils.beam_models import airy_disk_model
    from astroviper.utils.data_tree import release_data_tree
    from astroviper.utils.telescope_layout import read_telescope_layout

    work_dir = tmp_path_factory.mktemp("node_task_science")
    ps_store = str(work_dir / "point.ps.zarr")
    antenna_xds = read_telescope_layout("alma.cycle8.1")
    distributed_applications.simulation.simulate_processing_set(
        ps_store=ps_store,
        antenna_xds=antenna_xds,
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 600.0,
            "n_samples": 12,
        },
        frequency_params={
            "freq_start": F0,
            "freq_delta": 0.5e9,
            "n_channels": 1,
            "channel_width": 2e6,
            "spectral_window_name": "Band3",
        },
        polarization=["XX", "YY"],
        sky_components=[
            {
                "kind": "point",
                "flux": np.array([[1.0, 0.0, 0.0, 1.0]]),
                "ra_dec": PHASE_CENTER + np.array([8.0, 5.0]) * ARCSEC,
                "name": "source_00",
            }
        ],
        phase_center_ra_dec=PHASE_CENTER[None, :],
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
        n_time_chunks=1,
        n_frequency_chunks=1,
        overwrite=True,
    )
    ps_xdt = open_processing_set(ps_store)
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    image_params = {
        "image_size": [96, 96],
        "cell_size": np.array([-0.7, 0.7]) * ARCSEC,
        "phase_direction": combined.FIELD_PHASE_CENTER_DIRECTION.sel(
            field_name=combined.attrs["center_field_name"]
        ).values.copy(),
        "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values.copy(),
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }
    del combined
    release_data_tree(ps_xdt)
    ps_xdt = None

    captured = []
    original = node_tasks_imaging.image_cube_single_field

    # graphviper forwards only the keyword arguments the node task declares
    @functools.wraps(original)
    def capture(*args, **kwargs):
        captured.append(copy.deepcopy(kwargs))
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(node_tasks_imaging, "image_cube_single_field", capture)
        with dask.config.set(scheduler="synchronous"):
            image_cube_single_field(
                ps_store=ps_store,
                image_store=str(work_dir / "driver.img.zarr"),
                image_params=image_params,
                imaging_weights_params={"weighting": "natural"},
                iteration_control_params={
                    "max_iter": 40,
                    "max_cycles": 2,
                    "threshold": 0.0,
                    "primary_beam_limit": 0.2,
                    "gain": 0.1,
                    "psf_sidelobe_factor": 1.5,
                    "max_iter_per_cycle": 20,
                    "min_psf_fraction": 0.05,
                    "max_psf_fraction": 0.8,
                },
                instrument_polarization_basis="linear",
                gridder="prolate_spheroidal",
                deconvolver="hogbom",
                scan_intents="OBSERVE_TARGET#ON_SOURCE",
                image_data_variables_keep=KEEP,
                processing_set_data_group_name="base",
                single_precision_image=False,
                processing_function_threads=1,
                n_mapping_parallelism={"frequency": 1},
                fft_backend="scipy",
                restore=True,
                primary_beam_correction=True,
                overwrite=True,
            )
    assert len(captured) == 1, f"{len(captured)} node tasks, expected one"
    return captured[0]


def _image(node_task_inputs, image_store, frequencies):
    """Run the node task on image channels ``frequencies`` (the visibility
    selection unchanged), written to a new ``image_store``; returns the
    written image and the task's result."""
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    kwargs = copy.deepcopy(node_task_inputs)
    kwargs["task_coords"]["frequency"]["data"] = np.asarray(frequencies, float)
    kwargs["task_coords"]["frequency"]["slice"] = slice(0, len(frequencies))
    kwargs["image_store"] = str(image_store)
    kwargs["graph_mode"] = False  # write the task's image to its own store
    result = image_cube_single_field(**kwargs)
    return xr.open_zarr(image_store), result


def _assert_same(a, b, where):
    """``a`` and ``b`` are the same, bit for bit for numbers."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), where
        for key in a:
            _assert_same(a[key], b[key], f"{where}[{key!r}]")
    elif isinstance(a, list | tuple):
        assert type(a) is type(b) and len(a) == len(b), where
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(x, y, f"{where}[{index}]")
    elif isinstance(a, str | bytes | None):
        assert a == b, where
    else:
        x, y = np.asarray(a), np.asarray(b)
        assert x.dtype == y.dtype and x.shape == y.shape, where
        assert x.tobytes() == y.tobytes(), where


@pytest.mark.parametrize(
    "frequencies, without_visibilities",
    [((F0 - SPACING, F0), 0), ((F0, F0 + SPACING), 1)],
    ids=["first", "last"],
)
def test_channel_without_visibilities_next_to_a_mapped_channel(
    node_task_inputs, tmp_path, frequencies, without_visibilities
):
    """One task of two image channels, of which only one has the visibility
    channel mapped onto it: the other is imaged from views of the whole chunk
    (the science function grids the chunk's only visibility channel onto
    it). Both come out exactly as when each is imaged in a task of its own.
    Handing the loaded chunk itself to the whole-chunk channel registered the
    imaging weights and model visibilities on it, so a mapped channel after
    it failed with "AssertionError: Output data variable WEIGHT_IMAGING
    already exists"."""
    from astroviper.processing_functions.imaging.utils.imaging_dict import Key

    both, both_result = _image(
        node_task_inputs, tmp_path / "both.img.zarr", frequencies
    )
    assert both.sizes["frequency"] == 2
    for chan, frequency in enumerate(frequencies):
        alone, alone_result = _image(
            node_task_inputs, tmp_path / f"alone_{chan}.img.zarr", [frequency]
        )
        for name, variable in alone.data_vars.items():
            in_both = both[name]
            if "frequency" in variable.dims:
                in_both = in_both.isel(frequency=[chan])
            _assert_same(in_both.values, variable.values, f"channel {chan} {name}")
        if chan == without_visibilities:
            # the chunk's visibility channel is gridded onto it
            assert float(np.abs(alone.SKY_MODEL.values).max()) > 0.1
        alone_dict = alone_result["deconvolution"].data
        assert alone_dict
        for key, value in alone_dict.items():
            moved = Key(time=key.time, pol=key.pol, chan=key.chan + chan)
            _assert_same(
                both_result["deconvolution"].data[moved], value, f"imaging dict {moved}"
            )
        for key, statistics in alone_result["image_statistics"].items():
            in_both = both_result["image_statistics"][key].isel(frequency=[chan])
            for name, variable in statistics.data_vars.items():
                _assert_same(
                    in_both[name].values,
                    variable.values,
                    f"channel {chan} statistics {key} {name}",
                )
    assert len(both_result["deconvolution"].data) == 2 * len(alone_dict)


def test_node_task_releases_everything_by_reference_counting(node_task_inputs):
    """With the garbage collector off, a node task (graph mode, as the
    distributed application runs it) leaves no object in a reference cycle,
    and no garbage collection runs inside it (a ``gc.collect()`` call
    included): the loaded chunk, the per-channel views, the image and every
    lazy zarr and xarray object die by reference counting alone."""
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    # first use (imports, caches) outside the measured call
    image_cube_single_field(**copy.deepcopy(node_task_inputs))
    kwargs = copy.deepcopy(node_task_inputs)
    enable = gc.enable
    gc_was_enabled = gc.isenabled()
    gc.disable()
    # everything alive now, garbage of earlier calls and tests included, is
    # set aside (not collected): the collection below sees only what the
    # measured call leaves
    gc.freeze()
    # dask's disable_gc decorator turns the collector back on after any dask
    # slice; keep it off for the measured call
    gc.enable = lambda: None
    collections_inside = []

    def on_collection(phase, info):
        if phase == "start":
            collections_inside.append(info["generation"])

    try:
        gc.callbacks.append(on_collection)
        try:
            result = image_cube_single_field(**kwargs)
        finally:
            gc.callbacks.remove(on_collection)
        assert not collections_inside, (
            f"garbage collections inside the node task: {collections_inside}"
        )
        assert result["deconvolution"].data
        del result, kwargs
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()
        left = cyclic_garbage(gc.garbage)
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.unfreeze()
        gc.enable = enable
        if gc_was_enabled:
            gc.enable()
    assert not left, f"objects left in reference cycles: {left.most_common(20)}"
