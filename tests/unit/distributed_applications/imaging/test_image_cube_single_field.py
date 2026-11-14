"""Full-stack tests of the image_cube_single_field distributed application.

* On a small simulated processing set.
* With a Measurement Set v2 ``ps_store``: the guards (not gated: a directory
  that looks like a Measurement Set, and XRADIO's ``open_msv2`` removed or
  replaced by a stand-in), and, gated by XRADIO's ``xradio_msv2`` engine, the
  images of the generated ``imageable_msv2`` (``tests/unit/conftest.py``)
  against those of its conversion (``imageable_msv2_ps``), and a run that
  leaves the Measurement Set unwritten.
"""

import errno
import importlib
import os
import shutil

import dask
import numpy as np
import pytest
import xarray as xr
import zarr
from astropy.coordinates import SkyCoord
from xradio.image import load_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
import astroviper.node_tasks.imaging.find_continuum_first_round as find_continuum_node_tasks
from astroviper.distributed_applications.imaging.find_continuum_first_round import (
    FirstRoundStoreInputs,
    find_continuum_first_round_partials_path,
    validate_find_continuum_first_round,
)
from astroviper.distributed_applications.imaging.image_cube_single_field import (
    DISTRIBUTED_APPLICATION_TIMING_PHASES,
)
from astroviper.node_tasks.imaging.utils import msv2_engine_available
from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
    accumulate_moments,
    finalize_moments,
    first_round_channel_rows,
    new_moment_partials,
    primary_beam_masks,
    valid_voxels,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
    moment_channel_weights,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    FirstRoundRows,
)
from astroviper.processing_functions.imaging.utils.spectral_windows import (
    window_rest_frequency,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

_ENGINE_AVAILABLE, _ENGINE_REASON = msv2_engine_available()
requires_msv2_engine = pytest.mark.skipif(
    not _ENGINE_AVAILABLE, reason=_ENGINE_REASON or "needs XRADIO with open_msv2"
)

#: Every driver-level timing key of the phase layout.
TIMING_KEYS = {
    key for _, _, leaves in DISTRIBUTED_APPLICATION_TIMING_PHASES for _, key in leaves
}

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
N_ANTENNA = 8


def _simulate_point_source(tmp_path, n_channels=2):
    """1 Jy unpolarised point source at the phase centre, 8 VLA antennas,
    ``n_channels`` channels 0.4 GHz apart from 3 GHz."""
    result = distributed_applications.simulation.simulate_processing_set(
        ps_store=str(tmp_path / "point.ps.zarr"),
        antenna_xds=read_telescope_layout(
            "vla.d", antenna_selection=list(range(N_ANTENNA))
        ),
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 1800.0,
            "n_samples": 4,
        },
        frequency_params={
            "freq_start": 3e9,
            "freq_delta": 0.4e9,
            "n_channels": n_channels,
            "channel_width": 1e7,
        },
        polarization=["RR", "LL"],
        point_source_flux=np.array([1.0, 0, 0, 1.0])[None, None, None, :],
        point_source_ra_dec=PC[None, None, :],
        phase_center_ra_dec=PC[None, :],
        beam_models=[airy_disk_model("vla_nvss")],
        beam_model_map=np.zeros(N_ANTENNA, int),
        n_time_chunks=1,
        n_frequency_chunks=2,
        overwrite=True,
    )
    return result["ps_store"]


def test_image_store_name_without_img_zarr_extension(tmp_path):
    """A ``.zarr`` image store name is written as given by earlier XRADIO versions
    and as ``.img.zarr`` by later ones; the node tasks fill the store written."""
    ps_store = _simulate_point_source(tmp_path)
    ps_xdt = open_processing_set(ps_store)
    result = distributed_applications.imaging.image_cube_single_field(
        ps_store=ps_store,
        image_store=str(tmp_path / "cube.zarr"),
        image_params={
            "image_size": [64, 64],
            "cell_size": np.array([-8.0, 8.0]) * np.pi / (180 * 3600),
            "phase_direction": PC,
            "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
            "polarization_coords": ["I", "V"],
            "time_coords": [0],
            "fft_padding": 1.2,
            "cpp_gridder": True,
        },
        instrument_polarization_basis="circular",
        imaging_weights_params={
            "weighting": "natural",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        cycle_control_params={
            "max_iter": 0,
            "max_cycles": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        image_data_variables_keep=["sky_residual", "point_spread_function"],
        processing_set_data_group_name="base",
        single_precision_image=False,
        n_mapping_parallelism={"frequency": 2},
    )
    stores = [
        store
        for store in (tmp_path / "cube.zarr", tmp_path / "cube.img.zarr")
        if store.exists()
    ]
    assert len(stores) == 1
    img = load_image(str(stores[0]))
    # Both frequency chunks were written: the natural-weighted dirty image of the
    # 1 Jy source at the phase centre (primary beam 1) peaks at 1 Jy/beam.
    stokes_i = img.SKY_RESIDUAL.values[0, :, 0]  # [frequency, l, m]
    np.testing.assert_allclose(np.nanmax(stokes_i, axis=(1, 2)), [1.0, 1.0], rtol=1e-3)
    # Every driver step of the timing layout is timed; a processing set has no
    # lazy inputs to add.
    timing = result["timing_distributed_application"]
    assert TIMING_KEYS <= set(timing)
    assert timing["T_add_lazy_input_data"] == 0.0


def test_kept_variables_the_run_does_not_make_are_skipped(tmp_path):
    """A dirty image run (``max_iter=0``) that keeps ``mask``, which only
    deconvolution makes, writes every variable it does make: the mask is
    skipped instead of failing the write of every channel."""
    ps_store = _simulate_point_source(tmp_path)
    ps_xdt = open_processing_set(ps_store)
    result = distributed_applications.imaging.image_cube_single_field(
        ps_store=ps_store,
        image_store=str(tmp_path / "cube.img.zarr"),
        image_params={
            "image_size": [64, 64],
            "cell_size": np.array([-8.0, 8.0]) * np.pi / (180 * 3600),
            "phase_direction": PC,
            "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
            "polarization_coords": ["I", "V"],
            "time_coords": [0],
            "fft_padding": 1.2,
            "cpp_gridder": True,
        },
        instrument_polarization_basis="circular",
        imaging_weights_params={"weighting": "natural"},
        cycle_control_params={
            "max_iter": 0,
            "max_cycles": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        image_data_variables_keep=[
            "sky_residual",
            "sky_model",
            "mask",
            "point_spread_function",
        ],
        processing_set_data_group_name="base",
        single_precision_image=False,
        n_mapping_parallelism={"frequency": 2},
    )
    # No node task failed its write.
    assert "task_failed_phase" not in result["timing_node_tasks"]
    img = load_image(str(tmp_path / "cube.img.zarr"))
    stokes_i = img.SKY_RESIDUAL.values[0, :, 0]  # [frequency, l, m]
    np.testing.assert_allclose(np.nanmax(stokes_i, axis=(1, 2)), [1.0, 1.0], rtol=1e-3)
    assert np.isfinite(img.POINT_SPREAD_FUNCTION.values).all()


#: A threshold (Jy) above the 1 Jy/beam dirty peak of the point source.
NO_UPDATE_THRESHOLD = 100.0


def _image_restored(
    ps_store,
    image_store,
    n_tasks,
    clean_mask=None,
    restoration_output="both",
    **cycle_control_params,
):
    """The driver on the point source in ``n_tasks`` node tasks, restored and
    primary beam corrected, keeping the model, the mask and the primary
    beam."""
    ps_xdt = open_processing_set(ps_store)
    return distributed_applications.imaging.image_cube_single_field(
        ps_store=ps_store,
        image_store=image_store,
        image_params={
            "image_size": [64, 64],
            "cell_size": np.array([-8.0, 8.0]) * np.pi / (180 * 3600),
            "phase_direction": PC,
            "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
            "polarization_coords": ["I", "V"],
            "time_coords": [0],
            "fft_padding": 1.2,
        },
        instrument_polarization_basis="circular",
        imaging_weights_params={"weighting": "natural"},
        cycle_control_params={
            "max_iter": 40,
            "max_cycles": 2,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": 20,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "primary_beam_limit": 0.2,
            **cycle_control_params,
        },
        image_data_variables_keep=["sky_residual", "sky_model", "mask", "primary_beam"],
        processing_set_data_group_name="base",
        single_precision_image=False,
        n_mapping_parallelism={"frequency": n_tasks},
        restore=True,
        restoration_output=restoration_output,
        clean_mask=clean_mask,
    )


def _assert_restored_without_model_update(img, chan):
    """Channel ``chan`` has a zero model, its residual as the restored image
    and the residual divided by the primary beam as the primary beam corrected
    one (NaN below the cutoff)."""
    residual = img.SKY_RESIDUAL.values[:, chan]
    primary_beam = img.PRIMARY_BEAM.values[:, chan]
    corrected = img.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.values[:, chan]
    inside = primary_beam >= 0.2
    assert (img.SKY_MODEL.values[:, chan] == 0).all()
    np.testing.assert_array_equal(img.SKY_RESTORED.values[:, chan], residual)
    np.testing.assert_array_equal(
        corrected[inside], residual[inside] / primary_beam[inside]
    )
    assert np.isnan(corrected[~inside]).all()


@pytest.mark.parametrize(
    "cycle_control_params, mask_source",
    [
        ({"max_iter": 0}, "PRIMARY_BEAM > 0.2"),
        ({"max_cycles": 0}, "PRIMARY_BEAM > 0.2"),
        ({"threshold": NO_UPDATE_THRESHOLD}, "MASK"),
    ],
    ids=["max_iter_0", "max_cycles_0", "threshold_above_the_peak"],
)
def test_channel_without_model_update_is_restored(
    tmp_path, synchronous_dask, cycle_control_params, mask_source
):
    """A channel imaged in a node task of its own that makes no model update
    (``max_iter`` 0, ``max_cycles`` 0, or a threshold above its dirty peak)
    has a zero model and is restored to its residual, as in continuum
    imaging. Its masked statistics use the mask a model update would have
    made when one could run, else the primary beam fallback. Its model,
    restored and primary beam corrected images used to keep the image
    store's NaN fill value."""
    ps_store = _simulate_point_source(tmp_path)
    store = str(tmp_path / "cube.img.zarr")
    result = _image_restored(ps_store, store, 2, **cycle_control_params)
    _assert_imaged(result, 2)
    statistics = result["image_statistics"]["sky_residual"]
    assert statistics.attrs["mask_source"] == mask_source
    img = load_image(store)
    for chan in range(2):
        _assert_restored_without_model_update(img, chan)


@pytest.mark.parametrize("box_mask", [False, True], ids=["no_mask", "box_mask"])
@pytest.mark.parametrize("without_update", [0, 1], ids=["first", "second"])
def test_chunk_keeps_every_channel_when_one_makes_no_model_update(
    tmp_path, synchronous_dask, without_update, box_mask
):
    """One node task of two channels, one of which makes no model update (its
    threshold is above its dirty peak), with the primary beam mask or a clean
    mask: both come out exactly as when each is imaged in a node task of its
    own, the one without update with a zero model restored to its residual.
    The chunk takes its variables from its first channel, so a first channel
    without update lost the other's model and restored images (the image
    store's NaN fill value), and a second one failed the task: "channel 1
    result is missing variable 'SKY_MODEL' present for the chunk's first
    channel" ('MASK' with the primary beam mask, which only the model update
    made)."""
    ps_store = _simulate_point_source(tmp_path)
    clean_mask = None
    if box_mask:
        clean_mask = str(tmp_path / "mask.npy")
        box = np.zeros((64, 64), dtype=bool)
        box[24:40, 24:40] = True
        np.save(clean_mask, box)
    threshold = [[0.1, 0.1], [0.1, 0.1]]
    threshold[without_update] = [NO_UPDATE_THRESHOLD] * 2
    stores = {
        n_tasks: str(tmp_path / f"{n_tasks}_tasks.img.zarr") for n_tasks in (1, 2)
    }
    for n_tasks, store in stores.items():
        result = _image_restored(
            ps_store, store, n_tasks, clean_mask=clean_mask, threshold=threshold
        )
        _assert_imaged(result, n_tasks)
    _assert_images_equal(stores[1], stores[2])
    img = load_image(stores[1])
    _assert_restored_without_model_update(img, without_update)
    assert img.SKY_MODEL.values[:, 1 - without_update].max() > 0.5


#: Frequency offset (Hz) of the second execution block, of the order of those
#: between ALMA execution blocks converted to LSRK (0.7 to 4.9 kHz on 488 kHz
#: channels).
EXECUTION_BLOCK_FREQUENCY_OFFSET = 5.0e3


def _add_shifted_execution_block(ps_store, offset=EXECUTION_BLOCK_FREQUENCY_OFFSET):
    """Copy the processing set's only MSv4 as a second execution block whose
    channel centres are ``offset`` Hz higher and whose visibilities are three
    times brighter, and consolidate the metadata. Equal weights give the
    natural-weighted dirty image of both a peak of 2 Jy/beam (1 and 3 if one
    block were left out). Returns the frequencies of the first block."""
    import zarr

    (ms_name,) = open_processing_set(ps_store).keys()
    shifted_name = f"{ms_name}_shifted"
    shutil.copytree(
        os.path.join(ps_store, ms_name), os.path.join(ps_store, shifted_name)
    )
    frequency = zarr.open_array(os.path.join(ps_store, shifted_name, "frequency"))
    frequencies = frequency[:]
    frequency[:] = frequencies + offset
    visibility = zarr.open_array(os.path.join(ps_store, shifted_name, "VISIBILITY"))
    visibility[:] = 3.0 * visibility[:]
    zarr.consolidate_metadata(ps_store)
    shifted = open_processing_set(ps_store)
    assert sorted(shifted.keys()) == sorted([ms_name, shifted_name])
    np.testing.assert_array_equal(
        shifted[shifted_name].frequency.values, frequencies + offset
    )
    return frequencies


@pytest.mark.parametrize("n_tasks", [1, 2])
@pytest.mark.parametrize("skunk_works", [True, False])
def test_execution_blocks_with_offset_frequencies_are_all_imaged(
    tmp_path, skunk_works, n_tasks
):
    """Two execution blocks whose channel centres differ by a few kHz, imaged
    on the first one's frequencies in one node task or one per channel: every
    task selects and images both (each channel maps onto its nearest image
    channel), with the direct Zarr loader (``skunk_works=True``) as with the
    standard one, and the two loaders give the same images. The direct loader
    used to require every selected frequency to equal an image frequency to
    1e-12, so it failed the load and, as a load failure is logged and
    skipped, wrote no image at all; and graphviper's channel selection used
    to drop the second block from the task of an end channel."""
    ps_store = _simulate_point_source(tmp_path)
    frequencies = _add_shifted_execution_block(ps_store)

    def image(store_name, use_skunk_works):
        return distributed_applications.imaging.image_cube_single_field(
            ps_store=ps_store,
            image_store=str(tmp_path / store_name),
            image_params={
                "image_size": [64, 64],
                "cell_size": np.array([-8.0, 8.0]) * np.pi / (180 * 3600),
                "phase_direction": PC,
                "frequency_coords": frequencies,
                "polarization_coords": ["I", "V"],
                "time_coords": [0],
                "fft_padding": 1.2,
            },
            instrument_polarization_basis="circular",
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params={
                "max_iter": 0,
                "max_cycles": 0,
                "threshold": 0.0,
                "gain": 0.1,
                "psf_sidelobe_factor": 1.5,
                "max_iter_per_cycle": -1,
                "min_psf_fraction": 0.05,
                "max_psf_fraction": 0.8,
            },
            image_data_variables_keep=["sky_residual", "point_spread_function"],
            processing_set_data_group_name="base",
            single_precision_image=False,
            n_mapping_parallelism={"frequency": n_tasks},
            skunk_works=use_skunk_works,
        )

    result = image("cube.img.zarr", skunk_works)
    _assert_imaged(result, n_tasks=n_tasks)
    img = load_image(str(tmp_path / "cube.img.zarr"))
    # The natural-weighted dirty image of the source at the phase centre, 1 Jy
    # in the first execution block and 3 Jy in the second, peaks at 2 Jy/beam
    # in both channels only when both blocks are imaged in both.
    stokes_i = img.SKY_RESIDUAL.values[0, :, 0]  # [frequency, l, m]
    np.testing.assert_allclose(np.nanmax(stokes_i, axis=(1, 2)), [2.0, 2.0], rtol=1e-3)
    assert np.isfinite(img.POINT_SPREAD_FUNCTION.values).all()

    if skunk_works:
        reference = image("standard.img.zarr", False)
        _assert_imaged(reference, n_tasks=n_tasks)
        _assert_images_equal(
            str(tmp_path / "cube.img.zarr"), str(tmp_path / "standard.img.zarr")
        )


# --------------------------------------------------------------------------- #
# Measurement Set v2 ps_store
# --------------------------------------------------------------------------- #
#: The scan intent of the generated Measurement Set's MSv4s.
MSV2_SCAN_INTENTS = ["scan_intent#subscan_intent"]


def _image_cube(ps_store, image_store, frequency_coords, **overrides):
    """The driver on a small image (32 x 32 pixels, Stokes I and Q of linear
    feeds, Briggs weights): the dirty image unless ``max_iter`` is given."""
    max_iter = overrides.pop("max_iter", 0)
    keep = ["sky_residual", "point_spread_function", "primary_beam"]
    if max_iter:
        keep.append("sky_model")
    arguments = dict(
        ps_store=ps_store,
        image_store=image_store,
        image_params={
            "image_size": [32, 32],
            "cell_size": np.array([-0.5, 0.5]) * np.pi / (180 * 3600),
            "phase_direction": np.array([0.0, 0.5]),
            "frequency_coords": np.asarray(frequency_coords),
            "polarization_coords": ["I", "Q"],
            "time_coords": [0],
            "fft_padding": 1.2,
            "cpp_gridder": True,
        },
        imaging_weights_params={
            "weighting": "briggs",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        cycle_control_params={
            "max_iter": max_iter,
            "max_cycles": 2 if max_iter else 0,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        instrument_polarization_basis="linear",
        scan_intents=MSV2_SCAN_INTENTS,
        image_data_variables_keep=keep,
        processing_set_data_group_name="base",
        single_precision_image=False,
        n_mapping_parallelism={"frequency": 3},
        skunk_works=True,
    )
    arguments.update(overrides)
    return distributed_applications.imaging.image_cube_single_field(**arguments)


def _measurement_set_like(tmp_path):
    """A directory that looks like a Measurement Set v2 (a casacore table)."""
    ms_path = tmp_path / "input.ms"
    ms_path.mkdir()
    (ms_path / "table.dat").write_bytes(b"")
    return str(ms_path)


def _assert_nothing_written(tmp_path):
    """Nothing but the input was created (no image store, no FITS file)."""
    assert sorted(os.listdir(tmp_path)) == ["input.ms"]
    assert os.listdir(tmp_path / "input.ms") == ["table.dat"]


class _OpenMSv2:
    """Stands in for XRADIO's ``open_msv2``: records its calls, returns
    ``result`` (default: an empty processing set, as when no MSv4 has one of
    the scan intents)."""

    def __init__(self):
        self.calls = []
        self.result = xr.DataTree()

    def __call__(self, ms_path, **kwargs):
        self.calls.append((ms_path, kwargs))
        return self.result


@pytest.fixture
def stand_in_open_msv2(monkeypatch):
    """XRADIO with the stand-in ``open_msv2`` (so the guards run without the
    engine) and no partition-cache environment variable."""
    import xradio.measurement_set

    stand_in = _OpenMSv2()
    monkeypatch.setattr(xradio.measurement_set, "open_msv2", stand_in, raising=False)
    monkeypatch.delenv("XRADIO_MSV2_PARTITION_CACHE", raising=False)
    return stand_in


def test_msv2_requires_skunk_works(tmp_path):
    ms_path = _measurement_set_like(tmp_path)
    with pytest.raises(ValueError, match="skunk_works=True"):
        _image_cube(ms_path, str(tmp_path / "cube.img.zarr"), [1e11], skunk_works=False)
    _assert_nothing_written(tmp_path)


def test_msv2_without_the_engine(tmp_path, monkeypatch):
    import xradio.measurement_set

    monkeypatch.delattr(xradio.measurement_set, "open_msv2", raising=False)
    ms_path = _measurement_set_like(tmp_path)
    with pytest.raises(ImportError, match="xradio_msv2"):
        _image_cube(ms_path, str(tmp_path / "cube.img.zarr"), [1e11])
    _assert_nothing_written(tmp_path)


@pytest.mark.parametrize(
    "flag", ["write_visibility_model_to_ps", "write_imaging_weights_to_ps"]
)
def test_msv2_refuses_writing_into_the_input(tmp_path, flag):
    ms_path = _measurement_set_like(tmp_path)
    with pytest.raises(
        ValueError, match="not implemented for the in-memory cube workflow"
    ):
        _image_cube(ms_path, str(tmp_path / "cube.img.zarr"), [1e11], **{flag: True})
    _assert_nothing_written(tmp_path)


@pytest.mark.parametrize("key", ["array_backend", "scan_intents"])
def test_msv2_refused_open_option(tmp_path, stand_in_open_msv2, key):
    ms_path = _measurement_set_like(tmp_path)
    with pytest.raises(ValueError, match=f"msv2_open_options may not set.*{key}"):
        _image_cube(
            ms_path,
            str(tmp_path / "cube.img.zarr"),
            [1e11],
            msv2_open_options={key: "dask"},
        )
    assert stand_in_open_msv2.calls == []
    _assert_nothing_written(tmp_path)


def test_msv2_no_measurement_set_after_scan_intents(tmp_path, stand_in_open_msv2):
    """An empty selection is refused (with the hint) before anything is
    written; ``scan_intents=None`` passes the parameter check and reaches the
    engine with the driver's open options."""
    ms_path = _measurement_set_like(tmp_path)
    with pytest.raises(ValueError, match="scan_intents=None"):
        _image_cube(
            ms_path,
            str(tmp_path / "cube.img.zarr"),
            [1e11],
            scan_intents=["OBSERVE_TARGET#ON_SOURCE"],
        )
    _assert_nothing_written(tmp_path)

    with pytest.raises(ValueError, match="no visibilities"):
        _image_cube(
            ms_path,
            str(tmp_path / "cube.img.zarr"),
            [1e11],
            scan_intents=None,
            msv2_open_options={"partition_scheme": ["FIELD_ID"]},
        )
    _assert_nothing_written(tmp_path)
    assert stand_in_open_msv2.calls[-1] == (
        ms_path,
        {
            "scan_intents": None,
            "array_backend": "xarray",
            "with_pointing": False,
            "partition_cache": "read",
            "partition_scheme": ["FIELD_ID"],
        },
    )


@pytest.mark.parametrize(
    "data_groups, match",
    [
        (
            {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "uvw": "UVW",
                }
            },
            r"input_0 has no data group 'corrected' \(it has \['base'\]\)",
        ),
        (
            {
                "corrected": {
                    "correlated_data": "SPECTRUM_CORRECTED",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                }
            },
            r"has no \['uvw'\].*single-dish",
        ),
    ],
    ids=["missing_group", "single_dish"],
)
def test_msv2_data_group_refused_before_anything_is_written(
    tmp_path, stand_in_open_msv2, data_groups, match
):
    """An MSv4 without the data group, or whose group lacks a role the imaging
    reads, is refused right after the open: no image store is left behind."""
    stand_in_open_msv2.result = xr.DataTree.from_dict(
        {"input_0": xr.Dataset(attrs={"data_groups": data_groups})}
    )
    ms_path = _measurement_set_like(tmp_path)
    with pytest.raises(ValueError, match=match):
        _image_cube(
            ms_path,
            str(tmp_path / "cube.img.zarr"),
            [1e11],
            processing_set_data_group_name="corrected",
        )
    assert len(stand_in_open_msv2.calls) == 1
    _assert_nothing_written(tmp_path)


@pytest.fixture
def synchronous_dask(monkeypatch):
    """Run the graphs in this process, one task at a time (deterministic, and
    within the memory of one test process), from the default open options."""
    monkeypatch.delenv("XRADIO_MSV2_PARTITION_CACHE", raising=False)
    with dask.config.set(scheduler="synchronous"):
        yield


def _assert_imaged(result, n_tasks=3):
    """Every task imaged its channels (no task was skipped)."""
    timing = result["timing_node_tasks"]
    assert len(timing) == n_tasks
    if "task_failed_phase" in timing:
        assert timing["task_failed_phase"].isna().all(), timing["task_error"].tolist()


def _assert_images_equal(store, reference_store):
    """Every image variable and coordinate is the same, bit for bit (NaN where
    the reference has NaN), and the residual is finite in every plane."""
    image = load_image(store)
    reference = load_image(reference_store)
    assert sorted(image.data_vars) == sorted(reference.data_vars)
    for name in reference.data_vars:
        assert image[name].dtype == reference[name].dtype, name
        assert np.array_equal(
            image[name].values, reference[name].values, equal_nan=True
        ), name
    for name in ("frequency", "polarization"):
        np.testing.assert_array_equal(image[name].values, reference[name].values)
    residual = image["SKY_RESIDUAL"].values  # (time, frequency, polarization, l, m)
    assert np.isfinite(residual).any(axis=(-2, -1)).all()


@requires_msv2_engine
@pytest.mark.parametrize(
    "group, max_iter",
    [
        ("base", 20),
        ("corrected", 0),
        pytest.param(
            "corrected",
            20,
            marks=pytest.mark.skip(
                reason="deconvolution of a data group other than 'base' fails "
                "on every input path; fixed by casangi/astroviper#311"
            ),
        ),
    ],
    ids=["base_clean", "corrected_dirty", "corrected_clean"],
)
def test_msv2_images_equal_production_on_the_conversion(
    imageable_msv2, imageable_msv2_ps, tmp_path, synchronous_dask, group, max_iter
):
    """The Measurement Set v2, imaged directly, gives the images of its
    conversion imaged by the production path, bit for bit: both spectral
    windows (16 channels each, one decreasing) in 5 tasks of 7, 7, 7, 7 and 4
    channels, so that each spectral window has a task inside its channels
    (neither at its first nor at its last channel) and one task straddles
    both, with whole-cell tiles (``base``) and narrow tiles (``corrected``)."""
    frequencies = open_processing_set(imageable_msv2_ps).xr_ps.get_freq_axis().values
    assert frequencies.size == 32
    msv2_store = str(tmp_path / "msv2.img.zarr")
    production_store = str(tmp_path / "production.img.zarr")
    common = dict(
        processing_set_data_group_name=group,
        max_iter=max_iter,
        n_mapping_parallelism={"frequency": 5},
    )

    result = _image_cube(imageable_msv2, msv2_store, frequencies, **common)
    production = _image_cube(
        imageable_msv2_ps, production_store, frequencies, skunk_works=False, **common
    )

    _assert_imaged(result, n_tasks=5)
    _assert_imaged(production, n_tasks=5)
    _assert_images_equal(msv2_store, production_store)
    timing = result["timing_distributed_application"]
    assert TIMING_KEYS <= set(timing)
    assert timing["T_add_lazy_input_data"] > 0.0


@requires_msv2_engine
def test_msv2_images_equal_skunk_works_on_the_conversion(
    imageable_msv2, imageable_msv2_ps, tmp_path, synchronous_dask
):
    """The same against the Zarr skunk-works path, on the decreasing spectral
    window's channels only: that loader assumes the image's frequencies, so a
    task that straddles two spectral windows is skipped there."""
    converted = open_processing_set(imageable_msv2_ps)
    frequencies = converted["imageable_2"].frequency.values
    assert frequencies.size == 16
    msv2_store = str(tmp_path / "msv2.img.zarr")
    skunk_works_store = str(tmp_path / "skunk_works.img.zarr")

    result = _image_cube(imageable_msv2, msv2_store, frequencies, max_iter=20)
    skunk_works = _image_cube(
        imageable_msv2_ps, skunk_works_store, frequencies, max_iter=20
    )

    _assert_imaged(result)
    _assert_imaged(skunk_works)
    _assert_images_equal(msv2_store, skunk_works_store)


@requires_msv2_engine
def test_msv2_missing_data_group_with_the_engine(
    imageable_msv2, tmp_path, synchronous_dask
):
    """Through the engine: the generated Measurement Set has no MODEL_DATA, so
    no ``model`` data group; the run is refused before the image store is
    created."""
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match=r"no data group 'model'"):
        _image_cube(
            imageable_msv2,
            str(output / "msv2.img.zarr"),
            100e9 + 1e6 * np.arange(16),
            processing_set_data_group_name="model",
        )
    assert os.listdir(output) == []


def _file_state(path):
    """Every file under ``path``: its relative path, size and mtime."""
    state = {}
    for directory, _, files in os.walk(path):
        for name in files:
            full = os.path.join(directory, name)
            stat = os.stat(full)
            state[os.path.relpath(full, path)] = (stat.st_size, stat.st_mtime_ns)
    return state


@requires_msv2_engine
def test_msv2_run_does_not_write_the_measurement_set(
    imageable_msv2, tmp_path, synchronous_dask
):
    """A whole run with the default open options leaves the Measurement Set
    as it was: no partition cache stored and no file written or touched (on a
    copy of the generated Measurement Set)."""
    ms_path = str(tmp_path / "input" / os.path.basename(imageable_msv2))
    shutil.copytree(imageable_msv2, ms_path)
    before = _file_state(ms_path)
    frequencies = 100e9 + 1e6 * np.arange(16)  # spectral window 0

    result = _image_cube(ms_path, str(tmp_path / "msv2.img.zarr"), frequencies)

    _assert_imaged(result)
    assert _file_state(ms_path) == before
    assert not os.path.exists(os.path.join(ms_path, "XRADIO_PARTITIONS"))


@pytest.mark.parametrize("output", ["apparent", "primary_beam_corrected", "both"])
def test_cube_publishes_only_selected_restoration_products(
    tmp_path, synchronous_dask, output
):
    ps_store = _simulate_point_source(tmp_path)
    store = str(tmp_path / "selected.img.zarr")
    _image_restored(ps_store, store, 1, restoration_output=output, max_iter=0)
    image = load_image(store)
    assert ("SKY_RESTORED" in image) == (output in ("apparent", "both"))
    assert ("SKY_RESTORED_PRIMARY_BEAM_CORRECTED" in image) == (
        output in ("primary_beam_corrected", "both")
    )
    selected = (
        "SKY_RESTORED"
        if output == "apparent"
        else "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"
    )
    assert np.isfinite(image[selected]).any()


# --------------------------------------------------------------------------- #
# find_continuum's first round
# --------------------------------------------------------------------------- #
#: Offset (Hz) of the second spectral window from the first.
SECOND_WINDOW_OFFSET = 2.0e9
#: Spectral windows of the first-round cube, and its tiles (px).
WINDOW_RANGES = [[0, 3], [3, 6]]
FIRST_ROUND_TILE = 32


@pytest.fixture
def threaded_dask():
    """Run the graphs on the threaded scheduler: every node task adds its
    moments to the registry of this process."""
    with dask.config.set(scheduler="threads"):
        yield


def _image_first_round(
    tmp_path, n_channels=3, offset=SECOND_WINDOW_OFFSET, n_tasks=3, **kwargs
):
    """The dirty cube of the point source in two spectral windows of
    ``n_channels`` channels (the second ``offset`` Hz above the first and
    three times brighter), with find_continuum's first round: ``n_tasks``
    node tasks (by default three, the second straddling the windows), a chunk
    per channel, tiles of 32 px and the primary beam below its limit in the
    corners."""
    ps_store = _simulate_point_source(tmp_path, n_channels=n_channels)
    frequencies = _add_shifted_execution_block(ps_store, offset)
    return distributed_applications.imaging.image_cube_single_field(
        ps_store=ps_store,
        image_store=str(tmp_path / "cube.img.zarr"),
        image_params={
            "image_size": [64, 64],
            "cell_size": np.array([-40.0, 40.0]) * np.pi / (180 * 3600),
            "phase_direction": PC,
            "frequency_coords": np.concatenate([frequencies, frequencies + offset]),
            "polarization_coords": ["I", "V"],
            "time_coords": [0],
            "fft_padding": 1.2,
        },
        instrument_polarization_basis="circular",
        imaging_weights_params={"weighting": "natural"},
        cycle_control_params={
            "max_iter": 0,
            "max_cycles": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "primary_beam_limit": 0.2,
        },
        image_data_variables_keep=["sky_residual"],
        processing_set_data_group_name="base",
        n_mapping_parallelism={"frequency": n_tasks},
        image_chunking={"frequency": 1, "l": FIRST_ROUND_TILE, "m": FIRST_ROUND_TILE},
        find_continuum_first_round=True,
        **kwargs,
    )


def _first_round_array(store, name):
    return zarr.open_array(os.path.join(store, name), mode="r")[:]


@pytest.mark.parametrize("windows_in_memory", [2, 1])
def test_find_continuum_first_round(
    tmp_path, threaded_dask, monkeypatch, windows_in_memory
):
    """The imager writes findContinuum's first round (store version 3) of the
    cube it wrote into a store bound to it: the masks and rows of every
    channel, and the moments of both spectral windows from the registry of the
    worker process (holding both windows, or one and saving the other). Each
    window validates with its moments; a range across the two windows is
    refused. It keeps the variables find_continuum reads, but not the point
    spread function."""
    monkeypatch.setattr(
        find_continuum_node_tasks, "FIRST_ROUND_WINDOWS_IN_MEMORY", windows_in_memory
    )
    result = _image_first_round(tmp_path)

    _assert_imaged(result)
    image_store = str(tmp_path / "cube.img.zarr")
    store = str(tmp_path / "cube_find_continuum_first_round.zarr")
    assert result["find_continuum_first_round"] == {
        "store": store,
        "primary_beam_limit": 0.2,
        "complete": True,
        "moments_complete": [True, True],
    }
    for index, window in enumerate(WINDOW_RANGES):
        assert validate_find_continuum_first_round(
            store, image_store, 0.2, window
        ) == FirstRoundStoreInputs(store, index, True, "image_cube_single_field")
    with pytest.raises(ValueError, match=r"holds channels .* not \[2, 5\)"):
        validate_find_continuum_first_round(store, image_store, 0.2, [2, 5])
    assert not find_continuum_node_tasks._FIRST_ROUND_MOMENTS
    assert not os.path.exists(find_continuum_first_round_partials_path(store))

    img_xds = xr.open_zarr(image_store, chunks=None)
    frequency = img_xds.frequency.values
    attributes = zarr.open_group(store, mode="r").attrs["find_continuum_first_round"]
    assert attributes["version"] == 3
    assert attributes["written_by"] == "image_cube_single_field"
    assert attributes["spectrum_primary_beam_level"] == 0.21
    assert attributes["windows"] == [
        {
            "channel_range": [start, stop],
            "first_frequency": float(frequency[start]),
            "channel_increment": float(frequency[start + 1] - frequency[start]),
            "n_channels": stop - start,
        }
        for start, stop in WINDOW_RANGES
    ]
    assert attributes["n_missing"] == attributes["n_duplicated"] == [0, 0]
    task_seconds = result["timing_node_tasks"]["T_find_continuum_first_round"]
    assert (task_seconds > 0).all()
    timing = result["timing_distributed_application"]
    assert TIMING_KEYS <= set(timing)
    assert timing["T_find_continuum_first_round_write"] > 0
    assert attributes["imager"] == {
        "n_tasks": 3,
        "T_find_continuum_first_round_sum": pytest.approx(task_seconds.sum()),
        "T_drain": timing["T_find_continuum_first_round_drain"],
    }
    assert timing["T_find_continuum_first_round_drain"] > 0
    stages = [record.get("stage") for record in result["timing_reduce_nodes"]]
    assert stages.count("first_round_pop") == 1
    assert stages.count("first_round_merge") == len(WINDOW_RANGES)

    group = img_xds.attrs["data_groups"]["residual"]
    assert "point_spread_function" not in group
    assert "POINT_SPREAD_FUNCTION" not in img_xds
    beams = img_xds[group["beam_fit_params_point_spread_function"]].values
    assert np.isfinite(beams).all()
    stokes_i = list(img_xds.polarization.values).index("I")
    planes = slice(stokes_i, stokes_i + 1)
    sky = img_xds[group["sky"]].values[:, :, planes]
    beam = img_xds[group["primary_beam"]].values[:, :, planes]
    limit_mask, spectrum_mask = primary_beam_masks(beam, 0.2)
    assert not limit_mask.all() and spectrum_mask.any()
    assert (spectrum_mask != limit_mask).any()

    n_channels, n_tiles = sky.shape[1], 64 // FIRST_ROUND_TILE
    for name, mask in zip(
        ("LIMIT", "SPECTRUM"),
        (limit_mask[0, :, 0], spectrum_mask[0, :, 0]),
        strict=True,
    ):
        packed = _first_round_array(store, f"PRIMARY_BEAM_{name}_MASK")
        np.testing.assert_array_equal(
            np.unpackbits(packed, axis=-1, count=64).view(bool), mask
        )
        tiles = mask.reshape(n_channels, n_tiles, FIRST_ROUND_TILE, n_tiles, -1)
        np.testing.assert_array_equal(
            _first_round_array(store, f"PRIMARY_BEAM_{name}_TILES"),
            tiles.any(axis=(2, 4)),
        )

    rows = first_round_channel_rows(sky, beam, limit_mask)
    assert (rows["sky_valid_count"] > 0).all()
    for field in FirstRoundRows._fields:
        stored = _first_round_array(store, field.upper())
        assert stored.dtype == rows[field].dtype
        np.testing.assert_array_equal(stored, rows[field][0, :, 0])

    # The moments of each window are weighted at its own rest frequency, its
    # middle channel, not at the cube's.
    rest_frequency = float(img_xds.frequency.attrs["rest_frequency"]["data"])
    assert rest_frequency == frequency[3]
    maps = [
        _first_round_array(store, name)
        for name in ("SKY_MOMENT_INTEGRATED", "SKY_MOMENT_MAXIMUM")
    ]
    map_shape = (1, 1, 1, *sky.shape[-2:])
    valid = valid_voxels(sky, limit_mask) & (sky != 0)
    for window, (start, stop) in enumerate(WINDOW_RANGES):
        partials = new_moment_partials(map_shape, sky.dtype)
        accumulate_moments(
            partials,
            sky[:, start:stop],
            valid[:, start:stop],
            moment_channel_weights(
                float(frequency[start]),
                float(frequency[start + 1] - frequency[start]),
                stop - start,
                window_rest_frequency(frequency, [start, stop], rest_frequency),
            ),
        )
        integral, maximum = (
            values[0, 0, 0]
            for values in finalize_moments(partials, map_shape, sky.dtype)
        )
        assert np.isfinite(integral).any() and np.isnan(integral).any()
        np.testing.assert_array_equal(maps[1][window], maximum)
        np.testing.assert_allclose(maps[0][window], integral, rtol=1e-12)


def test_find_continuum_first_round_shard_waves_one_window_at_a_time(
    tmp_path, threaded_dask, monkeypatch
):
    """With sharded output and the moments, the imager dispatches the shard
    waves of one spectral window before those of the next, so that a worker
    saves the moments of each window about once; the waves over the whole
    cube would take a worker through both windows in every wave."""
    import graphviper.graph_tools

    import astroviper.node_tasks as node_tasks

    graphs = []
    graphviper_map = graphviper.graph_tools.map

    def recording_map(*args, **kwargs):
        graphs.append(kwargs)
        return graphviper_map(*args, **kwargs)

    monkeypatch.setattr(graphviper.graph_tools, "map", recording_map)
    result = _image_first_round(
        tmp_path,
        n_channels=4,
        n_tasks=8,
        skunk_works=True,
        image_sharding={"frequency": 2},
    )

    _assert_imaged(result, n_tasks=8)
    assert result["find_continuum_first_round"]["moments_complete"] == [True, True]
    (graph,) = (
        graph
        for graph in graphs
        if graph["node_task"] is node_tasks.imaging.image_cube_single_field
    )
    priorities = graph["task_priorities"]
    channels = {
        task: item["task_coords"]["frequency"]["slice"].start
        for task, item in graph["node_task_data_mapping"].items()
    }
    order = sorted(priorities, key=lambda task: (-priorities[task], task))
    # Shards of two channels, two per window: the waves of window [0, 4),
    # then those of window [4, 8).
    assert [channels[task] for task in order] == [0, 2, 1, 3, 4, 6, 5, 7]


def test_find_continuum_first_round_of_a_chunk_not_written(
    tmp_path, threaded_dask, monkeypatch
):
    """A chunk that cannot be written leaves the first round incomplete: the
    imager returns ``None``. The moments are still drained, so the process
    keeps none of them, and the window of that chunk has none."""
    node_task = importlib.import_module(
        "astroviper.node_tasks.imaging.image_cube_single_field"
    )
    select_chunk_writer = node_task._select_chunk_writer

    def select_failing_writer(*args):
        write = select_chunk_writer(*args)

        def write_or_fail(chunk_xds, chunk_task_coords):
            if chunk_task_coords["frequency"]["slice"].start == 4:
                raise OSError("no space left on device")
            write(chunk_xds, chunk_task_coords)

        return write_or_fail

    monkeypatch.setattr(node_task, "_select_chunk_writer", select_failing_writer)
    result = _image_first_round(tmp_path)

    assert result["find_continuum_first_round"] is None
    assert (result["timing_node_tasks"]["task_failed_phase"] == "write").sum() == 1
    store = str(tmp_path / "cube_find_continuum_first_round.zarr")
    attributes = zarr.open_group(store, mode="r").attrs["find_continuum_first_round"]
    assert not attributes["complete"]
    assert attributes["moments_complete"] == [True, False]
    assert attributes["n_missing"] == [0, 1]
    with pytest.raises(ValueError, match="incomplete"):
        validate_find_continuum_first_round(
            store, str(tmp_path / "cube.img.zarr"), 0.2, WINDOW_RANGES[0]
        )
    assert not find_continuum_node_tasks._FIRST_ROUND_MOMENTS
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


def test_find_continuum_first_round_of_a_chunk_that_fails(
    tmp_path, threaded_dask, monkeypatch
):
    """The first round of a chunk that cannot be written (after the retries of
    transient errors) is logged and leaves the first round incomplete; the
    imaging goes on, and no process keeps moments or slots."""
    write_masks = find_continuum_node_tasks.write_primary_beam_masks

    def write_or_fail(first_round_store, channels, limit_mask, spectrum_mask):
        if list(channels) == [4]:
            raise OSError(errno.EIO, "Input/output error")
        return write_masks(first_round_store, channels, limit_mask, spectrum_mask)

    monkeypatch.setattr(
        find_continuum_node_tasks, "write_primary_beam_masks", write_or_fail
    )
    result = _image_first_round(tmp_path)

    _assert_imaged(result)
    assert result["find_continuum_first_round"] is None
    assert result["timing_node_tasks"]["n_failed_first_round_chunks"].sum() == 1
    store = str(tmp_path / "cube_find_continuum_first_round.zarr")
    assert not zarr.open_group(store, mode="r").attrs["find_continuum_first_round"][
        "complete"
    ]
    assert not find_continuum_node_tasks._FIRST_ROUND_MOMENTS
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


def test_find_continuum_first_round_with_a_tile_that_fails_to_merge(
    tmp_path, threaded_dask, monkeypatch
):
    """A merge of the moments that fails after the imaging leaves only that
    window's moments incomplete; the first round is bound and returned."""
    from astroviper.distributed_applications.imaging import find_continuum_first_round

    merge = find_continuum_first_round.merge_first_round_moment_tile

    def merge_or_fail(store, partials_directory, window, slots, tile):
        if window == 1:
            raise OSError(errno.EIO, "Input/output error")
        return merge(store, partials_directory, window, slots, tile)

    monkeypatch.setattr(
        find_continuum_first_round, "merge_first_round_moment_tile", merge_or_fail
    )
    result = _image_first_round(tmp_path)

    _assert_imaged(result)
    store = str(tmp_path / "cube_find_continuum_first_round.zarr")
    assert result["find_continuum_first_round"] == {
        "store": store,
        "primary_beam_limit": 0.2,
        "complete": True,
        "moments_complete": [True, False],
    }
    assert [
        validate_find_continuum_first_round(
            store, str(tmp_path / "cube.img.zarr"), 0.2, window
        ).moments_usable
        for window in WINDOW_RANGES
    ] == [True, False]
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


def test_a_failed_imaging_compute_drops_the_first_round_moments(
    tmp_path, threaded_dask, monkeypatch
):
    """The compute fails after every task added its moments (here in the
    reduce): the call raises, no process keeps the moments of the run and
    their slots are removed."""
    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.image_cube_single_field"
    )
    monkeypatch.setattr(find_continuum_node_tasks, "FIRST_ROUND_WINDOWS_IN_MEMORY", 1)
    flushed = []
    flush = find_continuum_node_tasks._flush_first_round_window

    def recorded_flush(entry, window, partials_directory):
        flush(entry, window, partials_directory)
        flushed.append(window)

    def failing_reduce(input_data, input_params):
        raise RuntimeError("reduce failed")

    monkeypatch.setattr(
        find_continuum_node_tasks, "_flush_first_round_window", recorded_flush
    )
    monkeypatch.setattr(driver, "combine_return_data_frames", failing_reduce)
    with pytest.raises(RuntimeError, match="reduce failed"):
        _image_first_round(tmp_path)

    assert flushed
    assert not find_continuum_node_tasks._FIRST_ROUND_MOMENTS
    store = str(tmp_path / "cube_find_continuum_first_round.zarr")
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


def test_find_continuum_first_round_merge_tiles_without_l_and_m_chunking(tmp_path):
    """The default image chunking spans the plane; the moments of the first
    round are merged in tiles of 625 px chunks, not of the whole plane."""
    from xradio.image import make_empty_sky_image

    from astroviper.utils.io import (
        create_empty_data_variables_on_disk,
        write_zarr_image_store,
    )

    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.image_cube_single_field"
    )
    n_pixels = 1500
    img_xds = make_empty_sky_image(
        phase_center=[1.2, -0.4],
        image_size=[n_pixels, n_pixels],
        cell_size=[1.0e-7, 1.0e-7],
        frequency_coords=230.0e9 + 1.0e6 * np.arange(4),
        pol_coords=["I", "Q"],
        time_coords=[0.0],
    )
    img_xds.attrs["data_groups"] = {
        "residual": {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
    }
    image_store = write_zarr_image_store(img_xds, str(tmp_path / "cube.img.zarr"))
    create_empty_data_variables_on_disk(
        image_store,
        ["sky_residual", "primary_beam"],
        shape_dict=dict(img_xds.sizes),
        parallel_coords={"frequency": {"data_chunks": [[0, 1], [2, 3]]}},
        compressor=None,
        double_precision=False,
        data_variable_definitions="imaging",
    )
    params = driver._create_find_continuum_first_round(
        image_store, img_xds, 0.2, "dask"
    )
    store = params["first_round_store"]
    for name in ("SKY_MOMENT_INTEGRATED", "SKY_MOMENT_MAXIMUM"):
        array = zarr.open_array(os.path.join(store, name), mode="r")
        assert array.chunks == (1, 750, 750)
    shutil.rmtree(params["partials_directory"])


def test_find_continuum_first_round_refuses_fits(tmp_path):
    """findContinuum reads Zarr image stores: FITS output is refused before
    anything is written."""
    with pytest.raises(ValueError, match="output_image_format 'zarr'"):
        distributed_applications.imaging.image_cube_single_field(
            ps_store=str(tmp_path / "point.ps.zarr"),
            image_store=str(tmp_path / "cube"),
            image_params={
                "image_size": [64, 64],
                "cell_size": np.array([-40.0, 40.0]) * np.pi / (180 * 3600),
                "phase_direction": PC,
                "frequency_coords": np.array([3.0e9, 3.4e9]),
                "polarization_coords": ["I", "Q"],
                "time_coords": [0],
                "fft_padding": 1.2,
            },
            instrument_polarization_basis="linear",
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params={
                "max_iter": 0,
                "max_cycles": 0,
                "threshold": 0.0,
                "gain": 0.1,
                "psf_sidelobe_factor": 1.5,
                "max_iter_per_cycle": -1,
                "min_psf_fraction": 0.05,
                "max_psf_fraction": 0.8,
            },
            skunk_works=True,
            output_image_format="fits",
            find_continuum_first_round=True,
        )
    assert os.listdir(tmp_path) == []
