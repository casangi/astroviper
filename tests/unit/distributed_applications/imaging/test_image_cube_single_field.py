"""Full-stack tests of the image_cube_single_field distributed application.

* On a small simulated processing set.
* With a Measurement Set v2 ``ps_store``: the guards (not gated: a directory
  that looks like a Measurement Set, and XRADIO's ``open_msv2`` removed or
  replaced by a stand-in), and, gated by XRADIO's ``xradio_msv2`` engine, the
  images of the generated ``imageable_msv2`` (``tests/unit/conftest.py``)
  against those of its conversion (``imageable_msv2_ps``), and a run that
  leaves the Measurement Set unwritten.
"""

import os
import shutil

import dask
import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord
from xradio.image import load_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging.image_cube_single_field import (
    DISTRIBUTED_APPLICATION_TIMING_PHASES,
)
from astroviper.node_tasks.imaging.utils import msv2_engine_available
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


def _simulate_point_source(tmp_path):
    """1 Jy unpolarised point source at the phase centre, 8 VLA antennas, 2 channels."""
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
            "n_channels": 2,
            "channel_width": 1e7,
        },
        polarization=["RR", "LL"],
        point_source_flux=np.array([1.0, 0, 0, 1.0])[None, None, None, :],
        point_source_ra_dec=PC[None, None, :],
        phase_center_ra_dec=PC[None, :],
        beam_models=[airy_disk_model("vla")],
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
        iteration_control_params={
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
        iteration_control_params={
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
    with pytest.raises(ValueError, match="never writes into its input"):
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
