"""Component tests: imaging a Measurement Set v2 directly (skunk_works).

The real ALMA Measurement Set v2 ``Antennae_North.cal.lsrk.split.ms`` (4 MSv4s,
8 channels of a spectral window with decreasing frequencies, XX and YY) is
imaged by ``image_cube_single_field`` with ``ps_store`` set to the Measurement
Set itself and ``skunk_works=True``: the driver opens it with XRADIO's
``xradio_msv2`` engine and every node task reads its own selection through
the engine. The tests check that

* the images (Briggs weights, 2 imaging cycles of Hogbom CLEAN, Stokes I and
  Q) equal, bit for bit, those of the Measurement Set's conversion
  (``convert_msv2_to_processing_set``) imaged by the production path, and the
  run leaves the Measurement Set unwritten;
* the sharded (``image_sharding``) and FITS (``output_image_format="fits"``)
  writers give the same images from the Measurement Set v2 input.

The graphs run on a Dask ``LocalCluster`` of two worker processes with one
thread each (the deployment the driver documents for a Measurement Set v2: the
reads hold the GIL), so every task's lazy input is pickled to another process.

The tests need XRADIO's ``xradio_msv2`` engine (an XRADIO release that carries
it, with python-casacore or casatools) and skip without it. The data are
downloaded with ``toolviper.utils.data.download``.

Run with pytest::

    pytest tests/component/test_single_field_imaging_msv2.py
"""

import glob
import os
import shutil

import numpy as np
import pytest

from astroviper.node_tasks.imaging.utils import msv2_engine_available

_ENGINE_AVAILABLE, _ENGINE_REASON = msv2_engine_available()
pytestmark = pytest.mark.skipif(
    not _ENGINE_AVAILABLE, reason=_ENGINE_REASON or "needs XRADIO with open_msv2"
)

MS_NAME = "Antennae_North.cal.lsrk.split.ms"

#: Image variables kept by every run (all real-valued, so FITS can hold them).
IMAGE_DATA_VARIABLES_KEEP = [
    "sky_residual",
    "point_spread_function",
    "primary_beam",
    "sky_model",
]

#: Frequency chunks (node tasks): 2 channels each.
N_FREQUENCY_CHUNKS = 4


def _file_state(path):
    """Every file under ``path``: its relative path, size and mtime."""
    state = {}
    for directory, _, files in os.walk(path):
        for name in files:
            full = os.path.join(directory, name)
            stat = os.stat(full)
            state[os.path.relpath(full, path)] = (stat.st_size, stat.st_mtime_ns)
    return state


@pytest.fixture(scope="module")
def antennae_msv2(tmp_path_factory):
    """Path of the downloaded Measurement Set v2 (a private copy)."""
    from toolviper.utils.data import download

    directory = tmp_path_factory.mktemp("antennae_msv2")
    download(MS_NAME, folder=str(directory))
    return str(directory / MS_NAME)


@pytest.fixture(scope="module")
def antennae_ps(antennae_msv2, tmp_path_factory):
    """Path of the Measurement Set converted to a processing set.

    The converter reads a copy (named as the original, so the MSv4 names equal
    the engine's), so the Measurement Set the tests image is never written.
    """
    from xradio.measurement_set import convert_msv2_to_processing_set

    directory = tmp_path_factory.mktemp("antennae_ps")
    ms_copy = str(directory / "copy" / MS_NAME)
    shutil.copytree(antennae_msv2, ms_copy)
    ps_store = str(directory / "Antennae_North.ps.zarr")
    convert_msv2_to_processing_set(ms_copy, ps_store, partition_scheme=[])
    shutil.rmtree(os.path.dirname(ms_copy))
    return ps_store


@pytest.fixture(scope="module")
def image_params(antennae_ps):
    """A small image (64 x 64 pixels of 0.13 arcsec) of every channel, centred
    on the central field (read from the converted processing set)."""
    from xradio.measurement_set import open_processing_set

    ps_xdt = open_processing_set(antennae_ps)
    field_and_source_xds = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    phase_direction = field_and_source_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=field_and_source_xds.attrs["center_field_name"]
    )
    frequencies = ps_xdt.xr_ps.get_freq_axis().values
    assert frequencies.size == 2 * N_FREQUENCY_CHUNKS
    return {
        "image_size": [64, 64],
        "cell_size": np.array([-0.13, 0.13]) * np.pi / (180 * 3600),
        "phase_direction": np.asarray(phase_direction.values),
        "frequency_coords": frequencies,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }


@pytest.fixture(scope="module")
def dask_client():
    """A Dask cluster of two worker processes with one thread each."""
    from distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        memory_limit="2GiB",
        dashboard_address=None,
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


def _image_cube(ps_store, image_store, image_params, **overrides):
    """The driver with the settings shared by every run of this module."""
    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    arguments = dict(
        ps_store=ps_store,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params={
            "weighting": "briggs",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        iteration_control_params={
            "max_iter": 50,
            "max_cycles": 2,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        instrument_polarization_basis="linear",
        scan_intents=["OBSERVE_TARGET#ON_SOURCE"],
        image_data_variables_keep=IMAGE_DATA_VARIABLES_KEEP,
        processing_set_data_group_name="base",
        single_precision_image=False,
        processing_function_threads=1,
        n_mapping_parallelism={"frequency": N_FREQUENCY_CHUNKS},
        skunk_works=True,
        overwrite=True,
    )
    arguments.update(overrides)
    return image_cube_single_field(**arguments)


def _assert_imaged(result):
    """Every task imaged its channels (no task was skipped), in a worker
    process of the cluster (so its input was pickled to it)."""
    timing = result["timing_node_tasks"]
    assert len(timing) == N_FREQUENCY_CHUNKS
    if "task_failed_phase" in timing:
        assert timing["task_failed_phase"].isna().all(), timing["task_error"].tolist()
    assert (timing["process_pid"] != os.getpid()).all()
    assert timing["worker_name"].notna().all()


@pytest.fixture(scope="module")
def msv2_image(antennae_msv2, image_params, dask_client, tmp_path_factory):
    """The Measurement Set v2 imaged directly: ``(image store, result,
    Measurement Set file state before, after)``."""
    store = str(tmp_path_factory.mktemp("msv2_image") / "msv2.img.zarr")
    before = _file_state(antennae_msv2)
    result = _image_cube(antennae_msv2, store, image_params)
    return store, result, before, _file_state(antennae_msv2)


def _assert_images_equal(image, reference):
    """Every variable of ``reference`` is in ``image``, bit for bit (NaN where
    the reference has NaN)."""
    for name in reference.data_vars:
        assert image[name].dtype == reference[name].dtype, name
        assert np.array_equal(
            image[name].values, reference[name].values, equal_nan=True
        ), name


def test_msv2_images_equal_production_on_the_conversion(
    msv2_image, antennae_msv2, antennae_ps, image_params, dask_client, tmp_path
):
    """Imaged directly, the Measurement Set v2 gives the images of its
    conversion imaged by the production path, bit for bit, and is not
    written."""
    from xradio.image import load_image

    msv2_store, result, before, after = msv2_image
    production_store = str(tmp_path / "production.img.zarr")
    production = _image_cube(
        antennae_ps, production_store, image_params, skunk_works=False
    )

    _assert_imaged(result)
    _assert_imaged(production)
    image = load_image(msv2_store)
    reference = load_image(production_store)
    assert sorted(image.data_vars) == sorted(reference.data_vars)
    assert sorted(reference.data_vars) == sorted(
        name.upper() for name in IMAGE_DATA_VARIABLES_KEEP
    )
    _assert_images_equal(image, reference)
    for name in ("frequency", "polarization", "l", "m"):
        np.testing.assert_array_equal(image[name].values, reference[name].values)
    # A real image: finite residuals in every plane, and CLEAN found flux.
    assert np.isfinite(image["SKY_RESIDUAL"].values).any(axis=(-2, -1)).all()
    assert np.nanmax(np.abs(image["SKY_MODEL"].values)) > 0.0

    timing = result["timing_distributed_application"]
    assert timing["T_add_lazy_input_data"] > 0.0
    # Nothing in the Measurement Set was written or touched (default open
    # options: the partitions are not stored in it).
    assert after == before
    assert not os.path.exists(os.path.join(antennae_msv2, "XRADIO_PARTITIONS"))


def _chunk_files(store, variable):
    """The chunk (or shard) files of one variable of a Zarr v3 store."""
    return [
        path
        for path in glob.glob(os.path.join(store, variable, "c", "**"), recursive=True)
        if os.path.isfile(path)
    ]


def test_msv2_sharded_output(msv2_image, antennae_msv2, image_params, tmp_path):
    """The sharded writer gives the same image from a Measurement Set v2 input,
    in 2 shard files per variable (2 tasks of 2 channels each per shard)
    instead of one file per task."""
    from xradio.image import load_image

    msv2_store, _, _, _ = msv2_image
    sharded_store = str(tmp_path / "sharded.img.zarr")
    result = _image_cube(
        antennae_msv2,
        sharded_store,
        image_params,
        image_sharding={"frequency": 2 * 2},
    )

    _assert_imaged(result)
    reference = load_image(msv2_store)
    _assert_images_equal(load_image(sharded_store), reference)
    for name in reference.data_vars:
        assert len(_chunk_files(sharded_store, name)) == 2, name
        assert len(_chunk_files(msv2_store, name)) == N_FREQUENCY_CHUNKS, name


def test_msv2_fits_output(msv2_image, antennae_msv2, image_params, tmp_path):
    """The FITS writer gives the same image from a Measurement Set v2 input:
    one double-precision FITS file per kept variable."""
    from astropy.io import fits
    from xradio.image import load_image

    msv2_store, _, _, _ = msv2_image
    fits_store = str(tmp_path / "msv2_fits")
    result = _image_cube(
        antennae_msv2, fits_store, image_params, output_image_format="fits"
    )

    _assert_imaged(result)
    reference = load_image(msv2_store)
    assert sorted(os.listdir(fits_store)) == sorted(
        f"{name}.fits" for name in reference.data_vars
    )
    for name in reference.data_vars:
        with fits.open(os.path.join(fits_store, f"{name}.fits")) as hdulist:
            assert hdulist[0].header["BITPIX"] == -64, name
            data = hdulist[0].data
        # FITS stores (frequency, polarization, m, l) per time; the image
        # dataset (time, frequency, polarization, l, m).
        expected = reference[name].values[0].transpose(0, 1, 3, 2)
        assert np.array_equal(data, expected, equal_nan=True), name
