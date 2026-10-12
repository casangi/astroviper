"""Shared fixtures of the unit tests.

``tests/utils/conftest.py`` does not apply to ``tests/unit/`` (there is no root
conftest), so fixtures shared by several unit-test directories live here.

* ``imageable_msv2``: a small generated Measurement Set v2 that the imaging can
  read, built once per session and only when a test requests it. It needs
  XRADIO's ``xradio_msv2`` engine and python-casacore; without them the tests
  that request it are skipped.
* ``imageable_msv2_ps``: the same Measurement Set converted to a processing set
  with XRADIO's converter (``partition_scheme=[]``, the engine's default).

Tests must not write into ``imageable_msv2``: one that needs to change the
Measurement Set works on a copy.

Only the standard library and pytest are imported at module level, so the
``--doctest-modules`` collection never needs the optional packages.
"""

import os
import shutil

import pytest

#: Base name of the generated Measurement Set; XRADIO names the MSv4s of the
#: engine and of the converter after it (``imageable_0`` ... ``imageable_3``).
IMAGEABLE_MSV2_NAME = "imageable.ms"


def make_imageable_msv2(ms_path, seed=0):
    """Generate a small Measurement Set v2 that the imaging can read.

    XRADIO's ``gen_test_ms`` (without VLBI tables, with which the engine skips
    two of the partitions) writes 1,200 MAIN rows: 5 antennas, 2 spectral
    windows x 2 polarization setups (4 MSv4s), 16 channels and placeholder
    values. They are then made imageable and given the features the MSv2 path
    must handle:

    * real CHAN_FREQ: increasing in SPW 0, decreasing in SPW 1 (negative
      CHAN_WIDTH; the engine and the converter reverse it);
    * CORR_TYPE ``[XX, YY]``;
    * seeded random DATA, WEIGHT and UVW, and 5% of FLAG set;
    * a ``CORRECTED_DATA`` column in a TiledShapeStMan with narrow tiles
      (``[npol, 4, 64]``: 4 channels per tile, against whole-cell tiles for
      DATA), so the ``corrected`` data group exercises channel-sliced reads;
    * row 1 gets the keys of row 0 (a duplicated time and baseline: the last
      row wins);
    * 3 rows moved to a time of their own (a padded time and baseline grid).

    Parameters
    ----------
    ms_path : str
        Path of the Measurement Set to create.
    seed : int, optional
        Seed of the random values. Default 0.

    Returns
    -------
    str
        ``ms_path``.
    """
    import numpy as np
    from casacore import tables
    from casacore.tables.tableutil import makearrcoldesc
    from xradio.testing.measurement_set.msv2_io import gen_test_ms

    gen_test_ms(ms_path, vlbi_tables=False)
    rng = np.random.default_rng(seed)

    def random_visibilities(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            np.complex64
        )

    with tables.table(ms_path, readonly=False, ack=False) as main:
        n_row = main.nrows()
        n_channel, n_polarization = main.getcell("DATA", 0).shape
        cell = (n_row, n_channel, n_polarization)
        main.putcol("DATA", random_visibilities(cell))
        main.putcol("FLAG", rng.random(cell) < 0.05)
        main.putcol(
            "WEIGHT",
            rng.uniform(0.5, 2.0, (n_row, n_polarization)).astype(np.float32),
        )
        main.putcol("UVW", rng.uniform(-300.0, 300.0, (n_row, 3)))
        main.addcols(
            makearrcoldesc(
                "CORRECTED_DATA",
                0j,
                ndim=2,
                shape=[n_channel, n_polarization],
                valuetype="complex",
            ),
            {
                "TYPE": "TiledShapeStMan",
                "NAME": "CorrectedDataNarrowTiles",
                "SPEC": {"DEFAULTTILESHAPE": [n_polarization, 4, 64]},
            },
        )
        main.putcol("CORRECTED_DATA", random_visibilities(cell))
        for column in (
            "TIME",
            "ANTENNA1",
            "ANTENNA2",
            "DATA_DESC_ID",
            "FIELD_ID",
            "SCAN_NUMBER",
            "STATE_ID",
            "OBSERVATION_ID",
        ):
            main.putcell(column, 1, main.getcell(column, 0))
        for row in (5, 6, 7):
            main.putcell("TIME", row, main.getcell("TIME", row) + 1000.0)

    width = 1.0e6
    channels = np.arange(n_channel, dtype=np.float64)
    with tables.table(
        os.path.join(ms_path, "SPECTRAL_WINDOW"), readonly=False, ack=False
    ) as spectral_window:
        for row, frequency in enumerate(
            [100e9 + width * channels, 101e9 - width * channels]
        ):
            spectral_window.putcell("CHAN_FREQ", row, frequency)
            spectral_window.putcell(
                "CHAN_WIDTH", row, np.full(n_channel, width if row == 0 else -width)
            )
            spectral_window.putcell("EFFECTIVE_BW", row, np.full(n_channel, width))
            spectral_window.putcell("RESOLUTION", row, np.full(n_channel, width))
            spectral_window.putcell("REF_FREQUENCY", row, frequency[0])
            spectral_window.putcell("TOTAL_BANDWIDTH", row, width * n_channel)
    with tables.table(
        os.path.join(ms_path, "POLARIZATION"), readonly=False, ack=False
    ) as polarization:
        for row in range(polarization.nrows()):
            polarization.putcell("CORR_TYPE", row, np.array([9, 12], np.int32))
            polarization.putcell(
                "CORR_PRODUCT", row, np.array([[0, 0], [1, 1]], np.int32)
            )
    return ms_path


def _skip_without_msv2_engine():
    """Skip the requesting test without XRADIO's engine or python-casacore."""
    from astroviper.node_tasks.imaging.utils import msv2_engine_available

    available, reason = msv2_engine_available()
    if not available:
        pytest.skip(reason)
    pytest.importorskip("casacore.tables", reason="python-casacore not installed")


@pytest.fixture(scope="session")
def imageable_msv2(tmp_path_factory):
    """Path of the generated Measurement Set v2 (see ``make_imageable_msv2``).

    Read only: a test that writes into a Measurement Set copies this one.
    """
    _skip_without_msv2_engine()
    directory = tmp_path_factory.mktemp("imageable_msv2")
    return make_imageable_msv2(str(directory / IMAGEABLE_MSV2_NAME))


@pytest.fixture(scope="session")
def imageable_msv2_ps(imageable_msv2, tmp_path_factory):
    """Path of ``imageable_msv2`` converted to a processing set (Zarr).

    The converter reads a copy, so ``imageable_msv2`` is never written; the
    copy keeps the base name, so the MSv4 names equal the engine's.
    """
    from xradio.measurement_set import convert_msv2_to_processing_set

    directory = tmp_path_factory.mktemp("imageable_msv2_ps")
    ms_copy = str(directory / "copy" / IMAGEABLE_MSV2_NAME)
    shutil.copytree(imageable_msv2, ms_copy)
    ps_store = str(directory / "imageable.ps.zarr")
    convert_msv2_to_processing_set(ms_copy, ps_store, partition_scheme=[])
    shutil.rmtree(os.path.dirname(ms_copy))
    return ps_store
