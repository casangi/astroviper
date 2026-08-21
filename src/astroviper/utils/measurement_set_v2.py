"""Write a simulated MSv4 processing set as a CASA Measurement Set v2 via arcae.

Optional backend of the simulator (the SIRIUS ``write_to_ms`` equivalent):
``write_measurement_set_v2`` converts one simulated MSv4 into an MSv2 on disk
using `arcae <https://github.com/ska-sa/arcae>`_ (an **optional** dependency;
install with ``pip install arcae``).  ``arcae.Table.ms_from_descriptor`` builds
the complete casacore Measurement Set skeleton -- the MAIN table plus every
required subtable -- and the columns are then filled from the MSv4 datasets.

Conventions follow what CASA's own simulator writes (verified against an
``sm``-produced MS): the MSv4 ``UVW`` and ``VISIBILITY`` follow the archival /
VLBI convention (``uvw = P(antenna1) - P(antenna2)`` with the matching phase
sign) and are copied verbatim, ``TIME``/``TIME_CENTROID`` are integration
midpoints in MJD UTC seconds, ``FLAG_CATEGORY`` cells are left undefined, and
the ``STATE`` and ``PROCESSOR`` subtables get one default row each.
"""

from __future__ import annotations

import os
import shutil

import numpy as np

from astroviper.utils.measurement_set_tools import POLARIZATION_NAME_TO_CODE

# Seconds between the casacore MJD epoch (1858-11-17) and the unix epoch.
MJD_UNIX_OFFSET_SECONDS = 3506716800.0

# Receptor index pair of each correlation (CORR_PRODUCT).
_CORR_PRODUCT = {
    "XX": (0, 0), "XY": (0, 1), "YX": (1, 0), "YY": (1, 1),
    "RR": (0, 0), "RL": (0, 1), "LR": (1, 0), "LL": (1, 1),
}  # fmt: skip

# casacore MFrequency reference codes for the MSv4 ``observer`` label.
_FREQUENCY_FRAME_CODE = {
    "rest": 0, "lsrk": 1, "lsrd": 2, "bary": 3, "geo": 4, "topo": 5, "galacto": 6,
}  # fmt: skip

_MAIN_DATA_COLUMN_DESC = {
    "DATA": {
        "comment": "Complex visibilities",
        "dataManagerGroup": "StandardStMan",
        "dataManagerType": "StandardStMan",
        "keywords": {"UNIT": "Jy"},
        "maxlen": 0,
        "ndim": 2,
        "option": 0,
        "valueType": "COMPLEX",
    }
}


def _require_arcae():
    try:
        from arcae.lib.arrow_tables import Table
    except ImportError as error:
        raise ImportError(
            "Writing a Measurement Set v2 requires the optional dependency "
            "'arcae' (https://github.com/ska-sa/arcae): pip install arcae"
        ) from error
    return Table


def write_measurement_set_v2(ps_store, ms_path, ms_name=None, overwrite=False):
    """Convert one simulated MSv4 of a processing set to a Measurement Set v2.

    Parameters
    ----------
    ps_store : str
        Path of the simulated MSv4 processing set (Zarr store).  The MSv4 is
        loaded into memory, so this is meant for simulator outputs, not
        arbitrarily large archives.
    ms_path : str
        Output path of the Measurement Set (conventionally ``*.ms``).
    ms_name : str, optional
        Name of the MSv4 inside the processing set to convert.  Default: the
        only (first) one.
    overwrite : bool, optional
        Remove an existing ``ms_path`` first.  Default ``False`` (raise).

    Returns
    -------
    str
        ``ms_path``.

    Raises
    ------
    ImportError
        If ``arcae`` is not installed.
    FileExistsError
        If ``ms_path`` exists and ``overwrite`` is ``False``.
    """
    Table = _require_arcae()
    from xradio.measurement_set import load_processing_set

    if os.path.exists(ms_path):
        if not overwrite:
            raise FileExistsError(
                f"{ms_path} exists; pass overwrite=True to replace it."
            )
        shutil.rmtree(ms_path)

    ps_xdt = load_processing_set(ps_store)
    if ms_name is None:
        ms_name = next(iter(ps_xdt.children))
    ms_tree = ps_xdt[ms_name]
    ms_xds = ms_tree.ds
    antenna_xds = ms_tree["antenna_xds"].ds
    field_xds = ms_tree["field_and_source_base_xds"].ds

    n_time = ms_xds.sizes["time"]
    n_baseline = ms_xds.sizes["baseline_id"]
    n_frequency = ms_xds.sizes["frequency"]
    polarization = [str(p) for p in ms_xds.polarization.values]
    n_polarization = len(polarization)
    n_antenna = antenna_xds.sizes["antenna_name"]
    n_row = n_time * n_baseline

    antenna_names = [str(name) for name in antenna_xds.antenna_name.values]
    antenna_index = {name: k for k, name in enumerate(antenna_names)}
    antenna1 = np.array(
        [antenna_index[str(n)] for n in ms_xds.baseline_antenna1_name.values],
        dtype=np.int32,
    )
    antenna2 = np.array(
        [antenna_index[str(n)] for n in ms_xds.baseline_antenna2_name.values],
        dtype=np.int32,
    )

    time_mjd = (
        np.asarray(ms_xds.time.values, dtype=np.float64) + MJD_UNIX_OFFSET_SECONDS
    )
    integration_time = float(ms_xds.time.attrs["integration_time"]["data"])
    frequency = np.asarray(ms_xds.frequency.values, dtype=np.float64)
    channel_width = float(ms_xds.frequency.attrs["channel_width"]["data"])

    field_names = [str(name) for name in field_xds.field_name.values]
    field_of_row_time = np.array(
        [field_names.index(str(name)) for name in ms_xds.field_name.values],
        dtype=np.int32,
    )

    # ---------------- MAIN ----------------
    # MSv4 and MSv2 share the archival uvw/phase convention (see
    # utils/measurement_set_tools and calculate_uvw): copy verbatim.
    uvw = np.asarray(ms_xds.UVW.values, dtype=np.float64)
    visibility = np.asarray(ms_xds.VISIBILITY.values, dtype=np.complex64)
    weight = np.asarray(ms_xds.WEIGHT.values, dtype=np.float32)
    flag = np.asarray(ms_xds.FLAG.values, dtype=bool)

    main = Table.ms_from_descriptor(ms_path, "MAIN", table_desc=_MAIN_DATA_COLUMN_DESC)
    main.addrows(n_row)
    row_time = np.repeat(time_mjd, n_baseline)
    main.putcol("TIME", row_time)
    main.putcol("TIME_CENTROID", row_time)
    main.putcol("INTERVAL", np.full(n_row, integration_time))
    main.putcol("EXPOSURE", np.full(n_row, integration_time))
    main.putcol("ANTENNA1", np.tile(antenna1, n_time))
    main.putcol("ANTENNA2", np.tile(antenna2, n_time))
    main.putcol("FEED1", np.zeros(n_row, dtype=np.int32))
    main.putcol("FEED2", np.zeros(n_row, dtype=np.int32))
    main.putcol("DATA_DESC_ID", np.zeros(n_row, dtype=np.int32))
    main.putcol("FIELD_ID", np.repeat(field_of_row_time, n_baseline))
    main.putcol("SCAN_NUMBER", np.ones(n_row, dtype=np.int32))
    main.putcol("ARRAY_ID", np.zeros(n_row, dtype=np.int32))
    main.putcol("OBSERVATION_ID", np.zeros(n_row, dtype=np.int32))
    main.putcol("PROCESSOR_ID", np.zeros(n_row, dtype=np.int32))
    main.putcol("STATE_ID", np.zeros(n_row, dtype=np.int32))
    main.putcol("UVW", uvw.reshape(n_row, 3))
    main.putcol("DATA", visibility.reshape(n_row, n_frequency, n_polarization))
    # MSv2 WEIGHT/SIGMA have no channel axis; the simulator's weights are
    # channel-independent, so channel 0 is representative.
    row_weight = weight[:, :, 0, :].reshape(n_row, n_polarization)
    main.putcol("WEIGHT", row_weight)
    with np.errstate(divide="ignore"):
        main.putcol("SIGMA", (1.0 / np.sqrt(row_weight)).astype(np.float32))
    main.putcol(
        "FLAG",
        flag.reshape(n_row, n_frequency, n_polarization),
    )
    main.putcol("FLAG_ROW", np.zeros(n_row, dtype=bool))
    # FLAG_CATEGORY cells stay undefined, as in CASA-simulated sets.
    main.close()

    # ---------------- ANTENNA ----------------
    antenna = Table.ms_from_descriptor(ms_path, "ANTENNA")
    antenna.addrows(n_antenna)
    antenna.putcol("NAME", np.array(antenna_names))
    antenna.putcol("STATION", np.array(antenna_names))
    antenna.putcol("TYPE", np.array(["GROUND-BASED"] * n_antenna))
    antenna.putcol("MOUNT", np.array([str(m) for m in antenna_xds.mount.values]))
    antenna.putcol(
        "POSITION", np.asarray(antenna_xds.ANTENNA_POSITION.values, dtype=np.float64)
    )
    antenna.putcol("OFFSET", np.zeros((n_antenna, 3)))
    antenna.putcol(
        "DISH_DIAMETER",
        np.asarray(antenna_xds.ANTENNA_DISH_DIAMETER.values, dtype=np.float64),
    )
    antenna.putcol("FLAG_ROW", np.zeros(n_antenna, dtype=bool))
    antenna.close()

    # ---------------- SPECTRAL_WINDOW ----------------
    frame_label = str(ms_xds.frequency.attrs.get("observer", "topo")).lower()
    spectral_window = Table.ms_from_descriptor(ms_path, "SPECTRAL_WINDOW")
    spectral_window.addrows(1)
    spectral_window.putcol(
        "NAME", np.array([str(ms_xds.frequency.attrs["spectral_window_name"])])
    )
    spectral_window.putcol("NUM_CHAN", np.array([n_frequency], dtype=np.int32))
    spectral_window.putcol("CHAN_FREQ", frequency[None, :])
    for column in ("CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"):
        spectral_window.putcol(column, np.full((1, n_frequency), channel_width))
    spectral_window.putcol("REF_FREQUENCY", np.array([frequency[0]]))
    spectral_window.putcol(
        "TOTAL_BANDWIDTH", np.array([abs(frequency[-1] - frequency[0]) + channel_width])
    )
    spectral_window.putcol(
        "MEAS_FREQ_REF",
        np.array([_FREQUENCY_FRAME_CODE.get(frame_label, 5)], dtype=np.int32),
    )
    spectral_window.putcol("NET_SIDEBAND", np.array([1], dtype=np.int32))
    spectral_window.putcol("IF_CONV_CHAIN", np.array([0], dtype=np.int32))
    spectral_window.putcol("FREQ_GROUP", np.array([0], dtype=np.int32))
    spectral_window.putcol("FREQ_GROUP_NAME", np.array([""]))
    spectral_window.putcol("FLAG_ROW", np.array([False]))
    spectral_window.close()

    # ---------------- POLARIZATION ----------------
    pol_table = Table.ms_from_descriptor(ms_path, "POLARIZATION")
    pol_table.addrows(1)
    pol_table.putcol("NUM_CORR", np.array([n_polarization], dtype=np.int32))
    pol_table.putcol(
        "CORR_TYPE",
        np.array(
            [[POLARIZATION_NAME_TO_CODE[p] for p in polarization]], dtype=np.int32
        ),
    )
    pol_table.putcol(
        "CORR_PRODUCT",
        np.array([[_CORR_PRODUCT[p] for p in polarization]], dtype=np.int32),
    )
    pol_table.putcol("FLAG_ROW", np.array([False]))
    pol_table.close()

    # ---------------- DATA_DESCRIPTION ----------------
    data_description = Table.ms_from_descriptor(ms_path, "DATA_DESCRIPTION")
    data_description.addrows(1)
    data_description.putcol("SPECTRAL_WINDOW_ID", np.array([0], dtype=np.int32))
    data_description.putcol("POLARIZATION_ID", np.array([0], dtype=np.int32))
    data_description.putcol("FLAG_ROW", np.array([False]))
    data_description.close()

    # ---------------- FIELD ----------------
    field_direction = np.asarray(
        field_xds.FIELD_PHASE_CENTER_DIRECTION.values, dtype=np.float64
    ).reshape(len(field_names), 1, 2)
    field = Table.ms_from_descriptor(ms_path, "FIELD")
    field.addrows(len(field_names))
    field.putcol("NAME", np.array(field_names))
    field.putcol("CODE", np.array([""] * len(field_names)))
    for column in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
        field.putcol(column, field_direction)
    field.putcol("TIME", np.full(len(field_names), time_mjd[0]))
    field.putcol("NUM_POLY", np.zeros(len(field_names), dtype=np.int32))
    field.putcol("SOURCE_ID", np.zeros(len(field_names), dtype=np.int32))
    field.putcol("FLAG_ROW", np.zeros(len(field_names), dtype=bool))
    field.close()

    # ---------------- OBSERVATION ----------------
    telescope_name = str(
        antenna_xds.attrs.get(
            "overall_telescope_name",
            ms_xds.attrs.get("observation_info", {}).get("telescope_name", "unknown"),
        )
    )
    observation = Table.ms_from_descriptor(ms_path, "OBSERVATION")
    observation.addrows(1)
    observation.putcol("TELESCOPE_NAME", np.array([telescope_name]))
    observation.putcol(
        "TIME_RANGE",
        np.array(
            [[time_mjd[0] - integration_time / 2, time_mjd[-1] + integration_time / 2]]
        ),
    )
    observation.putcol("OBSERVER", np.array(["astroviper"]))
    observation.putcol("PROJECT", np.array(["astroviper simulation"]))
    observation.putcol("RELEASE_DATE", np.array([0.0]))
    observation.putcol("FLAG_ROW", np.array([False]))
    observation.close()

    # ---------------- FEED ----------------
    receptors = ["X", "Y"] if polarization[0][0] in ("X", "Y") else ["R", "L"]
    feed = Table.ms_from_descriptor(ms_path, "FEED")
    feed.addrows(n_antenna)
    feed.putcol("ANTENNA_ID", np.arange(n_antenna, dtype=np.int32))
    feed.putcol("FEED_ID", np.zeros(n_antenna, dtype=np.int32))
    feed.putcol("SPECTRAL_WINDOW_ID", np.full(n_antenna, -1, dtype=np.int32))
    feed.putcol("TIME", np.full(n_antenna, time_mjd[0]))
    feed.putcol("INTERVAL", np.full(n_antenna, 1e30))
    feed.putcol("NUM_RECEPTORS", np.full(n_antenna, 2, dtype=np.int32))
    feed.putcol("BEAM_ID", np.full(n_antenna, -1, dtype=np.int32))
    feed.putcol("BEAM_OFFSET", np.zeros((n_antenna, 2, 2)))
    feed.putcol("POLARIZATION_TYPE", np.array([receptors] * n_antenna))
    feed.putcol(
        "POL_RESPONSE", np.tile(np.eye(2, dtype=np.complex64), (n_antenna, 1, 1))
    )
    feed.putcol("POSITION", np.zeros((n_antenna, 3)))
    feed.putcol("RECEPTOR_ANGLE", np.zeros((n_antenna, 2)))
    feed.close()

    # STATE and PROCESSOR: one default row each, as CASA-simulated sets carry.
    for subtable in ("STATE", "PROCESSOR"):
        table = Table.ms_from_descriptor(ms_path, subtable)
        table.addrows(1)
        table.close()

    return ms_path
