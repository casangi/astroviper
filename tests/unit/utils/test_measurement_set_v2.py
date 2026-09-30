"""Unit tests for the arcae MSv2 backend of the simulator."""

import importlib.util

import numpy as np
import pytest

arcae_missing = importlib.util.find_spec("arcae") is None

pytestmark = pytest.mark.skipif(
    arcae_missing, reason="optional dependency arcae not installed"
)

MJD_UNIX_OFFSET_SECONDS = 3506716800.0


@pytest.fixture(scope="module")
def simulated_ms(tmp_path_factory):
    """A small simulated processing set converted to an MSv2."""
    from astropy.coordinates import SkyCoord

    import astroviper.distributed_applications as distributed_applications
    from astroviper.utils.beam_models import airy_disk_model
    from astroviper.utils.telescope_layout import read_telescope_layout

    tmp_path = tmp_path_factory.mktemp("msv2")
    ps_store = str(tmp_path / "sim.ps.zarr")
    ms_path = str(tmp_path / "sim.ms")
    pc = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="icrs")
    pc_rad = np.array([pc.ra.rad, pc.dec.rad])[None, :]
    ant = read_telescope_layout("vla.d").isel(antenna_name=slice(0, 6))
    distributed_applications.simulation.simulate_processing_set(
        ps_store=ps_store,
        antenna_xds=ant,
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 60.0,
            "n_samples": 3,
        },
        frequency_params={
            "freq_start": 3e9,
            "freq_delta": 1e7,
            "n_channels": 2,
            "channel_width": 1e7,
        },
        polarization=["RR", "LL"],
        point_source_flux=np.array([[[[1.0, 0, 0, 1.0]]]]),
        point_source_ra_dec=pc_rad[:, None, :],
        phase_center_ra_dec=pc_rad,
        beam_models=[airy_disk_model("vla")],
        beam_model_map=np.zeros(6, int),
        n_time_chunks=1,
        n_frequency_chunks=1,
        overwrite=True,
        ms_v2_path=ms_path,
    )
    return ps_store, ms_path


def test_main_table_round_trip(simulated_ms):
    """MAIN carries the MSv4 content in MSv2 conventions (uvw sign, conjugate)."""
    from arcae.lib.arrow_tables import Table
    from xradio.measurement_set import load_processing_set

    ps_store, ms_path = simulated_ms
    ms_xds = next(iter(load_processing_set(ps_store).children.values())).ds
    n_time = ms_xds.sizes["time"]
    n_baseline = ms_xds.sizes["baseline_id"]

    table = Table.from_filename(ms_path)
    assert table.nrow() == n_time * n_baseline
    time = table.getcol("TIME")
    np.testing.assert_allclose(
        np.unique(time) - MJD_UNIX_OFFSET_SECONDS, ms_xds.time.values
    )
    np.testing.assert_allclose(table.getcol("INTERVAL"), 60.0)
    uvw = table.getcol("UVW").reshape(n_time, n_baseline, 3)
    np.testing.assert_allclose(uvw, ms_xds.UVW.values, rtol=1e-12)
    data = table.getcol("DATA").reshape(n_time, n_baseline, 2, 2)
    np.testing.assert_allclose(data, ms_xds.VISIBILITY.values, rtol=2e-7)
    assert not table.getcol("FLAG").any()
    np.testing.assert_allclose(table.getcol("WEIGHT"), 1.0)
    table.close()


def test_subtables(simulated_ms):
    from arcae.lib.arrow_tables import Table

    _, ms_path = simulated_ms
    antenna = Table.from_filename(f"{ms_path}::ANTENNA")
    assert antenna.nrow() == 6
    assert list(antenna.getcol("DISH_DIAMETER")) == [25.0] * 6
    antenna.close()
    spectral_window = Table.from_filename(f"{ms_path}::SPECTRAL_WINDOW")
    np.testing.assert_allclose(spectral_window.getcol("CHAN_FREQ")[0], [3e9, 3.01e9])
    assert spectral_window.getcol("NUM_CHAN")[0] == 2
    spectral_window.close()
    polarization = Table.from_filename(f"{ms_path}::POLARIZATION")
    assert list(polarization.getcol("CORR_TYPE")[0]) == [5, 8]  # RR, LL
    polarization.close()
    field = Table.from_filename(f"{ms_path}::FIELD")
    assert field.nrow() == 1
    field.close()


def test_overwrite_flag(simulated_ms):
    from astroviper.utils.measurement_set_v2 import write_measurement_set_v2

    ps_store, ms_path = simulated_ms
    with pytest.raises(FileExistsError):
        write_measurement_set_v2(ps_store, ms_path, overwrite=False)
    write_measurement_set_v2(ps_store, ms_path, overwrite=True)


def test_measurement_set_structure(simulated_ms):
    """The written set is a coherent casacore Measurement Set: the subtables are
    reachable through the MAIN table's keywords (the ``::`` syntax below) and
    carry consistent metadata (field direction, observation range, TaQL access)."""
    from arcae.lib.arrow_tables import Table

    _, ms_path = simulated_ms
    field = Table.from_filename(f"{ms_path}::FIELD")
    direction = field.getcol("PHASE_DIR")[0, 0]
    assert abs(np.rad2deg(direction[1]) - 40.7337) < 1e-3
    field.close()
    observation = Table.from_filename(f"{ms_path}::OBSERVATION")
    time_range = observation.getcol("TIME_RANGE")[0]
    assert time_range[1] - time_range[0] == pytest.approx(3 * 60.0)
    observation.close()
    selected = Table.from_taql(
        f"SELECT FROM '{ms_path}' WHERE ANTENNA1 == 0 AND ANTENNA2 == 1"
    )
    assert selected.nrow() == 3
    selected.close()
