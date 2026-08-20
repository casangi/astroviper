"""Full-stack tests of the simulate_processing_set distributed application."""

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from xradio.measurement_set import load_processing_set, open_processing_set
from xradio.schema.check import check_datatree

import astroviper.distributed_applications as distributed_applications
from astroviper.processing_functions.simulation import (
    simulate_processing_set as simulate_processing_set_pf,
)
from astroviper.utils.beam_models import (
    airy_disk_model,
    read_aperture_polynomial_coefficients,
)
from astroviper.utils.telescope_layout import (
    observatory_position,
    read_telescope_layout,
)

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
SOURCE = SkyCoord(ra="19h59m50.51793355s", dec="+40d48m11.3694551s", frame="fk5")
PC = np.array([[PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad]])
SRC = np.array([[[SOURCE.ra.rad, SOURCE.dec.rad]]])
TIME_PARAMS = {
    "time_start": "2019-10-03T19:00:00.000",
    "time_delta": 1800.0,
    "n_samples": 4,
}
FREQ_PARAMS = {
    "freq_start": 3e9,
    "freq_delta": 0.2e9,
    "n_channels": 3,
    "channel_width": 1e7,
    "spectral_window_name": "SBand",
}


def run(tmp_path, **overrides):
    ant = read_telescope_layout("vla.d", antenna_selection=list(range(8)))
    kwargs = dict(
        ps_store=str(tmp_path / "vla_sim.ps.zarr"),
        antenna_xds=ant,
        time_params=TIME_PARAMS,
        frequency_params=FREQ_PARAMS,
        polarization=["RR", "LL"],
        point_source_flux=np.array([1.0, 0, 0, 1.0])[None, None, None, :],
        point_source_ra_dec=SRC,
        phase_center_ra_dec=PC,
        beam_models=[airy_disk_model("vla")],
        beam_model_map=np.zeros(8, int),
        n_time_chunks=2,
        n_frequency_chunks=3,
        overwrite=True,
    )
    kwargs.update(overrides)
    return distributed_applications.simulation.simulate_processing_set(**kwargs), kwargs


def test_full_stack_matches_processing_function(tmp_path):
    result, kwargs = run(tmp_path)
    assert isinstance(result["timing_node_tasks"], pd.DataFrame)
    assert len(result["timing_node_tasks"]) == 6
    assert list(result["timing_node_tasks"]["task_id"]) == list(range(6))
    assert result["ms_name"] == "VLA_SBand"
    assert result["timing_distributed_application"]["T_total"] > 0

    ps_xdt = open_processing_set(result["ps_store"])
    issues = check_datatree(ps_xdt)  # XRADIO MSv4 schema checker
    assert str(issues) == "No schema issues found", str(issues)
    ms = load_processing_set(result["ps_store"])["VLA_SBand"].ds
    assert ms.sizes == {
        "time": 4,
        "baseline_id": 28,
        "frequency": 3,
        "polarization": 2,
        "uvw_label": 3,
    }
    assert not np.isnan(ms.VISIBILITY.values).any()
    assert list(ms.polarization.values) == ["RR", "LL"]
    assert ms.frequency.attrs["spectral_window_name"] == "SBand"
    assert ms.frequency.attrs["channel_width"]["data"] == 1e7
    assert ms.time.attrs["integration_time"]["data"] == 1800.0
    assert ms.field_name.values[0] == "field_0"
    np.testing.assert_array_equal(ms.WEIGHT.values, 1.0)
    assert not ms.FLAG.values.any()

    # identical to the processing function on the full axes
    ant = kwargs["antenna_xds"]
    ref, _ = simulate_processing_set_pf(
        ms.time.values, ms.frequency.values, ["RR", "LL"], ant.ANTENNA_POSITION.values,
        observatory_position("VLA"), kwargs["point_source_flux"], SRC, PC, [airy_disk_model("vla")], np.zeros(8, int),
    )  # fmt: skip
    np.testing.assert_allclose(ms.VISIBILITY.values, ref.VISIBILITY.values, atol=1e-12)
    np.testing.assert_allclose(ms.UVW.values, ref.UVW.values, atol=1e-9)
    # antenna and field sub-datasets
    field = ps_xdt["VLA_SBand"]["field_and_source_base_xds"].ds
    np.testing.assert_allclose(field.FIELD_PHASE_CENTER_DIRECTION.values, PC)
    assert field.FIELD_PHASE_CENTER_DIRECTION.attrs["frame"] == "icrs"
    assert ps_xdt["VLA_SBand"]["antenna_xds"].ds.sizes["antenna_name"] == 8


def test_mosaic_fields_noise_and_zernike(tmp_path):
    pc2 = PC + np.array([[0.0, 2e-4]])
    phase_centers = np.concatenate([PC, PC, pc2, pc2])
    zpc = read_aperture_polynomial_coefficients("EVLA_avg_zcoeffs_SBand_lookup")
    result, kwargs = run(
        tmp_path,
        phase_center_ra_dec=phase_centers,
        field_name=["A", "A", "B", "B"],
        polarization=["RR", "RL", "LR", "LL"],
        beam_models=[zpc, airy_disk_model("vla")],
        beam_model_map=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        beam_params={"image_size": [128, 128], "mueller_selection": np.arange(16)},
        noise_params={"t_receiver": 50.0, "random_seed": 3},
        n_time_chunks=2,
        n_frequency_chunks=1,
        ms_name="mosaic",
    )
    ps_xdt = open_processing_set(result["ps_store"])
    issues = check_datatree(ps_xdt)  # XRADIO MSv4 schema checker
    assert str(issues) == "No schema issues found", str(issues)
    ms = load_processing_set(result["ps_store"])["mosaic"].ds
    assert list(ms.field_name.values) == ["A", "A", "B", "B"]
    field = ps_xdt["mosaic"]["field_and_source_base_xds"].ds
    assert list(field.field_name.values) == ["A", "B"]
    np.testing.assert_allclose(
        field.FIELD_PHASE_CENTER_DIRECTION.values, np.concatenate([PC, pc2])
    )
    assert ms.sizes["polarization"] == 4
    assert np.all(ms.WEIGHT.values > 0) and np.all(ms.WEIGHT.values < 1e6)
    assert not np.isnan(ms.VISIBILITY.values).any()
    # noise is reproducible for a fixed seed
    result2, _ = run(
        tmp_path,
        ps_store=str(tmp_path / "again.ps.zarr"),
        phase_center_ra_dec=phase_centers,
        field_name=["A", "A", "B", "B"],
        polarization=["RR", "RL", "LR", "LL"],
        beam_models=[zpc, airy_disk_model("vla")],
        beam_model_map=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        beam_params={"image_size": [128, 128], "mueller_selection": np.arange(16)},
        noise_params={"t_receiver": 50.0, "random_seed": 3},
        n_time_chunks=2,
        n_frequency_chunks=1,
        ms_name="mosaic",
    )
    ms2 = load_processing_set(result2["ps_store"])["mosaic"].ds
    np.testing.assert_array_equal(ms.VISIBILITY.values, ms2.VISIBILITY.values)


def test_automatic_chunking_and_validation_errors(tmp_path):
    result, _ = run(
        tmp_path,
        n_time_chunks=None,
        n_frequency_chunks=None,
        thread_info={"n_threads": 2, "memory_per_thread": 4.0},
    )
    assert len(result["timing_node_tasks"]) >= 1
    with pytest.raises(ValueError):
        run(tmp_path, beam_model_map=np.zeros(3, int))
    with pytest.raises(ValueError):
        run(tmp_path, point_source_flux=np.ones((1, 2, 1, 4)))  # time axis 2 != 1 or 4
    with pytest.raises(ValueError):
        run(tmp_path, polarization=["RR", "XX"])
    with pytest.raises(FileExistsError):
        run(tmp_path, overwrite=False)
    with pytest.raises(AssertionError):  # toolviper schema: unknown implementation
        run(tmp_path, implementation="fortran")
