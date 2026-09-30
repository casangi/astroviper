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


def test_disk_sources_through_the_driver_match_the_processing_function(tmp_path):
    """Limb-darkened disks are sliced per chunk and reach the processing function unchanged."""
    from astroviper.processing_functions.simulation import (
        limb_darkened_disk_uv_response,
    )

    arcsec = np.pi / (180 * 3600)
    disk_flux = np.array([[[[2.0, 0, 0, 2.0]]], [[[0.5, 0, 0, 0.5]]]])
    disk_ra_dec = np.concatenate([SRC, SRC + 1e-4], axis=1)  # [1, 2, 2]
    disk_shape = np.array(
        [[300 * arcsec, 200 * arcsec, 0.4], [120 * arcsec, 120 * arcsec, 0.0]]
    )
    limb_darkening = [1.0, -1.0]
    result, kwargs = run(
        tmp_path,
        point_source_flux=np.zeros((1, 1, 1, 4)),
        disk_source_flux=disk_flux,
        disk_source_ra_dec=disk_ra_dec,
        disk_source_shape=disk_shape,
        disk_source_limb_darkening=limb_darkening,
    )
    ms = load_processing_set(result["ps_store"])["VLA_SBand"].ds
    assert (
        "2 limb-darkened disk source(s)"
        in ms.attrs["data_groups"]["base"]["description"]
    )
    ant = kwargs["antenna_xds"]
    ref, _ = simulate_processing_set_pf(
        ms.time.values, ms.frequency.values, ["RR", "LL"], ant.ANTENNA_POSITION.values,
        observatory_position("VLA"), np.zeros((1, 1, 1, 4)), SRC, PC, [airy_disk_model("vla")], np.zeros(8, int),
        disk_source_flux=disk_flux, disk_source_ra_dec=disk_ra_dec, disk_source_shape=disk_shape,
        disk_source_limb_darkening=np.array(limb_darkening),
    )  # fmt: skip
    np.testing.assert_allclose(ms.VISIBILITY.values, ref.VISIBILITY.values, atol=1e-12)
    # the disks are resolved: the response departs from unity on the longer baselines
    u = ms.UVW.values[..., 0, None] * ms.frequency.values / 299792458.0
    v = ms.UVW.values[..., 1, None] * ms.frequency.values / 299792458.0
    assert limb_darkened_disk_uv_response(u, v, *disk_shape[0], 1.0).min() < 0.9

    with pytest.raises(ValueError, match="disk_source_shape"):
        run(tmp_path, disk_source_flux=disk_flux, disk_source_ra_dec=disk_ra_dec,
            disk_source_shape=disk_shape[:1])  # fmt: skip
    with pytest.raises(ValueError, match="limb_darkening"):
        run(tmp_path, disk_source_flux=disk_flux, disk_source_ra_dec=disk_ra_dec,
            disk_source_shape=disk_shape, disk_source_limb_darkening=[0.0, -3.0])  # fmt: skip
    with pytest.raises(ValueError, match="given together"):
        run(tmp_path, disk_source_flux=disk_flux)


def test_sky_components_through_the_driver(tmp_path):
    """The unified component list is validated once, sliced per chunk and reaches the kernel intact."""
    from astroviper.processing_functions.simulation import (
        normalize_sky_components,
        sky_components_from_arrays,
    )

    arcsec = np.pi / (180 * 3600)
    n_time, n_frequency = TIME_PARAMS["n_samples"], FREQ_PARAMS["n_channels"]
    # a time- and frequency-dependent point source, a SkyCoord-positioned annulus and
    # an m-ring given through the list; the bulk point-source arrays stay usable alongside
    spectrum = (
        np.ones((n_time, n_frequency, 4)) * np.arange(1, n_time + 1)[:, None, None]
    )
    spectrum[..., 1:3] = 0.0
    components = [
        {"kind": "point", "flux": spectrum, "ra_dec": SRC[0, 0]},
        {"kind": "annulus", "flux": 2.0, "ra_dec": SOURCE, "radius": 200 * arcsec,
         "inner_radius": 120 * arcsec, "inclination": 0.4, "pa": 1.0, "fwhm": 20 * arcsec},
        {"kind": "m_ring", "flux": [1.0, 0.0, 0.0, 0.8], "ra_dec": SRC[0, 0] + 1e-4,
         "radius": 150 * arcsec, "beta": [0.3j], "fwhm": 40 * arcsec},
    ]  # fmt: skip
    result, kwargs = run(
        tmp_path,
        point_source_flux=np.array([0.5, 0, 0, 0.5])[None, None, None, :],
        point_source_ra_dec=SRC + 2e-4,
        sky_components=components,
    )
    ms = load_processing_set(result["ps_store"])["VLA_SBand"].ds
    description = ms.attrs["data_groups"]["base"]["description"]
    assert "2 point source(s), 1 annulus source(s), 1 m-ring source(s)" in description
    ant = kwargs["antenna_xds"]
    all_components = sky_components_from_arrays(
        "point", kwargs["point_source_flux"], kwargs["point_source_ra_dec"]
    ) + normalize_sky_components(components, n_time, n_frequency, "icrs")
    ref, _ = simulate_processing_set_pf(
        ms.time.values, ms.frequency.values, ["RR", "LL"], ant.ANTENNA_POSITION.values,
        observatory_position("VLA"), None, None, PC, [airy_disk_model("vla")], np.zeros(8, int),
        sky_components=all_components,
    )  # fmt: skip
    np.testing.assert_allclose(ms.VISIBILITY.values, ref.VISIBILITY.values, atol=1e-12)
    assert np.abs(ms.VISIBILITY.values).max() > 3.0  # the sources are there
    # the chunks saw different times: the point-source spectrum ramps with time
    first = np.abs(ms.VISIBILITY.values[0, :, 0, 0]).mean()
    last = np.abs(ms.VISIBILITY.values[-1, :, 0, 0]).mean()
    assert last > first

    with pytest.raises(ValueError, match="no sources"):
        run(tmp_path, point_source_flux=None, point_source_ra_dec=None)
    with pytest.raises(ValueError, match="unknown sky component kind"):
        run(
            tmp_path,
            sky_components=[{"kind": "blob", "flux": 1.0, "ra_dec": SRC[0, 0]}],
        )
    with pytest.raises(ValueError, match="frequency axis"):
        run(
            tmp_path,
            sky_components=[
                {"kind": "point", "flux": np.ones((5, 4)), "ra_dec": SRC[0, 0]}
            ],
        )


def test_sky_image_output(tmp_path):
    """sky_image_params writes the simulated sky on the imager's grid as Stokes planes in Jy/pixel."""
    from xradio.image import load_image

    from astroviper.utils.coordinate_transforms import celestial_coord_to_sin_pixel

    arcsec = np.pi / (180 * 3600)
    image_size = [96, 96]
    cell_size = [-10 * arcsec, 10 * arcsec]
    second = PC[0]  # at the phase centre, far from the point source
    components = [
        # circular basis [RR, RL, LR, LL]: I = 2, Q = 0.4, U = 0, V = 1
        {"kind": "point", "flux": [3.0, 0.4, 0.4, 1.0], "ra_dec": SRC[0, 0]},
        {"kind": "gaussian", "flux": 2.0, "ra_dec": second, "major": 60 * arcsec,
         "minor": 40 * arcsec, "pa": 0.3},
    ]  # fmt: skip
    store = str(tmp_path / "sky.img.zarr")
    sky_image_params = {
        "image_store": store,
        "image_size": image_size,
        "cell_size": cell_size,
        "polarization_coords": ["I", "Q", "U", "V"],
    }
    result, _ = run(
        tmp_path,
        point_source_flux=None,
        point_source_ra_dec=None,
        sky_components=components,
        sky_image_params=sky_image_params,
    )
    assert result["sky_image_store"] == store
    assert result["timing_distributed_application"]["T_write_sky_image"] >= 0.0
    img = load_image(store)
    assert img.SKY.dims == ("time", "frequency", "polarization", "l", "m")
    assert list(img.polarization.values) == ["I", "Q", "U", "V"]
    assert img.sizes["frequency"] == FREQ_PARAMS["n_channels"]
    assert (img.sizes["l"], img.sizes["m"]) == tuple(image_size)
    assert img.attrs["data_groups"]["base"]["sky"] == "SKY"
    sky = img.SKY.values[0]  # [frequency, polarization, l, m]
    # the point source sits on the pixel the imaging convention predicts, with its Stokes fluxes
    i, j = np.round(
        celestial_coord_to_sin_pixel(PC[0], image_size, cell_size, SRC[0, 0])
    ).astype(int)
    np.testing.assert_allclose(sky[:, :, i, j], [[2.0, 0.4, 0.0, 1.0]] * 3, atol=1e-12)
    # the (unpolarised) Gaussian integrates to its flux in Stokes I only
    rest = sky.copy()
    rest[:, :, i, j] = 0.0
    np.testing.assert_allclose(
        rest.sum(axis=(2, 3)), [[2.0, 0.0, 0.0, 0.0]] * 3, atol=1e-3
    )
    # the store is a readable XRADIO image with the simulated frequency axis
    ms = load_processing_set(result["ps_store"])["VLA_SBand"].ds
    np.testing.assert_allclose(img.frequency.values, ms.frequency.values)
    with pytest.raises(ValueError, match="polarization_coords"):
        run(
            tmp_path,
            sky_image_params={**sky_image_params, "polarization_coords": ["XX"]},
        )
    with pytest.raises(ValueError, match="unknown keys"):
        run(tmp_path, sky_image_params={**sky_image_params, "cellsize": 1.0})
    with pytest.raises(ValueError, match="time_index"):
        run(tmp_path, sky_image_params={**sky_image_params, "time_index": 99})
