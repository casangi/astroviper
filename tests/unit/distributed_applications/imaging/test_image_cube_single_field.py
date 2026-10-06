"""Full-stack test of the image_cube_single_field distributed application on a
small simulated processing set."""

import numpy as np
from astropy.coordinates import SkyCoord
from xradio.image import load_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

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
    distributed_applications.imaging.image_cube_single_field(
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
