"""Tests of the point_spread_function_cube_single_field driver: the reduce,
the channels in the image's order whatever order the reduce leaves them in,
and the refusal of a Measurement Set v2."""

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord

import astroviper.distributed_applications as distributed_applications
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
N_ANTENNA = 6
# The module itself (the package re-exports the function under its name).
psf_module = importlib.import_module(
    "astroviper.distributed_applications.imaging.point_spread_function_cube_single_field"
)


def _image_params(frequency_coords):
    return {
        "image_size": [32, 32],
        "cell_size": np.array([-8.0, 8.0]) * np.pi / (180 * 3600),
        "phase_direction": PC,
        "frequency_coords": np.asarray(frequency_coords),
        "polarization_coords": ["I", "V"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }


def _chunk_result(task_id, frequency):
    return {
        "timing_node_tasks": pd.DataFrame({"task_id": [task_id]}),
        "point_spread_function_properties": xr.Dataset(
            {"SENSITIVITY": ("frequency", np.asarray(frequency) / 1e9)},
            coords={"frequency": frequency},
        ),
    }


def test_combine_point_spread_function_results_composes():
    """Concatenates in the order given and composes under tree reduction."""
    combine = psf_module.combine_point_spread_function_results
    pair = combine([_chunk_result(2, [5e9, 6e9]), _chunk_result(0, [1e9, 2e9])], {})
    result = combine([pair, _chunk_result(1, [3e9])], {})
    assert list(result["timing_node_tasks"]["task_id"]) == [2, 0, 1]
    properties = result["point_spread_function_properties"]
    np.testing.assert_array_equal(properties.frequency, [5e9, 6e9, 1e9, 2e9, 3e9])
    np.testing.assert_array_equal(properties.SENSITIVITY, [5, 6, 1, 2, 3])


@pytest.fixture(scope="module")
def ps_store(tmp_path_factory):
    """No source, 6 VLA antennas, 3 channels."""
    result = distributed_applications.simulation.simulate_processing_set(
        ps_store=str(tmp_path_factory.mktemp("psf_driver") / "empty.ps.zarr"),
        antenna_xds=read_telescope_layout(
            "vla.d", antenna_selection=list(range(N_ANTENNA))
        ),
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 1800.0,
            "n_samples": 2,
        },
        frequency_params={
            "freq_start": 3e9,
            "freq_delta": 0.2e9,
            "n_channels": 3,
            "channel_width": 1e7,
        },
        polarization=["RR", "LL"],
        point_source_flux=np.zeros((1, 1, 1, 4)),
        point_source_ra_dec=PC[None, None, :],
        phase_center_ra_dec=PC[None, :],
        beam_models=[airy_disk_model("vla_nvss")],
        beam_model_map=np.zeros(N_ANTENNA, int),
        n_time_chunks=1,
        n_frequency_chunks=3,
        overwrite=True,
    )
    return result["ps_store"]


def test_channels_in_the_image_order(ps_store, monkeypatch):
    """A reduce that reverses its inputs: the channels still come back in the
    order of the image's frequency coordinate."""
    from xradio.measurement_set import open_processing_set

    combine = psf_module.combine_point_spread_function_results
    monkeypatch.setattr(
        psf_module,
        "combine_point_spread_function_results",
        lambda input_data, input_params: combine(input_data[::-1], input_params),
    )
    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    properties = psf_module.point_spread_function_cube_single_field(
        ps_store,
        _image_params(frequency_coords),
        {"weighting": "natural"},
        instrument_polarization_basis="circular",
        processing_set_data_group_name="base",
        n_mapping_parallelism={"frequency": 3},
    )
    np.testing.assert_array_equal(properties.frequency.values, frequency_coords)
    # The uv extent in wavelengths grows with the channel frequency.
    assert np.all(np.diff(properties.UV_DISTANCE_MAX.values) > 0)


def test_measurement_set_v2_is_refused(tmp_path):
    ms_v2 = tmp_path / "data.ms"
    ms_v2.mkdir()
    (ms_v2 / "table.dat").write_bytes(b"")
    with pytest.raises(ValueError, match="is a Measurement Set v2"):
        psf_module.point_spread_function_cube_single_field(
            str(ms_v2), _image_params([3e9]), {"weighting": "natural"}
        )
