"""Full-stack tests of two cycle control options on a small simulated
processing set: the iteration history (``record_iteration_history``) and
``max_cycles="auto"`` reach the merged convergence record of the distributed
application through the node tasks and the reduce, and the record still
saves to feather; bad values of the two are refused before the image store
is created.
"""

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.processing_functions.imaging.utils.cycle_control import (
    IMAGING_THRESHOLD,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    ITERATION_HISTORY_FIELDS,
    imaging_dict_to_dataframe,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
N_ANTENNA = 8
CYCLE_CONTROL_PARAMS = {
    "max_iter": 500,
    "max_cycles": "auto",
    "threshold": 0.02,
    "primary_beam_limit": 0.2,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.0,
    "max_iter_per_cycle": -1,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
    "record_iteration_history": True,
}


@pytest.fixture(scope="module")
def point_source_ps(tmp_path_factory):
    """1 Jy unpolarised point source at the phase centre, 8 VLA antennas, two
    channels; Stokes V is empty."""
    return distributed_applications.simulation.simulate_processing_set(
        ps_store=str(tmp_path_factory.mktemp("ps") / "point.ps.zarr"),
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
        beam_models=[airy_disk_model("vla_nvss")],
        beam_model_map=np.zeros(N_ANTENNA, int),
        n_time_chunks=1,
        n_frequency_chunks=2,
        overwrite=True,
    )["ps_store"]


def _image_cube(ps_store, image_store, cycle_control_params, deconvolver="hogbom"):
    """The two channels in two node tasks."""
    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    return distributed_applications.imaging.image_cube_single_field(
        ps_store=ps_store,
        image_store=str(image_store),
        image_params={
            "image_size": [64, 64],
            "cell_size": np.array([-8.0, 8.0]) * np.pi / (180 * 3600),
            "phase_direction": PC,
            "frequency_coords": frequency_coords,
            "polarization_coords": ["I", "V"],
            "time_coords": [0],
            "fft_padding": 1.2,
            "cpp_gridder": True,
        },
        instrument_polarization_basis="circular",
        imaging_weights_params={"weighting": "natural"},
        cycle_control_params=cycle_control_params,
        deconvolver=deconvolver,
        image_data_variables_keep=["sky_residual", "sky_model"],
        processing_set_data_group_name="base",
        single_precision_image=False,
        n_mapping_parallelism={"frequency": 2},
    )


def test_iteration_history_and_auto_max_cycles_reach_the_result(
    tmp_path, point_source_ps
):
    result = _image_cube(
        point_source_ps, tmp_path / "cube.img.zarr", CYCLE_CONTROL_PARAMS
    )
    record = result["deconvolution"]
    assert {(key.chan, key.pol) for key in record.data} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    for key, entry in record.data.items():
        for field in ITERATION_HISTORY_FIELDS:
            assert [len(cycle) for cycle in entry[field]] == entry["iter_done"]
        if key.pol == 0:  # Stokes I: cleaned to the threshold within its limit
            assert entry["stop_code"].imaging == IMAGING_THRESHOLD
            # the model updates and the closing stop test
            assert 2 <= len(entry["iter_done"]) <= entry["max_cycles_auto"] + 1
            assert entry["iter_done"][-1] == 0
            assert sum(entry["iter_done"]) > 0
        else:  # the empty Stokes V plane has nothing to predict
            assert entry["max_cycles_auto"] is None
    frame = imaging_dict_to_dataframe(record)
    frame.to_feather(tmp_path / "convergence.ft")
    back = pd.read_feather(tmp_path / "convergence.ft")
    assert len(back) == 4
    assert back["iteration_residual_rms"].map(len).tolist() == (
        frame["iter_done"].map(len).tolist()
    )


@pytest.mark.parametrize(
    ("cycle_control_params", "deconvolver", "match"),
    [
        ({}, "asp", "record_iteration_history is supported by the Hogbom"),
        ({"record_iteration_history": 1}, "hogbom", "must be a bool; got 1"),
        ({"max_cycles": "Auto"}, "hogbom", 'integer >= -1 or "auto"'),
        ({"max_cycles": 2.5}, "hogbom", 'integer >= -1 or "auto"'),
        ({"max_psf_fraction": 1.0}, "hogbom", "max_psf_fraction below 1"),
    ],
    ids=[
        "history_asp",
        "history_int",
        "max_cycles_Auto",
        "max_cycles_float",
        "auto_psf_fraction",
    ],
)
def test_refused_before_the_image_store_is_created(
    tmp_path, point_source_ps, cycle_control_params, deconvolver, match
):
    with pytest.raises(ValueError, match=match):
        _image_cube(
            point_source_ps,
            tmp_path / "cube.img.zarr",
            {**CYCLE_CONTROL_PARAMS, **cycle_control_params},
            deconvolver,
        )
    assert list(tmp_path.iterdir()) == []
