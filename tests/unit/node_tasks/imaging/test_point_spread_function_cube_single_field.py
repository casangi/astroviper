"""The PSF node task on a tiny in-memory chunk: the properties of every image
channel on the labelled beam dimension, without the PSF image when nothing is
written, and an image channel no visibility channel maps onto, made from the
whole chunk."""

import numpy as np
import xarray as xr

from astroviper.node_tasks.imaging.point_spread_function_cube_single_field import (
    point_spread_function_cube_single_field,
)

FREQUENCY = 1.4e9
UV_EXTENT = 150.0  # metres
IMAGE_PARAMS = {
    "image_size": [32, 32],
    "cell_size": np.array([-1.0e-4, 1.0e-4]),
    "phase_direction": np.array([1.0, 0.5]),
    "time_coords": [0],
    "polarization_coords": ["I", "Q"],
    "fft_padding": 1.2,
}


def _processing_set(n_time=8, n_antenna=4):
    """One MSv4 with a single channel and equal weights."""
    rng = np.random.default_rng(0)
    antennas = [f"a{i}" for i in range(n_antenna)]
    pairs = [(a, b) for i, a in enumerate(antennas) for b in antennas[i + 1 :]]
    n_baseline = len(pairs)
    shape = (n_time, n_baseline, 1, 2)
    uvw = np.zeros((n_time, n_baseline, 3))
    uvw[..., :2] = rng.uniform(-UV_EXTENT, UV_EXTENT, (n_time, n_baseline, 2))
    dims = ("time", "baseline_id", "frequency", "polarization")
    ms_xds = xr.Dataset(
        data_vars={
            "VISIBILITY": (dims, np.ones(shape, dtype=np.complex128)),
            "FLAG": (dims, np.zeros(shape, dtype=bool)),
            "WEIGHT": (dims, np.full(shape, 4.0)),
            "UVW": (("time", "baseline_id", "uvw_label"), uvw),
        },
        coords={
            "time": np.arange(n_time, dtype=float),
            "baseline_id": np.arange(n_baseline),
            "baseline_antenna1_name": ("baseline_id", [a for a, _ in pairs]),
            "baseline_antenna2_name": ("baseline_id", [b for _, b in pairs]),
            "frequency": [FREQUENCY],
            "polarization": ["XX", "YY"],
            "uvw_label": ["u", "v", "w"],
        },
        attrs={
            "data_groups": {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "uvw": "UVW",
                }
            }
        },
    )
    return xr.DataTree.from_dict({"ms_0": ms_xds})


def test_channel_properties_and_whole_chunk_fallback():
    frequencies = np.array([FREQUENCY, 1.05 * FREQUENCY])
    result = point_spread_function_cube_single_field(
        IMAGE_PARAMS,
        {"weighting": "natural"},
        task_coords={"frequency": {"data": frequencies, "slice": slice(0, 2)}},
        data_selection={"ms_0": {"frequency": slice(0, 1)}},
        input_data_store="unused.ps.zarr",
        processing_set_data_group_name="base",
        single_precision_image=False,
        fft_backend="scipy",
        task_id=5,
        input_data=_processing_set(),
    )
    properties = result["point_spread_function_properties"]
    assert "POINT_SPREAD_FUNCTION" not in properties
    np.testing.assert_array_equal(properties.frequency.values, frequencies)
    assert list(properties.polarization.values) == ["I", "Q"]
    beam = properties.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION
    assert beam.dims == ("frequency", "polarization", "beam_params_label")
    assert list(beam.beam_params_label.values) == ["major", "minor", "pa"]
    assert np.all(beam.sel(beam_params_label=["major", "minor"]).values > 0)
    np.testing.assert_allclose(properties.SENSITIVITY.values, 1 / np.sqrt(2 * 48 * 4.0))
    # No visibility channel maps onto the second image channel: the whole
    # chunk is gridded onto it, the same samples, so the same properties.
    for name, variable in properties.data_vars.items():
        np.testing.assert_allclose(
            variable.isel(frequency=1),
            variable.isel(frequency=0),
            rtol=1e-12,
            err_msg=name,
        )
    timing = result["timing_node_tasks"]
    assert len(timing) == 1
    assert timing["task_id"].iloc[0] == 5
    assert timing["n_channels"].iloc[0] == 2
    assert timing["T_write"].iloc[0] >= 0.0
