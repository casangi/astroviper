"""The single-field anti-alias gridding functions refuse a measurement set whose
number of polarizations differs from that of the grid they accumulate into.

The C++ kernels do not bounds-check the polarization map: a later measurement
set with more polarizations was gridded past the end of the grid and of its
normalization, and one with fewer into the grid's first planes.
"""

import numpy as np
import pytest
import xarray as xr
from xradio.image import make_empty_sky_image

from astroviper.processing_functions.imaging.add_visibility_grid import (
    add_visibility_grid_single_field,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    make_gridding_kernel,
)
from astroviper.processing_functions.imaging.make_point_spread_function import (
    add_uv_sampling_grid_single_field,
)

FREQUENCIES = np.array([1.00e9, 1.05e9])
CORRELATIONS = {2: ["XX", "YY"], 4: ["XX", "XY", "YX", "YY"]}
DIMS = ("time", "baseline_id", "frequency", "polarization")
GUARD = 8192
SENTINEL = 1234.5


def _ms(n_pol):
    rng = np.random.default_rng(n_pol)
    shape = (2, 10, FREQUENCIES.size, n_pol)
    uvw = rng.uniform(-300.0, 300.0, (2, 10, 3))
    uvw[..., 2] *= 0.05
    ms_xds = xr.Dataset(
        {
            "VISIBILITY": (DIMS, rng.standard_normal(shape) + 0j),
            "UVW": (("time", "baseline_id", "uvw_label"), uvw),
            "WEIGHT_IMAGING": (DIMS, np.ones(shape)),
        },
        coords={"frequency": FREQUENCIES, "polarization": CORRELATIONS[n_pol]},
    )
    ms_xds.attrs["data_groups"] = {
        "base": {
            "correlated_data": "VISIBILITY",
            "uvw": "UVW",
            "weight_imaging": "WEIGHT_IMAGING",
        }
    }
    return ms_xds


def _guard(img_xds):
    """Move each array of ``img_xds`` into a buffer that ends in a sentinel tail,
    so that a write past an array lands in its tail; return the tails."""
    tails = {}
    for name in img_xds.data_vars:
        values = img_xds[name].values
        buffer = np.full(values.size + GUARD, SENTINEL, dtype=values.dtype)
        buffer[: values.size] = values.ravel()
        img_xds.variables[name].data = buffer[: values.size].reshape(values.shape)
        assert np.shares_memory(img_xds[name].values, buffer)
        tails[name] = buffer[values.size :]
    return tails


@pytest.mark.parametrize(
    "function", [add_visibility_grid_single_field, add_uv_sampling_grid_single_field]
)
@pytest.mark.parametrize(("n_pol", "later_n_pol"), [(2, 4), (4, 2)])
def test_a_later_measurement_set_with_other_polarizations_is_refused(
    function, n_pol, later_n_pol
):
    """The later call raises before gridding: the grid, its normalization and
    the memory after them are unchanged."""
    kernel = make_gridding_kernel(None, 1.2)
    img_xds = make_empty_sky_image(
        phase_center=[0.0, 0.0],
        image_size=[24, 24],
        cell_size=[-2.0e-5, 2.0e-5],
        frequency_coords=FREQUENCIES,
        pol_coords=CORRELATIONS[n_pol],
        time_coords=[0],
        do_sky_coords=False,
    )
    img_xds.attrs["data_groups"] = {"residual": {}}
    function(_ms(n_pol), kernel, img_xds)
    tails = _guard(img_xds)
    before = {name: img_xds[name].values.copy() for name in tails}

    with pytest.raises(ValueError, match="polarization axis must match"):
        function(_ms(later_n_pol), kernel, img_xds)

    for name, tail in tails.items():
        np.testing.assert_array_equal(img_xds[name].values, before[name])
        assert np.all(tail == SENTINEL), name
