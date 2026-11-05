"""Shared minimal datasets for the prolate spheroidal degridding unit tests.

Used by ``test_get_visibility_grid.py``, ``test_degrid_visibility_grid.py``
and ``utils/test_frequency_mapping.py`` (not collected by pytest: the file
name does not start with ``test_``).
"""

import numpy as np
import xarray as xr

# Registers the `xr_img` accessor used by the functions under test.
import xradio.image.image_xds  # noqa: F401

SUPPORT = 7
OVERSAMPLING = 100


def build_datasets(
    n_l=80,
    n_m=80,
    n_time=1,
    n_baseline=16,
    n_chan=2,
    n_pol=2,
    fft_padding=1.2,
    delta=2.0e-5,
    uv_extent=20.0,
    sky_value=2.0 + 0.0j,
    seed=0,
    visibility_frequencies=None,
    image_frequencies=None,
):
    """Build a minimal (ms_xds, img_xds, n_uv) triple for the degridder.

    `img_xds` holds a UV-domain model grid (not a sky image) with shape
    `(time, frequency, polarization, u, v)`. By default the image frequency
    axis equals the visibility axis; pass ``image_frequencies`` to build an
    image with more (or offset) channels than the measurement set.
    """
    rng = np.random.default_rng(seed)
    n_uv = (fft_padding * np.array([n_l, n_m])).astype(int)

    # l/m coordinates centred on zero so `get_lm_cell_size` returns `delta`.
    l_coord = (np.arange(n_l) - n_l / 2) * delta
    m_coord = (np.arange(n_m) - n_m / 2) * delta
    if visibility_frequencies is None:
        visibility_frequencies = np.linspace(1.0e9, 1.1e9, n_chan)
    visibility_frequencies = np.asarray(visibility_frequencies, dtype=np.float64)
    n_chan = visibility_frequencies.size
    if image_frequencies is None:
        image_frequencies = visibility_frequencies
    image_frequencies = np.asarray(image_frequencies, dtype=np.float64)

    uvw = np.concatenate(
        [
            rng.uniform(-uv_extent, uv_extent, (n_time, n_baseline, 2)),
            np.zeros((n_time, n_baseline, 1)),
        ],
        axis=-1,
    )

    ms_xds = xr.Dataset(
        data_vars={
            "VISIBILITY": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.zeros((n_time, n_baseline, n_chan, n_pol), dtype=np.complex128),
            ),
            "UVW": (("time", "baseline_id", "uvw_label"), uvw),
            "WEIGHT_IMAGING": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.ones((n_time, n_baseline, n_chan, n_pol)),
            ),
        },
        coords={"frequency": visibility_frequencies},
    )
    ms_xds.attrs["data_groups"] = {
        "base": {
            "correlated_data": "VISIBILITY",
            "uvw": "UVW",
            "weight_imaging": "WEIGHT_IMAGING",
        }
    }

    # UV model grid: shape (m_time, m_chan, m_pol, n_u, n_v)
    sky_model = np.full(
        (n_time, image_frequencies.size, n_pol, int(n_uv[0]), int(n_uv[1])),
        sky_value,
        dtype=np.complex128,
    )
    img_xds = xr.Dataset(
        data_vars={
            "SKY_MODEL": (
                ("time", "frequency", "polarization", "u", "v"),
                sky_model,
            ),
        },
        coords={"l": l_coord, "m": m_coord, "frequency": image_frequencies},
    )
    img_xds.attrs["type"] = "image_dataset"
    # get_visibility_grid_single_field degrids the image-side "visibility" uv
    # grid (default input data group "model") into ms model visibilities.
    img_xds.attrs["data_groups"] = {
        "model": {"visibility": "SKY_MODEL"},
    }

    return ms_xds, img_xds, n_uv
