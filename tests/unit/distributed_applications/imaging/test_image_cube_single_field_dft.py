"""Full-stack tests of the direct Fourier transform option of the cube imager.

``gridding_kernel_params={"type": "dft"}`` reaches the residual updates and the
point spread function of every node task: on a small simulated processing set
(two polarized point sources on pixel centres, four correlations, two channels in
two node tasks, the w term on) the clean recovers the apparent fluxes to 1
percent of the brightest (the criterion of the component test of flux recovery)
and writes the outputs of the gridder path. Kernel keys that do not apply to
the DFT, and kept uv grids, which it does not make, are refused before the image
store is created.
"""

import numpy as np
import pytest
from xradio.image import load_image, make_empty_sky_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import inverse_sin_project
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
CELL = 0.7 * np.pi / (180 * 3600)
IMAGE_SIZE = [64, 64]
# Stokes I, Q, U, V (Jy) and the pixel offset from the centre of each source
SOURCES = [
    (np.array([1.0, 0.1, -0.05, 0.02]), (13, -9)),
    (np.array([0.4, -0.04, 0.03, 0.0]), (-10, 7)),
]
CYCLE_CONTROL_PARAMS = {
    "max_iter": 4000,
    "max_cycles": 15,
    "threshold": 1e-5,
    "primary_beam_limit": 0.2,
    "gain": 0.2,
    "psf_sidelobe_factor": 1.0,
    "max_iter_per_cycle": 300,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}


def _pixel(offset):
    return IMAGE_SIZE[0] // 2 + offset[0], IMAGE_SIZE[1] // 2 + offset[1]


@pytest.fixture(scope="module")
def point_sources_ps(tmp_path_factory):
    img_xds = make_empty_sky_image(
        phase_center=PHASE_CENTER,
        image_size=IMAGE_SIZE,
        cell_size=[-CELL, CELL],
        frequency_coords=[100e9],
        pol_coords=["I"],
        time_coords=[0],
        do_sky_coords=False,
    )
    components = []
    for stokes, offset in SOURCES:
        i, j = _pixel(offset)
        lm = np.array([img_xds.l.values[i], img_xds.m.values[j]])
        flux = [
            stokes[0] + stokes[1],
            stokes[2] + 1j * stokes[3],
            stokes[2] - 1j * stokes[3],
            stokes[0] - stokes[1],
        ]
        components.append(
            {
                "kind": "point",
                "flux": np.array(flux),
                "ra_dec": inverse_sin_project(PHASE_CENTER, lm),
            }
        )
    antenna_xds = read_telescope_layout(
        "alma.cycle8.1", antenna_selection=list(range(8))
    )
    return distributed_applications.simulation.simulate_processing_set(
        ps_store=str(tmp_path_factory.mktemp("ps") / "points.ps.zarr"),
        antenna_xds=antenna_xds,
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 600.0,
            "n_samples": 6,
        },
        frequency_params={
            "freq_start": 100e9,
            "freq_delta": 0.5e9,
            "n_channels": 2,
            "channel_width": 2e6,
        },
        polarization=["XX", "XY", "YX", "YY"],
        sky_components=components,
        phase_center_ra_dec=PHASE_CENTER[None, :],
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
        n_time_chunks=1,
        n_frequency_chunks=1,
        overwrite=True,
    )["ps_store"]


def _image_cube(ps_store, image_store, gridding_kernel_params, keep=None):
    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    return distributed_applications.imaging.image_cube_single_field(
        ps_store=ps_store,
        image_store=str(image_store),
        image_params={
            "image_size": IMAGE_SIZE,
            "cell_size": np.array([-CELL, CELL]),
            "phase_direction": PHASE_CENTER,
            "frequency_coords": frequency_coords,
            "polarization_coords": ["I", "Q", "U", "V"],
            "time_coords": [0],
            "fft_padding": 1.2,
        },
        imaging_weights_params={"weighting": "natural"},
        cycle_control_params=CYCLE_CONTROL_PARAMS,
        image_data_variables_keep=keep
        or ["sky_residual", "sky_model", "point_spread_function", "primary_beam"],
        processing_set_data_group_name="base",
        single_precision_image=False,
        n_mapping_parallelism={"frequency": 2},
        gridding_kernel_params=gridding_kernel_params,
        processing_function_threads=2,
    )


def test_the_dft_cube_cleans_the_point_sources(tmp_path, point_sources_ps):
    _image_cube(point_sources_ps, tmp_path / "dft.img.zarr", {"type": "dft"})
    _image_cube(point_sources_ps, tmp_path / "grid.img.zarr", None)
    dft = load_image(str(tmp_path / "dft.img.zarr"))
    grid = load_image(str(tmp_path / "grid.img.zarr"))
    primary_beam = dft["PRIMARY_BEAM"].values[0, :, 0]
    np.testing.assert_array_equal(
        dft["POINT_SPREAD_FUNCTION"].values[0, :, :, 32, 32], 1.0
    )
    for stokes, offset in SOURCES:
        i, j = _pixel(offset)
        for chan in range(2):
            model = dft["SKY_MODEL"].values[0, chan, :, i - 1 : i + 2, j - 1 : j + 2]
            np.testing.assert_allclose(
                model.sum(axis=(1, 2)),
                stokes * primary_beam[chan, i, j],
                rtol=0,
                atol=0.01 * SOURCES[0][0][0] * primary_beam[chan, i, j],
            )
    # the outputs of the gridder path: the same variables, dimensions and types
    assert sorted(dft.data_vars) == sorted(grid.data_vars)
    for name in dft.data_vars:
        assert dft[name].dims == grid[name].dims, name
        assert dft[name].shape == grid[name].shape, name
        assert dft[name].dtype == grid[name].dtype, name


@pytest.mark.parametrize(
    ("gridding_kernel_params", "keep", "match"),
    [
        ({"type": "dft", "support": 7}, None, "do not apply to the type 'dft'"),
        (
            {"type": "dft", "oversampling": 100, "interpolation": "nearest"},
            None,
            "do not apply to the type 'dft'",
        ),
        ({"type": "fft"}, None, "is not one of"),
        ({"type": "dft"}, ["sky_residual", "visibility"], "uv grids"),
        ({"type": "dft"}, ["uv_sampling", "sky_model"], "uv grids"),
    ],
    ids=["support", "oversampling_interpolation", "unknown_type", "visibility", "uv"],
)
def test_refused_before_the_image_store_is_created(
    tmp_path, point_sources_ps, gridding_kernel_params, keep, match
):
    with pytest.raises(ValueError, match=match):
        _image_cube(
            point_sources_ps, tmp_path / "cube.img.zarr", gridding_kernel_params, keep
        )
    assert list(tmp_path.iterdir()) == []
