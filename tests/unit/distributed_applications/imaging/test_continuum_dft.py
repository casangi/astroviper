"""The direct Fourier transform option of the image_continuum_single_field application.

``gridding_kernel_params={"type": "dft"}`` makes the MFS Taylor residuals, the Taylor
PSFs and the model visibilities of every map task by the exact direct Fourier
transform; the reduce adds the Taylor images and the append node divides them by the
sum of the imaging weights. Small simulated processing sets (6 ALMA antennas, 4
times, no noise, the w term on) hold two polarized point sources, one at the phase
centre and one off centre.
"""

import dask
import numpy as np
import pytest
from xradio.image import load_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

DFT = {"type": "dft"}
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
CELL = 0.7 * np.pi / (180 * 3600)
IMAGE_SIZE = [64, 64]
# linear feeds: I, Q, U, V = 1, 0.2, 0.1, 0.05 and 0.5, -0.05, 0.1, -0.02
SOURCES = [
    (np.array([1.2, 0.1 + 0.05j, 0.1 - 0.05j, 0.8]), (0.0, 0.0)),
    (np.array([0.45, 0.1 - 0.02j, 0.1 + 0.02j, 0.55]), (12.0, -8.0)),
]
CONTROLS = {
    "max_iter": 1000,
    "max_cycles": 2,
    "threshold": 0.0,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.5,
    "max_iter_per_cycle": 30,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}
KEEP = ["sky_residual", "sky_model", "point_spread_function"]


@pytest.fixture(scope="module")
def processing_sets(tmp_path_factory):
    """``four``: one channel at 100 GHz, ``XX, XY, YX, YY``; ``two_hands``:
    two channels at 100 and 100.5 GHz, ``XX, YY``."""
    work_dir = tmp_path_factory.mktemp("continuum_dft")
    antenna_xds = read_telescope_layout(
        "alma.cycle8.1", antenna_selection=list(range(6))
    )
    sky_components = [
        {
            "kind": "point",
            "flux": flux,
            "ra_dec": PHASE_CENTER
            + np.deg2rad(np.array(offset) / 3600.0)
            / np.array([np.cos(PHASE_CENTER[1]), 1.0]),
        }
        for flux, offset in SOURCES
    ]
    stores = {}
    for name, polarization, n_channels in (
        ("four", ["XX", "XY", "YX", "YY"], 1),
        ("two_hands", ["XX", "YY"], 2),
    ):
        stores[name] = str(work_dir / f"{name}.ps.zarr")
        with dask.config.set(scheduler="synchronous"):
            distributed_applications.simulation.simulate_processing_set(
                ps_store=stores[name],
                antenna_xds=antenna_xds,
                time_params={
                    "time_start": "2019-10-03T19:00:00.000",
                    "time_delta": 600.0,
                    "n_samples": 4,
                },
                frequency_params={
                    "freq_start": 100e9,
                    "freq_delta": 0.5e9,
                    "n_channels": n_channels,
                    "channel_width": 2e6,
                },
                polarization=polarization,
                sky_components=sky_components,
                phase_center_ra_dec=PHASE_CENTER[None, :],
                beam_models=[airy_disk_model("alma")],
                beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
                n_time_chunks=1,
                n_frequency_chunks=1,
                overwrite=True,
            )
    return stores


def _image_params(ps_store, polarization_coords, n_taylor_terms=1):
    frequency = np.asarray(open_processing_set(ps_store).xr_ps.get_freq_axis().values)
    return {
        "image_size": IMAGE_SIZE,
        "cell_size": np.array([-CELL, CELL]),
        "phase_direction": PHASE_CENTER,
        "frequency_coords": frequency,
        "polarization_coords": polarization_coords,
        "time_coords": [0],
        "fft_padding": 1.2,
        "n_taylor_terms": n_taylor_terms,
        "reference_frequency": float(np.mean(frequency)),
    }


def _continuum(ps_store, image_store, image_params, **kwargs):
    arguments = {
        "imaging_weights_params": {"weighting": "natural"},
        "cycle_control_params": CONTROLS,
        "continuum_mode": "mfs",
        "primary_beam_limit": 0.0,
        "image_data_variables_keep": KEEP,
        "processing_set_data_group_name": "base",
        "single_precision_image": False,
        "n_mapping_parallelism": {"frequency": 1},
        "gridding_kernel_params": DFT,
        "processing_function_threads": 2,
        "overwrite": True,
        **kwargs,
    }
    with dask.config.set(scheduler="synchronous"):
        return distributed_applications.imaging.image_continuum_single_field(
            ps_store=ps_store,
            image_store=str(image_store),
            image_params=image_params,
            **arguments,
        )


def _planes(image_store, name):
    """``name`` as (polarization, l, m): the one channel or Taylor term zero."""
    data = load_image(str(image_store))[name]
    other = {dim: 0 for dim in data.dims if dim not in ("polarization", "l", "m")}
    return data.isel(other).transpose("polarization", "l", "m").values


def _relative_difference(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_mfs_with_the_dft_equals_the_cube_on_one_channel(tmp_path, processing_sets):
    """MFS with one Taylor term and natural weights images one channel with the DFT
    as the cube imager with the DFT does: the PSF, the residual and the model after
    two imaging cycles, all four Stokes planes, to within 1e-11 of each product's
    peak (measured on Linux x86-64: the PSF identical, the residual and the model
    within 7e-16 and 2e-16 of their peaks), while the default kernel moves them by
    5e-3 to 1e-2. The outputs are those of the gridder path."""
    store = processing_sets["four"]
    image_params = _image_params(store, ["I", "Q", "U", "V"])
    cube_store = tmp_path / "cube.img.zarr"
    with dask.config.set(scheduler="synchronous"):
        distributed_applications.imaging.image_cube_single_field(
            ps_store=store,
            image_store=str(cube_store),
            image_params={
                key: image_params[key]
                for key in (
                    "image_size",
                    "cell_size",
                    "phase_direction",
                    "frequency_coords",
                    "polarization_coords",
                    "time_coords",
                    "fft_padding",
                )
            },
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params={**CONTROLS, "primary_beam_limit": 0.0},
            image_data_variables_keep=KEEP,
            processing_set_data_group_name="base",
            single_precision_image=False,
            n_mapping_parallelism={"frequency": 1},
            gridding_kernel_params=DFT,
            processing_function_threads=2,
            overwrite=True,
        )
    continuum_store = tmp_path / "continuum.img.zarr"
    _continuum(store, continuum_store, image_params)
    gridder_store = tmp_path / "continuum_gridder.img.zarr"
    _continuum(store, gridder_store, image_params, gridding_kernel_params=None)

    for name in ("POINT_SPREAD_FUNCTION", "SKY_RESIDUAL", "SKY_MODEL"):
        cube = _planes(cube_store, name)
        continuum = _planes(continuum_store, name)
        assert cube.shape == continuum.shape == (4, *IMAGE_SIZE)
        assert _relative_difference(continuum, cube) <= 1e-11, name
    # the direct Fourier transform's PSF is 1 at the phase centre
    np.testing.assert_allclose(
        _planes(continuum_store, "POINT_SPREAD_FUNCTION")[:, 32, 32],
        1.0,
        rtol=0,
        atol=1e-15,
    )
    # the outputs of the gridder path: the same variables, dimensions, types and
    # data groups, less the role of the uv grid the DFT does not make (the
    # gridder path's store names its VISIBILITY grid, which it does not hold)
    dft, gridder = load_image(str(continuum_store)), load_image(str(gridder_store))
    assert sorted(dft.data_vars) == sorted(gridder.data_vars)
    for name in dft.data_vars:
        assert dft[name].dims == gridder[name].dims, name
        assert dft[name].shape == gridder[name].shape, name
        assert dft[name].dtype == gridder[name].dtype, name

    def roles(image, ignored=("date", "description")):
        return {
            name: {k: v for k, v in group.items() if k not in ignored}
            for name, group in image.attrs["data_groups"].items()
        }

    uv_grids = ("visibility", "uv_sampling")
    assert roles(dft) == roles(dft, ("date", "description", *uv_grids))
    assert roles(dft) == roles(gridder, ("date", "description", *uv_grids))
    assert (
        _relative_difference(
            _planes(gridder_store, "SKY_RESIDUAL"),
            _planes(continuum_store, "SKY_RESIDUAL"),
        )
        > 1e-6
    )


def test_the_reduce_adds_the_taylor_images_of_the_map_tasks(tmp_path, processing_sets):
    """Two map tasks give the products of one, to rounding: the Taylor residuals
    and PSFs of the tasks are added before their normalization, in every cycle,
    and every task predicts the model (MT-MFS, two Taylor terms; measured on Linux
    x86-64: within 8e-16, 1.3e-15 and 3.3e-14 of the peaks of the PSF, the
    residual and the model)."""
    store = processing_sets["two_hands"]
    image_params = _image_params(store, ["I", "Q"], n_taylor_terms=2)
    stores = {}
    for n_tasks in (1, 2):
        stores[n_tasks] = tmp_path / f"tasks{n_tasks}.img.zarr"
        _continuum(
            store,
            stores[n_tasks],
            image_params,
            deconvolver="mtmfs",
            n_mapping_parallelism={"frequency": n_tasks},
        )
    one, two = (load_image(str(stores[n])) for n in (1, 2))
    for name in ("POINT_SPREAD_FUNCTION", "SKY_RESIDUAL", "SKY_MODEL"):
        assert one[name].dims == two[name].dims, name
        assert one[name].shape == two[name].shape, name
        assert _relative_difference(two[name].values, one[name].values) <= 1e-12
    # the second Taylor term of the model is predicted, too
    assert np.abs(one["SKY_MODEL"].isel(taylor_term=1).values).max() > 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"continuum_mode": "mvc"}, "continuum_mode 'mvc' is not available"),
        (
            {"visibility_memory_mode": "in_memory"},
            "visibility_memory_mode 'in_memory' is not available",
        ),
        (
            {"visibility_memory_mode": "in_place"},
            "visibility_memory_mode 'in_place' is not available",
        ),
        ({"image_data_variables_keep": ["sky_residual", "visibility"]}, "uv grids"),
        ({"image_data_variables_keep": ["uv_sampling"]}, "uv grids"),
        (
            {"gridding_kernel_params": {"type": "dft", "support": 7}},
            "do not apply to the type 'dft'",
        ),
    ],
    ids=["mvc", "in_memory", "in_place", "visibility", "uv_sampling", "support"],
)
def test_refused_before_anything_is_written(tmp_path, overrides, match):
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": np.array([-CELL, CELL]),
        "phase_direction": PHASE_CENTER,
        "frequency_coords": [100e9],
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "n_taylor_terms": 1,
        "reference_frequency": 100e9,
    }
    with pytest.raises(ValueError, match=match):
        _continuum(
            str(tmp_path / "absent.ps.zarr"),
            tmp_path / "refused.img.zarr",
            image_params,
            **overrides,
        )
    assert list(tmp_path.iterdir()) == []
