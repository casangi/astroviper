"""The gridding kernel of the image_continuum_single_field distributed application.

``gridding_kernel_params`` selects the kernel of every continuum stage that
grids, degrids or corrects an image, as it does for cube imaging. Small
simulated processing sets (6 ALMA antennas, 4 times, no noise) hold two point
sources, one at the phase centre and one off centre, where the image plane
correction of the kernel differs from 1.
"""

import inspect
import re

import dask
import numpy as np
import pytest
from xradio.image import load_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.processing_functions.imaging._param_docs import IMAGING_PARAM_DOCS
from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    CASA_GRIDDING_KERNEL_PARAMS,
    DEFAULT_GRIDDING_KERNEL_PARAMS,
    GriddingKernel,
    make_gridding_kernel,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
CELL = 0.7 * np.pi / (180 * 3600)
IMAGE_SIZE = [64, 64]
FFT_PADDING = 1.2
# linear feeds: I, Q, U, V = 1, 0.2, 0.1, 0.05 and 0.5, -0.05, 0.1, -0.02
SOURCES = [
    (np.array([1.2, 0.1 + 0.05j, 0.1 - 0.05j, 0.8]), (0.0, 0.0)),
    (np.array([0.45, 0.1 - 0.02j, 0.1 + 0.02j, 0.55]), (12.0, -8.0)),
]
KERNEL = {
    "type": "prolate_spheroidal",
    "support": 9,
    "oversampling": 300,
    "interpolation": "linear",
}
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
    work_dir = tmp_path_factory.mktemp("continuum_gridding_kernel")
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


def _image_params(ps_store, polarization_coords, n_taylor_terms=1, **overrides):
    frequency = np.asarray(open_processing_set(ps_store).xr_ps.get_freq_axis().values)
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": np.array([-CELL, CELL]),
        "phase_direction": PHASE_CENTER,
        "frequency_coords": frequency,
        "polarization_coords": polarization_coords,
        "time_coords": [0],
        "fft_padding": FFT_PADDING,
        "n_taylor_terms": n_taylor_terms,
        "reference_frequency": float(np.mean(frequency)),
    }
    image_params.update(overrides)
    return image_params


def _continuum(
    ps_store,
    image_store,
    image_params,
    gridding_kernel_params,
    continuum_mode="mfs",
):
    with dask.config.set(scheduler="synchronous"):
        return distributed_applications.imaging.image_continuum_single_field(
            ps_store=ps_store,
            image_store=str(image_store),
            image_params=image_params,
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params=CONTROLS,
            continuum_mode=continuum_mode,
            primary_beam_limit=0.0,
            image_data_variables_keep=KEEP,
            processing_set_data_group_name="base",
            single_precision_image=False,
            n_mapping_parallelism={"frequency": 1},
            gridding_kernel_params=gridding_kernel_params,
            overwrite=True,
        )


def _planes(image_store, name):
    """``name`` as (polarization, l, m): the one channel or Taylor term zero."""
    data = load_image(str(image_store))[name]
    other = {dim: 0 for dim in data.dims if dim not in ("polarization", "l", "m")}
    return data.isel(other).transpose("polarization", "l", "m").values


def _relative_difference(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_kernel_parameter_matches_the_cube_api():
    """The same default and the same shared description as cube imaging."""
    continuum = distributed_applications.imaging.image_continuum_single_field
    cube = distributed_applications.imaging.image_cube_single_field
    for function in (continuum, cube):
        parameter = inspect.signature(function).parameters["gridding_kernel_params"]
        assert parameter.default is None
        assert " ".join(IMAGING_PARAM_DOCS["gridding_kernel_params"].split()) in (
            " ".join(inspect.getdoc(function).split())
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"oversamplng": 10},
        {"type": "boxcar"},
        {"support": 6},
        {"support": 1, "type": "exponential_semicircle"},
        {"type": "prolate_spheroidal_casa", "support": 9},
        {"oversampling": 0},
        {"oversampling": 2.5},
        {"interpolation": "cubic"},
    ],
)
def test_kernel_parameters_are_refused_like_the_cube_before_anything_is_written(
    tmp_path, bad
):
    """The check cube imaging makes when it makes its kernel refuses the same
    parameters here, with the same message, before the input is opened."""
    with pytest.raises(ValueError) as cube_refusal:
        make_gridding_kernel(bad, FFT_PADDING)
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": np.array([-CELL, CELL]),
        "phase_direction": PHASE_CENTER,
        "frequency_coords": [100e9],
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": FFT_PADDING,
        "n_taylor_terms": 1,
        "reference_frequency": 100e9,
    }
    with pytest.raises(ValueError, match=re.escape(str(cube_refusal.value))):
        _continuum(
            str(tmp_path / "absent.ps.zarr"),
            tmp_path / "refused.img.zarr",
            image_params,
            bad,
        )
    assert list(tmp_path.iterdir()) == []


def test_a_kernel_parameter_that_is_not_a_dict_is_refused_like_the_cube(tmp_path):
    """The schema of both applications refuses it before anything runs."""
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": np.array([-CELL, CELL]),
        "phase_direction": PHASE_CENTER,
        "frequency_coords": [100e9],
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": FFT_PADDING,
        "n_taylor_terms": 1,
        "reference_frequency": 100e9,
    }
    with pytest.raises(AssertionError):
        _continuum(
            str(tmp_path / "absent.ps.zarr"),
            tmp_path / "refused.img.zarr",
            image_params,
            "prolate_spheroidal",
        )
    with pytest.raises(AssertionError):
        distributed_applications.imaging.image_cube_single_field(
            ps_store=str(tmp_path / "absent.ps.zarr"),
            image_store=str(tmp_path / "refused_cube.img.zarr"),
            image_params=image_params,
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params=CONTROLS,
            gridding_kernel_params="prolate_spheroidal",
        )
    assert list(tmp_path.iterdir()) == []


def test_a_kernel_the_padding_cannot_hold_is_refused_like_the_cube(
    tmp_path, processing_sets
):
    """Support 17 at a padding of 1.05: refused where the kernel is made."""
    kernel = {"type": "prolate_spheroidal", "support": 17}
    with pytest.raises(ValueError) as cube_refusal:
        make_gridding_kernel(kernel, 1.05)
    store = processing_sets["two_hands"]
    with pytest.raises(ValueError, match=re.escape(str(cube_refusal.value))):
        _continuum(
            store,
            tmp_path / "refused.img.zarr",
            _image_params(store, ["I", "Q"], fft_padding=1.05),
            kernel,
        )


def test_mfs_with_a_designed_kernel_equals_the_cube_on_one_channel(
    tmp_path, processing_sets
):
    """MFS with one Taylor term and natural weights images one channel as the
    cube imager does, with the kernel of gridding_kernel_params in the PSF, the
    gridding, the degridding of the model and the image correction (two imaging
    cycles, all four Stokes planes); the default kernel gives other images.

    "As the cube" is to rounding: within 1e-11 of the peak of each product (on
    Linux x86-64 the PSF is identical, the model differs by 2e-16 and the
    residual by 2e-13). The CASA kernel in any one of the four stages moves a
    product by more than 1e-2, and the default kernel (least misfit
    9/300/linear) moves the residual by 2e-5.
    """
    store = processing_sets["four"]
    stokes = ["I", "Q", "U", "V"]
    image_params = _image_params(store, stokes)
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
            gridding_kernel_params=KERNEL,
            overwrite=True,
        )
    continuum_store = tmp_path / "continuum.img.zarr"
    _continuum(store, continuum_store, image_params, KERNEL)
    default_store = tmp_path / "continuum_default.img.zarr"
    _continuum(store, default_store, image_params, None)

    for cube_name, continuum_name in (
        ("POINT_SPREAD_FUNCTION", "POINT_SPREAD_FUNCTION"),
        ("SKY_RESIDUAL", "SKY_RESIDUAL"),
        ("SKY_MODEL", "SKY_MODEL"),
    ):
        cube = _planes(cube_store, cube_name)
        continuum = _planes(continuum_store, continuum_name)
        assert cube.shape == (4, *IMAGE_SIZE)
        assert _relative_difference(continuum, cube) <= 1e-11, cube_name
    residual = _planes(continuum_store, "SKY_RESIDUAL")
    default_residual = _planes(default_store, "SKY_RESIDUAL")
    assert _relative_difference(default_residual, residual) > 1e-6


def _spy_kernels(monkeypatch):
    """Record the kernel of every gridder, degridder and image correction call."""
    from astroviper.processing_functions.imaging.gridders import anti_alias_grid_cpp

    calls = {"grid": [], "uv_sampling": [], "degrid": [], "correction": []}
    table_position = {"grid": 9, "uv_sampling": 8, "degrid": 7}
    for kind, name in (
        ("grid", "anti_alias_grid"),
        ("uv_sampling", "anti_alias_grid_uv_sampling"),
        ("degrid", "anti_alias_degrid"),
    ):
        function = getattr(anti_alias_grid_cpp, name)

        def spy(*args, _kind=kind, _function=function, **kwargs):
            calls[_kind].append(
                (
                    np.array(args[table_position[_kind]]),
                    kwargs["support"],
                    kwargs["oversampling"],
                    kwargs.get("interpolation", 0),
                )
            )
            return _function(*args, **kwargs)

        monkeypatch.setattr(anti_alias_grid_cpp, name, spy)

    correction_1D = GriddingKernel.correction_1D

    def correction_spy(self, n_pixels):
        calls["correction"].append(self.params)
        return correction_1D(self, n_pixels)

    monkeypatch.setattr(GriddingKernel, "correction_1D", correction_spy)
    return calls


@pytest.mark.parametrize(
    ("continuum_mode", "store", "stokes", "n_taylor_terms"),
    [
        ("mfs", "four", ["I", "Q", "U", "V"], 1),
        ("mfs", "two_hands", ["I", "Q"], 2),
        ("mvc", "two_hands", ["I", "Q"], 2),
    ],
)
@pytest.mark.parametrize(
    "gridding_kernel_params",
    [KERNEL, None, {"type": "prolate_spheroidal_casa"}],
    ids=["prolate_spheroidal_9_300_linear", "default", "casa_type_alone"],
)
def test_every_continuum_stage_uses_the_requested_kernel(
    monkeypatch,
    tmp_path,
    processing_sets,
    continuum_mode,
    store,
    stokes,
    n_taylor_terms,
    gridding_kernel_params,
):
    """The PSF, the gridding, the degridding of the model and every image
    correction (the residual and PSF transforms and the model transform before
    degridding) use the kernel of gridding_kernel_params, in MFS and MVC; the
    type prolate_spheroidal_casa alone is CASA's kernel and sampling."""
    calls = _spy_kernels(monkeypatch)
    ps_store = processing_sets[store]
    _continuum(
        ps_store,
        tmp_path / "spy.img.zarr",
        _image_params(ps_store, stokes, n_taylor_terms),
        gridding_kernel_params,
        continuum_mode,
    )
    expected = make_gridding_kernel(gridding_kernel_params, FFT_PADDING)
    assert {kind: len(made) > 0 for kind, made in calls.items()} == {
        "grid": True,
        "uv_sampling": True,
        "degrid": True,
        "correction": True,
    }
    for kind in ("grid", "uv_sampling", "degrid"):
        for table, support, oversampling, interpolation in calls[kind]:
            np.testing.assert_array_equal(table, expected.table)
            assert (support, oversampling, interpolation) == (
                expected.support,
                expected.oversampling,
                expected.interpolation_code,
            ), kind
    assert all(params == expected.params for params in calls["correction"])
    if gridding_kernel_params is None:
        assert expected.params == DEFAULT_GRIDDING_KERNEL_PARAMS
    elif gridding_kernel_params == {"type": "prolate_spheroidal_casa"}:
        assert expected.params == CASA_GRIDDING_KERNEL_PARAMS
