"""Correlation selection of the image_continuum_single_field distributed application.

Small simulated processing sets (6 ALMA antennas, 4 times, 3 channels near
100 GHz) with one point source at the phase centre and no noise. The source
has the correlation fluxes ``[1.2, 0.1 + 0.05j, 0.1 - 0.05j, 0.8]`` (the
simulator picks the stored correlations by name), that is
``I, Q, U, V = 1, 0.2, 0.1, 0.05`` for linear feeds
(``XX, XY, YX, YY``) and ``I, Q, U, V = 1, 0.1, 0.05, 0.2`` for circular
feeds (``RR, RL, LR, LL``).
"""

import shutil

import dask
import numpy as np
import pytest
import zarr
from xradio.image import load_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
CORRELATION_FLUX = np.array([1.2, 0.1 + 0.05j, 0.1 - 0.05j, 0.8])
TRUTH = {
    "linear": {"I": 1.0, "Q": 0.2, "U": 0.1, "V": 0.05},
    "circular": {"I": 1.0, "Q": 0.1, "U": 0.05, "V": 0.2},
}
IMAGE_SIZE = [64, 64]
CENTRE = {"l": IMAGE_SIZE[0] // 2, "m": IMAGE_SIZE[1] // 2}
CORRELATIONS = {
    "two_hands": ["XX", "YY"],
    "four": ["XX", "XY", "YX", "YY"],
    "four_cross_hand_flagged": ["XX", "XY", "YX", "YY"],
    "reversed": ["YY", "XX"],
    "one": ["XX"],
    "circular_two_hands": ["RR", "LL"],
    "circular_four": ["RR", "RL", "LR", "LL"],
}


@pytest.fixture(scope="module")
def processing_sets(tmp_path_factory):
    """One pristine processing set per correlation layout of ``CORRELATIONS``.

    In ``four_cross_hand_flagged`` the ``XY`` correlation of the first time is
    flagged on every baseline and channel.
    """
    work_dir = tmp_path_factory.mktemp("continuum_polarization")
    antenna_xds = read_telescope_layout(
        "alma.cycle8.1", antenna_selection=list(range(6))
    )
    stores = {}
    for name, polarization in CORRELATIONS.items():
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
                    "n_channels": 3,
                    "channel_width": 2e6,
                },
                polarization=polarization,
                sky_components=[
                    {"kind": "point", "flux": CORRELATION_FLUX, "ra_dec": PHASE_CENTER}
                ],
                phase_center_ra_dec=PHASE_CENTER[None, :],
                beam_models=[airy_disk_model("alma")],
                beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
                n_time_chunks=1,
                n_frequency_chunks=1,
                overwrite=True,
            )

    stores["mtmfs_low_pb"] = str(work_dir / "mtmfs_low_pb.ps.zarr")
    off_axis = PHASE_CENTER + np.array(
        [np.deg2rad(45.0 / 3600.0) / np.cos(PHASE_CENTER[1]), 0.0]
    )
    with dask.config.set(scheduler="synchronous"):
        distributed_applications.simulation.simulate_processing_set(
            ps_store=stores["mtmfs_low_pb"],
            antenna_xds=antenna_xds,
            time_params={
                "time_start": "2019-10-03T19:00:00.000",
                "time_delta": 600.0,
                "n_samples": 4,
            },
            frequency_params={
                "freq_start": 100e9,
                "freq_delta": 0.5e9,
                "n_channels": 3,
                "channel_width": 2e6,
            },
            polarization=CORRELATIONS["two_hands"],
            sky_components=[
                {"kind": "point", "flux": CORRELATION_FLUX, "ra_dec": PHASE_CENTER},
                {"kind": "point", "flux": CORRELATION_FLUX, "ra_dec": off_axis},
            ],
            phase_center_ra_dec=PHASE_CENTER[None, :],
            beam_models=[airy_disk_model("alma")],
            beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
            n_time_chunks=1,
            n_frequency_chunks=1,
            overwrite=True,
        )
    root = zarr.open_group(
        stores["four_cross_hand_flagged"], mode="r+", use_consolidated=False
    )
    (ms_name,) = list(root.group_keys())
    flag = root[ms_name]["FLAG"]
    flag_values = np.asarray(flag[:])
    flag_values[0, :, :, 1] = True
    flag[:] = flag_values
    zarr.consolidate_metadata(stores["four_cross_hand_flagged"], path=ms_name)
    zarr.consolidate_metadata(stores["four_cross_hand_flagged"])

    # Derive the parallel-hand flag cases from the same simulated visibilities.
    # A sample with either loaded hand flagged must be omitted from both hands.
    for name, correlation_selection in (
        ("one_parallel_hand_flagged", [0]),
        ("both_parallel_hands_flagged", [0, 1]),
    ):
        stores[name] = str(work_dir / f"{name}.ps.zarr")
        shutil.copytree(stores["two_hands"], stores[name])
        root = zarr.open_group(stores[name], mode="r+", use_consolidated=False)
        (ms_name,) = list(root.group_keys())
        flag = root[ms_name]["FLAG"]
        flag_values = np.asarray(flag[:])
        flag_values[0, :, :, correlation_selection] = True
        flag[:] = flag_values
        zarr.consolidate_metadata(stores[name], path=ms_name)
        zarr.consolidate_metadata(stores[name])
    return stores


def _arrays_and_groups(ps_store):
    """Data variables and data groups of every measurement set of a store."""
    root = zarr.open_group(ps_store, mode="r", use_consolidated=False)
    return {
        name: (sorted(root[name].array_keys()), root[name].attrs.get("data_groups"))
        for name in root.group_keys()
    }


def _image(
    tmp_path,
    pristine_store,
    polarization_coords,
    *,
    basis="linear",
    continuum_mode="mfs",
    max_iter=0,
    weighting=None,
    image_param_overrides=None,
    cycle_control_overrides=None,
    image_data_variables_keep=None,
    return_result=False,
    **kwargs,
):
    """Image a fresh copy of ``pristine_store`` and return its image store path.

    The default in-place weight cache writes into the input processing set, so
    every run gets its own copy; afterwards the copy must hold exactly what it
    held before.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    ps_store = str(tmp_path / "input.ps.zarr")
    shutil.rmtree(ps_store, ignore_errors=True)
    shutil.copytree(pristine_store, ps_store)
    before = _arrays_and_groups(ps_store)
    frequency = np.asarray(open_processing_set(ps_store).xr_ps.get_freq_axis().values)
    image_store = str(
        tmp_path / f"{continuum_mode}_{'_'.join(polarization_coords)}.img.zarr"
    )
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": np.array([-0.7, 0.7]) * np.pi / (180 * 3600),
        "phase_direction": PHASE_CENTER,
        "frequency_coords": frequency,
        "polarization_coords": polarization_coords,
        "time_coords": [0],
        "fft_padding": 1.2,
        "n_taylor_terms": 1,
        "reference_frequency": float(np.mean(frequency)),
    }
    image_params.update(image_param_overrides or {})
    cycle_control_params = {
        "max_iter": max_iter,
        "max_cycles": 5 if max_iter else 0,
        "threshold": 0.0,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "max_iter_per_cycle": max(max_iter, 1),
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    cycle_control_params.update(cycle_control_overrides or {})
    if image_data_variables_keep is None:
        image_data_variables_keep = [
            "sky_residual",
            "sky_model",
            "point_spread_function",
        ]
    with dask.config.set(scheduler="synchronous"):
        result = distributed_applications.imaging.image_continuum_single_field(
            ps_store=ps_store,
            image_store=image_store,
            image_params=image_params,
            imaging_weights_params=weighting
            or {"weighting": "natural", "weighting_scope": "local"},
            cycle_control_params=cycle_control_params,
            continuum_mode=continuum_mode,
            instrument_polarization_basis=basis,
            image_data_variables_keep=image_data_variables_keep,
            processing_set_data_group_name="base",
            n_mapping_parallelism={"frequency": 1},
            overwrite=True,
            **kwargs,
        )
    assert _arrays_and_groups(ps_store) == before
    return (result, image_store) if return_result else image_store


def _centre(img, variable):
    """Value of every Stokes plane of ``variable`` at the phase centre."""
    plane = img[variable].isel(time=0, **CENTRE)
    plane = plane.isel({dim: 0 for dim in plane.dims if dim != "polarization"})
    return {str(p): float(plane.sel(polarization=p)) for p in img.polarization.values}


@pytest.mark.parametrize(
    ("layout", "basis", "polarization_coords", "continuum_mode"),
    [
        ("four", "linear", ["I", "Q"], "mfs"),
        ("four", "linear", ["I", "Q"], "mvc"),
        ("four", "linear", ["I", "Q", "U", "V"], "mfs"),
        ("reversed", "linear", ["I", "Q"], "mfs"),
        ("reversed", "linear", ["I", "Q"], "mvc"),
        ("circular_two_hands", "circular", ["I", "V"], "mfs"),
        ("circular_four", "circular", ["I", "V"], "mfs"),
        ("circular_four", "circular", ["I", "Q", "U", "V"], "mfs"),
    ],
)
def test_requested_stokes_planes_are_imaged_from_their_correlations(
    tmp_path, processing_sets, layout, basis, polarization_coords, continuum_mode
):
    """The dirty image of the source holds the requested Stokes planes, with the
    right labels and signs (a reversed ``YY, XX`` set gives ``Q > 0``), whatever
    other correlations the processing set holds and in whatever order."""
    img = load_image(
        _image(
            tmp_path,
            processing_sets[layout],
            polarization_coords,
            basis=basis,
            continuum_mode=continuum_mode,
        )
    )
    assert [str(p) for p in img.polarization.values] == polarization_coords
    centre = _centre(img, "SKY_RESIDUAL")
    expected = {stokes: TRUTH[basis][stokes] for stokes in polarization_coords}
    assert centre == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize(
    ("layout", "basis"), [("four", "linear"), ("circular_four", "circular")]
)
@pytest.mark.parametrize(
    "weighting",
    [
        {"weighting": "natural", "weighting_scope": "local"},
        {"weighting": "briggs", "robust": 0.5, "weighting_scope": "global"},
    ],
)
def test_four_correlations_are_cleaned_in_every_stokes_plane(
    tmp_path, processing_sets, layout, basis, weighting
):
    """Cleaning ``I, Q, U, V`` predicts and subtracts all four correlations: at
    the source the model flux plus the residual is the true flux of every
    plane (global Briggs weighting runs the separate weight graphs)."""
    polarization_coords = ["I", "Q", "U", "V"]
    img = load_image(
        _image(
            tmp_path,
            processing_sets[layout],
            polarization_coords,
            basis=basis,
            max_iter=50,
            weighting=weighting,
        )
    )
    residual = _centre(img, "SKY_RESIDUAL")
    model_flux = img.SKY_MODEL.isel(time=0).sum(dim=["l", "m"])
    model_flux = model_flux.isel(
        {dim: 0 for dim in model_flux.dims if dim != "polarization"}
    )
    for stokes in polarization_coords:
        total = float(model_flux.sel(polarization=stokes)) + residual[stokes]
        assert total == pytest.approx(TRUTH[basis][stokes], rel=1e-5, abs=1e-6)
        assert float(model_flux.sel(polarization=stokes)) > 0.5 * TRUTH[basis][stokes]


@pytest.fixture(scope="module")
def two_hand_images(tmp_path_factory, processing_sets):
    """``I, Q`` images of the two-hand ``XX, YY`` processing set, cleaned."""
    images = {}
    for continuum_mode in ("mfs", "mvc"):
        tmp_path = tmp_path_factory.mktemp(f"two_hands_{continuum_mode}")
        images[continuum_mode] = load_image(
            _image(
                tmp_path,
                processing_sets["two_hands"],
                ["I", "Q"],
                continuum_mode=continuum_mode,
                max_iter=30,
            )
        )
    return images


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
@pytest.mark.parametrize(
    ("layout", "kwargs"),
    [
        ("four", {}),
        ("four_cross_hand_flagged", {}),
        ("reversed", {}),
        ("reversed", {"skunk_works": True, "weight_memory_mode": "in_memory"}),
    ],
)
def test_parallel_hands_are_loaded_alone(
    tmp_path, processing_sets, two_hand_images, continuum_mode, layout, kwargs
):
    """``I, Q`` from four correlations, from four with a flagged cross hand, or
    from ``YY, XX`` loads only ``XX, YY`` in this order: the images equal those
    of the ``XX, YY`` set bit for bit. A flagged cross hand drops no
    parallel-hand sample."""
    img = load_image(
        _image(
            tmp_path,
            processing_sets[layout],
            ["I", "Q"],
            continuum_mode=continuum_mode,
            max_iter=30,
            **kwargs,
        )
    )
    reference = two_hand_images[continuum_mode]
    for variable in ("SKY_RESIDUAL", "SKY_MODEL", "POINT_SPREAD_FUNCTION"):
        assert np.isfinite(img[variable].values).all(), variable
        np.testing.assert_array_equal(
            img[variable].values, reference[variable].values, err_msg=variable
        )


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_single_task_visibility_cache_modes_match_driver_default(
    tmp_path, processing_sets, continuum_mode
):
    """One-leaf in-memory and on-disk caches reproduce default recomputation."""
    images = {}
    stores = {}
    for mode in ("recompute", "in_memory", "in_place"):
        options = {} if mode == "recompute" else {"visibility_memory_mode": mode}
        stores[mode] = _image(
            tmp_path / mode,
            processing_sets["two_hands"],
            ["I", "Q"],
            continuum_mode=continuum_mode,
            max_iter=2,
            **options,
        )
        images[mode] = load_image(stores[mode])

    for mode in ("in_memory", "in_place"):
        for variable in ("SKY_RESIDUAL", "SKY_MODEL", "POINT_SPREAD_FUNCTION"):
            np.testing.assert_allclose(
                images[mode][variable],
                images["recompute"][variable],
                rtol=1e-5,
                atol=5e-6,
            )
        cache_name = (
            "_MFS_VISIBILITY_GRID_CACHE"
            if continuum_mode == "mfs"
            else "_MVC_VISIBILITY_GRID_CACHE"
        )
        assert cache_name not in zarr.open_group(stores[mode])


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_one_flagged_parallel_hand_drops_the_whole_sample(
    tmp_path, processing_sets, continuum_mode
):
    """Flagging one loaded hand is equivalent to flagging both loaded hands."""
    images = {}
    for layout in ("one_parallel_hand_flagged", "both_parallel_hands_flagged"):
        images[layout] = load_image(
            _image(
                tmp_path / layout,
                processing_sets[layout],
                ["I", "Q"],
                continuum_mode=continuum_mode,
            )
        )

    for variable in ("SKY_RESIDUAL", "POINT_SPREAD_FUNCTION"):
        np.testing.assert_array_equal(
            images["one_parallel_hand_flagged"][variable],
            images["both_parallel_hands_flagged"][variable],
        )


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_mtmfs_ignores_masked_structure_below_the_primary_beam_limit(
    tmp_path, processing_sets, continuum_mode
):
    """MT-MFS cleans valid support and converges with masked low-PB structure."""
    threshold = 0.053
    clean_mask = tmp_path / "clean_mask.npy"
    np.save(clean_mask, np.ones(IMAGE_SIZE, dtype=bool))
    result, image_store = _image(
        tmp_path,
        processing_sets["mtmfs_low_pb"],
        ["I", "Q"],
        continuum_mode=continuum_mode,
        max_iter=100,
        image_param_overrides={
            "cell_size": np.array([-2.0, 2.0]) * np.pi / (180 * 3600),
            "n_taylor_terms": 2,
        },
        cycle_control_overrides={
            "max_cycles": 15,
            "threshold": threshold,
            "max_iter_per_cycle": 20,
            "scales": [0.0],
        },
        image_data_variables_keep=[
            "sky_residual",
            "sky_model",
            "point_spread_function",
            "primary_beam",
        ],
        return_result=True,
        deconvolver="mtmfs",
        primary_beam_limit=0.2,
        clean_mask=str(clean_mask),
    )
    image = load_image(image_store)
    primary_beam = image.PRIMARY_BEAM.isel(time=0, frequency=0, polarization=0)
    low_beam = (primary_beam > 0.05) & (primary_beam < 0.2)
    residual = np.abs(image.SKY_RESIDUAL.isel(time=0, taylor_term=0, polarization=0))
    model = np.abs(image.SKY_MODEL.isel(time=0, taylor_term=0, polarization=0))

    assert result["n_cycles"] < 15
    assert result["controller"].total_iter_done > 0
    assert float(model.max()) > 0.0
    if continuum_mode == "mfs":
        assert float(residual.where(low_beam).max()) > threshold
    assert float(model.where(low_beam, 0).max()) == 0.0


@pytest.mark.parametrize(
    ("layout", "basis", "polarization_coords", "continuum_mode", "message"),
    [
        ("one", "linear", ["I", "Q"], "mfs", r"holds the correlations \['XX'\]"),
        (
            "circular_two_hands",
            "linear",
            ["I", "Q"],
            "mfs",
            r"holds the correlations \['RR', 'LL'\]",
        ),
        ("two_hands", "linear", ["I"], "mfs", "polarization_coords"),
        ("two_hands", "linear", ["I", "V"], "mfs", "polarization_coords"),
        ("two_hands", "linear", ["Q", "I"], "mfs", "polarization_coords"),
        ("four", "linear", ["I", "Q", "U", "V"], "mvc", "continuum_mode='mvc'"),
    ],
)
def test_unsupported_requests_are_refused_before_anything_is_written(
    tmp_path,
    processing_sets,
    layout,
    basis,
    polarization_coords,
    continuum_mode,
    message,
):
    """A request the processing set cannot give, a Stokes set outside the two
    supported ones, or ``I, Q, U, V`` with MVC raises ``ValueError`` before the
    image store or the weight cache in the input processing set is written."""
    with pytest.raises(ValueError, match=message):
        _image(
            tmp_path,
            processing_sets[layout],
            polarization_coords,
            basis=basis,
            continuum_mode=continuum_mode,
        )
    assert [path.name for path in tmp_path.iterdir()] == ["input.ps.zarr"]
    input_store = str(tmp_path / "input.ps.zarr")
    assert _arrays_and_groups(input_store) == _arrays_and_groups(
        processing_sets[layout]
    )
