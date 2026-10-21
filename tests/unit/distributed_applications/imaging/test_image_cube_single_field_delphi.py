"""Tests of image_cube_single_field_delphi and of the parameters file of image_cube_single_field.

* ``parameters_file`` of image_cube_single_field: the arguments passed win
  over the file, a call with the file alone, and the arrays of
  ``image_params`` after a JSON round trip.
* image_cube_single_field_delphi on a small simulated processing set with the
  PSF pass replaced by a stand-in of its contract: the parameters file, and
  the cube imaged from it.
* The metadata delphi reads, its input checks, the warning of a cell too
  large for the uv coverage and the MPI options of its two computes.
"""

import functools
import importlib
import inspect
import json
import sys
import types

import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord
from xradio.image import load_image

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging.image_cube_single_field import (
    _apply_parameters_file,
    image_cube_single_field,
    read_parameters_file,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    IMAGING_THRESHOLD,
)
from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
    NOISY_N_SIGMA,
    noisy_cycle_control_params,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
N_ANTENNA = 8
PSF_MODULE = "astroviper.distributed_applications.imaging.point_spread_function_cube_single_field"
# The module itself (the package may re-export the function under its name).
delphi_module = importlib.import_module(
    "astroviper.distributed_applications.imaging.image_cube_single_field_delphi"
)
PROLATE_SPHEROIDAL_9 = {
    "type": "prolate_spheroidal",
    "support": 9,
    "oversampling": 300,
    "interpolation": "linear",
}


@pytest.fixture(scope="module")
def ps_store(tmp_path_factory):
    """1 Jy unpolarised point source at the phase centre, 8 VLA antennas, 2 channels."""
    result = distributed_applications.simulation.simulate_processing_set(
        ps_store=str(tmp_path_factory.mktemp("delphi") / "point.ps.zarr"),
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
    )
    return result["ps_store"]


def _write_parameters_file(path, arguments):
    with open(path, "w") as file:
        json.dump({"image_cube_single_field": arguments}, file)
    return str(path)


# --------------------------------------------------------------------------- #
# parameters_file of image_cube_single_field
# --------------------------------------------------------------------------- #
def _target(ps_store, image_store, image_params, gridder="a", parameters_file=None):
    """Stand-in with the shape of the imager's signature."""
    return {
        "ps_store": ps_store,
        "image_store": image_store,
        "image_params": image_params,
        "gridder": gridder,
        "parameters_file": parameters_file,
    }


def test_parameters_file_fills_every_argument(tmp_path):
    """A call with the file alone takes every argument from it; the arrays of
    image_params come back as float arrays."""
    path = _write_parameters_file(
        tmp_path / "p.json",
        {
            "ps_store": "file.ps.zarr",
            "image_store": "file.img.zarr",
            "image_params": {
                "cell_size": [-1e-6, 1e-6],
                "phase_direction": [0.1, 0.2],
                "frequency_coords": [1e9, 2e9],
                "image_size": [8, 8],
                "polarization_coords": ["I", "V"],
            },
            "gridder": "b",
            "parameters_file": "ignored.json",
        },
    )
    called = _apply_parameters_file(_target)(parameters_file=path)
    assert called["ps_store"] == "file.ps.zarr"
    assert called["gridder"] == "b"
    assert called["parameters_file"] == path
    image_params = called["image_params"]
    for key in ("cell_size", "phase_direction", "frequency_coords"):
        assert isinstance(image_params[key], np.ndarray)
        assert image_params[key].dtype == float
    np.testing.assert_array_equal(image_params["cell_size"], [-1e-6, 1e-6])
    assert image_params["image_size"] == [8, 8]
    assert image_params["polarization_coords"] == ["I", "V"]


def test_parameters_file_explicit_arguments_win(tmp_path):
    """Arguments passed by position or keyword replace the file's values whole."""
    path = _write_parameters_file(
        tmp_path / "p.json",
        {
            "ps_store": "file.ps.zarr",
            "image_store": "file.img.zarr",
            "image_params": {"image_size": [8, 8]},
            "gridder": "b",
        },
    )
    called = _apply_parameters_file(_target)(
        "call.ps.zarr", image_params={"image_size": [4, 4]}, parameters_file=path
    )
    assert called["ps_store"] == "call.ps.zarr"
    assert called["image_store"] == "file.img.zarr"
    assert called["image_params"] == {"image_size": [4, 4]}
    assert called["gridder"] == "b"
    # An explicit default value also wins.
    assert (
        _apply_parameters_file(_target)(gridder="a", parameters_file=path)["gridder"]
        == "a"
    )


def test_parameters_file_absent_forwards_the_call_unchanged():
    called = _apply_parameters_file(_target)("p", "i", {"image_size": [2, 2]})
    assert called["gridder"] == "a" and called["parameters_file"] is None
    with pytest.raises(TypeError):
        _apply_parameters_file(_target)("p")


def test_image_cube_single_field_signature_keeps_required_arguments():
    """The signature seen by callers and tools is the imager's, plus parameters_file."""
    parameters = inspect.signature(image_cube_single_field).parameters
    assert parameters["parameters_file"].default is None
    assert parameters["ps_store"].default is inspect.Parameter.empty
    assert list(parameters)[-1] == "parameters_file"


def test_read_parameters_file_json_round_trip(tmp_path):
    """The JSON a delphi run writes reads back to arrays equal to the originals."""
    image_params = {
        "image_size": [16, 16],
        "cell_size": np.array([-2.5e-6, 2.5e-6]),
        "phase_direction": PC,
        "frequency_coords": np.array([3e9, 3.4e9]),
        "polarization_coords": ["I", "V"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }
    arguments = delphi_module._json_compatible(
        {
            "ps_store": "x",
            "image_params": image_params,
            "cycle_control_params": {"threshold": np.array([[1.0, 2.0], [3.0, 4.0]])},
            "n": np.int64(3),
        }
    )
    path = _write_parameters_file(tmp_path / "p.json", arguments)
    read = read_parameters_file(path)
    for key in ("cell_size", "phase_direction", "frequency_coords"):
        np.testing.assert_array_equal(read["image_params"][key], image_params[key])
    assert read["cycle_control_params"]["threshold"] == [[1.0, 2.0], [3.0, 4.0]]
    assert read["n"] == 3 and isinstance(read["n"], int)


def _imaging_arguments(ps_store, image_store, frequency_coords, threshold):
    return {
        "ps_store": ps_store,
        "image_store": image_store,
        "image_params": {
            "image_size": [64, 64],
            "cell_size": [-8.0 * np.pi / (180 * 3600), 8.0 * np.pi / (180 * 3600)],
            "phase_direction": PC.tolist(),
            "frequency_coords": list(frequency_coords),
            "polarization_coords": ["I", "V"],
            "time_coords": [0],
            "fft_padding": 1.2,
        },
        "instrument_polarization_basis": "circular",
        "imaging_weights_params": {"weighting": "natural"},
        "cycle_control_params": {
            "max_iter": 50,
            "max_cycles": 2,
            "threshold": threshold,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        "image_data_variables_keep": ["sky_residual", "sky_model"],
        "processing_set_data_group_name": "base",
        "single_precision_image": False,
        "n_mapping_parallelism": {"frequency": 2},
        "overwrite": True,
    }


def test_image_cube_single_field_from_parameters_file_per_channel_threshold(
    ps_store, tmp_path
):
    """Run from the file alone: channel 0 (threshold above the 1 Jy peak) is
    not cleaned, channel 1 (in the second node task) is."""
    from xradio.measurement_set import open_processing_set

    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    image_store = str(tmp_path / "cube.img.zarr")
    path = _write_parameters_file(
        tmp_path / "p.json",
        _imaging_arguments(
            ps_store, image_store, frequency_coords, [[10.0, 10.0], [0.1, 0.1]]
        ),
    )
    image_cube_single_field(parameters_file=path)
    model = load_image(image_store).SKY_MODEL.values[0, :, 0]  # [frequency, l, m]
    assert np.nansum(np.abs(model[0])) == 0.0
    assert np.nansum(model[1]) > 0.5


def test_image_cube_single_field_per_channel_threshold_shape(ps_store, tmp_path):
    from xradio.measurement_set import open_processing_set

    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    path = _write_parameters_file(
        tmp_path / "p.json",
        _imaging_arguments(
            ps_store, str(tmp_path / "c.img.zarr"), frequency_coords, [[1.0, 1.0]]
        ),
    )
    with pytest.raises(ValueError, match="per channel"):
        image_cube_single_field(parameters_file=path)


# --------------------------------------------------------------------------- #
# image_cube_single_field_delphi with a stand-in PSF pass
# --------------------------------------------------------------------------- #
#: Fitted minor axis of the stand-in PSF, in units of the cell of the PSF pass.
MINOR_AXIS_IN_CELLS = 4.0


@pytest.fixture
def psf_calls(monkeypatch):
    """Replace the PSF pass by a stand-in of its contract; records the calls."""
    calls = []

    def point_spread_function_cube_single_field(
        ps_store, image_params, imaging_weights_params, **kwargs
    ):
        calls.append(
            {
                "ps_store": ps_store,
                "image_params": image_params,
                "imaging_weights_params": imaging_weights_params,
                **kwargs,
            }
        )
        frequency = np.asarray(image_params["frequency_coords"])
        polarization = list(image_params["polarization_coords"])
        shape = (frequency.size, len(polarization))
        cell = abs(float(image_params["cell_size"][0]))
        beam = np.empty(shape + (3,))
        # The beam shrinks with frequency; the highest channel sets the cell.
        scale = frequency.max() / frequency
        beam[..., 0] = 1.2 * MINOR_AXIS_IN_CELLS * cell * scale[:, None]
        beam[..., 1] = MINOR_AXIS_IN_CELLS * cell * scale[:, None]
        beam[..., 2] = 0.3
        sensitivity = np.array([[1e-3, 2e-3], [3e-3, 4e-3]])[: shape[0], : shape[1]]
        dims = ("frequency", "polarization")
        return xr.Dataset(
            {
                "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION": (
                    dims + ("beam_params_label",),
                    beam,
                ),
                "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (dims, np.full(shape, 0.2)),
                "SENSITIVITY": (dims, sensitivity),
                "SENSITIVITY_NATURAL": (dims, 0.8 * sensitivity),
                "SUM_OF_WEIGHTS": (dims, np.full(shape, 100.0)),
                "EFFECTIVE_NUMBER_OF_VISIBILITIES": (dims, np.full(shape, 50.0)),
                "UV_DISTANCE_MAX": ("frequency", np.full(shape[0], 1e4)),
                "UV_DISTANCE_MIN": ("frequency", np.full(shape[0], 1e2)),
                "ABS_W_MAX": ("frequency", np.full(shape[0], 1e3)),
                "FLAGGED_FRACTION": ("frequency", np.zeros(shape[0])),
            },
            coords={
                "frequency": frequency,
                "polarization": polarization,
                "beam_params_label": ["major", "minor", "pa"],
            },
        )

    module = types.ModuleType(PSF_MODULE)
    module.point_spread_function_cube_single_field = (
        point_spread_function_cube_single_field
    )
    monkeypatch.setitem(sys.modules, PSF_MODULE, module)
    return calls


def test_delphi_calculate_parameters_only(ps_store, tmp_path, psf_calls):
    image_store = str(tmp_path / "cube.img.zarr")
    parameters = delphi_module.image_cube_single_field_delphi(
        ps_store, image_store, calculate_parameters_only=True, n_sigma=3.0
    )
    path = f"{image_store}.parameters.json"
    with open(path) as file:
        assert json.load(file) == parameters
    assert not (tmp_path / "cube.img.zarr").exists()

    # The PSF pass: the derived polarization, kernel and beam, no image written.
    (call,) = psf_calls
    assert call["image_store"] is None
    assert call["instrument_polarization_basis"] == "circular"
    assert call["image_params"]["polarization_coords"] == ["I", "V"]
    assert call["gridding_kernel_params"] == PROLATE_SPHEROIDAL_9
    assert call["image_params"]["fft_padding"] == 1.2
    assert call["primary_beam_params"] == {"telescope": "vla"}
    assert call["imaging_weights_params"] == {"weighting": "briggs", "robust": 2.0}

    arguments = parameters["image_cube_single_field"]
    expected_names = set(inspect.signature(image_cube_single_field).parameters) - {
        "parameters_file"
    }
    assert set(arguments) == expected_names
    derived = parameters["derived"]
    assert derived["telescope"] == "vla"
    cell_guess = derived["cell_size_guess"]
    assert abs(call["image_params"]["cell_size"][0]) == pytest.approx(cell_guess)
    assert call["image_params"]["image_size"] == [derived["image_size_guess"]] * 2
    # pixels_per_beam = 5 across the minor axis of the highest channel.
    cell = MINOR_AXIS_IN_CELLS * cell_guess / 5.0
    assert derived["cell_size"] == pytest.approx(cell)
    image_params = arguments["image_params"]
    assert image_params["cell_size"] == pytest.approx([-cell, cell])
    size = image_params["image_size"][0]
    assert size % 2 == 0 and size * cell >= derived["field_of_view"]
    assert image_params["phase_direction"] == pytest.approx(PC.tolist())
    assert image_params["polarization_coords"] == ["I", "V"]
    assert len(image_params["frequency_coords"]) == 2
    assert image_params["fft_padding"] == 1.2

    # The controls of the PSF pass, every key written out.
    controls = arguments["cycle_control_params"]
    stand_in = sys.modules[PSF_MODULE].point_spread_function_cube_single_field(**call)
    assert controls == noisy_cycle_control_params(stand_in, n_sigma=3.0)
    np.testing.assert_allclose(
        controls["threshold"], 3.0 * np.array([[1e-3, 2e-3], [3e-3, 4e-3]])
    )
    assert controls["threshold_tolerance"] == 0.02
    assert controls["exact_psf_subtraction"] is False
    assert not {"psf_fraction", "predicted_imaging_cycles"} & set(derived)
    assert arguments["gridding_kernel_params"] == PROLATE_SPHEROIDAL_9
    assert arguments["restore"] is True and arguments["primary_beam_correction"]
    np.testing.assert_allclose(
        derived["sensitivity_natural"], 0.8 * np.array([[1e-3, 2e-3], [3e-3, 4e-3]])
    )
    provenance = parameters["provenance"]
    assert provenance["delphi_inputs"]["n_sigma"] == 3.0
    assert isinstance(provenance["astroviper"], str)

    # An existing parameters file is kept unless overwrite.
    with pytest.raises(FileExistsError):
        delphi_module.image_cube_single_field_delphi(
            ps_store, image_store, calculate_parameters_only=True
        )


def test_delphi_images_from_its_parameters_file(
    ps_store, tmp_path, psf_calls, monkeypatch
):
    """Without calculate_parameters_only the cube is imaged from the file alone."""
    imaged = []

    @functools.wraps(image_cube_single_field)
    def fake_image_cube_single_field(**kwargs):
        imaged.append(kwargs)
        return {"deconvolution": "stand-in"}

    monkeypatch.setattr(
        delphi_module, "image_cube_single_field", fake_image_cube_single_field
    )
    parameters_file = str(tmp_path / "chosen.json")
    result = delphi_module.image_cube_single_field_delphi(
        ps_store,
        str(tmp_path / "cube.img.zarr"),
        weighting="natural",
        parameters_file=parameters_file,
    )
    assert imaged == [{"parameters_file": parameters_file}]
    assert result["deconvolution"] == "stand-in"
    with open(parameters_file) as file:
        assert json.load(file) == result["parameters"]
    assert result["parameters"]["image_cube_single_field"][
        "imaging_weights_params"
    ] == {"weighting": "natural"}


def test_delphi_n_sigma_default_is_the_heuristic():
    parameters = inspect.signature(
        delphi_module.image_cube_single_field_delphi
    ).parameters
    assert parameters["n_sigma"].default == NOISY_N_SIGMA
    assert "max_dynamic_range" not in parameters


def test_delphi_end_to_end_with_stand_in_psf_pass(ps_store, tmp_path, psf_calls):
    """The parameters file drives the real imager: both channels are cleaned
    (per channel thresholds of a few mJy) until every plane stops on its
    threshold within the tolerance, then restored and primary beam corrected.
    A round trip check: delphi's former controls stop there too, so the
    values are guarded by the tests of noisy_cycle_control_params."""
    image_store = str(tmp_path / "cube.img.zarr")
    result = delphi_module.image_cube_single_field_delphi(
        ps_store,
        image_store,
        processing_set_data_group_name="base",
        n_mapping_parallelism={"frequency": 2},
    )
    assert set(result["parameters"]) == {
        "image_cube_single_field",
        "derived",
        "provenance",
    }
    img = load_image(image_store)
    size = result["parameters"]["image_cube_single_field"]["image_params"]["image_size"]
    assert img.sizes["l"] == size[0] and img.sizes["m"] == size[1]
    model = img.SKY_MODEL.values[0, :, 0]  # [frequency, l, m]
    np.testing.assert_allclose(np.nansum(model, axis=(1, 2)), [1.0, 1.0], rtol=0.05)
    for name in ("SKY_RESTORED", "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"):
        assert name in img.data_vars
    controls = result["parameters"]["image_cube_single_field"]["cycle_control_params"]
    record = result["deconvolution"]
    assert {(key.chan, key.pol) for key in record.data} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    for key, entry in record.data.items():
        threshold = controls["threshold"][key.chan][key.pol]
        assert entry["stop_code"].imaging == IMAGING_THRESHOLD, key
        assert len(entry["iter_done"]) - 1 <= controls["max_cycles"], key
        assert abs(entry["peakres"][-1]) <= threshold * (
            1 + controls["threshold_tolerance"]
        ), key


def test_delphi_skips_a_fully_flagged_highest_channel(ps_store, tmp_path, psf_calls):
    """A flagged channel has a zero beam fit and infinite sensitivity."""
    module = sys.modules[PSF_MODULE]
    psf = module.point_spread_function_cube_single_field

    def flagged_highest_channel(*args, **kwargs):
        xds = psf(*args, **kwargs)
        top = int(np.argmax(xds.frequency.values))
        xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"][top] = 0.0
        xds["SENSITIVITY"][top] = np.inf
        xds["FLAGGED_FRACTION"][top] = 1.0
        return xds

    module.point_spread_function_cube_single_field = flagged_highest_channel
    image_store = str(tmp_path / "cube.img.zarr")
    parameters = delphi_module.image_cube_single_field_delphi(
        ps_store, image_store, calculate_parameters_only=True
    )
    with open(f"{image_store}.parameters.json") as file:
        text = file.read()
    assert "NaN" not in text and "Infinity" not in text
    # The cell comes from the remaining channel, whose beam is wider.
    derived = parameters["derived"]
    frequency = np.asarray(derived["frequency"])
    scale = frequency.max() / frequency.min()
    cell = MINOR_AXIS_IN_CELLS * derived["cell_size_guess"] * scale / 5.0
    assert derived["cell_size"] == pytest.approx(cell)
    threshold = parameters["image_cube_single_field"]["cycle_control_params"][
        "threshold"
    ]
    assert threshold[int(np.argmax(frequency))] == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# Metadata, input checks, uv coverage and MPI options of delphi
# --------------------------------------------------------------------------- #
def _metadata_processing_set(
    correlations=(("RR", "LL"),),
    observation_info=None,
    antenna_attrs=None,
    antenna_coords=None,
    field_names=("target",),
):
    """The parts of a processing set that _processing_set_metadata reads."""
    measurement_sets = {
        f"ms_{i}": types.SimpleNamespace(
            ds=xr.Dataset(
                coords={"polarization": list(labels)},
                attrs={"observation_info": observation_info or {}},
            )
        )
        for i, labels in enumerate(correlations)
    }
    antenna_xds = xr.Dataset(
        {"ANTENNA_POSITION": (("antenna_name", "cartesian_pos_label"), np.eye(3)[:2])},
        coords={
            "antenna_name": ["a0", "a1"],
            "cartesian_pos_label": ["x", "y", "z"],
            **(antenna_coords or {}),
        },
        attrs=antenna_attrs or {},
    )
    field_xds = xr.Dataset(
        {
            "FIELD_PHASE_CENTER_DIRECTION": (
                ("field_name", "sky_dir_label"),
                np.tile(PC, (len(field_names), 1)),
            )
        },
        coords={"field_name": list(field_names), "sky_dir_label": ["ra", "dec"]},
        attrs={"center_field_name": field_names[0]},
    )
    ps_xdt = types.SimpleNamespace(
        values=measurement_sets.values,
        xr_ps=types.SimpleNamespace(
            get_combined_antenna_xds=lambda: antenna_xds,
            get_combined_field_and_source_xds=lambda: field_xds,
            get_freq_axis=lambda: xr.DataArray([3e9, 3.4e9], dims="frequency"),
        ),
    )
    return ps_xdt


@pytest.mark.parametrize(
    "where, expected",
    [
        ({"observation_info": {"telescope_name": "ALMA"}}, "ALMA"),
        ({"antenna_attrs": {"overall_telescope_name": "VLA"}}, "VLA"),
        (
            {"antenna_coords": {"telescope_name": ("antenna_name", ["EVLA", "EVLA"])}},
            "EVLA",
        ),
    ],
)
def test_metadata_telescope_name_fallbacks(where, expected):
    """The observation's telescope name, else the antennas' overall name, else
    the first antenna's."""
    metadata = delphi_module._processing_set_metadata(
        _metadata_processing_set(**where), None
    )
    assert metadata["telescope_name"] == expected
    assert metadata["dish_diameters"] is None
    np.testing.assert_array_equal(metadata["phase_direction"], PC)
    np.testing.assert_array_equal(metadata["frequency_coords"], [3e9, 3.4e9])


def test_metadata_correlations_every_measurement_set_holds():
    metadata = delphi_module._processing_set_metadata(
        _metadata_processing_set(
            correlations=(("RR", "RL", "LR", "LL"), ("LL", "RR")),
            observation_info={"telescope_name": "VLA"},
        ),
        "target",
    )
    assert metadata["correlations"] == ["LL", "RR"]


def test_metadata_refuses_several_fields():
    with pytest.raises(ValueError, match="The imager makes one field"):
        delphi_module._processing_set_metadata(
            _metadata_processing_set(field_names=("a", "b")), None
        )


@pytest.mark.parametrize("name", ["pixels_per_beam", "n_sigma"])
def test_delphi_refuses_a_non_positive_input(tmp_path, name):
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        delphi_module.image_cube_single_field_delphi(
            "unused.ps.zarr", str(tmp_path / "cube.img.zarr"), **{name: 0}
        )


def test_delphi_warns_when_the_cell_is_too_large_for_the_uv_grid(
    ps_store, tmp_path, psf_calls, monkeypatch
):
    """The cell of the cube may leave the longest baselines off its uv grid
    when it exceeds 1 / (2 u_max) = 1.5 times the cell of the PSF pass."""
    import toolviper.utils.logger as logger

    warnings = []
    monkeypatch.setattr(
        logger, "warning", lambda message, *args, **kwargs: warnings.append(message)
    )
    derived = delphi_module.image_cube_single_field_delphi(
        ps_store, str(tmp_path / "narrow.img.zarr"), calculate_parameters_only=True
    )["derived"]
    assert derived["cell_size"] < 1.5 * derived["cell_size_guess"]
    assert not any("off the uv grid" in message for message in warnings)

    module = sys.modules[PSF_MODULE]
    psf = module.point_spread_function_cube_single_field

    def wide_beam(*args, **kwargs):
        xds = psf(*args, **kwargs)
        xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] *= 3.0
        return xds

    monkeypatch.setattr(module, "point_spread_function_cube_single_field", wide_beam)
    parameters = delphi_module.image_cube_single_field_delphi(
        ps_store, str(tmp_path / "wide.img.zarr"), calculate_parameters_only=True
    )
    derived = parameters["derived"]
    # 3 * MINOR_AXIS_IN_CELLS cells of the PSF pass across 5 pixels
    assert derived["cell_size"] == pytest.approx(2.4 * derived["cell_size_guess"])
    assert any("off the uv grid" in message for message in warnings)


def test_delphi_gives_the_teardown_to_the_last_compute_only(
    ps_store, tmp_path, psf_calls, monkeypatch
):
    """graphviper arms teardown_force_exit_seconds at the end of a compute, so
    the PSF pass runs without it unless it is the only compute."""

    @functools.wraps(image_cube_single_field)
    def fake_image_cube_single_field(**kwargs):
        return {}

    monkeypatch.setattr(
        delphi_module, "image_cube_single_field", fake_image_cube_single_field
    )
    mpi_cluster_setup = {"max_workers": 2, "teardown_force_exit_seconds": 30}
    result = delphi_module.image_cube_single_field_delphi(
        ps_store,
        str(tmp_path / "imaged.img.zarr"),
        compute_backend="mpi",
        mpi_cluster_setup=mpi_cluster_setup,
    )
    assert psf_calls[-1]["mpi_cluster_setup"] == {"max_workers": 2}
    arguments = result["parameters"]["image_cube_single_field"]
    assert arguments["mpi_cluster_setup"] == mpi_cluster_setup
    delphi_module.image_cube_single_field_delphi(
        ps_store,
        str(tmp_path / "parameters_only.img.zarr"),
        calculate_parameters_only=True,
        compute_backend="mpi",
        mpi_cluster_setup=mpi_cluster_setup,
    )
    assert psf_calls[-1]["mpi_cluster_setup"] == mpi_cluster_setup
    assert mpi_cluster_setup == {"max_workers": 2, "teardown_force_exit_seconds": 30}
