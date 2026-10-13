"""Exception contracts using deliberately incomplete in-memory images only."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    accumulate_continuum_model,
    apply_mvc_primary_beam_convention,
    make_mvc_taylor_normal_equation_contributions,
    model_update_mtmfs_single_field,
    point_spread_function_gaussian_fit_continuum,
    restore_image,
)
from astroviper.processing_functions.imaging.residual_cycle_continuum_single_field import (
    make_visibility_model_continuum_single_field,
)


def _image():
    dims = ("time", "taylor_term", "polarization", "l", "m")
    return xr.Dataset(
        {
            "SKY_RESIDUAL": (dims, np.ones((1, 2, 1, 2, 2))),
            "SKY_MODEL": (dims, np.zeros((1, 2, 1, 2, 2))),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                np.ones((1, 3, 1, 2, 2)),
            ),
            "BEAM": ("beam_parameter", [1.0, 1.0, 0.0]),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                    "beam_fit_params_point_spread_function": "BEAM",
                },
                "model": {"sky": "SKY_MODEL"},
            }
        },
    )


@pytest.mark.parametrize("group", ["residual", "model"])
@pytest.mark.parametrize(
    "fault,message",
    [
        ("group", "data group"),
        ("role", "define 'sky'"),
        ("variable", "variable.*missing"),
        ("dimension", "taylor_term"),
        ("empty", "no Taylor terms"),
    ],
)
def test_restore_rejects_invalid_model_or_residual(group, fault, message):
    image = _image()
    variable = image.attrs["data_groups"][group]["sky"]
    if fault == "group":
        del image.attrs["data_groups"][group]
    elif fault == "role":
        del image.attrs["data_groups"][group]["sky"]
    elif fault == "variable":
        image = image.drop_vars(variable)
    elif fault == "dimension":
        image[variable] = image[variable].rename({"taylor_term": "other"})
    else:
        image = image.isel(taylor_term=slice(0, 0))
    snapshot = image.copy(deep=True)
    with pytest.raises(
        ValueError if fault in ("dimension", "empty") else KeyError, match=message
    ):
        restore_image(image)
    xr.testing.assert_identical(image, snapshot)


@pytest.mark.parametrize(
    "role,variable",
    [
        ("point_spread_function", "POINT_SPREAD_FUNCTION"),
        ("beam_fit_params_point_spread_function", "BEAM"),
    ],
)
@pytest.mark.parametrize("registered", [False, True])
def test_restore_rejects_missing_registered_or_fallback_products(
    role, variable, registered
):
    image = _image().drop_vars(variable)
    if not registered:
        del image.attrs["data_groups"]["residual"][role]
    with pytest.raises(KeyError, match="absent|missing"):
        restore_image(image)


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("deconvolver", NotImplementedError, "hogbom"),
        ("residual", KeyError, "SKY_RESIDUAL"),
        ("psf", KeyError, "POINT_SPREAD_FUNCTION"),
        ("group", KeyError, "Input data group"),
        ("residual_dimension", ValueError, "taylor_term"),
        ("psf_dimension", ValueError, "psf_taylor_order"),
        ("empty_residual", ValueError, "no Taylor terms"),
        ("empty_psf", ValueError, "no PSF Taylor terms"),
        ("control_dimensions", ValueError, "three dimensions"),
        ("empty_controls", ValueError, "no image planes"),
        ("sidelobe", KeyError, "MAX_SIDELOBE"),
    ],
)
def test_model_update_rejects_incomplete_images_and_controls(fault, error, message):
    image = _image()
    controls = {}
    deconvolver = "hogbom"
    if fault == "deconvolver":
        deconvolver = "unsupported"
    elif fault in ("residual", "psf"):
        image = image.drop_vars(
            "SKY_RESIDUAL" if fault == "residual" else "POINT_SPREAD_FUNCTION"
        )
    elif fault == "group":
        del image.attrs["data_groups"]["residual"]
    elif fault in ("residual_dimension", "psf_dimension"):
        axis = "taylor_term" if fault == "residual_dimension" else "psf_taylor_order"
        image = image.rename({axis: "other"})
    elif fault in ("empty_residual", "empty_psf"):
        image = image.isel(
            {
                (
                    "taylor_term" if fault == "empty_residual" else "psf_taylor_order"
                ): slice(0, 0)
            }
        )
    elif fault == "control_dimensions":
        controls["max_iter_per_cycle"] = np.ones((1, 2))
    elif fault == "empty_controls":
        controls["threshold_per_cycle"] = np.ones((1, 0, 1))
    snapshot = image.copy(deep=True)
    with pytest.raises(error, match=message):
        model_update_mtmfs_single_field(image, deconvolver, controls)
    xr.testing.assert_identical(image, snapshot)


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("group", KeyError, "Input data group"),
        ("role", KeyError, "point_spread_function"),
        ("variable", KeyError, "PSF variable"),
        ("order_dimension", ValueError, "psf_taylor_order"),
        ("empty", ValueError, "no Taylor-order"),
        ("spatial_dimension", ValueError, "missing required dimensions"),
    ],
)
def test_psf_fitter_rejects_incomplete_products(fault, error, message):
    image = _image()
    if fault == "group":
        image.attrs["data_groups"] = {}
    elif fault == "role":
        del image.attrs["data_groups"]["residual"]["point_spread_function"]
    elif fault == "variable":
        image = image.drop_vars("POINT_SPREAD_FUNCTION")
    elif fault == "order_dimension":
        image = image.rename({"psf_taylor_order": "other"})
    elif fault == "empty":
        image = image.isel(psf_taylor_order=slice(0, 0))
    else:
        image["POINT_SPREAD_FUNCTION"] = image.POINT_SPREAD_FUNCTION.isel(
            l=0, drop=True
        )
    with pytest.raises(error, match=message):
        point_spread_function_gaussian_fit_continuum(image)


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("order", ValueError, "nterms"),
        ("reference", ValueError, "reference_frequency"),
        ("group", KeyError, "data group"),
        ("role", KeyError, "visibility"),
        ("variable", KeyError, "not present"),
        ("dimension", ValueError, "taylor_term"),
        ("terms", ValueError, "Taylor terms"),
        ("frequency", KeyError, "frequency"),
        ("nonfinite", ValueError, "non-finite"),
        ("frequency_dimension", ValueError, "one-dimensional"),
    ],
)
def test_prediction_rejects_incomplete_models_and_frequencies(fault, error, message):
    model = xr.Dataset(
        {
            "GRID": (
                ("time", "taylor_term", "polarization", "u", "v"),
                np.ones((1, 2, 1, 2, 2), dtype=complex),
            )
        },
        attrs={"data_groups": {"model": {"visibility": "GRID"}}},
    )
    child = xr.Dataset(coords={"frequency": [100.0, 110.0]})
    nterms, reference = 2, 100.0
    if fault == "order":
        nterms = 0
    elif fault == "reference":
        reference = np.inf
    elif fault == "group":
        model.attrs["data_groups"] = {}
    elif fault == "role":
        model.attrs["data_groups"]["model"] = {}
    elif fault == "variable":
        model = model.drop_vars("GRID")
    elif fault == "dimension":
        model = model.rename({"taylor_term": "other"})
    elif fault == "terms":
        nterms = 3
    elif fault == "frequency":
        child = xr.Dataset()
    elif fault == "nonfinite":
        child = child.assign_coords(frequency=[100.0, np.nan])
    else:
        child = xr.Dataset({"frequency": (("a", "b"), [[100.0, 110.0]])})
    with pytest.raises(error, match=message):
        make_visibility_model_continuum_single_field(
            {"child": child}, model, np.ones(8), nterms, reference
        )


def _cubes():
    coords = dict(
        time=[0.0],
        frequency=[100.0, 110.0],
        polarization=["I"],
        l=[0.0, 1.0],
        m=[0.0, 1.0],
    )
    model = xr.DataArray(np.ones((1, 2, 1, 2, 2)), dims=tuple(coords), coords=coords)
    return (
        model,
        model.copy(deep=True),
        model.isel(frequency=0, drop=True).copy(deep=True),
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_pb_prediction_rejects_wrong_dimension_order(index):
    cubes = list(_cubes())
    cubes[index] = cubes[index].transpose(*reversed(cubes[index].dims))
    with pytest.raises(ValueError, match="dimensions"):
        apply_mvc_primary_beam_convention(*cubes)


@pytest.mark.parametrize("index", [1, 2])
def test_pb_prediction_rejects_incompatible_shapes(index):
    cubes = list(_cubes())
    cubes[index] = cubes[index].isel(l=[0])
    with pytest.raises(ValueError, match="incompatible shapes"):
        apply_mvc_primary_beam_convention(*cubes)


@pytest.mark.parametrize(
    "index,coordinate",
    [
        (1, "time"),
        (1, "frequency"),
        (1, "l"),
        (1, "m"),
        (2, "time"),
        (2, "l"),
        (2, "m"),
    ],
)
def test_pb_prediction_rejects_coordinate_misalignment(index, coordinate):
    cubes = list(_cubes())
    cubes[index] = cubes[index].assign_coords(
        {coordinate: cubes[index][coordinate] + 1}
    )
    with pytest.raises(ValueError, match=f"{coordinate} coordinates.*not aligned"):
        apply_mvc_primary_beam_convention(*cubes)


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("increment_type", TypeError, "model_increment_xds"),
        ("mode", ValueError, "specmode"),
        ("initial_pb", KeyError, "PRIMARY_BEAM"),
        ("previous_type", TypeError, "previous_model_xds"),
        ("previous_model", KeyError, "SKY_MODEL"),
    ],
)
def test_accumulation_rejects_invalid_initial_or_previous_state(fault, error, message):
    increment = _image()
    previous = None
    mode = "mfs"
    if fault == "increment_type":
        increment = None
    elif fault == "mode":
        mode = "unknown"
    elif fault == "initial_pb":
        mode = "mvc"
    elif fault == "previous_type":
        previous = []
    else:
        previous = xr.Dataset()
    with pytest.raises(error, match=message):
        accumulate_continuum_model(increment, previous, specmode=mode)


@pytest.mark.parametrize(
    "fault,message",
    [
        ("residual", "residual_cube"),
        ("psf", "psf_cube"),
        ("pb", "primary_beam_cube"),
        ("weights", "residual_normalization"),
        ("psf_weights", "psf_normalization"),
        ("missing_psf_weights", "required"),
        ("empty", "at least one channel"),
    ],
)
def test_mvc_equations_reject_invalid_cube_and_weight_layouts(fault, message):
    residual, pb, _ = _cubes()
    psf = residual.copy(deep=True)
    weights = residual.isel(l=0, m=0, drop=True)
    inputs = [residual, psf, pb, weights, weights.copy(deep=True)]
    indexes = {"residual": 0, "psf": 1, "pb": 2, "weights": 3, "psf_weights": 4}
    if fault in indexes:
        i = indexes[fault]
        inputs[i] = inputs[i].rename({"frequency": "other"})
    elif fault == "missing_psf_weights":
        inputs[4] = None
    else:
        inputs = [value.isel(frequency=slice(0, 0)) for value in inputs]
    with pytest.raises(ValueError, match=message):
        make_mvc_taylor_normal_equation_contributions(
            *inputs, nterms=2, reference_frequency=100.0
        )
