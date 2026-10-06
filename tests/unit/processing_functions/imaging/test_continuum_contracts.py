"""Continuum numerical and state contracts, independent of CASA references."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    accumulate_continuum_model,
    convert_mvc_cubes_to_taylor_normal_equations,
    finalize_mvc_taylor_normal_equations,
    form_residual_grid_from_cache,
    make_mvc_taylor_normal_equation_contributions,
    primary_beam_correct_restored_continuum,
)


def _mvc_inputs(dtype=np.float64):
    # Multiple times and polarizations must each retain their own weight sum.
    coords = dict(
        time=[0, 1],
        frequency=[80.0, 100.0, 120.0],
        polarization=["I", "Q"],
        l=[0, 1],
        m=[0, 1],
    )
    shape = tuple(len(v) for v in coords.values())
    residual = xr.DataArray(
        np.arange(np.prod(shape), dtype=dtype).reshape(shape) + 1,
        dims=tuple(coords),
        coords=coords,
    )
    beam = xr.ones_like(residual) * dtype(0.8)
    psf = xr.ones_like(residual)
    weights = xr.DataArray(
        np.arange(12, dtype=float).reshape(2, 3, 2) + 1,
        dims=("time", "frequency", "polarization"),
        coords={k: coords[k] for k in ("time", "frequency", "polarization")},
    )
    return residual, psf, beam, weights, weights.copy(deep=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nterms", [1, 2, 3])
def test_mvc_moments_preserve_independent_time_and_polarization_planes(dtype, nterms):
    inputs = _mvc_inputs(dtype)
    residual, psf, beam = convert_mvc_cubes_to_taylor_normal_equations(
        *inputs, nterms=nterms, reference_frequency=100.0
    )
    # Independent scalar sums, deliberately avoiding xarray's reduction/alignment.
    expected = np.zeros((2, nterms, 2, 2, 2))
    expected_psf = np.zeros((2, 2 * nterms - 1, 2, 2, 2))
    for time in range(2):
        for pol in range(2):
            weights = inputs[3].values[time, :, pol]
            for term in range(nterms):
                for channel, x in enumerate([-0.2, 0.0, 0.2]):
                    expected[time, term, pol] += (
                        weights[channel]
                        * x**term
                        * inputs[0].values[time, channel, pol]
                        / weights.sum()
                    )
            for term in range(2 * nterms - 1):
                expected_psf[time, term, pol] = (
                    sum(
                        w * x**term
                        for w, x in zip(weights, [-0.2, 0.0, 0.2], strict=True)
                    )
                    / weights.sum()
                )
    np.testing.assert_allclose(
        residual.transpose("time", "taylor_term", "polarization", "l", "m"),
        expected,
        rtol=2e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        psf.transpose("time", "psf_taylor_order", "polarization", "l", "m"),
        expected_psf,
        rtol=2e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(beam, 0.8)
    assert residual.dtype == psf.dtype == beam.dtype == dtype


@pytest.mark.parametrize("bad_weight", [0.0, -1.0, np.nan, np.inf])
def test_mvc_ignores_invalid_channel_weights_but_rejects_empty_planes(bad_weight):
    inputs = list(_mvc_inputs())
    for weights in inputs[3:]:
        weights.values[:, 0, :] = bad_weight
    actual = convert_mvc_cubes_to_taylor_normal_equations(
        *inputs, nterms=2, reference_frequency=100.0
    )
    expected = convert_mvc_cubes_to_taylor_normal_equations(
        *(array.isel(frequency=slice(1, None)) for array in inputs),
        nterms=2,
        reference_frequency=100.0,
    )
    for left, right in zip(actual, expected, strict=True):
        xr.testing.assert_allclose(left, right)
    inputs[3].values[1, :, 1] = bad_weight
    with pytest.raises(ValueError, match="residual normalization.*empty"):
        convert_mvc_cubes_to_taylor_normal_equations(
            *inputs, nterms=2, reference_frequency=100.0
        )


def test_mvc_rejects_empty_psf_plane_independently_of_residual_weights():
    inputs = list(_mvc_inputs())
    inputs[4].values[0, :, 1] = 0
    with pytest.raises(ValueError, match="PSF normalization.*empty"):
        convert_mvc_cubes_to_taylor_normal_equations(
            *inputs, nterms=2, reference_frequency=100.0
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("nterms", 0),
        ("reference_frequency", 0),
        ("reference_frequency", np.nan),
        ("reference_frequency", np.inf),
        ("pblimit", -0.1),
        ("pblimit", np.nan),
    ],
)
def test_mvc_rejects_invalid_equation_parameters(parameter, value):
    kwargs = dict(nterms=2, reference_frequency=100.0, pblimit=0.2)
    kwargs[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        make_mvc_taylor_normal_equation_contributions(*_mvc_inputs(), **kwargs)


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_mvc_rejects_misaligned_channel_coordinates(index):
    inputs = list(_mvc_inputs())
    inputs[index] = inputs[index].assign_coords(frequency=[81.0, 101.0, 121.0])
    with pytest.raises(ValueError, match="coordinate-aligned"):
        make_mvc_taylor_normal_equation_contributions(
            *inputs, nterms=2, reference_frequency=100.0
        )


def test_mvc_later_residual_update_reuses_effective_beam_without_psf():
    inputs = _mvc_inputs()
    expected, _, beam = convert_mvc_cubes_to_taylor_normal_equations(
        *inputs, nterms=2, reference_frequency=100.0
    )
    contribution = make_mvc_taylor_normal_equation_contributions(
        inputs[0], None, inputs[2], inputs[3], None, nterms=2, reference_frequency=100.0
    )
    actual, psf, reused_beam = finalize_mvc_taylor_normal_equations(
        contribution, effective_primary_beam=beam
    )
    xr.testing.assert_allclose(actual, expected)
    xr.testing.assert_allclose(reused_beam, beam)
    assert psf is None
    assert not any("PSF" in name for name in contribution)


def _grid(axis, value):
    return xr.Dataset(
        {
            "GRID": ((axis, "u"), np.full((2, 2), value, dtype=complex)),
            "SUM": ((axis,), [2.0, 3.0]),
        },
        coords={axis: [1.0, 2.0], "u": [0, 1]},
        attrs={
            "data_groups": {
                "residual": {"visibility": "GRID", "visibility_normalization": "SUM"}
            }
        },
    )


@pytest.mark.parametrize("axis", ["frequency", "taylor_term"])
def test_cached_residual_owns_arrays_and_nested_metadata(axis):
    observed, model = _grid(axis, 4 + 3j), _grid(axis, 1 + 2j)
    residual = form_residual_grid_from_cache(observed, model)
    np.testing.assert_array_equal(residual.GRID, 3 + 1j)
    residual.GRID.values[:] = 100
    residual.SUM.values[:] = 100
    residual.attrs["data_groups"]["residual"]["visibility"] = "OTHER"
    np.testing.assert_array_equal(observed.GRID, 4 + 3j)
    np.testing.assert_array_equal(model.GRID, 1 + 2j)
    np.testing.assert_array_equal(observed.SUM, [2, 3])
    assert model.attrs["data_groups"]["residual"]["visibility"] == "GRID"


@pytest.mark.parametrize("axis", ["frequency", "taylor_term"])
@pytest.mark.parametrize(
    "fault",
    ["coordinates", "dimensions", "missing_group", "missing_role", "missing_array"],
)
def test_cached_residual_rejects_incompatible_cache(axis, fault):
    observed, model = _grid(axis, 4), _grid(axis, 1)
    error = ValueError
    if fault == "coordinates":
        model = model.assign_coords({axis: [3.0, 4.0]})
    elif fault == "dimensions":
        model = model.rename({"u": "v"})
    elif fault == "missing_group":
        model.attrs["data_groups"] = {}
        error = KeyError
    elif fault == "missing_role":
        del model.attrs["data_groups"]["residual"]["visibility"]
        error = KeyError
    else:
        model = model.drop_vars("GRID")
        error = KeyError
    with pytest.raises(error):
        form_residual_grid_from_cache(observed, model)


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
@pytest.mark.parametrize("fault", ["missing_model", "dimensions", "shape"])
def test_model_accumulation_rejects_incompatible_state(specmode, fault):
    previous = xr.Dataset({"SKY_MODEL": (("taylor_term", "l"), np.ones((2, 2)))})
    increment = previous.copy(deep=True)
    if fault == "missing_model":
        increment = increment.drop_vars("SKY_MODEL")
    elif fault == "dimensions":
        increment = increment.rename({"l": "m"})
    else:
        increment = increment.isel(l=slice(0, 1))
    with pytest.raises(KeyError if fault == "missing_model" else ValueError):
        accumulate_continuum_model(increment, previous, specmode=specmode)


@pytest.mark.parametrize(
    "fault",
    [
        "channel_beam",
        "missing_beam",
        "missing_restored",
        "negative_limit",
        "unit_limit",
    ],
)
def test_pb_correction_rejects_invalid_final_products(fault):
    """An unreduced channel beam must not broadcast a continuum image to a cube."""
    image = xr.Dataset(
        {
            "RESTORED": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.ones((1, 2, 1, 2, 2)),
            ),
            "PRIMARY_BEAM": (
                ("time", "frequency", "polarization", "l", "m"),
                np.ones((1, 2 if fault == "channel_beam" else 1, 1, 2, 2)),
            ),
        },
        attrs={"data_groups": {"restored": {"sky": "RESTORED"}}},
    )
    pblimit = 0.2
    if fault == "missing_beam":
        image = image.drop_vars("PRIMARY_BEAM")
    elif fault == "missing_restored":
        image = image.drop_vars("RESTORED")
    elif fault == "negative_limit":
        pblimit = -0.1
    elif fault == "unit_limit":
        pblimit = 1.0
    with pytest.raises(KeyError if fault.startswith("missing") else ValueError):
        primary_beam_correct_restored_continuum(image, pblimit=pblimit)
