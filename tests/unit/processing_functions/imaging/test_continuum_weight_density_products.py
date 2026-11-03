"""Runtime density contracts shared by reduction, factor preparation and application."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from astroviper.processing_functions.imaging.calculate_imaging_weights import (
    check_continuum_briggs_factor_shape,
    check_continuum_weight_density_products,
    degrid_imaging_weights_continuum,
)


@pytest.fixture
def density():
    return xr.Dataset(
        {
            "WEIGHT_DENSITY_GRID": (
                ("frequency", "weight_polarization", "u", "v"),
                np.ones((1, 1, 2, 2)),
            ),
            "SUM_WEIGHT": (("frequency", "weight_polarization"), np.ones((1, 1))),
        },
        coords={"frequency": [1e9]},
        attrs={"continuum_frequency_collapsed": True},
    )


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("type", TypeError, "Dataset"),
        ("missing", KeyError, "SUM_WEIGHT"),
        ("coordinate", KeyError, "frequency coordinate"),
        ("density_dims", ValueError, "WEIGHT_DENSITY_GRID has dimensions"),
        ("sum_dims", ValueError, "SUM_WEIGHT has dimensions"),
        ("nan", ValueError, "non-finite"),
        ("empty", ValueError, "no frequency"),
        ("uncollapsed", ValueError, "frequency-collapsed"),
        ("multiple", ValueError, "exactly one"),
        ("duplicate", ValueError, "duplicate"),
    ],
)
def test_malformed_density_metadata(density, fault, error, message):
    if fault == "type":
        density = []
    elif fault == "missing":
        density = density.drop_vars("SUM_WEIGHT")
    elif fault == "coordinate":
        density = density.drop_vars("frequency")
    elif fault == "density_dims":
        density = density.rename({"u": "other"})
    elif fault == "sum_dims":
        density["SUM_WEIGHT"] = density.SUM_WEIGHT.transpose(
            "weight_polarization", "frequency"
        )
    elif fault == "nan":
        density = density.assign_coords(frequency=[np.nan])
    elif fault == "empty":
        density = density.isel(frequency=slice(0, 0))
    elif fault == "uncollapsed":
        density.attrs["continuum_frequency_collapsed"] = False
    elif fault in ("multiple", "duplicate"):
        density = xr.concat([density, density], dim="frequency").assign_coords(
            frequency=[1e9, 2e9 if fault == "multiple" else 1e9]
        )
    with pytest.raises(error, match=message):
        check_continuum_weight_density_products(density, name="test result")


def test_metadata_checks_leave_inputs_lazy_and_unchanged(density):
    @delayed
    def forbidden_read():
        pytest.fail("density pixels were loaded by a metadata check")

    density["WEIGHT_DENSITY_GRID"] = (
        density.WEIGHT_DENSITY_GRID.dims,
        da.from_delayed(forbidden_read(), shape=(1, 1, 2, 2), dtype=float),
    )
    original_data = density.WEIGHT_DENSITY_GRID.data
    original_attrs = density.attrs.copy()
    check_continuum_weight_density_products(density)
    assert density.WEIGHT_DENSITY_GRID.data is original_data
    assert density.attrs == original_attrs


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 1), (2, 1, 2), (2, 1)])
def test_factor_shape_matches_density_planes(density, shape):
    with pytest.raises(ValueError, match="unexpected shape"):
        check_continuum_briggs_factor_shape(np.ones(shape), density)


def test_factor_shape_accepts_array_and_stored_product(density):
    values = np.ones((2, 1, 1))
    check_continuum_briggs_factor_shape(values, density)
    check_continuum_briggs_factor_shape(
        xr.DataArray(
            values, dims=("briggs_parameter", "frequency", "weight_polarization")
        ),
        density,
    )


@pytest.mark.parametrize("input_checked", [False, True])
def test_runtime_density_validation_cannot_be_skipped(density, input_checked):
    density.attrs["continuum_frequency_collapsed"] = False
    # Validation fails before touching the processing set or image products.
    with pytest.raises(ValueError, match="frequency-collapsed"):
        degrid_imaging_weights_continuum(
            None, None, density, {"weighting": "uniform"}, input_checked=input_checked
        )


@pytest.mark.parametrize("fault", ["factor_dims", "factor_count", "image_geometry"])
def test_application_retains_stage_specific_product_checks(density, fault):
    density["BRIGGS_FACTORS"] = (
        ("briggs_parameter", "frequency", "weight_polarization"),
        np.ones((1 if fault == "factor_count" else 2, 1, 1)),
    )
    if fault == "factor_dims":
        density["BRIGGS_FACTORS"] = density.BRIGGS_FACTORS.transpose(
            "frequency", "briggs_parameter", "weight_polarization"
        )
    image = xr.Dataset(coords={"l": [0, 1, 2], "m": [0, 1]})
    expected = {
        "factor_dims": "BRIGGS_FACTORS has dimensions",
        "factor_count": "unexpected shape",
        "image_geometry": "geometry does not match",
    }
    with pytest.raises(ValueError, match=expected[fault]):
        degrid_imaging_weights_continuum(
            None, image, density, {"weighting": "uniform"}, input_checked=True
        )
