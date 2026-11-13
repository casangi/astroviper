"""Final-only MFS beam fits and globally reduced MVC support keep their equations."""

import importlib

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    finalize_mvc_taylor_normal_equations,
    fit_mfs_primary_beam_taylor,
    make_mvc_taylor_normal_equation_contributions,
)
from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _convert_primary_beam_to_reference_frequency,
)
from tests.unit.processing_functions.imaging.test_continuum_contracts import _mvc_inputs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("beam_model", ["airy", "casa_airy"])
@pytest.mark.parametrize("nterms", [2, 3, 4])
def test_reduced_mfs_fit_matches_partition_local_projection(dtype, beam_model, nterms):
    # Each map task streams its local channels before the additive products are
    # globally reduced and fitted in the first append node.
    frequency = np.asarray([0.8, 0.95, 1.1, 1.2, 1.3, 1.5, 1.7]) * 1e9
    params = {
        "image_size": [11, 9],
        "cell_size": [-2e-3, 2e-3],
        "n_taylor_terms": nterms,
        "reference_frequency": 1e9,
        "primary_beam_model": beam_model,
    }
    values = np.random.default_rng(42).uniform(0.2, 5, (len(frequency), 2))
    partitions = ([0, 2, 4], [1, 3, 5, 6])
    images = []
    lookup_frequency_max = np.empty_like(frequency)
    for indices in partitions:
        local_frequency = frequency[indices]
        lookup_frequency_max[indices] = local_frequency.max()
        image = xr.Dataset(
            coords={
                "time": [0, 1],
                "frequency": local_frequency,
                "polarization": ["XX", "YY"],
                "l": np.arange(11),
                "m": np.arange(9),
            },
            attrs={"data_groups": {"residual": {}}},
        )
        weights = xr.DataArray(
            values[indices],
            dims=("frequency", "polarization"),
            coords={"frequency": local_frequency, "polarization": ["XX", "YY"]},
            name="MFS_CHANNEL_IMAGING_WEIGHT_SUM",
        )
        images.append(
            _convert_primary_beam_to_reference_frequency(
                image,
                image_params=params,
                reference_frequency=1e9,
                n_taylor_terms=nterms,
                float_dtype=dtype,
                channel_imaging_weight_sums=weights,
            )
        )
    global_image = xr.Dataset(
        coords={
            "time": [0, 1],
            "frequency": frequency,
            "polarization": ["XX", "YY"],
            "l": np.arange(11),
            "m": np.arange(9),
        },
        attrs={"data_groups": {"residual": {}}},
    )
    global_weights = xr.DataArray(
        values,
        dims=("frequency", "polarization"),
        coords={"frequency": frequency, "polarization": ["XX", "YY"]},
    )
    previous_deferred_projection = _convert_primary_beam_to_reference_frequency(
        global_image,
        image_params=params,
        reference_frequency=1e9,
        n_taylor_terms=nterms,
        float_dtype=dtype,
        channel_imaging_weight_sums=global_weights,
        channel_lookup_frequency_max=lookup_frequency_max,
    )
    names = ["MFS_PRIMARY_BEAM_TAYLOR_NUMERATOR", "MFS_PRIMARY_BEAM_FIT_MOMENTS"]
    expected = fit_mfs_primary_beam_taylor(previous_deferred_projection[names])
    actual = fit_mfs_primary_beam_taylor(
        images[1][names] + images[0][names], primary_beam_limit=0.2
    )
    xr.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-12)


def _global_mvc_contributions():
    inputs = list(_mvc_inputs())
    # One pixel has only one supported channel, another has none. This exercises
    # the scalar Taylor-zero fallback and the primary-beam exclusion as well.
    inputs[2].values[:, 1:, :, 0, 0] = 0
    inputs[2].values[:, :, :, 0, 1] = 0
    chunks = [
        make_mvc_taylor_normal_equation_contributions(
            *(array.isel(frequency=selection) for array in inputs),
            n_taylor_terms=2,
            reference_frequency=100.0,
        )
        for selection in (slice(0, 1), slice(1, None))
    ]
    combined = chunks[0] + chunks[1]
    combined.attrs = dict(chunks[0].attrs)
    return combined


def test_cached_boolean_support_matches_integer_count_and_fallback():
    raw = _global_mvc_contributions()
    expected = finalize_mvc_taylor_normal_equations(raw)
    compact = raw.drop_vars("MVC_RESIDUAL_BEAM_CHANNEL_COUNT")
    compact["MVC_RESIDUAL_BEAM_SUPPORTED"] = raw.MVC_RESIDUAL_BEAM_CHANNEL_COUNT >= 2
    actual = finalize_mvc_taylor_normal_equations(compact)
    assert compact.MVC_RESIDUAL_BEAM_SUPPORTED.dtype == np.bool_
    assert (
        compact.MVC_RESIDUAL_BEAM_SUPPORTED.nbytes * 8
        == raw.MVC_RESIDUAL_BEAM_CHANNEL_COUNT.nbytes
    )
    for result, reference in zip(actual, expected, strict=True):
        xr.testing.assert_identical(result, reference)


@pytest.mark.parametrize("invalid", ["dtype", "order"])
def test_cached_support_rejects_incompatible_mask(invalid):
    raw = _global_mvc_contributions()
    compact = raw.drop_vars("MVC_RESIDUAL_BEAM_CHANNEL_COUNT")
    support = raw.MVC_RESIDUAL_BEAM_CHANNEL_COUNT >= 2
    if invalid == "dtype":
        support = support.astype(np.int64)
        error, message = TypeError, "must be Boolean"
    else:
        support.attrs["n_taylor_terms"] = 3
        error, message = ValueError, "different Taylor order"
    compact["MVC_RESIDUAL_BEAM_SUPPORTED"] = support
    with pytest.raises(error, match=message):
        finalize_mvc_taylor_normal_equations(compact)


def test_global_append_converts_counts_only_after_reduction(monkeypatch):
    node = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    processing = importlib.import_module(
        "astroviper.processing_functions.imaging.image_continuum_single_field"
    )
    raw = _global_mvc_contributions()
    expected = raw.MVC_RESIDUAL_BEAM_CHANNEL_COUNT >= 2
    reached = []

    def check_global_mask(image, **kwargs):
        assert "MVC_RESIDUAL_BEAM_CHANNEL_COUNT" not in image
        xr.testing.assert_equal(
            image.MVC_RESIDUAL_BEAM_SUPPORTED,
            expected.rename("MVC_RESIDUAL_BEAM_SUPPORTED"),
        )
        reached.append(True)
        raise StopIteration

    monkeypatch.setattr(
        processing, "finalize_mvc_taylor_normal_equations", check_global_mask
    )
    with pytest.raises(StopIteration):
        node._prepare_continuum_image(
            raw,
            {
                "continuum_mode": "mvc",
                "image_params": {"n_taylor_terms": 2, "reference_frequency": 100.0},
            },
            initialize_static_products=True,
        )
    assert reached == [True]


@pytest.mark.parametrize(
    "restoring_beam, refused",
    [([5.0e-5, 3.5e-5, 0.6], False), ([5.0e-5, 1.5e-5, 0.3], True)],
)
def test_global_append_refuses_a_target_that_does_not_contain_the_psf_beam(
    monkeypatch, restoring_beam, refused
):
    """The first append node refuses the target right after the PSF fit, before
    any imaging cycle; the step after the fit is a sentinel here."""
    node = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    processing = importlib.import_module(
        "astroviper.processing_functions.imaging.image_continuum_single_field"
    )
    polarization = importlib.import_module(
        "astroviper.processing_functions.image_analysis.transform_polarization_basis"
    )
    fitted = []

    def fit(img_xds, image_data_group_out_name, **kwargs):
        img_xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] = xr.DataArray(
            np.broadcast_to([3.0e-5, 2.0e-5, 0.3], (2, 1, 2, 3)),
            dims=("time", "frequency", "polarization", "beam_params"),
        )
        img_xds["MAX_SIDELOBE_POINT_SPREAD_FUNCTION"] = xr.DataArray(
            np.zeros((2, 1, 2)), dims=("time", "frequency", "polarization")
        )
        img_xds.attrs["data_groups"][image_data_group_out_name].update(
            beam_fit_params_point_spread_function="BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
            max_sidelobe_point_spread_function="MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
        )
        fitted.append(True)
        return img_xds, None

    def after_the_fit(*args, **kwargs):
        raise StopIteration

    monkeypatch.setattr(processing, "point_spread_function_gaussian_fit_continuum", fit)
    monkeypatch.setattr(polarization, "transform_polarization_basis", after_the_fit)
    raw = _global_mvc_contributions()
    raw.attrs["data_groups"] = {"residual": {}}
    with pytest.raises(
        ValueError if refused else StopIteration,
        match="does not contain" if refused else None,
    ):
        node._prepare_continuum_image(
            raw,
            {
                "continuum_mode": "mvc",
                "image_params": {"n_taylor_terms": 2, "reference_frequency": 100.0},
                "restoring_beam": restoring_beam,
            },
            initialize_static_products=True,
        )
    assert fitted == [True]
