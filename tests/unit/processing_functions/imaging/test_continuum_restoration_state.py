"""Final-only MFS beam fits and globally reduced MVC support keep their equations."""

import importlib

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    _fit_mfs_primary_beam_for_restoration,
    finalize_mvc_taylor_normal_equations,
    fit_mfs_primary_beam_taylor,
    make_mvc_taylor_normal_equation_contributions,
)
from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _combine_mfs_primary_beam_weight_summaries,
    _convert_primary_beam_to_reference_frequency,
)
from tests.unit.processing_functions.imaging.test_continuum_contracts import _mvc_inputs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("beam_model", ["airy", "casa_airy"])
@pytest.mark.parametrize("nterms", [2, 3, 4])
def test_deferred_mfs_fit_matches_original_partition_local_projection(
    dtype, beam_model, nterms
):
    # The chunks have different upper frequencies. Preserve their original
    # lookup bounds, weights, and correlation order when fitting at the end.
    frequency = np.asarray([0.8, 0.95, 1.1, 1.2, 1.3, 1.5, 1.7]) * 1e9
    params = {
        "image_size": [11, 9],
        "cell_size": [-2e-3, 2e-3],
        "n_taylor_terms": nterms,
        "reference_frequency": 1e9,
        "primary_beam_model": beam_model,
    }
    values = np.random.default_rng(42).uniform(0.2, 5, (len(frequency), 2))
    images, summaries = [], []
    for indices in ([0, 2, 4], [1, 3, 5, 6]):
        local_frequency = frequency[indices]
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
        summary = weights.to_dataset()
        summary["MFS_BEAM_LOOKUP_FREQUENCY_MAX"] = weights.frequency.copy(
            data=np.full(len(indices), local_frequency.max())
        )
        summaries.append(summary)
        images.append(
            _convert_primary_beam_to_reference_frequency(
                image,
                image_params=params,
                reference_frequency_hz=1e9,
                float_dtype=dtype,
                channel_imaging_weight_sums=weights,
            )
        )
    names = ["MFS_PRIMARY_BEAM_TAYLOR_NUMERATOR", "MFS_PRIMARY_BEAM_FIT_MOMENTS"]
    expected = fit_mfs_primary_beam_taylor(images[0][names] + images[1][names])
    expected = expected.assign_coords(polarization=["I", "Q"])
    image = xr.Dataset(
        {
            "PRIMARY_BEAM": images[0].PRIMARY_BEAM_REFERENCE.assign_coords(
                polarization=["I", "Q"]
            )
        }
    )
    actual = _fit_mfs_primary_beam_for_restoration(
        image, _combine_mfs_primary_beam_weight_summaries(summaries), params, 0.2
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-12, equal_nan=True)


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


def test_beam_weight_summary_tree_reduction_preserves_ownership_and_lookup_bounds():
    chunks = []
    for frequencies in ([1.0, 1.4], [1.1, 1.5], [1.2, 1.6]):
        chunks.append(
            xr.Dataset(
                {
                    "MFS_CHANNEL_IMAGING_WEIGHT_SUM": (
                        ("frequency", "polarization"),
                        np.ones((2, 2)),
                    ),
                    "MFS_BEAM_LOOKUP_FREQUENCY_MAX": (
                        "frequency",
                        [max(frequencies)] * 2,
                    ),
                },
                coords={"frequency": frequencies, "polarization": ["XX", "YY"]},
            )
        )
    flat = _combine_mfs_primary_beam_weight_summaries(chunks)
    tree = _combine_mfs_primary_beam_weight_summaries(
        [_combine_mfs_primary_beam_weight_summaries(chunks[:2]), chunks[2]]
    )
    xr.testing.assert_identical(tree, flat)
    np.testing.assert_array_equal(
        tree.MFS_BEAM_LOOKUP_FREQUENCY_MAX, [1.4, 1.5, 1.6, 1.4, 1.5, 1.6]
    )
    with pytest.raises(ValueError, match="duplicate frequency ownership"):
        _combine_mfs_primary_beam_weight_summaries([chunks[0], chunks[0]])


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
