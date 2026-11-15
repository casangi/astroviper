"""Streaming MFS beam moments preserve the batched beam and Taylor equations."""

import weakref

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _convert_primary_beam_to_reference_frequency,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("beam_model", ["airy", "casa_airy"])
@pytest.mark.parametrize("nterms", [1, 2, 3, 4])
def test_streamed_beam_moments_match_dense_channel_sum(
    monkeypatch, dtype, beam_model, nterms
):
    """Keep exact beam samples, independent weights, all times, and bounded storage."""
    import astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam as beam

    # Reordered and repeated frequencies must contribute in the supplied order.
    frequencies = np.asarray([1.3, 0.8, 1.1, 1.3, 0.95, 1.0, 1.2]) * 1e9
    reference = 1e9
    polarizations = ["XX", "XY", "YX", "YY"]
    image_size = [17, 13]
    image = xr.Dataset(
        coords={
            "time": [0, 1],
            "frequency": frequencies,
            "polarization": polarizations,
            "l": np.arange(image_size[0]),
            "m": np.arange(image_size[1]),
        },
        attrs={"data_groups": {"residual": {}}},
    )
    params = {
        "image_size": image_size,
        "cell_size": [-2e-3, 2e-3],
        "image_center": [size // 2 for size in image_size],
        "n_taylor_terms": nterms,
        "primary_beam_model": beam_model,
        "list_dish_diameters": [10.7],
        "list_blockage_diameters": [0.75],
    }
    pb_params = {
        "list_dish_diameters": np.asarray([10.7]),
        "list_blockage_diameters": np.asarray([0.75]),
        "ipower": 2,
    }
    values = np.random.default_rng(42).uniform(0.2, 3, (len(frequencies), 4))
    values *= np.logspace(-2, 2, len(frequencies))[:, None]
    values[1, 0] = 0
    values[:, -1] = 0  # A completely flagged polarization still has zero moments.
    weights = xr.DataArray(
        values,
        dims=("frequency", "polarization"),
        coords={"frequency": frequencies, "polarization": polarizations},
    )
    evaluate = beam.evaluate_primary_beam
    dense_beam = evaluate(frequencies, polarizations, pb_params, params, dtype=dtype)
    reference_beam = evaluate(
        np.asarray([reference]), polarizations, pb_params, params, dtype=dtype
    )
    x = (frequencies - reference) / reference
    expected_numerator = np.stack(
        [
            np.sum(
                dense_beam[0]
                * values[:, :, None, None]
                * x[:, None, None, None] ** order,
                axis=0,
            )
            for order in range(nterms)
        ]
    )
    expected_moments = np.stack(
        [
            np.sum(values * x[:, None] ** order, axis=0)
            for order in range(2 * nterms - 1)
        ]
    )
    calls = []
    previous_storage = None

    def evaluate_one_channel(frequency, *args, **kwargs):
        nonlocal previous_storage
        assert len(frequency) == 1
        # No previous channel's backing allocation may survive until the next
        # evaluation. Holding a list of singleton beams would still use cube memory.
        assert previous_storage is None or previous_storage() is None
        result = evaluate(frequency, *args, **kwargs)
        if len(calls) < len(frequencies):
            index = len(calls)
            assert frequency[0] == frequencies[index]
            assert kwargs["lookup_frequency_max"] == frequencies.max()
            np.testing.assert_array_equal(result, dense_beam[:, index : index + 1])
        else:
            assert frequency[0] == reference
            assert kwargs.get("lookup_frequency_max") is None
        storage = result
        while isinstance(storage.base, np.ndarray):
            storage = storage.base
        previous_storage = weakref.ref(storage)
        calls.append(frequency[0])
        return result

    monkeypatch.setattr(beam, "evaluate_primary_beam", evaluate_one_channel)
    result = _convert_primary_beam_to_reference_frequency(
        image,
        image_params=params,
        reference_frequency=reference,
        n_taylor_terms=nterms,
        float_dtype=dtype,
        channel_imaging_weight_sums=weights,
    )
    assert len(calls) == len(frequencies) + 1
    numerator = result.MFS_PRIMARY_BEAM_TAYLOR_NUMERATOR
    moments = result.MFS_PRIMARY_BEAM_FIT_MOMENTS
    assert numerator.dtype == moments.dtype == np.float64
    for time in range(image.sizes["time"]):
        np.testing.assert_allclose(
            numerator.isel(time=time), expected_numerator, rtol=2e-14, atol=1e-13
        )
        np.testing.assert_array_equal(moments.isel(time=time), expected_moments)
        np.testing.assert_array_equal(
            result.PRIMARY_BEAM_REFERENCE.isel(time=time), reference_beam[0, 0]
        )
    assert result.PRIMARY_BEAM_REFERENCE.dtype == dtype
