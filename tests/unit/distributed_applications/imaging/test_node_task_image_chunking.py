"""Unit tests for the ``node_task_image_chunking`` validation of the
``image_cube_single_field`` distributed application (no imaging run needed --
the helper is exercised directly)."""

import numpy as np
import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_cube_single_field import (
    _validate_node_task_image_chunking,
)


def _img_xds(n_freq=8, n_lm=16):
    return xr.Dataset(
        coords={
            "time": ("time", [0.0]),
            "frequency": ("frequency", np.arange(n_freq, dtype=float)),
            "polarization": ("polarization", ["I"]),
            "l": ("l", np.arange(n_lm, dtype=float)),
            "m": ("m", np.arange(n_lm, dtype=float)),
        }
    )


def _parallel_coords(chunk_lengths):
    """Frequency parallel coords with the given per-task chunk lengths."""
    chunks, start = {}, 0
    for i, length in enumerate(chunk_lengths):
        chunks[i] = list(range(start, start + length))
        start += length
    return {"frequency": {"data_chunks": chunks}}


def test_valid_chunking_passes():
    _validate_node_task_image_chunking(
        {"l": 4, "m": 4, "frequency": 1}, _img_xds(), _parallel_coords([2, 2, 2, 2])
    )


def test_uv_dims_are_valid_keys():
    _validate_node_task_image_chunking(
        {"u": 4, "v": 4}, _img_xds(), _parallel_coords([2, 2, 2, 2])
    )


def test_unknown_dimension_rejected():
    with pytest.raises(ValueError, match="not an image dimension"):
        _validate_node_task_image_chunking(
            {"chan": 1}, _img_xds(), _parallel_coords([2, 2, 2, 2])
        )


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4"])
def test_non_positive_or_non_int_chunk_size_rejected(bad):
    with pytest.raises(ValueError, match="positive int"):
        _validate_node_task_image_chunking(
            {"l": bad}, _img_xds(), _parallel_coords([2, 2, 2, 2])
        )


def test_frequency_chunk_must_divide_task_chunks():
    # Task chunks of 4 channels; a 3-channel on-disk chunk would straddle tasks.
    with pytest.raises(ValueError, match="must divide"):
        _validate_node_task_image_chunking(
            {"frequency": 3}, _img_xds(), _parallel_coords([4, 4])
        )


def test_frequency_chunk_larger_than_task_chunk_is_clipped_and_passes():
    # Sizes >= the per-task chunk are clipped on creation (no subdivision), so
    # they must validate even when they do not divide the task chunk.
    _validate_node_task_image_chunking(
        {"frequency": 5}, _img_xds(), _parallel_coords([4, 4])
    )


def test_partial_last_task_chunk_is_allowed():
    # 8 channels in tasks of [3, 3, 2]: frequency=3 divides every task chunk
    # except the last (partial, at the array edge) -- allowed.
    _validate_node_task_image_chunking(
        {"frequency": 3}, _img_xds(), _parallel_coords([3, 3, 2])
    )


class TestValidateNMappingParallelism:
    """Cube imaging's n_mapping_parallelism dict: frequency-only, positive int
    or None count."""

    @staticmethod
    def _validate(value):
        from astroviper.distributed_applications.imaging.image_cube_single_field import (
            _validate_n_mapping_parallelism,
        )

        _validate_n_mapping_parallelism(value)

    def test_frequency_count_passes(self):
        self._validate({"frequency": 500})

    def test_frequency_none_count_passes(self):
        self._validate({"frequency": None})

    @pytest.mark.parametrize(
        "bad_keys",
        [{"l": 4}, {"frequency": 2, "l": 4}, {}],
        ids=["wrong-axis", "extra-axis", "empty"],
    )
    def test_non_frequency_keys_rejected(self, bad_keys):
        with pytest.raises(ValueError, match="single key\\s+'frequency'"):
            self._validate(bad_keys)

    @pytest.mark.parametrize("bad", [0, -3, 2.5, True, "5"])
    def test_bad_count_rejected(self, bad):
        with pytest.raises(ValueError, match="positive int"):
            self._validate({"frequency": bad})
