"""Unit tests for the ``image_chunking`` / ``image_sharding`` validation the
``image_cube_single_field`` distributed application runs before creating the
image store (no imaging run needed -- the helper is exercised directly), plus
the ``n_mapping_parallelism`` validation."""

import pytest

from astroviper.utils.io import validate_image_chunking_and_sharding

SHAPE = {"time": 1, "frequency": 8, "polarization": 1, "l": 16, "m": 16}


def _parallel_coords(chunk_lengths):
    """Frequency parallel coords with the given per-task chunk lengths."""
    chunks, start = {}, 0
    for i, length in enumerate(chunk_lengths):
        chunks[i] = list(range(start, start + length))
        start += length
    return {"frequency": {"data_chunks": chunks}}


def test_valid_chunking_passes():
    validate_image_chunking_and_sharding(
        {"l": 4, "m": 4, "frequency": 1}, None, SHAPE, _parallel_coords([2, 2, 2, 2])
    )


def test_uv_dims_are_valid_keys():
    validate_image_chunking_and_sharding(
        {"u": 4, "v": 4}, {"u": 8}, SHAPE, _parallel_coords([2, 2, 2, 2])
    )


def test_none_and_empty_pass():
    validate_image_chunking_and_sharding(None, None, SHAPE, _parallel_coords([2, 2]))
    validate_image_chunking_and_sharding({}, {}, SHAPE, _parallel_coords([2, 2]))


@pytest.mark.parametrize("label", ["image_chunking", "image_sharding"])
def test_unknown_dimension_rejected(label):
    kwargs = {"image_chunking": None, "image_sharding": None, label: {"chan": 1}}
    with pytest.raises(
        ValueError, match=f"{label} key 'chan' is not an image dimension"
    ):
        validate_image_chunking_and_sharding(
            kwargs["image_chunking"],
            kwargs["image_sharding"],
            SHAPE,
            _parallel_coords([2, 2, 2, 2]),
        )


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4"])
def test_non_positive_or_non_int_size_rejected(bad):
    with pytest.raises(ValueError, match="positive int"):
        validate_image_chunking_and_sharding(
            {"l": bad}, None, SHAPE, _parallel_coords([2, 2, 2, 2])
        )
    with pytest.raises(ValueError, match="positive int"):
        validate_image_chunking_and_sharding(
            None, {"l": bad}, SHAPE, _parallel_coords([2, 2, 2, 2])
        )


def test_frequency_chunk_must_divide_task_chunks():
    # Task chunks of 4 channels; a 3-channel on-disk chunk would straddle tasks.
    with pytest.raises(ValueError, match="must divide"):
        validate_image_chunking_and_sharding(
            {"frequency": 3}, None, SHAPE, _parallel_coords([4, 4])
        )


def test_chunk_larger_than_task_extent_is_rejected():
    # Previously clipped silently; now an explicit error.
    with pytest.raises(ValueError, match=r"image_chunking\['frequency'\]=5 exceeds"):
        validate_image_chunking_and_sharding(
            {"frequency": 5}, None, SHAPE, _parallel_coords([4, 4])
        )
    with pytest.raises(ValueError, match=r"image_chunking\['l'\]=17 exceeds"):
        validate_image_chunking_and_sharding(
            {"l": 17}, None, SHAPE, _parallel_coords([4, 4])
        )


def test_partial_last_task_chunk_is_allowed():
    # 8 channels in tasks of [3, 3, 2]: frequency=3 divides every task chunk
    # except the last (partial, at the array edge) -- allowed.
    validate_image_chunking_and_sharding(
        {"frequency": 3}, None, SHAPE, _parallel_coords([3, 3, 2])
    )


def test_shards_may_span_tasks_and_must_hold_whole_chunks():
    # 1-channel tasks, 200-channel shards (the Frontera layout): fine.
    validate_image_chunking_and_sharding(
        None, {"frequency": 200}, SHAPE, _parallel_coords([1] * 8)
    )
    # Shard of 3 channels over 2-channel chunks: not a multiple.
    with pytest.raises(
        ValueError, match=r"image_sharding\['frequency'\]=3 must be a multiple"
    ):
        validate_image_chunking_and_sharding(
            None, {"frequency": 3}, SHAPE, _parallel_coords([2, 2, 2, 2])
        )
    # Sharding l needs an l chunk that divides the shard (default chunk = 16).
    with pytest.raises(ValueError, match=r"image_sharding\['l'\]=8 must be a multiple"):
        validate_image_chunking_and_sharding(
            None, {"l": 8}, SHAPE, _parallel_coords([2, 2, 2, 2])
        )
    validate_image_chunking_and_sharding(
        {"l": 4}, {"l": 8, "frequency": 4}, SHAPE, _parallel_coords([2, 2, 2, 2])
    )


def test_list_data_chunks_are_accepted():
    # Test helpers pass data_chunks as a list rather than graphviper's dict.
    validate_image_chunking_and_sharding(
        {"frequency": 1},
        {"frequency": 4},
        SHAPE,
        {"frequency": {"data_chunks": [[0, 1], [2, 3]]}},
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
