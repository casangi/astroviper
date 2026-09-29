"""Unit tests for :mod:`astroviper.utils.io` helpers."""

from astroviper.utils.io import (
    image_data_groups_for_kept_variables,
    imaging_data_variable_data_group_roles,
    imaging_data_variables_and_dims_double_precision,
    imaging_data_variables_and_dims_single_precision,
)


class TestImageDataGroupsForKeptVariables:
    def test_standard_clean_keep_list(self):
        """The notebook/driver CLEAN keep list produces the documented groups."""
        keep = [
            "sky_residual",
            "sky_model",
            "mask",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
        ]
        groups = image_data_groups_for_kept_variables(keep)
        assert groups == {
            "residual": {
                "sky": "SKY_RESIDUAL",
                "mask": "MASK",
                "point_spread_function": "POINT_SPREAD_FUNCTION",
                "primary_beam": "PRIMARY_BEAM",
                "beam_fit_params_point_spread_function": (
                    "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
                ),
            },
            "model": {"sky": "SKY_MODEL"},
        }

    def test_restored_goes_to_its_own_group(self):
        groups = image_data_groups_for_kept_variables(["sky_restored"])
        assert groups == {"restored": {"sky": "SKY_RESTORED"}}

    def test_variables_without_membership_are_skipped(self):
        # sky_dirty has no data-group membership; only sky_residual registers.
        groups = image_data_groups_for_kept_variables(["sky_dirty", "sky_residual"])
        assert groups == {"residual": {"sky": "SKY_RESIDUAL"}}

    def test_empty_keep_list(self):
        assert image_data_groups_for_kept_variables([]) == {}

    def test_membership_keys_exist_in_both_registries(self):
        """Every group-role key must be a real variable in both precision maps."""
        for key in imaging_data_variable_data_group_roles:
            assert key in imaging_data_variables_and_dims_double_precision
            assert key in imaging_data_variables_and_dims_single_precision
            # Variable names must agree across precisions (groups store names).
            assert (
                imaging_data_variables_and_dims_double_precision[key]["name"]
                == imaging_data_variables_and_dims_single_precision[key]["name"]
            )


class TestCreateEmptyDataVariablesImageChunkingSharding:
    """On-disk chunk and shard shapes produced by ``image_chunking`` and
    ``image_sharding`` (4 tasks x 2 channels on an 8-channel, 16x16 image)."""

    @staticmethod
    def _make_store(tmp_path, image_chunking=None, image_sharding=None):
        import zarr

        from astroviper.utils.io import create_empty_data_variables_on_disk

        store = str(tmp_path / "img.zarr")
        zarr.open_group(store, mode="w")
        shape_dict = {
            "time": 1,
            "frequency": 8,
            "polarization": 2,
            "l": 16,
            "m": 16,
        }
        freq_chunks = [list(range(2)) for _ in range(4)]  # 4 tasks x 2 channels
        create_empty_data_variables_on_disk(
            store,
            ["sky_residual"],
            shape_dict=shape_dict,
            parallel_coords={"frequency": {"data_chunks": freq_chunks}},
            compressor=None,
            double_precision=False,
            data_variable_definitions="imaging",
            image_chunking=image_chunking,
            image_sharding=image_sharding,
        )
        return store

    def test_default_chunks_span_full_lm(self, tmp_path):
        import zarr

        store = self._make_store(tmp_path)
        arr = zarr.open_array(store + "/SKY_RESIDUAL")
        assert arr.chunks == (1, 2, 2, 16, 16)

    def test_chunking_subdivides_lm_and_frequency(self, tmp_path):
        import zarr

        store = self._make_store(
            tmp_path, image_chunking={"l": 8, "m": 4, "frequency": 1}
        )
        arr = zarr.open_array(store + "/SKY_RESIDUAL")
        assert arr.chunks == (1, 1, 2, 8, 4)

    def test_chunking_larger_than_task_extent_raises(self, tmp_path):
        import pytest

        # Larger than the axis / per-task chunk -> rejected, never clipped.
        with pytest.raises(ValueError, match=r"image_chunking\['l'\]=999 exceeds"):
            self._make_store(tmp_path, image_chunking={"l": 999})
        with pytest.raises(ValueError, match=r"image_chunking\['frequency'\]=999"):
            self._make_store(tmp_path, image_chunking={"frequency": 999})

    def test_sharded_inner_chunking_keeps_one_shard_per_channel_block(self, tmp_path):
        import zarr

        store = self._make_store(
            tmp_path,
            image_chunking={"l": 8, "m": 8},
            image_sharding={"frequency": 4},
        )
        arr = zarr.open_array(store + "/SKY_RESIDUAL")
        # Shard: 4 channels per shard file, FULL l/m extent.
        assert arr.shards == (1, 4, 2, 16, 16)
        # Inner (read/write) chunk carries the l/m sub-chunking.
        assert arr.chunks == (1, 2, 2, 8, 8)

    def test_sharding_on_any_dimension(self, tmp_path):
        import zarr

        store = self._make_store(
            tmp_path,
            image_chunking={"l": 4, "m": 8},
            image_sharding={"frequency": 4, "l": 8, "m": 8},
        )
        arr = zarr.open_array(store + "/SKY_RESIDUAL")
        assert arr.shards == (1, 4, 2, 8, 8)
        assert arr.chunks == (1, 2, 2, 4, 8)

    def test_shard_larger_than_axis_is_clipped(self, tmp_path):
        import zarr

        store = self._make_store(tmp_path, image_sharding={"frequency": 200})
        arr = zarr.open_array(store + "/SKY_RESIDUAL")
        assert arr.shards == (1, 8, 2, 16, 16)
        assert arr.chunks == (1, 2, 2, 16, 16)

    def test_shard_not_multiple_of_chunk_raises(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="must be a multiple of the on-disk chunk"):
            self._make_store(tmp_path, image_sharding={"frequency": 3})
        # A shard on l needs an l chunk that divides it (default chunk = 16).
        with pytest.raises(ValueError, match=r"image_sharding\['l'\]=8"):
            self._make_store(tmp_path, image_sharding={"l": 8})
