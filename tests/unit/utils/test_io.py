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


class TestCreateEmptyDataVariablesNodeTaskImageChunking:
    """On-disk chunk shapes produced by ``node_task_image_chunking``."""

    @staticmethod
    def _make_store(tmp_path, shard_channels=None, node_task_image_chunking=None):
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
            shard_channels=shard_channels,
            node_task_image_chunking=node_task_image_chunking,
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
            tmp_path, node_task_image_chunking={"l": 8, "m": 4, "frequency": 1}
        )
        arr = zarr.open_array(store + "/SKY_RESIDUAL")
        assert arr.chunks == (1, 1, 2, 8, 4)

    def test_chunking_is_clipped_to_defaults(self, tmp_path):
        import zarr

        # Larger than the axis / per-task chunk -> clipped, never enlarged.
        store = self._make_store(
            tmp_path, node_task_image_chunking={"l": 999, "frequency": 999}
        )
        arr = zarr.open_array(store + "/SKY_RESIDUAL")
        assert arr.chunks == (1, 2, 2, 16, 16)

    def test_sharded_inner_chunking_keeps_one_shard_per_channel_block(self, tmp_path):
        import zarr

        store = self._make_store(
            tmp_path,
            shard_channels=4,
            node_task_image_chunking={"l": 8, "m": 8},
        )
        arr = zarr.open_array(store + "/SKY_RESIDUAL")
        # Shard: 4 channels per shard file, FULL l/m extent.
        assert arr.shards == (1, 4, 2, 16, 16)
        # Inner (read/write) chunk carries the l/m sub-chunking.
        assert arr.chunks == (1, 2, 2, 8, 8)
