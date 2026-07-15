"""Unit tests for :mod:`astroviper.utils.io` helpers."""

import pytest

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
