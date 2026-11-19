"""
Unit tests for the cycle control module.

Tests cover:
- imaging_dict core functionality (initialization, add_to_imaging_dict,
  select_from_imaging_dict)
- imaging_dict utility functions (merge, extraction)
- Cycle control state initialization and parameter validation
- Adaptive threshold_per_cycle calculation
- Convergence checking with all stop codes
- Count updates and state management
- Threshold string parsing
- Full imaging cycle workflows
"""

import unittest

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.image_statistics import (
    RESIDUAL_STATISTICS,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (  # Stop codes; Utility functions; Cycle control functions
    IMAGING_CONTINUE,
    IMAGING_DIVERGENCE,
    IMAGING_MAX_CYCLES,
    IMAGING_MAX_ITER,
    IMAGING_THRESHOLD,
    IMAGING_ZERO_MASK,
    MODEL_UPDATE_CONTINUE,
    MODEL_UPDATE_DIVERGENCE,
    MODEL_UPDATE_MAX_ITER_PER_CYCLE,
    MODEL_UPDATE_THRESHOLD_PER_CYCLE,
    StopCode,
    build_residual_imaging_dict,
    calculate_per_cycle_limits,
    check_convergence,
    check_refreshed_residual_divergence,
    ensure_planes,
    get_iterations_done_from_imaging_dict,
    get_masksum_from_imaging_dict,
    get_max_psf_sidelobe_from_imaging_dict,
    get_model_flux_from_imaging_dict,
    get_peak_residual_from_imaging_dict,
    get_state,
    make_cycle_control_state,
    merge_imaging_dicts,
    per_plane_threshold_per_cycle,
    plot_convergence_history,
    record_model_update,
    reset,
    reset_stopcode,
    update_counts,
    update_parameters,
)
from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
    THRESHOLD_TOLERANCE,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    COMPONENT_COUNT_FIELDS,
    FIELD_ACCUM,
    FIELD_SINGLE_VALUE,
    ITERATION_HISTORY_FIELDS,
    Key,
    add_to_imaging_dict,
    select_from_imaging_dict,
)

# =============================================================================
# imaging_dict Core Functionality Tests
# =============================================================================


class TestImagingDictBasics(unittest.TestCase):
    """Test imaging_dict initialization and layout."""

    def test_empty_initialization(self):
        """Test creating an empty imaging_dict."""
        rd = {}
        self.assertIsNotNone(rd)
        self.assertEqual(len(rd), 0)
        self.assertIsInstance(rd, dict)

    def test_data_property_getter(self):
        """Test the imaging_dict is a dictionary keyed by Key."""
        rd = {}
        add_to_imaging_dict(rd, {"peakres": 1.0}, time=0, pol=0, chan=0)

        data = rd
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 1)

        # Verify the key structure
        key = Key(0, 0, 0)
        self.assertIn(key, data)


class TestImagingDictAdd(unittest.TestCase):
    """Test add_to_imaging_dict() edge cases and history tracking."""

    def test_add_first_entry_field_accum(self):
        """Test adding first entry with FIELD_ACCUM fields."""
        rd = {}
        add_to_imaging_dict(
            rd, {"peakres": 1.5, "iter_done": 100}, time=0, pol=0, chan=0
        )

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # FIELD_ACCUM fields should be stored as lists
        self.assertEqual(entry["peakres"], [1.5])
        self.assertEqual(entry["iter_done"], [100])
        self.assertIsInstance(entry["peakres"], list)
        self.assertIsInstance(entry["iter_done"], list)

    def test_add_first_entry_field_single_value(self):
        """Test adding first entry with FIELD_SINGLE_VALUE fields."""
        rd = {}
        add_to_imaging_dict(
            rd, {"max_psf_sidelobe": 0.2, "gain": 0.1}, time=0, pol=0, chan=0
        )

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # FIELD_SINGLE_VALUE fields should be stored as scalars
        self.assertEqual(entry["max_psf_sidelobe"], 0.2)
        self.assertEqual(entry["gain"], 0.1)
        self.assertNotIsInstance(entry["max_psf_sidelobe"], list)
        self.assertNotIsInstance(entry["gain"], list)

    def test_add_first_entry_mixed_fields(self):
        """Test adding first entry with both FIELD_ACCUM and FIELD_SINGLE_VALUE."""
        rd = {}
        add_to_imaging_dict(
            rd,
            {
                "peakres": 1.0,  # FIELD_ACCUM
                "iter_done": 50,  # FIELD_ACCUM
                "max_psf_sidelobe": 0.15,  # FIELD_SINGLE_VALUE
                "threshold": 0.01,  # FIELD_SINGLE_VALUE
            },
            time=0,
            pol=0,
            chan=0,
        )

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # Check FIELD_ACCUM are lists
        self.assertEqual(entry["peakres"], [1.0])
        self.assertEqual(entry["iter_done"], [50])

        # Check FIELD_SINGLE_VALUE are scalars
        self.assertEqual(entry["max_psf_sidelobe"], 0.15)
        self.assertEqual(entry["threshold"], 0.01)

    def test_add_unknown_field(self):
        """Test adding a field not in FIELD_ACCUM or FIELD_SINGLE_VALUE."""
        rd = {}
        add_to_imaging_dict(
            rd, {"custom_field": 42, "peakres": 1.0}, time=0, pol=0, chan=0
        )

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # Unknown fields should be treated as FIELD_SINGLE_VALUE (not accumulated)
        self.assertEqual(entry["custom_field"], 42)
        self.assertNotIsInstance(entry["custom_field"], list)

        # Known FIELD_ACCUM should still be a list
        self.assertEqual(entry["peakres"], [1.0])

    def test_add_empty_dict(self):
        """Test adding an empty dictionary."""
        rd = {}
        add_to_imaging_dict(rd, {}, time=0, pol=0, chan=0)

        # Key should exist but with empty entry
        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)
        self.assertIsNotNone(entry)
        self.assertEqual(entry, {})

    def test_backward_compat_scalar_to_list(self):
        """Test backward compatibility: scalar converted to list on second add."""
        rd = {}

        # Manually create a scalar entry (simulating old data format)
        key = Key(0, 0, 0)
        rd[key] = {"peakres": 1.0}  # Scalar instead of list

        # Now add new value - should convert to list and append
        add_to_imaging_dict(rd, {"peakres": 0.8}, time=0, pol=0, chan=0)

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # Should have converted scalar to list and appended new value
        self.assertEqual(entry["peakres"], [1.0, 0.8])
        self.assertIsInstance(entry["peakres"], list)

    def test_multiple_adds_same_key_accumulate(self):
        """Test multiple adds to same key accumulate FIELD_ACCUM values."""
        rd = {}

        # First add
        add_to_imaging_dict(
            rd, {"peakres": 1.0, "iter_done": 100}, time=0, pol=0, chan=0
        )

        # Second add
        add_to_imaging_dict(
            rd, {"peakres": 0.8, "iter_done": 80}, time=0, pol=0, chan=0
        )

        # Third add
        add_to_imaging_dict(
            rd, {"peakres": 0.6, "iter_done": 60}, time=0, pol=0, chan=0
        )

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # FIELD_ACCUM should accumulate in lists
        self.assertEqual(entry["peakres"], [1.0, 0.8, 0.6])
        self.assertEqual(entry["iter_done"], [100, 80, 60])

    def test_multiple_adds_same_key_replace_single_value(self):
        """Test multiple adds to same key replace FIELD_SINGLE_VALUE values."""
        rd = {}

        # First add
        add_to_imaging_dict(
            rd, {"max_psf_sidelobe": 0.2, "threshold": 0.01}, time=0, pol=0, chan=0
        )

        # Second add with different values
        add_to_imaging_dict(
            rd, {"max_psf_sidelobe": 0.15, "threshold": 0.005}, time=0, pol=0, chan=0
        )

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # FIELD_SINGLE_VALUE should be replaced (last value wins)
        self.assertEqual(entry["max_psf_sidelobe"], 0.15)
        self.assertEqual(entry["threshold"], 0.005)

    def test_add_partial_fields_to_existing_key(self):
        """Test adding partial fields to an existing key."""
        rd = {}

        # First add with some fields
        add_to_imaging_dict(
            rd, {"peakres": 1.0, "iter_done": 100}, time=0, pol=0, chan=0
        )

        # Second add with different fields
        add_to_imaging_dict(
            rd, {"masksum": 500, "max_psf_sidelobe": 0.2}, time=0, pol=0, chan=0
        )

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # First fields should still exist
        self.assertEqual(entry["peakres"], [1.0])
        self.assertEqual(entry["iter_done"], [100])

        # New fields should be added
        self.assertEqual(entry["masksum"], [500])  # FIELD_ACCUM
        self.assertEqual(entry["max_psf_sidelobe"], 0.2)  # FIELD_SINGLE_VALUE


class TestImagingDictSel(unittest.TestCase):
    """Test select_from_imaging_dict() edge cases and filtering."""

    def setUp(self):
        """Create test imaging_dict with multiple entries."""
        self.rd = {}
        add_to_imaging_dict(self.rd, {"peakres": 1.0}, time=0, pol=0, chan=0)
        add_to_imaging_dict(self.rd, {"peakres": 0.9}, time=0, pol=0, chan=1)
        add_to_imaging_dict(self.rd, {"peakres": 0.8}, time=0, pol=1, chan=0)
        add_to_imaging_dict(self.rd, {"peakres": 0.7}, time=1, pol=0, chan=0)

    def test_sel_no_matches_returns_none(self):
        """Test select_from_imaging_dict() returns None when no matches found."""
        result = select_from_imaging_dict(self.rd, time=99, pol=0, chan=0)
        self.assertIsNone(result)

        result = select_from_imaging_dict(self.rd, time=0, pol=99, chan=0)
        self.assertIsNone(result)

        result = select_from_imaging_dict(self.rd, time=0, pol=0, chan=99)
        self.assertIsNone(result)

    def test_sel_single_match_returns_dict(self):
        """Test select_from_imaging_dict() returns single dict (not list) when one
        match."""
        result = select_from_imaging_dict(self.rd, time=0, pol=0, chan=0)

        # Should return a dict, not a list
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, list)
        self.assertEqual(result["peakres"], [1.0])

    def test_sel_multiple_matches_returns_list(self):
        """Test select_from_imaging_dict() returns list when multiple matches."""
        # Filter by time=0 only (should match 3 entries)
        result = select_from_imaging_dict(self.rd, time=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

        # Verify all are dicts
        for entry in result:
            self.assertIsInstance(entry, dict)
            self.assertIn("peakres", entry)

    def test_sel_all_none_returns_all(self):
        """Test select_from_imaging_dict() with all None filters returns all entries
        (or empty list)."""
        result = select_from_imaging_dict(self.rd)

        # With all None, it should return a list of all entries
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)

    def test_sel_empty_returndict(self):
        """Test select_from_imaging_dict() on empty imaging_dict returns None."""
        rd_empty = {}
        result = select_from_imaging_dict(rd_empty, time=0, pol=0, chan=0)

        self.assertIsNone(result)

    def test_sel_filter_by_time_only(self):
        """Test filtering by time parameter only."""
        result = select_from_imaging_dict(self.rd, time=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 3 entries with time=0

    def test_sel_filter_by_pol_only(self):
        """Test filtering by pol parameter only."""
        result = select_from_imaging_dict(self.rd, pol=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 3 entries with pol=0

    def test_sel_filter_by_chan_only(self):
        """Test filtering by chan parameter only."""
        result = select_from_imaging_dict(self.rd, chan=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 3 entries with chan=0

    def test_sel_combined_filters(self):
        """Test filtering with multiple parameters."""
        result = select_from_imaging_dict(self.rd, time=0, pol=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # time=0, pol=0 matches 2 entries

    def test_sel_preserves_order(self):
        """Test select_from_imaging_dict() preserves insertion order."""
        rd = {}

        # Add in specific order
        add_to_imaging_dict(rd, {"peakres": 1.0}, time=0, pol=0, chan=0)
        add_to_imaging_dict(rd, {"peakres": 0.9}, time=0, pol=0, chan=1)
        add_to_imaging_dict(rd, {"peakres": 0.8}, time=0, pol=0, chan=2)

        result = select_from_imaging_dict(rd, time=0, pol=0)

        # Should return in insertion order
        self.assertEqual(result[0]["peakres"], [1.0])
        self.assertEqual(result[1]["peakres"], [0.9])
        self.assertEqual(result[2]["peakres"], [0.8])

    def test_sel_nonexistent_values(self):
        """Test select_from_imaging_dict() with values that don't exist in any
        entry."""
        result = select_from_imaging_dict(self.rd, time=999, pol=999, chan=999)
        self.assertIsNone(result)


class TestImagingDictFieldClassification(unittest.TestCase):
    """Test FIELD_ACCUM and FIELD_SINGLE_VALUE classification behavior."""

    def test_field_accum_constants(self):
        """Test FIELD_ACCUM contains expected fields."""
        expected_fields = {
            "peakres",
            "peakres_nomask",
            "iter_done",
            "masksum",
            "model_flux",
            "start_peakres",
            "start_peakres_nomask",
            "start_model_flux",
            "entropy",
            "residual_snr",
            "model_update_stop_code",
            "threshold_per_cycle",
            *RESIDUAL_STATISTICS,
            *(f"start_{field}" for field in RESIDUAL_STATISTICS),
            *ITERATION_HISTORY_FIELDS,
            *COMPONENT_COUNT_FIELDS,
            "model_flux_positive",
            "model_flux_negative",
            "t_residual_update",
            "t_model_update",
        }
        self.assertEqual(FIELD_ACCUM, expected_fields)

    def test_field_single_value_constants(self):
        """Test FIELD_SINGLE_VALUE contains expected fields."""
        expected_fields = {
            "max_psf_sidelobe",
            "gain",
            "max_iter",
            "threshold",
            "max_cycles_auto",
        }
        self.assertEqual(FIELD_SINGLE_VALUE, expected_fields)

    def test_all_field_accum_become_lists(self):
        """Test all FIELD_ACCUM fields are stored as lists."""
        rd = {}

        # Add all FIELD_ACCUM fields
        add_to_imaging_dict(
            rd,
            {
                "peakres": 1.0,
                "peakres_nomask": 1.1,
                "iter_done": 100,
                "masksum": 500,
                "model_flux": 2.5,
                "start_peakres": 1.0,
                "start_peakres_nomask": 1.1,
                "start_model_flux": 0.0,
                "entropy": 7.5,
                "residual_snr": 3.0,
                "model_update_stop_code": 2,
                **{field: 1.0 for field in FIELD_ACCUM},
                "iteration_peak_residual": np.array([1.0, 0.9]),
                "iteration_residual_rms": np.array([0.2, 0.19]),
            },
            time=0,
            pol=0,
            chan=0,
        )

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # All should be lists
        for field in FIELD_ACCUM:
            self.assertIn(field, entry)
            self.assertIsInstance(entry[field], list)
            self.assertEqual(len(entry[field]), 1)

    def test_all_field_single_value_remain_scalars(self):
        """Test all FIELD_SINGLE_VALUE fields remain scalars."""
        rd = {}

        # Add all FIELD_SINGLE_VALUE fields
        add_to_imaging_dict(
            rd,
            {
                "max_psf_sidelobe": 0.2,
                "gain": 0.1,
                "max_iter": 1000,
                "threshold": 0.01,
                "max_cycles_auto": 7,
            },
            time=0,
            pol=0,
            chan=0,
        )

        entry = select_from_imaging_dict(rd, time=0, pol=0, chan=0)

        # All should be scalars (not lists)
        for field in FIELD_SINGLE_VALUE:
            self.assertIn(field, entry)
            self.assertNotIsInstance(entry[field], list)


class TestImagingDictKeyBehavior(unittest.TestCase):
    """Test Key namedtuple behavior and insertion order."""

    def test_key_namedtuple_creation(self):
        """Test Key namedtuple can be created and accessed."""
        key = Key(time=0, pol=1, chan=2)

        self.assertEqual(key.time, 0)
        self.assertEqual(key.pol, 1)
        self.assertEqual(key.chan, 2)

    def test_key_namedtuple_immutability(self):
        """Test Key namedtuple is immutable."""
        key = Key(0, 1, 2)

        with self.assertRaises(AttributeError):
            key.time = 99

    def test_key_equality(self):
        """Test Key namedtuples with same values are equal."""
        key1 = Key(0, 1, 2)
        key2 = Key(0, 1, 2)
        key3 = Key(0, 1, 3)

        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_key_hashable(self):
        """Test Key can be used as dictionary key (hashable)."""
        test_dict = {}
        key = Key(0, 1, 2)

        test_dict[key] = "value"
        self.assertEqual(test_dict[key], "value")

    def test_ordered_dict_preserves_insertion_order(self):
        """Test imaging_dict preserves insertion order."""
        rd = {}

        # Add keys in specific order
        add_to_imaging_dict(rd, {"peakres": 1.0}, time=2, pol=0, chan=0)
        add_to_imaging_dict(rd, {"peakres": 0.9}, time=0, pol=0, chan=0)
        add_to_imaging_dict(rd, {"peakres": 0.8}, time=1, pol=0, chan=0)

        # Verify order is preserved
        keys = list(rd.keys())
        self.assertEqual(keys[0].time, 2)
        self.assertEqual(keys[1].time, 0)
        self.assertEqual(keys[2].time, 1)


# =============================================================================
# imaging_dict Utility Function Tests (Merge, Extraction)
# =============================================================================


class TestMergeImagingDicts(unittest.TestCase):
    """Test merging multiple imaging_dict objects."""

    def test_invalid_strategy_rejected_before_inspecting_records(self):
        left = {}
        add_to_imaging_dict(left, {"peakres": 1.0}, time=0, pol=0, chan=0)
        right = {}
        add_to_imaging_dict(right, {"peakres": 0.5}, time=0, pol=0, chan=1)
        for records in ([], [left], [left, right]):
            with self.subTest(records=len(records)):
                with self.assertRaisesRegex(ValueError, "Unknown merge_strategy"):
                    merge_imaging_dicts(records, merge_strategy="invalid")
        self.assertEqual(
            select_from_imaging_dict(left, time=0, pol=0, chan=0)["peakres"], [1.0]
        )

    def setUp(self):
        """Create test imaging_dict objects."""
        self.rd1 = {}
        add_to_imaging_dict(
            self.rd1, {"peakres": 0.5, "iter_done": 100}, time=0, pol=0, chan=0
        )
        add_to_imaging_dict(
            self.rd1, {"peakres": 0.4, "iter_done": 90}, time=0, pol=0, chan=1
        )

        self.rd2 = {}
        add_to_imaging_dict(
            self.rd2, {"peakres": 0.3, "iter_done": 120}, time=0, pol=0, chan=2
        )
        add_to_imaging_dict(
            self.rd2, {"peakres": 0.6, "iter_done": 80}, time=0, pol=0, chan=3
        )

        self.rd3 = {}
        add_to_imaging_dict(
            self.rd3, {"peakres": 0.2, "iter_done": 110}, time=0, pol=1, chan=0
        )

    def test_merge_non_overlapping_keys(self):
        """Test merging dicts with no overlapping keys."""
        merged = merge_imaging_dicts([self.rd1, self.rd2, self.rd3])
        self.assertEqual(len(merged), 5)

        # Check all entries present
        self.assertIsNotNone(select_from_imaging_dict(merged, time=0, pol=0, chan=0))
        self.assertIsNotNone(select_from_imaging_dict(merged, time=0, pol=0, chan=1))
        self.assertIsNotNone(select_from_imaging_dict(merged, time=0, pol=0, chan=2))
        self.assertIsNotNone(select_from_imaging_dict(merged, time=0, pol=0, chan=3))
        self.assertIsNotNone(select_from_imaging_dict(merged, time=0, pol=1, chan=0))

    def test_merge_strategy_latest(self):
        """Test 'latest' merge strategy overwrites with last value."""
        rd_conflict = {}
        add_to_imaging_dict(
            rd_conflict, {"peakres": 0.9, "iter_done": 50}, time=0, pol=0, chan=0
        )

        merged = merge_imaging_dicts([self.rd1, rd_conflict], merge_strategy="latest")

        # Should use value from rd_conflict (last in list)
        # Note: fields are stored as lists even for single values (history tracking)
        entry = select_from_imaging_dict(merged, time=0, pol=0, chan=0)
        self.assertEqual(entry["peakres"], [0.9])  # Now a list
        self.assertEqual(entry["iter_done"], [50])  # Now a list

    def test_merge_strategy_error_raises_on_conflict(self):
        """Test 'error' merge strategy raises on overlapping keys."""
        rd_conflict = {}
        add_to_imaging_dict(
            rd_conflict, {"peakres": 0.9, "iter_done": 50}, time=0, pol=0, chan=0
        )

        with self.assertRaises(ValueError) as context:
            merge_imaging_dicts([self.rd1, rd_conflict], merge_strategy="error")

        self.assertIn("Conflicting key", str(context.exception))

    def test_merge_strategy_update(self):
        """Test 'update' merge strategy merges dict values."""
        rd_partial = {}
        add_to_imaging_dict(rd_partial, {"new_field": 123}, time=0, pol=0, chan=0)

        merged = merge_imaging_dicts([self.rd1, rd_partial], merge_strategy="update")

        entry = select_from_imaging_dict(merged, time=0, pol=0, chan=0)
        # Should have both original and new fields
        # History-tracked fields (FIELD_ACCUM) are stored as lists
        self.assertEqual(entry["peakres"], [0.5])  # Now a list
        self.assertEqual(entry["iter_done"], [100])  # Now a list
        # new_field is not in FIELD_ACCUM, so it's a single value
        self.assertEqual(entry["new_field"], 123)

    def test_merge_strategy_update_concatenates_history(self):
        """Test 'update' merge strategy concatenates FIELD_ACCUM history lists."""
        # Create a second dict with same key but new values
        rd_second_cycle = {}
        add_to_imaging_dict(
            rd_second_cycle, {"peakres": 0.3, "iter_done": 50}, time=0, pol=0, chan=0
        )

        # Merge with update strategy
        merged = merge_imaging_dicts(
            [self.rd1, rd_second_cycle], merge_strategy="update"
        )

        entry = select_from_imaging_dict(merged, time=0, pol=0, chan=0)
        # FIELD_ACCUM fields should be concatenated
        self.assertEqual(entry["peakres"], [0.5, 0.3])  # Concatenated history
        self.assertEqual(entry["iter_done"], [100, 50])  # Concatenated history

    def test_merge_empty_list(self):
        """Test merging empty list returns empty imaging_dict."""
        merged = merge_imaging_dicts([])
        self.assertEqual(len(merged), 0)

    def test_merge_invalid_strategy(self):
        """Test invalid merge strategy raises error."""
        # Create conflicting dicts to trigger the merge strategy check
        rd_conflict = {}
        add_to_imaging_dict(rd_conflict, {"peakres": 0.9}, time=0, pol=0, chan=0)

        with self.assertRaises(ValueError) as context:
            merge_imaging_dicts([self.rd1, rd_conflict], merge_strategy="invalid")

        self.assertIn("Unknown merge_strategy", str(context.exception))


class TestImagingDictUtilities(unittest.TestCase):
    """Test imaging_dict extraction utility functions."""

    def setUp(self):
        """Create test imaging_dict with realistic deconvolution data."""
        self.rd = {}

        # Add entries for multiple planes
        add_to_imaging_dict(
            self.rd,
            {
                "peakres": 1.0,
                "peakres_nomask": 1.2,
                "masksum": 100,
                "iter_done": 50,
                "max_psf_sidelobe": 0.2,
            },
            time=0,
            pol=0,
            chan=0,
        )

        add_to_imaging_dict(
            self.rd,
            {
                "peakres": 0.8,
                "peakres_nomask": 0.9,
                "masksum": 80,
                "iter_done": 40,
                "max_psf_sidelobe": 0.15,
            },
            time=0,
            pol=0,
            chan=1,
        )

        add_to_imaging_dict(
            self.rd,
            {
                "peakres": 0.5,
                "peakres_nomask": 0.6,
                "masksum": 0,  # Zero mask!
                "iter_done": 20,
                "max_psf_sidelobe": 0.1,
            },
            time=0,
            pol=1,
            chan=0,
        )

    def test_get_peak_residual_with_mask(self):
        """Test extracting peak residual with mask."""
        peak = get_peak_residual_from_imaging_dict(self.rd, use_mask=True)

        # Should return max across valid (non-zero mask) planes
        # Plane (0,0,0): 1.0, Plane (0,0,1): 0.8
        # Plane (0,1,0): masksum=0, ignored
        self.assertEqual(peak, 1.0)

    def test_get_peak_residual_without_mask(self):
        """Test extracting peak residual without mask."""
        peak = get_peak_residual_from_imaging_dict(self.rd, use_mask=False)

        # Should use peakres_nomask and include all planes
        self.assertEqual(peak, 1.2)

    def test_get_peak_residual_with_filter(self):
        """Test extracting peak residual with filtering."""
        peak = get_peak_residual_from_imaging_dict(self.rd, use_mask=True, chan=1)

        # Should only look at chan=1
        self.assertEqual(peak, 0.8)

    def test_get_masksum_total(self):
        """Test extracting total masksum."""
        total_masksum = get_masksum_from_imaging_dict(self.rd)

        # Sum: 100 + 80 + 0 = 180
        self.assertEqual(total_masksum, 180)

    def test_get_masksum_with_filter(self):
        """Test extracting masksum with filtering."""
        masksum = get_masksum_from_imaging_dict(self.rd, pol=0)

        # Sum for pol=0: 100 + 80 = 180
        self.assertEqual(masksum, 180)

        masksum_zero = get_masksum_from_imaging_dict(self.rd, pol=1)
        self.assertEqual(masksum_zero, 0)

    def test_get_iterations_done_total(self):
        """Test extracting total iterations done."""
        total_iters = get_iterations_done_from_imaging_dict(self.rd)

        # Sum: 50 + 40 + 20 = 110
        self.assertEqual(total_iters, 110)

    def test_get_iterations_done_with_filter(self):
        """Test extracting iterations with filtering."""
        iters = get_iterations_done_from_imaging_dict(self.rd, pol=0)

        # Sum for pol=0: 50 + 40 = 90
        self.assertEqual(iters, 90)

    def test_get_max_psf_sidelobe(self):
        """Test extracting max PSF sidelobe level."""
        max_sidelobe = get_max_psf_sidelobe_from_imaging_dict(self.rd)

        # Max across all planes: max(0.2, 0.15, 0.1) = 0.2
        self.assertEqual(max_sidelobe, 0.2)

    def test_get_max_psf_sidelobe_with_filter(self):
        """Test extracting max PSF sidelobe with filtering."""
        max_sidelobe = get_max_psf_sidelobe_from_imaging_dict(self.rd, chan=1)

        # Only chan=1: 0.15
        self.assertEqual(max_sidelobe, 0.15)

    def test_get_max_psf_sidelobe_default(self):
        """Test default PSF sidelobe when not in imaging_dict."""
        rd_empty = {}
        add_to_imaging_dict(
            rd_empty, {"peakres": 1.0, "iter_done": 50}, time=0, pol=0, chan=0
        )

        max_sidelobe = get_max_psf_sidelobe_from_imaging_dict(rd_empty)

        # Should return conservative default
        self.assertEqual(max_sidelobe, 0.2)

    def test_get_peak_residual_empty_returndict(self):
        """Test getting peak residual from empty imaging_dict."""
        rd_empty = {}
        peak = get_peak_residual_from_imaging_dict(rd_empty)

        self.assertEqual(peak, 0.0)

    def test_get_masksum_empty_returndict(self):
        """Test getting masksum from empty imaging_dict."""
        rd_empty = {}
        masksum = get_masksum_from_imaging_dict(rd_empty)

        self.assertEqual(masksum, 0.0)


class TestCycleControllerInitialization(unittest.TestCase):
    """Test cycle control state initialization and default values."""

    def test_default_initialization(self):
        """Test controller with all default parameters."""
        controller = make_cycle_control_state()

        # max_iter_remaining is a per-plane array allocated lazily; max_iter
        # holds the scalar per-plane budget.
        self.assertIsNone(controller["max_iter_remaining"])
        self.assertEqual(controller["max_iter"], 1000)
        self.assertEqual(controller["max_cycles"], -1)
        self.assertEqual(controller["threshold"], 0.0)
        self.assertEqual(controller["gain"], 0.1)
        self.assertEqual(controller["psf_sidelobe_factor"], 1.0)
        self.assertEqual(controller["min_psf_fraction"], 0.05)
        self.assertEqual(controller["max_psf_fraction"], 0.8)
        self.assertEqual(controller["max_iter_per_cycle"], -1)
        self.assertEqual(controller["threshold_sigma"], 0.0)

        # Tracking state
        self.assertEqual(controller["cycles_done"], 0)
        self.assertEqual(controller["total_iter_done"], 0)

        # Convergence state
        self.assertEqual(controller["stopcode"].imaging, IMAGING_CONTINUE)
        self.assertEqual(controller["stopcode"].model_update, MODEL_UPDATE_CONTINUE)
        self.assertIn("Continue", controller["stopdescription"])

    def test_custom_initialization(self):
        """Test controller with custom parameters."""
        controller = make_cycle_control_state(
            max_iter=500,
            max_cycles=10,
            threshold=0.01,
            gain=0.2,
            psf_sidelobe_factor=1.5,
            min_psf_fraction=0.1,
            max_psf_fraction=0.9,
            max_iter_per_cycle=100,
            threshold_sigma=5.0,
        )

        self.assertEqual(controller["max_iter"], 500)
        self.assertIsNone(controller["max_iter_remaining"])
        self.assertEqual(controller["max_cycles"], 10)
        self.assertEqual(controller["threshold"], 0.01)
        self.assertEqual(controller["gain"], 0.2)
        self.assertEqual(controller["psf_sidelobe_factor"], 1.5)
        self.assertEqual(controller["min_psf_fraction"], 0.1)
        self.assertEqual(controller["max_psf_fraction"], 0.9)
        self.assertEqual(controller["max_iter_per_cycle"], 100)
        self.assertEqual(controller["threshold_sigma"], 5.0)


class TestCalculatePerCycleLimits(unittest.TestCase):
    """Test adaptive threshold_per_cycle calculation."""

    def setUp(self):
        """Create controller and test imaging_dict."""
        self.controller = make_cycle_control_state(
            max_iter=1000,
            psf_sidelobe_factor=1.5,
            min_psf_fraction=0.05,
            max_psf_fraction=0.8,
        )

    def test_basic_threshold_per_cycle_calculation(self):
        """Test basic adaptive threshold calculation."""
        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 1.0, "max_psf_sidelobe": 0.2},
            time=0,
            pol=0,
            chan=0,
        )

        max_iter_per_cycle, threshold_per_cycle = calculate_per_cycle_limits(
            self.controller, rd
        )

        # Expected: psf_fraction = 1.5 × 0.2 = 0.3
        #           threshold_per_cycle = 0.3 × 1.0 = 0.3
        self.assertEqual(max_iter_per_cycle, 1000)
        self.assertAlmostEqual(threshold_per_cycle, 0.3, places=10)

    def test_threshold_per_cycle_clamping_min(self):
        """Test threshold_per_cycle clamping to min_psf_fraction."""
        rd = {}
        # Very small PSF sidelobe
        add_to_imaging_dict(
            rd,
            {"peakres": 1.0, "max_psf_sidelobe": 0.01},
            time=0,
            pol=0,
            chan=0,
        )

        max_iter_per_cycle, threshold_per_cycle = calculate_per_cycle_limits(
            self.controller, rd
        )

        # psf_fraction = 1.5 × 0.01 = 0.015
        # Clamped to min_psf_fraction = 0.05
        # threshold_per_cycle = 0.05 × 1.0 = 0.05
        self.assertEqual(threshold_per_cycle, 0.05)

    def test_threshold_per_cycle_clamping_max(self):
        """Test threshold_per_cycle clamping to max_psf_fraction."""
        rd = {}
        # Very large PSF sidelobe
        add_to_imaging_dict(
            rd,
            {"peakres": 1.0, "max_psf_sidelobe": 0.9},
            time=0,
            pol=0,
            chan=0,
        )

        max_iter_per_cycle, threshold_per_cycle = calculate_per_cycle_limits(
            self.controller, rd
        )

        # psf_fraction = 1.5 × 0.9 = 1.35
        # Clamped to max_psf_fraction = 0.8
        # threshold_per_cycle = 0.8 × 1.0 = 0.8
        self.assertEqual(threshold_per_cycle, 0.8)

    def test_threshold_per_cycle_respects_global_threshold(self):
        """Test threshold_per_cycle respects global threshold as minimum."""
        controller = make_cycle_control_state(
            max_iter=1000,
            threshold=0.5,  # High global threshold
            psf_sidelobe_factor=1.0,
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 1.0, "max_psf_sidelobe": 0.2},
            time=0,
            pol=0,
            chan=0,
        )

        max_iter_per_cycle, threshold_per_cycle = calculate_per_cycle_limits(
            controller, rd
        )

        # psf_fraction = 1.0 × 0.2 = 0.2
        # threshold_per_cycle_calc = 0.2 × 1.0 = 0.2
        # But threshold = 0.5 is higher, so use that
        self.assertEqual(threshold_per_cycle, 0.5)

    def test_max_iter_per_cycle_cap_applied(self):
        """Test max_iter_per_cycle cap is applied."""
        controller = make_cycle_control_state(
            max_iter=1000,
            max_iter_per_cycle=100,  # Cap at 100 per cycle
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 1.0, "max_psf_sidelobe": 0.2},
            time=0,
            pol=0,
            chan=0,
        )

        max_iter_per_cycle_updated, threshold_per_cycle = calculate_per_cycle_limits(
            controller, rd
        )

        # Should use min(100, 1000) = 100
        self.assertEqual(max_iter_per_cycle_updated, 100)

    def test_max_iter_per_cycle_respects_remaining_iterations(self):
        """Test max_iter_per_cycle respects remaining max_iter."""
        controller = make_cycle_control_state(
            max_iter=50,  # Only 50 iterations left
            max_iter_per_cycle=100,
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 1.0, "max_psf_sidelobe": 0.2},
            time=0,
            pol=0,
            chan=0,
        )

        max_iter_per_cycle_updated, threshold_per_cycle = calculate_per_cycle_limits(
            controller, rd
        )

        # Should use min(100, 50) = 50
        self.assertEqual(max_iter_per_cycle_updated, 50)

    def test_default_psf_sidelobe_used(self):
        """Test default PSF sidelobe when not in imaging_dict."""
        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 1.0},  # No max_psf_sidelobe
            time=0,
            pol=0,
            chan=0,
        )

        max_iter_per_cycle, threshold_per_cycle = calculate_per_cycle_limits(
            self.controller, rd
        )

        # Should use default 0.2
        # psf_fraction = 1.5 × 0.2 = 0.3
        # threshold_per_cycle = 0.3 × 1.0 = 0.3
        self.assertAlmostEqual(threshold_per_cycle, 0.3, places=10)


class TestBuildResidualImagingDict(unittest.TestCase):
    """Test seeding of the first-model-update imaging_dict from the residual image."""

    def _make_img_xds(self, residual_peak, real_sidelobe):
        residual = np.zeros((1, 1, 1, 4, 4))
        residual[0, 0, 0, 1, 1] = residual_peak
        sidelobe = np.full((1, 1, 1), real_sidelobe)
        img_xds = xr.Dataset(
            {
                "SKY_RESIDUAL": (
                    ("time", "frequency", "polarization", "l", "m"),
                    residual,
                ),
                "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                    ("time", "frequency", "polarization"),
                    sidelobe,
                ),
            }
        )
        img_xds.attrs["data_groups"] = {
            "residual": {
                "sky": "SKY_RESIDUAL",
                "max_sidelobe_point_spread_function": (
                    "MAX_SIDELOBE_POINT_SPREAD_FUNCTION"
                ),
            }
        }
        return img_xds

    def test_max_cycles_zero_measures_only_what_the_stop_test_needs(self):
        img_xds = self._make_img_xds(residual_peak=1.0, real_sidelobe=0.3)
        key = Key(time=0, chan=0, pol=0)
        skipped = build_residual_imaging_dict(
            img_xds, "residual", {"gain": 0.1, "max_cycles": 0}
        )[key]
        measured = build_residual_imaging_dict(
            img_xds, "residual", {"gain": 0.1, "max_cycles": 1}
        )[key]
        for field in ("peakres", "peakres_nomask", "masksum", "max_psf_sidelobe"):
            self.assertEqual(skipped[field], measured[field])
        for field in ("entropy", "residual_snr", *RESIDUAL_STATISTICS):
            self.assertTrue(np.isnan(skipped[field]), field)
        self.assertTrue(np.isfinite(measured["rms_in_mask"]))

    def test_max_psf_sidelobe_is_the_measured_value(self):
        real_sidelobe = 0.3
        img_xds = self._make_img_xds(residual_peak=1.0, real_sidelobe=real_sidelobe)
        rd = build_residual_imaging_dict(
            img_xds, "residual", {"gain": 0.1, "max_psf_fraction": 0.9}
        )
        entry = rd[Key(time=0, chan=0, pol=0)]
        self.assertEqual(entry["max_psf_sidelobe"], real_sidelobe)

    def test_threshold_per_cycle_uses_measured_sidelobe_not_max_psf_fraction(self):
        real_sidelobe = 0.3
        residual_peak = 1.0
        psf_sidelobe_factor = 1.0
        max_psf_fraction = 0.9
        img_xds = self._make_img_xds(residual_peak, real_sidelobe)
        rd = build_residual_imaging_dict(
            img_xds,
            "residual",
            {
                "gain": 0.1,
                "min_psf_fraction": 0.05,
                "max_psf_fraction": max_psf_fraction,
            },
        )
        controller = make_cycle_control_state(
            max_iter=100,
            psf_sidelobe_factor=psf_sidelobe_factor,
            min_psf_fraction=0.05,
            max_psf_fraction=max_psf_fraction,
            threshold=0.0,
        )

        _, threshold_per_cycle = calculate_per_cycle_limits(controller, rd)

        self.assertAlmostEqual(
            threshold_per_cycle, real_sidelobe * psf_sidelobe_factor * residual_peak
        )
        self.assertNotAlmostEqual(threshold_per_cycle, max_psf_fraction * residual_peak)


class TestCheckConvergence(unittest.TestCase):
    """Test convergence checking with all stop codes."""

    def test_continue_when_not_converged(self):
        """Test continuing when no stopping criteria met."""
        controller = make_cycle_control_state(
            max_iter=1000,
            threshold=0.01,
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertEqual(stopcode.model_update, MODEL_UPDATE_CONTINUE)
        self.assertIn("Continue", desc)

    def test_preserves_model_update_stop_code(self):
        """A post-model-update check keeps the deconvolver's reason per plane."""
        controller = make_cycle_control_state(max_iter=1000, threshold=0.01)
        rd = {}
        add_to_imaging_dict(
            rd,
            {
                "peakres": 0.5,
                "masksum": 100,
                "stop_code": StopCode(
                    imaging=IMAGING_CONTINUE,
                    model_update=MODEL_UPDATE_MAX_ITER_PER_CYCLE,
                ),
            },
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd, model_update_ran=True)

        self.assertEqual(
            stopcode,
            StopCode(imaging=IMAGING_CONTINUE, model_update=MODEL_UPDATE_CONTINUE),
        )
        self.assertEqual(
            controller["stop_code_model_update"][0, 0, 0],
            MODEL_UPDATE_MAX_ITER_PER_CYCLE,
        )
        self.assertEqual(
            select_from_imaging_dict(rd, time=0, pol=0, chan=0)["stop_code"],
            StopCode(
                imaging=IMAGING_CONTINUE,
                model_update=MODEL_UPDATE_MAX_ITER_PER_CYCLE,
            ),
        )
        self.assertEqual(desc, "Continue imaging cycles")

    def test_scalar_stop_code_is_not_treated_as_a_model_update_code(self):
        """Only the typed StopCode contract can report a model-update reason."""
        controller = make_cycle_control_state(max_iter=1000, threshold=0.01)
        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.5, "masksum": 100, "iter_done": 1, "stop_code": 4},
            time=0,
            pol=0,
            chan=0,
        )

        check_convergence(controller, rd, model_update_ran=True)

        self.assertEqual(
            controller["stop_code_model_update"][0, 0, 0], MODEL_UPDATE_CONTINUE
        )
        self.assertEqual(
            select_from_imaging_dict(rd, time=0, pol=0, chan=0)["stop_code"],
            StopCode(IMAGING_CONTINUE, MODEL_UPDATE_CONTINUE),
        )

    def test_fresh_residual_check_keeps_model_update_stop_code(self):
        """A later residual-only check keeps the code until the next model update."""
        controller = make_cycle_control_state(max_iter=1000, threshold=0.01)
        rd = {}
        add_to_imaging_dict(
            rd,
            {
                "peakres": 0.5,
                "masksum": 100,
                "stop_code": StopCode(
                    imaging=IMAGING_CONTINUE,
                    model_update=MODEL_UPDATE_THRESHOLD_PER_CYCLE,
                ),
            },
            time=0,
            pol=0,
            chan=0,
        )
        check_convergence(controller, rd, model_update_ran=True)

        stopcode, desc = check_convergence(controller, rd, model_update_ran=False)

        self.assertEqual(
            stopcode,
            StopCode(imaging=IMAGING_CONTINUE, model_update=MODEL_UPDATE_CONTINUE),
        )
        self.assertEqual(
            controller["stop_code_model_update"][0, 0, 0],
            MODEL_UPDATE_THRESHOLD_PER_CYCLE,
        )
        self.assertEqual(desc, "Continue imaging cycles")

    def test_aggregates_mixed_model_update_stop_codes(self):
        """Mixed plane results remain exact per plane and aggregate safely."""
        controller = make_cycle_control_state(max_iter=1000, threshold=0.01)
        rd = {}
        for pol, model_update in enumerate(
            (MODEL_UPDATE_THRESHOLD_PER_CYCLE, MODEL_UPDATE_DIVERGENCE)
        ):
            add_to_imaging_dict(
                rd,
                {
                    "peakres": 0.5,
                    "masksum": 100,
                    "stop_code": StopCode(
                        imaging=IMAGING_CONTINUE,
                        model_update=model_update,
                    ),
                },
                time=0,
                pol=pol,
                chan=0,
            )

        stopcode, desc = check_convergence(controller, rd, model_update_ran=True)

        self.assertEqual(
            stopcode,
            StopCode(imaging=IMAGING_CONTINUE, model_update=MODEL_UPDATE_CONTINUE),
        )
        np.testing.assert_array_equal(
            controller["stop_code_model_update"][0, 0],
            [MODEL_UPDATE_THRESHOLD_PER_CYCLE, MODEL_UPDATE_DIVERGENCE],
        )
        self.assertEqual(desc, "Continue imaging cycles")
        self.assertEqual(
            [entry["stop_code"].model_update for entry in rd.values()],
            [MODEL_UPDATE_THRESHOLD_PER_CYCLE, MODEL_UPDATE_DIVERGENCE],
        )

    def test_rejects_unknown_model_update_stop_code(self):
        """Do not silently propagate an undocumented native return code."""
        controller = make_cycle_control_state(max_iter=1000)
        rd = {}
        add_to_imaging_dict(
            rd,
            {
                "peakres": 0.5,
                "masksum": 100,
                "stop_code": StopCode(imaging=IMAGING_CONTINUE, model_update=99),
            },
            time=0,
            pol=0,
            chan=0,
        )

        with self.assertRaisesRegex(ValueError, "Unknown model-update stop code 99"):
            check_convergence(controller, rd, model_update_ran=True)

    def test_stop_on_zero_mask(self):
        """Test stopping when mask is zero (priority 1)."""
        controller = make_cycle_control_state(max_iter=1000)

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.5, "masksum": 0},  # Zero mask!
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        self.assertEqual(stopcode.imaging, IMAGING_ZERO_MASK)
        self.assertIn("Zero mask", desc)

    def test_stop_on_iteration_limit(self):
        """Test stopping when max_iter exhausted (priority 2)."""
        controller = make_cycle_control_state(max_iter=0)  # No iterations left

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        self.assertEqual(stopcode.imaging, IMAGING_MAX_ITER)
        self.assertIn("max_iter", desc)

    def test_stop_on_threshold(self):
        """Test stopping when threshold reached (priority 3)."""
        controller = make_cycle_control_state(
            max_iter=1000,
            threshold=0.5,  # Set threshold
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.3, "masksum": 100},  # Below threshold
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        self.assertEqual(stopcode.imaging, IMAGING_THRESHOLD)
        self.assertIn("threshold", desc)

    def test_threshold_exactly_at_limit(self):
        """Test stopping when peak residual exactly equals threshold."""
        controller = make_cycle_control_state(
            max_iter=1000,
            threshold=0.5,
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.5, "masksum": 100},  # Exactly at threshold
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        self.assertEqual(stopcode.imaging, IMAGING_THRESHOLD)

    def test_stop_on_max_cycles(self):
        """Test stopping when max_cycles exhausted (priority 4)."""
        controller = make_cycle_control_state(
            max_iter=1000,
            max_cycles=0,  # No imaging cycles left
            threshold=0.0,
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        self.assertEqual(stopcode.imaging, IMAGING_MAX_CYCLES)
        self.assertIn("max_cycles", desc)

    def test_priority_zero_mask_over_others(self):
        """Test zero mask has highest priority."""
        controller = make_cycle_control_state(
            max_iter=0,  # Also at iteration limit
            max_cycles=0,  # Also at imaging cycle limit
            threshold=1.0,  # Also below threshold
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.5, "masksum": 0},  # Zero mask
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        # Zero mask should win
        self.assertEqual(stopcode.imaging, IMAGING_ZERO_MASK)

    def test_priority_iter_limit_over_threshold(self):
        """Test iteration limit has priority over threshold."""
        controller = make_cycle_control_state(
            max_iter=0,  # At iteration limit
            threshold=1.0,  # Also below threshold
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        # Iteration limit should win
        self.assertEqual(stopcode.imaging, IMAGING_MAX_ITER)

    def test_max_cycles_unlimited_never_stops(self):
        """Test max_cycles=-1 (unlimited) never triggers cycle limit."""
        controller = make_cycle_control_state(
            max_iter=1000,
            max_cycles=-1,  # Unlimited
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)

    def test_threshold_zero_never_stops(self):
        """Test threshold=0 (disabled) never triggers threshold stop."""
        controller = make_cycle_control_state(
            max_iter=1000,
            threshold=0.0,  # Disabled
        )

        rd = {}
        add_to_imaging_dict(
            rd,
            {"peakres": 0.001, "masksum": 100},  # Very low residual
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = check_convergence(controller, rd)

        # Should not stop on threshold
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)

    def test_stopcode_state_updated(self):
        """Test controller's internal stopcode is updated."""
        controller = make_cycle_control_state(max_iter=0)

        rd = {}
        add_to_imaging_dict(rd, {"peakres": 0.5, "masksum": 100}, time=0, pol=0, chan=0)

        stopcode, desc = check_convergence(controller, rd)

        # Internal state should match returned values
        self.assertEqual(controller["stopcode"].imaging, stopcode.imaging)
        self.assertEqual(controller["stopdescription"], desc)


class TestUpdateCounts(unittest.TestCase):
    """Test iteration count updates after imaging cycles."""

    def test_basic_count_update(self):
        """Test basic count decrementing."""
        controller = make_cycle_control_state(max_iter=1000, max_cycles=5)

        rd = {}
        add_to_imaging_dict(rd, {"iter_done": 100}, time=0, pol=0, chan=0)

        update_counts(controller, rd)

        # max_iter: 1000 - 100 = 900 (per-plane, plane (0,0,0))
        # max_cycles: 5 - 1 = 4
        # cycles_done: 0 + 1 = 1
        # total_iter_done: 0 + 100 = 100
        self.assertEqual(controller["max_iter_remaining"][0, 0, 0], 900)
        self.assertEqual(controller["max_cycles"], 4)
        self.assertEqual(controller["cycles_done"], 1)
        self.assertEqual(controller["total_iter_done"], 100)

    def test_multiple_updates(self):
        """Test multiple count updates accumulate correctly."""
        controller = make_cycle_control_state(max_iter=1000, max_cycles=5)

        rd1 = {}
        add_to_imaging_dict(rd1, {"iter_done": 100}, time=0, pol=0, chan=0)

        rd2 = {}
        add_to_imaging_dict(rd2, {"iter_done": 150}, time=0, pol=0, chan=0)

        update_counts(controller, rd1)
        update_counts(controller, rd2)

        self.assertEqual(
            controller["max_iter_remaining"][0, 0, 0], 750
        )  # 1000 - 100 - 150
        self.assertEqual(controller["max_cycles"], 3)  # 5 - 1 - 1
        self.assertEqual(controller["cycles_done"], 2)
        self.assertEqual(controller["total_iter_done"], 250)  # 100 + 150

    def test_max_iter_floor_at_zero(self):
        """Test max_iter doesn't go negative."""
        controller = make_cycle_control_state(max_iter=50)

        rd = {}
        add_to_imaging_dict(rd, {"iter_done": 100}, time=0, pol=0, chan=0)

        update_counts(controller, rd)

        # Should floor at 0, not go negative
        self.assertEqual(controller["max_iter_remaining"][0, 0, 0], 0)

    def test_max_cycles_floor_at_zero(self):
        """Test max_cycles doesn't go negative."""
        controller = make_cycle_control_state(max_cycles=1)

        rd1 = {}
        add_to_imaging_dict(rd1, {"iter_done": 50}, time=0, pol=0, chan=0)

        rd2 = {}
        add_to_imaging_dict(rd2, {"iter_done": 50}, time=0, pol=0, chan=0)

        update_counts(controller, rd1)
        update_counts(controller, rd2)

        # Should floor at 0, not go negative
        self.assertEqual(controller["max_cycles"], 0)

    def test_max_cycles_unlimited_never_decrements(self):
        """Test max_cycles=-1 (unlimited) never decrements."""
        controller = make_cycle_control_state(max_iter=1000, max_cycles=-1)

        rd = {}
        add_to_imaging_dict(rd, {"iter_done": 100}, time=0, pol=0, chan=0)

        update_counts(controller, rd)

        # max_cycles should stay -1
        self.assertEqual(controller["max_cycles"], -1)
        self.assertEqual(controller["cycles_done"], 1)

    def test_update_with_multiple_planes(self):
        """Test update sums iterations across multiple planes."""
        controller = make_cycle_control_state(max_iter=1000)

        rd = {}
        add_to_imaging_dict(rd, {"iter_done": 50}, time=0, pol=0, chan=0)
        add_to_imaging_dict(rd, {"iter_done": 30}, time=0, pol=0, chan=1)
        add_to_imaging_dict(rd, {"iter_done": 20}, time=0, pol=1, chan=0)

        update_counts(controller, rd)

        # Each plane's remaining max_iter is decremented independently by its own
        # iter_done. Arrays are indexed (time, chan, pol).
        self.assertEqual(
            controller["max_iter_remaining"][0, 0, 0], 950
        )  # Key(0,0,0): 1000-50
        self.assertEqual(
            controller["max_iter_remaining"][0, 1, 0], 970
        )  # Key(0,0,1): 1000-30
        self.assertEqual(
            controller["max_iter_remaining"][0, 0, 1], 980
        )  # Key(0,1,0): 1000-20
        # total_iter_done sums across planes: 50 + 30 + 20 = 100
        self.assertEqual(controller["total_iter_done"], 100)
        self.assertEqual(controller["cycles_done"], 1)

    def test_no_update_when_converged(self):
        """Test update_counts does nothing when already converged."""
        controller = make_cycle_control_state(max_iter=1000, max_cycles=5)

        # Set converged state
        controller["stopcode"] = StopCode(
            imaging=IMAGING_THRESHOLD, model_update=MODEL_UPDATE_CONTINUE
        )

        rd = {}
        add_to_imaging_dict(rd, {"iter_done": 100}, time=0, pol=0, chan=0)

        update_counts(controller, rd)

        # Counts should not change (update_counts returns early when converged,
        # so the per-plane array is never even allocated).
        self.assertIsNone(controller["max_iter_remaining"])
        self.assertEqual(controller["max_iter"], 1000)
        self.assertEqual(controller["max_cycles"], 5)
        self.assertEqual(controller["cycles_done"], 0)
        self.assertEqual(controller["total_iter_done"], 0)


class TestUpdateParameters(unittest.TestCase):
    """Test interactive parameter updates with validation."""

    def test_update_max_iter(self):
        """Test updating max_iter parameter."""
        controller = make_cycle_control_state(max_iter=1000)

        code, msg = update_parameters(controller, max_iter=500)

        self.assertEqual(code, 0)
        self.assertEqual(msg, "")
        # max_iter sets the per-plane budget; with no array allocated yet this is
        # reflected in the "max_iter" of the state.
        self.assertEqual(controller["max_iter"], 500)

    def test_update_max_iter_per_cycle(self):
        """Test updating max_iter_per_cycle parameter."""
        controller = make_cycle_control_state(max_iter_per_cycle=-1)

        code, msg = update_parameters(controller, max_iter_per_cycle=100)

        self.assertEqual(code, 0)
        self.assertEqual(controller["max_iter_per_cycle"], 100)

    def test_update_max_cycles(self):
        """Test updating max_cycles parameter."""
        controller = make_cycle_control_state(max_cycles=-1)

        code, msg = update_parameters(controller, max_cycles=10)

        self.assertEqual(code, 0)
        self.assertEqual(controller["max_cycles"], 10)

    def test_update_threshold_numeric(self):
        """Test updating threshold with numeric value."""
        controller = make_cycle_control_state(threshold=0.0)

        code, msg = update_parameters(controller, threshold=0.5)

        self.assertEqual(code, 0)
        self.assertEqual(controller["threshold"], 0.5)

    def test_update_threshold_string_jy(self):
        """Test updating threshold with Jy string."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(controller, threshold="0.5Jy")

        self.assertEqual(code, 0)
        self.assertEqual(controller["threshold"], 0.5)

    def test_update_threshold_string_mjy(self):
        """Test updating threshold with mJy string."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(controller, threshold="10mJy")

        self.assertEqual(code, 0)
        self.assertAlmostEqual(controller["threshold"], 0.01)

    def test_update_threshold_string_ujy(self):
        """Test updating threshold with uJy string."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(controller, threshold="100uJy")

        self.assertEqual(code, 0)
        self.assertAlmostEqual(controller["threshold"], 0.0001)

    def test_update_cyclefactor(self):
        """Test updating psf_sidelobe_factor parameter."""
        controller = make_cycle_control_state(psf_sidelobe_factor=1.0)

        code, msg = update_parameters(controller, psf_sidelobe_factor=1.5)

        self.assertEqual(code, 0)
        self.assertEqual(controller["psf_sidelobe_factor"], 1.5)

    def test_update_multiple_parameters(self):
        """Test updating multiple parameters at once."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(
            controller,
            max_iter=500,
            threshold="5mJy",
            psf_sidelobe_factor=1.5,
        )

        self.assertEqual(code, 0)
        self.assertEqual(controller["max_iter"], 500)
        self.assertAlmostEqual(controller["threshold"], 0.005)
        self.assertEqual(controller["psf_sidelobe_factor"], 1.5)

    def test_reject_negative_max_iter(self):
        """Test rejecting max_iter < -1."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(controller, max_iter=-2)

        self.assertEqual(code, -1)
        self.assertIn("max_iter must be >= -1", msg)
        self.assertEqual(controller["max_iter"], 1000)  # Unchanged

    def test_reject_negative_threshold_numeric(self):
        """Test rejecting negative numeric threshold."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(controller, threshold=-0.5)

        self.assertEqual(code, -1)
        self.assertIn("threshold must be >= 0", msg)

    def test_reject_negative_threshold_string(self):
        """Test rejecting negative threshold string (Bug #1 fix)."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(controller, threshold="-10mJy")

        self.assertEqual(code, -1)
        self.assertIn("threshold must be >= 0", msg)

    def test_reject_zero_cyclefactor(self):
        """Test rejecting psf_sidelobe_factor <= 0."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(controller, psf_sidelobe_factor=0)

        self.assertEqual(code, -1)
        self.assertIn("psf_sidelobe_factor must be > 0", msg)

        code, msg = update_parameters(controller, psf_sidelobe_factor=-1.0)

        self.assertEqual(code, -1)
        self.assertIn("psf_sidelobe_factor must be > 0", msg)

    def test_reject_invalid_threshold_string(self):
        """Test rejecting threshold string with unknown units."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(controller, threshold="10kJy")

        self.assertEqual(code, -1)
        self.assertIn("number with units", msg)

    def test_reject_non_numeric_max_iter(self):
        """Test rejecting non-numeric max_iter."""
        controller = make_cycle_control_state()

        code, msg = update_parameters(controller, max_iter="abc")

        self.assertEqual(code, -1)
        self.assertIn("integer", msg)


class TestResetMethods(unittest.TestCase):
    """Test reset functionality."""

    def test_reset(self):
        """Test full reset restores initial state."""
        controller = make_cycle_control_state(max_iter=1000, max_cycles=5)

        # Simulate some work
        rd = {}
        add_to_imaging_dict(rd, {"iter_done": 300}, time=0, pol=0, chan=0)
        update_counts(controller, rd)

        # Modify stopcode
        controller["stopcode"] = StopCode(
            imaging=IMAGING_THRESHOLD, model_update=MODEL_UPDATE_CONTINUE
        )

        # Reset
        reset(controller)

        # Should restore to initial state (per-plane array reset to the budget)
        self.assertEqual(controller["max_iter_remaining"][0, 0, 0], 1000)
        self.assertEqual(controller["cycles_done"], 0)
        self.assertEqual(controller["total_iter_done"], 0)
        self.assertEqual(controller["stopcode"].imaging, IMAGING_CONTINUE)
        self.assertEqual(controller["stopcode"].model_update, MODEL_UPDATE_CONTINUE)

    def test_reset_stopcode_only(self):
        """Test reset_stopcode only resets convergence state."""
        controller = make_cycle_control_state(max_iter=1000)

        # Simulate some work
        rd = {}
        add_to_imaging_dict(rd, {"iter_done": 300}, time=0, pol=0, chan=0)
        update_counts(controller, rd)

        # Set converged
        controller["stopcode"] = StopCode(
            imaging=IMAGING_THRESHOLD, model_update=MODEL_UPDATE_CONTINUE
        )

        # Reset only stopcode
        reset_stopcode(controller)

        # Stopcode should be reset
        self.assertEqual(controller["stopcode"].imaging, IMAGING_CONTINUE)
        self.assertEqual(controller["stopcode"].model_update, MODEL_UPDATE_CONTINUE)

        # But counts should remain
        self.assertEqual(controller["max_iter_remaining"][0, 0, 0], 700)
        self.assertEqual(controller["cycles_done"], 1)


class TestGetState(unittest.TestCase):
    """Test state serialization."""

    def test_get_state(self):
        """Test getting controller state as dictionary."""
        controller = make_cycle_control_state(
            max_iter=1000,
            max_cycles=5,
            threshold=0.01,
            psf_sidelobe_factor=1.5,
        )

        # Simulate some work
        rd = {}
        add_to_imaging_dict(rd, {"iter_done": 200}, time=0, pol=0, chan=0)
        update_counts(controller, rd)

        state = get_state(controller)

        # Check all fields present. max_iter is serialized per-plane (nested list).
        self.assertEqual(state["max_iter_remaining"][0][0][0], 800)
        self.assertEqual(state["max_cycles"], 4)
        self.assertEqual(state["max_iter"], 1000)
        self.assertEqual(state["threshold"], 0.01)
        self.assertEqual(state["psf_sidelobe_factor"], 1.5)
        self.assertEqual(state["cycles_done"], 1)
        self.assertEqual(state["total_iter_done"], 200)

        # Stopcode should be serialized as dict
        self.assertIsInstance(state["stopcode"], dict)
        self.assertEqual(state["stopcode"]["imaging"], IMAGING_CONTINUE)
        self.assertEqual(state["stopcode"]["model_update"], MODEL_UPDATE_CONTINUE)


class TestFullImagingCycleWorkflow(unittest.TestCase):
    """Integration tests for complete imaging cycle workflows."""

    def test_basic_convergence_workflow(self):
        """Test basic imaging cycle workflow until convergence."""
        controller = make_cycle_control_state(
            max_iter=300,
            max_cycles=5,
            threshold=0.1,
            psf_sidelobe_factor=1.5,
        )

        # Simulate 3 imaging cycles with decreasing residual
        residuals = [1.0, 0.5, 0.08]  # Last one below threshold
        iterations_per_cycle = [100, 100, 50]

        for cycle, (residual, iters) in enumerate(
            zip(residuals, iterations_per_cycle, strict=False)
        ):
            # Create imaging_dict for this cycle
            rd = {}
            add_to_imaging_dict(
                rd,
                {
                    "peakres": residual,
                    "masksum": 100,
                    "iter_done": iters,
                    "max_psf_sidelobe": 0.2,
                },
                time=0,
                pol=0,
                chan=0,
            )

            # Calculate per cycle limits
            max_iter_per_cycle, threshold_per_cycle = calculate_per_cycle_limits(
                controller, rd
            )

            # Update counts
            update_counts(controller, rd)

            # Check convergence
            stopcode, desc = check_convergence(controller, rd)

            if stopcode.imaging != IMAGING_CONTINUE:
                # Should converge on cycle 3 (index 2)
                self.assertEqual(cycle, 2)
                self.assertEqual(stopcode.imaging, IMAGING_THRESHOLD)
                break

        # Verify final state
        self.assertEqual(controller["cycles_done"], 3)
        self.assertEqual(controller["total_iter_done"], 250)
        self.assertEqual(controller["max_iter_remaining"][0, 0, 0], 50)  # 300 - 250

    def test_iteration_limit_workflow(self):
        """Test workflow stopping at iteration limit."""
        controller = make_cycle_control_state(
            max_iter=250,  # Limited iterations
            threshold=0.01,  # Low threshold (hard to reach)
        )

        # Simulate imaging cycles
        for cycle in range(5):
            rd = {}
            add_to_imaging_dict(
                rd,
                {
                    "peakres": 0.5,  # Stays high
                    "masksum": 100,
                    "iter_done": 100,
                    "max_psf_sidelobe": 0.2,
                },
                time=0,
                pol=0,
                chan=0,
            )

            update_counts(controller, rd)
            stopcode, desc = check_convergence(controller, rd)

            if stopcode.imaging != IMAGING_CONTINUE:
                # Should stop after cycle 3 (3 × 100 = 300 > 250)
                # But max_iter floors at 0 after cycle 2
                self.assertEqual(cycle, 2)
                self.assertEqual(stopcode.imaging, IMAGING_MAX_ITER)
                break

    def test_max_cycles_workflow(self):
        """Test workflow stopping at imaging cycle limit."""
        controller = make_cycle_control_state(
            max_iter=1000,
            max_cycles=3,  # Limited imaging cycles
            threshold=0.01,
        )

        # Simulate imaging cycles
        for cycle in range(5):
            rd = {}
            add_to_imaging_dict(
                rd,
                {
                    "peakres": 0.5,  # Stays high
                    "masksum": 100,
                    "iter_done": 100,
                    "max_psf_sidelobe": 0.2,
                },
                time=0,
                pol=0,
                chan=0,
            )

            # Check convergence first, then update counts
            # After 3 updates, max_cycles will be 0, triggering stop
            update_counts(controller, rd)
            stopcode, desc = check_convergence(controller, rd)

            if stopcode.imaging != IMAGING_CONTINUE:
                # Should stop when max_cycles reaches 0 after 3 cycles
                # Cycle 0: max_cycles 3→2, Cycle 1: max_cycles 2→1, Cycle 2: max_cycles 1→0
                # Check on cycle 2 detects max_cycles==0
                self.assertEqual(cycle, 2)
                self.assertEqual(stopcode.imaging, IMAGING_MAX_CYCLES)
                break

        self.assertEqual(controller["cycles_done"], 3)

    def test_zero_mask_workflow(self):
        """Test workflow stopping when mask becomes zero."""
        controller = make_cycle_control_state(
            max_iter=1000,
            threshold=0.1,
        )

        # Simulate cycles with decreasing mask
        masksums = [100, 50, 0]  # Mask disappears

        for cycle, masksum in enumerate(masksums):
            rd = {}
            add_to_imaging_dict(
                rd,
                {
                    "peakres": 0.5,
                    "masksum": masksum,
                    "iter_done": 100,
                    "max_psf_sidelobe": 0.2,
                },
                time=0,
                pol=0,
                chan=0,
            )

            update_counts(controller, rd)
            stopcode, desc = check_convergence(controller, rd)

            if stopcode.imaging != IMAGING_CONTINUE:
                # Should stop when mask becomes zero
                self.assertEqual(cycle, 2)
                self.assertEqual(stopcode.imaging, IMAGING_ZERO_MASK)
                break

    def test_interactive_continue_workflow(self):
        """Test interactive workflow: stop, update params, continue."""
        controller = make_cycle_control_state(
            max_iter=100,
            threshold=0.5,
        )

        # First phase: run until iteration limit
        rd1 = {}
        add_to_imaging_dict(
            rd1,
            {
                "peakres": 0.8,
                "masksum": 100,
                "iter_done": 100,
                "max_psf_sidelobe": 0.2,
            },
            time=0,
            pol=0,
            chan=0,
        )

        update_counts(controller, rd1)
        stopcode, desc = check_convergence(controller, rd1)

        self.assertEqual(stopcode.imaging, IMAGING_MAX_ITER)

        # User decides to continue with more iterations
        code, msg = update_parameters(controller, max_iter=200)
        self.assertEqual(code, 0)

        # Reset stopcode to continue
        reset_stopcode(controller)

        # Second phase: run more cycles
        rd2 = {}
        add_to_imaging_dict(
            rd2,
            {
                "peakres": 0.3,  # Below threshold now
                "masksum": 100,
                "iter_done": 100,
                "max_psf_sidelobe": 0.2,
            },
            time=0,
            pol=0,
            chan=0,
        )

        update_counts(controller, rd2)
        stopcode, desc = check_convergence(controller, rd2)

        self.assertEqual(stopcode.imaging, IMAGING_THRESHOLD)
        self.assertEqual(controller["cycles_done"], 2)
        self.assertEqual(controller["total_iter_done"], 200)


class TestMultiPlaneWorkflows(unittest.TestCase):
    """Test workflows with multiple time/pol/chan planes."""

    def test_merge_and_converge_multi_channel(self):
        """Test merging results from multiple channels."""
        controller = make_cycle_control_state(
            max_iter=300,
            threshold=0.1,
        )

        # Simulate 3 workers processing different channels
        rd1 = {}
        add_to_imaging_dict(
            rd1,
            {
                "peakres": 0.5,
                "masksum": 100,
                "iter_done": 50,
                "max_psf_sidelobe": 0.2,
            },
            time=0,
            pol=0,
            chan=0,
        )

        rd2 = {}
        add_to_imaging_dict(
            rd2,
            {
                "peakres": 0.8,  # Highest residual
                "masksum": 80,
                "iter_done": 60,
                "max_psf_sidelobe": 0.15,
            },
            time=0,
            pol=0,
            chan=1,
        )

        rd3 = {}
        add_to_imaging_dict(
            rd3,
            {
                "peakres": 0.3,
                "masksum": 120,
                "iter_done": 40,
                "max_psf_sidelobe": 0.25,  # Highest sidelobe
            },
            time=0,
            pol=0,
            chan=2,
        )

        # Merge results
        merged = merge_imaging_dicts([rd1, rd2, rd3])

        # Get global statistics
        peak = get_peak_residual_from_imaging_dict(merged)
        total_iters = get_iterations_done_from_imaging_dict(merged)
        max_sidelobe = get_max_psf_sidelobe_from_imaging_dict(merged)

        self.assertEqual(peak, 0.8)  # Max across channels
        self.assertEqual(total_iters, 150)  # 50 + 60 + 40
        self.assertEqual(max_sidelobe, 0.25)  # Max sidelobe

        # Check global convergence
        update_counts(controller, merged)
        stopcode, desc = check_convergence(controller, merged)

        # Should not converge: every plane's own peak (0.5, 0.8, 0.3) is above
        # the threshold 0.1, so all planes stay active and the aggregate is
        # CONTINUE.
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        # Each channel's remaining max_iter is decremented by its own iter_done.
        self.assertEqual(
            controller["max_iter_remaining"][0, 0, 0], 250
        )  # chan0: 300-50
        self.assertEqual(
            controller["max_iter_remaining"][0, 1, 0], 240
        )  # chan1: 300-60
        self.assertEqual(
            controller["max_iter_remaining"][0, 2, 0], 260
        )  # chan2: 300-40
        self.assertEqual(controller["total_iter_done"], 150)  # 50 + 60 + 40

    def test_partial_zero_mask_handling(self):
        """Test per-plane handling when some planes have zero mask.

        With independent per-plane stopping, the zero-mask plane stops with
        ZERO_MASK while the valid plane stops on its own threshold; the
        aggregate reports that all planes have stopped.
        """
        controller = make_cycle_control_state(threshold=0.5)

        # Chan 0: zero mask, Chan 1: valid
        rd = {}
        add_to_imaging_dict(
            rd,
            {
                "peakres": 1.0,
                "masksum": 0,  # Zero mask
                "iter_done": 50,
            },
            time=0,
            pol=0,
            chan=0,
        )
        add_to_imaging_dict(
            rd,
            {
                "peakres": 0.3,
                "masksum": 100,  # Valid mask
                "iter_done": 50,
            },
            time=0,
            pol=0,
            chan=1,
        )

        # Global masksum is non-zero
        total_masksum = get_masksum_from_imaging_dict(rd)
        self.assertEqual(total_masksum, 100)

        # Peak should ignore zero-mask plane
        peak = get_peak_residual_from_imaging_dict(rd, use_mask=True)
        self.assertEqual(peak, 0.3)  # Only chan 1

        # Each plane stops for its own reason: chan 0 (zero mask) -> ZERO_MASK,
        # chan 1 (peak 0.3 <= threshold 0.5) -> THRESHOLD.
        stopcode, desc = check_convergence(controller, rd)
        self.assertEqual(
            controller["stop_code_imaging"][0, 0, 0], IMAGING_ZERO_MASK
        )  # chan 0
        self.assertEqual(
            controller["stop_code_imaging"][0, 1, 0], IMAGING_THRESHOLD
        )  # chan 1
        # Both planes have stopped, so the aggregate is non-CONTINUE.
        self.assertNotEqual(stopcode.imaging, IMAGING_CONTINUE)


# =============================================================================
# Key Validation Tests (Issue #157)
# =============================================================================


class TestImagingDictKeyValidation(unittest.TestCase):
    """Test key validation in imaging_dict utility functions."""

    def setUp(self):
        """Create test imaging_dict with known keys."""
        self.rd = {}
        add_to_imaging_dict(
            self.rd,
            {
                "peakres": 1.0,
                "masksum": 100,
                "iter_done": 50,
                "max_psf_sidelobe": 0.2,
                "model_flux": 1.5,
            },
            time=0,
            pol=0,
            chan=0,
        )
        add_to_imaging_dict(
            self.rd,
            {
                "peakres": 0.8,
                "masksum": 80,
                "iter_done": 40,
                "max_psf_sidelobe": 0.15,
                "model_flux": 1.2,
            },
            time=0,
            pol=0,
            chan=1,
        )

    # --- get_peak_residual_from_imaging_dict tests ---

    def test_get_peak_residual_invalid_time_raises_keyerror(self):
        """Test KeyError raised for non-existent time."""
        with self.assertRaises(KeyError) as context:
            get_peak_residual_from_imaging_dict(self.rd, time=99, pol=0, chan=0)
        self.assertIn("time=99", str(context.exception))
        self.assertIn("Available keys", str(context.exception))

    def test_get_peak_residual_invalid_pol_raises_keyerror(self):
        """Test KeyError raised for non-existent pol."""
        with self.assertRaises(KeyError) as context:
            get_peak_residual_from_imaging_dict(self.rd, time=0, pol=99, chan=0)
        self.assertIn("pol=99", str(context.exception))

    def test_get_peak_residual_invalid_chan_raises_keyerror(self):
        """Test KeyError raised for non-existent chan."""
        with self.assertRaises(KeyError) as context:
            get_peak_residual_from_imaging_dict(self.rd, time=0, pol=0, chan=99)
        self.assertIn("chan=99", str(context.exception))

    def test_get_peak_residual_partial_match_failure(self):
        """Test KeyError when some filters match but not all."""
        # time=0 exists, pol=0 exists, but chan=2 doesn't
        with self.assertRaises(KeyError) as context:
            get_peak_residual_from_imaging_dict(self.rd, time=0, pol=0, chan=2)
        self.assertIn("chan=2", str(context.exception))

    def test_get_peak_residual_valid_key_succeeds(self):
        """Test no error for valid key."""
        result = get_peak_residual_from_imaging_dict(self.rd, time=0, pol=0, chan=0)
        self.assertEqual(result, 1.0)

    def test_get_peak_residual_wildcard_no_error(self):
        """Test no error when using wildcards (None values)."""
        # These should NOT raise errors
        result = get_peak_residual_from_imaging_dict(self.rd, time=0)
        self.assertEqual(result, 1.0)  # Max of 1.0 and 0.8

        result = get_peak_residual_from_imaging_dict(self.rd)  # All None
        self.assertEqual(result, 1.0)

    def test_get_peak_residual_empty_returndict_with_filter(self):
        """Test KeyError on empty imaging_dict with explicit filter."""
        rd_empty = {}
        with self.assertRaises(KeyError) as context:
            get_peak_residual_from_imaging_dict(rd_empty, time=0, pol=0, chan=0)
        self.assertIn("empty", str(context.exception))

    def test_get_peak_residual_empty_returndict_no_filter(self):
        """Test no error on empty imaging_dict without filters."""
        rd_empty = {}
        result = get_peak_residual_from_imaging_dict(rd_empty)  # All None
        self.assertEqual(result, 0.0)  # Default value

    # --- get_masksum_from_imaging_dict tests ---

    def test_get_masksum_invalid_key_raises_keyerror(self):
        """Test KeyError raised for non-existent key."""
        with self.assertRaises(KeyError):
            get_masksum_from_imaging_dict(self.rd, time=99, pol=0, chan=0)

    def test_get_masksum_valid_key_succeeds(self):
        """Test no error for valid key."""
        result = get_masksum_from_imaging_dict(self.rd, time=0, pol=0, chan=0)
        self.assertEqual(result, 100)

    # --- get_iterations_done_from_imaging_dict tests ---

    def test_get_iterations_done_invalid_key_raises_keyerror(self):
        """Test KeyError raised for non-existent key."""
        with self.assertRaises(KeyError):
            get_iterations_done_from_imaging_dict(self.rd, time=0, pol=0, chan=99)

    def test_get_iterations_done_valid_key_succeeds(self):
        """Test no error for valid key."""
        result = get_iterations_done_from_imaging_dict(self.rd, time=0, pol=0, chan=0)
        self.assertEqual(result, 50)

    # --- get_max_psf_sidelobe_from_imaging_dict tests ---

    def test_get_max_psf_sidelobe_invalid_key_raises_keyerror(self):
        """Test KeyError raised for non-existent key."""
        with self.assertRaises(KeyError):
            get_max_psf_sidelobe_from_imaging_dict(self.rd, time=0, pol=99, chan=0)

    def test_get_max_psf_sidelobe_valid_key_succeeds(self):
        """Test no error for valid key."""
        result = get_max_psf_sidelobe_from_imaging_dict(self.rd, time=0, pol=0, chan=0)
        self.assertEqual(result, 0.2)

    # --- get_model_flux_from_imaging_dict tests ---

    def test_get_model_flux_invalid_key_raises_keyerror(self):
        """Test KeyError raised for non-existent key."""
        with self.assertRaises(KeyError):
            get_model_flux_from_imaging_dict(self.rd, time=99, pol=99, chan=99)

    def test_get_model_flux_valid_key_succeeds(self):
        """Test no error for valid key."""
        result = get_model_flux_from_imaging_dict(self.rd, time=0, pol=0, chan=0)
        self.assertEqual(result, 1.5)

    # --- Error message format tests ---

    def test_error_message_includes_filter_values(self):
        """Test error message includes the filter values that were specified."""
        with self.assertRaises(KeyError) as context:
            get_peak_residual_from_imaging_dict(self.rd, time=5, pol=3, chan=10)

        error_msg = str(context.exception)
        self.assertIn("time=5", error_msg)
        self.assertIn("pol=3", error_msg)
        self.assertIn("chan=10", error_msg)

    def test_error_message_includes_available_keys(self):
        """Test error message includes available keys for debugging."""
        with self.assertRaises(KeyError) as context:
            get_peak_residual_from_imaging_dict(self.rd, time=0, pol=0, chan=99)

        error_msg = str(context.exception)
        self.assertIn("Available keys", error_msg)

    def test_error_message_partial_filter(self):
        """Test error message when only some filters specified."""
        with self.assertRaises(KeyError) as context:
            get_peak_residual_from_imaging_dict(self.rd, chan=99)

        error_msg = str(context.exception)
        self.assertIn("chan=99", error_msg)
        # Should NOT include time or pol since they weren't specified
        self.assertNotIn("time=None", error_msg)
        self.assertNotIn("pol=None", error_msg)


class TestNoProgressStop(unittest.TestCase):
    """A plane whose model updates do no iteration any more cannot change its
    residual; with ``threshold = 0`` and ``max_cycles = -1`` nothing else would
    ever stop it, so it is stopped with the no progress code."""

    @staticmethod
    def _model_update(iter_done, peakres=(0.5, 0.0), stop_codes=(None, None)):
        """imaging_dict of one model update for two planes (pol 0 and pol 1)."""
        rd = {}
        for pol, (iters, peak, code) in enumerate(
            zip(iter_done, peakres, stop_codes, strict=True)
        ):
            add_to_imaging_dict(
                rd,
                {
                    "peakres": peak,
                    "masksum": 100,
                    "iter_done": iters,
                    "stop_code": code,
                },
                time=0,
                pol=pol,
                chan=0,
            )
        return rd

    def _controller(self, **kwargs):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            make_cycle_control_state,
        )

        params = dict(max_iter=100, max_cycles=-1, threshold=0.0)
        params.update(kwargs)
        return make_cycle_control_state(**params)

    def test_two_model_updates_without_iterations_stop_the_plane(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_CONTINUE,
            IMAGING_NO_PROGRESS,
            IMAGING_STOP_DESCRIPTIONS,
        )

        controller = self._controller()
        first = self._model_update(iter_done=(10, 0))
        update_counts(controller, first)
        stopcode, _ = check_convergence(controller, first, model_update_ran=True)
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_CONTINUE)

        second = self._model_update(iter_done=(10, 0))
        update_counts(controller, second)
        stopcode, _ = check_convergence(controller, second, model_update_ran=True)
        # pol 0 still cleans, so the imaging cycles go on ...
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        # ... but pol 1 is stopped.
        self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_NO_PROGRESS)
        entry = second[list(second)[1]]
        self.assertEqual(entry["stop_code"].imaging, IMAGING_NO_PROGRESS)
        self.assertEqual(
            entry["stop_description"], IMAGING_STOP_DESCRIPTIONS[IMAGING_NO_PROGRESS]
        )
        self.assertIn("No progress", entry["stop_description"])

    def test_imaging_cycles_end_when_the_other_plane_stops_too(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_MAX_ITER,
            IMAGING_NO_PROGRESS,
        )

        controller = self._controller(max_iter=20)
        for iters in ((10, 0), (10, 0)):
            rd = self._model_update(iter_done=iters)
            update_counts(controller, rd)
            stopcode, description = check_convergence(
                controller, rd, model_update_ran=True
            )
        self.assertNotEqual(stopcode.imaging, 0)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_MAX_ITER)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_NO_PROGRESS)
        self.assertEqual(description, "All planes stopped (mixed reasons)")

    def test_one_model_update_without_iterations_is_tolerated(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_CONTINUE,
        )

        controller = self._controller()
        for iters in ((10, 0), (10, 7), (10, 0), (10, 3)):
            rd = self._model_update(iter_done=iters)
            update_counts(controller, rd)
            stopcode, _ = check_convergence(controller, rd, model_update_ran=True)
            self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
            self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_CONTINUE)
        self.assertEqual(controller["zero_iter_model_updates"][0, 0, 1], 0)

    def test_checks_without_a_model_update_do_not_count(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_CONTINUE,
        )

        controller = self._controller()
        # The residual-only check of every imaging cycle carries iter_done = 0.
        for _ in range(5):
            rd = self._model_update(iter_done=(0, 0))
            stopcode, _ = check_convergence(controller, rd)
            self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertEqual(int(controller["zero_iter_model_updates"].sum()), 0)

    def test_other_stop_codes_take_precedence(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_MAX_CYCLES,
            IMAGING_THRESHOLD,
        )

        controller = self._controller(threshold=0.1, max_cycles=2)
        for _ in range(2):
            rd = self._model_update(iter_done=(0, 0), peakres=(0.5, 0.05))
            update_counts(controller, rd)
            check_convergence(controller, rd, model_update_ran=True)
        # pol 1 is below the threshold; pol 0 has used up max_cycles.
        self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_THRESHOLD)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_MAX_CYCLES)

    def test_reset_clears_the_count(self):
        controller = self._controller()
        for _ in range(2):
            rd = self._model_update(iter_done=(0, 0))
            check_convergence(controller, rd, model_update_ran=True)
        self.assertEqual(int(controller["zero_iter_model_updates"].max()), 2)
        reset(controller)
        self.assertEqual(int(controller["zero_iter_model_updates"].max()), 0)

    def test_model_update_divergence_code_is_kept(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_CONTINUE,
            MODEL_UPDATE_CONTINUE,
            MODEL_UPDATE_DIVERGENCE,
            StopCode,
        )

        controller = self._controller()
        rd = self._model_update(
            iter_done=(40, 12),
            stop_codes=(StopCode(IMAGING_CONTINUE, MODEL_UPDATE_DIVERGENCE), None),
        )
        update_counts(controller, rd)
        stopcode, _ = check_convergence(controller, rd, model_update_ran=True)
        # A diverged model update does not end the imaging cycles.
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertEqual(stopcode.model_update, MODEL_UPDATE_CONTINUE)
        entries = list(rd.values())
        self.assertEqual(entries[0]["stop_code"].model_update, MODEL_UPDATE_DIVERGENCE)
        self.assertEqual(entries[1]["stop_code"].model_update, MODEL_UPDATE_CONTINUE)
        self.assertEqual(
            controller["stop_code_model_update"][0, 0, 0], MODEL_UPDATE_DIVERGENCE
        )


class TestModelUpdateStopCode(unittest.TestCase):
    """The stop code the deconvolver reports for every plane (iteration
    control) reaches the controller and stays until the next model update;
    with ``max_cycles_divergence`` repeated divergence stops the plane (cycle
    control)."""

    _model_update = staticmethod(TestNoProgressStop._model_update)

    @staticmethod
    def _controller(**kwargs):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            make_cycle_control_state,
        )

        params = dict(max_iter=1000, max_cycles=-1, threshold=0.0)
        params.update(kwargs)
        return make_cycle_control_state(**params)

    def _run(self, controller, codes, iter_done=(10, 10)):
        """One model update of the two planes, ending with ``codes``."""
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            StopCode,
        )

        rd = self._model_update(
            iter_done=iter_done,
            stop_codes=tuple(StopCode(IMAGING_CONTINUE, code) for code in codes),
        )
        update_counts(controller, rd)
        stopcode, _ = check_convergence(controller, rd, model_update_ran=True)
        return stopcode, rd

    def test_every_code_reaches_the_controller(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            MODEL_UPDATE_ALGORITHM,
            MODEL_UPDATE_DIVERGENCE,
            MODEL_UPDATE_MAX_ITER_PER_CYCLE,
            MODEL_UPDATE_THRESHOLD_PER_CYCLE,
        )

        controller = self._controller()
        for codes in (
            (MODEL_UPDATE_MAX_ITER_PER_CYCLE, MODEL_UPDATE_THRESHOLD_PER_CYCLE),
            (MODEL_UPDATE_DIVERGENCE, MODEL_UPDATE_ALGORITHM),
        ):
            stopcode, rd = self._run(controller, codes)
            self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
            self.assertEqual(
                controller["stop_code_model_update"][0, 0].tolist(), list(codes)
            )
            self.assertEqual(
                [e["stop_code"].model_update for e in rd.values()], list(codes)
            )

    def test_a_check_on_the_residual_keeps_the_code(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            MODEL_UPDATE_DIVERGENCE,
            MODEL_UPDATE_THRESHOLD_PER_CYCLE,
        )

        codes = [MODEL_UPDATE_THRESHOLD_PER_CYCLE, MODEL_UPDATE_DIVERGENCE]
        controller = self._controller()
        self._run(controller, codes)
        # the residual of the next residual update carries no stop code
        residual = self._model_update(iter_done=(0, 0))
        check_convergence(controller, residual)
        self.assertEqual(controller["stop_code_model_update"][0, 0].tolist(), codes)
        self.assertEqual(
            [e["stop_code"].model_update for e in residual.values()], codes
        )

    def test_a_model_update_that_did_not_run_keeps_the_code(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            MODEL_UPDATE_DIVERGENCE,
            MODEL_UPDATE_MAX_ITER_PER_CYCLE,
        )

        codes = [MODEL_UPDATE_MAX_ITER_PER_CYCLE, MODEL_UPDATE_DIVERGENCE]
        controller = self._controller()
        self._run(controller, codes)
        # both planes given no iterations (e.g. their max_iter is spent while
        # another plane of the channel still cleans)
        _, rd = self._run(
            controller, (MODEL_UPDATE_CONTINUE, MODEL_UPDATE_CONTINUE), (0, 0)
        )
        self.assertEqual(controller["stop_code_model_update"][0, 0].tolist(), codes)
        self.assertEqual([e["stop_code"].model_update for e in rd.values()], codes)
        # nor does it count for, or break, a run of diverged model updates
        self.assertEqual(controller["diverged_model_updates"][0, 0].tolist(), [0, 1])

    def test_repeated_divergence_stops_the_plane(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_DIVERGENCE,
            IMAGING_STOP_DESCRIPTIONS,
            MODEL_UPDATE_DIVERGENCE,
            MODEL_UPDATE_MAX_ITER_PER_CYCLE,
            get_per_cycle_limits,
        )

        diverged = (MODEL_UPDATE_DIVERGENCE, MODEL_UPDATE_MAX_ITER_PER_CYCLE)
        controller = self._controller(max_cycles_divergence=2)
        self._run(controller, diverged)
        self.assertEqual(
            controller["stop_code_imaging"][0, 0].tolist(),
            [IMAGING_CONTINUE, IMAGING_CONTINUE],
        )
        stopcode, rd = self._run(controller, diverged)
        # pol 1 carries on, so the imaging cycles go on
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertEqual(
            controller["stop_code_imaging"][0, 0].tolist(),
            [IMAGING_DIVERGENCE, IMAGING_CONTINUE],
        )
        entry = rd[list(rd)[0]]
        self.assertEqual(
            entry["stop_description"], IMAGING_STOP_DESCRIPTIONS[IMAGING_DIVERGENCE]
        )
        # the plane gets no more iterations ...
        for fields in rd.values():
            fields["max_psf_sidelobe"] = 0.2
        limits, _ = get_per_cycle_limits(controller, rd)
        self.assertEqual(limits[0, 0].tolist(), [0, 980])
        # ... and the stop is final
        self._run(
            controller,
            (MODEL_UPDATE_CONTINUE, MODEL_UPDATE_MAX_ITER_PER_CYCLE),
            (0, 10),
        )
        self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_DIVERGENCE)

    def test_the_divergence_must_be_in_a_row(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            MODEL_UPDATE_DIVERGENCE,
            MODEL_UPDATE_MAX_ITER_PER_CYCLE,
        )

        controller = self._controller(max_cycles_divergence=2)
        for first in (
            MODEL_UPDATE_DIVERGENCE,
            MODEL_UPDATE_MAX_ITER_PER_CYCLE,
            MODEL_UPDATE_DIVERGENCE,
        ):
            stopcode, _ = self._run(
                controller, (first, MODEL_UPDATE_MAX_ITER_PER_CYCLE)
            )
            self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_CONTINUE)
        self.assertEqual(controller["diverged_model_updates"][0, 0, 0], 1)

    def test_off_by_default(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            MODEL_UPDATE_DIVERGENCE,
        )

        controller = self._controller()
        for _ in range(5):
            stopcode, _ = self._run(
                controller, (MODEL_UPDATE_DIVERGENCE, MODEL_UPDATE_DIVERGENCE)
            )
            self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertEqual(controller["max_cycles_divergence"], -1)
        self.assertFalse(controller["divergence_stopped"].any())

    def test_invalid_values_are_refused(self):
        for value in (0, -2, 1.5, True, "2"):
            with self.assertRaises(ValueError):
                self._controller(max_cycles_divergence=value)

    def test_reset_clears_the_divergence_state(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            MODEL_UPDATE_DIVERGENCE,
            MODEL_UPDATE_MAX_ITER_PER_CYCLE,
        )

        controller = self._controller(max_cycles_divergence=1)
        self._run(
            controller, (MODEL_UPDATE_DIVERGENCE, MODEL_UPDATE_MAX_ITER_PER_CYCLE)
        )
        self.assertTrue(controller["divergence_stopped"][0, 0, 0])
        reset(controller)
        self.assertFalse(controller["divergence_stopped"].any())
        self.assertEqual(int(controller["diverged_model_updates"].max()), 0)
        self.assertEqual(get_state(controller)["max_cycles_divergence"], 1)


class TestRefreshedResidualDivergence(unittest.TestCase):
    """A CASA-style threefold rise in an exact residual stops only that plane."""

    @staticmethod
    def _residual(*peaks):
        residual = {}
        for pol, peak in enumerate(peaks):
            add_to_imaging_dict(
                residual,
                {"peakres": peak, "masksum": 16, "iter_done": 0},
                time=0,
                chan=0,
                pol=pol,
            )
        return residual

    @staticmethod
    def _controller():
        return make_cycle_control_state(max_iter=100, max_cycles=-1, threshold=0.0)

    def test_threefold_rise_is_strict_and_per_plane(self):
        controller = self._controller()
        ensure_planes(controller, 1, 1, 2)
        # Cube and MFS controllers pay no storage cost unless the MVC check is
        # explicitly invoked.
        self.assertIsNone(controller["previous_refreshed_peak_residual"])
        self.assertEqual(
            check_refreshed_residual_divergence(controller, self._residual(1.0, 2.0)),
            [],
        )
        # Exactly threefold is permitted; only a larger rise is a runaway.
        self.assertEqual(
            check_refreshed_residual_divergence(controller, self._residual(3.0, 2.5)),
            [],
        )
        stopped = check_refreshed_residual_divergence(
            controller, self._residual(9.1, 2.0)
        )
        self.assertEqual(stopped, [((0, 0, 0), 3.0, 9.1)])

        stopcode, _ = check_convergence(controller, self._residual(9.1, 2.0))
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertEqual(
            controller["stop_code_imaging"][0, 0].tolist(),
            [IMAGING_DIVERGENCE, IMAGING_CONTINUE],
        )

    def test_empty_statistics_do_not_allocate_state(self):
        controller = self._controller()
        self.assertEqual(check_refreshed_residual_divergence(controller, {}), [])
        self.assertIsNone(controller["previous_refreshed_peak_residual"])

    def test_stop_is_final_and_reset_clears_the_baseline(self):
        controller = self._controller()
        check_refreshed_residual_divergence(controller, self._residual(1.0))
        check_refreshed_residual_divergence(controller, self._residual(4.0))
        self.assertTrue(controller["refreshed_residual_divergence_stopped"][0, 0, 0])

        # Later measurements update the diagnostic peak but cannot revoke or
        # duplicate the final divergence stop.
        self.assertEqual(
            check_refreshed_residual_divergence(controller, self._residual(0.1)),
            [],
        )
        reset(controller)
        self.assertFalse(controller["divergence_stopped"].any())
        self.assertFalse(controller["refreshed_residual_divergence_stopped"].any())
        self.assertTrue(np.isnan(controller["previous_refreshed_peak_residual"]).all())

    def test_invalid_growth_factors_are_refused(self):
        controller = self._controller()
        for factor in (1.0, 0.0, np.inf, np.nan, True, "3"):
            with self.assertRaises(ValueError):
                check_refreshed_residual_divergence(
                    controller, self._residual(1.0), factor=factor
                )


class TestRecordModelUpdate(unittest.TestCase):
    """``record_model_update`` books a model update for the stop test after
    the next residual update and decides nothing; ``check_convergence`` with
    ``model_update_ran=True`` is that booking followed by the stop test."""

    _model_update = staticmethod(TestNoProgressStop._model_update)

    @staticmethod
    def _controller():
        return make_cycle_control_state(
            max_iter=100, max_cycles=-1, threshold=0.0, max_cycles_divergence=2
        )

    def test_books_without_deciding(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_DIVERGENCE,
            IMAGING_NO_PROGRESS,
            MODEL_UPDATE_DIVERGENCE,
        )

        diverged = StopCode(IMAGING_CONTINUE, MODEL_UPDATE_DIVERGENCE)
        controller = self._controller()
        for _ in range(2):
            rd = self._model_update(iter_done=(10, 0), stop_codes=(diverged, None))
            update_counts(controller, rd)
            record_model_update(controller, rd)
        self.assertEqual(controller["zero_iter_model_updates"][0, 0].tolist(), [0, 2])
        self.assertEqual(
            controller["stop_code_model_update"][0, 0].tolist(),
            [MODEL_UPDATE_DIVERGENCE, MODEL_UPDATE_CONTINUE],
        )
        self.assertEqual(controller["divergence_stopped"][0, 0].tolist(), [True, False])
        self.assertEqual(controller["stop_code_imaging"][0, 0].tolist(), [0, 0])
        self.assertEqual(controller["stopcode"], StopCode(0, MODEL_UPDATE_CONTINUE))
        self.assertEqual(list(rd.values())[0]["stop_code"], diverged)
        # the stop test on the residual of the next residual update decides
        stopcode, _ = check_convergence(controller, self._model_update((0, 0)))
        self.assertEqual(
            controller["stop_code_imaging"][0, 0].tolist(),
            [IMAGING_DIVERGENCE, IMAGING_NO_PROGRESS],
        )
        self.assertEqual(stopcode.imaging, IMAGING_DIVERGENCE)

    def test_a_check_after_a_model_update_books_then_tests(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_DIVERGENCE,
            IMAGING_NO_PROGRESS,
            MODEL_UPDATE_DIVERGENCE,
            MODEL_UPDATE_MAX_ITER_PER_CYCLE,
            MODEL_UPDATE_THRESHOLD_PER_CYCLE,
        )

        div, mic, tpc = (
            MODEL_UPDATE_DIVERGENCE,
            MODEL_UPDATE_MAX_ITER_PER_CYCLE,
            MODEL_UPDATE_THRESHOLD_PER_CYCLE,
        )
        cont = MODEL_UPDATE_CONTINUE
        nop, dvg = IMAGING_NO_PROGRESS, IMAGING_DIVERGENCE
        # (iter_done, reported codes) -> after booking and testing, per plane:
        # max_iter_remaining, zero_iter_model_updates, stop_code_model_update,
        # diverged_model_updates, divergence_stopped, stop_code_imaging
        sequence = (
            ((10, 0), (div, cont), ([90, 100], [0, 1], [div, cont], [1, 0], 0, [0, 0])),
            (
                (0, 0),
                (mic, cont),
                ([90, 100], [1, 2], [mic, cont], [0, 0], 0, [0, nop]),
            ),
            ((10, 0), (div, tpc), ([80, 100], [0, 3], [div, tpc], [1, 0], 0, [0, nop])),
            (
                (5, 0),
                (div, cont),
                ([75, 100], [0, 4], [div, tpc], [2, 0], 1, [dvg, nop]),
            ),
        )
        controller = self._controller()
        for iter_done, codes, expected in sequence:
            stop_codes = tuple(StopCode(IMAGING_CONTINUE, code) for code in codes)
            rd = self._model_update(iter_done, stop_codes=stop_codes)
            update_counts(controller, rd)
            stopcode, _ = check_convergence(controller, rd, model_update_ran=True)
            remaining, zero_iter, mu_code, diverged, stopped, imaging = expected
            self.assertEqual(controller["max_iter_remaining"][0, 0].tolist(), remaining)
            self.assertEqual(
                controller["zero_iter_model_updates"][0, 0].tolist(), zero_iter
            )
            self.assertEqual(
                controller["stop_code_model_update"][0, 0].tolist(), mu_code
            )
            self.assertEqual(
                controller["diverged_model_updates"][0, 0].tolist(), diverged
            )
            self.assertEqual(
                controller["divergence_stopped"][0, 0].tolist(), [bool(stopped), False]
            )
            self.assertEqual(controller["stop_code_imaging"][0, 0].tolist(), imaging)
            self.assertEqual(
                [fields["stop_code"] for fields in rd.values()],
                [StopCode(i, m) for i, m in zip(imaging, mu_code, strict=True)],
            )
        # every plane has stopped, for different reasons: the largest code
        self.assertEqual(stopcode.imaging, dvg)


class TestPerPlaneStateGrowth(unittest.TestCase):
    def test_arrays_grow_when_a_larger_plane_index_appears(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            make_cycle_control_state,
        )

        controller = make_cycle_control_state(max_iter=100)
        small = {}
        add_to_imaging_dict(
            small,
            {"peakres": 1.0, "masksum": 10, "iter_done": 30},
            time=0,
            pol=0,
            chan=0,
        )
        update_counts(controller, small)
        self.assertEqual(controller["max_iter_remaining"].shape, (1, 1, 1))
        large = {}
        add_to_imaging_dict(
            large,
            {"peakres": 1.0, "masksum": 10, "iter_done": 5},
            time=0,
            pol=1,
            chan=2,
        )
        check_convergence(controller, large, model_update_ran=True)
        self.assertEqual(controller["max_iter_remaining"].shape, (1, 3, 2))
        self.assertEqual(controller["zero_iter_model_updates"].shape, (1, 3, 2))
        # existing values kept, new planes start at the full budget
        self.assertEqual(controller["max_iter_remaining"][0, 0, 0], 70)
        self.assertEqual(controller["max_iter_remaining"][0, 2, 1], 100)
        # the state of the entropy stop grows with them
        self.assertEqual(controller["max_entropy"].shape, (1, 3, 2))
        self.assertEqual(controller["entropy_stopped"].shape, (1, 3, 2))
        self.assertTrue(np.isnan(controller["max_entropy"]).all())
        self.assertFalse(controller["entropy_stopped"].any())


class TestEntropyStop(unittest.TestCase):
    """The entropy of the residual rises while the clean removes emission and
    falls once the clean fits noise. With ``entropy_stop`` a plane is stopped
    when the entropy of its residual is lower than in an earlier imaging
    cycle."""

    @staticmethod
    def _residual(entropy, snr=(3.0, 3.0), peakres=(0.5, 0.4)):
        """imaging_dict of the residual of one residual update, two planes."""
        rd = {}
        for pol in range(2):
            add_to_imaging_dict(
                rd,
                {
                    "peakres": peakres[pol],
                    "masksum": 100,
                    "iter_done": 0,
                    "entropy": entropy[pol],
                    "residual_snr": snr[pol],
                },
                time=0,
                pol=pol,
                chan=0,
            )
        return rd

    @staticmethod
    def _model_update(iter_done):
        rd = {}
        for pol, iters in enumerate(iter_done):
            add_to_imaging_dict(
                rd,
                {"peakres": 0.3, "masksum": 100, "iter_done": iters},
                time=0,
                pol=pol,
                chan=0,
            )
        return rd

    @staticmethod
    def _controller(**kwargs):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            make_cycle_control_state,
        )

        params = dict(max_iter=1000, max_cycles=-1, threshold=0.0, entropy_stop=True)
        params.update(kwargs)
        return make_cycle_control_state(**params)

    def _cycle(self, controller, entropy, iter_done=(50, 50), **residual):
        """One imaging cycle; returns the stop code after the residual update
        and the per-plane budget of the model update that follows."""
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_CONTINUE,
        )

        rd = self._residual(entropy, **residual)
        stopcode, _ = check_convergence(controller, rd)
        if stopcode.imaging != IMAGING_CONTINUE:
            return stopcode, None, rd
        budget = controller["max_iter_remaining"].copy()
        budget[controller["entropy_stopped"]] = 0
        update = self._model_update(np.minimum(iter_done, budget[0, 0]).tolist())
        update_counts(controller, update)
        check_convergence(controller, update, model_update_ran=True)
        return stopcode, budget, update

    def test_a_fall_of_the_entropy_stops_the_plane(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_CONTINUE,
            IMAGING_ENTROPY,
            IMAGING_STOP_DESCRIPTIONS,
        )

        controller = self._controller()
        stopcode, _, _ = self._cycle(controller, (7.50, 7.40))
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        stopcode, _, _ = self._cycle(controller, (7.52, 7.45))
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        np.testing.assert_allclose(controller["max_entropy"][0, 0], [7.52, 7.45])
        # pol 0 falls, pol 1 still rises: the channel cycles on for pol 1
        stopcode, budget, update = self._cycle(controller, (7.51, 7.46))
        self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_ENTROPY)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_CONTINUE)
        self.assertEqual(budget[0, 0].tolist(), [0, 900])
        entries = list(update.values())
        self.assertEqual(entries[0]["stop_code"].imaging, IMAGING_ENTROPY)
        self.assertEqual(
            entries[0]["stop_description"], IMAGING_STOP_DESCRIPTIONS[IMAGING_ENTROPY]
        )
        self.assertIn("entropy", entries[0]["stop_description"])
        # pol 1 falls too: all planes have stopped
        stopcode, _, rd = self._cycle(controller, (7.51, 7.44))
        self.assertEqual(stopcode.imaging, IMAGING_ENTROPY)
        self.assertEqual(controller["stop_code_imaging"][0, 0].tolist(), [10, 10])
        self.assertEqual([e["stop_code"].imaging for e in rd.values()], [10, 10])

    def test_the_stop_is_final(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_ENTROPY,
        )

        controller = self._controller()
        self._cycle(controller, (7.50, 7.40))
        self._cycle(controller, (7.49, 7.45))
        self.assertTrue(controller["entropy_stopped"][0, 0, 0])
        # a higher entropy later on does not revive the plane, and model
        # updates without iterations do not change its stop code
        for entropy in ((7.60, 7.46), (7.70, 7.47), (7.80, 7.48)):
            _, budget, _ = self._cycle(controller, entropy)
            self.assertEqual(budget[0, 0, 0], 0)
            self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_ENTROPY)
        self.assertEqual(controller["max_entropy"][0, 0, 0], 7.50)
        # the plane spent 50 iterations, in the one model update it took part in
        self.assertEqual(controller["max_iter_remaining"][0, 0, 0], 950)

    def test_entropy_that_is_not_a_number_is_skipped(self):
        # image_residual_entropy returns no number while the peak of the
        # residual is above entropy_max_snr times its RMS
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_CONTINUE,
        )

        controller = self._controller()
        for entropy in ((np.nan, 7.2), (np.nan, 7.3), (7.5, 7.4), (7.6, 7.5)):
            stopcode, _, _ = self._cycle(controller, entropy)
            self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        np.testing.assert_allclose(controller["max_entropy"][0, 0], [7.6, 7.5])
        self.assertFalse(controller["entropy_stopped"].any())

    def test_equal_entropy_does_not_stop(self):
        controller = self._controller()
        for _ in range(3):
            self._cycle(controller, (7.5, 7.5))
        self.assertFalse(controller["entropy_stopped"].any())

    def test_off_by_default(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_CONTINUE,
        )

        controller = self._controller(entropy_stop=False)
        for entropy in ((7.5, 7.5), (7.4, 7.4), (7.3, 7.3)):
            stopcode, budget, _ = self._cycle(controller, entropy)
            self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
        self.assertFalse(controller["entropy_stopped"].any())
        self.assertTrue(np.isnan(controller["max_entropy"]).all())
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            make_cycle_control_state,
        )

        self.assertFalse(make_cycle_control_state()["entropy_stop"])

    def test_threshold_and_iteration_limit_come_first(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_ENTROPY,
            IMAGING_MAX_ITER,
            IMAGING_THRESHOLD,
        )

        controller = self._controller(threshold=0.45, max_iter=100)
        self._cycle(controller, (7.5, 7.5), iter_done=(100, 10))
        # pol 0 has spent its budget, pol 1 is below the threshold; both fall
        rd = self._residual((7.4, 7.4), peakres=(0.5, 0.4))
        check_convergence(controller, rd)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_MAX_ITER)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_THRESHOLD)
        self.assertTrue(controller["entropy_stopped"].all())
        self.assertNotEqual(IMAGING_ENTROPY, IMAGING_MAX_ITER)

    def test_entropy_comes_before_the_cycle_limit(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            IMAGING_ENTROPY,
            IMAGING_MAX_CYCLES,
        )

        controller = self._controller(max_cycles=2)
        self._cycle(controller, (7.5, 7.4))
        self._cycle(controller, (7.4, 7.5))  # the last cycle allowed
        rd = self._residual((7.4, 7.6))
        stopcode, _ = check_convergence(controller, rd)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_ENTROPY)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_MAX_CYCLES)
        self.assertEqual(stopcode.imaging, IMAGING_ENTROPY)  # highest of the two

    def test_reset_clears_the_state(self):
        controller = self._controller()
        self._cycle(controller, (7.5, 7.4))
        self._cycle(controller, (7.4, 7.5))
        self.assertTrue(controller["entropy_stopped"][0, 0, 0])
        reset_stopcode(controller)
        self.assertFalse(controller["entropy_stopped"].any())
        self.assertTrue(np.isnan(controller["max_entropy"]).all())
        self._cycle(controller, (7.5, 7.4))
        self._cycle(controller, (7.4, 7.5))
        reset(controller)
        self.assertFalse(controller["entropy_stopped"].any())
        self.assertTrue(np.isnan(controller["max_entropy"]).all())

    def test_state_lists_the_parameters(self):
        state = get_state(
            self._controller(
                entropy_max_snr=5.0, entropy_spatial_bins=9, entropy_flux_bins=20
            )
        )
        self.assertIs(state["entropy_stop"], True)
        self.assertEqual(state["entropy_max_snr"], 5.0)
        self.assertEqual(state["entropy_spatial_bins"], 9)
        self.assertEqual(state["entropy_flux_bins"], 20)

    def test_per_cycle_limits_give_a_stopped_plane_no_iterations(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            get_per_cycle_limits,
        )

        controller = self._controller(max_iter_per_cycle=200)
        self._cycle(controller, (7.5, 7.4))
        _, _, update = self._cycle(controller, (7.4, 7.5))
        for fields in update.values():
            fields["max_psf_sidelobe"] = 0.2
        max_iter_per_cycle_updated, _ = get_per_cycle_limits(controller, update)
        self.assertEqual(max_iter_per_cycle_updated[0, 0].tolist(), [0, 200])
        # the budget of the controller is not touched
        self.assertEqual(controller["max_iter_remaining"][0, 0].tolist(), [950, 900])


class TestEntropyParameters(unittest.TestCase):
    def test_defaults(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            validate_entropy_params,
        )

        self.assertEqual(validate_entropy_params(None), (False, 6.0, 7, 10))
        self.assertEqual(validate_entropy_params({"gain": 0.1}), (False, 6.0, 7, 10))

    def test_values_are_taken_over(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            validate_entropy_params,
        )

        out = validate_entropy_params(
            {
                "entropy_stop": True,
                "entropy_max_snr": 5,
                "entropy_spatial_bins": np.int64(9),
                "entropy_flux_bins": 20,
            }
        )
        self.assertEqual(out, (True, 5.0, 9, 20))
        self.assertIsInstance(out[1], float)
        self.assertIsInstance(out[2], int)

    def test_wrong_values_are_refused(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            make_cycle_control_state,
            validate_entropy_params,
        )

        for key, values in {
            "entropy_stop": [1, 0, "True", None],
            "entropy_max_snr": [0, -1.0, np.nan, np.inf, "6", True, None],
            "entropy_spatial_bins": [0, -3, 7.0, "7", True, None],
            "entropy_flux_bins": [0, -3, 10.0, "10", False, None],
        }.items():
            for value in values:
                with self.assertRaisesRegex(ValueError, key):
                    validate_entropy_params({key: value})
                with self.assertRaisesRegex(ValueError, key):
                    make_cycle_control_state(**{key: value})


class TestResidualEntropyInImagingDict(unittest.TestCase):
    @staticmethod
    def _img_xds(seed=0, n=64):
        rng = np.random.default_rng(seed)
        residual = rng.standard_normal((1, 2, 2, n, n))
        residual[0, 1, 1, 10, 10] = 50.0  # peak far above 6 times the RMS
        img_xds = xr.Dataset(
            {
                "SKY_RESIDUAL": (
                    ("time", "frequency", "polarization", "l", "m"),
                    residual,
                ),
                "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                    ("time", "frequency", "polarization"),
                    np.full((1, 2, 2), 0.2),
                ),
            }
        )
        img_xds.attrs["data_groups"] = {
            "residual": {
                "sky": "SKY_RESIDUAL",
                "max_sidelobe_point_spread_function": (
                    "MAX_SIDELOBE_POINT_SPREAD_FUNCTION"
                ),
            }
        }
        return img_xds

    def test_entropy_is_recorded_with_and_without_entropy_stop(self):
        img_xds = self._img_xds()
        off = build_residual_imaging_dict(img_xds, "residual", {"gain": 0.1})
        on = build_residual_imaging_dict(
            img_xds, "residual", {"gain": 0.1, "entropy_stop": True}
        )
        for key, fields in off.items():
            np.testing.assert_equal(fields["entropy"], on[key]["entropy"])
        self.assertEqual(len(on), 4)
        for key, fields in on.items():
            self.assertEqual(len(fields["entropy"]), 1)
            self.assertEqual(len(fields["residual_snr"]), 1)
            if (key.chan, key.pol) == (1, 1):
                self.assertTrue(np.isnan(fields["entropy"][0]))
                self.assertGreater(fields["residual_snr"][0], 30)
            else:
                # white noise in 49 spatial bins: close to ln(49) + 3.72
                self.assertAlmostEqual(fields["entropy"][0], 7.3, delta=0.15)
                self.assertLess(fields["residual_snr"][0], 6)

    def test_parameters_reach_the_entropy(self):
        img_xds = self._img_xds()
        coarse = build_residual_imaging_dict(
            img_xds,
            "residual",
            {
                "gain": 0.1,
                "entropy_stop": True,
                "entropy_spatial_bins": 1,
                "entropy_flux_bins": 2,
            },
        )
        fine = build_residual_imaging_dict(
            img_xds, "residual", {"gain": 0.1, "entropy_stop": True}
        )
        key = Key(time=0, chan=0, pol=0)
        self.assertLess(coarse[key]["entropy"][0], 2.5)
        self.assertGreater(fine[key]["entropy"][0], 7.0)
        # a lower limit on the ratio of peak to RMS leaves the noise planes out
        strict = build_residual_imaging_dict(
            img_xds,
            "residual",
            {"gain": 0.1, "entropy_stop": True, "entropy_max_snr": 2.0},
        )
        self.assertTrue(np.isnan(strict[key]["entropy"][0]))

    def test_wrong_parameter_is_refused(self):
        with self.assertRaisesRegex(ValueError, "entropy_stop"):
            build_residual_imaging_dict(
                self._img_xds(), "residual", {"gain": 0.1, "entropy_stop": "yes"}
            )

    def test_copy_into_the_result_of_a_model_update(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            copy_residual_statistics,
            merge_imaging_dicts,
        )

        img_xds = self._img_xds()
        residual = build_residual_imaging_dict(
            img_xds, "residual", {"gain": 0.1, "entropy_stop": True}
        )
        update = {}
        for key in residual:
            add_to_imaging_dict(
                update,
                {"peakres": 0.1, "masksum": 10, "iter_done": 5},
                time=key.time,
                pol=key.pol,
                chan=key.chan,
            )
        copy_residual_statistics(update, residual)
        key = Key(time=0, chan=0, pol=1)
        self.assertEqual(update[key]["entropy"], residual[key]["entropy"])
        self.assertEqual(update[key]["residual_snr"], residual[key]["residual_snr"])
        # the statistics of the residual the model update started from
        for field in RESIDUAL_STATISTICS:
            self.assertEqual(update[key][f"start_{field}"], residual[key][field])
        # one value per imaging cycle in the merged record
        merged = merge_imaging_dicts([update, update])
        self.assertEqual(len(merged[key]["entropy"]), 2)
        self.assertEqual(len(merged[key]["start_rms_in_mask"]), 2)

    def test_residual_statistics_of_every_plane(self):
        from astroviper.processing_functions.image_analysis.image_statistics import (
            plane_residual_statistics,
        )

        img_xds = self._img_xds()
        record = build_residual_imaging_dict(img_xds, "residual", {"gain": 0.1})
        for key, fields in record.items():
            plane = img_xds["SKY_RESIDUAL"].values[key.time, key.chan, key.pol]
            for field, value in plane_residual_statistics(plane).items():
                np.testing.assert_equal(fields[field], [value])
        self.assertEqual(record[Key(time=0, chan=1, pol=1)]["max_in_mask"], [50.0])

    def test_record_cycle_time(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            record_cycle_time,
        )

        record = build_residual_imaging_dict(self._img_xds(), "residual", {"gain": 0.1})
        record_cycle_time(record, "t_residual_update", 1.5)
        for fields in record.values():
            self.assertEqual(fields["t_residual_update"], [1.5])


if __name__ == "__main__":
    unittest.main()


class TestPerStokesThreshold(unittest.TestCase):
    """``threshold`` with one value per polarization (Stokes) plane."""

    @staticmethod
    def _planes(peaks):
        rd = {}
        for pol, peak in enumerate(peaks):
            add_to_imaging_dict(
                rd,
                {"peakres": peak, "masksum": 100, "max_psf_sidelobe": 0.2},
                time=0,
                pol=pol,
                chan=0,
            )
        return rd

    def test_each_plane_stops_at_its_own_threshold(self):
        controller = make_cycle_control_state(max_iter=1000, threshold=[1e-3, 1e-4])
        check_convergence(controller, self._planes([5e-4, 5e-4]))
        self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_THRESHOLD)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_CONTINUE)

    def test_threshold_per_cycle_is_floored_per_plane(self):
        controller = make_cycle_control_state(
            max_iter=1000, threshold=[1e-3, 1e-4], psf_sidelobe_factor=1.0
        )
        ensure_planes(controller, 1, 1, 2)
        per_plane = per_plane_threshold_per_cycle(
            controller, self._planes([2e-3, 2e-3])
        )
        np.testing.assert_allclose(per_plane[0, 0], [1e-3, 0.2 * 2e-3])

    def test_one_value_applies_to_every_plane(self):
        controller = make_cycle_control_state(max_iter=1000, threshold=1e-3)
        check_convergence(controller, self._planes([5e-4, 2e-3]))
        self.assertEqual(controller["stop_code_imaging"][0, 0, 0], IMAGING_THRESHOLD)
        self.assertEqual(controller["stop_code_imaging"][0, 0, 1], IMAGING_CONTINUE)

    def test_one_value_per_plane_is_required(self):
        controller = make_cycle_control_state(
            max_iter=1000, threshold=[1e-3, 1e-4, 1e-5]
        )
        with self.assertRaisesRegex(ValueError, "polarization planes"):
            ensure_planes(controller, 1, 1, 2)

    def test_negative_value_is_refused(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            make_cycle_control_state(max_iter=1000, threshold=[1e-3, -1e-4])

    def test_state_lists_the_thresholds(self):
        controller = make_cycle_control_state(max_iter=1000, threshold=[1e-3, 1e-4])
        self.assertEqual(get_state(controller)["threshold"], [1e-3, 1e-4])


class TestThresholdTolerance(unittest.TestCase):
    """``threshold_tolerance``: the threshold stop at ``threshold * (1 + tau)``."""

    THRESHOLDS = (1e-3, 2.5e-4, 0.0)

    @classmethod
    def _peaks(cls):
        peaks = [0.0, 1e-9, 0.5, 2.0]
        for value in cls.THRESHOLDS:
            peaks += [
                value,
                np.nextafter(value, 0.0),
                np.nextafter(value, np.inf),
                value * 1.01,
                value * 1.02,
                value * 0.99,
            ]
        return peaks

    @staticmethod
    def _record(peaks, npol):
        rd = {}
        for chan, peak in enumerate(peaks):
            for pol in range(npol):
                add_to_imaging_dict(
                    rd,
                    {"peakres": float(peak), "masksum": 100, "max_psf_sidelobe": 0.2},
                    time=0,
                    pol=pol,
                    chan=chan,
                )
        return rd

    def _codes(self, threshold, peaks, **kwargs):
        controller = make_cycle_control_state(
            max_iter=1000, threshold=threshold, **kwargs
        )
        check_convergence(controller, self._record(peaks, np.size(threshold)))
        return controller["stop_code_imaging"][0]

    def test_zero_reproduces_the_threshold_stop(self):
        """``0`` gives the stop codes of ``peak <= threshold``."""
        peaks = self._peaks()
        thresholds = [*self.THRESHOLDS, list(self.THRESHOLDS)]
        for threshold in thresholds:
            planes = np.broadcast_to(threshold, (len(peaks), np.size(threshold)))
            expected = np.where(
                (planes > 0) & (np.asarray(peaks)[:, None] <= planes),
                IMAGING_THRESHOLD,
                IMAGING_CONTINUE,
            )
            for kwargs in ({"threshold_tolerance": 0.0}, {"threshold_tolerance": 0}):
                np.testing.assert_array_equal(
                    self._codes(threshold, peaks, **kwargs), expected
                )

    def test_a_peak_within_the_tolerance_stops_the_plane(self):
        threshold, tau = 1e-3, 0.02
        limit = threshold * (1 + tau)
        peaks = [threshold, 1.01 * threshold, limit, np.nextafter(limit, np.inf)]
        np.testing.assert_array_equal(
            self._codes(threshold, peaks, threshold_tolerance=tau)[:, 0],
            [IMAGING_THRESHOLD] * 3 + [IMAGING_CONTINUE],
        )
        np.testing.assert_array_equal(
            self._codes(threshold, peaks, threshold_tolerance=0)[:, 0],
            [IMAGING_THRESHOLD] + [IMAGING_CONTINUE] * 3,
        )

    def test_default_is_the_heuristic(self):
        self.assertEqual(
            make_cycle_control_state()["threshold_tolerance"], THRESHOLD_TOLERANCE
        )
        threshold = 1e-3
        limit = threshold * (1 + THRESHOLD_TOLERANCE)
        peaks = [1.01 * threshold, np.nextafter(limit, np.inf)]
        np.testing.assert_array_equal(
            self._codes(threshold, peaks)[:, 0], [IMAGING_THRESHOLD, IMAGING_CONTINUE]
        )

    def test_per_plane_threshold_and_disabled_threshold(self):
        codes = self._codes(
            [1e-3, 1e-4, 0.0], [1.01e-3, 1.01e-4, 0.0], threshold_tolerance=0.02
        )
        np.testing.assert_array_equal(
            np.diag(codes), [IMAGING_THRESHOLD, IMAGING_THRESHOLD, IMAGING_CONTINUE]
        )

    def test_threshold_per_cycle_is_unchanged(self):
        rd = self._record([1.01e-3, 2e-3, 1.0], 1)
        per_plane = []
        for tau in (0.0, 0.02):
            controller = make_cycle_control_state(
                max_iter=1000, threshold=1e-3, threshold_tolerance=tau
            )
            ensure_planes(controller, 1, 3, 1)
            per_plane.append(per_plane_threshold_per_cycle(controller, rd))
        np.testing.assert_array_equal(per_plane[0], per_plane[1])

    def test_a_stopped_plane_gets_no_iterations(self):
        """Stokes I within the tolerance, Q above it and U below the
        threshold: with a tolerance I and U get no iterations while Q goes
        on; without one I goes on and U keeps its budget, which the
        deconvolver does not spend (its peak is below threshold_per_cycle)."""
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            get_per_cycle_limits,
        )

        threshold = 1e-3
        rd = {}
        for pol, peak in enumerate((1.01 * threshold, 5 * threshold, 0.5 * threshold)):
            add_to_imaging_dict(
                rd,
                {"peakres": peak, "masksum": 100, "max_psf_sidelobe": 0.2},
                time=0,
                pol=pol,
                chan=0,
            )
        for tau, codes, limits in (
            (0.0, [IMAGING_CONTINUE, IMAGING_CONTINUE, IMAGING_THRESHOLD], [800] * 3),
            (
                0.02,
                [IMAGING_THRESHOLD, IMAGING_CONTINUE, IMAGING_THRESHOLD],
                [0, 800, 0],
            ),
        ):
            controller = make_cycle_control_state(
                max_iter=800, threshold=[threshold] * 3, threshold_tolerance=tau
            )
            stopcode, _ = check_convergence(controller, rd)
            self.assertEqual(stopcode.imaging, IMAGING_CONTINUE)
            self.assertEqual(controller["stop_code_imaging"][0, 0].tolist(), codes)
            max_iter_per_cycle_updated, threshold_per_cycle = get_per_cycle_limits(
                controller, rd
            )
            self.assertEqual(max_iter_per_cycle_updated[0, 0].tolist(), limits)
            # the peak of I lies above its threshold_per_cycle
            self.assertLess(threshold_per_cycle[0, 0, 0], 1.01 * threshold)

    def test_invalid_values_are_refused(self):
        for value in (-0.01, -1, np.nan, np.inf, True, "0.02", None):
            with self.assertRaisesRegex(ValueError, "threshold_tolerance"):
                make_cycle_control_state(threshold=1e-3, threshold_tolerance=value)

    def test_state_lists_the_tolerance(self):
        self.assertEqual(
            get_state(make_cycle_control_state())["threshold_tolerance"],
            THRESHOLD_TOLERANCE,
        )
        controller = make_cycle_control_state(threshold_tolerance=np.float32(0.25))
        self.assertEqual(get_state(controller)["threshold_tolerance"], 0.25)
        self.assertIs(type(controller["threshold_tolerance"]), float)


def test_plot_convergence_history_of_two_cycles():
    """The plot of a plane is built from a two-cycle imaging_dict, without a
    display."""
    hv = pytest.importorskip("holoviews")
    pytest.importorskip("bokeh")
    rd = {}
    for start_peakres, peakres, start_model_flux, model_flux in (
        (1.0, 0.5, 0.0, 0.5),
        (0.5, 0.2, 0.5, 0.8),
    ):
        add_to_imaging_dict(
            rd,
            {
                "start_peakres": start_peakres,
                "peakres": peakres,
                "iter_done": 100,
                "start_model_flux": start_model_flux,
                "model_flux": model_flux,
            },
            time=0,
            pol=0,
            chan=0,
        )

    plot = plot_convergence_history(rd)

    assert isinstance(plot, hv.DynamicMap)
    assert [(kdim.name, kdim.values) for kdim in plot.kdims] == [
        ("Stokes", ["I"]),
        ("Channel", [0]),
    ]
    peakres_curve, model_flux_curve = plot["I", 0]
    np.testing.assert_array_equal(peakres_curve.dimension_values(0), [0, 100, 200])
    np.testing.assert_array_equal(peakres_curve.dimension_values(1), [1.0, 0.5, 0.2])
    np.testing.assert_array_equal(model_flux_curve.dimension_values(0), [0, 100, 200])
    np.testing.assert_array_equal(model_flux_curve.dimension_values(1), [0.0, 0.5, 0.8])
