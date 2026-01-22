"""
Unit tests for the iteration control module.

Tests cover:
- ReturnDict core functionality (initialization, add, sel, to_dict, repr)
- ReturnDict utility functions (merge, extraction)
- IterationController initialization and parameter validation
- Adaptive cyclethreshold calculation
- Convergence checking with all stop codes
- Count updates and state management
- Threshold string parsing
- Full major cycle workflows
"""

import unittest
import numpy as np
from astroviper.core.imaging.imaging_utils.iteration_control import (
    # Stop codes
    StopCode,
    MAJOR_CONTINUE,
    MAJOR_ITER_LIMIT,
    MAJOR_THRESHOLD,
    MAJOR_ZERO_MASK,
    MAJOR_CYCLE_LIMIT,
    MINOR_CONTINUE,
    MINOR_ITER_LIMIT,
    MINOR_THRESHOLD,
    # Utility functions
    merge_return_dicts,
    get_peak_residual_from_returndict,
    get_masksum_from_returndict,
    get_iterations_done_from_returndict,
    get_max_psf_sidelobe_from_returndict,
    # Main class
    IterationController,
)
from astroviper.core.imaging.imaging_utils.return_dict import (
    ReturnDict,
    FIELD_ACCUM,
    FIELD_SINGLE_VALUE,
    Key,
)

# =============================================================================
# ReturnDict Core Functionality Tests
# =============================================================================


class TestReturnDictBasics(unittest.TestCase):
    """Test ReturnDict initialization, properties, and basic methods."""

    def test_empty_initialization(self):
        """Test creating an empty ReturnDict."""
        rd = ReturnDict()
        self.assertIsNotNone(rd)
        self.assertEqual(len(rd.data), 0)
        self.assertIsInstance(rd.data, dict)

    def test_data_property_getter(self):
        """Test data property getter returns internal dictionary."""
        rd = ReturnDict()
        rd.add({"peakres": 1.0}, time=0, pol=0, chan=0)

        data = rd.data
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 1)

        # Verify the key structure
        key = Key(0, 0, 0)
        self.assertIn(key, data)

    def test_data_property_setter(self):
        """Test data property setter updates internal dictionary."""
        rd = ReturnDict()

        # Create new data dict
        new_data = {Key(0, 0, 0): {"peakres": [2.5], "iter_done": [50]}}
        rd.data = new_data

        self.assertEqual(len(rd.data), 1)
        entry = rd.sel(time=0, pol=0, chan=0)
        self.assertEqual(entry["peakres"], [2.5])
        self.assertEqual(entry["iter_done"], [50])

    def test_to_dict_method(self):
        """Test to_dict() returns the internal data dictionary."""
        rd = ReturnDict()
        rd.add({"peakres": 1.5, "iter_done": 100}, time=0, pol=0, chan=0)
        rd.add({"peakres": 1.2, "iter_done": 80}, time=0, pol=0, chan=1)

        result = rd.to_dict()

        # Should return the same object as .data
        self.assertIs(result, rd.data)
        self.assertEqual(len(result), 2)

    def test_repr_empty(self):
        """Test __repr__() for empty ReturnDict."""
        rd = ReturnDict()
        repr_str = repr(rd)

        # Empty dict should produce empty string (no entries)
        self.assertEqual(repr_str, "")

    def test_repr_single_entry(self):
        """Test __repr__() for single entry."""
        rd = ReturnDict()
        rd.add({"peakres": 1.0, "iter_done": 50}, time=0, pol=0, chan=0)

        repr_str = repr(rd)

        # Should contain the key and values
        self.assertIn("Key(time=0, pol=0, chan=0)", repr_str)
        self.assertIn("peakres", repr_str)
        self.assertIn("[1.0]", repr_str)  # FIELD_ACCUM stored as list
        self.assertIn("iter_done", repr_str)
        self.assertIn("[50]", repr_str)  # FIELD_ACCUM stored as list

    def test_repr_multiple_entries(self):
        """Test __repr__() for multiple entries."""
        rd = ReturnDict()
        rd.add({"peakres": 1.0}, time=0, pol=0, chan=0)
        rd.add({"peakres": 0.8}, time=0, pol=0, chan=1)
        rd.add({"peakres": 0.6}, time=0, pol=1, chan=0)

        repr_str = repr(rd)

        # Should contain all keys
        self.assertIn("Key(time=0, pol=0, chan=0)", repr_str)
        self.assertIn("Key(time=0, pol=0, chan=1)", repr_str)
        self.assertIn("Key(time=0, pol=1, chan=0)", repr_str)

        # Should have multiple lines (newline separated)
        lines = repr_str.split("\n")
        self.assertEqual(len(lines), 3)


class TestReturnDictAdd(unittest.TestCase):
    """Test ReturnDict.add() method edge cases and history tracking."""

    def test_add_first_entry_field_accum(self):
        """Test adding first entry with FIELD_ACCUM fields."""
        rd = ReturnDict()
        rd.add({"peakres": 1.5, "iter_done": 100}, time=0, pol=0, chan=0)

        entry = rd.sel(time=0, pol=0, chan=0)

        # FIELD_ACCUM fields should be stored as lists
        self.assertEqual(entry["peakres"], [1.5])
        self.assertEqual(entry["iter_done"], [100])
        self.assertIsInstance(entry["peakres"], list)
        self.assertIsInstance(entry["iter_done"], list)

    def test_add_first_entry_field_single_value(self):
        """Test adding first entry with FIELD_SINGLE_VALUE fields."""
        rd = ReturnDict()
        rd.add({"max_psf_sidelobe": 0.2, "loop_gain": 0.1}, time=0, pol=0, chan=0)

        entry = rd.sel(time=0, pol=0, chan=0)

        # FIELD_SINGLE_VALUE fields should be stored as scalars
        self.assertEqual(entry["max_psf_sidelobe"], 0.2)
        self.assertEqual(entry["loop_gain"], 0.1)
        self.assertNotIsInstance(entry["max_psf_sidelobe"], list)
        self.assertNotIsInstance(entry["loop_gain"], list)

    def test_add_first_entry_mixed_fields(self):
        """Test adding first entry with both FIELD_ACCUM and FIELD_SINGLE_VALUE."""
        rd = ReturnDict()
        rd.add(
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

        entry = rd.sel(time=0, pol=0, chan=0)

        # Check FIELD_ACCUM are lists
        self.assertEqual(entry["peakres"], [1.0])
        self.assertEqual(entry["iter_done"], [50])

        # Check FIELD_SINGLE_VALUE are scalars
        self.assertEqual(entry["max_psf_sidelobe"], 0.15)
        self.assertEqual(entry["threshold"], 0.01)

    def test_add_unknown_field(self):
        """Test adding a field not in FIELD_ACCUM or FIELD_SINGLE_VALUE."""
        rd = ReturnDict()
        rd.add({"custom_field": 42, "peakres": 1.0}, time=0, pol=0, chan=0)

        entry = rd.sel(time=0, pol=0, chan=0)

        # Unknown fields should be treated as FIELD_SINGLE_VALUE (not accumulated)
        self.assertEqual(entry["custom_field"], 42)
        self.assertNotIsInstance(entry["custom_field"], list)

        # Known FIELD_ACCUM should still be a list
        self.assertEqual(entry["peakres"], [1.0])

    def test_add_empty_dict(self):
        """Test adding an empty dictionary."""
        rd = ReturnDict()
        rd.add({}, time=0, pol=0, chan=0)

        # Key should exist but with empty entry
        entry = rd.sel(time=0, pol=0, chan=0)
        self.assertIsNotNone(entry)
        self.assertEqual(entry, {})

    def test_backward_compat_scalar_to_list(self):
        """Test backward compatibility: scalar converted to list on second add."""
        rd = ReturnDict()

        # Manually create a scalar entry (simulating old data format)
        key = Key(0, 0, 0)
        rd.data[key] = {"peakres": 1.0}  # Scalar instead of list

        # Now add new value - should convert to list and append
        rd.add({"peakres": 0.8}, time=0, pol=0, chan=0)

        entry = rd.sel(time=0, pol=0, chan=0)

        # Should have converted scalar to list and appended new value
        self.assertEqual(entry["peakres"], [1.0, 0.8])
        self.assertIsInstance(entry["peakres"], list)

    def test_multiple_adds_same_key_accumulate(self):
        """Test multiple adds to same key accumulate FIELD_ACCUM values."""
        rd = ReturnDict()

        # First add
        rd.add({"peakres": 1.0, "iter_done": 100}, time=0, pol=0, chan=0)

        # Second add
        rd.add({"peakres": 0.8, "iter_done": 80}, time=0, pol=0, chan=0)

        # Third add
        rd.add({"peakres": 0.6, "iter_done": 60}, time=0, pol=0, chan=0)

        entry = rd.sel(time=0, pol=0, chan=0)

        # FIELD_ACCUM should accumulate in lists
        self.assertEqual(entry["peakres"], [1.0, 0.8, 0.6])
        self.assertEqual(entry["iter_done"], [100, 80, 60])

    def test_multiple_adds_same_key_replace_single_value(self):
        """Test multiple adds to same key replace FIELD_SINGLE_VALUE values."""
        rd = ReturnDict()

        # First add
        rd.add({"max_psf_sidelobe": 0.2, "threshold": 0.01}, time=0, pol=0, chan=0)

        # Second add with different values
        rd.add({"max_psf_sidelobe": 0.15, "threshold": 0.005}, time=0, pol=0, chan=0)

        entry = rd.sel(time=0, pol=0, chan=0)

        # FIELD_SINGLE_VALUE should be replaced (last value wins)
        self.assertEqual(entry["max_psf_sidelobe"], 0.15)
        self.assertEqual(entry["threshold"], 0.005)

    def test_add_partial_fields_to_existing_key(self):
        """Test adding partial fields to an existing key."""
        rd = ReturnDict()

        # First add with some fields
        rd.add({"peakres": 1.0, "iter_done": 100}, time=0, pol=0, chan=0)

        # Second add with different fields
        rd.add({"masksum": 500, "max_psf_sidelobe": 0.2}, time=0, pol=0, chan=0)

        entry = rd.sel(time=0, pol=0, chan=0)

        # First fields should still exist
        self.assertEqual(entry["peakres"], [1.0])
        self.assertEqual(entry["iter_done"], [100])

        # New fields should be added
        self.assertEqual(entry["masksum"], [500])  # FIELD_ACCUM
        self.assertEqual(entry["max_psf_sidelobe"], 0.2)  # FIELD_SINGLE_VALUE


class TestReturnDictSel(unittest.TestCase):
    """Test ReturnDict.sel() method edge cases and filtering."""

    def setUp(self):
        """Create test ReturnDict with multiple entries."""
        self.rd = ReturnDict()
        self.rd.add({"peakres": 1.0}, time=0, pol=0, chan=0)
        self.rd.add({"peakres": 0.9}, time=0, pol=0, chan=1)
        self.rd.add({"peakres": 0.8}, time=0, pol=1, chan=0)
        self.rd.add({"peakres": 0.7}, time=1, pol=0, chan=0)

    def test_sel_no_matches_returns_none(self):
        """Test sel() returns None when no matches found."""
        result = self.rd.sel(time=99, pol=0, chan=0)
        self.assertIsNone(result)

        result = self.rd.sel(time=0, pol=99, chan=0)
        self.assertIsNone(result)

        result = self.rd.sel(time=0, pol=0, chan=99)
        self.assertIsNone(result)

    def test_sel_single_match_returns_dict(self):
        """Test sel() returns single dict (not list) when one match."""
        result = self.rd.sel(time=0, pol=0, chan=0)

        # Should return a dict, not a list
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, list)
        self.assertEqual(result["peakres"], [1.0])

    def test_sel_multiple_matches_returns_list(self):
        """Test sel() returns list when multiple matches."""
        # Filter by time=0 only (should match 3 entries)
        result = self.rd.sel(time=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

        # Verify all are dicts
        for entry in result:
            self.assertIsInstance(entry, dict)
            self.assertIn("peakres", entry)

    def test_sel_all_none_returns_all(self):
        """Test sel() with all None filters returns all entries (or empty list)."""
        result = self.rd.sel()

        # With all None, it should return a list of all entries
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)

    def test_sel_empty_returndict(self):
        """Test sel() on empty ReturnDict returns None."""
        rd_empty = ReturnDict()
        result = rd_empty.sel(time=0, pol=0, chan=0)

        self.assertIsNone(result)

    def test_sel_filter_by_time_only(self):
        """Test filtering by time parameter only."""
        result = self.rd.sel(time=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 3 entries with time=0

    def test_sel_filter_by_pol_only(self):
        """Test filtering by pol parameter only."""
        result = self.rd.sel(pol=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 3 entries with pol=0

    def test_sel_filter_by_chan_only(self):
        """Test filtering by chan parameter only."""
        result = self.rd.sel(chan=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 3 entries with chan=0

    def test_sel_combined_filters(self):
        """Test filtering with multiple parameters."""
        result = self.rd.sel(time=0, pol=0)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # time=0, pol=0 matches 2 entries

    def test_sel_preserves_order(self):
        """Test sel() preserves insertion order (OrderedDict behavior)."""
        rd = ReturnDict()

        # Add in specific order
        rd.add({"peakres": 1.0}, time=0, pol=0, chan=0)
        rd.add({"peakres": 0.9}, time=0, pol=0, chan=1)
        rd.add({"peakres": 0.8}, time=0, pol=0, chan=2)

        result = rd.sel(time=0, pol=0)

        # Should return in insertion order
        self.assertEqual(result[0]["peakres"], [1.0])
        self.assertEqual(result[1]["peakres"], [0.9])
        self.assertEqual(result[2]["peakres"], [0.8])

    def test_sel_nonexistent_values(self):
        """Test sel() with values that don't exist in any entry."""
        result = self.rd.sel(time=999, pol=999, chan=999)
        self.assertIsNone(result)


class TestReturnDictFieldClassification(unittest.TestCase):
    """Test FIELD_ACCUM and FIELD_SINGLE_VALUE classification behavior."""

    def test_field_accum_constants(self):
        """Test FIELD_ACCUM contains expected fields."""
        expected_fields = {
            "peakres",
            "peakres_nomask",
            "iter_done",
            "masksum",
            "model_flux",
        }
        self.assertEqual(FIELD_ACCUM, expected_fields)

    def test_field_single_value_constants(self):
        """Test FIELD_SINGLE_VALUE contains expected fields."""
        expected_fields = {"max_psf_sidelobe", "loop_gain", "niter", "threshold"}
        self.assertEqual(FIELD_SINGLE_VALUE, expected_fields)

    def test_all_field_accum_become_lists(self):
        """Test all FIELD_ACCUM fields are stored as lists."""
        rd = ReturnDict()

        # Add all FIELD_ACCUM fields
        rd.add(
            {
                "peakres": 1.0,
                "peakres_nomask": 1.1,
                "iter_done": 100,
                "masksum": 500,
                "model_flux": 2.5,
            },
            time=0,
            pol=0,
            chan=0,
        )

        entry = rd.sel(time=0, pol=0, chan=0)

        # All should be lists
        for field in FIELD_ACCUM:
            self.assertIn(field, entry)
            self.assertIsInstance(entry[field], list)
            self.assertEqual(len(entry[field]), 1)

    def test_all_field_single_value_remain_scalars(self):
        """Test all FIELD_SINGLE_VALUE fields remain scalars."""
        rd = ReturnDict()

        # Add all FIELD_SINGLE_VALUE fields
        rd.add(
            {
                "max_psf_sidelobe": 0.2,
                "loop_gain": 0.1,
                "niter": 1000,
                "threshold": 0.01,
            },
            time=0,
            pol=0,
            chan=0,
        )

        entry = rd.sel(time=0, pol=0, chan=0)

        # All should be scalars (not lists)
        for field in FIELD_SINGLE_VALUE:
            self.assertIn(field, entry)
            self.assertNotIsInstance(entry[field], list)


class TestReturnDictKeyBehavior(unittest.TestCase):
    """Test Key namedtuple behavior and OrderedDict properties."""

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
        """Test ReturnDict preserves insertion order."""
        rd = ReturnDict()

        # Add keys in specific order
        rd.add({"peakres": 1.0}, time=2, pol=0, chan=0)
        rd.add({"peakres": 0.9}, time=0, pol=0, chan=0)
        rd.add({"peakres": 0.8}, time=1, pol=0, chan=0)

        # Verify order is preserved
        keys = list(rd.data.keys())
        self.assertEqual(keys[0].time, 2)
        self.assertEqual(keys[1].time, 0)
        self.assertEqual(keys[2].time, 1)


# =============================================================================
# ReturnDict Utility Function Tests (Merge, Extraction)
# =============================================================================


class TestMergeReturnDicts(unittest.TestCase):
    """Test merging multiple ReturnDict objects."""

    def setUp(self):
        """Create test ReturnDict objects."""
        self.rd1 = ReturnDict()
        self.rd1.add({"peakres": 0.5, "iter_done": 100}, time=0, pol=0, chan=0)
        self.rd1.add({"peakres": 0.4, "iter_done": 90}, time=0, pol=0, chan=1)

        self.rd2 = ReturnDict()
        self.rd2.add({"peakres": 0.3, "iter_done": 120}, time=0, pol=0, chan=2)
        self.rd2.add({"peakres": 0.6, "iter_done": 80}, time=0, pol=0, chan=3)

        self.rd3 = ReturnDict()
        self.rd3.add({"peakres": 0.2, "iter_done": 110}, time=0, pol=1, chan=0)

    def test_merge_non_overlapping_keys(self):
        """Test merging dicts with no overlapping keys."""
        merged = merge_return_dicts([self.rd1, self.rd2, self.rd3])
        self.assertEqual(len(merged.data), 5)

        # Check all entries present
        self.assertIsNotNone(merged.sel(time=0, pol=0, chan=0))
        self.assertIsNotNone(merged.sel(time=0, pol=0, chan=1))
        self.assertIsNotNone(merged.sel(time=0, pol=0, chan=2))
        self.assertIsNotNone(merged.sel(time=0, pol=0, chan=3))
        self.assertIsNotNone(merged.sel(time=0, pol=1, chan=0))

    def test_merge_strategy_latest(self):
        """Test 'latest' merge strategy overwrites with last value."""
        rd_conflict = ReturnDict()
        rd_conflict.add({"peakres": 0.9, "iter_done": 50}, time=0, pol=0, chan=0)

        merged = merge_return_dicts([self.rd1, rd_conflict], merge_strategy="latest")

        # Should use value from rd_conflict (last in list)
        # Note: fields are stored as lists even for single values (history tracking)
        entry = merged.sel(time=0, pol=0, chan=0)
        self.assertEqual(entry["peakres"], [0.9])  # Now a list
        self.assertEqual(entry["iter_done"], [50])  # Now a list

    def test_merge_strategy_error_raises_on_conflict(self):
        """Test 'error' merge strategy raises on overlapping keys."""
        rd_conflict = ReturnDict()
        rd_conflict.add({"peakres": 0.9, "iter_done": 50}, time=0, pol=0, chan=0)

        with self.assertRaises(ValueError) as context:
            merge_return_dicts([self.rd1, rd_conflict], merge_strategy="error")

        self.assertIn("Conflicting key", str(context.exception))

    def test_merge_strategy_update(self):
        """Test 'update' merge strategy merges dict values."""
        rd_partial = ReturnDict()
        rd_partial.add({"new_field": 123}, time=0, pol=0, chan=0)

        merged = merge_return_dicts([self.rd1, rd_partial], merge_strategy="update")

        entry = merged.sel(time=0, pol=0, chan=0)
        # Should have both original and new fields
        # History-tracked fields (FIELD_ACCUM) are stored as lists
        self.assertEqual(entry["peakres"], [0.5])  # Now a list
        self.assertEqual(entry["iter_done"], [100])  # Now a list
        # new_field is not in FIELD_ACCUM, so it's a single value
        self.assertEqual(entry["new_field"], 123)

    def test_merge_strategy_update_concatenates_history(self):
        """Test 'update' merge strategy concatenates FIELD_ACCUM history lists."""
        # Create a second dict with same key but new values
        rd_second_cycle = ReturnDict()
        rd_second_cycle.add({"peakres": 0.3, "iter_done": 50}, time=0, pol=0, chan=0)

        # Merge with update strategy
        merged = merge_return_dicts(
            [self.rd1, rd_second_cycle], merge_strategy="update"
        )

        entry = merged.sel(time=0, pol=0, chan=0)
        # FIELD_ACCUM fields should be concatenated
        self.assertEqual(entry["peakres"], [0.5, 0.3])  # Concatenated history
        self.assertEqual(entry["iter_done"], [100, 50])  # Concatenated history

    def test_merge_empty_list(self):
        """Test merging empty list returns empty ReturnDict."""
        merged = merge_return_dicts([])
        self.assertEqual(len(merged.data), 0)

    def test_merge_invalid_strategy(self):
        """Test invalid merge strategy raises error."""
        # Create conflicting dicts to trigger the merge strategy check
        rd_conflict = ReturnDict()
        rd_conflict.add({"peakres": 0.9}, time=0, pol=0, chan=0)

        with self.assertRaises(ValueError) as context:
            merge_return_dicts([self.rd1, rd_conflict], merge_strategy="invalid")

        self.assertIn("Unknown merge_strategy", str(context.exception))


class TestReturnDictUtilities(unittest.TestCase):
    """Test ReturnDict extraction utility functions."""

    def setUp(self):
        """Create test ReturnDict with realistic deconvolution data."""
        self.rd = ReturnDict()

        # Add entries for multiple planes
        self.rd.add(
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

        self.rd.add(
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

        self.rd.add(
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
        peak = get_peak_residual_from_returndict(self.rd, use_mask=True)

        # Should return max across valid (non-zero mask) planes
        # Plane (0,0,0): 1.0, Plane (0,0,1): 0.8
        # Plane (0,1,0): masksum=0, ignored
        self.assertEqual(peak, 1.0)

    def test_get_peak_residual_without_mask(self):
        """Test extracting peak residual without mask."""
        peak = get_peak_residual_from_returndict(self.rd, use_mask=False)

        # Should use peakres_nomask and include all planes
        self.assertEqual(peak, 1.2)

    def test_get_peak_residual_with_filter(self):
        """Test extracting peak residual with filtering."""
        peak = get_peak_residual_from_returndict(self.rd, use_mask=True, chan=1)

        # Should only look at chan=1
        self.assertEqual(peak, 0.8)

    def test_get_masksum_total(self):
        """Test extracting total masksum."""
        total_masksum = get_masksum_from_returndict(self.rd)

        # Sum: 100 + 80 + 0 = 180
        self.assertEqual(total_masksum, 180)

    def test_get_masksum_with_filter(self):
        """Test extracting masksum with filtering."""
        masksum = get_masksum_from_returndict(self.rd, pol=0)

        # Sum for pol=0: 100 + 80 = 180
        self.assertEqual(masksum, 180)

        masksum_zero = get_masksum_from_returndict(self.rd, pol=1)
        self.assertEqual(masksum_zero, 0)

    def test_get_iterations_done_total(self):
        """Test extracting total iterations done."""
        total_iters = get_iterations_done_from_returndict(self.rd)

        # Sum: 50 + 40 + 20 = 110
        self.assertEqual(total_iters, 110)

    def test_get_iterations_done_with_filter(self):
        """Test extracting iterations with filtering."""
        iters = get_iterations_done_from_returndict(self.rd, pol=0)

        # Sum for pol=0: 50 + 40 = 90
        self.assertEqual(iters, 90)

    def test_get_max_psf_sidelobe(self):
        """Test extracting max PSF sidelobe level."""
        max_sidelobe = get_max_psf_sidelobe_from_returndict(self.rd)

        # Max across all planes: max(0.2, 0.15, 0.1) = 0.2
        self.assertEqual(max_sidelobe, 0.2)

    def test_get_max_psf_sidelobe_with_filter(self):
        """Test extracting max PSF sidelobe with filtering."""
        max_sidelobe = get_max_psf_sidelobe_from_returndict(self.rd, chan=1)

        # Only chan=1: 0.15
        self.assertEqual(max_sidelobe, 0.15)

    def test_get_max_psf_sidelobe_default(self):
        """Test default PSF sidelobe when not in ReturnDict."""
        rd_empty = ReturnDict()
        rd_empty.add({"peakres": 1.0, "iter_done": 50}, time=0, pol=0, chan=0)

        max_sidelobe = get_max_psf_sidelobe_from_returndict(rd_empty)

        # Should return conservative default
        self.assertEqual(max_sidelobe, 0.2)

    def test_get_peak_residual_empty_returndict(self):
        """Test getting peak residual from empty ReturnDict."""
        rd_empty = ReturnDict()
        peak = get_peak_residual_from_returndict(rd_empty)

        self.assertEqual(peak, 0.0)

    def test_get_masksum_empty_returndict(self):
        """Test getting masksum from empty ReturnDict."""
        rd_empty = ReturnDict()
        masksum = get_masksum_from_returndict(rd_empty)

        self.assertEqual(masksum, 0.0)


class TestIterationControllerInitialization(unittest.TestCase):
    """Test IterationController initialization and default values."""

    def test_default_initialization(self):
        """Test controller with all default parameters."""
        controller = IterationController()

        self.assertEqual(controller.niter, 1000)
        self.assertEqual(controller.nmajor, -1)
        self.assertEqual(controller.threshold, 0.0)
        self.assertEqual(controller.gain, 0.1)
        self.assertEqual(controller.cyclefactor, 1.0)
        self.assertEqual(controller.minpsffraction, 0.05)
        self.assertEqual(controller.maxpsffraction, 0.8)
        self.assertEqual(controller.cycleniter, -1)
        self.assertEqual(controller.nsigma, 0.0)

        # Tracking state
        self.assertEqual(controller.major_done, 0)
        self.assertEqual(controller.total_iter_done, 0)

        # Convergence state
        self.assertEqual(controller.stopcode.major, MAJOR_CONTINUE)
        self.assertEqual(controller.stopcode.minor, MINOR_CONTINUE)
        self.assertIn("Continue", controller.stopdescription)

    def test_custom_initialization(self):
        """Test controller with custom parameters."""
        controller = IterationController(
            niter=500,
            nmajor=10,
            threshold=0.01,
            gain=0.2,
            cyclefactor=1.5,
            minpsffraction=0.1,
            maxpsffraction=0.9,
            cycleniter=100,
            nsigma=5.0,
        )

        self.assertEqual(controller.niter, 500)
        self.assertEqual(controller.nmajor, 10)
        self.assertEqual(controller.threshold, 0.01)
        self.assertEqual(controller.gain, 0.2)
        self.assertEqual(controller.cyclefactor, 1.5)
        self.assertEqual(controller.minpsffraction, 0.1)
        self.assertEqual(controller.maxpsffraction, 0.9)
        self.assertEqual(controller.cycleniter, 100)
        self.assertEqual(controller.nsigma, 5.0)


class TestCalculateCycleControls(unittest.TestCase):
    """Test adaptive cyclethreshold calculation."""

    def setUp(self):
        """Create controller and test ReturnDict."""
        self.controller = IterationController(
            niter=1000,
            cyclefactor=1.5,
            minpsffraction=0.05,
            maxpsffraction=0.8,
        )

    def test_basic_cyclethreshold_calculation(self):
        """Test basic adaptive threshold calculation."""
        rd = ReturnDict()
        rd.add(
            {"peakres": 1.0, "max_psf_sidelobe": 0.2},
            time=0,
            pol=0,
            chan=0,
        )

        cycleniter, cyclethresh = self.controller.calculate_cycle_controls(rd)

        # Expected: psf_fraction = 1.5 × 0.2 = 0.3
        #           cyclethresh = 0.3 × 1.0 = 0.3
        self.assertEqual(cycleniter, 1000)
        self.assertAlmostEqual(cyclethresh, 0.3, places=10)

    def test_cyclethreshold_clamping_min(self):
        """Test cyclethreshold clamping to minpsffraction."""
        rd = ReturnDict()
        # Very small PSF sidelobe
        rd.add(
            {"peakres": 1.0, "max_psf_sidelobe": 0.01},
            time=0,
            pol=0,
            chan=0,
        )

        cycleniter, cyclethresh = self.controller.calculate_cycle_controls(rd)

        # psf_fraction = 1.5 × 0.01 = 0.015
        # Clamped to minpsffraction = 0.05
        # cyclethresh = 0.05 × 1.0 = 0.05
        self.assertEqual(cyclethresh, 0.05)

    def test_cyclethreshold_clamping_max(self):
        """Test cyclethreshold clamping to maxpsffraction."""
        rd = ReturnDict()
        # Very large PSF sidelobe
        rd.add(
            {"peakres": 1.0, "max_psf_sidelobe": 0.9},
            time=0,
            pol=0,
            chan=0,
        )

        cycleniter, cyclethresh = self.controller.calculate_cycle_controls(rd)

        # psf_fraction = 1.5 × 0.9 = 1.35
        # Clamped to maxpsffraction = 0.8
        # cyclethresh = 0.8 × 1.0 = 0.8
        self.assertEqual(cyclethresh, 0.8)

    def test_cyclethreshold_respects_global_threshold(self):
        """Test cyclethreshold respects global threshold as minimum."""
        controller = IterationController(
            niter=1000,
            threshold=0.5,  # High global threshold
            cyclefactor=1.0,
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 1.0, "max_psf_sidelobe": 0.2},
            time=0,
            pol=0,
            chan=0,
        )

        cycleniter, cyclethresh = controller.calculate_cycle_controls(rd)

        # psf_fraction = 1.0 × 0.2 = 0.2
        # cyclethresh_calc = 0.2 × 1.0 = 0.2
        # But threshold = 0.5 is higher, so use that
        self.assertEqual(cyclethresh, 0.5)

    def test_cycleniter_cap_applied(self):
        """Test cycleniter cap is applied."""
        controller = IterationController(
            niter=1000,
            cycleniter=100,  # Cap at 100 per cycle
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 1.0, "max_psf_sidelobe": 0.2},
            time=0,
            pol=0,
            chan=0,
        )

        cycleniter, cyclethresh = controller.calculate_cycle_controls(rd)

        # Should use min(100, 1000) = 100
        self.assertEqual(cycleniter, 100)

    def test_cycleniter_respects_remaining_iterations(self):
        """Test cycleniter respects remaining niter."""
        controller = IterationController(
            niter=50,  # Only 50 iterations left
            cycleniter=100,
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 1.0, "max_psf_sidelobe": 0.2},
            time=0,
            pol=0,
            chan=0,
        )

        cycleniter, cyclethresh = controller.calculate_cycle_controls(rd)

        # Should use min(100, 50) = 50
        self.assertEqual(cycleniter, 50)

    def test_default_psf_sidelobe_used(self):
        """Test default PSF sidelobe when not in ReturnDict."""
        rd = ReturnDict()
        rd.add(
            {"peakres": 1.0},  # No max_psf_sidelobe
            time=0,
            pol=0,
            chan=0,
        )

        cycleniter, cyclethresh = self.controller.calculate_cycle_controls(rd)

        # Should use default 0.2
        # psf_fraction = 1.5 × 0.2 = 0.3
        # cyclethresh = 0.3 × 1.0 = 0.3
        self.assertAlmostEqual(cyclethresh, 0.3, places=10)


class TestCheckConvergence(unittest.TestCase):
    """Test convergence checking with all stop codes."""

    def test_continue_when_not_converged(self):
        """Test continuing when no stopping criteria met."""
        controller = IterationController(
            niter=1000,
            threshold=0.01,
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        self.assertEqual(stopcode.major, MAJOR_CONTINUE)
        self.assertEqual(stopcode.minor, MINOR_CONTINUE)
        self.assertIn("Continue", desc)

    def test_stop_on_zero_mask(self):
        """Test stopping when mask is zero (priority 1)."""
        controller = IterationController(niter=1000)

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.5, "masksum": 0},  # Zero mask!
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        self.assertEqual(stopcode.major, MAJOR_ZERO_MASK)
        self.assertIn("Zero mask", desc)

    def test_stop_on_iteration_limit(self):
        """Test stopping when niter exhausted (priority 2)."""
        controller = IterationController(niter=0)  # No iterations left

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        self.assertEqual(stopcode.major, MAJOR_ITER_LIMIT)
        self.assertIn("iteration limit", desc)

    def test_stop_on_threshold(self):
        """Test stopping when threshold reached (priority 3)."""
        controller = IterationController(
            niter=1000,
            threshold=0.5,  # Set threshold
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.3, "masksum": 100},  # Below threshold
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        self.assertEqual(stopcode.major, MAJOR_THRESHOLD)
        self.assertIn("threshold", desc)

    def test_threshold_exactly_at_limit(self):
        """Test stopping when peak residual exactly equals threshold."""
        controller = IterationController(
            niter=1000,
            threshold=0.5,
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.5, "masksum": 100},  # Exactly at threshold
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        self.assertEqual(stopcode.major, MAJOR_THRESHOLD)

    def test_stop_on_major_cycle_limit(self):
        """Test stopping when nmajor exhausted (priority 4)."""
        controller = IterationController(
            niter=1000,
            nmajor=0,  # No major cycles left
            threshold=0.0,
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        self.assertEqual(stopcode.major, MAJOR_CYCLE_LIMIT)
        self.assertIn("major cycle", desc)

    def test_priority_zero_mask_over_others(self):
        """Test zero mask has highest priority."""
        controller = IterationController(
            niter=0,  # Also at iteration limit
            nmajor=0,  # Also at major cycle limit
            threshold=1.0,  # Also below threshold
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.5, "masksum": 0},  # Zero mask
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        # Zero mask should win
        self.assertEqual(stopcode.major, MAJOR_ZERO_MASK)

    def test_priority_iter_limit_over_threshold(self):
        """Test iteration limit has priority over threshold."""
        controller = IterationController(
            niter=0,  # At iteration limit
            threshold=1.0,  # Also below threshold
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        # Iteration limit should win
        self.assertEqual(stopcode.major, MAJOR_ITER_LIMIT)

    def test_nmajor_unlimited_never_stops(self):
        """Test nmajor=-1 (unlimited) never triggers cycle limit."""
        controller = IterationController(
            niter=1000,
            nmajor=-1,  # Unlimited
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.5, "masksum": 100},
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        self.assertEqual(stopcode.major, MAJOR_CONTINUE)

    def test_threshold_zero_never_stops(self):
        """Test threshold=0 (disabled) never triggers threshold stop."""
        controller = IterationController(
            niter=1000,
            threshold=0.0,  # Disabled
        )

        rd = ReturnDict()
        rd.add(
            {"peakres": 0.001, "masksum": 100},  # Very low residual
            time=0,
            pol=0,
            chan=0,
        )

        stopcode, desc = controller.check_convergence(rd)

        # Should not stop on threshold
        self.assertEqual(stopcode.major, MAJOR_CONTINUE)

    def test_stopcode_state_updated(self):
        """Test controller's internal stopcode is updated."""
        controller = IterationController(niter=0)

        rd = ReturnDict()
        rd.add({"peakres": 0.5, "masksum": 100}, time=0, pol=0, chan=0)

        stopcode, desc = controller.check_convergence(rd)

        # Internal state should match returned values
        self.assertEqual(controller.stopcode.major, stopcode.major)
        self.assertEqual(controller.stopdescription, desc)


class TestUpdateCounts(unittest.TestCase):
    """Test iteration count updates after major cycles."""

    def test_basic_count_update(self):
        """Test basic count decrementing."""
        controller = IterationController(niter=1000, nmajor=5)

        rd = ReturnDict()
        rd.add({"iter_done": 100}, time=0, pol=0, chan=0)

        controller.update_counts(rd)

        # niter: 1000 - 100 = 900
        # nmajor: 5 - 1 = 4
        # major_done: 0 + 1 = 1
        # total_iter_done: 0 + 100 = 100
        self.assertEqual(controller.niter, 900)
        self.assertEqual(controller.nmajor, 4)
        self.assertEqual(controller.major_done, 1)
        self.assertEqual(controller.total_iter_done, 100)

    def test_multiple_updates(self):
        """Test multiple count updates accumulate correctly."""
        controller = IterationController(niter=1000, nmajor=5)

        rd1 = ReturnDict()
        rd1.add({"iter_done": 100}, time=0, pol=0, chan=0)

        rd2 = ReturnDict()
        rd2.add({"iter_done": 150}, time=0, pol=0, chan=0)

        controller.update_counts(rd1)
        controller.update_counts(rd2)

        self.assertEqual(controller.niter, 750)  # 1000 - 100 - 150
        self.assertEqual(controller.nmajor, 3)  # 5 - 1 - 1
        self.assertEqual(controller.major_done, 2)
        self.assertEqual(controller.total_iter_done, 250)  # 100 + 150

    def test_niter_floor_at_zero(self):
        """Test niter doesn't go negative."""
        controller = IterationController(niter=50)

        rd = ReturnDict()
        rd.add({"iter_done": 100}, time=0, pol=0, chan=0)

        controller.update_counts(rd)

        # Should floor at 0, not go negative
        self.assertEqual(controller.niter, 0)

    def test_nmajor_floor_at_zero(self):
        """Test nmajor doesn't go negative."""
        controller = IterationController(nmajor=1)

        rd1 = ReturnDict()
        rd1.add({"iter_done": 50}, time=0, pol=0, chan=0)

        rd2 = ReturnDict()
        rd2.add({"iter_done": 50}, time=0, pol=0, chan=0)

        controller.update_counts(rd1)
        controller.update_counts(rd2)

        # Should floor at 0, not go negative
        self.assertEqual(controller.nmajor, 0)

    def test_nmajor_unlimited_never_decrements(self):
        """Test nmajor=-1 (unlimited) never decrements."""
        controller = IterationController(niter=1000, nmajor=-1)

        rd = ReturnDict()
        rd.add({"iter_done": 100}, time=0, pol=0, chan=0)

        controller.update_counts(rd)

        # nmajor should stay -1
        self.assertEqual(controller.nmajor, -1)
        self.assertEqual(controller.major_done, 1)

    def test_update_with_multiple_planes(self):
        """Test update sums iterations across multiple planes."""
        controller = IterationController(niter=1000)

        rd = ReturnDict()
        rd.add({"iter_done": 50}, time=0, pol=0, chan=0)
        rd.add({"iter_done": 30}, time=0, pol=0, chan=1)
        rd.add({"iter_done": 20}, time=0, pol=1, chan=0)

        controller.update_counts(rd)

        # Total iterations: 50 + 30 + 20 = 100
        self.assertEqual(controller.niter, 900)
        self.assertEqual(controller.total_iter_done, 100)
        self.assertEqual(controller.major_done, 1)

    def test_no_update_when_converged(self):
        """Test update_counts does nothing when already converged."""
        controller = IterationController(niter=1000, nmajor=5)

        # Set converged state
        controller.stopcode = StopCode(major=MAJOR_THRESHOLD, minor=MINOR_CONTINUE)

        rd = ReturnDict()
        rd.add({"iter_done": 100}, time=0, pol=0, chan=0)

        controller.update_counts(rd)

        # Counts should not change
        self.assertEqual(controller.niter, 1000)
        self.assertEqual(controller.nmajor, 5)
        self.assertEqual(controller.major_done, 0)
        self.assertEqual(controller.total_iter_done, 0)


class TestUpdateParameters(unittest.TestCase):
    """Test interactive parameter updates with validation."""

    def test_update_niter(self):
        """Test updating niter parameter."""
        controller = IterationController(niter=1000)

        code, msg = controller.update_parameters(niter=500)

        self.assertEqual(code, 0)
        self.assertEqual(msg, "")
        self.assertEqual(controller.niter, 500)

    def test_update_cycleniter(self):
        """Test updating cycleniter parameter."""
        controller = IterationController(cycleniter=-1)

        code, msg = controller.update_parameters(cycleniter=100)

        self.assertEqual(code, 0)
        self.assertEqual(controller.cycleniter, 100)

    def test_update_nmajor(self):
        """Test updating nmajor parameter."""
        controller = IterationController(nmajor=-1)

        code, msg = controller.update_parameters(nmajor=10)

        self.assertEqual(code, 0)
        self.assertEqual(controller.nmajor, 10)

    def test_update_threshold_numeric(self):
        """Test updating threshold with numeric value."""
        controller = IterationController(threshold=0.0)

        code, msg = controller.update_parameters(threshold=0.5)

        self.assertEqual(code, 0)
        self.assertEqual(controller.threshold, 0.5)

    def test_update_threshold_string_jy(self):
        """Test updating threshold with Jy string."""
        controller = IterationController()

        code, msg = controller.update_parameters(threshold="0.5Jy")

        self.assertEqual(code, 0)
        self.assertEqual(controller.threshold, 0.5)

    def test_update_threshold_string_mjy(self):
        """Test updating threshold with mJy string."""
        controller = IterationController()

        code, msg = controller.update_parameters(threshold="10mJy")

        self.assertEqual(code, 0)
        self.assertAlmostEqual(controller.threshold, 0.01)

    def test_update_threshold_string_ujy(self):
        """Test updating threshold with uJy string."""
        controller = IterationController()

        code, msg = controller.update_parameters(threshold="100uJy")

        self.assertEqual(code, 0)
        self.assertAlmostEqual(controller.threshold, 0.0001)

    def test_update_cyclefactor(self):
        """Test updating cyclefactor parameter."""
        controller = IterationController(cyclefactor=1.0)

        code, msg = controller.update_parameters(cyclefactor=1.5)

        self.assertEqual(code, 0)
        self.assertEqual(controller.cyclefactor, 1.5)

    def test_update_multiple_parameters(self):
        """Test updating multiple parameters at once."""
        controller = IterationController()

        code, msg = controller.update_parameters(
            niter=500,
            threshold="5mJy",
            cyclefactor=1.5,
        )

        self.assertEqual(code, 0)
        self.assertEqual(controller.niter, 500)
        self.assertAlmostEqual(controller.threshold, 0.005)
        self.assertEqual(controller.cyclefactor, 1.5)

    def test_reject_negative_niter(self):
        """Test rejecting niter < -1."""
        controller = IterationController()

        code, msg = controller.update_parameters(niter=-2)

        self.assertEqual(code, -1)
        self.assertIn("niter must be >= -1", msg)
        self.assertEqual(controller.niter, 1000)  # Unchanged

    def test_reject_negative_threshold_numeric(self):
        """Test rejecting negative numeric threshold."""
        controller = IterationController()

        code, msg = controller.update_parameters(threshold=-0.5)

        self.assertEqual(code, -1)
        self.assertIn("threshold must be >= 0", msg)

    def test_reject_negative_threshold_string(self):
        """Test rejecting negative threshold string (Bug #1 fix)."""
        controller = IterationController()

        code, msg = controller.update_parameters(threshold="-10mJy")

        self.assertEqual(code, -1)
        self.assertIn("threshold must be >= 0", msg)

    def test_reject_zero_cyclefactor(self):
        """Test rejecting cyclefactor <= 0."""
        controller = IterationController()

        code, msg = controller.update_parameters(cyclefactor=0)

        self.assertEqual(code, -1)
        self.assertIn("cyclefactor must be > 0", msg)

        code, msg = controller.update_parameters(cyclefactor=-1.0)

        self.assertEqual(code, -1)
        self.assertIn("cyclefactor must be > 0", msg)

    def test_reject_invalid_threshold_string(self):
        """Test rejecting threshold string with unknown units."""
        controller = IterationController()

        code, msg = controller.update_parameters(threshold="10kJy")

        self.assertEqual(code, -1)
        self.assertIn("number with units", msg)

    def test_reject_non_numeric_niter(self):
        """Test rejecting non-numeric niter."""
        controller = IterationController()

        code, msg = controller.update_parameters(niter="abc")

        self.assertEqual(code, -1)
        self.assertIn("integer", msg)


class TestResetMethods(unittest.TestCase):
    """Test reset functionality."""

    def test_reset(self):
        """Test full reset restores initial state."""
        controller = IterationController(niter=1000, nmajor=5)

        # Simulate some work
        rd = ReturnDict()
        rd.add({"iter_done": 300}, time=0, pol=0, chan=0)
        controller.update_counts(rd)

        # Modify stopcode
        controller.stopcode = StopCode(major=MAJOR_THRESHOLD, minor=MINOR_CONTINUE)

        # Reset
        controller.reset()

        # Should restore to initial state
        self.assertEqual(controller.niter, 1000)
        self.assertEqual(controller.major_done, 0)
        self.assertEqual(controller.total_iter_done, 0)
        self.assertEqual(controller.stopcode.major, MAJOR_CONTINUE)
        self.assertEqual(controller.stopcode.minor, MINOR_CONTINUE)

    def test_reset_stopcode_only(self):
        """Test reset_stopcode only resets convergence state."""
        controller = IterationController(niter=1000)

        # Simulate some work
        rd = ReturnDict()
        rd.add({"iter_done": 300}, time=0, pol=0, chan=0)
        controller.update_counts(rd)

        # Set converged
        controller.stopcode = StopCode(major=MAJOR_THRESHOLD, minor=MINOR_CONTINUE)

        # Reset only stopcode
        controller.reset_stopcode()

        # Stopcode should be reset
        self.assertEqual(controller.stopcode.major, MAJOR_CONTINUE)
        self.assertEqual(controller.stopcode.minor, MINOR_CONTINUE)

        # But counts should remain
        self.assertEqual(controller.niter, 700)
        self.assertEqual(controller.major_done, 1)


class TestGetState(unittest.TestCase):
    """Test state serialization."""

    def test_get_state(self):
        """Test getting controller state as dictionary."""
        controller = IterationController(
            niter=1000,
            nmajor=5,
            threshold=0.01,
            cyclefactor=1.5,
        )

        # Simulate some work
        rd = ReturnDict()
        rd.add({"iter_done": 200}, time=0, pol=0, chan=0)
        controller.update_counts(rd)

        state = controller.get_state()

        # Check all fields present
        self.assertEqual(state["niter"], 800)
        self.assertEqual(state["nmajor"], 4)
        self.assertEqual(state["initial_niter"], 1000)
        self.assertEqual(state["threshold"], 0.01)
        self.assertEqual(state["cyclefactor"], 1.5)
        self.assertEqual(state["major_done"], 1)
        self.assertEqual(state["total_iter_done"], 200)

        # Stopcode should be serialized as dict
        self.assertIsInstance(state["stopcode"], dict)
        self.assertEqual(state["stopcode"]["major"], MAJOR_CONTINUE)
        self.assertEqual(state["stopcode"]["minor"], MINOR_CONTINUE)


class TestFullMajorCycleWorkflow(unittest.TestCase):
    """Integration tests for complete major cycle workflows."""

    def test_basic_convergence_workflow(self):
        """Test basic major cycle workflow until convergence."""
        controller = IterationController(
            niter=300,
            nmajor=5,
            threshold=0.1,
            cyclefactor=1.5,
        )

        # Simulate 3 major cycles with decreasing residual
        residuals = [1.0, 0.5, 0.08]  # Last one below threshold
        iterations_per_cycle = [100, 100, 50]

        for cycle, (residual, iters) in enumerate(zip(residuals, iterations_per_cycle)):
            # Create ReturnDict for this cycle
            rd = ReturnDict()
            rd.add(
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

            # Calculate cycle controls
            cycleniter, cyclethresh = controller.calculate_cycle_controls(rd)

            # Update counts
            controller.update_counts(rd)

            # Check convergence
            stopcode, desc = controller.check_convergence(rd)

            if stopcode.major != MAJOR_CONTINUE:
                # Should converge on cycle 3 (index 2)
                self.assertEqual(cycle, 2)
                self.assertEqual(stopcode.major, MAJOR_THRESHOLD)
                break

        # Verify final state
        self.assertEqual(controller.major_done, 3)
        self.assertEqual(controller.total_iter_done, 250)
        self.assertEqual(controller.niter, 50)  # 300 - 250

    def test_iteration_limit_workflow(self):
        """Test workflow stopping at iteration limit."""
        controller = IterationController(
            niter=250,  # Limited iterations
            threshold=0.01,  # Low threshold (hard to reach)
        )

        # Simulate major cycles
        for cycle in range(5):
            rd = ReturnDict()
            rd.add(
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

            controller.update_counts(rd)
            stopcode, desc = controller.check_convergence(rd)

            if stopcode.major != MAJOR_CONTINUE:
                # Should stop after cycle 3 (3 × 100 = 300 > 250)
                # But niter floors at 0 after cycle 2
                self.assertEqual(cycle, 2)
                self.assertEqual(stopcode.major, MAJOR_ITER_LIMIT)
                break

    def test_major_cycle_limit_workflow(self):
        """Test workflow stopping at major cycle limit."""
        controller = IterationController(
            niter=1000,
            nmajor=3,  # Limited major cycles
            threshold=0.01,
        )

        # Simulate major cycles
        for cycle in range(5):
            rd = ReturnDict()
            rd.add(
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
            # After 3 updates, nmajor will be 0, triggering stop
            controller.update_counts(rd)
            stopcode, desc = controller.check_convergence(rd)

            if stopcode.major != MAJOR_CONTINUE:
                # Should stop when nmajor reaches 0 after 3 cycles
                # Cycle 0: nmajor 3→2, Cycle 1: nmajor 2→1, Cycle 2: nmajor 1→0
                # Check on cycle 2 detects nmajor==0
                self.assertEqual(cycle, 2)
                self.assertEqual(stopcode.major, MAJOR_CYCLE_LIMIT)
                break

        self.assertEqual(controller.major_done, 3)

    def test_zero_mask_workflow(self):
        """Test workflow stopping when mask becomes zero."""
        controller = IterationController(
            niter=1000,
            threshold=0.1,
        )

        # Simulate cycles with decreasing mask
        masksums = [100, 50, 0]  # Mask disappears

        for cycle, masksum in enumerate(masksums):
            rd = ReturnDict()
            rd.add(
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

            controller.update_counts(rd)
            stopcode, desc = controller.check_convergence(rd)

            if stopcode.major != MAJOR_CONTINUE:
                # Should stop when mask becomes zero
                self.assertEqual(cycle, 2)
                self.assertEqual(stopcode.major, MAJOR_ZERO_MASK)
                break

    def test_interactive_continue_workflow(self):
        """Test interactive workflow: stop, update params, continue."""
        controller = IterationController(
            niter=100,
            threshold=0.5,
        )

        # First phase: run until iteration limit
        rd1 = ReturnDict()
        rd1.add(
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

        controller.update_counts(rd1)
        stopcode, desc = controller.check_convergence(rd1)

        self.assertEqual(stopcode.major, MAJOR_ITER_LIMIT)

        # User decides to continue with more iterations
        code, msg = controller.update_parameters(niter=200)
        self.assertEqual(code, 0)

        # Reset stopcode to continue
        controller.reset_stopcode()

        # Second phase: run more cycles
        rd2 = ReturnDict()
        rd2.add(
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

        controller.update_counts(rd2)
        stopcode, desc = controller.check_convergence(rd2)

        self.assertEqual(stopcode.major, MAJOR_THRESHOLD)
        self.assertEqual(controller.major_done, 2)
        self.assertEqual(controller.total_iter_done, 200)


class TestMultiPlaneWorkflows(unittest.TestCase):
    """Test workflows with multiple time/pol/chan planes."""

    def test_merge_and_converge_multi_channel(self):
        """Test merging results from multiple channels."""
        controller = IterationController(
            niter=300,
            threshold=0.1,
        )

        # Simulate 3 workers processing different channels
        rd1 = ReturnDict()
        rd1.add(
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

        rd2 = ReturnDict()
        rd2.add(
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

        rd3 = ReturnDict()
        rd3.add(
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
        merged = merge_return_dicts([rd1, rd2, rd3])

        # Get global statistics
        peak = get_peak_residual_from_returndict(merged)
        total_iters = get_iterations_done_from_returndict(merged)
        max_sidelobe = get_max_psf_sidelobe_from_returndict(merged)

        self.assertEqual(peak, 0.8)  # Max across channels
        self.assertEqual(total_iters, 150)  # 50 + 60 + 40
        self.assertEqual(max_sidelobe, 0.25)  # Max sidelobe

        # Check global convergence
        controller.update_counts(merged)
        stopcode, desc = controller.check_convergence(merged)

        # Should not converge (peak 0.8 > threshold 0.1)
        self.assertEqual(stopcode.major, MAJOR_CONTINUE)
        self.assertEqual(controller.niter, 150)  # 300 - 150

    def test_partial_zero_mask_handling(self):
        """Test handling when some planes have zero mask."""
        controller = IterationController(threshold=0.5)

        # Chan 0: zero mask, Chan 1: valid
        rd = ReturnDict()
        rd.add(
            {
                "peakres": 1.0,
                "masksum": 0,  # Zero mask
                "iter_done": 50,
            },
            time=0,
            pol=0,
            chan=0,
        )
        rd.add(
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
        total_masksum = get_masksum_from_returndict(rd)
        self.assertEqual(total_masksum, 100)

        # Peak should ignore zero-mask plane
        peak = get_peak_residual_from_returndict(rd, use_mask=True)
        self.assertEqual(peak, 0.3)  # Only chan 1

        # Should converge on threshold
        stopcode, desc = controller.check_convergence(rd)
        self.assertEqual(stopcode.major, MAJOR_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
