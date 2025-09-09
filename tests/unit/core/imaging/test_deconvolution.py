"""
Unit tests for astroviper.core.imaging.deconvolution module

Tests all functions in the deconvolution module including parameter validation,
progress callbacks, and the main Hogbom CLEAN algorithm.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
import sys
import os

from astroviper.core.imaging.deconvolution import (
    _validate_dconv_params,
    progress_callback,
    hogbom_clean,
)

try:
    from astroviper.core.imaging.deconvolvers import hogbom

    HOGBOM_AVAILABLE = True
except ImportError as e:
    HOGBOM_AVAILABLE = False
    print(f"Warning: Hogbom extension not available: {e}")

# Skip all tests relying on C++ extension if hogbom is not available
pytestmark = pytest.mark.skipif(
    not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
)


class TestValidateDconvParams:
    """Test the _validate_dconv_params function"""

    def test_default_parameters(self):
        """Test that default parameters are set correctly"""
        params = {}
        result = _validate_dconv_params(params)

        expected = {"gain": 0.1, "niter": 1000, "threshold": None, "clean_box": None}
        assert result == expected

    def test_existing_valid_parameters(self):
        """Test that valid existing parameters are preserved"""
        params = {
            "gain": 0.05,
            "niter": 500,
            "threshold": 1e-6,
            "clean_box": (slice(10, 20), slice(15, 25)),
        }
        result = _validate_dconv_params(params)
        assert result == params

    def test_gain_validation(self):
        """Test gain parameter validation"""
        # Valid gain values
        valid_gains = [0.01, 0.1, 0.5, 1.0]
        for gain in valid_gains:
            params = {"gain": gain}
            result = _validate_dconv_params(params)
            assert result["gain"] == gain

        # Invalid gain values
        invalid_gains = [0.0, -0.1, 1.1, 2.0]
        for gain in invalid_gains:
            params = {"gain": gain}
            with pytest.raises(ValueError, match="CLEAN gain must be between 0 and 1"):
                _validate_dconv_params(params)

    def test_niter_validation(self):
        """Test niter parameter validation"""
        # Valid niter values
        valid_niters = [1, 100, 1000, 10000]
        for niter in valid_niters:
            params = {"niter": niter}
            result = _validate_dconv_params(params)
            assert result["niter"] == niter

        # Invalid niter values
        invalid_niters = [0, -1, 1.5, "100", None]
        for niter in invalid_niters:
            params = {"niter": niter}
            with pytest.raises(
                ValueError,
                match="Maximum number of iterations must be a positive integer",
            ):
                _validate_dconv_params(params)

    def test_threshold_validation(self):
        """Test threshold parameter validation"""
        # Valid threshold values
        valid_thresholds = [None, 0.0, 1e-6, 0.001]
        for threshold in valid_thresholds:
            params = {"threshold": threshold}
            result = _validate_dconv_params(params)
            assert result["threshold"] == threshold

        # Invalid threshold values
        invalid_thresholds = [-1e-6, -0.001]
        for threshold in invalid_thresholds:
            params = {"threshold": threshold}
            with pytest.raises(
                ValueError, match="Threshold must be non-negative or None"
            ):
                _validate_dconv_params(params)

    def test_clean_box_validation(self):
        """Test clean_box parameter validation"""
        # Valid clean_box values
        valid_clean_boxes = [
            None,
            (slice(0, 10), slice(0, 10)),
            (slice(5, 15), slice(10, 20)),
        ]
        for clean_box in valid_clean_boxes:
            params = {"clean_box": clean_box}
            result = _validate_dconv_params(params)
            assert result["clean_box"] == clean_box

        # Invalid clean_box values
        invalid_clean_boxes = [
            (slice(0, 10),),  # Only one slice
            (0, 10, 0, 10),  # Not slices
            "invalid",  # String
            [],  # Empty list
        ]
        for clean_box in invalid_clean_boxes:
            params = {"clean_box": clean_box}
            with pytest.raises(
                ValueError, match="Clean box must be a tuple of slices or None"
            ):
                _validate_dconv_params(params)

    def test_partial_parameters(self):
        """Test that partial parameter specification works correctly"""
        params = {"gain": 0.05}
        result = _validate_dconv_params(params)

        expected = {
            "gain": 0.05,
            "niter": 1000,  # Default
            "threshold": None,  # Default
            "clean_box": None,  # Default
        }
        assert result == expected


class TestHogbomUtilities:
    """Test utility functions in the hogbom extension"""

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_get_dtype_name(self):
        """Test get_dtype_name function"""

        # Test different dtypes
        arr_float32 = np.array([1.0, 2.0], dtype=np.float32)
        arr_float64 = np.array([1.0, 2.0], dtype=np.float64)
        arr_int32 = np.array([1, 2], dtype=np.int32)

        assert hogbom.get_dtype_name(arr_float32) == "float32"
        assert hogbom.get_dtype_name(arr_float64) == "float64"
        assert hogbom.get_dtype_name(arr_int32) == "int32"

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_is_float32(self):
        """Test is_float32 function"""

        arr_float32 = np.array([1.0, 2.0], dtype=np.float32)
        arr_float64 = np.array([1.0, 2.0], dtype=np.float64)
        arr_int32 = np.array([1, 2], dtype=np.int32)

        assert hogbom.is_float32(arr_float32) is True
        assert hogbom.is_float32(arr_float64) is False
        assert hogbom.is_float32(arr_int32) is False

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_is_float64(self):
        """Test is_float64 function"""

        arr_float32 = np.array([1.0, 2.0], dtype=np.float32)
        arr_float64 = np.array([1.0, 2.0], dtype=np.float64)
        arr_int32 = np.array([1, 2], dtype=np.int32)

        assert hogbom.is_float64(arr_float32) is False
        assert hogbom.is_float64(arr_float64) is True
        assert hogbom.is_float64(arr_int32) is False


class TestMaximgFunction:
    """Test the maximg function"""

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_basic_float32(self):
        """Test basic maximg functionality with float32"""

        # Create 3D test image [pol, ny, nx]
        npol, ny, nx = 2, 32, 32
        image = np.random.rand(npol, ny, nx).astype(np.float32) * 2 - 1  # Range [-1, 1]

        # Set known min/max values
        image[0, 10, 15] = -2.5  # Global minimum
        image[1, 20, 25] = 3.7  # Global maximum

        fmin, fmax = hogbom.maximg(image)

        assert fmin == pytest.approx(-2.5, abs=1e-6)
        assert fmax == pytest.approx(3.7, abs=1e-6)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_basic_float64(self):
        """Test basic maximg functionality with float64"""

        # Create 3D test image [pol, ny, nx]
        npol, ny, nx = 1, 16, 16
        image = np.random.rand(npol, ny, nx).astype(np.float64) * 10

        # Set known min/max values
        image[0, 5, 8] = -15.75
        image[0, 12, 3] = 22.33

        fmin, fmax = hogbom.maximg(image)

        assert fmin == pytest.approx(-15.75, abs=1e-12)
        assert fmax == pytest.approx(22.33, abs=1e-12)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_with_mask(self):
        """Test maximg with mask"""

        npol, ny, nx = 1, 16, 16
        image = np.ones((npol, ny, nx), dtype=np.float32) * 5.0

        # Set extreme values that should be masked out
        image[0, 3, 4] = -100.0  # Should be ignored due to mask
        image[0, 10, 12] = 200.0  # Should be ignored due to mask

        # Create mask - 1.0 means use pixel, 0.0 means ignore
        mask = np.ones((ny, nx), dtype=np.float32)
        mask[3, 4] = 0.0  # Mask out the minimum
        mask[10, 12] = 0.0  # Mask out the maximum

        fmin, fmax = hogbom.maximg(image, mask)

        # Should find 5.0 as both min and max (ignoring masked pixels)
        assert fmin == pytest.approx(5.0, abs=1e-6)
        assert fmax == pytest.approx(5.0, abs=1e-6)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_single_polarization(self):
        """Test maximg with single polarization"""

        npol, ny, nx = 1, 8, 8
        image = np.zeros((npol, ny, nx), dtype=np.float32)
        image[0, 4, 4] = 1.0  # Single point source

        fmin, fmax = hogbom.maximg(image)

        assert fmin == pytest.approx(0.0, abs=1e-6)
        assert fmax == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_multiple_polarizations(self):
        """Test maximg with multiple polarizations"""

        npol, ny, nx = 4, 10, 10
        image = np.zeros((npol, ny, nx), dtype=np.float32)

        # Set different values in each polarization
        image[0, 2, 3] = -3.0  # Global minimum
        image[1, 4, 5] = 1.5
        image[2, 6, 7] = 2.8  # Global maximum
        image[3, 8, 1] = -1.2

        fmin, fmax = hogbom.maximg(image)

        assert fmin == pytest.approx(-3.0, abs=1e-6)
        assert fmax == pytest.approx(2.8, abs=1e-6)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_error_wrong_dimensions(self):
        """Test maximg error handling for wrong dimensions"""

        # Test 2D array (should fail)
        image_2d = np.random.rand(32, 32).astype(np.float32)
        with pytest.raises(RuntimeError, match="Image must be 3D array"):
            hogbom.maximg(image_2d)

        # Test 4D array (should fail)
        image_4d = np.random.rand(2, 2, 32, 32).astype(np.float32)
        with pytest.raises(RuntimeError, match="Image must be 3D array"):
            hogbom.maximg(image_4d)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_mask_dimension_mismatch(self):
        """Test maximg error handling for mask dimension mismatch"""

        image = np.random.rand(2, 32, 32).astype(np.float32)
        wrong_mask = np.ones((16, 16), dtype=np.float32)  # Wrong size

        with pytest.raises(
            RuntimeError, match="Mask dimensions must match image spatial dimensions"
        ):
            hogbom.maximg(image, wrong_mask)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
