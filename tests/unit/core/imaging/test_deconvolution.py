"""
Unit tests for astroviper.core.imaging.deconvolution module

Tests all functions in the deconvolution module including parameter validation,
progress callbacks, and the main Hogbom CLEAN algorithm.
"""

import pytest
import numpy as np
import xarray as xr
import sys
import os
import shutil
import toolviper
from xradio.image import load_image

from astroviper.core.imaging.deconvolution import (
    _validate_deconv_params,
    progress_callback,
    hogbom_clean,
)

try:
    from astroviper.core.imaging.deconvolvers import hogbom

    HOGBOM_AVAILABLE = True
except ImportError as e:
    HOGBOM_AVAILABLE = False
    print(f"Warning: Hogbom extension not available: {e}")


def _wipe_files(file_list):
    """Helper function to remove files or directories if they exist"""
    for f in file_list:
        if os.path.exists(f):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)


@pytest.fixture()
def hogbom_images():
    """
    Create a new copy of the test images for hogbom tests.
    """
    resid_image_orig = "refim_point_im.residual"
    psf_image_orig = "refim_point_im.psf"

    if not os.path.exists(resid_image_orig) or not os.path.exists(psf_image_orig):
        _wipe_files([resid_image_orig, psf_image_orig])
        toolviper.utils.data.update()
        files = [resid_image_orig, psf_image_orig]
        toolviper.utils.data.download(file=files, folder=".")

    resid_image = "test_hogbom_resid.im"
    psf_image = "test_hogbom_psf.im"

    if os.path.exists(resid_image):
        shutil.rmtree(resid_image)
    if os.path.exists(psf_image):
        shutil.rmtree(psf_image)

    shutil.copytree(resid_image_orig, resid_image)
    shutil.copytree(psf_image_orig, psf_image)

    return resid_image, psf_image


class TestValidateDeconvParams:
    """Test the _validate_deconv_params function"""

    def test_default_parameters(self):
        """Test that default parameters are set correctly"""
        params = {}
        result = _validate_deconv_params(params)

        expected = {
            "gain": 0.1,
            "niter": 1000,
            "threshold": 0.0,
            "clean_box": (-1, -1, -1, -1),
        }
        assert result == expected

    def test_existing_valid_parameters(self):
        """Test that valid existing parameters are preserved"""
        params = {
            "gain": 0.05,
            "niter": 500,
            "threshold": 1e-6,
            "clean_box": (slice(10, 20), slice(15, 25)),
        }
        result = _validate_deconv_params(params)
        assert result == params

    def test_gain_validation(self):
        """Test gain parameter validation"""
        # Valid gain values
        valid_gains = [0.01, 0.1, 0.5, 1.0]
        for gain in valid_gains:
            params = {"gain": gain}
            result = _validate_deconv_params(params)
            assert result["gain"] == gain

        # Invalid gain values
        invalid_gains = [0.0, -0.1, 1.1, 2.0]
        for gain in invalid_gains:
            params = {"gain": gain}
            with pytest.raises(ValueError, match="CLEAN gain must be between 0 and 1"):
                _validate_deconv_params(params)

    def test_niter_validation(self):
        """Test niter parameter validation"""
        # Valid niter values
        valid_niters = [1, 100, 1000, 10000]
        for niter in valid_niters:
            params = {"niter": niter}
            result = _validate_deconv_params(params)
            assert result["niter"] == niter

        # Invalid niter values
        invalid_niters = [0, -1, 1.5, "100", None]
        for niter in invalid_niters:
            params = {"niter": niter}
            with pytest.raises(
                ValueError,
                match="Maximum number of iterations must be a positive integer",
            ):
                _validate_deconv_params(params)

    def test_threshold_validation(self):
        """Test threshold parameter validation"""
        # Valid threshold values
        valid_thresholds = [None, 0.0, 1e-6, 0.001]
        for threshold in valid_thresholds:
            params = {"threshold": threshold}
            result = _validate_deconv_params(params)
            assert result["threshold"] == threshold

        # Invalid threshold values
        invalid_thresholds = [-1e-6, -0.001]
        for threshold in invalid_thresholds:
            params = {"threshold": threshold}
            with pytest.raises(
                ValueError, match="Threshold must be non-negative or None"
            ):
                _validate_deconv_params(params)

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
            result = _validate_deconv_params(params)
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
                _validate_deconv_params(params)

    def test_partial_parameters(self):
        """Test that partial parameter specification works correctly"""
        params = {"gain": 0.05}
        result = _validate_deconv_params(params)

        expected = {
            "gain": 0.05,
            "niter": 1000,  # Default
            "threshold": 0.0,  # Default
            "clean_box": (-1, -1, -1, -1),  # Default
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


class TestHogbomClean:
    """Test the hogbom_clean function"""

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_hogbom_clean_basic(self, hogbom_images):
        """Test basic hogbom_clean functionality"""
        # Setup mock returns

        resid_image, psf_image = hogbom_images
        dirty_xds = load_image(resid_image)
        psf_xds = load_image(psf_image)

        deconv_params = {"gain": 0.1, "niter": 100}

        # Run function
        result, resid_xds, psf_xds = hogbom_clean(dirty_xds, psf_xds, deconv_params)

        # Verify result structure
        assert isinstance(result, dict)
        assert "iterations_performed" in result
        assert "final_peak" in result
        assert "total_flux_cleaned" in result
        assert result["iterations_performed"] == 100

    # @pytest.mark.skipif(
    #     not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    # )
    # def test_hogbom_clean_multipol(self, hogbom_images):
    #     """Test hogbom_clean with multiple time and frequency slices"""
    #
    #     resid_image, psf_image = hogbom_images
    #     dirty_xds = load_image(resid_image)
    #     psf_xds = load_image(psf_image)
    #
    #     # Grab data to duplicate across frequency
    #     original_data = dirty_xds["SKY"].data
    #     original_psf = psf_xds["SKY"].data
    #
    #     pol_coords = ['I', 'Q', 'U', 'V']
    #
    #     # Replicate the single plane data to 4 polarizations
    #     # Original shape should be [time, freq, pol, ny, nx] - expand polarization axis
    #     cube_data = np.repeat(original_data, 4, axis=2)  # Repeat along pol axis
    #     cube_psf = np.repeat(original_psf, 4, axis=2)
    #
    #     # Create new xarray datasets with frequency dimension
    #     dirty_xds_multipol = dirty_xds.copy()
    #     dirty_xds_multipol['SKY'] = xr.DataArray(
    #         cube_data,
    #         dims=['time', 'frequency', 'polarization', 'l', 'm'],
    #         coords={
    #             'time': dirty_xds['SKY'].coords['time'],
    #             'frequency': dirty_xds['SKY'].coords['frequency'],
    #             'polarization': pol_coords,
    #             'l': dirty_xds['SKY'].coords['l'],
    #             'm': dirty_xds['SKY'].coords['m']
    #         }
    #     )
    #
    #     psf_xds_multipol = psf_xds.copy()
    #     psf_xds_multipol['SKY'] = xr.DataArray(
    #         cube_psf,
    #         dims=['time', 'frequency', 'polarization', 'l', 'm'],
    #         coords={
    #             'time': psf_xds['SKY'].coords['time'],
    #             'frequency': psf_xds['SKY'].coords['frequency'],
    #             'polarization': pol_coords,
    #             'l': psf_xds['SKY'].coords['l'],
    #             'm': psf_xds['SKY'].coords['m']
    #         }
    #     )
    #
    #     deconv_params = {'gain': 0.1, 'niter': 100}
    #
    #     result = hogbom_clean(dirty_xds_multipol, psf_xds_multipol, deconv_params)
    #
    #
    # def test_hogbom_clean_logging(self, mock_logger, mock_hogbom):
    #
    #     """Test that proper logging occurs during deconvolution"""
    #     mock_hogbom.maximg.return_value = (-0.5, 1.0)
    #     mock_hogbom.clean.return_value = {
    #         'model_image': np.zeros((1, 32, 32), dtype=np.float32),
    #         'residual_image': np.random.rand(1, 32, 32).astype(np.float32),
    #         'iterations_performed': 25,
    #         'final_peak': 0.05,
    #         'total_flux_cleaned': 1.2,
    #         'converged': True
    #     }
    #
    #     dirty_xds = self.create_mock_xds(nx=32, ny=32)
    #     psf_xds = self.create_mock_psf_xds(nx=32, ny=32)
    #     deconv_params = {'gain': 0.1}
    #
    #     hogbom_clean(dirty_xds, psf_xds, deconv_params)
    #
    #     # Check that debug and info logging occurred
    #     assert mock_logger.debug.called
    #     assert mock_logger.info.called
    #
    # def test_hogbom_clean_output_structure(self):
    #     """Test the structure of hogbom_clean output"""
    #
    #     expected_result = {
    #         'model_image': np.zeros((1, 64, 64), dtype=np.float32),
    #         'residual_image': np.random.rand(1, 64, 64).astype(np.float32),
    #         'iterations_performed': 75,
    #         'final_peak': 0.02,
    #         'total_flux_cleaned': 3.1,
    #         'converged': False
    #     }
    #
    #     mock_hogbom.maximg.return_value = (-0.3, 0.8)
    #     mock_hogbom.clean.return_value = expected_result
    #
    #     dirty_xds = self.create_mock_xds()
    #     psf_xds = self.create_mock_psf_xds()
    #     deconv_params = {'gain': 0.1}
    #
    #     result = hogbom_clean(dirty_xds, psf_xds, deconv_params)
    #
    #     # Verify all expected keys are present
    #     required_keys = [
    #         'model_image', 'residual_image', 'iterations_performed',
    #         'final_peak', 'total_flux_cleaned', 'converged'
    #     ]
    #     for key in required_keys:
    #         assert key in result
    #
    #     # Verify data types
    #     assert isinstance(result['iterations_performed'], int)
    #     assert isinstance(result['final_peak'], (int, float))
    #     assert isinstance(result['total_flux_cleaned'], (int, float))
    #     assert isinstance(result['converged'], bool)
    #


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
