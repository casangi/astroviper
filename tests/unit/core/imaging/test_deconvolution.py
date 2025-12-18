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
    deconvolve,
)
from astroviper.core.imaging.imaging_utils.return_dict import ReturnDict

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
            "clean_box": (10, 20, 15, 25),
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
        # Valid clean_box values (4-tuples of integers)
        valid_clean_boxes = [
            None,
            (-1, -1, -1, -1),  # No clean box
            (0, 10, 0, 10),  # Valid box
            (5, 15, 10, 20),  # Another valid box
        ]
        for clean_box in valid_clean_boxes:
            params = {"clean_box": clean_box}
            result = _validate_deconv_params(params)
            assert result["clean_box"] == clean_box

        # Invalid clean_box values
        invalid_clean_boxes = [
            (0, 10),  # Only 2 elements
            (0, 10, 0),  # Only 3 elements
            (0, 10, 0, 10, 0),  # 5 elements
            "invalid",  # String
            [],  # Empty list
        ]
        for clean_box in invalid_clean_boxes:
            params = {"clean_box": clean_box}
            with pytest.raises(
                ValueError, match="Clean box must be a 4-tuple.*or None"
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

        # Create 2D test image [ny, nx]
        ny, nx = 32, 32
        image = np.random.rand(ny, nx).astype(np.float32) * 2 - 1  # Range [-1, 1]

        # Set known min/max values
        image[10, 15] = -2.5  # Global minimum
        image[20, 25] = 3.7  # Global maximum

        fmin, fmax = hogbom.maximg(image)

        assert fmin == pytest.approx(-2.5, abs=1e-6)
        assert fmax == pytest.approx(3.7, abs=1e-6)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_basic_float64(self):
        """Test basic maximg functionality with float64"""

        # Create 2D test image [ny, nx]
        ny, nx = 16, 16
        image = np.random.rand(ny, nx).astype(np.float64) * 10

        # Set known min/max values
        image[5, 8] = -15.75
        image[12, 3] = 22.33

        fmin, fmax = hogbom.maximg(image)

        assert fmin == pytest.approx(-15.75, abs=1e-12)
        assert fmax == pytest.approx(22.33, abs=1e-12)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_with_mask(self):
        """Test maximg with mask"""

        ny, nx = 16, 16
        image = np.ones((ny, nx), dtype=np.float32) * 5.0

        # Set extreme values that should be masked out
        image[3, 4] = -100.0  # Should be ignored due to mask
        image[10, 12] = 200.0  # Should be ignored due to mask

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
    def test_maximg_single_plane(self):
        """Test maximg with single 2D plane"""

        ny, nx = 8, 8
        image = np.zeros((ny, nx), dtype=np.float32)
        image[4, 4] = 1.0  # Single point source

        fmin, fmax = hogbom.maximg(image)

        assert fmin == pytest.approx(0.0, abs=1e-6)
        assert fmax == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_multiple_sources(self):
        """Test maximg with multiple sources in 2D plane"""

        ny, nx = 10, 10
        image = np.zeros((ny, nx), dtype=np.float32)

        # Set different values at different locations
        image[2, 3] = -3.0  # Global minimum
        image[4, 5] = 1.5
        image[6, 7] = 2.8  # Global maximum
        image[8, 1] = -1.2

        fmin, fmax = hogbom.maximg(image)

        assert fmin == pytest.approx(-3.0, abs=1e-6)
        assert fmax == pytest.approx(2.8, abs=1e-6)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_error_wrong_dimensions(self):
        """Test maximg error handling for wrong dimensions"""

        # Test 1D array (should fail)
        image_1d = np.random.rand(32).astype(np.float32)
        with pytest.raises(RuntimeError, match="Image must be 2D array"):
            hogbom.maximg(image_1d)

        # Test 3D array (should fail - now expects 2D)
        image_3d = np.random.rand(2, 32, 32).astype(np.float32)
        with pytest.raises(RuntimeError, match="Image must be 2D array"):
            hogbom.maximg(image_3d)

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_maximg_mask_dimension_mismatch(self):
        """Test maximg error handling for mask dimension mismatch"""

        image = np.random.rand(32, 32).astype(np.float32)
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

        resid_image, psf_image = hogbom_images
        dirty_xds = load_image({"residual": resid_image})
        psf_xds = load_image({"point_spread_function": psf_image})

        deconv_params = {"gain": 0.1, "niter": 100}

        # Extract 2D numpy arrays from xarray datasets
        dirty_array = dirty_xds.isel(time=0, frequency=0, polarization=0)[
            "RESIDUAL"
        ].values
        psf_array = psf_xds.isel(time=0, frequency=0, polarization=0)[
            "POINT_SPREAD_FUNCTION"
        ].values

        # Run function
        # Hogbom clean now expects 2D numpy arrays and returns numpy arrays
        # The iteration over time/freq/pol is done in the deconvolve() function
        result, model_array, residual_array = hogbom_clean(
            dirty_array, psf_array, deconv_params
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "iterations_performed" in result
        assert "final_peak" in result
        assert "total_flux_cleaned" in result
        assert result["iterations_performed"] == 100

        # Verify arrays are numpy arrays with correct shape
        assert isinstance(model_array, np.ndarray)
        assert isinstance(residual_array, np.ndarray)
        assert model_array.ndim == 2
        assert residual_array.ndim == 2
        assert model_array.shape == dirty_array.shape
        assert residual_array.shape == dirty_array.shape

        # Verify values
        assert residual_array[125, 129] == pytest.approx(0.004976, abs=1e-5)
        assert model_array[128, 128] == pytest.approx(0.97746, abs=1e-4)

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


class TestReturnDict:
    """Test the ReturnDict structure, indexing, and values"""

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_returndict_structure_single_plane(self, hogbom_images):
        """Test ReturnDict structure with single plane (1 time, 1 freq, 1 pol)"""

        resid_image, psf_image = hogbom_images
        dirty_xds = load_image({"residual": resid_image})
        psf_xds = load_image({"point_spread_function": psf_image})

        deconv_params = {"gain": 0.1, "niter": 100, "threshold": 0.0}

        returndict, model_xds, residual_xds = deconvolve(
            dirty_xds, psf_xds, deconv_params=deconv_params
        )

        # Verify ReturnDict has exactly one entry (1 time × 1 freq × 1 pol)
        assert len(returndict.data) == 1

        # Get the single entry
        key = list(returndict.data.keys())[0]
        entry = returndict.data[key]

        # Verify key structure
        assert key.time == 0
        assert key.pol == 0
        assert key.chan == 0

        # Verify all required fields exist
        required_fields = [
            "niter",
            "threshold",
            "iter_done",
            "loop_gain",
            "min_psf_fraction",
            "max_psf_fraction",
            "max_psf_sidelobe",
            "stop_code",
            "stokes",
            "frequency",
            "phase_center",
            "time",
            "start_model_flux",
            "start_peakres",
            "start_peakres_nomask",
            "peakres",
            "peakres_nomask",
            "masksum",
        ]
        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_returndict_values_consistency(self, hogbom_images):
        """Test that ReturnDict values are internally consistent"""

        resid_image, psf_image = hogbom_images
        dirty_xds = load_image({"residual": resid_image})

        psf_xds = load_image({"point_spread_function": psf_image})

        deconv_params = {"gain": 0.1, "niter": 100, "threshold": 0.001}

        returndict, model_xds, residual_xds = deconvolve(
            dirty_xds, psf_xds, deconv_params=deconv_params
        )

        entry = list(returndict.data.values())[0]

        # Check parameter consistency
        assert entry["niter"] == deconv_params["niter"]
        assert entry["threshold"] == deconv_params["threshold"]
        assert entry["loop_gain"] == deconv_params["gain"]

        # Extract latest values from history-tracked fields (FIELD_ACCUM)
        # These are now stored as lists to enable convergence tracking
        iter_done = (
            entry["iter_done"][-1]
            if isinstance(entry["iter_done"], list)
            else entry["iter_done"]
        )
        peakres = (
            entry["peakres"][-1]
            if isinstance(entry["peakres"], list)
            else entry["peakres"]
        )
        peakres_nomask = (
            entry["peakres_nomask"][-1]
            if isinstance(entry["peakres_nomask"], list)
            else entry["peakres_nomask"]
        )
        masksum = (
            entry["masksum"][-1]
            if isinstance(entry["masksum"], list)
            else entry["masksum"]
        )

        # Check that iterations performed is reasonable
        assert iter_done >= 0
        assert iter_done <= entry["niter"]

        # Check that peak residual decreased (using absolute values)
        # Note: May be NaN if mask excludes all pixels
        if not np.isnan(peakres) and not np.isnan(entry["start_peakres"]):
            assert abs(peakres) <= abs(entry["start_peakres"])
        if not np.isnan(peakres_nomask) and not np.isnan(entry["start_peakres_nomask"]):
            assert abs(peakres_nomask) <= abs(entry["start_peakres_nomask"])

        # Check that PSF fractions are in valid range
        assert 0 < entry["min_psf_fraction"] <= 1
        assert 0 < entry["max_psf_fraction"] <= 1
        assert entry["min_psf_fraction"] <= entry["max_psf_fraction"]

        # Check that PSF sidelobe is in reasonable range [0, 1)
        assert 0 <= entry["max_psf_sidelobe"] < 1

        # Check that masksum is non-negative
        assert masksum >= 0

        # Check coordinate values
        assert entry["stokes"] is not None
        assert entry["frequency"] is not None
        assert entry["time"] is not None
        assert entry["phase_center"] is not None

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_returndict_indexing_single_plane(self, hogbom_images):
        """Test ReturnDict indexing methods with single plane"""

        resid_image, psf_image = hogbom_images
        dirty_xds = load_image({"residual": resid_image})
        psf_xds = load_image({"point_spread_function": psf_image})

        deconv_params = {"gain": 0.1, "niter": 50}

        returndict, _, _ = deconvolve(dirty_xds, psf_xds, deconv_params=deconv_params)

        # Test exact indexing (should return single dict)
        result = returndict.sel(time=0, pol=0, chan=0)
        assert isinstance(result, dict)
        assert "iter_done" in result

        # Test partial indexing - time only
        result = returndict.sel(time=0)
        assert isinstance(result, dict)  # Single match returns dict

        # Test partial indexing - pol only
        result = returndict.sel(pol=0)
        assert isinstance(result, dict)

        # Test partial indexing - chan only
        result = returndict.sel(chan=0)
        assert isinstance(result, dict)

        # Test no indexing (get all)
        # When only one match, returns dict not list
        result = returndict.sel()
        assert isinstance(result, dict)  # Single match
        assert "iter_done" in result

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_returndict_with_multipol_synthetic(self, hogbom_images):
        """Test ReturnDict with synthetic multi-polarization data"""

        resid_image, psf_image = hogbom_images
        dirty_xds = load_image({"residual": resid_image})
        psf_xds = load_image({"point_spread_function": psf_image})

        # Create synthetic multi-pol data by replicating along polarization axis
        # Original shape: [1, 1, 1, ny, nx]
        original_dirty = dirty_xds["RESIDUAL"].values
        original_psf = psf_xds["POINT_SPREAD_FUNCTION"].values

        # Replicate to 4 polarizations
        npol = 4
        multi_dirty = np.repeat(original_dirty, npol, axis=2)
        multi_psf = np.repeat(original_psf, npol, axis=2)

        # Create new coordinates
        pol_coords = ["I", "Q", "U", "V"]

        # Create completely new datasets with proper dimensions
        # Extract original coordinates - copy all coordinates from original
        coords_dict = {
            "time": dirty_xds.coords["time"],
            "frequency": dirty_xds.coords["frequency"],
            "polarization": pol_coords,
            "l": dirty_xds.coords["l"],
            "m": dirty_xds.coords["m"],
        }

        # Add 2D spatial coordinates if present
        if "right_ascension" in dirty_xds.coords:
            coords_dict["right_ascension"] = dirty_xds.coords["right_ascension"]
        if "declination" in dirty_xds.coords:
            coords_dict["declination"] = dirty_xds.coords["declination"]
        if "velocity" in dirty_xds.coords:
            coords_dict["velocity"] = dirty_xds.coords["velocity"]
        if "beam_params_label" in dirty_xds.coords:
            coords_dict["beam_params_label"] = dirty_xds.coords["beam_params_label"]

        # Create new xarray datasets from scratch
        # Include mask if present in original
        data_vars_dirty = {
            "RESIDUAL": (["time", "frequency", "polarization", "l", "m"], multi_dirty),
        }
        data_vars_psf = {
            "POINT_SPREAD_FUNCTION": (
                ["time", "frequency", "polarization", "l", "m"],
                multi_psf,
            ),
        }

        # Add mask if present
        if "MASK_0" in dirty_xds:
            mask_data = np.repeat(dirty_xds["MASK_0"].values, npol, axis=2)
            data_vars_dirty["MASK_0"] = (
                ["time", "frequency", "polarization", "l", "m"],
                mask_data,
            )

        if "MASK_SKY" in dirty_xds:
            mask_data = np.repeat(dirty_xds["MASK_SKY"].values, npol, axis=2)
            data_vars_dirty["MASK_SKY"] = (
                ["time", "frequency", "polarization", "l", "m"],
                mask_data,
            )

        dirty_xds_new = xr.Dataset(
            data_vars_dirty,
            coords=coords_dict,
        )
        # Copy dataset attributes
        if hasattr(dirty_xds, "attrs"):
            dirty_xds_new.attrs = dirty_xds.attrs
        # Copy RESIDUAL DataArray attributes (including active_mask)
        if hasattr(dirty_xds["RESIDUAL"], "attrs"):
            dirty_xds_new["RESIDUAL"].attrs = dirty_xds["RESIDUAL"].attrs

        # Create PSF coords dict (same as dirty)
        psf_coords_dict = {
            "time": psf_xds.coords["time"],
            "frequency": psf_xds.coords["frequency"],
            "polarization": pol_coords,
            "l": psf_xds.coords["l"],
            "m": psf_xds.coords["m"],
        }
        if "right_ascension" in psf_xds.coords:
            psf_coords_dict["right_ascension"] = psf_xds.coords["right_ascension"]
        if "declination" in psf_xds.coords:
            psf_coords_dict["declination"] = psf_xds.coords["declination"]
        if "velocity" in psf_xds.coords:
            psf_coords_dict["velocity"] = psf_xds.coords["velocity"]
        if "beam_param" in psf_xds.coords:
            psf_coords_dict["beam_param"] = psf_xds.coords["beam_param"]

        psf_xds_new = xr.Dataset(
            data_vars_psf,
            coords=psf_coords_dict,
        )
        # Copy dataset attributes
        if hasattr(psf_xds, "attrs"):
            psf_xds_new.attrs = psf_xds.attrs
        # Copy POINT_SPREAD_FUNCTION DataArray attributes
        if hasattr(psf_xds["POINT_SPREAD_FUNCTION"], "attrs"):
            psf_xds_new["POINT_SPREAD_FUNCTION"].attrs = psf_xds[
                "POINT_SPREAD_FUNCTION"
            ].attrs

        deconv_params = {"gain": 0.1, "niter": 50}

        returndict, _, _ = deconvolve(
            dirty_xds_new, psf_xds_new, deconv_params=deconv_params
        )

        # Verify we have 4 entries (1 time × 1 freq × 4 pol)
        assert len(returndict.data) == 4

        # Test indexing by polarization
        for pp in range(npol):
            result = returndict.sel(pol=pp)
            assert isinstance(result, dict)
            assert result["stokes"] == pol_coords[pp]

        # Test getting all entries for a specific time
        results = returndict.sel(time=0)
        assert isinstance(results, list)
        assert len(results) == 4

        # Test getting all entries for specific channel
        results = returndict.sel(chan=0)
        assert isinstance(results, list)
        assert len(results) == 4

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_returndict_with_multichan_synthetic(self, hogbom_images):
        """Test ReturnDict with synthetic multi-channel data"""

        resid_image, psf_image = hogbom_images
        dirty_xds = load_image({"residual": resid_image})
        psf_xds = load_image({"point_spread_function": psf_image})

        # Create synthetic multi-channel data
        original_dirty = dirty_xds["RESIDUAL"].values
        original_psf = psf_xds["POINT_SPREAD_FUNCTION"].values

        # Replicate to 3 channels
        nchan = 3
        multi_dirty = np.repeat(original_dirty, nchan, axis=1)
        multi_psf = np.repeat(original_psf, nchan, axis=1)

        # Create new frequency coordinates
        freq_coords = [1.4e9, 1.5e9, 1.6e9]

        # Create completely new datasets with proper dimensions
        # Extract original coordinates - copy all coordinates from original
        coords_dict = {
            "time": dirty_xds.coords["time"],
            "frequency": freq_coords,
            "polarization": dirty_xds.coords["polarization"],
            "l": dirty_xds.coords["l"],
            "m": dirty_xds.coords["m"],
        }

        # Add 2D spatial coordinates if present
        if "right_ascension" in dirty_xds.coords:
            coords_dict["right_ascension"] = dirty_xds.coords["right_ascension"]
        if "declination" in dirty_xds.coords:
            coords_dict["declination"] = dirty_xds.coords["declination"]
        if "beam_param" in dirty_xds.coords:
            coords_dict["beam_param"] = dirty_xds.coords["beam_param"]
        # Note: velocity coord will be recreated per-channel, so skip it

        # Create new xarray datasets from scratch
        data_vars_dirty = {
            "RESIDUAL": (["time", "frequency", "polarization", "l", "m"], multi_dirty),
        }
        data_vars_psf = {
            "POINT_SPREAD_FUNCTION": (
                ["time", "frequency", "polarization", "l", "m"],
                multi_psf,
            ),
        }

        # Add mask if present
        if "MASK_0" in dirty_xds:
            mask_data = np.repeat(dirty_xds["MASK_0"].values, nchan, axis=1)
            data_vars_dirty["MASK_0"] = (
                ["time", "frequency", "polarization", "l", "m"],
                mask_data,
            )

        if "MASK_SKY" in dirty_xds:
            mask_data = np.repeat(dirty_xds["MASK_SKY"].values, nchan, axis=1)
            data_vars_dirty["MASK_SKY"] = (
                ["time", "frequency", "polarization", "l", "m"],
                mask_data,
            )

        dirty_xds_new = xr.Dataset(
            data_vars_dirty,
            coords=coords_dict,
        )
        # Copy dataset attributes
        if hasattr(dirty_xds, "attrs"):
            dirty_xds_new.attrs = dirty_xds.attrs
        # Copy RESIDUAL DataArray attributes (including active_mask)
        if hasattr(dirty_xds["RESIDUAL"], "attrs"):
            dirty_xds_new["RESIDUAL"].attrs = dirty_xds["RESIDUAL"].attrs

        # Create PSF coords dict (same as dirty)
        psf_coords_dict = {
            "time": psf_xds.coords["time"],
            "frequency": freq_coords,
            "polarization": psf_xds.coords["polarization"],
            "l": psf_xds.coords["l"],
            "m": psf_xds.coords["m"],
        }
        if "right_ascension" in psf_xds.coords:
            psf_coords_dict["right_ascension"] = psf_xds.coords["right_ascension"]
        if "declination" in psf_xds.coords:
            psf_coords_dict["declination"] = psf_xds.coords["declination"]
        if "beam_param" in psf_xds.coords:
            psf_coords_dict["beam_param"] = psf_xds.coords["beam_param"]

        psf_xds_new = xr.Dataset(
            data_vars_psf,
            coords=psf_coords_dict,
        )
        # Copy dataset attributes
        if hasattr(psf_xds, "attrs"):
            psf_xds_new.attrs = psf_xds.attrs
        # Copy POINT_SPREAD_FUNCTION DataArray attributes
        if hasattr(psf_xds["POINT_SPREAD_FUNCTION"], "attrs"):
            psf_xds_new["POINT_SPREAD_FUNCTION"].attrs = psf_xds[
                "POINT_SPREAD_FUNCTION"
            ].attrs

        deconv_params = {"gain": 0.1, "niter": 50}

        returndict, _, _ = deconvolve(
            dirty_xds_new, psf_xds_new, deconv_params=deconv_params
        )

        # Verify we have 3 entries (1 time × 3 freq × 1 pol)
        assert len(returndict.data) == 3

        # Test indexing by channel
        for cc in range(nchan):
            result = returndict.sel(chan=cc)
            assert isinstance(result, dict)
            assert result["frequency"] == freq_coords[cc]

        # Test getting all channels for specific polarization
        results = returndict.sel(pol=0)
        assert isinstance(results, list)
        assert len(results) == 3

        # Verify frequencies are different across channels
        freqs = [float(returndict.sel(chan=cc)["frequency"]) for cc in range(nchan)]
        assert len(set(freqs)) == nchan  # All unique
        assert np.allclose(freqs, freq_coords)


class TestDeconvolve:
    """Test the main deconvolve orchestration function"""

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_deconvolve_basic(self, hogbom_images):
        """Test basic deconvolve functionality with single plane"""

        resid_image, psf_image = hogbom_images
        dirty_xds = load_image({"residual": resid_image})
        psf_xds = load_image({"point_spread_function": psf_image})

        deconv_params = {"gain": 0.1, "niter": 100, "threshold": 0.0}

        # Run deconvolve
        returndict, model_xds, residual_xds = deconvolve(
            dirty_xds, psf_xds, deconv_params=deconv_params
        )

        # Verify ReturnDict structure
        assert isinstance(returndict, ReturnDict)
        assert len(returndict.data) > 0

        # Verify model and residual output
        assert isinstance(model_xds, xr.Dataset)
        assert isinstance(residual_xds, xr.Dataset)
        assert "MODEL" in model_xds
        assert "RESIDUAL" in residual_xds

    @pytest.mark.skipif(
        not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
    )
    def test_deconvolve_returndict_contents(self, hogbom_images):
        """Test that ReturnDict contains expected fields"""

        resid_image, psf_image = hogbom_images
        dirty_xds = load_image({"residual": resid_image})
        psf_xds = load_image({"point_spread_function": psf_image})

        deconv_params = {"gain": 0.1, "niter": 50, "threshold": 0.0}

        returndict, _, _ = deconvolve(dirty_xds, psf_xds, deconv_params=deconv_params)

        # Get first entry from ReturnDict
        if len(returndict.data) > 0:
            first_key = list(returndict.data.keys())[0]
            entry = returndict.data[first_key]

            # Verify expected fields exist
            expected_fields = [
                "iter_done",
                "niter",
                "threshold",
                "loop_gain",
                "peakres",
                "peakres_nomask",
                "masksum",
                "max_psf_sidelobe",
            ]
            for field in expected_fields:
                assert field in entry, f"Missing field: {field}"

        # @pytest.mark.skipif(
        #    not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
        # )
        # def test_deconvolve_with_initial_model(self, hogbom_images):
        #    """Test deconvolve with an initial model image"""

        #    resid_image, psf_image = hogbom_images
        #    dirty_xds = load_image(resid_image)
        #    psf_xds = load_image(psf_image)

        #    # Create a simple initial model
        #    model_xds = dirty_xds.copy(deep=True)
        #    model_xds["SKY"].values[:] = 0.0  # Zero model to start

        #    deconv_params = {"gain": 0.1, "niter": 50, "threshold": 0.0}

        #    # Run with initial model
        #    returndict, final_model, residual_xds = deconvolve(
        #        dirty_xds, psf_xds, model_xds=model_xds, deconv_params=deconv_params
        #    )

        #    # Verify outputs are valid
        #    assert isinstance(returndict, ReturnDict)
        #    assert isinstance(final_model, xr.Dataset)
        #    assert isinstance(residual_xds, xr.Dataset)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
