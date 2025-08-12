"""
Tests for Hogbom CLEAN algorithm implementation
"""

import numpy as np
import pytest
import sys
import os

# Add the python directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import hogbom_clean as hc
except ImportError:
    pytest.skip("hogbom_clean module not available", allow_module_level=True)


class TestBasicFunctionality:
    """Test basic functionality of the CLEAN algorithm"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create a simple 2D Gaussian beam
        self.beam_size = 32
        y, x = np.ogrid[:self.beam_size, :self.beam_size]
        center = self.beam_size // 2
        sigma = 3.0
        self.beam = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        
        # Create a simple dirty map with point source
        self.map_size = 128
        self.dirty_map = np.zeros((self.map_size, self.map_size))
        self.dirty_map[64, 64] = 1.0  # Point source at center
        
        # Add some noise
        np.random.seed(42)
        self.dirty_map += 0.01 * np.random.random(self.dirty_map.shape)
    
    def test_clean_basic(self):
        """Test basic CLEAN functionality"""
        residual = hc.clean(
            self.dirty_map.copy(), 
            self.beam, 
            gain=0.1, 
            threshold=0.01,
            max_iter=100
        )
        
        # Check that residual is modified
        assert not np.allclose(residual, self.dirty_map)
        
        # Check that peak is reduced
        original_peak = np.max(np.abs(self.dirty_map))
        residual_peak = np.max(np.abs(residual))
        assert residual_peak < original_peak
    
    def test_clean_with_components(self):
        """Test CLEAN with component extraction"""
        results = hc.clean_with_components(
            self.dirty_map.copy(),
            self.beam,
            gain=0.1,
            threshold=0.01,
            max_iter=100
        )
        
        # Check return structure
        required_keys = [
            'residual_map', 'component_flux', 'component_x', 'component_y',
            'iterations', 'final_peak', 'total_flux_cleaned'
        ]
        for key in required_keys:
            assert key in results
        
        # Check that components were found
        assert len(results['component_flux']) > 0
        assert len(results['component_x']) == len(results['component_flux'])
        assert len(results['component_y']) == len(results['component_flux'])
        
        # Check that iterations were performed
        assert results['iterations'] > 0
        
        # Check that flux was cleaned
        assert results['total_flux_cleaned'] > 0
    
    def test_clean_convergence(self):
        """Test that CLEAN converges properly"""
        # High threshold should converge quickly
        results = hc.clean_with_components(
            self.dirty_map.copy(),
            self.beam,
            gain=0.1,
            threshold=0.1,  # High threshold
            max_iter=100
        )
        
        # Should converge due to threshold
        assert results['final_peak'] <= 0.1
        assert results['iterations'] < 100
    
    def test_clean_max_iterations(self):
        """Test that CLEAN respects max iterations"""
        results = hc.clean_with_components(
            self.dirty_map.copy(),
            self.beam,
            gain=0.01,  # Small gain
            threshold=0.0,  # No threshold
            max_iter=10  # Few iterations
        )
        
        # Should stop due to max iterations
        assert results['iterations'] == 10


class TestCleanWindow:
    """Test CLEAN window functionality"""
    
    def setup_method(self):
        """Set up test fixtures with multiple sources"""
        # Create beam
        self.beam_size = 16
        y, x = np.ogrid[:self.beam_size, :self.beam_size]
        center = self.beam_size // 2
        self.beam = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2**2))
        
        # Create dirty map with two sources
        self.map_size = 64
        self.dirty_map = np.zeros((self.map_size, self.map_size))
        self.dirty_map[16, 16] = 1.0  # Source in lower left quadrant
        self.dirty_map[48, 48] = 0.8  # Source in upper right quadrant
    
    def test_full_window_cleaning(self):
        """Test cleaning with full window"""
        results = hc.clean_with_components(
            self.dirty_map.copy(),
            self.beam,
            gain=0.1,
            threshold=0.01,
            clean_window=(-1, -1, -1, -1)  # Full window
        )
        
        # Should find both sources
        assert len(results['component_flux']) >= 2
    
    def test_partial_window_cleaning(self):
        """Test cleaning with restricted window"""
        # Clean only lower left quadrant
        results = hc.clean_with_components(
            self.dirty_map.copy(),
            self.beam,
            gain=0.1,
            threshold=0.01,
            clean_window=(0, 32, 0, 32)  # Lower left quadrant
        )
        
        # Should primarily find the source in that quadrant
        components_x = np.array(results['component_x'])
        components_y = np.array(results['component_y'])
        
        # Most components should be in the specified window
        in_window = (components_x < 32) & (components_y < 32)
        assert np.sum(in_window) > len(components_x) // 2


class TestMultipolarization:
    """Test multi-polarization CLEAN"""
    
    def setup_method(self):
        """Set up multi-pol test data"""
        # Create beam
        self.beam_size = 16
        y, x = np.ogrid[:self.beam_size, :self.beam_size]
        center = self.beam_size // 2
        self.beam = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2**2))
        
        # Create 4-pol dirty maps (I, Q, U, V)
        self.npol = 4
        self.map_size = 64
        self.dirty_maps = np.zeros((self.npol, self.map_size, self.map_size))
        
        # Add sources with different polarization properties
        self.dirty_maps[0, 32, 32] = 1.0  # Stokes I
        self.dirty_maps[1, 32, 32] = 0.1  # Stokes Q
        self.dirty_maps[2, 32, 32] = 0.05  # Stokes U
        self.dirty_maps[3, 32, 32] = 0.02  # Stokes V
    
    def test_multipol_clean(self):
        """Test multi-polarization cleaning"""
        results = hc.clean_multipol(
            self.dirty_maps.copy(),
            self.beam,
            gain=0.1,
            threshold=0.01,
            max_iter=50
        )
        
        # Check return structure
        required_keys = [
            'residual_maps', 'component_flux', 'component_x', 'component_y',
            'iterations', 'final_peak', 'total_flux_cleaned'
        ]
        for key in required_keys:
            assert key in results
        
        # Check shapes
        assert results['residual_maps'].shape == (self.npol, self.map_size, self.map_size)
        assert len(results['component_flux']) > 0
        
        # Check that components were found
        assert results['iterations'] > 0
        assert results['total_flux_cleaned'] > 0


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_find_peak(self):
        """Test peak finding function"""
        # Create test data with known peak
        data = np.random.random((32, 32))
        data[15, 20] = 2.0  # Known peak
        
        peak_val, peak_x, peak_y = hc.find_peak(data, window=(0, -1, 0, -1))
        
        assert peak_val == 2.0
        assert peak_x == 20
        assert peak_y == 15
    
    def test_find_peak_with_window(self):
        """Test peak finding with restricted window"""
        data = np.random.random((32, 32))
        data[5, 5] = 2.0    # Peak outside window
        data[15, 15] = 1.5  # Peak inside window
        
        # Search only in right half
        peak_val, peak_x, peak_y = hc.find_peak(data, window=(10, 32, 10, 32))
        
        assert peak_val == 1.5
        assert peak_x == 15
        assert peak_y == 15


class TestParameterStructures:
    """Test parameter and result structures"""
    
    def test_clean_params(self):
        """Test CleanParams structure"""
        params = hc.CleanParams()
        
        # Test default values
        assert params.gain == 0.1
        assert params.threshold == 0.0
        assert params.max_iter == 100
        assert params.x_begin == 0
        assert params.x_end == -1
        assert params.y_begin == 0
        assert params.y_end == -1
        
        # Test modification
        params.gain = 0.05
        params.threshold = 0.01
        assert params.gain == 0.05
        assert params.threshold == 0.01
    
    def test_clean_results(self):
        """Test CleanResults structure"""
        results = hc.CleanResults()
        
        # Test initial state
        assert len(results.component_flux) == 0
        assert len(results.component_x) == 0
        assert len(results.component_y) == 0
        assert results.iterations_performed == 0
        assert results.final_peak == 0.0
        assert results.total_flux_cleaned == 0.0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_dimensions(self):
        """Test handling of invalid array dimensions"""
        # 1D array should raise error
        with pytest.raises(RuntimeError):
            hc.clean(np.array([1, 2, 3]), np.array([[1, 2], [3, 4]]))
        
        # 3D array for single-pol should raise error
        with pytest.raises(RuntimeError):
            hc.clean(np.random.random((2, 32, 32)), np.random.random((16, 16)))
    
    def test_empty_arrays(self):
        """Test handling of empty arrays"""
        with pytest.raises((RuntimeError, ValueError)):
            hc.clean(np.array([[]]), np.array([[]]))
    
    def test_invalid_clean_window(self):
        """Test handling of invalid clean windows"""
        dirty = np.random.random((32, 32))
        beam = np.random.random((8, 8))
        
        # Window outside array bounds should be handled gracefully
        # (should clip to valid bounds)
        results = hc.clean_with_components(
            dirty, beam, clean_window=(100, 200, 100, 200)
        )
        # Should not crash, but may find no components


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks"""
    
    def test_clean_performance(self, benchmark):
        """Benchmark basic CLEAN performance"""
        # Create larger test case
        map_size = 512
        beam_size = 64
        
        # Create dirty map with multiple sources
        dirty_map = np.zeros((map_size, map_size))
        np.random.seed(42)
        
        # Add random sources
        for _ in range(10):
            x = np.random.randint(beam_size, map_size - beam_size)
            y = np.random.randint(beam_size, map_size - beam_size)
            flux = np.random.uniform(0.1, 1.0)
            dirty_map[y, x] = flux
        
        # Create beam
        y, x = np.ogrid[:beam_size, :beam_size]
        center = beam_size // 2
        beam = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 8**2))
        
        # Benchmark
        def run_clean():
            return hc.clean_with_components(
                dirty_map.copy(), beam, gain=0.1, threshold=0.01, max_iter=200
            )
        
        result = benchmark(run_clean)
        assert result['iterations'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])