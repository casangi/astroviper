"""
Test examples to demonstrate usage and validate functionality
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


class TestRealisticExamples:
    """Test with realistic astronomy examples"""
    
    def create_gaussian_beam(self, size, sigma):
        """Create a 2D Gaussian beam"""
        y, x = np.ogrid[:size, :size]
        center = size // 2
        return np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    
    def create_point_source_map(self, map_size, sources):
        """Create a map with point sources
        
        Args:
            map_size: Size of the map (square)
            sources: List of tuples (x, y, flux)
        """
        dirty_map = np.zeros((map_size, map_size))
        for x, y, flux in sources:
            if 0 <= x < map_size and 0 <= y < map_size:
                dirty_map[y, x] = flux
        return dirty_map
    
    def test_single_point_source(self):
        """Test cleaning a single point source"""
        # Parameters
        map_size = 128
        beam_size = 32
        beam_sigma = 4.0
        source_flux = 1.0
        
        # Create beam and dirty map
        beam = self.create_gaussian_beam(beam_size, beam_sigma)
        sources = [(64, 64, source_flux)]  # Center source
        dirty_map = self.create_point_source_map(map_size, sources)
        
        # Add realistic noise
        np.random.seed(42)
        noise_level = 0.01
        dirty_map += noise_level * np.random.normal(size=dirty_map.shape)
        
        # Clean the map
        results = hc.clean_with_components(
            dirty_map.copy(),
            beam,
            gain=0.1,
            threshold=3 * noise_level,  # 3-sigma threshold
            max_iter=1000
        )
        
        # Validate results
        assert results['iterations'] > 0
        assert len(results['component_flux']) > 0
        
        # Check that the brightest component is near the source location
        max_component_idx = np.argmax(np.abs(results['component_flux']))
        max_x = results['component_x'][max_component_idx]
        max_y = results['component_y'][max_component_idx]
        
        # Should be within a few pixels of the true source
        assert abs(max_x - 64) < 5
        assert abs(max_y - 64) < 5
        
        # Total cleaned flux should be close to source flux
        assert abs(results['total_flux_cleaned'] - source_flux) < 0.1
    
    def test_multiple_point_sources(self):
        """Test cleaning multiple point sources"""
        # Parameters
        map_size = 256
        beam_size = 32
        beam_sigma = 4.0
        
        # Create multiple sources
        sources = [
            (64, 64, 1.0),    # Bright source
            (128, 128, 0.8),  # Medium source  
            (192, 64, 0.6),   # Weaker source
            (64, 192, 0.4),   # Weakest source
        ]
        
        # Create beam and dirty map
        beam = self.create_gaussian_beam(beam_size, beam_sigma)
        dirty_map = self.create_point_source_map(map_size, sources)
        
        # Add noise
        np.random.seed(123)
        noise_level = 0.02
        dirty_map += noise_level * np.random.normal(size=dirty_map.shape)
        
        # Clean the map
        results = hc.clean_with_components(
            dirty_map.copy(),
            beam,
            gain=0.05,  # Conservative gain
            threshold=3 * noise_level,
            max_iter=2000
        )
        
        # Should find multiple components
        assert len(results['component_flux']) >= len(sources)
        assert results['iterations'] > 0
        
        # Total flux should be reasonable
        total_source_flux = sum(flux for _, _, flux in sources)
        assert abs(results['total_flux_cleaned'] - total_source_flux) < 0.2
    
    def test_extended_source_simulation(self):
        """Test with a simulated extended source"""
        # Parameters
        map_size = 128
        beam_size = 24
        beam_sigma = 3.0
        
        # Create extended Gaussian source
        y, x = np.ogrid[:map_size, :map_size]
        center_x, center_y = 64, 64
        source_sigma = 15.0
        peak_flux = 1.0
        
        # Extended Gaussian source
        dirty_map = peak_flux * np.exp(
            -((x - center_x)**2 + (y - center_y)**2) / (2 * source_sigma**2)
        )
        
        # Create beam
        beam = self.create_gaussian_beam(beam_size, beam_sigma)
        
        # Add noise
        np.random.seed(456)
        noise_level = 0.01
        dirty_map += noise_level * np.random.normal(size=dirty_map.shape)
        
        # Clean the map with conservative parameters
        results = hc.clean_with_components(
            dirty_map.copy(),
            beam,
            gain=0.02,  # Very conservative for extended source
            threshold=2 * noise_level,
            max_iter=3000
        )
        
        # Should find many components for extended source
        assert len(results['component_flux']) > 50
        assert results['iterations'] > 100
        
        # Components should be concentrated near source center
        comp_x = np.array(results['component_x'])
        comp_y = np.array(results['component_y'])
        distances = np.sqrt((comp_x - center_x)**2 + (comp_y - center_y)**2)
        
        # Most components should be within 3*source_sigma of center
        close_components = np.sum(distances < 3 * source_sigma)
        assert close_components > 0.5 * len(distances)
    
    def test_clean_with_mask(self):
        """Test cleaning with a restricted clean window (mask)"""
        # Create dirty map with sources inside and outside mask region
        map_size = 128
        beam_size = 16
        beam = self.create_gaussian_beam(beam_size, 2.0)
        
        sources = [
            (32, 32, 1.0),   # Inside mask
            (96, 96, 0.8),   # Outside mask
        ]
        dirty_map = self.create_point_source_map(map_size, sources)
        
        # Define clean window (mask) covering only lower-left quadrant
        clean_window = (0, 64, 0, 64)
        
        # Clean with mask
        results = hc.clean_with_components(
            dirty_map.copy(),
            beam,
            gain=0.1,
            threshold=0.01,
            clean_window=clean_window
        )
        
        # All components should be within the clean window
        for x, y in zip(results['component_x'], results['component_y']):
            assert 0 <= x < 64
            assert 0 <= y < 64
        
        # Should find the source inside the mask
        assert len(results['component_flux']) > 0


class TestComparisonWithFortran:
    """Tests that compare behavior with original Fortran implementation"""
    
    def test_fortran_parameter_mapping(self):
        """Test that our parameters map correctly to Fortran version"""
        # This test ensures our API matches the original Fortran subroutine
        # SUBROUTINE HCLEAN(MAP,NXP,NYP,BEAM,NXB,NYB,GAIN,THRESH,NITER,
        #                   XBEG,XEND,YBEG,YEND,NPOL)
        
        map_size_x, map_size_y = 64, 64
        beam_size_x, beam_size_y = 16, 16
        
        # Create test data
        dirty_map = np.random.random((map_size_y, map_size_x))
        beam = np.random.random((beam_size_y, beam_size_x))
        
        # Fortran parameters
        gain = 0.1
        thresh = 0.01
        niter = 100
        xbeg, xend = 10, 54  # 1-indexed in Fortran
        ybeg, yend = 10, 54
        
        # Convert to our 0-indexed clean window
        clean_window = (xbeg-1, xend, ybeg-1, yend)  # Convert to 0-indexed
        
        # Should not crash and should respect parameters
        results = hc.clean_with_components(
            dirty_map,
            beam,
            gain=gain,
            threshold=thresh,
            max_iter=niter,
            clean_window=clean_window
        )
        
        # Basic sanity checks
        assert isinstance(results, dict)
        assert results['iterations'] <= niter
    
    def test_gain_behavior(self):
        """Test that gain parameter behaves as expected"""
        # Create identical test case
        map_size = 64
        beam_size = 16
        
        dirty_map = np.zeros((map_size, map_size))
        dirty_map[32, 32] = 1.0
        beam = self.create_gaussian_beam(beam_size, 2.0)
        
        # Test with different gain values
        gains = [0.01, 0.1, 0.5]
        results_by_gain = {}
        
        for gain in gains:
            results = hc.clean_with_components(
                dirty_map.copy(),
                beam,
                gain=gain,
                threshold=0.001,
                max_iter=1000
            )
            results_by_gain[gain] = results
        
        # Higher gain should converge faster (fewer iterations)
        # but might be less stable
        assert (results_by_gain[0.1]['iterations'] <= 
                results_by_gain[0.01]['iterations'])
    
    def create_gaussian_beam(self, size, sigma):
        """Helper to create Gaussian beam"""
        y, x = np.ogrid[:size, :size]
        center = size // 2
        return np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_beam(self):
        """Test behavior with zero beam"""
        dirty_map = np.random.random((32, 32))
        zero_beam = np.zeros((8, 8))
        
        # Should handle gracefully (though not physically meaningful)
        results = hc.clean_with_components(
            dirty_map,
            zero_beam,
            gain=0.1,
            threshold=0.01,
            max_iter=10
        )
        
        # Should perform iterations but not clean much
        assert results['iterations'] >= 0
    
    def test_very_small_arrays(self):
        """Test with very small arrays"""
        dirty_map = np.array([[1.0, 0.5], [0.3, 0.2]])
        beam = np.array([[1.0]])
        
        results = hc.clean_with_components(
            dirty_map,
            beam,
            gain=0.1,
            threshold=0.05,
            max_iter=10
        )
        
        assert results['iterations'] >= 0
        assert isinstance(results['component_flux'], list)
    
    def test_beam_larger_than_map(self):
        """Test when beam is larger than dirty map"""
        dirty_map = np.random.random((16, 16))
        large_beam = np.random.random((32, 32))
        
        # Should handle gracefully by using only the overlapping region
        results = hc.clean_with_components(
            dirty_map,
            large_beam,
            gain=0.1,
            threshold=0.01,
            max_iter=10
        )
        
        # Should not crash
        assert results['iterations'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])