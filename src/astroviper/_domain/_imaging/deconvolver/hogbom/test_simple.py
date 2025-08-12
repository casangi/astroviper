#!/usr/bin/env python3
"""
Simple test to verify the hogbom package works
"""
import numpy as np

try:
    import hogbom
    print("✓ Successfully imported hogbom package")
    
    # Test basic functionality
    dirty_map = np.random.random((32, 32))
    beam = np.random.random((8, 8))
    
    print("✓ Created test arrays")
    
    # Test basic clean
    residual = hogbom.clean(dirty_map.copy(), beam, max_iter=10)
    print(f"✓ Basic clean works, residual shape: {residual.shape}")
    
    # Test clean with components
    results = hogbom.clean_with_components(dirty_map.copy(), beam, max_iter=10)
    print(f"✓ Clean with components works, iterations: {results['iterations']}")
    
    # Test find_peak
    peak_val, peak_x, peak_y = hogbom.find_peak(dirty_map)
    print(f"✓ Find peak works: peak={peak_val:.3f} at ({peak_x}, {peak_y})")
    
    print("\n🎉 All tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()