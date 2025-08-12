# Hogbom CLEAN Algorithm

A high-performance implementation of the Hogbom CLEAN algorithm for radio astronomy image deconvolution.

## Overview

The Hogbom CLEAN algorithm is a fundamental deconvolution method used in radio astronomy to remove the effects of the instrument's point spread function (PSF) from observed images. This implementation provides:

- Fast C++ core with Python bindings
- Support for single and multi-polarization cleaning
- NumPy array interface for easy integration
- Configurable cleaning parameters and windows

## Installation

```bash
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import hogbom

# Create sample data
dirty_map = np.random.random((256, 256))
beam = np.random.random((64, 64))

# Run CLEAN algorithm
residual = hogbom.clean(dirty_map, beam, gain=0.1, threshold=0.01)

# Get full results including components
results = hogbom.clean_with_components(dirty_map, beam)
print(f"Cleaned {results['iterations']} iterations")
print(f"Found {len(results['component_flux'])} components")
```

## API Reference

### Main Functions

#### `clean(dirty_map, beam, gain=0.1, threshold=0.0, max_iter=100, clean_window=(-1,-1,-1,-1))`

Basic CLEAN algorithm that returns only the residual map.

**Parameters:**
- `dirty_map` (np.ndarray): 2D input dirty map
- `beam` (np.ndarray): 2D clean beam (PSF)  
- `gain` (float): Loop gain (fraction of peak to subtract each iteration)
- `threshold` (float): Cleaning threshold (stop when peak < threshold)
- `max_iter` (int): Maximum number of iterations
- `clean_window` (tuple): Clean window bounds (x_start, x_end, y_start, y_end)

**Returns:**
- `np.ndarray`: Residual map after cleaning

#### `clean_with_components(dirty_map, beam, ...)`

Full CLEAN algorithm that returns components and statistics.

**Returns:**
- `dict`: Dictionary containing:
  - `residual_map`: Cleaned residual map
  - `component_flux`: Array of component flux values
  - `component_x`: Array of component x positions  
  - `component_y`: Array of component y positions
  - `iterations`: Number of iterations performed
  - `final_peak`: Final peak value in residual
  - `total_flux_cleaned`: Total flux cleaned

#### `clean_multipol(dirty_maps, beam, ...)`

Multi-polarization CLEAN algorithm.

**Parameters:**
- `dirty_maps` (np.ndarray): 3D array (npol, ny, nx) of dirty maps
- Other parameters same as `clean()`

**Returns:**
- `dict`: Same as `clean_with_components()` but with `residual_maps` (3D)

### Utility Functions

#### `find_peak(data, window=(-1,-1,-1,-1))`

Find peak pixel in 2D array within specified window.

**Returns:**
- `tuple`: (peak_value, peak_x, peak_y)

## Algorithm Details

The Hogbom CLEAN algorithm follows these steps:

1. Find the peak pixel in the dirty map (within clean window if specified)
2. If peak < threshold or max iterations reached, stop
3. Subtract a scaled version of the beam centered at the peak location
4. Record the component (position and flux)  
5. Repeat from step 1

The algorithm is based on the original Fortran implementation from CASACORE, adapted for modern C++ with Python bindings.

## Performance

This implementation is optimized for performance:

- C++ core with compiler optimizations
- Memory-efficient in-place operations  
- NumPy integration for zero-copy data access
- Support for different data types and layouts

## Examples

### Basic Cleaning
```python
import numpy as np
import hogbom

# Create synthetic dirty map with point source
dirty = np.zeros((128, 128))
dirty[64, 64] = 1.0  # Point source at center

# Gaussian beam
y, x = np.ogrid[:32, :32]
beam = np.exp(-((x-16)**2 + (y-16)**2) / (2*3**2))

# Clean the image
results = hogbom.clean_with_components(dirty, beam, gain=0.1, threshold=0.01)

print(f"Found {len(results['component_flux'])} components")
print(f"Peak component at ({results['component_x'][0]}, {results['component_y'][0]})")
```

### Multi-polarization Cleaning
```python
# Create 4-polarization data (I, Q, U, V)
npol = 4
dirty_maps = np.random.random((npol, 128, 128))

# Clean all polarizations
results = hogbom.clean_multipol(dirty_maps, beam, gain=0.05)

print(f"Residual shape: {results['residual_maps'].shape}")
```

### Custom Clean Window
```python
# Clean only central region
window = (32, 96, 32, 96)  # x_start, x_end, y_start, y_end
results = hogbom.clean_with_components(dirty, beam, clean_window=window)
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run benchmarks:
```bash
pytest tests/ -m benchmark
```

## License

MIT License - see LICENSE file for details.