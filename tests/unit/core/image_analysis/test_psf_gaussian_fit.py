import dask.array as da
import numpy as np
import xarray as xr
from astroviper.core.image_analysis.psf_gaussian_fit import psf_gaussian_fit


def create_test_xds(shape=(1, 1, 1, 9, 9)):
    # Create a simple 2D Gaussian as test data
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    gaussian = np.exp(-(xv**2 + yv**2) / (2 * 0.2**2))
    data = np.zeros(shape)
    print("data = ", data)
    data[0, 0, 0, :, :] = gaussian
    da_data = da.from_array(data, chunks=(1, 1, 1, 9, 9))
    dims = ["time", "frequency", "polarization", "l", "m"]
    data_coords = {
        "time": np.arange(shape[0]),
        "frequency": np.arange(shape[1]),
        "polarization": np.arange(shape[2]),
        "l": np.linspace(-1, 1, shape[-2]),
        "m": np.linspace(-1, 1, shape[-1]),
    }
    sky = xr.DataArray(da_data, dims=dims, coords=data_coords)

    beam_data = np.zeros((1, shape[1], shape[2], 3))
    da_beam_data = da.from_array(beam_data, chunks=(1, 1, 1, 1))
    beam_dims = ["time", "frequency", "polarization", "beam_param"]
    beam_coords = {
        "time": np.arange(shape[0]),
        "frequency": np.arange(shape[1]),
        "polarization": np.arange(shape[2]),
        "beam_param": np.arange(3),
    }
    beam = xr.DataArray(da_beam_data, dims=beam_dims, coords=beam_coords)

    test_dataset = xr.Dataset({"SKY": sky, "BEAM": beam})

    return test_dataset


def test_psf_gaussian_fit_basic():
    test_dataset = create_test_xds()
    result = psf_gaussian_fit(test_dataset)
    assert "BEAM" in result
    assert "beam_param" in result["BEAM"].dims
    params = result["BEAM"]["beam_param"]
    assert params.shape[-1] == 3
    # Check that the fitted widths are positive
    assert np.all(result["BEAM"].data[:, :, :-1] > 0)


def test_psf_gaussian_fit_custom_window():
    test_dataset = create_test_xds(shape=(1, 1, 1, 15, 15))
    result = psf_gaussian_fit(test_dataset, npix_window=[7, 7], sampling=[7, 7])
    params = result["BEAM"]["beam_param"]
    assert params.shape == (3,)
    assert np.all(result["BEAM"].data[:, :, :-1] > 0)
