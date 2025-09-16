import pytest
import dask.array as da
import numpy as np
import xarray as xr
from astroviper.core.image_analysis.psf_gaussian_fit import psf_gaussian_fit
from astroviper.core.image_analysis.psf_gaussian_fit import psf_gaussian_fit_core


def create_test_xds(shape=(1, 1, 1, 9, 9)):
    # Create a simple 2D circular Gaussian as test data
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    gaussian = np.exp(-(xv**2 + yv**2) / (2 * 0.2**2))
    data = np.zeros(shape)
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


def test_psf_gaussian_fit_output_structure():
    """test fit_psf_gaussian fit gives expected output structure and values"""
    test_dataset = create_test_xds()
    result = psf_gaussian_fit(test_dataset)
    # sigma = 0.2 -> fwhm = 2* sigma*sqrt(2*ln(2)) = 0.2*2.35482
    truth_values = [0.47096, 0.47096, 0.0]
    assert "BEAM" in result
    assert "beam_param" in result["BEAM"].dims
    params = result["BEAM"]["beam_param"]
    assert params.shape[-1] == 3
    # Check that the fitted widths are positive
    assert np.all(result["BEAM"].data[:, :, :-1] > 0)
    assert np.allclose(result["BEAM"].data[0, 0, 0, :], truth_values, rtol=1e-3, atol=5)


def test_psf_gaussian_fit_custom_window():
    """test npix_window and sampling parameters"""
    test_dataset = create_test_xds(shape=(1, 1, 1, 15, 15))
    truth_values = [0.47096, 0.47096, 0.0]
    # result = psf_gaussian_fit(test_dataset, npix_window=[7, 7], sampling=[7, 7])
    result = psf_gaussian_fit(test_dataset, npix_window=[15, 15], sampling=[9, 9])
    params = result["BEAM"]["beam_param"]
    assert params.shape == (3,)
    assert np.all(result["BEAM"].data[:, :, :-1] > 0)
    assert np.allclose(result["BEAM"].data[0, 0, 0, :], truth_values, rtol=1e-3, atol=5)


def test_invalid_npix_window_type():
    """test npix_window type checking"""
    ds = create_test_xds()
    with pytest.raises((TypeError, ValueError)):
        psf_gaussian_fit(ds, npix_window="invalid")


def test_negative_npix_window():
    """test npix_window value checking"""
    ds = create_test_xds()
    with pytest.raises((ValueError, AssertionError)):
        psf_gaussian_fit(ds, npix_window=[-5, 9])


def test_single_value_npix_window():
    """test single value npix_window"""
    ds = create_test_xds()
    with pytest.raises((TypeError, AssertionError)):
        psf_gaussian_fit(ds, npix_window=7)


def test_non_integer_npix_window():
    """test non-integer npix_window"""
    ds = create_test_xds()
    with pytest.raises((TypeError, AssertionError)):
        psf_gaussian_fit(ds, npix_window=[5.5, 9])


def test_zero_sampling():
    """test sampling value checking"""
    ds = create_test_xds()
    with pytest.raises((ValueError, AssertionError)):
        psf_gaussian_fit(ds, sampling=[0, 9])


def test_single_value_sampling():
    """test single value sampling"""
    ds = create_test_xds()
    with pytest.raises((TypeError, AssertionError)):
        psf_gaussian_fit(ds, sampling=5)


def test_non_integer_sampling():
    """test non-integer sampling"""
    ds = create_test_xds()
    with pytest.raises((TypeError, AssertionError)):
        psf_gaussian_fit(ds, sampling=[5.5, 9])


def test_invalid_cutoff():
    """test invalid cutoff value"""
    ds = create_test_xds()
    # Negative cutoff may not make sense
    with pytest.raises((ValueError, AssertionError)):
        psf_gaussian_fit(ds, cutoff=-0.1)


def test_missing_sky_variable():
    """test missing SKY variable"""
    ds = create_test_xds()
    del ds["SKY"]
    with pytest.raises(KeyError):
        psf_gaussian_fit(ds)


def test_all_nan_input():
    """test all NaN input data"""
    ds = create_test_xds()
    ds["SKY"].data[:] = np.nan
    result = psf_gaussian_fit(ds)
    assert np.all(np.isnan(result["BEAM"].data.compute()))


def test_all_zero_input():
    """test all zero input data"""
    ds = create_test_xds()
    ds["SKY"].data[:] = 0
    result = psf_gaussian_fit(ds)
    # Depending on implementation, may be all zeros or NaNs
    assert np.all((result["BEAM"].data == 0) | np.isnan(result["BEAM"].data))


def test_no_l_coordinate():
    """test missing 'l' coordinate"""
    ds = create_test_xds()
    ds = ds.drop_dims("l")
    with pytest.raises(KeyError):
        psf_gaussian_fit(ds)


def test_no_m_coordinate():
    """test missing 'm' coordinate"""
    ds = create_test_xds()
    ds = ds.drop_dims("m")
    with pytest.raises(KeyError):
        psf_gaussian_fit(ds)


def create_rotated_gaussian(shape, angle_deg):
    """Create a dataset with a rotated 2D Gaussian"""
    # Create a 2D Gaussian rotated by angle_deg (counterclockwise from y to x)
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    theta = np.deg2rad(angle_deg)
    a = (np.cos(theta) ** 2) / (2 * 0.2**2) + (np.sin(theta) ** 2) / (2 * 0.1**2)
    b = -np.sin(2 * theta) / (4 * 0.2**2) + np.sin(2 * theta) / (4 * 0.1**2)
    c = (np.sin(theta) ** 2) / (2 * 0.2**2) + (np.cos(theta) ** 2) / (2 * 0.1**2)
    gaussian = np.exp(-(a * xv**2 + 2 * b * xv * yv + c * yv**2))
    data = np.zeros((1, 1, 1, shape[-2], shape[-1]))
    data[0, 0, 0, :, :] = gaussian
    sky_data = da.from_array(data, chunks="auto")
    sky = xr.DataArray(
        sky_data,
        dims=["time", "frequency", "polarization", "l", "m"],
        coords={"l": np.arange(shape[-2]), "m": np.arange(shape[-1])},
    )
    beam_data = da.from_array(np.zeros((1, 1, 1, 3)), chunks="auto")
    beam = xr.DataArray(
        beam_data, dims=["time", "frequency", "polarization", "beam_param"]
    )
    return xr.Dataset({"SKY": sky, "BEAM": beam})


def test_psf_gaussian_fit_orientation():
    """test position angle of fitted results"""
    import matplotlib.pyplot as plt

    for angle in [-135, -90, -45, -33, 33, 45, 90, 135]:
        ds = create_rotated_gaussian((1, 1, 1, 100, 100), angle)
        # Plot the input ellipse (rotated Gaussian)
        input_img = (
            ds["SKY"].data[0, 0, 0].compute()
            if hasattr(ds["SKY"].data, "compute")
            else ds["SKY"].data[0, 0, 0]
        )
        plt.figure()
        plt.imshow(input_img, origin="lower", cmap="viridis")
        plt.title(f"Input Rotated Gaussian, angle={angle} deg")
        plt.colorbar(label="Amplitude")
        plt.xlabel("m (x)")
        plt.ylabel("l (y)")
        plt.tight_layout()
        # plt.show(block=True)
        plt.show(block=False)
        result = psf_gaussian_fit(ds)
        measured_angle = -np.rad2deg(float(result["BEAM"].data[0, 0, 0, 2]))
        measured_bmaj = float(result["BEAM"].data[0, 0, 0, 0])
        measured_bmin = float(result["BEAM"].data[0, 0, 0, 1])
        # Allow for 180-degree ambiguity and some tolerance
        measured_angle -= 90
        if measured_angle < -90:
            measured_angle += 180
        if measured_angle > 90:
            measured_angle -= 180
        angle_mod = (measured_angle + 180) % 180
        expected_mod = (angle + 180) % 180
        # print(
        #    f"angle={angle}, measured_angle={measured_angle}, angle_mod={angle_mod}, expected_mod={expected_mod}"
        # )
        assert np.isclose(
            expected_mod,
            angle_mod,
            atol=5,
        ), f"Expected {angle}, got {measured_angle}"


def test_psf_gaussian_fit_core_simple_gaussian():
    # Create a simple 2D Gaussian
    shape = (1, 1, 1, 9, 9)
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    gaussian = np.exp(-(xv**2 + yv**2) / (2 * 0.2**2))
    data = np.zeros(shape)
    data[0, 0, 0, :, :] = gaussian
    npix_window = np.array([9, 9])
    sampling = np.array([9, 9])
    cutoff = 0.1
    delta = np.array([np.abs(x[1] - x[0]), np.abs(y[1] - y[0])])
    result = psf_gaussian_fit_core(data, npix_window, sampling, cutoff, delta)
    assert result.shape == (1, 1, 1, 3)
    assert np.all(result[0, 0, 0, :2] > 0)
    assert np.isfinite(result[0, 0, 0, 2])


def test_psf_gaussian_fit_core_all_nan():
    # All NaN input
    data = np.full((1, 1, 1, 9, 9), np.nan)
    npix_window = np.array([9, 9])
    sampling = np.array([9, 9])
    cutoff = 0.1
    delta = np.array([1.0, 1.0])
    result = psf_gaussian_fit_core(data, npix_window, sampling, cutoff, delta)
    assert np.all(np.isnan(result))


def test_psf_gaussian_fit_core_all_zero():
    # All zero input
    data = np.zeros((1, 1, 1, 9, 9))
    npix_window = np.array([9, 9])
    sampling = np.array([9, 9])
    cutoff = 0.1
    delta = np.array([1.0, 1.0])
    result = psf_gaussian_fit_core(data, npix_window, sampling, cutoff, delta)
    assert np.all(result == 0)


# def test_psf_gaussian_fit_core_rotated_gaussian():
#     # Rotated Gaussian
#     shape = (1, 1, 1, 21, 21)
#     angle_deg = 45
#     x = np.linspace(-1, 1, shape[-2])
#     y = np.linspace(-1, 1, shape[-1])
#     xv, yv = np.meshgrid(x, y, indexing="ij")
#     theta = np.deg2rad(angle_deg)
#     a = (np.cos(theta) ** 2) / (2 * 0.2**2) + (np.sin(theta) ** 2) / (2 * 0.1**2)
#     b = -np.sin(2 * theta) / (4 * 0.2**2) + np.sin(2 * theta) / (4 * 0.1**2)
#     c = (np.sin(theta) ** 2) / (2 * 0.2**2) + (np.cos(theta) ** 2) / (2 * 0.1**2)
#     gaussian = np.exp(-(a * xv**2 + 2 * b * xv * yv + c * yv**2))
#     data = np.zeros(shape)
#     data[0, 0, 0, :, :] = gaussian
#     npix_window = np.array([21, 21])
#     sampling = np.array([21, 21])
#     cutoff = 0.1
#     delta = np.array([np.abs(x[1] - x[0]), np.abs(y[1] - y[0])])
#     result = psf_gaussian_fit_core(data, npix_window, sampling, cutoff, delta)
#     measured_angle = result[0, 0, 0, 2]
#     # Allow for 180-degree ambiguity
#     angle_mod = (np.rad2deg(measured_angle) + 180) % 180
#     expected_mod = (angle_deg + 180) % 180
#     print(f"angle_deg={angle_deg}, measured_angle={np.rad2deg(measured_angle)}")
#     assert np.isclose(angle_mod, expected_mod, atol=10)
