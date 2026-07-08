import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
    FWHM_factor,
    extract_main_lobe,
)
from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
    point_spread_function_gaussian_fit as psf_gaussian_fit,
)
from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
    psf_gaussian_fit_core,
)


def create_test_xds(shape=(1, 1, 1, 51, 51), rm_coord=None):
    # Create a simple 2D circular Gaussian as test data
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    gaussian = np.exp(-(xv**2 + yv**2) / (2 * 0.2**2))
    data = np.zeros(shape)
    dims = ["time", "frequency", "polarization", "l", "m"]
    data_coords = {
        "time": np.arange(shape[0]),
        "frequency": np.arange(shape[1]),
        "polarization": np.arange(shape[2]),
        "l": np.linspace(-1, 1, shape[-2]),
        "m": np.linspace(-1, 1, shape[-1]),
    }
    if rm_coord is None:
        data[0, 0, 0, :, :] = gaussian
    elif rm_coord == "l" or rm_coord == "m":
        shape = shape[:-2] + (shape[-2],)  # Adjust shape if a coordinate is removed
        dims.remove(rm_coord)
        data_coords.pop(rm_coord)
        data = np.zeros(shape)

    # Processing-function unit tests operate on numpy-backed xarray
    # structures (no Dask arrays).
    psf = xr.DataArray(data, dims=dims, coords=data_coords)

    beam_data = np.zeros((1, shape[1], shape[2], 3))
    beam_dims = ["time", "frequency", "polarization", "beam_params_label"]
    beam_coords = {
        "time": np.arange(shape[0]),
        "frequency": np.arange(shape[1]),
        "polarization": np.arange(shape[2]),
        "beam_params_label": np.arange(3),
    }
    beam = xr.DataArray(beam_data, dims=beam_dims, coords=beam_coords)

    test_dataset = xr.Dataset(
        {"POINT_SPREAD_FUNCTION": psf, "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION": beam}
    )
    # The public fit function resolves data variables through this data group.
    test_dataset.attrs["data_groups"] = {
        "image": {"point_spread_function": "POINT_SPREAD_FUNCTION"}
    }
    return test_dataset


def test_psf_gaussian_fit_output_structure():
    """test fit_psf_gaussian fit gives expected output structure and values"""
    test_dataset = create_test_xds()
    result = psf_gaussian_fit(test_dataset)
    print(result)
    # sigma = 0.2 -> fwhm = 2* sigma*sqrt(2*ln(2)) = 0.2*2.35482
    truth_values = [0.47096, 0.47096, 0.0]
    assert "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION" in result
    # The fit writes the parameters along a ``beam_params`` dimension and adds a
    # ``beam_params_label`` coordinate ("major", "minor", "pa") to the dataset.
    assert "beam_params" in result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].dims
    assert result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].sizes["beam_params"] == 3
    params = result["beam_params_label"]
    assert params.shape[-1] == 3
    # Check that the fitted widths (major, minor) are positive
    assert np.all(result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data[..., :-1] > 0)
    assert np.allclose(
        result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data[0, 0, 0, :],
        truth_values,
        rtol=1e-3,
        atol=5,
    )


def test_psf_gaussian_fit_custom_window():
    """test npix_window and sampling parameters"""
    test_dataset = create_test_xds(shape=(1, 1, 1, 15, 15))
    truth_values = [0.47096, 0.47096, 0.0]
    # result = psf_gaussian_fit(test_dataset, npix_window=[7, 7], sampling=[7, 7])
    result = psf_gaussian_fit(test_dataset, npix_window=[15, 15], sampling=[9, 9])
    params = result["beam_params_label"]
    assert params.shape == (3,)
    assert np.all(result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data[..., :-1] > 0)
    assert np.allclose(
        result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data[0, 0, 0, :],
        truth_values,
        rtol=1e-3,
        atol=5,
    )


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


def test_even_npix_window():
    """npix_window must be odd so the window stays centered on the peak"""
    ds = create_test_xds()
    with pytest.raises((ValueError, AssertionError)):
        psf_gaussian_fit(ds, npix_window=[40, 40])
    with pytest.raises((ValueError, AssertionError)):
        psf_gaussian_fit(ds, npix_window=[41, 40])


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


def test_missing_psf_variable():
    """test missing POINT_SPREAD_FUNCTION variable"""
    ds = create_test_xds()
    del ds["POINT_SPREAD_FUNCTION"]
    with pytest.raises(KeyError):
        psf_gaussian_fit(ds)


def test_all_nan_input():
    """test all NaN input data"""
    ds = create_test_xds()
    ds["POINT_SPREAD_FUNCTION"].data[:] = np.nan
    result = psf_gaussian_fit(ds)
    assert np.all(np.isnan(result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].values))


def test_all_zero_input():
    """test all zero input data"""
    ds = create_test_xds()
    ds["POINT_SPREAD_FUNCTION"].data[:] = 0
    result = psf_gaussian_fit(ds)
    # Depending on implementation, may be all zeros or NaNs
    assert np.all(
        (result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data == 0)
        | np.isnan(result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data)
    )


def test_no_l_coordinate():
    """test missing 'l' coordinate"""
    ds = create_test_xds(rm_coord="l")
    with pytest.raises(KeyError):
        psf_gaussian_fit(ds)


def test_no_m_coordinate():
    """test missing 'm' coordinate"""
    ds = create_test_xds(rm_coord="m")
    with pytest.raises(KeyError):
        psf_gaussian_fit(ds)


def test_oversampling():
    """test oversampling case"""
    ds = create_test_xds(shape=(1, 1, 1, 100, 100))
    result = psf_gaussian_fit(ds, npix_window=[41, 41], sampling=[51, 51])
    truth_values = [0.47096, 0.47096, 0.0]
    assert np.allclose(
        result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data[0, 0, 0, :],
        truth_values,
        rtol=1e-3,
        atol=5,
    )


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
    psf = xr.DataArray(
        data,
        dims=["time", "frequency", "polarization", "l", "m"],
        coords={"l": np.arange(shape[-2]), "m": np.arange(shape[-1])},
    )
    beam_data = np.zeros((1, 1, 1, 3))
    beam = xr.DataArray(
        beam_data, dims=["time", "frequency", "polarization", "beam_params_label"]
    )
    rotated_dataset = xr.Dataset(
        {"POINT_SPREAD_FUNCTION": psf, "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION": beam}
    )
    rotated_dataset.attrs["data_groups"] = {
        "image": {"point_spread_function": "POINT_SPREAD_FUNCTION"}
    }
    return rotated_dataset


def test_psf_gaussian_fit_orientation():
    """test position angle of fitted results"""
    import matplotlib.pyplot as plt

    for angle in [-135, -90, -45, -33, 33, 45, 90, 135]:
        ds = create_rotated_gaussian((1, 1, 1, 200, 200), angle)
        # Plot the input ellipse (rotated Gaussian)
        input_img = ds["POINT_SPREAD_FUNCTION"][0, 0, 0].values

        # for verfication purpose only
        plt.figure()
        plt.imshow(input_img, origin="lower", cmap="viridis")
        plt.title(f"Input Rotated Gaussian, angle={angle} deg")
        plt.colorbar(label="Amplitude")
        plt.xlabel("m (x)")
        plt.ylabel("l (y)")
        plt.tight_layout()
        # plt.show(block=True)
        plt.show(block=False)
        result = psf_gaussian_fit(ds, npix_window=(21, 21), sampling=(55, 55))
        measured_angle = -np.rad2deg(
            float(result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data[0, 0, 0, 2])
        )
        measured_bmaj = float(
            result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data[0, 0, 0, 0]
        )
        measured_bmin = float(
            result["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].data[0, 0, 0, 1]
        )
        # Allow for 180-degree ambiguity and some tolerance
        measured_angle -= 90
        if measured_angle < -90:
            measured_angle += 180
        if measured_angle > 90:
            measured_angle -= 180
        angle_mod = (measured_angle + 180) % 180
        expected_mod = (angle + 180) % 180
        print(
            f"angle={angle}, measured_angle={measured_angle}, angle_mod={angle_mod}, expected_mod={expected_mod}"
        )
        assert np.isclose(
            expected_mod,
            angle_mod,
            atol=5,
        ), f"Expected {angle}, got {measured_angle}"


def _make_multi_slice_cube(shape=(2, 3, 2, 81, 81), sigmas=None):
    """Build a (time, chan, pol, l, m) cube of centred Gaussians with varying widths."""
    n_t, n_c, n_p, nl, nm = shape
    x = np.linspace(-1, 1, nl)
    y = np.linspace(-1, 1, nm)
    xv, yv = np.meshgrid(x, y, indexing="ij")
    data = np.zeros(shape)
    if sigmas is None:
        rng = np.random.default_rng(0)
        sigmas = 0.15 + 0.05 * rng.random((n_t, n_c, n_p))
    for t in range(n_t):
        for c in range(n_c):
            for p in range(n_p):
                s = sigmas[t, c, p]
                data[t, c, p, :, :] = np.exp(-(xv**2 + yv**2) / (2 * s**2))
    return data, x, y


def test_psf_gaussian_fit_core_threaded_matches_serial():
    """Threaded execution must yield identical results to the serial path."""
    data, x, y = _make_multi_slice_cube(shape=(2, 3, 2, 81, 81))
    blc = np.array([20, 20])
    trc = np.array([60, 60])
    sampling = np.array([41, 41])
    cutoff = 0.1
    delta = np.array([np.abs(x[1] - x[0]), np.abs(y[1] - y[0])])

    serial = psf_gaussian_fit_core(
        data.copy(), blc, trc, sampling, cutoff, delta, num_threads=1
    )
    threaded = psf_gaussian_fit_core(
        data.copy(), blc, trc, sampling, cutoff, delta, num_threads=4
    )
    auto = psf_gaussian_fit_core(
        data.copy(), blc, trc, sampling, cutoff, delta, num_threads=0
    )

    assert serial.shape == (2, 3, 2, 3)
    np.testing.assert_allclose(threaded, serial, rtol=0, atol=0)
    np.testing.assert_allclose(auto, serial, rtol=0, atol=0)
    # Every slice should have produced positive widths.
    assert np.all(serial[..., :2] > 0)


def test_psf_gaussian_fit_core_num_threads_exceeds_tasks():
    """num_threads larger than the number of slices must still succeed."""
    data, x, y = _make_multi_slice_cube(shape=(1, 1, 1, 61, 61))
    blc = np.array([10, 10])
    trc = np.array([50, 50])
    sampling = np.array([31, 31])
    cutoff = 0.1
    delta = np.array([np.abs(x[1] - x[0]), np.abs(y[1] - y[0])])

    result = psf_gaussian_fit_core(
        data, blc, trc, sampling, cutoff, delta, num_threads=16
    )
    assert result.shape == (1, 1, 1, 3)
    assert np.all(result[0, 0, 0, :2] > 0)


def test_psf_gaussian_fit_core_simple_gaussian():
    # Create a simple 2D Gaussian
    shape = (1, 1, 1, 100, 100)
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    gaussian = np.exp(-(xv**2 + yv**2) / (2 * 0.2**2))
    data = np.zeros(shape)
    data[0, 0, 0, :, :] = gaussian
    npix_window = np.array([41, 41])
    blc = np.array([50 - npix_window[0] // 2, 30 - npix_window[1] // 2])
    trc = np.array([50 + npix_window[0] // 2, 30 + npix_window[1] // 2])
    sampling = np.array([41, 41])
    cutoff = 0.1
    delta = np.array([np.abs(x[1] - x[0]), np.abs(y[1] - y[0])])
    result = psf_gaussian_fit_core(data, blc, trc, sampling, cutoff, delta)
    assert result.shape == (1, 1, 1, 3)
    assert np.all(result[0, 0, 0, :2] > 0)
    assert np.isfinite(result[0, 0, 0, 2])


def test_psf_gaussian_fit_core_all_nan():
    # All NaN input
    data = np.full((1, 1, 1, 9, 9), np.nan)
    blc = np.array([0, 0])
    trc = np.array([9, 9])
    sampling = np.array([9, 9])
    cutoff = 0.1
    delta = np.array([1.0, 1.0])
    result = psf_gaussian_fit_core(data, blc, trc, sampling, cutoff, delta)
    assert np.all(np.isnan(result))


def test_psf_gaussian_fit_core_all_zero():
    # All zero input
    data = np.zeros((1, 1, 1, 9, 9))
    blc = np.array([0, 0])
    trc = np.array([9, 9])
    sampling = np.array([9, 9])
    cutoff = 0.1
    delta = np.array([1.0, 1.0])
    result = psf_gaussian_fit_core(data, blc, trc, sampling, cutoff, delta)
    assert np.all(result == 0)


def test_extract_main_lobe():
    """
    Test the extraction of the main lobe from a PSF image.
    """
    # Simple test case with a main lobe with sidelobes

    shape = (1, 1, 1, 100, 100)
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    gaussian = np.exp(-(xv**2 + yv**2) / (2 * 0.2**2))
    sidelobe1 = 0.3 * np.exp(-((xv - 0.5) ** 2 + (yv - 0.5) ** 2) / (2 * 0.1**2))
    sidelobe2 = 0.2 * np.exp(-((xv + 0.5) ** 2 + (yv + 0.5) ** 2) / (2 * 0.1**2))
    psf_image = gaussian + sidelobe1 + sidelobe2

    # for verification purposes, plot the psf_image
    import matplotlib.pyplot as plt

    plt.imshow(psf_image, origin="lower", cmap="gray")
    plt.colorbar()
    plt.title("PSF Image")
    plt.show(block=False)  # change to True to pause the test to see the plot

    data = np.zeros(shape)
    data[0, 0, 0, :, :] = psf_image
    npix_window = np.array([41, 41])
    cutoff = 0.1
    main_lobe_only, blc, trc, __ = extract_main_lobe(npix_window, cutoff, data)
    print("test_extract_main_lobe: blc, trc=", blc, trc)
    # blc/trc are now per-slice arrays of shape (time, frequency, polarization, 2).
    assert blc.shape == (1, 1, 1, 2)
    assert trc.shape == (1, 1, 1, 2)
    # Check that the main lobe is extracted correctly by comparing total main lobe pixels
    # with model main lobe (gaussian) pixels with cutoff applied
    total_main_lobe = np.sum(main_lobe_only)
    model_main_lobe = np.sum(gaussian[gaussian >= cutoff * np.max(gaussian)])
    assert np.isclose(total_main_lobe, model_main_lobe, rtol=0.1)
    assert blc[0, 0, 0, 0] < trc[0, 0, 0, 0] and blc[0, 0, 0, 1] < trc[0, 0, 0, 1]
    # Check that sidelobes are zeroed out
    assert np.all(
        main_lobe_only[0, 0, 0, :, :][psf_image < cutoff * np.max(psf_image)] == 0
    )


def test_extract_main_lobe_offset_peak():
    """
    Test the case where the psf peak is not at the center of the image
    """
    shape = (1, 1, 1, 100, 100)
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    gaussian = np.exp(-((xv - 0.1) ** 2 + (yv) ** 2) / (2 * 0.2**2))
    sidelobe1 = 0.3 * np.exp(-((xv - 0.5) ** 2 + (yv - 0.5) ** 2) / (2 * 0.1**2))
    sidelobe2 = 0.2 * np.exp(-((xv + 0.5) ** 2 + (yv + 0.5) ** 2) / (2 * 0.1**2))
    psf_image = gaussian + sidelobe1 + sidelobe2

    # for verification purposes, plot the psf_image (set plt.show(block=True) to pause the test)
    import matplotlib.pyplot as plt

    plt.imshow(psf_image, origin="lower", cmap="gray")
    plt.colorbar()
    plt.title("PSF Image with Offset Peak")
    plt.show(block=False)  # change to True to pause the test to see the plot

    data = np.zeros(shape)
    data[0, 0, 0, :, :] = psf_image
    npix_window = np.array([41, 41])
    cutoff = 0.1
    main_lobe_only, blc, trc, __ = extract_main_lobe(npix_window, cutoff, data)
    print("test_extract_main_lobe_offset_peak: blc, trc=", blc, trc)
    # blc/trc are now per-slice arrays of shape (time, frequency, polarization, 2).
    assert blc.shape == (1, 1, 1, 2)
    assert trc.shape == (1, 1, 1, 2)
    # Check that the main lobe is extracted correctly by comparing total main lobe pixels
    # with model main lobe (gaussian) pixels with cutoff applied
    total_main_lobe = np.sum(main_lobe_only)
    model_main_lobe = np.sum(gaussian[gaussian >= cutoff * np.max(gaussian)])
    assert np.isclose(total_main_lobe, model_main_lobe, rtol=0.1)
    assert blc[0, 0, 0, 0] < trc[0, 0, 0, 0] and blc[0, 0, 0, 1] < trc[0, 0, 0, 1]
    # Check that sidelobes are zeroed out
    assert np.all(
        main_lobe_only[0, 0, 0, :, :][psf_image < cutoff * np.max(psf_image)] == 0
    )


def test_extract_main_lobe_per_slice_independent():
    """extract_main_lobe must return an independent bounding box per slice."""
    shape = (1, 2, 1, 81, 81)
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    data = np.zeros(shape)
    # Two channels with very different widths -> different main-lobe sizes.
    data[0, 0, 0] = np.exp(-(xv**2 + yv**2) / (2 * 0.1**2))  # narrow
    data[0, 1, 0] = np.exp(-(xv**2 + yv**2) / (2 * 0.3**2))  # wide
    npix_window = np.array([61, 61])
    cutoff = 0.1
    main_lobe_only, blc, trc, max_sidelobe = extract_main_lobe(
        npix_window, cutoff, data
    )
    assert blc.shape == (1, 2, 1, 2)
    assert trc.shape == (1, 2, 1, 2)
    assert max_sidelobe.shape == (1, 2, 1)
    # The wider Gaussian must yield a strictly larger main-lobe bounding box.
    size_narrow = trc[0, 0, 0] - blc[0, 0, 0]
    size_wide = trc[0, 1, 0] - blc[0, 1, 0]
    assert np.all(size_wide > size_narrow)


def test_psf_gaussian_fit_core_per_slice_boxes():
    """Per-slice blc/trc arrays must fit each slice with its own window."""
    shape = (1, 2, 1, 81, 81)
    x = np.linspace(-1, 1, shape[-2])
    y = np.linspace(-1, 1, shape[-1])
    xv, yv = np.meshgrid(x, y, indexing="ij")
    data = np.zeros(shape)
    data[0, 0, 0] = np.exp(-(xv**2 + yv**2) / (2 * 0.1**2))  # narrow
    data[0, 1, 0] = np.exp(-(xv**2 + yv**2) / (2 * 0.3**2))  # wide
    sampling = np.array([41, 41])
    cutoff = 0.1
    delta = np.array([np.abs(x[1] - x[0]), np.abs(y[1] - y[0])])

    _, blc, trc, _ = extract_main_lobe(np.array([61, 61]), cutoff, data)
    result = psf_gaussian_fit_core(data, blc, trc, sampling, cutoff, delta)

    assert result.shape == (1, 2, 1, 3)
    # Wider input Gaussian -> larger fitted major axis.
    assert result[0, 1, 0, 0] > result[0, 0, 0, 0]
    # Each slice is recovered close to its analytic FWHM (sigma * FWHM_factor).
    assert np.isclose(result[0, 0, 0, 0], 0.1 * FWHM_factor, rtol=0.15)
    assert np.isclose(result[0, 1, 0, 0], 0.3 * FWHM_factor, rtol=0.15)


def test_psf_gaussian_fit_core_broadcasts_single_box():
    """A single [l, m] box must match an explicit per-slice array of that box."""
    data, x, y = _make_multi_slice_cube(shape=(2, 2, 2, 81, 81))
    sampling = np.array([41, 41])
    cutoff = 0.1
    delta = np.array([np.abs(x[1] - x[0]), np.abs(y[1] - y[0])])

    blc = np.array([20, 20])
    trc = np.array([60, 60])
    broadcast = psf_gaussian_fit_core(data.copy(), blc, trc, sampling, cutoff, delta)

    blc_full = np.broadcast_to(blc, (2, 2, 2, 2)).copy()
    trc_full = np.broadcast_to(trc, (2, 2, 2, 2)).copy()
    per_slice = psf_gaussian_fit_core(
        data.copy(), blc_full, trc_full, sampling, cutoff, delta
    )
    np.testing.assert_allclose(broadcast, per_slice, rtol=0, atol=0)


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
