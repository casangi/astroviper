import numpy as np
from numba import jit, objmode
import numba
import xarray as xr
import scipy.optimize as optimize
from scipy.interpolate import interpn
from scipy.ndimage import label

FWHM_factor = 2.355  # approx 2*sqrt(2*ln(2))
beam_initial_guess = [2.5, 2.5, 0.0]  # initial guess for [bmaj, bmin, pa]
expand_pixel = 5  # expand the fitting window by this many pixels on each side


def _get_main_lobe_bounding_box(masked_psf, max_coords):
    """
    Given a psf image of main lobe in full 5D with zero values outside the lobe, find the bounding box
    based on slice of a 2D image at max_coords.

    Args:
        masked_psf (np.ndarray): A 2D binary mask where the main lobe is True.
        max_coords (tuple): Coordinates of the peak intensity within the main lobe.
    Returns:
        tuple: Bounding box coordinates (blc, trc) as ((x_min, y_min), (x_max, y_max))
    """
    # make slice of 2d image using the first 3 indices (time, freq, pol) of max_coords
    psf_2d = masked_psf[max_coords[0], max_coords[1], max_coords[2], :, :]
    valid_pixels = psf_2d != 0

    valid_indices = np.where(valid_pixels)

    if len(valid_indices[0]) == 0:
        return None, None  # No valid pixels found

    x_min, x_max = valid_indices[0].min(), valid_indices[0].max()
    y_min, y_max = valid_indices[1].min(), valid_indices[1].max()

    # Ensure the bounding box is square
    if x_min > y_min:
        x_min = y_min
    if x_max < y_max:
        x_max = y_max

    return np.array([x_min, y_min]), np.array([x_max, y_max])


def extract_main_lobe(npix_window, threshold, psf_image):
    """
    Extracts the main lobe from a PSF image, within the window defined by npix_window
    to handle large images efficiently.

    Args:
        npix_window (tuple): The size of the window in pixels for searching features.
        threshold (float): A threshold in fraction of peak value for determining the main lobe.
        psf_image (np.ndarray): The input PSF image with 5 dimensions (time, frequency, polarization, x, y).

    Returns:
        np.ndarray: A new array containing only the main lobe, with other regions zeroed out.
        blc (np.ndarray): Bottom-left corner of the bounding box of the main lobe.
        trc (np.ndarray): Top-right corner of the bounding box of the main lobe.
    """
    # Find the peak intensity of the image
    peak_intensity = np.max(psf_image)
    if peak_intensity == 0:
        return (
            np.zeros_like(psf_image),
            np.array([0, 0]),
            np.array([psf_image.shape[3] - 1, psf_image.shape[4] - 1]),
            0.0,  # max_sidelobe is 0.0 if peak_intensity is 0
        )
    # find peak location in the psf_image
    itm, ifrq, ipol, peak_y, peak_x = np.unravel_index(
        np.argmax(psf_image), psf_image.shape
    )
    print("peak_x, peak_y=", peak_x, peak_y)

    # set window outside of psf image to 0
    windowed_psf = np.zeros_like(psf_image)
    windowed_psf[
        :,
        :,
        :,
        max(0, peak_x - npix_window[0] // 2) : min(
            psf_image.shape[3], peak_x + npix_window[0] // 2
        ),
        max(0, peak_y - npix_window[1] // 2) : min(
            psf_image.shape[4], peak_y + npix_window[1] // 2
        ),
    ] = psf_image[
        :,
        :,
        :,
        max(0, peak_x - npix_window[0] // 2) : min(
            psf_image.shape[3], peak_x + npix_window[0] // 2
        ),
        max(0, peak_y - npix_window[1] // 2) : min(
            psf_image.shape[4], peak_y + npix_window[1] // 2
        ),
    ]

    # print("windowed_psf.shape=", windowed_psf.shape)
    # create sub
    # Determine a threshold based on the peak intensity
    abs_threshold = peak_intensity * threshold

    # Create a binary mask where pixels above the threshold are True
    binary_mask = windowed_psf > abs_threshold
    # print("binary_mask=", binary_mask)
    # Use SciPy's `label` to find connected components in the binary mask
    # This is efficient for large images and helps find distinct lobes
    labels, num_features = label(binary_mask)

    # Find the label corresponding to the main lobe
    # We assume the main lobe is the region containing the global maximum
    max_coords = np.unravel_index(np.argmax(windowed_psf), windowed_psf.shape)
    main_lobe_label = labels[max_coords]

    # Create a new image containing only the main lobe
    main_lobe_only = np.where(labels == main_lobe_label, windowed_psf, 0)
    # return also max sidelobe level?, applying main lobe mask on the original psf_image
    # masked_main = np.where(labels != main_lobe_label, psf_image, 0)
    # max_side_lobe = np.max(masked_main)
    max_sidelobe = np.max(psf_image * (labels != main_lobe_label))
    print("maximum sidelobe level: ", max_sidelobe)
    blc, trc = _get_main_lobe_bounding_box(main_lobe_only, max_coords)
    # print("extract_main_lobe: blc, trc=", blc, trc)
    return main_lobe_only, blc, trc, max_sidelobe


def psf_gaussian_fit(
    xds: xr.Dataset,
    dv: str = "POINT_SPREAD_FUNCTION",
    npix_window: tuple = (41, 41),
    sampling: tuple = (55, 55),
    cutoff: float = 0.35,
):
    """
    fit 2D gaussian to psf

    Parameters
    ----------
    xds : xarray.Dataset
        The input data cube.
    dv : str
        The data variable to fit. Default is 'POINT_SPREAD_FUNCTION'.
    npix_window : tuple
        The size of the fitting window in pixels.
    sampling : tuple
        The sampling of the fitting grid in pixels.
    cutoff : float
        The cutoff value for the fitting.

    Returns
    -------
    xds : xarray.Dataset
        The image with the fitted parameters added.
        The l and m coordinates of the input data are assumed to be in radians.
        The units of beam size (major and minor) and position angle are in radians.

    Notes:
    -----
    - Returns NaN values for beam parameters if the fitting fails
    - L-BFGS-B optimization method is used with bounds on parameters
    """

    if not isinstance(npix_window, (list, tuple, np.ndarray)):
        raise TypeError("npix_window must be a list, tuple, or numpy array")
    if npix_window[0] <= 0 or npix_window[1] <= 0:
        raise ValueError("npix_window must be positive")
    if type(npix_window[0]) is not int or type(npix_window[1]) is not int:
        raise TypeError("npix_window must be integers")
    if not isinstance(sampling, (list, tuple, np.ndarray)):
        raise TypeError("sampling must be a list, tuple, or numpy array")
    if sampling[0] <= 0 or sampling[1] <= 0:
        raise ValueError("sampling must be positive")
    if type(sampling[0]) is not int or type(sampling[1]) is not int:
        raise TypeError("sampling must be integers")
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")
    if dv not in xds:
        raise KeyError(f"{dv} not found in the dataset")
    if "l" not in xds[dv].coords:
        raise KeyError("'l' coordinate not found in xds")
    if "m" not in xds[dv].coords:
        raise KeyError("'m' coordinate not found in xds")

    _xds = xds.copy(deep=True)

    sampling = np.array(sampling)
    npix_window = np.array(npix_window)

    # pixel increments (radians)
    delta = np.array([_xds[dv].l[1] - _xds[dv].l[0], _xds[dv].m[1] - _xds[dv].m[0]])
    # To make fitting result expressed in arcsecond uncomment the below
    #    * 3600
    #    * 180
    #    / np.pi

    # px, py, amp, psf2d = locate_peak_psf(_xds[dv].data.compute())
    # print(" locate_peak_psf: px, py, amp=", px, py, amp)
    # if amp < 1e-7:
    #    raise ValueError("PSF peak amplitude is zero")

    # npoints, blc, trc, x, y, sigma = find_n_points(
    #    npix_window, cutoff, px, py, psf2d, delta
    # )
    # print(" after find_n_points blc, trc=", blc, trc)
    main_lobe_im, blc, trc, __ = extract_main_lobe(
        npix_window, cutoff, _xds[dv].data.compute()
    )
    blc = blc - expand_pixel
    trc = trc + expand_pixel
    # print(" blc, trc after expanding=", blc, trc)
    if blc[0] < 0:
        blc[0] = 0
    if blc[1] < 0:
        blc[1] = 0
    if trc[0] >= main_lobe_im.shape[3]:
        trc[0] = main_lobe_im.shape[3] - 1
    if trc[1] >= main_lobe_im.shape[4]:
        trc[1] = main_lobe_im.shape[4] - 1

    ellipse_params = psf_gaussian_fit_core(
        _xds[dv].data.compute(), blc, trc, sampling, cutoff, delta
    )

    # Uncomment line below to change beam_param units to arcsec and deg
    # psf_gaussian_fit_core returns bmaj and bmin in  and pa in deg.
    # Converting to radian for storing to the xradio image
    # ellipse_params[..., :2] = np.deg2rad(ellipse_params[..., :2] / 3600.0)
    # ellipse_params[..., 2] = np.deg2rad(ellipse_params[..., 2])

    import dask.array as da

    _xds["BEAM_FIT_PARAMS"].data = da.from_array(ellipse_params, chunks="auto")
    # assume l, m in radians
    if "unit" not in _xds["POINT_SPREAD_FUNCTION"].l.attrs:
        _xds["BEAM_FIT_PARAMS"].beam_params_label.attrs["unit"] = "rad"
    return _xds


def beam_chi2(params, psf, sampling):
    """
    Chi-squared function for beam fitting.
    """
    psf_ravel = np.ravel(psf)
    psf_mask = np.invert(np.isnan(psf_ravel))
    psf_ravel = psf_ravel[psf_mask]

    width_x, width_y, rotation = params

    ### comment out assuming everthing done in radians
    # rotation is assumed to be in degrees
    # rotation = 90 - rotation
    # rotation = np.deg2rad(rotation)
    # for debugging
    # print("rotation original (rad) =", rotation, " (deg) =", np.rad2deg(rotation))
    # print("rotation after mod (rad) =", rotation, " (deg) =", np.rad2deg(rotation))
    x_size = sampling[0] * 2 + 1
    y_size = sampling[1] * 2 + 1

    x = np.repeat(np.arange(x_size), y_size).reshape(x_size, y_size)
    y = np.repeat(np.arange(y_size), x_size).reshape(x_size, y_size).T

    x = x - sampling[0]
    y = y - sampling[1]
    xp = x * np.cos(rotation) - y * np.sin(rotation)
    yp = x * np.sin(rotation) + y * np.cos(rotation)

    gaussian = 1.0 * np.exp(-(((xp) / width_x) ** 2 + ((yp) / width_y) ** 2) / 2.0)

    gaussian_ravel = np.ravel(gaussian)
    gaussian_ravel = gaussian_ravel[psf_mask]

    chi2 = np.sum((gaussian_ravel - psf_ravel) ** 2)
    return chi2


@jit(nopython=True, cache=True, nogil=True)
def psf_gaussian_fit_core(image_to_fit, blc, trc, sampling, cutoff, delta):
    """
    core function to fit gaussian to psf
    Parameters
    ----------
    image_to_fit : np.ndarray
        The input data cube.
    blc : np.ndarray
        The bottom left corner of the fitting window.
    trc : np.ndarray
        The top right corner of the fitting window.
    #npix_window : np.ndarray
    #    The size of the fitting window in pixels.
    sampling : np.ndarray
        The sampling of the fitting grid in pixels.
    cutoff : float
        The cutoff value for the fitting.
    delta : np.ndarray
        The pixel size in radians.
    """
    ellipse_params = np.zeros(image_to_fit.shape[0:3] + (3,), dtype=numba.double)
    if np.all(np.isnan(image_to_fit)):
        return ellipse_params + np.nan
    elif np.all(image_to_fit == 0):
        return ellipse_params
    xmin = blc[0]
    ymin = blc[1]
    xmax = trc[0] + 1
    ymax = trc[1] + 1
    npix_window = np.array([xmax - xmin, ymax - ymin])
    image_to_fit = image_to_fit[
        #    :, :, :, start_window[0] : end_window[0], start_window[1] : end_window[1]
        :,
        :,
        :,
        xmin:xmax,
        ymin:ymax,
    ]

    d0 = np.arange(0, npix_window[0]) * np.abs(delta[0])
    d1 = np.arange(0, npix_window[1]) * np.abs(delta[1])
    interp_d0 = np.linspace(0, npix_window[0] - 1, sampling[0]) * np.abs(delta[0])
    interp_d1 = np.linspace(0, npix_window[1] - 1, sampling[1]) * np.abs(delta[1])

    d0_shape = interp_d0.shape[0]
    d1_shape = interp_d1.shape[0]

    xp = np.repeat(interp_d0, d1_shape).reshape(d0_shape, d1_shape)
    yp = np.repeat(interp_d1, d0_shape).reshape(d1_shape, d0_shape).T

    points = np.vstack((np.ravel(xp), np.ravel(yp))).T

    for time in range(image_to_fit.shape[0]):
        for chan in range(image_to_fit.shape[1]):
            for pol in range(image_to_fit.shape[2]):

                with objmode(res_x="f8[:]"):  # return type annotation

                    interp_image_to_fit = np.reshape(
                        interpn(
                            (d0, d1),
                            image_to_fit[time, chan, pol, :, :],
                            points,
                            method="splinef2d",
                        ),
                        [sampling[1], sampling[0]],
                    ).T
                    interp_image_to_fit[interp_image_to_fit < cutoff] = np.nan

                    p0 = beam_initial_guess
                    bound = [(None, None), (None, None), (-np.pi / 2, np.pi / 2)]
                    res = optimize.minimize(
                        beam_chi2,
                        p0,
                        args=(interp_image_to_fit, sampling // 2),
                        bounds=bound,
                    )
                    if not res.success:
                        # could retry with lowerling cutoff as done in CASA but
                        # since the cutoff is used also outside the loop
                        # implementing retry requires some refactoring...
                        res_x = np.array([np.nan, np.nan, np.nan])
                    else:
                        res_x = res.x
                phi = res_x[2]
                # phi = res_x[2] - np.pi / 2
                # if phi < -np.pi / 2:
                #    phi += np.pi
                #    phi = (np.pi / 2 - res_x[2]) % np.pi

                # phi = np.rad2deg(res_x[2])
                # phi = res_x[2] - 90.0
                # if phi < -90.0:
                #    phi += 180.0

                if np.argmax(res_x[0:2]) == 1:
                    phi = -(np.pi / 2 - phi)
                ellipse_params[time, chan, pol, 0] = np.max(
                    np.abs(res_x[0:2])
                    # ) * np.abs(delta[0] * 2.355 / sampling[0] / npix_window[0])
                ) * np.abs(delta[0] * FWHM_factor / (sampling[0] / npix_window[0]))
                # ) * np.abs(delta[0] * 2.355)
                ellipse_params[time, chan, pol, 1] = np.min(
                    np.abs(res_x[0:2])
                    # ) * np.abs(delta[1] * 2.355 / sampling[1] / npix_window[1])
                ) * np.abs(delta[1] * FWHM_factor / (sampling[1] / npix_window[1]))
                # ) * np.abs(delta[1] * 2.355)
                # ellipse_params[time, chan, pol, 2] = -phi
                if phi < 0:
                    phi = (phi + np.pi) % np.pi
                ellipse_params[time, chan, pol, 2] = phi
    return ellipse_params
