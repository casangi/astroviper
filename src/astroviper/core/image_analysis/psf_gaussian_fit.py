import numpy as np
from numba import jit, objmode
import numba
import xarray
import scipy.optimize as optimize
from scipy.interpolate import interpn


def psf_gaussian_fit(
    xds: xarray.Dataset,
    dv: str = "SKY",
    npix_window: tuple = [9, 9],
    sampling: tuple = [9, 9],
    cutoff: float = 0.35,
):
    """
    fit 2D gaussian to psf

    Parameters
    ----------
    xds : xarray.Dataset
        The input data cube.
    dv : str
        The data variable to fit. Default is 'SKY'.
    npix_window : list
        The size of the fitting window in pixels.
    sampling : list
        The sampling of the fitting grid in pixels.
    cutoff : float
        The cutoff value for the fitting.

    Returns
    -------
    xds : xarray.Dataset
        The image with the fitted parameters added.
        The unit of beam size (major and minor) will be the same unit as
        that of the input image's (l,m) coordintes, which assumed to be
        radian. The position angle is given in degrees.
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
    if "l" not in xds.dims:
        raise KeyError("'l' coordinate not found in xds")
    if "m" not in xds.dims:
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

    ellipse_params = psf_gaussian_fit_core(
        _xds[dv].data.compute(), npix_window, sampling, cutoff, delta
    )

    # Uncomment line below to change beam_param units to arcsec and deg
    # psf_gaussian_fit_core returns bmaj and bmin in  and pa in deg.
    # Converting to radian for storing to the xradio image
    # ellipse_params[..., :2] = np.deg2rad(ellipse_params[..., :2] / 3600.0)
    # ellipse_params[..., 2] = np.deg2rad(ellipse_params[..., 2])

    import dask.array as da

    _xds["BEAM"].data = da.from_array(ellipse_params, chunks="auto")
    # assume l, m in radians
    if not "unit" in _xds["SKY"].l.attrs:
        _xds["BEAM"].beam_param.attrs["unit"] = "rad"
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
def psf_gaussian_fit_core(image_to_fit, npix_window, sampling, cutoff, delta):
    """
    core function to fit gaussian to psf
    Parameters
    ----------
    image_to_fit : np.ndarray
        The input data cube.
    npix_window : np.ndarray
        The size of the fitting window in pixels.
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
    image_size = np.array(image_to_fit.shape[3:5])
    image_center = image_size // 2
    start_window = image_center - npix_window // 2
    end_window = image_center + npix_window // 2 + 1
    image_to_fit = image_to_fit[
        :, :, :, start_window[0] : end_window[0], start_window[1] : end_window[1]
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

                    p0 = [2.5, 2.5, 0]
                    bound = [(None, None), (None, None), (-np.pi / 4, np.pi / 4)]
                    res = optimize.minimize(
                        beam_chi2,
                        p0,
                        args=(interp_image_to_fit, sampling // 2),
                        bounds=bound,
                    )
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
                ) * np.abs(delta[0] * 2.355)
                ellipse_params[time, chan, pol, 1] = np.min(
                    np.abs(res_x[0:2])
                    # ) * np.abs(delta[1] * 2.355 / sampling[1] / npix_window[1])
                ) * np.abs(delta[1] * 2.355)
                # ellipse_params[time, chan, pol, 2] = -phi
                ellipse_params[time, chan, pol, 2] = phi
    return ellipse_params
