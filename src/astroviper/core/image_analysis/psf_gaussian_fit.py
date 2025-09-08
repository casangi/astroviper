import numpy as np
from numba import jit, objmode
import numba
import scipy.optimize as optimize
from scipy.interpolate import interpn


def psf_gaussian_fit(
    xds,
    dv="SKY",
    npix_window=[9, 9],
    sampling=[9, 9],
    cutoff=0.35,
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
        The sampling of the fitting grid.
    cutoff : float
        The cutoff value for the fitting.

    Returns
    -------
    xds : xarray.Dataset
        The image with the fitted parameters added.
        The unit of beam size (major and minor) will be the same unit as
        that of the input image. If there is no unit attribute for the
        input image, the unit will be radian. The position angle is given
        in degrees.
    """

    _xds = xds.copy(deep=True)

    sampling = np.array(sampling)
    npix_window = np.array(npix_window)

    in_delta = np.array([_xds[dv].l[1] - _xds[dv].l[0], _xds[dv].m[1] - _xds[dv].m[0]])
    # pixel increments in arcsecond
    delta = (
        np.array([_xds[dv].l[1] - _xds[dv].l[0], _xds[dv].m[1] - _xds[dv].m[0]])
        * 3600
        * 180
        / np.pi
    )

    print(f"in_delta: {in_delta}")
    print(f"delta: {delta}")
    ellipse_params = psf_gaussian_fit_core(
        _xds[dv].data.compute(), npix_window, sampling, cutoff, delta
    )
    # psf_gaussian_fit_core returns bmaj and bmin in arcsecond and pa in deg.
    # Converting to radian for storing to the xradio image
    print(ellipse_params.shape)
    ellipse_params[..., :2] = np.deg2rad(ellipse_params[..., :2] / 3600.0)
    ellipse_params[..., 2] = np.deg2rad(ellipse_params[..., 2])

    _xds["BEAM"].data = ellipse_params
    if "unit" in _xds["SKY"].l.attrs:
        _xds["BEAM"].beam_param.attrs["unit"] = _xds["SKY"].l.attrs["unit"]
    else:
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
    rotation = 90 - rotation
    rotation = np.deg2rad(rotation)

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
    """
    ellipse_params = np.zeros(image_to_fit.shape[0:3] + (3,), dtype=numba.double)

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

                    p0 = [2.5, 2.5, 0.0]
                    res = optimize.minimize(
                        beam_chi2, p0, args=(interp_image_to_fit, sampling // 2)
                    )
                    res_x = res.x

                #
                # phi = res_x[2] - 90.0
                phi = res_x[2]
                if phi < -90.0:
                    phi += 180.0

                ellipse_params[time, chan, pol, 0] = np.max(
                    np.abs(res_x[0:2])
                    # ) * np.abs(delta[0] * 2.355 / sampling[0] / npix_window[0])
                ) * np.abs(delta[0] * 2.355)
                ellipse_params[time, chan, pol, 1] = np.min(
                    np.abs(res_x[0:2])
                    # ) * np.abs(delta[1] * 2.355 / sampling[1] / npix_window[1])
                ) * np.abs(delta[1] * 2.355)
                ellipse_params[time, chan, pol, 2] = -phi
                # ellipse_params[time, chan, pol, 2] = -res_x[2]
                print("res_x=", res_x)
    return ellipse_params
