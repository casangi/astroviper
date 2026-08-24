import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.optimize as optimize
import xarray as xr
from scipy.interpolate import interpn
from scipy.ndimage import label

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)

FWHM_factor = 2.355  # approx 2*sqrt(2*ln(2))
beam_initial_guess = [2.5, 2.5, 0.0]  # initial guess for [bmaj, bmin, pa]
expand_pixel = 5  # expand the fitting window by this many pixels on each side


def point_spread_function_gaussian_fit(
    img_xds: xr.Dataset,
    image_data_group_in_name="image",
    image_data_group_out_name="image",
    image_data_group_out_modified={
        "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
        "max_sidelobe_point_spread_function": "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
    },
    overwrite=True,
    npix_window: tuple = (41, 41),
    sampling: tuple = (55, 55),
    cutoff: float = 0.35,
    interpolation_method: str = "splinef2d",
    processing_function_threads: int = 1,
):
    """
    fit 2D gaussian to psf

    For every (time, frequency, polarization) slice the main lobe of the PSF is
    isolated and a 2D Gaussian is fit to it. The maximum sidelobe is measured
    after subtracting that fitted main beam, following CASA's cycle-threshold
    convention. Both results are written back into ``img_xds``.

    Parameters
    ----------
    img_xds : xarray.Dataset
        The input data cube.
    image_data_group_in_name : str
        The name of the image data group to fit. Default is 'image'.
    image_data_group_out_name : str
        The name of the image data group to write the results to. Default is
        'image'.
    image_data_group_out_modified : dict
        Maps the logical output names to the dataset variable names used to
        store the results. Defaults to writing the Gaussian fit parameters to
        ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION`` (key
        ``beam_fit_params_point_spread_function``) and the maximum sidelobe
        level to ``MAX_SIDELOBE_POINT_SPREAD_FUNCTION`` (key
        ``max_sidelobe_point_spread_function``).
    overwrite : bool, default True
        Whether to overwrite an existing output data group of the same name.
    npix_window : tuple
        The size of the search window in pixels. Both values must be odd
        integers so the window is centered on the peak.
    sampling : tuple
        The sampling of the fitting grid in pixels.
    cutoff : float
        The cutoff value for the fitting.
    interpolation_method : str
        The interpolation method to use when resampling the PSF for fitting.
    processing_function_threads : int, default 1
        Number of worker threads used to fit individual (time, frequency,
        polarization) PSF slices concurrently via a ``ThreadPoolExecutor``.
        Values ``<= 0`` fall back to ``os.cpu_count()``.  Each slice writes to
        its own output entry, so the threaded result is identical to the
        serial one.

    Returns
    -------
    xds : xarray.Dataset
        The input image with two data variables added (named according to
        ``image_data_group_out_modified``):

        - ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION`` with dims
          ``(time, frequency, polarization, beam_params)`` holding the fitted
          ``[major, minor, pa]`` for each slice.
        - ``MAX_SIDELOBE_POINT_SPREAD_FUNCTION`` with dims
          ``(time, frequency, polarization)`` holding the maximum sidelobe
          magnitude after subtracting the fitted main beam for each slice.

        The l and m coordinates of the input data are assumed to be in radians.
        The units of beam size (major and minor) and position angle are in radians.

    Notes:
    -----
    - Returns NaN values for beam parameters if the fitting fails
    - L-BFGS-B optimization method is used with bounds on parameters
    - The maximum sidelobe level is a fraction of the PSF peak (dimensionless);
      it is ``0.0`` for an all-zero slice.
    """

    if not isinstance(npix_window, (list, tuple, np.ndarray)):
        raise TypeError("npix_window must be a list, tuple, or numpy array")
    if npix_window[0] <= 0 or npix_window[1] <= 0:
        raise ValueError("npix_window must be positive")
    if type(npix_window[0]) is not int or type(npix_window[1]) is not int:
        raise TypeError("npix_window must be integers")
    if npix_window[0] % 2 == 0 or npix_window[1] % 2 == 0:
        raise ValueError(
            "npix_window must be odd so the search window is centered on the peak"
        )
    if not isinstance(sampling, (list, tuple, np.ndarray)):
        raise TypeError("sampling must be a list, tuple, or numpy array")
    if sampling[0] <= 0 or sampling[1] <= 0:
        raise ValueError("sampling must be positive")
    if type(sampling[0]) is not int or type(sampling[1]) is not int:
        raise TypeError("sampling must be integers")
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")

    image_data_group_in, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=overwrite,
    )
    # print(img_xds.attrs["data_groups"])
    # print("image_data_group_in ", image_data_group_in)
    psf_name = image_data_group_in["point_spread_function"]

    sampling = np.array(sampling)
    npix_window = np.array(npix_window)

    # pixel increments (radians)
    delta = np.array(
        [
            img_xds[psf_name]["l"][1] - img_xds[psf_name]["l"][0],
            img_xds[psf_name]["m"][1] - img_xds[psf_name]["m"][0],
        ]
    )
    # To make fitting result expressed in arcsecond uncomment the below
    #    * 3600
    #    * 180
    #    / np.pi

    # px, py, amp, psf2d = locate_peak_psf(img_xds[psf_name].values)
    # print(" locate_peak_psf: px, py, amp=", px, py, amp)
    # if amp < 1e-7:
    #    raise ValueError("PSF peak amplitude is zero")

    # npoints, blc, trc, x, y, sigma = find_n_points(
    #    npix_window, cutoff, px, py, psf2d, delta
    # )
    # print(" after find_n_points blc, trc=", blc, trc)
    main_lobe_im, blc, trc, max_sidelobe = extract_main_lobe(
        npix_window, cutoff, img_xds[psf_name].values
    )
    # blc/trc are per-slice arrays of shape (time, frequency, polarization, 2).
    # Expand each slice's fitting window and clamp it to the image bounds.
    blc = blc - expand_pixel
    trc = trc + expand_pixel
    # print(" blc, trc after expanding=", blc, trc)
    blc = np.maximum(blc, 0)
    trc[..., 0] = np.minimum(trc[..., 0], main_lobe_im.shape[3] - 1)
    trc[..., 1] = np.minimum(trc[..., 1], main_lobe_im.shape[4] - 1)

    ellipse_params = psf_gaussian_fit_core(
        img_xds[psf_name].values,
        blc,
        trc,
        sampling,
        cutoff,
        delta,
        interpolation_method,
        processing_function_threads=processing_function_threads,
    )
    max_sidelobe = _max_sidelobe_after_gaussian_subtraction(
        img_xds[psf_name].values,
        ellipse_params,
        delta,
        fallback=max_sidelobe,
    )

    # Uncomment line below to change beam_param units to arcsec and deg
    # psf_gaussian_fit_core returns bmaj and bmin in  and pa in deg.
    # Converting to radian for storing to the xradio image
    # ellipse_params[..., :2] = np.deg2rad(ellipse_params[..., :2] / 3600.0)
    # ellipse_params[..., 2] = np.deg2rad(ellipse_params[..., 2])

    img_xds = img_xds.assign_coords(
        beam_params_label=["major", "minor", "pa"],
    )

    img_xds[
        image_data_group_out["beam_fit_params_point_spread_function"]
    ] = xr.DataArray(
        ellipse_params,
        dims=["time", "frequency", "polarization", "beam_params"],
        attrs={"unit": "rad"},
    )

    img_xds[image_data_group_out["max_sidelobe_point_spread_function"]] = xr.DataArray(
        max_sidelobe,
        dims=["time", "frequency", "polarization"],
        attrs={"unit": ""},
    )

    modify_data_groups_xds(
        img_xds,
        image_data_group_out_name,
        image_data_group_out,
        description="Added UV sampling grid to img_xds with add_uv_sampling_grid_single_field.",
    )

    return img_xds


def _max_sidelobe_after_gaussian_subtraction(
    psf_image,
    ellipse_params,
    delta,
    fallback=None,
):
    """Measure CASA-style PSF sidelobes after removing the fitted main beam."""
    psf_image = np.asarray(psf_image)
    ellipse_params = np.asarray(ellipse_params)
    output = np.zeros(psf_image.shape[:3], dtype=np.float64)

    for index in np.ndindex(psf_image.shape[:3]):
        psf_2d = psf_image[index]
        finite = np.isfinite(psf_2d)
        beam = ellipse_params[index]
        valid_beam = np.all(np.isfinite(beam)) and np.all(beam[:2] > 0.0)
        if not np.any(finite) or not valid_beam:
            if fallback is not None:
                output[index] = fallback[index]
            continue

        finite_psf = np.where(finite, psf_2d, 0.0)
        peak_l, peak_m = np.unravel_index(np.argmax(finite_psf), psf_2d.shape)
        peak = finite_psf[peak_l, peak_m]
        if peak <= 0.0:
            if fallback is not None:
                output[index] = fallback[index]
            continue

        l_offset = (np.arange(psf_2d.shape[0]) - peak_l) * abs(delta[0])
        m_offset = (np.arange(psf_2d.shape[1]) - peak_m) * abs(delta[1])
        l_grid, m_grid = np.meshgrid(l_offset, m_offset, indexing="ij")
        cos_pa = np.cos(beam[2])
        sin_pa = np.sin(beam[2])
        major_offset = l_grid * cos_pa - m_grid * sin_pa
        minor_offset = l_grid * sin_pa + m_grid * cos_pa
        sigma_major = beam[0] / FWHM_factor
        sigma_minor = beam[1] / FWHM_factor
        fitted_main_beam = peak * np.exp(
            -0.5
            * ((major_offset / sigma_major) ** 2 + (minor_offset / sigma_minor) ** 2)
        )

        delobed_maximum = np.max(np.where(finite, psf_2d - fitted_main_beam, -np.inf))
        original_minimum = np.min(np.where(finite, psf_2d, np.inf))
        output[index] = max(abs(original_minimum), abs(delobed_maximum))

    return output


def _get_main_lobe_bounding_box(masked_psf_2d):
    """
    Given a 2D image of the main lobe with zero values outside the lobe, find the
    bounding box that encloses all of its non-zero pixels. The l-extent is
    widened to cover the m-extent where needed (the original near-square
    heuristic, preserved here).

    Args:
        masked_psf_2d (np.ndarray): A 2D (l, m) image where the main lobe is
            non-zero and all other pixels are zero.
    Returns:
        tuple: Bounding box corners (blc, trc) as
            (np.array([l_min, m_min]), np.array([l_max, m_max])), or (None, None)
            when there are no non-zero pixels.
    """
    valid_pixels = masked_psf_2d != 0

    valid_indices = np.where(valid_pixels)

    if len(valid_indices[0]) == 0:
        return None, None  # No valid pixels found

    l_min, l_max = valid_indices[0].min(), valid_indices[0].max()
    m_min, m_max = valid_indices[1].min(), valid_indices[1].max()

    # Ensure the bounding box is square
    if l_min > m_min:
        l_min = m_min
    if l_max < m_max:
        l_max = m_max

    return np.array([l_min, m_min]), np.array([l_max, m_max])


def _extract_main_lobe_2d(npix_window, threshold, psf_2d):
    """
    Extract the main lobe from a single 2D (l, m) PSF slice.

    Args:
        npix_window (tuple): The size of the search window in pixels.
        threshold (float): A threshold as a fraction of the peak value used to
            isolate the main lobe.
        psf_2d (np.ndarray): A single 2D PSF slice with dimensions (l, m).

    Returns:
        main_lobe_only (np.ndarray): The 2D slice with everything outside the
            main lobe zeroed out.
        blc (np.ndarray): Bottom-left corner of the main lobe as [l, m].
        trc (np.ndarray): Top-right corner of the main lobe as [l, m].
        max_sidelobe (float): Largest value outside the main lobe.
    """
    n_l, n_m = psf_2d.shape

    # Find the peak intensity of this slice
    peak_intensity = np.max(psf_2d)
    if peak_intensity == 0:
        return (
            np.zeros_like(psf_2d),
            np.array([0, 0]),
            np.array([n_l - 1, n_m - 1]),
            0.0,  # max_sidelobe is 0.0 if peak_intensity is 0
        )

    # Locate the peak within this slice
    peak_l, peak_m = np.unravel_index(np.argmax(psf_2d), psf_2d.shape)

    # Zero everything outside a window centred on the peak to limit the search.
    # npix_window is enforced odd upstream, so peak +/- npix_window // 2 (with an
    # inclusive upper bound) yields exactly npix_window pixels centred on the peak.
    windowed_psf = np.zeros_like(psf_2d)
    l0 = max(0, peak_l - npix_window[0] // 2)
    l1 = min(n_l, peak_l + npix_window[0] // 2 + 1)
    m0 = max(0, peak_m - npix_window[1] // 2)
    m1 = min(n_m, peak_m + npix_window[1] // 2 + 1)
    windowed_psf[l0:l1, m0:m1] = psf_2d[l0:l1, m0:m1]

    # Determine a threshold based on the peak intensity
    abs_threshold = peak_intensity * threshold

    # Create a binary mask where pixels above the threshold are True and use
    # SciPy's `label` to find connected components within this slice only.
    binary_mask = windowed_psf > abs_threshold
    labels, num_features = label(binary_mask)

    # The main lobe is the connected component containing this slice's peak.
    main_lobe_label = labels[peak_l, peak_m]
    main_lobe_only = np.where(labels == main_lobe_label, windowed_psf, 0)

    # Largest value that does not belong to the main lobe.
    # Cycle control depends on the largest sidelobe magnitude. A negative
    # sidelobe is just as capable of destabilizing a minor cycle as a positive
    # one, so do not discard it when estimating the safe cycle threshold.
    max_sidelobe = np.max(np.abs(psf_2d) * (labels != main_lobe_label))

    blc, trc = _get_main_lobe_bounding_box(main_lobe_only)
    if blc is None:
        # No non-zero pixels survived; fall back to the full slice.
        blc = np.array([0, 0])
        trc = np.array([n_l - 1, n_m - 1])
    return main_lobe_only, blc, trc, max_sidelobe


def extract_main_lobe(npix_window, threshold, psf_image):
    """
    Extracts the main lobe from a PSF image independently for every
    (time, frequency, polarization) slice.

    Each 2D (l, m) slice is processed on its own, so the returned bounding box
    and sidelobe level reflect that slice's main lobe rather than a single box
    derived from the global peak. The window defined by ``npix_window`` is used
    to handle large images efficiently.

    Args:
        npix_window (tuple): The size of the window in pixels for searching features.
        threshold (float): A threshold in fraction of peak value for determining the main lobe.
        psf_image (np.ndarray): The input PSF image with 5 dimensions (time, frequency, polarization, l, m).

    Returns:
        main_lobe_only (np.ndarray): A 5D array containing only each slice's main
            lobe, with other regions zeroed out.
        blc (np.ndarray): Per-slice bottom-left corners of the bounding box,
            shape (time, frequency, polarization, 2) as [l, m].
        trc (np.ndarray): Per-slice top-right corners of the bounding box,
            shape (time, frequency, polarization, 2) as [l, m].
        max_sidelobe (np.ndarray): Per-slice maximum sidelobe level,
            shape (time, frequency, polarization).
    """
    n_time, n_freq, n_pol = psf_image.shape[0:3]

    main_lobe_only = np.zeros_like(psf_image)
    blc = np.zeros((n_time, n_freq, n_pol, 2), dtype=int)
    trc = np.zeros((n_time, n_freq, n_pol, 2), dtype=int)
    max_sidelobe = np.zeros((n_time, n_freq, n_pol), dtype=np.float64)

    for itime in range(n_time):
        for ifreq in range(n_freq):
            for ipol in range(n_pol):
                (
                    lobe_2d,
                    slice_blc,
                    slice_trc,
                    slice_sidelobe,
                ) = _extract_main_lobe_2d(
                    npix_window, threshold, psf_image[itime, ifreq, ipol]
                )
                main_lobe_only[itime, ifreq, ipol] = lobe_2d
                blc[itime, ifreq, ipol] = slice_blc
                trc[itime, ifreq, ipol] = slice_trc
                max_sidelobe[itime, ifreq, ipol] = slice_sidelobe

    return main_lobe_only, blc, trc, max_sidelobe


def beam_chi2(params, psf_ravel_masked, x_grid, y_grid, psf_mask):
    """
    Chi-squared function for beam fitting.

    psf_ravel_masked : pre-ravelled and masked PSF values (non-NaN pixels)
    x_grid, y_grid   : pre-computed centred coordinate grids (shape: sampling_x, sampling_y)
    psf_mask         : boolean mask of non-NaN pixels in the ravelled PSF
    """
    width_x, width_y, rotation = params
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    xp = x_grid * cos_r - y_grid * sin_r
    yp = x_grid * sin_r + y_grid * cos_r
    gaussian = np.exp(-(xp**2 / width_x**2 + yp**2 / width_y**2) / 2.0)
    return np.sum((gaussian.ravel()[psf_mask] - psf_ravel_masked) ** 2)


def psf_gaussian_fit_core(
    image_to_fit,
    blc,
    trc,
    sampling,
    cutoff,
    delta,
    interpolation_method="linear",
    processing_function_threads: int = 1,
):
    """
    core function to fit gaussian to psf

    Each (time, frequency, polarization) slice is fit independently using its
    own bounding box, so a different main-lobe window may be used per slice.

    Parameters
    ----------
    image_to_fit : np.ndarray
        The input data cube (time, frequency, polarization, l, m).
    blc : np.ndarray
        Bottom-left corner(s) of the fitting window. Either a single ``[l, m]``
        pair applied to every slice, or a per-slice array of shape
        ``(time, frequency, polarization, 2)``.
    trc : np.ndarray
        Top-right corner(s) of the fitting window, with the same shape
        conventions as ``blc``.
    sampling : np.ndarray
        The sampling of the fitting grid in pixels.
    cutoff : float
        The cutoff value for the fitting.
    delta : np.ndarray
        The pixel size in radians.
    processing_function_threads : int, default 1
        Number of worker threads used to fit (time, chan, pol) slices
        concurrently via a ``ThreadPoolExecutor``. Values ``<= 0`` fall
        back to ``os.cpu_count()``. Each slice writes to its own entry in
        the output, so results are independent of the thread count.
    """
    ellipse_params = np.zeros(image_to_fit.shape[0:3] + (3,), dtype=np.float64)
    if np.all(np.isnan(image_to_fit)):
        return ellipse_params + np.nan
    elif np.all(image_to_fit == 0):
        return ellipse_params

    n_time, n_chan, n_pol = image_to_fit.shape[0:3]

    # Normalise blc/trc to per-slice arrays so a single ``[l, m]`` pair is
    # broadcast to every (time, frequency, polarization) slice, while per-slice
    # arrays of shape (time, frequency, polarization, 2) are used as-is.
    blc = np.asarray(blc, dtype=int)
    trc = np.asarray(trc, dtype=int)
    if blc.ndim == 1:
        blc = np.broadcast_to(blc, (n_time, n_chan, n_pol, 2))
        trc = np.broadcast_to(trc, (n_time, n_chan, n_pol, 2))

    # Pre-compute centred coordinate grids for beam_chi2 (depend only on sampling)
    half_s = sampling // 2
    ix = np.arange(sampling[0]) - half_s[0]
    iy = np.arange(sampling[1]) - half_s[1]
    x_grid = (
        np.repeat(ix, sampling[1]).reshape(sampling[0], sampling[1]).astype(np.float64)
    )
    y_grid = (
        np.repeat(iy, sampling[0])
        .reshape(sampling[1], sampling[0])
        .T.astype(np.float64)
    )

    bound = [(None, None), (None, None), (-np.pi / 2, np.pi / 2)]

    def _fit_slice(time, chan, pol):
        # Each slice uses its own bounding box, so the fitting window and the
        # interpolation grid are rebuilt per slice.
        xmin = blc[time, chan, pol, 0]
        ymin = blc[time, chan, pol, 1]
        xmax = trc[time, chan, pol, 0] + 1
        ymax = trc[time, chan, pol, 1] + 1
        npix_window = np.array([xmax - xmin, ymax - ymin])
        slice_to_fit = image_to_fit[time, chan, pol, xmin:xmax, ymin:ymax]

        d0 = np.arange(0, npix_window[0]) * np.abs(delta[0])
        d1 = np.arange(0, npix_window[1]) * np.abs(delta[1])
        interp_d0 = np.linspace(0, npix_window[0] - 1, sampling[0]) * np.abs(delta[0])
        interp_d1 = np.linspace(0, npix_window[1] - 1, sampling[1]) * np.abs(delta[1])

        d0_shape = interp_d0.shape[0]
        d1_shape = interp_d1.shape[0]

        xp_grid = np.repeat(interp_d0, d1_shape).reshape(d0_shape, d1_shape)
        yp_grid = np.repeat(interp_d1, d0_shape).reshape(d1_shape, d0_shape).T
        points = np.vstack((np.ravel(xp_grid), np.ravel(yp_grid))).T

        bmaj_scale = np.abs(delta[0] * FWHM_factor / (sampling[0] / npix_window[0]))
        bmin_scale = np.abs(delta[1] * FWHM_factor / (sampling[1] / npix_window[1]))

        interp_image_to_fit = np.reshape(
            interpn(
                (d0, d1),
                slice_to_fit,
                points,
                method=interpolation_method,
            ),
            [sampling[1], sampling[0]],
        ).T
        interp_image_to_fit[interp_image_to_fit < cutoff] = np.nan

        # Pre-compute mask and masked PSF values once per slice
        psf_mask = ~np.isnan(interp_image_to_fit.ravel())
        psf_ravel_masked = interp_image_to_fit.ravel()[psf_mask]

        res = optimize.minimize(
            beam_chi2,
            beam_initial_guess,
            args=(psf_ravel_masked, x_grid, y_grid, psf_mask),
            bounds=bound,
        )
        if not res.success:
            # Could retry with a lowered cutoff as CASA does, but since the
            # cutoff is also used outside the loop, implementing retry would
            # require some refactoring.
            res_x = np.array([np.nan, np.nan, np.nan])
        else:
            res_x = res.x

        phi = res_x[2]
        if np.argmax(res_x[0:2]) == 1:
            phi = -(np.pi / 2 - phi)
        if phi < 0:
            phi = (phi + np.pi) % np.pi

        ellipse_params[time, chan, pol, 0] = np.max(np.abs(res_x[0:2])) * bmaj_scale
        ellipse_params[time, chan, pol, 1] = np.min(np.abs(res_x[0:2])) * bmin_scale
        ellipse_params[time, chan, pol, 2] = phi

    tasks = [
        (time, chan, pol)
        for time in range(n_time)
        for chan in range(n_chan)
        for pol in range(n_pol)
    ]

    if processing_function_threads <= 0:
        resolved_threads = os.cpu_count() or 1
    else:
        resolved_threads = processing_function_threads
    resolved_threads = max(1, min(resolved_threads, len(tasks)))

    if resolved_threads == 1 or len(tasks) <= 1:
        for t, c, p in tasks:
            _fit_slice(t, c, p)
    else:
        with ThreadPoolExecutor(max_workers=resolved_threads) as pool:
            # Consume the iterator to surface any exceptions raised by a worker.
            for _ in pool.map(lambda tcp: _fit_slice(*tcp), tasks):
                pass

    return ellipse_params
