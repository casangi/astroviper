import time

import numpy as np
import scipy.fft
import xarray as xr
import toolviper.utils.logger as logger


from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)


def _fft_module(backend):
    """Return the FFT module for the requested backend.

    Parameters
    ----------
    backend : {"scipy", "pyfftw"}
        ``"scipy"``  — use ``scipy.fft`` (default, always available).
        ``"pyfftw"`` — use ``pyfftw.interfaces.scipy_fft``, which is a
        drop-in replacement for ``scipy.fft`` backed by FFTW.  The plan
        cache is enabled on first call so that repeated transforms of the
        same shape reuse pre-computed plans.

    Returns
    -------
    module
        An object exposing ``fft2``, ``ifft2``, ``fftshift``, and
        ``ifftshift`` with the same signatures as ``scipy.fft``.

    Raises
    ------
    ImportError
        If ``backend="pyfftw"`` and pyfftw is not installed.
    ValueError
        If ``backend`` is not a recognised string.
    """
    if backend == "scipy":
        return scipy.fft
    if backend == "pyfftw":
        try:
            import pyfftw
            import pyfftw.interfaces.scipy_fft as _pyfftw_fft

            pyfftw.interfaces.cache.enable()
            return _pyfftw_fft
        except ImportError:
            raise ImportError(
                "pyfftw is not installed. Install it with: pip install pyfftw"
            )
    raise ValueError(
        f"Unknown FFT backend '{backend}'. Supported values: 'scipy', 'pyfftw'."
    )


def ifft_norm_img_xds(
    img_xds,
    image_params,
    image_data_group_in_name="single_field",
    image_data_group_out_name="single_field",
    image_data_group_out_modified={
        "sky": "SKY_RESIDUAL",
        "point_spread_function": "POINT_SPREAD_FUNCTION",
    },
    overwrite=True,
    image_data_variables_keep=[],
    threads=1,
    fft_backend="scipy",
):
    """Normalize and inverse-Fourier-transform gridded UV data to sky images.

    For each data variable present in ``data_group_in`` (``"aperture"``,
    ``"uv_sampling"``, ``"visibility"``), the function:

    1. Retrieves the gridded UV array from ``img_xds``.
    2. Applies a 2-D inverse FFT (UV → lm) independently on each
       ``(time, frequency, polarization)`` slice so that only one 2-D plane
       is in memory during the transform.
    3. Divides in-place by the prolate-spheroidal gridding-correction function
       along each image axis.
    4. Crops the zero-padded margins to the requested ``image_size``.
    5. Normalises by the corresponding grid-weight sum, replacing zero weights
       with 1 to avoid division by zero.
    6. Stores the result back into ``img_xds`` under the name given by
       ``data_group_out``.
    7. Frees the raw UV grid from ``img_xds`` and from memory unless the
       variable is listed in ``image_data_variables_keep``.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Dataset containing gridded UV arrays and their normalisation scalars.
        Grid arrays must have dimensions
        ``(time, frequency, polarization, u, v)``; normalisation scalars must
        have dimensions ``(time, frequency, polarization)``.
    image_params : dict
        Imaging configuration.  Must contain:

        ``"image_size"`` : array-like of int, shape (2,)
            Pixel dimensions ``[n_l, n_m]`` of the *unpadded* output image.
    data_group_in : dict
        Maps a logical data-variable role to the corresponding variable name
        in ``img_xds``.  Supported roles for grids: ``"aperture"``,
        ``"uv_sampling"``, ``"visibility"``.  Also maps normalisation roles to
        their variable names: ``"aperture_normalization"``,
        ``"uv_sampling_normalization"``, ``"visibility_normalization"``.

        Example::

            {
                "visibility": "VISIBILITY",
                "uv_sampling": "UV_SAMPLING",
                "visibility_normalization": "VISIBILITY_NORMALIZATION",
                "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
            }

    data_group_out : dict
        Maps a logical output role to the target variable name in ``img_xds``.
        Roles consumed: ``"sky"``, ``"point_spread_function"``,
        ``"primary_beam"``.
    image_data_variables_keep : list of str
        Logical data-variable roles (``"aperture"``, ``"uv_sampling"``,
        ``"visibility"``) whose raw UV grid should be retained in ``img_xds``
        after the FFT.  Any role *not* in this list has its grid replaced with
        an empty array and freed from memory.  Pass an empty list to free all
        grids (lowest peak memory); pass all roles to keep all grids
        (useful for diagnostics).

    threads : int, optional
        Number of threads passed to the FFT backend for each 2-D transform.
        Default is ``1``, which is appropriate for a single-threaded dask
        worker.  Set to a higher value only when calling outside of a dask
        context (e.g. in a notebook or test).
    fft_backend : {"scipy", "pyfftw"}, optional
        FFT library to use.  Default is ``"scipy"``.  Use ``"pyfftw"`` for
        potentially faster transforms when pyfftw is installed; plan caching
        is especially beneficial when the same grid shape is transformed
        repeatedly across major cycles.

    Returns
    -------
    None
        ``img_xds`` is modified in place.  Sky / PSF / primary-beam arrays are
        added as new data variables with dimensions
        ``(time, frequency, polarization, l, m)``.

    Notes
    -----
    Peak memory is dominated by the raw UV grid
    (``time × frequency × polarization × u × v × 16`` bytes for complex128).
    The FFT step adds at most one extra 2-D float64 plane per iteration,
    which for a 12 000 × 12 000 grid is ≈ 1.15 GB.
    """
    _image_params = image_params  # no mutation below; deep copy not needed

    data_group_in, data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=overwrite,
    )

    fft_pair = {
        "uv_sampling": "point_spread_function",
        "visibility": "sky",
    }
    normalization_key = {
        "uv_sampling": "uv_sampling_normalization",
        "visibility": "visibility_normalization",
    }

    from astroviper.core.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_correcting_image_1D,
    )

    kernel_image_1D_l, kernel_image_1D_m = (
        create_prolate_spheroidal_correcting_image_1D(
            n_lm_padded=[img_xds.sizes["u"], img_xds.sizes["v"]]
        )
    )

    for data_variable in ["aperture", "uv_sampling", "visibility"]:
        if data_variable not in data_group_in:
            continue

        grid_var_name = data_group_in[data_variable]
        raw_grid = img_xds[grid_var_name].values  # (time, freq, pol, u, v)

        normalization = img_xds[
            data_group_in[normalization_key[data_variable]]
        ].values.copy()
        normalization[normalization == 0] = 1  # avoid division by zero

        n_time, n_freq, n_pol = raw_grid.shape[:3]
        image_size = np.asarray(_image_params["image_size"])
        result = np.empty(
            (n_time, n_freq, n_pol, image_size[0], image_size[1]),
            dtype=np.float64,
        )

        # Process one 2-D plane at a time to keep FFT temporaries small.
        # At 12 000 × 12 000 this limits the extra allocation to ≈ 1.15 GB
        # instead of allocating the full (time, freq, pol, u, v) float64 array.
        for t in range(n_time):
            for f in range(n_freq):
                for p in range(n_pol):
                    plane = ifft_uv_to_lm(
                        raw_grid[t, f, p], threads=threads, fft_backend=fft_backend
                    )  # (u_pad, v_pad)
                    # Divide in-place to avoid allocating temporaries.
                    plane /= kernel_image_1D_l[:, None]
                    plane /= kernel_image_1D_m[None, :]
                    result[t, f, p] = (
                        remove_padding(plane, image_size) / normalization[t, f, p]
                    )

        if data_variable not in image_data_variables_keep:
            # Release the large grid from the dataset so it can be freed as soon
            # as `del raw_grid` is called after the loop.
            del img_xds[grid_var_name]
            del raw_grid  # free ≈ 9 GB as early as possible

        img_xds[data_group_out[fft_pair[data_variable]]] = xr.DataArray(
            result, dims=("time", "frequency", "polarization", "l", "m")
        )

        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            data_group_out,
            description="Transformed from aperture uv plane to sky lm plane.",
        )

    return img_xds


def ifft_uv_to_lm(grid_2d, fft_plane_dims=(-2, -1), threads=1, fft_backend="scipy"):
    """Apply a 2-D inverse FFT to transform a UV grid to a sky-plane image.

    Applies the standard radio-astronomy convention:

    * ``ifftshift`` the UV grid to move the DC component to the array origin
      before the transform.
    * Compute the 2-D inverse FFT.
    * ``fftshift`` the result to place the image centre at the array centre.
    * Multiply by the number of UV cells to obtain the correct flux scale
      (undoing NumPy's implicit 1/N normalisation).
    * Discard the imaginary part; for conjugate-symmetric UV data (real sky
      emission) it should be negligible.

    Parameters
    ----------
    grid_2d : numpy.ndarray
        Input UV grid.  Can be any number of dimensions; the FFT is applied
        along ``fft_plane_dims``.  For the slice-by-slice path in
        ``ifft_norm_img_xds`` this is a 2-D array of shape ``(u, v)``.
        dtype must be complex128.
    fft_plane_dims : tuple of int, optional
        Axes over which to apply the 2-D FFT.  Default is ``(-2, -1)``.
    threads : int, optional
        Number of threads passed to the FFT backend.  Default is ``1``.
    fft_backend : {"scipy", "pyfftw"}, optional
        FFT library to use.  Default is ``"scipy"``.

    Returns
    -------
    numpy.ndarray
        Real-valued sky image of the same spatial shape as ``grid_2d``.
        Pixel values are in units of Jy/beam (flux scale) consistent with
        the gridded weights.

    Notes
    -----
    When running inside a single-threaded dask worker keep ``threads=1`` to
    avoid spawning threads that fight dask's own scheduler.  Set
    ``threads > 1`` only when calling outside a dask context.
    """
    fft = _fft_module(fft_backend)
    start = time.time()
    n_v, n_u = grid_2d.shape[fft_plane_dims[0]], grid_2d.shape[fft_plane_dims[1]]
    sky = fft.fftshift(
        fft.ifft2(
            fft.ifftshift(grid_2d, axes=fft_plane_dims),
            axes=fft_plane_dims,
            workers=threads,
        ),
        axes=fft_plane_dims,
    ).real * (n_u * n_v)
    logger.debug("Time for ifft_uv_to_lm: " + str(time.time() - start))
    return sky


def fft_lm_to_uv(image, fft_plane_dims=(-2, -1), threads=1, fft_backend="scipy"):
    """Apply a 2-D FFT to transform a sky-plane image to a UV grid.

    Applies the standard radio-astronomy convention:

    * ``ifftshift`` the image to move the image centre to the array origin
      before the transform.
    * Compute the 2-D FFT.
    * ``fftshift`` the result to place the DC component at the array centre.
    * Return only the real part (valid when the input image is real-valued,
      which holds for Stokes I, Q, U, V images).

    Parameters
    ----------
    image : numpy.ndarray
        Real-valued sky image.  Can be any number of dimensions; the FFT is
        applied along ``fft_plane_dims``.
    fft_plane_dims : tuple of int, optional
        Axes over which to apply the 2-D FFT.  Default is ``(-2, -1)``.
    threads : int, optional
        Number of threads passed to the FFT backend.  Default is ``1``.
    fft_backend : {"scipy", "pyfftw"}, optional
        FFT library to use.  Default is ``"scipy"``.

    Returns
    -------
    numpy.ndarray
        Real part of the 2-D FFT of the input image, with the DC component
        shifted to the array centre.  Same shape as ``image``.

    Notes
    -----
    Discarding the imaginary part is only valid when ``image`` is strictly
    real-valued.  If complex images are passed the imaginary component is
    silently lost.
    """
    fft = _fft_module(fft_backend)
    return fft.fftshift(
        fft.fft2(
            fft.ifftshift(image, axes=fft_plane_dims),
            axes=fft_plane_dims,
            workers=threads,
        ),
        axes=fft_plane_dims,
    ).real


def remove_padding(image, image_size):
    """Crop a zero-padded image to the requested unpadded size.

    Assumes the padded image is centred and symmetric, and extracts the
    central ``image_size`` pixels along the last two axes.

    Parameters
    ----------
    image : numpy.ndarray
        Padded image array.  Any number of leading dimensions are supported;
        the crop is applied only to the last two axes ``(..., n_u_pad, n_v_pad)``.
    image_size : array-like of int, shape (2,)
        Target pixel dimensions ``[n_l, n_m]`` of the unpadded output.
        Must satisfy ``image_size <= image.shape[-2:]`` element-wise.

    Returns
    -------
    numpy.ndarray
        A *view* (not a copy) into ``image`` with the last two dimensions
        trimmed to ``image_size``.  Subsequent in-place operations on the
        returned array will modify ``image``.

    Notes
    -----
    Because a view is returned, avoid in-place modification of the output
    if the padded ``image`` array is still needed.
    """
    image_size = np.asarray(image_size)
    image_size_padded = np.array(image.shape[-2:])
    start_xy = image_size_padded // 2 - image_size // 2
    end_xy = start_xy + image_size
    return image[..., start_xy[0] : end_xy[0], start_xy[1] : end_xy[1]]
