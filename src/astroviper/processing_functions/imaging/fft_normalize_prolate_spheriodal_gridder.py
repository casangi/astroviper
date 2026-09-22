import time

import numpy as np
import scipy.fft
import toolviper.utils.logger as logger
import xarray as xr

# from memory_profiler import profile
from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_correcting_image_1D,
)
from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)

ifft_pair = {
    "uv_sampling": "point_spread_function",
    "visibility": "sky",
}
normalization_key = {
    "uv_sampling": "uv_sampling_normalization",
    "visibility": "visibility_normalization",
}
fft_pair = {
    "point_spread_function": "uv_sampling",
    "sky": "visibility",
}


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
        except ImportError as err:
            raise ImportError(
                "pyfftw is not installed. Install it with: pip install pyfftw"
            ) from err
    raise ValueError(
        f"Unknown FFT backend '{backend}'. Supported values: 'scipy', 'pyfftw'."
    )


# @profile(precision=1)
def ifft_norm_img_xds(
    img_xds,
    image_params,
    image_data_group_in_name="single_field",
    image_data_group_out_name="single_field",
    image_data_group_out_modified=None,
    overwrite=True,
    image_data_variables_keep=None,
    processing_function_threads=1,
    fft_backend="pyfftw",
    complex_dtype=np.complex128,
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

    processing_function_threads : int, optional
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
    When the grid is not in ``image_data_variables_keep`` (the common case)
    each plane is transformed in place (``overwrite_input=True``); keeping the
    grid adds one extra 2-D complex plane per iteration for the defensive
    copy, ≈ 1.15 GB for a 12 000 × 12 000 complex128 grid.
    """
    if image_data_group_out_modified is None:
        image_data_group_out_modified = {
            "sky": "SKY_RESIDUAL",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        }
    if image_data_variables_keep is None:
        image_data_variables_keep = []

    _image_params = image_params  # no mutation below; deep copy not needed

    data_group_in, data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=overwrite,
    )

    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_correcting_image_1D,
    )

    (
        kernel_image_1D_l,
        kernel_image_1D_m,
    ) = create_prolate_spheroidal_correcting_image_1D(
        n_lm_padded=[img_xds.sizes["u"], img_xds.sizes["v"]]
    )

    # for data_variable in ["aperture", "uv_sampling", "visibility"]:
    for data_variable in image_data_group_out_modified:
        data_variable_out = fft_pair[
            data_variable
        ]  # Maps the input data variable to the corresponding output data variable (e.g. "uv_sampling" to "point_spread_function", "visibility" to "sky")
        if data_variable_out not in data_group_in:
            continue

        grid_var_name = data_group_in[data_variable_out]
        raw_grid = img_xds[grid_var_name].values  # (time, freq, pol, u, v)

        normalization = img_xds[
            data_group_in[normalization_key[data_variable_out]]
        ].values.copy()
        normalization[normalization == 0] = 1  # avoid division by zero

        n_time, n_freq, n_pol = raw_grid.shape[:3]
        image_size = np.asarray(_image_params["image_size"])
        padded_image_shape = raw_grid.shape[-2:]
        flux_scale = padded_image_shape[0] * padded_image_shape[1]

        # Process one 2-D plane at a time to keep FFT temporaries small.
        # At 12 000 × 12 000 this limits the extra allocation to ≈ 1.15 GB
        # instead of allocating the full (time, freq, pol, u, v) array.
        #
        # The sky/PSF image inherits the precision of the gridded UV data: a
        # complex64 grid yields a float32 image (half the memory), complex128 a
        # float64 image. The grid is fed to the iFFT at ``complex_dtype`` via
        # ``np.asarray(..., dtype=complex_dtype)`` — a no-op (no copy) when the
        # grid is already that dtype, which is the case for the cube imager.
        float_out_dtype = (
            np.float32 if np.dtype(complex_dtype) == np.complex64 else np.float64
        )
        out_name = data_group_out[ifft_pair[data_variable_out]]
        if out_name not in img_xds:
            img_xds[out_name] = xr.DataArray(
                np.empty(
                    (n_time, n_freq, n_pol, image_size[0], image_size[1]),
                    dtype=float_out_dtype,
                ),
                dims=("time", "frequency", "polarization", "l", "m"),
            )
        out_arr = img_xds[out_name].values  # numpy view for in-place writes
        img_xds[out_name].attrs["type"] = data_variable

        # Each grid plane is read exactly once and the grid variable is
        # deleted right after this loop unless explicitly kept, so the iFFT
        # may destroy the plane in place and skip its defensive copy
        # (≈ 1.5 GB per plane at 13 500² complex64).
        grid_discarded = data_variable_out not in image_data_variables_keep

        for t in range(n_time):
            for f in range(n_freq):
                for p in range(n_pol):
                    plane = ifft_uv_to_lm(
                        np.asarray(raw_grid[t, f, p], dtype=complex_dtype),
                        processing_function_threads=processing_function_threads,
                        fft_backend=fft_backend,
                        overwrite_input=grid_discarded,
                    )
                    # In-place ops below preserve `plane`'s precision (the float64
                    # kernel / normalization are broadcast without upcasting).
                    plane /= kernel_image_1D_l[:, None]
                    plane /= kernel_image_1D_m[None, :]
                    plane *= flux_scale / normalization[t, f, p]

                    out_arr[t, f, p] = remove_padding(plane.real, image_size)

        if data_variable_out not in image_data_variables_keep:
            # Release the large grid from the dataset so it can be freed as soon
            # as `del raw_grid` is called after the loop.
            img_xds.xr_img.delete_data_variables(variables=[grid_var_name])
            del raw_grid  # free ≈ 9 GB as early as possible

        # img_xds[data_group_out[ifft_pair[data_variable_out]]] = xr.DataArray(
        #     result, dims=("time", "frequency", "polarization", "l", "m")
        # )

        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            data_group_out,
            description="Transformed from aperture uv plane to sky lm plane.",
        )

    return img_xds


# @profile(precision=1)
def fft_norm_img_xds(
    img_xds,
    image_params,
    image_data_group_in_name="model",
    image_data_group_out_name="model",
    image_data_group_out_modified=None,
    overwrite=True,
    image_data_variables_keep=None,
    processing_function_threads=1,
    fft_backend="pyfftw",
    data_variables_to_process=None,
    complex_dtype=np.complex128,
):
    """Forward-transform a model sky image into a model UV grid.

    Options are data_variables_to_process = ["primary_beam", "point_spread_function", "sky"]

    ``complex_dtype`` sets the precision of the output model UV grid
    (``complex64`` for a single-precision image, ``complex128`` otherwise). The
    model visibilities degridded from this grid stay double precision regardless
    (the degridder widens each grid cell to ``complex128`` for the
    accumulation), so only the image-domain grid is affected here.
    """
    if image_data_group_out_modified is None:
        image_data_group_out_modified = {
            "visibility": "VISIBILITY_MODEL",
        }
    if image_data_variables_keep is None:
        image_data_variables_keep = []
    if data_variables_to_process is None:
        data_variables_to_process = ["sky"]

    _image_params = image_params  # no mutation below; deep copy not needed

    data_group_in, data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=overwrite,
    )

    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )

    n_uv = padded_grid_size(
        [img_xds.sizes["l"], img_xds.sizes["m"]], _image_params["fft_padding"]
    )

    for data_variable in data_variables_to_process:
        if data_variable not in data_group_in:
            continue

        grid_var_name = data_group_in[data_variable]
        raw_grid = img_xds[grid_var_name].values  # (time, freq, pol, u, v)

        n_time, n_freq, n_pol = raw_grid.shape[:3]

        out_name = data_group_out[fft_pair[data_variable]]
        if out_name not in img_xds:
            img_xds[out_name] = xr.DataArray(
                np.empty(
                    (n_time, n_freq, n_pol, n_uv[0], n_uv[1]), dtype=complex_dtype
                ),
                dims=("time", "frequency", "polarization", "u", "v"),
            )
            # img_xds[out_name] = xr.DataArray(
            #     np.empty((n_time, n_freq, n_pol, img_xds.sizes["l"], img_xds.sizes["m"]), dtype=np.complex128),
            #     dims=("time", "frequency", "polarization", "l", "m"),
            # )
        out_arr = img_xds[out_name].values  # numpy view for in-place writes
        img_xds[out_name].attrs["type"] = data_variable

        (
            kernel_image_1D_l,
            kernel_image_1D_m,
        ) = create_prolate_spheroidal_correcting_image_1D(n_lm_padded=n_uv)

        # kernel_image_1D_l, kernel_image_1D_m = (
        #     create_prolate_spheroidal_correcting_image_1D(
        #         n_lm_padded=[250, 250]
        #     )
        # )

        # Process one 2-D plane at a time to keep FFT temporaries small, and do
        # ALL arithmetic in place on the plane's own buffer. The out-of-place
        # spelling ``out_arr[t, f, p] / kernel_image_1D_l[:, None] / ...`` is a
        # memory disaster: the float64 kernel arrays PROMOTE a complex64 plane
        # to complex128 (array/array promotion), allocating two full-grid
        # complex128 temporaries (≈ 2.9 GB each at 13 500²) plus the cast back
        # -- the 2026-08-16 multi-cycle OOM. In-place division by the float64
        # kernels keeps the plane's dtype and allocates nothing (same pattern,
        # same reason, as ifft_norm_img_xds above). The padded borders are
        # zero, so dividing the full padded plane is harmless.
        for t in range(n_time):
            for f in range(n_freq):
                for p in range(n_pol):
                    add_padding(raw_grid[t, f, p], out_arr[t, f, p])
                    out_arr[t, f, p] /= kernel_image_1D_l[:, None]
                    out_arr[t, f, p] /= kernel_image_1D_m[None, :]

                    # The plane is the output buffer being overwritten anyway,
                    # so the FFT may destroy it (skips the defensive copy).
                    out_arr[t, f, p] = fft_lm_to_uv(
                        out_arr[t, f, p],
                        processing_function_threads=processing_function_threads,
                        fft_backend=fft_backend,
                        complex_dtype=complex_dtype,
                        overwrite_input=True,
                    )

        if data_variable not in image_data_variables_keep:
            # Release the large grid from the dataset so it can be freed as soon
            # as `del raw_grid` is called after the loop.
            img_xds.xr_img.delete_data_variables(
                variables=[grid_var_name]
            )  # Deletes the raw grid from the xds.
            del raw_grid  # free ≈ 9 GB as early as possible

        # img_xds[data_group_out[fft_pair[data_variable]]] = xr.DataArray(
        #     result, dims=("time", "frequency", "polarization", "l", "m")
        # )

        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            data_group_out,
            description="Transformed from lm plane to aperture uv plane.",
        )

    return img_xds


def _fold_shift_checkerboard(arr, axes):
    """Multiply ``arr`` in place by the separable checkerboard ``(-1)**(i+j+...)``.

    For even-length axes an ``fftshift`` / ``ifftshift`` (each a roll by ``N/2``)
    can be replaced by a pointwise multiply by ``(-1)**k`` in the opposite
    domain, so ``fftshift(fft(ifftshift(x))) == cb * fft(cb * x)`` with
    ``cb = (-1)**(sum of axis indices)``.  The checkerboard is applied as a 1-D
    sign broadcast per axis (no full 2-D kernel), which avoids the two
    full-array roll-copies the shifts would make.  ``arr`` keeps its dtype (the
    real sign array broadcasts without upcasting a complex array).
    """
    sign_dtype = arr.real.dtype
    for ax in axes:
        n = arr.shape[ax]
        sign = np.ones(n, dtype=sign_dtype)
        sign[1::2] = -1.0
        shape = [1] * arr.ndim
        shape[ax] = n
        arr *= sign.reshape(shape)
    return arr


def ifft_uv_to_lm(
    grid_2d,
    fft_plane_dims=(-2, -1),
    processing_function_threads=1,
    fft_backend="pyfftw",
    overwrite_input=False,
):
    """Apply a 2-D inverse FFT to transform a UV grid to a sky-plane image.

    Applies the standard radio-astronomy convention:

    * ``ifftshift`` the UV grid to move the DC component to the array origin
      before the transform.
    * Compute the 2-D inverse FFT.
    * ``fftshift`` the result to place the image centre at the array centre.

    The result is returned **complex and unscaled**: this routine applies
    neither the flux normalisation (undoing NumPy's implicit 1/N) nor a
    real-part projection.  Callers are responsible for both --
    :func:`ifft_norm_img_xds` divides by the prolate-spheroidal correction
    and the summed weights and takes ``.real``.  For conjugate-symmetric UV
    data (real sky emission) the imaginary part is negligible, but it is
    still present in the return value.

    Parameters
    ----------
    grid_2d : numpy.ndarray
        Input UV grid.  Can be any number of dimensions; the FFT is applied
        along ``fft_plane_dims``.  For the slice-by-slice path in
        ``ifft_norm_img_xds`` this is a 2-D array of shape ``(u, v)``, in
        either complex precision (``complex64`` or ``complex128``).
    fft_plane_dims : tuple of int, optional
        Axes over which to apply the 2-D FFT.  Default is ``(-2, -1)``.
    threads : int, optional
        Number of threads passed to the FFT backend.  Default is ``1``.
    fft_backend : {"scipy", "pyfftw"}, optional
        FFT library to use.  Default is ``"pyfftw"``.
    overwrite_input : bool, optional
        If ``True`` the function may reuse ``grid_2d``'s buffer as scratch
        space, destroying its contents (on the even-sized fast path this
        skips the one plane-sized defensive copy — ≈ 1.5 GB for a
        13 500 × 13 500 complex64 grid).  The caller must not use ``grid_2d``
        after the call.  Default ``False`` (input preserved).

    Returns
    -------
    numpy.ndarray
        Complex sky-plane image, same spatial shape and dtype as ``grid_2d``.
        Unscaled -- apply the flux normalisation and take ``.real`` in the
        caller (see :func:`ifft_norm_img_xds`).

    Notes
    -----
    When running inside a single-threaded dask worker keep ``threads=1`` to
    avoid spawning threads that fight dask's own scheduler.  Set
    ``threads > 1`` only when calling outside a dask context.
    """
    fft = _fft_module(fft_backend)
    start = time.time()
    even = all(grid_2d.shape[ax] % 2 == 0 for ax in fft_plane_dims)
    if even:
        # Fold the ifftshift/fftshift into checkerboard multiplies (lower peak
        # memory, fewer passes): fftshift(ifft2(ifftshift(x))) == cb*ifft2(cb*x).
        # The padded grid axes are even (next_fft_friendly_size(even=True)).
        # Own the buffer unless the caller ceded it via overwrite_input.
        work = grid_2d if overwrite_input else grid_2d.copy()
        _fold_shift_checkerboard(work, fft_plane_dims)
        sky = fft.ifft2(
            work,
            axes=fft_plane_dims,
            workers=processing_function_threads,
            overwrite_x=True,
        )
        _fold_shift_checkerboard(sky, fft_plane_dims)
    else:
        # Odd axis: the two shifts are not a simple checkerboard; use the rolls.
        sky = fft.fftshift(
            fft.ifft2(
                fft.ifftshift(grid_2d, axes=fft_plane_dims),
                axes=fft_plane_dims,
                workers=processing_function_threads,
            ),
            axes=fft_plane_dims,
        )
    logger.debug("Time for ifft_uv_to_lm: " + str(time.time() - start))
    return sky


def fft_lm_to_uv(
    image,
    fft_plane_dims=(-2, -1),
    processing_function_threads=1,
    fft_backend="pyfftw",
    complex_dtype=np.complex128,
    overwrite_input=False,
):
    """Apply a 2-D FFT to transform a sky-plane image to a UV grid.

    Applies the standard radio-astronomy convention:

    * ``ifftshift`` the image to move the image centre to the array origin
      before the transform.
    * Compute the 2-D FFT.
    * ``fftshift`` the result to place the DC component at the array centre.

    The full **complex** UV grid is returned (no real-part projection and no
    flux normalisation are applied here).  The input is cast to
    ``complex_dtype`` before transforming, so a real sky image is handled
    correctly.

    Parameters
    ----------
    image : numpy.ndarray
        Sky image (typically real-valued -- Stokes I, Q, U, V).  Can be any
        number of dimensions; the FFT is applied along ``fft_plane_dims``.
        Cast to ``complex_dtype`` internally.
    fft_plane_dims : tuple of int, optional
        Axes over which to apply the 2-D FFT.  Default is ``(-2, -1)``.
    threads : int, optional
        Number of threads passed to the FFT backend.  Default is ``1``.
    fft_backend : {"scipy", "pyfftw"}, optional
        FFT library to use.  Default is ``"pyfftw"``.
    complex_dtype : numpy dtype, optional
        Complex dtype the image is cast to and the grid is returned in.
        Default ``numpy.complex128``.
    overwrite_input : bool, optional
        If ``True`` and ``image`` is already ``complex_dtype``, the function
        may use ``image``'s buffer as scratch space, destroying its contents
        (skips the one plane-sized defensive copy — ≈ 1.5 GB for a
        13 500 × 13 500 complex64 grid).  The caller must not rely on
        ``image``'s contents after the call.  Default ``False``.

    Returns
    -------
    numpy.ndarray
        Complex UV grid (dtype ``complex_dtype``), the 2-D FFT of the input
        image with the DC component shifted to the array centre.  Same shape
        as ``image``.
    """
    fft = _fft_module(fft_backend)
    work = np.asarray(image, dtype=complex_dtype)
    even = all(work.shape[ax] % 2 == 0 for ax in fft_plane_dims)
    if even:
        # Fold the shifts into checkerboard multiplies (see ifft_uv_to_lm):
        # fftshift(fft2(ifftshift(x))) == cb * fft2(cb * x).
        # Own the buffer unless the caller ceded it via overwrite_input
        # (asarray is a no-copy view when the dtype already matches).
        if work is image and not overwrite_input:
            work = work.copy()
        _fold_shift_checkerboard(work, fft_plane_dims)
        uv = fft.fft2(
            work,
            axes=fft_plane_dims,
            workers=processing_function_threads,
            overwrite_x=True,
        )
        _fold_shift_checkerboard(uv, fft_plane_dims)
        return uv
    return fft.fftshift(
        fft.fft2(
            fft.ifftshift(work, axes=fft_plane_dims),
            axes=fft_plane_dims,
            workers=processing_function_threads,
        ),
        axes=fft_plane_dims,
    )


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


def add_padding(src, dst):
    """Embed ``src`` into the centre of ``dst`` in place; zero ``dst``'s borders.

    Inverse of :func:`remove_padding`:
    ``remove_padding(dst, src.shape[-2:])`` returns a view of the central
    ``src.shape[-2:]`` pixels of ``dst``; ``add_padding(src, dst)`` writes
    ``src`` into that same centre and zeros every pixel outside it.

    Operates entirely in place on ``dst`` — no new array is allocated.
    The only data movement is the unavoidable copy of ``src`` into the
    central region.  Intended for tight loops where the padded buffer is
    allocated once and reused across iterations.

    Parameters
    ----------
    src : numpy.ndarray
        Unpadded source array.  Any number of leading dimensions are
        supported; the centring is applied to the last two axes
        ``(..., n_l, n_m)``.  Leading dimensions must match ``dst``.
    dst : numpy.ndarray
        Pre-allocated padded buffer to modify in place.  Must have shape
        ``src.shape[:-2] + padded_size`` with
        ``padded_size >= src.shape[-2:]`` element-wise and a dtype
        compatible with ``src``.

    Returns
    -------
    numpy.ndarray
        The same ``dst`` array, with ``src`` placed at the centre of its
        last two axes and the surrounding border strips set to zero.
    """
    src_size = np.array(src.shape[-2:])
    dst_size = np.array(dst.shape[-2:])
    start_xy = dst_size // 2 - src_size // 2
    end_xy = start_xy + src_size
    dst[..., start_xy[0] : end_xy[0], start_xy[1] : end_xy[1]] = src
    dst[..., : start_xy[0], :] = 0
    dst[..., end_xy[0] :, :] = 0
    dst[..., start_xy[0] : end_xy[0], : start_xy[1]] = 0
    dst[..., start_xy[0] : end_xy[0], end_xy[1] :] = 0
    return dst
