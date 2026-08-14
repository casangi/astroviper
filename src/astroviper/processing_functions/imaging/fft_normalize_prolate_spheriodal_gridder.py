import time

import numpy as np
import scipy.fft
import toolviper.utils.logger as logger
import xarray as xr

# from memory_profiler import profile
from toolviper.utils.memory_management import get_rss_gb

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
        except ImportError:
            raise ImportError(
                "pyfftw is not installed. Install it with: pip install pyfftw"
            )
    raise ValueError(
        f"Unknown FFT backend '{backend}'. Supported values: 'scipy', 'pyfftw'."
    )


# @profile(precision=1)
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
        grid_da = img_xds[grid_var_name]

        if grid_da.dims[-2:] != ("u", "v"):
            raise ValueError(
                f"{grid_var_name} must end in dimensions ('u', 'v'); "
                f"received {grid_da.dims}."
            )

        plane_dims = [
            dim for dim in grid_da.dims[:-2] if dim not in {"time", "polarization"}
        ]
        if len(plane_dims) != 1:
            raise ValueError(
                f"Could not identify one plane dimension in {grid_var_name}; "
                f"received {grid_da.dims}."
            )

        plane_dim = plane_dims[0]
        grid_da = grid_da.transpose("time", plane_dim, "polarization", "u", "v")
        raw_grid = grid_da.values

        normalization_name = data_group_in[normalization_key[data_variable_out]]
        normalization = (
            img_xds[normalization_name]
            .transpose("time", plane_dim, "polarization")
            .values.copy()
        )
        normalization[normalization == 0] = 1

        n_time, n_planes, n_pol = raw_grid.shape[:3]
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
        output_dims = (
            "time",
            plane_dim,
            "polarization",
            "l",
            "m",
        )
        output_shape = (
            n_time,
            n_planes,
            n_pol,
            image_size[0],
            image_size[1],
        )

        if out_name in img_xds:
            existing = img_xds[out_name]
            if existing.dims != output_dims or existing.shape != output_shape:
                if not overwrite:
                    raise ValueError(
                        f"Existing {out_name} has dimensions {existing.dims} "
                        f"and shape {existing.shape}; expected {output_dims} "
                        f"and {output_shape}."
                    )
                img_xds = img_xds.drop_vars(out_name)

        if out_name not in img_xds:
            output_coords = {
                dim: img_xds.coords[dim] for dim in output_dims if dim in img_xds.coords
            }
            img_xds[out_name] = xr.DataArray(
                np.empty(output_shape, dtype=float_out_dtype),
                dims=output_dims,
                coords=output_coords,
            )

        out_arr = img_xds[out_name].values
        img_xds[out_name].attrs["type"] = data_variable

        for t in range(n_time):
            for plane_index in range(n_planes):
                for p in range(n_pol):
                    plane = ifft_uv_to_lm(
                        np.asarray(
                            raw_grid[t, plane_index, p],
                            dtype=complex_dtype,
                        ),
                        processing_function_threads=processing_function_threads,
                        fft_backend=fft_backend,
                    )
                    plane /= kernel_image_1D_l[:, None]
                    plane /= kernel_image_1D_m[None, :]
                    plane *= flux_scale / normalization[t, plane_index, p]

                    out_arr[t, plane_index, p] = remove_padding(
                        plane.real,
                        image_size,
                    )

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
    image_data_group_out_modified={
        "visibility": "VISIBILITY_MODEL",
    },
    overwrite=True,
    image_data_variables_keep=[],
    processing_function_threads=1,
    fft_backend="pyfftw",
    data_variables_to_process=["sky"],
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
        image_size = np.asarray(_image_params["image_size"])

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

        # Process one 2-D plane at a time to keep FFT temporaries small.
        # At 12 000 × 12 000 this limits the extra allocation to ≈ 1.15 GB
        # instead of allocating the full (time, freq, pol, u, v) float64 array.
        for t in range(n_time):
            for f in range(n_freq):
                for p in range(n_pol):
                    add_padding(raw_grid[t, f, p], out_arr[t, f, p])

                    out_arr[t, f, p] = fft_lm_to_uv(
                        out_arr[t, f, p]
                        / kernel_image_1D_l[:, None]
                        / kernel_image_1D_m[None, :],
                        processing_function_threads=processing_function_threads,
                        fft_backend=fft_backend,
                        complex_dtype=complex_dtype,
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
        ``ifft_norm_img_xds`` this is a 2-D array of shape ``(u, v)``.
        dtype must be complex128.
    fft_plane_dims : tuple of int, optional
        Axes over which to apply the 2-D FFT.  Default is ``(-2, -1)``.
    threads : int, optional
        Number of threads passed to the FFT backend.  Default is ``1``.
    fft_backend : {"scipy", "pyfftw"}, optional
        FFT library to use.  Default is ``"pyfftw"``.

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
        work = grid_2d.copy()  # own the buffer; do not mutate the caller's grid
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
        if work is image:
            work = work.copy()  # own the buffer if asarray returned a view
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


def fft_norm_continuum_img_xds(
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
    """Forward-transform continuum Taylor-model images into Taylor UV grids.

    This is the MT-MFS equivalent of :func:`fft_norm_img_xds`. The input sky
    model must have dimensions

        (time, taylor_term, polarization, l, m)

    and the output visibility grid is created with dimensions

        (time, taylor_term, polarization, u, v).

    One FFT is performed for each Taylor term, rather than for every frequency
    channel.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Continuum image dataset containing the Taylor-term sky model.
    image_params : dict
        Imaging parameters. Must contain ``image_size`` and ``fft_padding``.
    image_data_group_in_name : str, optional
        Input image data group containing the sky model.
    image_data_group_out_name : str, optional
        Output image data group receiving the Taylor UV grid.
    image_data_group_out_modified : dict, optional
        Output data-group variable mapping. Defaults to
        ``{"visibility": "VISIBILITY_MODEL"}``.
    overwrite : bool, optional
        Replace an existing output variable when its dimensions or shape do not
        match the expected Taylor UV grid.
    image_data_variables_keep : list of str, optional
        Logical input variables to retain after the FFT.
    processing_function_threads : int, optional
        Number of FFT threads.
    fft_backend : {"scipy", "pyfftw"}, optional
        FFT backend.
    data_variables_to_process : list of str, optional
        Logical variables to transform. Defaults to ``["sky"]``.
    complex_dtype : numpy dtype, optional
        Complex precision of the UV grid.

    Returns
    -------
    xarray.Dataset
        Dataset containing the Taylor-term visibility grid.
    """
    if image_data_group_out_modified is None:
        image_data_group_out_modified = {
            "visibility": "VISIBILITY_MODEL",
        }

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    if data_variables_to_process is None:
        data_variables_to_process = ["sky"]

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
        [img_xds.sizes["l"], img_xds.sizes["m"]],
        image_params["fft_padding"],
    )

    for data_variable in data_variables_to_process:
        if data_variable not in data_group_in:
            continue

        image_var_name = data_group_in[data_variable]

        if image_var_name not in img_xds:
            raise KeyError(
                f"Input image variable {image_var_name!r} is not present " "in img_xds."
            )

        image_da = img_xds[image_var_name]

        required_dims = {
            "time",
            "taylor_term",
            "polarization",
            "l",
            "m",
        }

        missing_dims = required_dims.difference(image_da.dims)

        if missing_dims:
            raise ValueError(
                f"{image_var_name!r} is missing dimensions "
                f"{sorted(missing_dims)}. Received {image_da.dims}."
            )

        image_da = image_da.transpose(
            "time",
            "taylor_term",
            "polarization",
            "l",
            "m",
        )

        raw_image = image_da.values

        n_time = image_da.sizes["time"]
        n_terms = image_da.sizes["taylor_term"]
        n_pol = image_da.sizes["polarization"]

        output_dims = (
            "time",
            "taylor_term",
            "polarization",
            "u",
            "v",
        )

        output_shape = (
            n_time,
            n_terms,
            n_pol,
            n_uv[0],
            n_uv[1],
        )

        out_name = data_group_out[fft_pair[data_variable]]

        if out_name in img_xds:
            existing = img_xds[out_name]

            if existing.dims != output_dims or existing.shape != output_shape:
                if not overwrite:
                    raise ValueError(
                        f"Existing {out_name!r} has dimensions "
                        f"{existing.dims} and shape {existing.shape}; "
                        f"expected {output_dims} and {output_shape}."
                    )

                img_xds = img_xds.drop_vars(out_name)

        if out_name not in img_xds:
            output_coords = {}

            for dim in ("time", "taylor_term", "polarization"):
                if dim in image_da.coords:
                    output_coords[dim] = image_da.coords[dim]
                elif dim in img_xds.coords:
                    output_coords[dim] = img_xds.coords[dim]

            # Do not reuse any existing frequency coordinate. The output has
            # a Taylor axis, not a frequency axis.
            img_xds[out_name] = xr.DataArray(
                np.empty(
                    output_shape,
                    dtype=complex_dtype,
                ),
                dims=output_dims,
                coords=output_coords,
            )

        out_arr = img_xds[out_name].values
        img_xds[out_name].attrs["type"] = data_variable

        (
            kernel_image_1D_l,
            kernel_image_1D_m,
        ) = create_prolate_spheroidal_correcting_image_1D(n_lm_padded=n_uv)

        padded_plane = np.empty(
            (n_uv[0], n_uv[1]),
            dtype=complex_dtype,
        )

        for t in range(n_time):
            for term in range(n_terms):
                for p in range(n_pol):
                    add_padding(
                        raw_image[t, term, p],
                        padded_plane,
                    )

                    # Apply the same prolate-spheroidal correction as the cube
                    # FFT routine before transforming to the UV plane.
                    padded_plane /= kernel_image_1D_l[:, None]
                    padded_plane /= kernel_image_1D_m[None, :]

                    out_arr[t, term, p] = fft_lm_to_uv(
                        padded_plane,
                        processing_function_threads=(processing_function_threads),
                        fft_backend=fft_backend,
                        complex_dtype=complex_dtype,
                    )

        if data_variable not in image_data_variables_keep:
            img_xds.xr_img.delete_data_variables(variables=[image_var_name])
            del raw_image

        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            data_group_out,
            description=(
                "Transformed continuum Taylor model from lm plane "
                "to aperture uv plane."
            ),
        )

    return img_xds
