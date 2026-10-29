"""Exact direct Fourier transform imaging and prediction for single-field imaging.

The gridding kernel type ``"dft"`` (``gridding_kernel_params={"type": "dft"}``)
replaces the gridding, the FFT and the image plane correction of the residual
update and of the point spread function by exact sums over the visibilities
and the image pixels, made by the threaded C++ kernels of
:mod:`~astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp`.
The image grid is the image itself: there is no padding (``fft_padding`` is
not used), no FFT and no correction.

Convention. A sample of baseline coordinates ``u, v, w`` in wavelengths
(``UVW * frequency / c``) and a pixel of direction cosines ``l, m`` (the
image's ``l`` and ``m`` coordinates), with ``n = sqrt(1 - l**2 - m**2)``, have
the phase ``2 pi (u l + v m + w (n - 1))``, the phase of the measurement
equation of :func:`~astroviper.distributed_applications.simulation.simulate_processing_set.simulate_processing_set`
(and its C++ visibility kernel), and

* prediction: ``V = sum over the pixels of M exp(+i phase) / n``, with ``M``
  the model in Jy per pixel (apparent flux, the primary beam included), the
  simulator's sum over point sources of flux ``M`` at the pixels: the
  residual of the true model is zero to rounding;
* imaging: ``I = n sum over the samples of W V exp(-i phase) / sum of W``,
  with ``W`` the imaging weights: the image of a point source is its
  apparent flux at its pixel, and the point spread function (the image of
  unit visibilities) is 1 at the phase centre.

The w term is exact (no small field or w projection approximation); with
``w = 0`` and ``n = 1`` these are the two dimensional transforms the gridder
and the degridder approximate, with the same signs. The two dimensional
gridder applies neither ``n`` nor ``1 / n``: even without a w term its image
of a point source differs from this one by about ``(1 / n - 1)`` times the
flux besides the kernel's error, so a comparison of the two resolves the
kernel's error only where ``1 - n`` is far below it.

The samples, the weights, the correlations, the frequency mapping and the
output variables are those of the gridder path; unlike the gridder, no
sample is dropped for lying beyond the uv grid, as there is none, and a
sample whose ``w`` alone is not finite is dropped. The normalization
(``VISIBILITY_NORMALIZATION``, ``UV_SAMPLING_NORMALIZATION``) is the sum of
the imaging weights of the samples imaged, per image plane, where the
gridder's is that sum times the kernel's stencil sums. No uv grid is made,
so the image data groups get no ``visibility`` or ``uv_sampling`` role.

Continuum (MT-MFS) imaging uses the same transforms for every Taylor term,
with ``x = (nu - nu_ref) / nu_ref`` the channel's offset from the reference
frequency: Taylor term ``t`` of the residual is the image of the visibilities
with the weights ``W x**t``, order ``k`` of the point spread function the
image of the weights ``W x**k``, every plane normalised by the sum of ``W``
(order zero) once the map tasks' sums are added, and the model visibilities
are ``sum over t of x**t`` times the prediction of Taylor term ``t`` of the
model, the equations of the gridder path with exact transforms.
"""

import time

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from astroviper.utils.param_docs import shares_param_docs

_SKY_ROLES = {
    "sky": "SKY_RESIDUAL",
    "visibility_normalization": "VISIBILITY_NORMALIZATION",
}
_UV_SAMPLING_ROLES = {
    "point_spread_function": "POINT_SPREAD_FUNCTION",
    "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
}


def image_direction_cosines(img_xds):
    """The direction cosines of the image rows and columns.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset with ``l`` and ``m`` coordinates (radians of the SIN
        projection, i.e. direction cosines).

    Returns
    -------
    l, m : numpy.ndarray
        C-contiguous float64 arrays of ``img_xds.sizes["l"]`` and
        ``img_xds.sizes["m"]`` values.
    """
    return (
        np.ascontiguousarray(img_xds.l.values, dtype=np.float64),
        np.ascontiguousarray(img_xds.m.values, dtype=np.float64),
    )


def _frequency_map(ms_xds, img_xds, chan_mode):
    """The image channel of every visibility channel."""
    if chan_mode == "cube":
        from astroviper.processing_functions.imaging.utils.frequency_mapping import (
            map_visibility_frequencies_to_image,
        )

        frequency_map = map_visibility_frequencies_to_image(
            ms_xds.frequency.values, img_xds.frequency.values
        )
    else:  # continuum: every channel onto image channel 0
        frequency_map = np.zeros(ms_xds.sizes["frequency"], dtype=np.int64)
    return np.ascontiguousarray(frequency_map, dtype=np.int64)


def _maps(ms_xds, img_xds, chan_mode):
    """Frequency, time and polarization maps of a measurement set onto the image."""
    n_pol = ms_xds.sizes["polarization"]
    if img_xds.sizes["polarization"] != n_pol:
        raise ValueError(
            "the image polarization axis must match the measurement set: "
            f"{img_xds.sizes['polarization']} image polarizations for {n_pol} "
            "visibility polarizations."
        )
    # Time mapping is not implemented: every visibility time maps onto plane 0.
    time_map = np.zeros(ms_xds.sizes["time"], dtype=np.int64)
    pol_map = np.arange(n_pol, dtype=np.int64)
    return _frequency_map(ms_xds, img_xds, chan_mode), time_map, pol_map


def _model_visibilities(
    ms_xds, ms_data_group_in, ms_data_group_out, ms_data_group_out_name, description
):
    """The output model visibilities of a prediction, made (zero) when absent."""
    # The model visibilities are double precision, as the degridder's.
    output_name = ms_data_group_out["correlated_data"]
    if output_name not in ms_xds:
        template_name = ms_data_group_in["correlated_data"]
        if template_name not in ms_xds:
            template_name = ms_data_group_in.get("weight_imaging")
        template = ms_xds[template_name]
        ms_xds[output_name] = xr.DataArray(
            np.zeros(template.shape, dtype=np.complex128), dims=template.dims
        )
        modify_data_groups_xds(
            ms_xds,
            ms_data_group_out_name,
            ms_data_group_out,
            description=description,
        )
    return ms_xds[output_name]


def add_dft_image_single_field(
    ms_xds,
    img_xds,
    ms_data_group_in_name="base",
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    image_data_group_out_modified=None,
    uv_sampling=False,
    overwrite=True,
    chan_mode="cube",
    processing_function_threads=1,
    complex_dtype=None,
):
    """Accumulate the direct Fourier transform of one measurement set into an image.

    Adds ``n sum W V exp(-i phase)`` (``uv_sampling=False``, the visibilities)
    or ``n sum W cos(phase)`` (``uv_sampling=True``, the uv sampling function
    that becomes the point spread function) of the samples of ``ms_xds`` into
    the image of the output data group, and the sum of their imaging weights
    ``W`` into its normalization (see the module docstring for the
    convention). The image and its normalization are created (zero) when the
    image variable is absent and accumulated otherwise, so that repeated calls
    sum over measurement sets; :func:`normalize_dft_image` divides by the
    normalization afterwards. This is the direct Fourier transform counterpart
    of
    :func:`~astroviper.processing_functions.imaging.add_visibility_grid.add_visibility_grid_single_field`
    and
    :func:`~astroviper.processing_functions.imaging.make_point_spread_function.add_uv_sampling_grid_single_field`.

    A sample is imaged when its uvw is finite and, for the visibilities, its
    weighted visibility ``W V`` is neither NaN nor zero, for the uv sampling
    its weight neither NaN nor zero (flagged samples carry NaN imaging
    weights): the samples the gridder grids, except that the gridder, which
    checks ``u`` and ``v`` only, also grids a sample whose ``w`` alone is not
    finite.

    Parameters
    ----------
    ms_xds : xarray.Dataset
        Measurement set. Must provide the ``correlated_data`` (not read for
        the uv sampling), ``uvw`` and ``weight_imaging`` roles of
        ``ms_data_group_in_name`` and a ``frequency`` coordinate. Not
        modified.
    img_xds : xarray.Dataset
        Image dataset in the polarization basis of the measurement set (one
        image plane per correlation). Modified in place.
    ms_data_group_in_name : str, optional
        Measurement-set data group to image. Default ``"base"``.
    image_data_group_in_name : str, optional
        Existing image data group the output group is formed from. Default
        ``"residual"``.
    image_data_group_out_name : str, optional
        Image data group the image and its normalization are registered
        under. Default ``"residual"``.
    image_data_group_out_modified : dict, optional
        Role to variable names of the outputs. Default
        ``{"sky": "SKY_RESIDUAL", "visibility_normalization":
        "VISIBILITY_NORMALIZATION"}``, or with ``uv_sampling``
        ``{"point_spread_function": "POINT_SPREAD_FUNCTION",
        "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION"}``.
    uv_sampling : bool, optional
        Transform the imaging weights (the point spread function) instead of
        the weighted visibilities. Default ``False``.
    overwrite : bool, optional
        Passed to the data-group helper. Default ``True``.
    chan_mode : str, optional
        ``"cube"`` maps each visibility channel onto the image channel
        nearest in frequency, ``"continuum"`` every channel onto image
        channel 0. Default ``"cube"``.
    processing_function_threads : int, optional
        Threads of the C++ kernel (``<= 0``: the hardware concurrency). The
        result does not depend on it.
    complex_dtype : numpy.dtype, optional
        Precision of a new image: ``complex64`` gives a float32 image (a
        complex64 sky in a four-correlation basis), ``complex128`` (the
        default) a float64 one. The sums are made in double precision.

    Returns
    -------
    None
        ``img_xds`` is modified in place.

    Raises
    ------
    ValueError
        If the image and the measurement set have different numbers of
        polarizations, or (from the kernel) if an image pixel lies beyond
        ``l**2 + m**2 < 1``.
    """
    from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
        dft_image,
        dft_image_uv_sampling,
    )
    from astroviper.processing_functions.imaging.utils.imaging_polarization import (
        is_four_correlation_basis,
    )

    if image_data_group_out_modified is None:
        image_data_group_out_modified = dict(
            _UV_SAMPLING_ROLES if uv_sampling else _SKY_ROLES
        )
    image_role, normalization_role = (
        ("point_spread_function", "uv_sampling_normalization")
        if uv_sampling
        else ("sky", "visibility_normalization")
    )
    if complex_dtype is None:
        complex_dtype = np.complex128

    ms_data_group_in = ms_xds.attrs["data_groups"][ms_data_group_in_name]
    _, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=dict(image_data_group_out_modified),
        overwrite=overwrite,
    )
    frequency_map, time_map, pol_map = _maps(ms_xds, img_xds, chan_mode)

    image_name = image_data_group_out[image_role]
    normalization_name = image_data_group_out[normalization_role]
    if image_name not in img_xds:
        # The sky in a four-correlation basis is complex (a cross hand is not
        # conjugate symmetric on its own), as the gridder path makes it.
        float_dtype = (
            np.float32 if np.dtype(complex_dtype) == np.complex64 else np.float64
        )
        image_dtype = (
            np.dtype(complex_dtype)
            if not uv_sampling
            and is_four_correlation_basis(img_xds.polarization.values)
            else float_dtype
        )
        image_dims = ("time", "frequency", "polarization", "l", "m")
        img_xds[image_name] = xr.DataArray(
            np.zeros(
                tuple(img_xds.sizes[dim] for dim in image_dims), dtype=image_dtype
            ),
            dims=image_dims,
            coords={
                dim: img_xds.coords[dim] for dim in image_dims if dim in img_xds.coords
            },
        )
        img_xds[image_name].attrs["type"] = image_role
        normalization_dims = ("time", "frequency", "polarization")
        img_xds[normalization_name] = xr.DataArray(
            np.zeros(tuple(img_xds.sizes[dim] for dim in normalization_dims)),
            dims=normalization_dims,
        )
        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            image_data_group_out,
            description="Added the direct Fourier transform of the "
            + ("imaging weights" if uv_sampling else "visibilities")
            + " to img_xds with add_dft_image_single_field.",
        )

    image = img_xds[image_name].values
    normalization = img_xds[normalization_name].values
    uvw = np.ascontiguousarray(ms_xds[ms_data_group_in["uvw"]].values, dtype=np.float64)
    weight = np.ascontiguousarray(
        ms_xds[ms_data_group_in["weight_imaging"]].values, dtype=np.float64
    )
    frequency_coord = np.ascontiguousarray(ms_xds.frequency.values, dtype=np.float64)
    l, m = image_direction_cosines(img_xds)
    if uv_sampling:
        dft_image_uv_sampling(
            image,
            normalization,
            uvw,
            frequency_coord,
            frequency_map,
            time_map,
            pol_map,
            weight,
            l,
            m,
            processing_function_threads=processing_function_threads,
        )
    else:
        vis_data = np.ascontiguousarray(
            ms_xds[ms_data_group_in["correlated_data"]].values, dtype=np.complex128
        )
        dft_image(
            image,
            normalization,
            vis_data,
            uvw,
            frequency_coord,
            frequency_map,
            time_map,
            pol_map,
            weight,
            l,
            m,
            processing_function_threads=processing_function_threads,
        )


def normalize_dft_image(
    img_xds,
    image_data_group_name="residual",
    uv_sampling=False,
):
    """Divide an accumulated direct Fourier transform image by its normalization.

    Every image plane is divided by its sum of imaging weights (a plane
    without weights by 1), in place, as
    :func:`~astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder.ifft_norm_img_xds`
    normalises the gridded image.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset holding the image and normalization of
        :func:`add_dft_image_single_field` or
        :func:`add_dft_image_continuum_single_field`. Modified in place (a
        read-only image is replaced by its normalised copy).
    image_data_group_name : str, optional
        The image data group of the image. Default ``"residual"``.
    uv_sampling : bool, optional
        ``True`` normalises the point spread function
        (``point_spread_function`` and ``uv_sampling_normalization`` roles),
        ``False`` the sky (``sky`` and ``visibility_normalization``). Default
        ``False``.

    Returns
    -------
    None
    """
    image_role, normalization_role = (
        ("point_spread_function", "uv_sampling_normalization")
        if uv_sampling
        else ("sky", "visibility_normalization")
    )
    data_group = img_xds.attrs["data_groups"][image_data_group_name]
    image_name = data_group[image_role]
    image = img_xds[image_name].values
    if not image.flags.writeable:
        # e.g. the sum of continuum map tasks received from another process
        image = image.copy()
        img_xds[image_name] = img_xds[image_name].copy(data=image)
    normalization = img_xds[data_group[normalization_role]].values.copy()
    normalization[normalization == 0] = 1
    for index in np.ndindex(normalization.shape):
        plane = image[index]
        np.divide(plane, normalization[index], out=plane)


@shares_param_docs
def make_dft_undeconvolved_image_single_field(
    ps_xdt,
    img_xds,
    ms_data_group_in_name="corrected",
    image_data_group_out_name="residual",
    processing_function_threads=1,
    complex_dtype=None,
):
    """Image the (residual) visibilities of every measurement set by the direct Fourier transform.

    The direct Fourier transform counterpart of gridding
    (:func:`~astroviper.processing_functions.imaging.make_undeconvolved_image.make_undeconvolved_image_single_field`)
    and the inverse FFT
    (:func:`~astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder.ifft_norm_img_xds`):
    drops the auto-correlations, sums the transforms of all measurement sets
    (:func:`add_dft_image_single_field`) into a new ``SKY_RESIDUAL`` and its
    ``VISIBILITY_NORMALIZATION`` and divides by the latter
    (:func:`normalize_dft_image`). The image is in the polarization basis of
    the visibilities (complex in a four-correlation basis).

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set; auto-correlations are dropped in place.
    img_xds : xarray.Dataset
        Image dataset the residual image is written into (an existing
        ``SKY_RESIDUAL`` is replaced). Modified in place.
    ms_data_group_in_name : str, optional
        Measurement-set data group to image. Default ``"corrected"``.
    image_data_group_out_name : str, optional
        Image data group the image is registered under. Default
        ``"residual"``.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    complex_dtype : numpy.dtype, optional
        ``complex64`` for a single-precision image, ``complex128`` (the
        default) otherwise.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with the residual image.
    return_df : pandas.DataFrame
        One-row timing frame with ``T_vis_mask``, ``T_uv_sampling_grid``
        (zero) and ``T_vis_grid`` (the transform and its normalization), the
        columns of the gridder path.
    """
    import pandas as pd

    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    data_group = img_xds.attrs["data_groups"].get(image_data_group_out_name, {})
    image_name = data_group.get("sky", _SKY_ROLES["sky"])
    if image_name in img_xds:
        img_xds.xr_img.delete_data_variables(variables=[image_name])

    T_vis_mask = 0.0
    T_vis_grid = 0.0
    for ms_xdt in ps_xdt.values():
        start = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - start

        start = time.time()
        add_dft_image_single_field(
            ms_xdt,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            uv_sampling=False,
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
        )
        T_vis_grid += time.time() - start

    start = time.time()
    normalize_dft_image(img_xds, image_data_group_out_name, uv_sampling=False)
    T_vis_grid += time.time() - start

    return_df = pd.DataFrame(
        {
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [0.0],
            "T_vis_grid": [T_vis_grid],
        }
    )
    return img_xds, return_df


@shares_param_docs
def make_dft_point_spread_function_single_field(
    ps_xdt,
    img_xds,
    ms_data_group_in_name="base",
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    processing_function_threads=1,
    complex_dtype=None,
):
    """Build the point spread function by the direct Fourier transform of the imaging weights.

    The direct Fourier transform counterpart of
    :func:`~astroviper.processing_functions.imaging.make_point_spread_function.make_point_spread_function_single_field`:
    drops the auto-correlations, sums the transforms of the imaging weights of
    all measurement sets (:func:`add_dft_image_single_field` with
    ``uv_sampling=True``) into a new ``POINT_SPREAD_FUNCTION`` and its
    ``UV_SAMPLING_NORMALIZATION`` and divides by the latter. The point spread
    function is the image of unit visibilities: 1 at the phase centre, real,
    one plane per correlation, with the w term of the samples.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set; auto-correlations are dropped in place.
    img_xds : xarray.Dataset
        Image dataset to populate (an existing ``POINT_SPREAD_FUNCTION`` is
        replaced). The data group ``image_data_group_in_name`` must exist.
        Modified in place.
    ms_data_group_in_name : str, optional
        Measurement-set data group that supplies the weights and uvw.
        Default ``"base"``.
    image_data_group_in_name : str, optional
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    image_data_group_out_name : str, optional
        Image data group the point spread function is registered under.
        Default ``"residual"``.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    complex_dtype : numpy.dtype, optional
        ``complex64`` gives a float32 point spread function, ``complex128``
        (the default) a float64 one.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with ``POINT_SPREAD_FUNCTION`` added.
    return_df : pandas.DataFrame
        One-row timing frame with the ``T_gcf`` (zero), ``T_vis_mask``,
        ``T_uv_sampling_grid`` (the transform) and ``T_fft_norm`` (the
        normalization) columns of the gridder path.
    """
    import pandas as pd

    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    data_group = img_xds.attrs["data_groups"].get(image_data_group_out_name, {})
    image_name = data_group.get(
        "point_spread_function", _UV_SAMPLING_ROLES["point_spread_function"]
    )
    if image_name in img_xds:
        img_xds.xr_img.delete_data_variables(variables=[image_name])

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0
    for ms_xdt in ps_xdt.values():
        start = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - start

        start = time.time()
        add_dft_image_single_field(
            ms_xdt,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_in_name,
            image_data_group_out_name=image_data_group_out_name,
            uv_sampling=True,
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
        )
        T_uv_sampling_grid += time.time() - start

    start = time.time()
    normalize_dft_image(img_xds, image_data_group_out_name, uv_sampling=True)
    T_fft_norm = time.time() - start

    return_df = pd.DataFrame(
        {
            "T_gcf": [0.0],
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [T_uv_sampling_grid],
            "T_fft_norm": [T_fft_norm],
        }
    )
    return img_xds, return_df


def predict_dft_visibilities_single_field(
    ms_xds,
    img_xds,
    ms_data_group_in_name="base",
    ms_data_group_out_name="model",
    ms_data_group_out_modified=None,
    image_data_group_in_name="model",
    overwrite=True,
    chan_mode="cube",
    processing_function_threads=1,
):
    """Predict the model visibilities of one measurement set by the direct Fourier transform.

    Writes ``V = sum over the non-zero model pixels of M exp(+i phase) / n``
    (see the module docstring) into the output visibilities, for every sample
    whose uvw is finite and whose output value is not NaN (others are left
    unchanged; a new output array is zero), as the degridder does
    (:func:`~astroviper.processing_functions.imaging.degrid_visibility_grid.degrid_visibility_grid_single_field`),
    which checks ``u`` and ``v`` only and so also predicts a sample whose
    ``w`` alone is not finite.
    The model image is read in the polarization basis of the visibilities;
    no FFT, padding or correction is involved.

    Parameters
    ----------
    ms_xds : xarray.Dataset
        Measurement set, modified in place. Must provide the
        ``correlated_data`` (or ``weight_imaging``, for the shape of a new
        output) and ``uvw`` roles of ``ms_data_group_in_name`` and a
        ``frequency`` coordinate.
    img_xds : xarray.Dataset
        Image dataset whose ``image_data_group_in_name`` group's ``sky`` role
        names the model, dims ``(time, frequency, polarization, l, m)``, in
        Jy per pixel.
    ms_data_group_in_name : str, optional
        Measurement-set data group being imaged. Default ``"base"``.
    ms_data_group_out_name : str, optional
        Measurement-set data group of the model visibilities. Default
        ``"model"``.
    ms_data_group_out_modified : dict, optional
        Default ``{"correlated_data": "VISIBILITY_MODEL"}``.
    image_data_group_in_name : str, optional
        Image data group of the model. Default ``"model"``.
    overwrite : bool, optional
        Whether an existing output data group may be replaced. Default
        ``True``.
    chan_mode : str, optional
        ``"cube"`` (each visibility channel from the image channel nearest in
        frequency) or ``"continuum"`` (every channel from image channel 0).
    processing_function_threads : int, optional
        Threads of the C++ kernel (``<= 0``: the hardware concurrency). The
        result does not depend on it.

    Returns
    -------
    None
        ``ms_xds`` is modified in place.

    Raises
    ------
    ValueError
        If the image and the measurement set have different numbers of
        polarizations.
    """
    from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
        dft_predict,
    )

    if ms_data_group_out_modified is None:
        ms_data_group_out_modified = {"correlated_data": "VISIBILITY_MODEL"}
    ms_data_group_in, ms_data_group_out = create_data_groups_in_and_out(
        ms_xds,
        data_group_in_name=ms_data_group_in_name,
        data_group_out_name=ms_data_group_out_name,
        data_group_out_modified=dict(ms_data_group_out_modified),
        overwrite=overwrite,
    )
    frequency_map, time_map, pol_map = _maps(ms_xds, img_xds, chan_mode)
    model_visibilities = _model_visibilities(
        ms_xds,
        ms_data_group_in,
        ms_data_group_out,
        ms_data_group_out_name,
        description="Predicted visibilities from img_xds "
        + image_data_group_in_name
        + " with predict_dft_visibilities_single_field.",
    )

    model_name = img_xds.attrs["data_groups"][image_data_group_in_name]["sky"]
    model = np.ascontiguousarray(
        img_xds[model_name]
        .transpose("time", "frequency", "polarization", "l", "m")
        .values
    )
    l, m = image_direction_cosines(img_xds)
    dft_predict(
        model,
        model_visibilities.values,
        np.ascontiguousarray(ms_xds[ms_data_group_in["uvw"]].values, dtype=np.float64),
        np.ascontiguousarray(ms_xds.frequency.values, dtype=np.float64),
        frequency_map,
        time_map,
        pol_map,
        l,
        m,
        processing_function_threads=processing_function_threads,
    )


def make_dft_visibility_model_single_field(
    ps_xdt,
    img_xds,
    ms_data_group_in_name="base",
    ms_data_group_out_name="model",
    ms_data_group_out_modified=None,
    image_data_group_in_name="model",
    processing_function_threads=1,
):
    """Predict the model visibilities of every measurement set by the direct Fourier transform.

    Wraps :func:`predict_dft_visibilities_single_field` over the measurement
    sets of the processing set, as
    :func:`~astroviper.processing_functions.imaging.residual_update.make_visibility_model_single_field`
    wraps the degridder.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set; model visibilities are written into each measurement
        set in place.
    img_xds : xarray.Dataset
        Image dataset holding the model image (in the polarization basis of
        the visibilities).
    ms_data_group_in_name : str, optional
        Measurement-set data group being imaged. Default ``"base"``.
    ms_data_group_out_name : str, optional
        Measurement-set data group for the model visibilities. Default
        ``"model"``.
    ms_data_group_out_modified : dict, optional
        Default ``{"correlated_data": "VISIBILITY_MODEL"}``.
    image_data_group_in_name : str, optional
        Image data group holding the model image. Default ``"model"``.
    processing_function_threads : int, optional
        Threads handed to the C++ kernel. Default ``1``.

    Returns
    -------
    None
    """
    for ms_xdt in ps_xdt.values():
        predict_dft_visibilities_single_field(
            ms_xdt,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            ms_data_group_out_name=ms_data_group_out_name,
            ms_data_group_out_modified=ms_data_group_out_modified,
            image_data_group_in_name=image_data_group_in_name,
            overwrite=True,
            chan_mode="cube",
            processing_function_threads=processing_function_threads,
        )


###############################################################################
# Continuum (MT-MFS) imaging
###############################################################################


def check_dft_continuum_parameters(
    continuum_mode="mfs",
    visibility_memory_mode="recompute",
    image_data_variables_keep=(),
):
    """Refuse the continuum options the direct Fourier transform does not take.

    Continuum imaging takes ``gridding_kernel_params={"type": "dft"}`` in MFS
    with ``visibility_memory_mode="recompute"``: MVC and the cached observed
    uv grids of ``"in_memory"`` and ``"in_place"`` are made of uv grids, which
    the direct Fourier transform does not make, and so are the ``visibility``
    and ``uv_sampling`` variables that ``image_data_variables_keep`` could
    keep.

    Parameters
    ----------
    continuum_mode : str, optional
        ``"mfs"`` or ``"mvc"``. Default ``"mfs"``.
    visibility_memory_mode : str, optional
        ``"recompute"``, ``"in_memory"`` or ``"in_place"``. Default
        ``"recompute"``.
    image_data_variables_keep : sequence of str, optional
        Logical image-variable keys to keep. Default none.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        For ``continuum_mode="mvc"``, a ``visibility_memory_mode`` other than
        ``"recompute"`` or a kept ``visibility`` or ``uv_sampling`` grid.
    """
    refusal = (
        'with the gridding_kernel_params type "dft" (the direct Fourier '
        "transform), which makes no uv grids: "
    )
    if str(continuum_mode).lower() != "mfs":
        raise ValueError(
            f"continuum_mode {continuum_mode!r} is not available "
            + refusal
            + 'MVC forms and caches its channels as uv grids; use continuum_mode="mfs"'
        )
    if visibility_memory_mode != "recompute":
        raise ValueError(
            f"visibility_memory_mode {visibility_memory_mode!r} is not available "
            + refusal
            + 'it caches the observed uv grid; use visibility_memory_mode="recompute"'
        )
    kept_grids = sorted({"visibility", "uv_sampling"} & set(image_data_variables_keep))
    if kept_grids:
        raise ValueError(
            f"image_data_variables_keep {kept_grids} are uv grids, which the "
            'gridding_kernel_params type "dft" (the direct Fourier transform) '
            "does not make"
        )


def add_dft_image_continuum_single_field(
    ms_xds,
    img_xds,
    n_taylor_terms,
    reference_frequency,
    ms_data_group_in_name="base",
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    image_data_group_out_modified=None,
    uv_sampling=False,
    overwrite=True,
    processing_function_threads=1,
    complex_dtype=None,
):
    """Accumulate the Taylor-weighted direct Fourier transforms of one measurement set.

    The continuum (MT-MFS) form of :func:`add_dft_image_single_field` and the
    direct Fourier transform counterpart of
    :func:`~astroviper.processing_functions.imaging.add_visibility_grid_continuum.add_visibility_grid_continuum_single_field`
    (``uv_sampling=False``) and
    :func:`~astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field.add_uv_sampling_grid_continuum_single_field`
    (``uv_sampling=True``): plane ``t`` of the output receives the transform
    of all channels with the imaging weights ``W x**t``, where
    ``x = (frequency - reference_frequency) / reference_frequency``, and every
    plane of the normalization the sum of ``W`` (order zero), as the gridder
    path normalises every Taylor term by the zeroth-order sum of weights.
    The outputs are

    ``SKY_RESIDUAL(time, taylor_term, polarization, l, m)`` and
    ``VISIBILITY_NORMALIZATION(time, taylor_term, polarization)``, or with
    ``uv_sampling``
    ``POINT_SPREAD_FUNCTION(time, psf_taylor_order, polarization, l, m)`` and
    ``UV_SAMPLING_NORMALIZATION(time, psf_taylor_order, polarization)``,

    created (zero) when the image variable is absent and accumulated
    otherwise, so that repeated calls sum over measurement sets, and the
    unnormalised sums can be added over map tasks before
    :func:`normalize_dft_image` divides them by their normalization. The
    convention of the transforms is that of the module docstring; the samples
    imaged are those of :func:`add_dft_image_single_field`.

    Parameters
    ----------
    ms_xds : xarray.Dataset
        Measurement set. Must provide the ``correlated_data`` (not read for
        the uv sampling), ``uvw`` and ``weight_imaging`` roles of
        ``ms_data_group_in_name`` and a ``frequency`` coordinate. Not
        modified.
    img_xds : xarray.Dataset
        Image dataset in the polarization basis of the measurement set, with
        one image time. Modified in place.
    n_taylor_terms : int
        Number of output planes: the Taylor terms of the residual or, with
        ``uv_sampling``, the ``2 * n_taylor_terms - 1`` orders of the point
        spread function.
    reference_frequency : float
        Taylor-expansion reference frequency in Hz.
    ms_data_group_in_name : str, optional
        Measurement-set data group to image. Default ``"base"``.
    image_data_group_in_name : str, optional
        Existing image data group the output group is formed from. Default
        ``"residual"``.
    image_data_group_out_name : str, optional
        Image data group the image and its normalization are registered
        under. Default ``"residual"``.
    image_data_group_out_modified : dict, optional
        Role to variable names of the outputs. Default those of
        :func:`add_dft_image_single_field`.
    uv_sampling : bool, optional
        Transform the imaging weights (the Taylor point spread functions)
        instead of the weighted visibilities. Default ``False``.
    overwrite : bool, optional
        Passed to the data-group helper. Default ``True``.
    processing_function_threads : int, optional
        Threads of the C++ kernel (``<= 0``: the hardware concurrency). The
        result does not depend on it.
    complex_dtype : numpy.dtype, optional
        Precision of a new image: ``complex64`` gives a float32 image (a
        complex64 sky in a four-correlation basis), ``complex128`` (the
        default) a float64 one. The sums are made in double precision.

    Returns
    -------
    None
        ``img_xds`` is modified in place.

    Raises
    ------
    ValueError
        For fewer than one plane, a reference frequency that is not finite
        and positive, an existing output of another shape, or as
        :func:`add_dft_image_single_field`.
    """
    from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
        dft_image,
        dft_image_uv_sampling,
    )
    from astroviper.processing_functions.imaging.utils.imaging_polarization import (
        is_four_correlation_basis,
    )

    n_taylor_terms = int(n_taylor_terms)
    reference_frequency = float(reference_frequency)
    if n_taylor_terms < 1:
        raise ValueError("n_taylor_terms must be at least 1.")
    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "reference_frequency must be a positive finite frequency in Hz."
        )
    if image_data_group_out_modified is None:
        image_data_group_out_modified = dict(
            _UV_SAMPLING_ROLES if uv_sampling else _SKY_ROLES
        )
    image_role, normalization_role, plane_dim = (
        ("point_spread_function", "uv_sampling_normalization", "psf_taylor_order")
        if uv_sampling
        else ("sky", "visibility_normalization", "taylor_term")
    )
    if complex_dtype is None:
        complex_dtype = np.complex128

    ms_data_group_in = ms_xds.attrs["data_groups"][ms_data_group_in_name]
    _, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=dict(image_data_group_out_modified),
        overwrite=overwrite,
    )
    frequency_map, time_map, pol_map = _maps(ms_xds, img_xds, "continuum")

    # The Taylor coordinates of the gridder path: int32 Taylor terms and
    # int64 orders of the point spread function.
    if plane_dim not in img_xds.coords:
        img_xds.coords[plane_dim] = np.arange(
            n_taylor_terms, dtype=np.int64 if uv_sampling else np.int32
        )
    elif img_xds.sizes[plane_dim] != n_taylor_terms:
        raise ValueError(
            f"Existing {plane_dim} coordinate has length "
            f"{img_xds.sizes[plane_dim]}, but {n_taylor_terms} planes were requested."
        )
    image_dims = ("time", plane_dim, "polarization", "l", "m")
    normalization_dims = image_dims[:3]
    image_shape = tuple(
        n_taylor_terms if dim == plane_dim else img_xds.sizes[dim] for dim in image_dims
    )

    image_name = image_data_group_out[image_role]
    normalization_name = image_data_group_out[normalization_role]
    if image_name not in img_xds:
        # The sky in a four-correlation basis is complex (a cross hand is not
        # conjugate symmetric on its own), as the gridder path makes it.
        float_dtype = (
            np.float32 if np.dtype(complex_dtype) == np.complex64 else np.float64
        )
        image_dtype = (
            np.dtype(complex_dtype)
            if not uv_sampling
            and is_four_correlation_basis(img_xds.polarization.values)
            else float_dtype
        )
        img_xds[image_name] = xr.DataArray(
            np.zeros(image_shape, dtype=image_dtype),
            dims=image_dims,
            coords={
                dim: img_xds.coords[dim] for dim in image_dims if dim in img_xds.coords
            },
        )
        img_xds[image_name].attrs["type"] = image_role
        img_xds[normalization_name] = xr.DataArray(
            np.zeros(image_shape[:3]),
            dims=normalization_dims,
            coords={
                dim: img_xds.coords[dim]
                for dim in normalization_dims
                if dim in img_xds.coords
            },
        )
        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            image_data_group_out,
            description="Added the Taylor-weighted direct Fourier transform of the "
            + ("imaging weights" if uv_sampling else "visibilities")
            + " to img_xds with add_dft_image_continuum_single_field.",
        )
    elif (
        img_xds[image_name].dims != image_dims
        or img_xds[image_name].shape != image_shape
        or img_xds[normalization_name].shape != image_shape[:3]
    ):
        raise ValueError(
            f"Existing {image_name} has dimensions {img_xds[image_name].dims} and "
            f"shape {img_xds[image_name].shape}; expected {image_dims} and "
            f"{image_shape}."
        )

    image = img_xds[image_name].values
    normalization = img_xds[normalization_name].values
    uvw = np.ascontiguousarray(ms_xds[ms_data_group_in["uvw"]].values, dtype=np.float64)
    weight = np.ascontiguousarray(
        ms_xds[ms_data_group_in["weight_imaging"]].values, dtype=np.float64
    )
    frequency_coord = np.ascontiguousarray(ms_xds.frequency.values, dtype=np.float64)
    l, m = image_direction_cosines(img_xds)
    if not uv_sampling:
        vis_data = np.ascontiguousarray(
            ms_xds[ms_data_group_in["correlated_data"]].values, dtype=np.complex128
        )
    x = (frequency_coord - reference_frequency) / reference_frequency

    # With one image time, the plane of every Taylor term is a C-contiguous
    # view that the kernel accumulates into in place.
    zeroth_normalization = np.zeros(
        (normalization.shape[0], 1, normalization.shape[2]), dtype=np.float64
    )
    for taylor_term in range(n_taylor_terms):
        taylor_weight = (
            weight
            if taylor_term == 0
            else np.ascontiguousarray(weight * (x**taylor_term)[None, None, :, None])
        )
        plane_normalization = np.zeros_like(zeroth_normalization)
        if uv_sampling:
            dft_image_uv_sampling(
                image[:, taylor_term : taylor_term + 1],
                plane_normalization,
                uvw,
                frequency_coord,
                frequency_map,
                time_map,
                pol_map,
                taylor_weight,
                l,
                m,
                processing_function_threads=processing_function_threads,
            )
        else:
            dft_image(
                image[:, taylor_term : taylor_term + 1],
                plane_normalization,
                vis_data,
                uvw,
                frequency_coord,
                frequency_map,
                time_map,
                pol_map,
                taylor_weight,
                l,
                m,
                processing_function_threads=processing_function_threads,
            )
        # Only the ordinary sum of the imaging weights normalises.
        if taylor_term == 0:
            zeroth_normalization += plane_normalization
    normalization += zeroth_normalization


@shares_param_docs
def make_dft_undeconvolved_image_continuum_single_field(
    ps_xdt,
    img_xds,
    n_taylor_terms,
    reference_frequency,
    ms_data_group_in_name="corrected",
    image_data_group_out_name="residual",
    processing_function_threads=1,
    complex_dtype=None,
):
    """Image the (residual) visibilities of every measurement set into Taylor terms by the direct Fourier transform.

    The direct Fourier transform counterpart of
    :func:`~astroviper.processing_functions.imaging.make_undeconvolved_image_continuum.make_undeconvolved_image_continuum_single_field`:
    drops the auto-correlations and sums the Taylor-weighted transforms of
    all measurement sets (:func:`add_dft_image_continuum_single_field`) into a
    new ``SKY_RESIDUAL(time, taylor_term, polarization, l, m)`` and its
    ``VISIBILITY_NORMALIZATION`` (an existing ``SKY_RESIDUAL`` is replaced).
    The sums are not normalised: as the gridded Taylor uv grids, they are
    added over the map tasks first and divided by their normalization
    afterwards (:func:`normalize_dft_image`).

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set; auto-correlations are dropped in place.
    img_xds : xarray.Dataset
        Continuum image dataset in the polarization basis of the
        visibilities. Modified in place.
    n_taylor_terms : int
        Number of residual Taylor terms.
    reference_frequency : float
        Taylor-expansion reference frequency in Hz.
    ms_data_group_in_name : str, optional
        Measurement-set data group to image. Default ``"corrected"``.
    image_data_group_out_name : str, optional
        Image data group the image is registered under. Default
        ``"residual"``.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    complex_dtype : numpy.dtype, optional
        ``complex64`` for a single-precision image, ``complex128`` (the
        default) otherwise.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with the unnormalised residual Taylor terms.
    return_df : pandas.DataFrame
        One-row timing frame with ``T_vis_mask``, ``T_uv_sampling_grid``
        (zero) and ``T_vis_grid`` (the transforms), the columns of the
        gridder path.
    """
    import pandas as pd

    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    data_group = img_xds.attrs["data_groups"].get(image_data_group_out_name, {})
    image_name = data_group.get("sky", _SKY_ROLES["sky"])
    if image_name in img_xds:
        img_xds.xr_img.delete_data_variables(variables=[image_name])

    T_vis_mask = 0.0
    T_vis_grid = 0.0
    for ms_xdt in ps_xdt.values():
        start = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - start

        start = time.time()
        add_dft_image_continuum_single_field(
            ms_xdt,
            img_xds,
            n_taylor_terms,
            reference_frequency,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            uv_sampling=False,
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
        )
        T_vis_grid += time.time() - start

    return_df = pd.DataFrame(
        {
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [0.0],
            "T_vis_grid": [T_vis_grid],
        }
    )
    return img_xds, return_df


@shares_param_docs
def make_dft_point_spread_function_continuum_single_field(
    ps_xdt,
    img_xds,
    n_psf_taylor_terms,
    reference_frequency,
    ms_data_group_in_name="base",
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    processing_function_threads=1,
    complex_dtype=None,
):
    """Build the MT-MFS Taylor point spread functions by the direct Fourier transform of the imaging weights.

    The direct Fourier transform counterpart of
    :func:`~astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field.make_point_spread_function_continuum_single_field`:
    drops the auto-correlations and sums the transforms of the imaging
    weights times ``x**k`` of all measurement sets
    (:func:`add_dft_image_continuum_single_field` with ``uv_sampling=True``)
    into a new ``POINT_SPREAD_FUNCTION(time, psf_taylor_order, polarization,
    l, m)`` and its ``UV_SAMPLING_NORMALIZATION`` (an existing
    ``POINT_SPREAD_FUNCTION`` is replaced). As the residual, the sums are
    normalised after they are added over the map tasks
    (:func:`normalize_dft_image`); order zero is then 1 at the phase centre.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set; auto-correlations are dropped in place.
    img_xds : xarray.Dataset
        Continuum image dataset in the polarization basis of the
        visibilities. The data group ``image_data_group_in_name`` must exist.
        Modified in place.
    n_psf_taylor_terms : int
        Number of point spread function orders, ``2 * n_taylor_terms - 1``.
    reference_frequency : float
        Taylor-expansion reference frequency in Hz.
    ms_data_group_in_name : str, optional
        Measurement-set data group that supplies the weights and uvw.
        Default ``"base"``.
    image_data_group_in_name : str, optional
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    image_data_group_out_name : str, optional
        Image data group the point spread functions are registered under.
        Default ``"residual"``.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    complex_dtype : numpy.dtype, optional
        ``complex64`` gives float32 point spread functions, ``complex128``
        (the default) float64 ones.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with the unnormalised Taylor point spread functions.
    return_df : pandas.DataFrame
        One-row timing frame with the ``T_gcf`` (zero), ``T_vis_mask`` and
        ``T_uv_sampling_grid`` (the transforms) columns of the gridder path.
    """
    import pandas as pd

    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    data_group = img_xds.attrs["data_groups"].get(image_data_group_out_name, {})
    image_name = data_group.get(
        "point_spread_function", _UV_SAMPLING_ROLES["point_spread_function"]
    )
    if image_name in img_xds:
        img_xds.xr_img.delete_data_variables(variables=[image_name])

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0
    for ms_xdt in ps_xdt.values():
        start = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - start

        start = time.time()
        add_dft_image_continuum_single_field(
            ms_xdt,
            img_xds,
            n_psf_taylor_terms,
            reference_frequency,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_in_name,
            image_data_group_out_name=image_data_group_out_name,
            uv_sampling=True,
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
        )
        T_uv_sampling_grid += time.time() - start

    return_df = pd.DataFrame(
        {
            "T_gcf": [0.0],
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [T_uv_sampling_grid],
        }
    )
    return img_xds, return_df


def make_dft_visibility_model_continuum_single_field(
    ps_xdt,
    model_xds,
    n_taylor_terms,
    reference_frequency,
    ms_data_group_in_name="base",
    ms_data_group_out_name="model",
    ms_data_group_out_modified=None,
    image_data_group_in_name="model",
    processing_function_threads=1,
):
    """Predict channel-dependent model visibilities from a Taylor model by the direct Fourier transform.

    The direct Fourier transform counterpart of
    :func:`~astroviper.processing_functions.imaging.residual_cycle_continuum_single_field.make_visibility_model_continuum_single_field`:
    in every measurement set, the model visibility of channel frequency
    ``nu`` is

        ``V(nu) = sum over t of x**t V_t``,
        ``x = (nu - reference_frequency) / reference_frequency``,

    with ``V_t = sum over the non-zero pixels of M_t exp(+i phase) / n`` the
    prediction of Taylor term ``M_t`` of the model at the sample's ``uvw``
    and frequency (the convention of the module docstring). The samples
    predicted are those of :func:`predict_dft_visibilities_single_field`; no
    FFT, padding or correction is involved.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing-set partition receiving the predicted model visibilities.
    model_xds : xarray.Dataset
        The Taylor model in the polarization basis of the visibilities: the
        ``sky`` role of ``image_data_group_in_name`` names a variable with
        dimensions ``(time, taylor_term, polarization, l, m)``, in Jy per
        pixel (apparent flux).
    n_taylor_terms : int
        Number of Taylor terms to predict (extra terms of the model are
        ignored).
    reference_frequency : float
        Taylor-expansion reference frequency in Hz.
    ms_data_group_in_name : str, optional
        Processing-set data group being imaged. Default ``"base"``.
    ms_data_group_out_name : str, optional
        Processing-set data group receiving the model visibilities. Default
        ``"model"``.
    ms_data_group_out_modified : dict, optional
        Default ``{"correlated_data": "VISIBILITY_MODEL"}``.
    image_data_group_in_name : str, optional
        Image data group of the model. Default ``"model"``.
    processing_function_threads : int, optional
        Threads of the C++ kernel. The result does not depend on it.

    Returns
    -------
    None
        The measurement sets of ``ps_xdt`` are modified in place.

    Raises
    ------
    KeyError
        If the model's data group or sky variable is absent.
    ValueError
        For fewer than one Taylor term or fewer in the model, a reference
        frequency that is not finite and positive, or as
        :func:`predict_dft_visibilities_single_field`.
    """
    from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
        dft_predict,
    )

    n_taylor_terms = int(n_taylor_terms)
    reference_frequency = float(reference_frequency)
    if n_taylor_terms < 1:
        raise ValueError("n_taylor_terms must be at least 1.")
    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "reference_frequency must be a positive finite frequency in Hz."
        )
    if ms_data_group_out_modified is None:
        ms_data_group_out_modified = {"correlated_data": "VISIBILITY_MODEL"}

    data_groups = model_xds.attrs.get("data_groups", {})
    if image_data_group_in_name not in data_groups:
        raise KeyError(
            f"Image data group {image_data_group_in_name!r} is not present "
            "in model_xds.attrs['data_groups']."
        )
    model_name = data_groups[image_data_group_in_name].get("sky")
    if model_name is None or model_name not in model_xds:
        raise KeyError(
            f"Image data group {image_data_group_in_name!r} does not name a sky "
            "variable of model_xds."
        )
    model = model_xds[model_name]
    if "taylor_term" not in model.dims or model.sizes["taylor_term"] < n_taylor_terms:
        raise ValueError(
            f"The model {model_name!r} must have at least {n_taylor_terms} Taylor "
            f"terms; it has dimensions {dict(model.sizes)}."
        )
    model_values = model.transpose(
        "time", "taylor_term", "polarization", "l", "m"
    ).values
    l, m = image_direction_cosines(model_xds)

    for ms_xdt in ps_xdt.values():
        ms_data_group_in, ms_data_group_out = create_data_groups_in_and_out(
            ms_xdt,
            data_group_in_name=ms_data_group_in_name,
            data_group_out_name=ms_data_group_out_name,
            data_group_out_modified=dict(ms_data_group_out_modified),
            overwrite=True,
        )
        frequency_map, time_map, pol_map = _maps(ms_xdt, model_xds, "continuum")
        model_visibilities = _model_visibilities(
            ms_xdt,
            ms_data_group_in,
            ms_data_group_out,
            ms_data_group_out_name,
            description="Continuum Taylor model predicted with "
            "make_dft_visibility_model_continuum_single_field.",
        )
        vis_data = model_visibilities.values
        uvw = np.ascontiguousarray(
            ms_xdt[ms_data_group_in["uvw"]].values, dtype=np.float64
        )
        frequency_coord = np.ascontiguousarray(
            ms_xdt.frequency.values, dtype=np.float64
        )
        x = (frequency_coord - reference_frequency) / reference_frequency
        weight_shape = [1] * vis_data.ndim
        weight_shape[model_visibilities.get_axis_num("frequency")] = x.size

        # Every Taylor term is predicted into the output; the zeroth Taylor
        # weight is one for every channel, a higher term is weighted and summed
        # in visibility space.
        accumulated = None
        for taylor_term in range(n_taylor_terms):
            dft_predict(
                np.ascontiguousarray(model_values[:, taylor_term : taylor_term + 1]),
                vis_data,
                uvw,
                frequency_coord,
                frequency_map,
                time_map,
                pol_map,
                l,
                m,
                processing_function_threads=processing_function_threads,
            )
            if taylor_term == 0:
                if n_taylor_terms > 1:
                    accumulated = vis_data.copy()
                continue
            np.multiply(vis_data, (x**taylor_term).reshape(weight_shape), out=vis_data)
            accumulated += vis_data
        if accumulated is not None:
            vis_data[...] = accumulated
