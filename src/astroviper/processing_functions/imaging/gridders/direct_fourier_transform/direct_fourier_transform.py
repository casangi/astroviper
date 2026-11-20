"""Exact direct Fourier transform imaging and prediction for single-field imaging.

The gridder ``"direct_fourier_transform"``
replaces the gridding, the FFT and the image plane correction of the residual
update and of the point spread function by exact sums over the visibilities
and the image pixels, made by
:class:`~astroviper.processing_functions.imaging.gridders.direct_fourier_transform.gridder_direct_fourier_transform.GridderDirectFourierTransform`
with the threaded C++ kernels of
:mod:`~astroviper.processing_functions.imaging.gridders.direct_fourier_transform.direct_fourier_transform_cpp`.
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
sample is dropped for lying beyond the uv grid, as there is none, or for a
zero visibility, and a sample whose ``w`` alone is not finite is dropped.
The normalization (``VISIBILITY_NORMALIZATION``,
``UV_SAMPLING_NORMALIZATION``) is the sum of
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

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import modify_data_groups_xds


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


def _maps(ms_xds, img_xds):
    """Time and polarization maps of a measurement set onto the image."""
    n_pol = ms_xds.sizes["polarization"]
    _check_polarizations(img_xds, n_pol)
    # Time mapping is not implemented: every visibility time maps onto plane 0.
    time_map = np.zeros(ms_xds.sizes["time"], dtype=np.int64)
    pol_map = np.arange(n_pol, dtype=np.int64)
    return time_map, pol_map


def _check_polarizations(img_xds, n_pol):
    """Refuse an image whose polarization axis differs from the measurement set's."""
    if img_xds.sizes["polarization"] != n_pol:
        raise ValueError(
            "the image polarization axis must match the measurement set: "
            f"{img_xds.sizes['polarization']} image polarizations for {n_pol} "
            "visibility polarizations."
        )


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
        Image dataset holding the image and normalization that
        :class:`~astroviper.processing_functions.imaging.gridders.direct_fourier_transform.gridder_direct_fourier_transform.GridderDirectFourierTransform`
        accumulated. Modified in place (a read-only image is replaced by its
        normalised copy).
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


###############################################################################
# Continuum (MT-MFS) imaging
###############################################################################


def check_dft_continuum_parameters(
    continuum_mode="mfs",
    visibility_memory_mode="recompute",
    image_data_variables_keep=(),
):
    """Refuse the continuum options the direct Fourier transform does not take.

    Continuum imaging takes ``gridder="direct_fourier_transform"`` in MFS
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
    from astroviper.processing_functions.imaging.config_check import (
        CONTINUUM_SINGLE_FIELD,
        check_driver_workflow,
    )

    check_driver_workflow(
        {
            "gridder": "direct_fourier_transform",
            "continuum_mode": continuum_mode,
            "visibility_memory_mode": visibility_memory_mode,
            "image_data_variables_keep": image_data_variables_keep,
        },
        mode=CONTINUUM_SINGLE_FIELD,
    )
