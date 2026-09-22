import numpy as np
import xarray as xr


def get_visibility_grid_single_field(
    ms_xds: xr.Dataset,
    cgk_1D: np.ndarray,
    img_xds: xr.Dataset,
    ms_data_group_in_name: str = "base",
    ms_data_group_out_name: str = "model",
    ms_data_group_out_modified: dict | None = None,
    image_data_group_in_name: str = "model",
    overwrite: bool = True,
    chan_mode: str = "cube",
    fft_padding: float = 1.2,
    processing_function_threads: int = 1,
):
    """Degrid a UV-domain model onto measurement-set visibility coordinates.

    Samples a model UV grid stored in ``img_xds`` at each visibility's
    ``(u, v)`` coordinate using a separable 1-D convolutional gridding kernel
    (``cgk_1D``), and writes the resulting predicted visibilities into
    ``ms_xds`` under the data variable named by
    ``ms_data_group_out_modified["correlated_data"]`` (default
    ``"VISIBILITY_MODEL"``).  A corresponding output data group is registered
    on ``ms_xds`` via
    :func:`~astroviper.utils.data_group_tools.modify_data_groups_xds`.

    This function is the inverse of
    :func:`~astroviper.processing_functions.imaging.add_visibility_grid.add_visibility_grid_single_field`:
    it predicts visibilities *from* a UV-plane model rather than gridding
    observed visibilities onto a UV plane.  It uses the separable C++ prolate
    spheroidal degridder (the standard gridder in CASA) and therefore does not
    require a GCF dataset. Cube and continuum callers share that numerical
    primitive without sharing their spectral-model preparation APIs.

    Parameters
    ----------
    ms_xds : xr.Dataset
        Measurement set dataset.  Must contain the data variables referenced
        by ``ms_data_group_in_name`` (``correlated_data`` and ``uvw``) and a
        ``frequency`` coordinate.  On the first call the output visibility
        data variable is created on ``ms_xds`` as a ``complex128`` array shaped
        like the input ``correlated_data``.
    cgk_1D : np.ndarray
        Oversampled 1-D prolate spheroidal wave function (PSWF) kernel.
        Shape ``(oversampling * (support // 2 + 1),)``; passed directly to
        the prolate spheroidal degridder.
    img_xds : xr.Dataset
        Image dataset holding the model UV grid.  Must expose an image data
        group under ``image_data_group_in_name`` whose ``"visibility"`` role
        names a data variable shaped ``(time, frequency, polarization, u, v)``,
        the same ``(u, v)`` axis order used by the C++
        ``prolate_spheroidal_grid`` kernel.
        Cell size and image dimensions are read directly from ``img_xds`` via
        ``img_xds.xr_img.get_lm_cell_size()`` and ``img_xds.sizes``.
    ms_data_group_in_name : str, default ``"base"``
        Key of the MS input data group in ``ms_xds.attrs["data_groups"]``.
        Must provide the ``"correlated_data"`` and ``"uvw"`` role keys.
    ms_data_group_out_name : str, default ``"model"``
        Key under which the output data group is registered in
        ``ms_xds.attrs["data_groups"]``.
    ms_data_group_out_modified : dict, default ``{"correlated_data": "VISIBILITY_MODEL"}``
        Mapping of role keys to the data-variable names written into
        ``ms_xds``.  ``"correlated_data"`` stores the complex predicted
        visibilities.
    image_data_group_in_name : str, default ``"model"``
        Key of the image input data group in ``img_xds.attrs["data_groups"]``
        whose ``"visibility"`` role names the model UV grid.
    overwrite : bool, default ``True``
        If ``True``, an existing output data group or output data variable is
        silently overwritten.  Defaults to ``True`` because this function is
        typically called repeatedly in an iterative cycle.
    chan_mode : str, default ``"cube"``
        Channel mapping mode.  ``"cube"`` maps each visibility channel to the
        image channel nearest in frequency (see
        :func:`~astroviper.processing_functions.imaging.utils.frequency_mapping.map_visibility_frequencies_to_image`);
        ``"continuum"`` sources every visibility channel from image channel 0.
    fft_padding : float, default ``1.2``
        Padding factor applied to the image size when computing the UV-grid
        dimensions.
    processing_function_threads : int, default ``1``
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.

    Returns
    -------
    None
        Modifies ``ms_xds`` in place (output data variable and ``data_groups``
        attribute); no return value.

    Raises
    ------
    ValueError
        If, in ``"cube"`` mode, a visibility channel lies more than half an
        image channel width from every image channel, or if the grid's
        polarization axis does not match ``ms_xds``.

    Notes
    -----
    - Flags must be applied by the caller before invoking this function.  The
      underlying degridder has no flag argument and only skips samples whose
      ``vis_data`` value is ``NaN``. Newly
      allocated output arrays are zero-initialised (not ``NaN``), so every
      visibility whose ``uvw`` is finite and whose support falls inside the
      grid will receive a prediction.
    - Time mapping is not currently implemented: every visibility time index
      is mapped to image time index 0.
    - ``processing_function_threads <= 0`` lets the C++ degridder fall back
      to the hardware concurrency; ``1`` runs serially.

    See Also
    --------
    astroviper.processing_functions.imaging.add_visibility_grid.add_visibility_grid_single_field :
        Forward (gridding) counterpart.
    degrid_visibility_grid_single_field :
        Shared prolate spheroidal gridder numerical primitive.
    """
    if ms_data_group_out_modified is None:
        ms_data_group_out_modified = {
            "correlated_data": "VISIBILITY_MODEL",
        }
    model_name = img_xds.attrs["data_groups"][image_data_group_in_name]["visibility"]

    n_chan = ms_xds.sizes["frequency"]
    if chan_mode == "cube":
        from astroviper.processing_functions.imaging.utils.frequency_mapping import (
            map_visibility_frequencies_to_image,
        )

        frequency_map = map_visibility_frequencies_to_image(
            ms_xds.frequency.values,
            img_xds.frequency.values,
        )
    else:  # continuum
        frequency_map = (np.zeros(n_chan)).astype(int)

    from astroviper.processing_functions.imaging.degrid_visibility_grid import (
        degrid_visibility_grid_single_field,
    )

    degrid_visibility_grid_single_field(
        ms_xds,
        cgk_1D,
        img_xds,
        img_xds[model_name].values,
        frequency_map,
        ms_data_group_in_name=ms_data_group_in_name,
        ms_data_group_out_name=ms_data_group_out_name,
        ms_data_group_out_modified=ms_data_group_out_modified,
        overwrite=overwrite,
        fft_padding=fft_padding,
        processing_function_threads=processing_function_threads,
        description="Degridded visibilities from img_xds "
        + image_data_group_in_name
        + " to ms_xds "
        + ms_data_group_out_name
        + " with get_visibility_grid_single_field.",
    )
