from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def model_update_cycle_cube_single_field(
    img_xds,
    deconvolver,
    deconvolve_params,
    is_n_iter_0,
    processing_function_threads=1,
    image_data_group_in_name="residual",
    image_data_group_out_name="model",
):
    """Run one model-update (minor) cycle: build a mask and deconvolve.

    Ensures a primary-beam mask exists on the input data group, then runs the
    configured deconvolver (Hogbom or Asp CLEAN) which updates the sky model in
    place and returns per-plane convergence statistics.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset holding the residual image (input data group) and the sky
        model (output data group).  Modified in place.
    deconvolver : str
        Deconvolution algorithm for the minor cycle. One of ``"hogbom"`` (C++, threaded across planes), ``"hogbom_many_threads"``
        (C++, threaded across *and* within planes -- faster when there are
        few planes, e.g. single-channel imaging) or ``"asp"``.
    deconvolve_params : dict
        Per-cycle deconvolution / iteration-control parameters passed straight to
        the deconvolver: the absolute ``threshold`` (floor) plus the adaptive
        ``cyclethreshold``, ``gain``, ``cycleniter``, ``niter_per_plane`` and
        ``cyclethreshold_per_plane`` ...  The separate ``primary_beam_limit``
        builds the primary-beam mask (a chunk-independent quantity, so the mask
        does not depend on how the cube was split across tasks); it is distinct
        from the deconvolver ``threshold``.
    is_n_iter_0 : bool
        ``True`` on the very first model update.  Currently informational.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    image_data_group_in_name : str, optional
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    image_data_group_out_name : str, optional
        Data group that the updated sky model is registered under.  Default
        ``"model"``.

    Returns
    -------
    deconvolve_dict : ReturnDict
        Per-plane deconvolution statistics for this cycle.
    return_df : pandas.DataFrame
        One-row timing frame with the ``T_make_mask`` and ``T_deconvolve``
        columns.
    """
    import time

    import pandas as pd

    from astroviper.processing_functions.image_analysis.make_mask import make_mask
    from astroviper.processing_functions.imaging.deconvolution import (
        _validate_deconvolve_params,
        deconvolve,
    )

    # Apply deconvolve-parameter defaults (e.g. ``primary_beam_limit``) up front
    # so both the mask step below and the deconvolver see a complete dict.
    deconvolve_params = _validate_deconvolve_params(deconvolve_params)

    # Add a primary-beam mask to the input data group if one is not present yet.
    T_make_mask = 0.0
    mask_name = img_xds.attrs["data_groups"][image_data_group_in_name].get("mask", None)
    if mask_name is None:
        start = time.time()
        make_mask(
            img_xds,
            primary_beam_limit=deconvolve_params["primary_beam_limit"],
            image_data_group_in_name=image_data_group_in_name,
            image_data_group_out_name=image_data_group_in_name,
            combine_mask=False,
            overwrite=False,
        )
        T_make_mask = time.time() - start

    start = time.time()
    deconvolve_dict = deconvolve(
        img_xds=img_xds,
        algorithm=deconvolver,
        deconvolve_params=deconvolve_params,
        image_data_group_in_name=image_data_group_in_name,
        image_data_group_out_name=image_data_group_out_name,
        processing_function_threads=processing_function_threads,
    )
    T_deconvolve = time.time() - start

    return_df = pd.DataFrame(
        {"T_make_mask": [T_make_mask], "T_deconvolve": [T_deconvolve]}
    )

    return deconvolve_dict, return_df
