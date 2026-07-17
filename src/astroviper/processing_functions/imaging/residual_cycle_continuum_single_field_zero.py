from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def residual_cycle_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    is_n_iter_0,
    processing_set_data_group_name="corrected",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    image_data_group_in_name="model",
    image_data_group_out_name="residual",
    last_residual_cycle=False,
):
    """Create zero-valued continuum residual Taylor terms for testing.

    This is a temporary scaffolding implementation of the continuum residual
    (major) cycle. It does not yet degrid a model, calculate residual
    visibilities, grid visibility data, run an FFT, or transform polarization.

    Instead, it creates a zero-valued ``SKY_RESIDUAL`` array with dimensions

    ``(time, taylor_term, polarization, l, m)``

    and registers that array as the ``"sky"`` variable of
    ``image_data_group_out_name``.

    For ``nterms=2``, the function therefore creates two zero-valued residual
    Taylor images, corresponding to ``R_0`` and ``R_1``.

    The function deliberately retains the full intended continuum residual
    cycle signature, so the placeholder can later be replaced incrementally
    without changing its callers.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Visibility data for this frequency chunk. Currently unused.
    img_xds : xarray.Dataset
        Image dataset to which the zero-valued residual Taylor terms are added.
    image_params : dict
        Image parameters. ``image_params.get("nterms", 2)`` determines the
        number of residual Taylor terms.
    is_n_iter_0 : bool
        Whether this is the first residual cycle. Currently recorded only in
        the returned timing metadata.
    processing_set_data_group_name : str, optional
        Processing-set input group. Currently unused.
    instrument_polarization_basis : str, optional
        Instrument polarization basis. Currently unused because zeros are
        basis invariant.
    single_precision_image : bool, optional
        If true, create ``float32`` residual images; otherwise use ``float64``.
    processing_function_threads : int, optional
        Retained for interface compatibility.
    fft_backend : str, optional
        Retained for interface compatibility.
    image_data_variables_keep : list of str, optional
        Retained for interface compatibility.
    image_data_group_in_name : str, optional
        Model data group. Currently unused.
    image_data_group_out_name : str, optional
        Output residual data group. Defaults to ``"residual"``.
    last_residual_cycle : bool, optional
        Retained for interface compatibility.

    Returns
    -------
    img_xds : xarray.Dataset
        Dataset containing zero-valued residual Taylor images.
    return_df : pandas.DataFrame
        One-row timing dataframe with the same broad timing keys as the cube
        residual-cycle implementation.
    """
    import time

    import numpy as np
    import pandas as pd
    import toolviper.utils.logger as logger
    import xarray as xr

    from astroviper.utils.data_group_tools import modify_data_groups_xds

    cycle_start = time.time()

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    nterms = int(image_params.get("nterms", 2))

    if nterms < 1:
        raise ValueError("image_params['nterms'] must be at least 1.")

    float_dtype = np.float32 if single_precision_image else np.float64

    required_dimensions = ("time", "polarization", "l", "m")
    missing_dimensions = [
        dimension for dimension in required_dimensions if dimension not in img_xds.sizes
    ]

    if missing_dimensions:
        raise ValueError(
            "img_xds is missing dimensions required for continuum residual "
            f"terms: {missing_dimensions}. Available dimensions are "
            f"{list(img_xds.sizes)}."
        )

    # Add or replace the Taylor coordinate. assign_coords returns a new Dataset,
    # which is important because adding a new dimension cannot be performed by
    # mutating only an existing DataArray.
    img_xds = img_xds.assign_coords(taylor_term=np.arange(nterms, dtype=np.int32))

    residual_shape = (
        img_xds.sizes["time"],
        nterms,
        img_xds.sizes["polarization"],
        img_xds.sizes["l"],
        img_xds.sizes["m"],
    )

    residual_dims = (
        "time",
        "taylor_term",
        "polarization",
        "l",
        "m",
    )

    residual_coords = {
        dimension: img_xds.coords[dimension]
        for dimension in residual_dims
        if dimension in img_xds.coords
    }

    residual_name = "SKY_RESIDUAL"

    img_xds[residual_name] = xr.DataArray(
        np.zeros(residual_shape, dtype=float_dtype),
        dims=residual_dims,
        coords=residual_coords,
        attrs={
            "description": ("Temporary zero-valued continuum residual Taylor terms."),
            "units": "Jy/beam",
            "nterms": nterms,
            "placeholder": True,
        },
    )

    # Ensure that the output data group exists.
    data_groups = img_xds.attrs.setdefault("data_groups", {})

    if image_data_group_out_name not in data_groups:
        img_xds.attrs["type"] = "image_dataset"
        img_xds = img_xds.xr_img.add_data_group(
            new_data_group_name=image_data_group_out_name,
            new_data_group={
                "description": ("Continuum residual Taylor products."),
                "date": "2026",
            },
        )

    # Register SKY_RESIDUAL as the group's sky variable. This mirrors the role
    # used by the cube residual-cycle implementation.
    residual_data_group = dict(img_xds.attrs["data_groups"][image_data_group_out_name])
    residual_data_group["sky"] = residual_name

    modify_data_groups_xds(
        img_xds,
        data_group_out_name=image_data_group_out_name,
        data_group_out=residual_data_group,
        description=(
            "Created zero-valued continuum residual Taylor terms for testing."
        ),
    )

    logger.debug(
        "Created zero-valued continuum residual Taylor terms with shape "
        f"{residual_shape} and dtype {np.dtype(float_dtype).name}."
    )

    # Preserve the timing interface expected by accumulate_timing() and by the
    # existing cube-imaging timing summaries. All real processing phases are
    # currently placeholders.
    return_df = pd.DataFrame(
        {
            "T_gcf": [0.0],
            "T_degrid": [0.0],
            "T_fft_degrid": [0.0],
            "T_residual_vis": [0.0],
            "T_grid": [0.0],
            "T_fft_grid": [0.0],
            "T_transform_pol": [0.0],
            "T_vis_mask": [0.0],
            "T_uv_sampling_grid": [0.0],
            "T_vis_grid": [0.0],
            "T_residual_cycle_placeholder": [time.time() - cycle_start],
            "nterms": [nterms],
            "is_n_iter_0": [bool(is_n_iter_0)],
            "last_residual_cycle": [bool(last_residual_cycle)],
        }
    )

    return img_xds, return_df
