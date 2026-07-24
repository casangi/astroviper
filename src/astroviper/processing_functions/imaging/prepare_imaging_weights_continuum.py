from astroviper.utils.param_docs import shares_param_docs


def _validate_prepared_imaging_weights(
    ps_xdt,
    *,
    processing_set_data_group_name,
    expected_weight_name,
):
    """Verify that imaging weights were created and registered."""

    datasets_checked = 0

    for node_name, node in ps_xdt.subtree_with_keys:

        ms_xds = node.ds

        if ms_xds is None:
            continue

        data_groups = ms_xds.attrs.get("data_groups", {})

        if processing_set_data_group_name not in data_groups:
            continue

        datasets_checked += 1

        data_group = data_groups[processing_set_data_group_name]
        registered_weight_name = data_group.get("weight_imaging")

        if registered_weight_name is None:
            raise RuntimeError(
                "'weight_imaging' was not registered in processing-set "
                f"data group {processing_set_data_group_name!r} for "
                f"DataTree node {node_name!r}."
            )

        if registered_weight_name != expected_weight_name:
            raise RuntimeError(
                "The imaging-weight variable was registered under the "
                f"unexpected name {registered_weight_name!r}; expected "
                f"{expected_weight_name!r}."
            )

        if registered_weight_name not in ms_xds:
            raise RuntimeError(
                f"Imaging-weight variable {registered_weight_name!r} was "
                f"registered for DataTree node {node_name!r}, but the variable "
                "is absent from that dataset."
            )

        weight_da = ms_xds[registered_weight_name]

        if weight_da.ndim != 4:
            raise ValueError(
                f"Imaging weights in DataTree node {node_name!r} must have "
                "four dimensions "
                "(time, baseline, frequency, polarization); "
                f"found dimensions {weight_da.dims}."
            )

    if datasets_checked == 0:
        raise RuntimeError(
            "No processing-set dataset containing data group "
            f"{processing_set_data_group_name!r} was found."
        )


@shares_param_docs
def prepare_imaging_weights_continuum(
    ps_xdt,
    img_xds,
    imaging_weights_params,
    processing_set_data_group_name="corrected",
    processing_function_threads=1,
    weight_imaging_name="WEIGHT_IMAGING",
):
    """Calculate and register imaging weights for one continuum data chunk.

    This function prepares the visibility-domain imaging weights required by
    continuum gridding. It is intended to be called once for each processing-set
    chunk before entering the major-cycle loop.

    The calculated weight array is installed in the selected processing-set
    data group under the logical role ``"weight_imaging"``. Subsequent residual
    and point-spread-function gridding can then reuse the prepared weights
    without recalculating them.

    This function modifies ``ps_xdt`` in place through
    :func:`calculate_imaging_weights`.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing-set data for one frequency chunk.

    img_xds : xarray.Dataset
        Image dataset providing the image geometry required by the weighting
        calculation.

    imaging_weights_params : dict
        Imaging-weight configuration, such as natural or Briggs weighting and
        the associated robustness parameter.

    processing_set_data_group_name : str, optional
        Processing-set data group containing the visibility data and receiving
        the imaging-weight registration.

    processing_function_threads : int, optional
        Number of threads supplied to the imaging-weight calculation.

    weight_imaging_name : str, optional
        Name assigned to the calculated imaging-weight variable. Defaults to
        ``"WEIGHT_IMAGING"``.

    Returns
    -------
    ps_xdt : xarray.DataTree
        Input processing-set chunk with the imaging weights calculated and
        registered.

    return_df : pandas.DataFrame
        One-row timing dataframe containing ``T_imaging_weights``.

    Notes
    -----
    This function only prepares the in-memory processing-set object. If every
    major cycle reloads the processing-set chunk from storage, the resulting
    imaging weights must also be persisted or cached and reattached when the
    chunk is loaded.
    """
    import time

    import pandas as pd

    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        calculate_imaging_weights,
    )

    if not isinstance(imaging_weights_params, dict):
        raise TypeError(
            "imaging_weights_params must be a dictionary; received "
            f"{type(imaging_weights_params).__name__}."
        )

    if not isinstance(processing_set_data_group_name, str):
        raise TypeError(
            "processing_set_data_group_name must be a string; received "
            f"{type(processing_set_data_group_name).__name__}."
        )

    if not processing_set_data_group_name:
        raise ValueError("processing_set_data_group_name must not be empty.")

    if not isinstance(weight_imaging_name, str):
        raise TypeError(
            "weight_imaging_name must be a string; received "
            f"{type(weight_imaging_name).__name__}."
        )

    if not weight_imaging_name:
        raise ValueError("weight_imaging_name must not be empty.")

    start = time.time()

    calculate_imaging_weights(
        ps_xdt,
        img_xds,
        imaging_weights_params=imaging_weights_params,
        return_weight_density_grid=False,
        ms_data_group_in_name=processing_set_data_group_name,
        ms_data_group_out_name=processing_set_data_group_name,
        ms_data_group_out_modified={
            "weight_imaging": weight_imaging_name,
        },
        processing_function_threads=processing_function_threads,
    )

    elapsed = time.time() - start

    _validate_prepared_imaging_weights(
        ps_xdt,
        processing_set_data_group_name=processing_set_data_group_name,
        expected_weight_name=weight_imaging_name,
    )

    return_df = pd.DataFrame(
        {
            "T_imaging_weights": [elapsed],
        }
    )

    return ps_xdt, return_df
