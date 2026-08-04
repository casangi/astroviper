"""Gridding of residual visibilities into an MVC frequency cube."""

import time

from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def make_undeconvolved_image_mvc_single_field(
    ps_xdt,
    img_xds,
    image_params,
    cgk_1D,
    ms_data_group_in_name="corrected",
    image_data_group_out_name="residual",
    processing_function_threads=1,
    complex_dtype=None,
):
    """Grid residual visibilities into a frequency-resolved MVC UV cube."""
    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.add_visibility_grid_continuum_mvc import (
        add_visibility_grid_mvc_single_field,
    )
    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    if complex_dtype is None:
        complex_dtype = np.complex128

    T_vis_mask = 0.0
    T_vis_grid = 0.0

    for _, ms_xdt in ps_xdt.items():
        start = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - start

        start = time.time()
        add_visibility_grid_mvc_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            image_data_group_out_modified={
                "visibility": "VISIBILITY",
                "visibility_normalization": ("VISIBILITY_NORMALIZATION"),
            },
            overwrite=True,
            fft_padding=image_params["fft_padding"],
            processing_function_threads=(processing_function_threads),
            complex_dtype=complex_dtype,
        )
        T_vis_grid += time.time() - start

    return_df = pd.DataFrame(
        {
            "T_vis_mask": [T_vis_mask],
            "T_vis_grid": [T_vis_grid],
        }
    )

    return img_xds, return_df
