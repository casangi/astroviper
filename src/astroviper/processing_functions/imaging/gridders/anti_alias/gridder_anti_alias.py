"""The anti-alias gridder (the standard gridder in CASA).

It grids with the separable kernel of the anti-alias term of ``gcf_xds`` (the
C++ kernels of ``gridders/anti_alias/anti_alias_grid_cpp``) onto a padded uv grid, and
transforms between the uv grid and the image with an FFT and the image plane
correction of the kernel (``normalization/normalize_image``).
"""

from types import MappingProxyType

import numpy as np

from astroviper.processing_functions.imaging.gridders.gridder_base import Gridder


class GridderAntiAlias(Gridder):
    """The anti-alias gridder; its methods follow the contracts of :class:`Gridder`.

    Its normalization holds the sums of the weights times the stencil sums of
    the kernel. All times go to time plane 0, polarization ``p`` to plane
    ``p``, and the uv grids have no ``type`` attribute.
    """

    name = "anti_alias"
    normalization = "sum_of_weights"
    gcf_terms = ("anti_alias",)
    grid_roles = MappingProxyType(
        {
            "visibility": "VISIBILITY",
            "visibility_normalization": "VISIBILITY_NORMALIZATION",
        }
    )
    grid_uv_sampling_roles = MappingProxyType(
        {
            "uv_sampling": "UV_SAMPLING",
            "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
        }
    )
    degrid_role = "visibility"

    @staticmethod
    def new_grid(img_xds, leading_shape, uv_sampling, fft_padding, complex_dtype):
        """:meth:`Gridder.new_grid`: zeros of ``complex_dtype`` on the padded uv grid.

        The grids of :meth:`grid` and :meth:`grid_uv_sampling` are the same.
        """
        from astroviper.processing_functions.imaging.utils.fft_sizing import (
            padded_grid_size,
        )

        n_uv = padded_grid_size([img_xds.sizes["l"], img_xds.sizes["m"]], fft_padding)
        return (
            np.zeros((*leading_shape, n_uv[0], n_uv[1]), dtype=complex_dtype),
            ("u", "v"),
            None,
        )

    @staticmethod
    def grid(
        ms_xds,
        img_xds,
        gcf_xds,
        weights,
        frequency_map,
        grid,
        normalization,
        ms_data_group_in_name,
        fft_padding,
        processing_function_threads,
    ):
        """:meth:`Gridder.grid` with the C++ ``anti_alias_grid``."""
        from astroviper.processing_functions.imaging.gridders.anti_alias.anti_alias_grid_cpp import (
            anti_alias_grid,
        )

        GridderAntiAlias._call_kernel(
            anti_alias_grid,
            True,
            ms_xds,
            img_xds,
            gcf_xds,
            weights,
            frequency_map,
            grid,
            normalization,
            ms_data_group_in_name,
            fft_padding,
            processing_function_threads,
        )

    @staticmethod
    def grid_uv_sampling(
        ms_xds,
        img_xds,
        gcf_xds,
        weights,
        frequency_map,
        grid,
        normalization,
        ms_data_group_in_name,
        fft_padding,
        processing_function_threads,
    ):
        """:meth:`Gridder.grid_uv_sampling` with the C++ ``anti_alias_grid_uv_sampling``."""
        from astroviper.processing_functions.imaging.gridders.anti_alias.anti_alias_grid_cpp import (
            anti_alias_grid_uv_sampling,
        )

        GridderAntiAlias._call_kernel(
            anti_alias_grid_uv_sampling,
            False,
            ms_xds,
            img_xds,
            gcf_xds,
            weights,
            frequency_map,
            grid,
            normalization,
            ms_data_group_in_name,
            fft_padding,
            processing_function_threads,
        )

    @staticmethod
    def _call_kernel(
        kernel,
        with_correlated_data,
        ms_xds,
        img_xds,
        gcf_xds,
        weights,
        frequency_map,
        grid,
        normalization,
        ms_data_group_in_name,
        fft_padding,
        processing_function_threads,
    ):
        """Call ``kernel``, the C++ gridding of the visibilities or of the weights."""
        from astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term import (
            anti_alias_kernel_arguments,
        )
        from astroviper.processing_functions.imaging.utils.fft_sizing import (
            padded_grid_size,
        )

        ms_data_group_in = ms_xds.attrs["data_groups"][ms_data_group_in_name]
        time_map = (np.zeros(ms_xds.sizes["time"])).astype(int)
        pol_map = (np.arange(0, weights.shape[3])).astype(int)
        n_uv = padded_grid_size([img_xds.sizes["l"], img_xds.sizes["m"]], fft_padding)
        delta_lm = img_xds.xr_img.get_lm_cell_size()
        correlated_data = (
            (ms_xds[ms_data_group_in["correlated_data"]].values,)
            if with_correlated_data
            else ()
        )
        uvw = ms_xds[ms_data_group_in["uvw"]].values
        table, support, oversampling, interpolation = anti_alias_kernel_arguments(
            gcf_xds
        )
        kernel(
            grid,
            normalization,
            *correlated_data,
            uvw,
            ms_xds.frequency.values,
            frequency_map,
            time_map,
            pol_map,
            weights,
            table,
            n_uv,
            delta_lm,
            support=support,
            oversampling=oversampling,
            interpolation=interpolation,
            processing_function_threads=processing_function_threads,
        )

    @staticmethod
    def degrid(
        ms_xds,
        img_xds,
        gcf_xds,
        grid,
        frequency_map,
        ms_data_group_in_name,
        ms_data_group_out_name,
        ms_data_group_out_modified,
        overwrite,
        fft_padding,
        processing_function_threads,
        description,
    ):
        """:meth:`Gridder.degrid` by :func:`~astroviper.processing_functions.imaging.gridders.anti_alias.degrid_visibility_grid.degrid_visibility_grid_single_field`.

        It allocates a missing output variable as complex128 zeros and refuses
        a frequency map or a grid that does not fit the measurement set.
        """
        from astroviper.processing_functions.imaging.gridders.anti_alias.degrid_visibility_grid import (
            degrid_visibility_grid_single_field,
        )

        degrid_visibility_grid_single_field(
            ms_xds,
            gcf_xds,
            img_xds,
            grid,
            frequency_map,
            ms_data_group_in_name=ms_data_group_in_name,
            ms_data_group_out_name=ms_data_group_out_name,
            ms_data_group_out_modified=ms_data_group_out_modified,
            overwrite=overwrite,
            fft_padding=fft_padding,
            processing_function_threads=processing_function_threads,
            description=description,
        )

    @staticmethod
    def uv_to_image(
        img_xds,
        image_params,
        gcf_xds,
        image_data_group_in_name,
        image_data_group_out_name,
        image_data_group_out_modified,
        overwrite,
        image_data_variables_keep,
        processing_function_threads,
        fft_backend,
        complex_dtype,
    ):
        """:meth:`Gridder.uv_to_image` by :func:`~astroviper.processing_functions.imaging.normalization.normalize_image.ifft_norm_img_xds`."""
        from astroviper.processing_functions.imaging.normalization.normalize_image import (
            ifft_norm_img_xds,
        )

        return ifft_norm_img_xds(
            img_xds,
            image_params=image_params,
            image_data_group_in_name=image_data_group_in_name,
            image_data_group_out_name=image_data_group_out_name,
            image_data_group_out_modified=image_data_group_out_modified,
            overwrite=overwrite,
            image_data_variables_keep=image_data_variables_keep,
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            complex_dtype=complex_dtype,
            gcf_xds=gcf_xds,
        )

    @staticmethod
    def image_to_uv(
        img_xds,
        image_params,
        gcf_xds,
        plane_dim,
        image_data_group_in_name,
        image_data_group_out_name,
        image_data_group_out_modified,
        overwrite,
        image_data_variables_keep,
        processing_function_threads,
        fft_backend,
        data_variables_to_process,
        complex_dtype,
        new_polarization_basis,
    ):
        """:meth:`Gridder.image_to_uv` by the FFT normalization functions.

        :func:`~astroviper.processing_functions.imaging.normalization.normalize_image.fft_norm_img_xds`
        transforms frequency planes and
        :func:`~astroviper.processing_functions.imaging.normalization.normalize_image.fft_norm_continuum_img_xds`
        Taylor planes.

        Raises
        ------
        ValueError
            If ``plane_dim`` is not ``"frequency"`` or ``"taylor_term"``, or
            is ``"taylor_term"`` with a ``new_polarization_basis``.
        """
        from astroviper.processing_functions.imaging.normalization.normalize_image import (
            fft_norm_continuum_img_xds,
            fft_norm_img_xds,
        )

        if plane_dim == "frequency":
            return fft_norm_img_xds(
                img_xds,
                image_params=image_params,
                image_data_group_in_name=image_data_group_in_name,
                image_data_group_out_name=image_data_group_out_name,
                image_data_group_out_modified=image_data_group_out_modified,
                overwrite=overwrite,
                image_data_variables_keep=image_data_variables_keep,
                processing_function_threads=processing_function_threads,
                fft_backend=fft_backend,
                data_variables_to_process=data_variables_to_process,
                complex_dtype=complex_dtype,
                gcf_xds=gcf_xds,
                new_polarization_basis=new_polarization_basis,
            )
        if plane_dim == "taylor_term" and new_polarization_basis is None:
            return fft_norm_continuum_img_xds(
                img_xds,
                image_params=image_params,
                image_data_group_in_name=image_data_group_in_name,
                image_data_group_out_name=image_data_group_out_name,
                image_data_group_out_modified=image_data_group_out_modified,
                overwrite=overwrite,
                image_data_variables_keep=image_data_variables_keep,
                processing_function_threads=processing_function_threads,
                fft_backend=fft_backend,
                data_variables_to_process=data_variables_to_process,
                complex_dtype=complex_dtype,
                gcf_xds=gcf_xds,
            )
        raise ValueError(
            "plane_dim must be 'frequency', or 'taylor_term' without "
            f"new_polarization_basis; received plane_dim={plane_dim!r} and "
            f"new_polarization_basis={new_polarization_basis!r}."
        )
