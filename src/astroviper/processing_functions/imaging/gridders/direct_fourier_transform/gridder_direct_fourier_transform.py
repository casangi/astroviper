"""The direct Fourier transform between the visibilities and the image.

It sums over the visibilities and the image pixels with the threaded C++
kernels of ``gridders/direct_fourier_transform_cpp``, in the convention of
:mod:`~astroviper.processing_functions.imaging.direct_fourier_transform`: no uv
grid, padding, FFT or correction.
"""

from types import MappingProxyType

import numpy as np

from astroviper.processing_functions.imaging.gridders.gridder_base import Gridder


class GridderDirectFourierTransform(Gridder):
    """The direct Fourier transform; its methods follow the contracts of :class:`Gridder`.

    It accumulates into the images themselves, typed ``"sky"`` and
    ``"point_spread_function"``, and its normalization holds the sums of the
    weights. It applies no term of ``gcf_xds``: its methods accept and ignore
    ``gcf_xds`` and ``fft_padding``. All times go to time plane 0,
    polarization ``p`` to plane ``p``.
    """

    name = "direct_fourier_transform"
    normalization = "sum_of_weights"
    gcf_terms = ()
    grid_roles = MappingProxyType(
        {
            "sky": "SKY_RESIDUAL",
            "visibility_normalization": "VISIBILITY_NORMALIZATION",
        }
    )
    grid_uv_sampling_roles = MappingProxyType(
        {
            "point_spread_function": "POINT_SPREAD_FUNCTION",
            "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
        }
    )
    degrid_role = "sky"

    @staticmethod
    def new_grid(img_xds, leading_shape, uv_sampling, fft_padding, complex_dtype):
        """:meth:`Gridder.new_grid`: zeros of the image size, typed ``"sky"`` or ``"point_spread_function"``.

        The sky is ``complex_dtype`` in a four-correlation basis; otherwise the
        array is float32 for ``complex64`` and float64 for ``complex128``. A
        polarization size other than the image's is refused.
        """
        from astroviper.processing_functions.imaging.direct_fourier_transform import (
            _check_polarizations,
        )
        from astroviper.processing_functions.imaging.utils.imaging_polarization import (
            is_four_correlation_basis,
        )

        _check_polarizations(img_xds, leading_shape[2])
        # The sky in a four-correlation basis is complex (a cross hand is not
        # conjugate symmetric on its own), as the anti-alias gridder makes it.
        float_dtype = (
            np.float32 if np.dtype(complex_dtype) == np.complex64 else np.float64
        )
        image_dtype = (
            np.dtype(complex_dtype)
            if not uv_sampling
            and is_four_correlation_basis(img_xds.polarization.values)
            else float_dtype
        )
        return (
            np.zeros(
                (*leading_shape, img_xds.sizes["l"], img_xds.sizes["m"]),
                dtype=image_dtype,
            ),
            ("l", "m"),
            "point_spread_function" if uv_sampling else "sky",
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
        """:meth:`Gridder.grid` with the C++ ``dft_image``."""
        from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
            dft_image,
        )

        GridderDirectFourierTransform._call_kernel(
            dft_image,
            True,
            ms_xds,
            img_xds,
            weights,
            frequency_map,
            grid,
            normalization,
            ms_data_group_in_name,
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
        """:meth:`Gridder.grid_uv_sampling` with the C++ ``dft_image_uv_sampling``."""
        from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
            dft_image_uv_sampling,
        )

        GridderDirectFourierTransform._call_kernel(
            dft_image_uv_sampling,
            False,
            ms_xds,
            img_xds,
            weights,
            frequency_map,
            grid,
            normalization,
            ms_data_group_in_name,
            processing_function_threads,
        )

    @staticmethod
    def _call_kernel(
        kernel,
        with_correlated_data,
        ms_xds,
        img_xds,
        weights,
        frequency_map,
        grid,
        normalization,
        ms_data_group_in_name,
        processing_function_threads,
    ):
        """Call ``kernel``, the C++ transform of the visibilities or of the weights."""
        from astroviper.processing_functions.imaging.direct_fourier_transform import (
            _maps,
            image_direction_cosines,
        )

        ms_data_group_in = ms_xds.attrs["data_groups"][ms_data_group_in_name]
        time_map, pol_map = _maps(ms_xds, img_xds)
        # The bindings take the visibilities, uvw and weights without conversion.
        uvw = np.ascontiguousarray(
            ms_xds[ms_data_group_in["uvw"]].values, dtype=np.float64
        )
        weight = np.ascontiguousarray(weights, dtype=np.float64)
        frequency_coord = np.ascontiguousarray(
            ms_xds.frequency.values, dtype=np.float64
        )
        l, m = image_direction_cosines(img_xds)
        correlated_data = (
            (
                np.ascontiguousarray(
                    ms_xds[ms_data_group_in["correlated_data"]].values,
                    dtype=np.complex128,
                ),
            )
            if with_correlated_data
            else ()
        )
        kernel(
            grid,
            normalization,
            *correlated_data,
            uvw,
            frequency_coord,
            np.ascontiguousarray(frequency_map, dtype=np.int64),
            time_map,
            pol_map,
            weight,
            l,
            m,
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
        """:meth:`Gridder.degrid` with the C++ ``dft_predict``.

        It allocates a missing output variable as complex128 zeros, registers
        the output data group at every call and refuses an image whose
        polarization axis differs from the measurement set's.
        """
        from astroviper.processing_functions.imaging.direct_fourier_transform import (
            _maps,
            _model_visibilities,
            image_direction_cosines,
        )
        from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
            dft_predict,
        )
        from astroviper.utils.data_group_tools import create_data_groups_in_and_out

        if ms_data_group_out_modified is None:
            ms_data_group_out_modified = {"correlated_data": "VISIBILITY_MODEL"}
        ms_data_group_in, ms_data_group_out = create_data_groups_in_and_out(
            ms_xds,
            data_group_in_name=ms_data_group_in_name,
            data_group_out_name=ms_data_group_out_name,
            data_group_out_modified=dict(ms_data_group_out_modified),
            overwrite=overwrite,
        )
        time_map, pol_map = _maps(ms_xds, img_xds)
        model_visibilities = _model_visibilities(
            ms_xds,
            ms_data_group_in,
            ms_data_group_out,
            ms_data_group_out_name,
            description=description,
        )
        l, m = image_direction_cosines(img_xds)
        dft_predict(
            np.ascontiguousarray(grid),
            model_visibilities.values,
            np.ascontiguousarray(
                ms_xds[ms_data_group_in["uvw"]].values, dtype=np.float64
            ),
            np.ascontiguousarray(ms_xds.frequency.values, dtype=np.float64),
            np.ascontiguousarray(frequency_map, dtype=np.int64),
            time_map,
            pol_map,
            l,
            m,
            processing_function_threads=processing_function_threads,
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
        """:meth:`Gridder.uv_to_image` by :func:`~astroviper.processing_functions.imaging.direct_fourier_transform.normalize_dft_image`.

        The accumulated arrays are the images of the roles of
        ``image_data_group_out_modified`` in the output data group; they are
        divided by their normalization in place.
        """
        from astroviper.processing_functions.imaging.direct_fourier_transform import (
            normalize_dft_image,
        )

        for role in image_data_group_out_modified:
            normalize_dft_image(
                img_xds,
                image_data_group_out_name,
                uv_sampling=role == "point_spread_function",
            )
        return img_xds

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
        """:meth:`Gridder.image_to_uv` without a transform: :meth:`degrid` reads the images.

        An output data group of another name than the input group is
        registered with the images of the roles of ``data_variables_to_process``.
        """
        if image_data_group_out_name != image_data_group_in_name:
            from astroviper.utils.data_group_tools import modify_data_groups_xds

            data_group_in = img_xds.attrs["data_groups"][image_data_group_in_name]
            modify_data_groups_xds(
                img_xds,
                image_data_group_out_name,
                {
                    role: data_group_in[role]
                    for role in (data_variables_to_process or ["sky"])
                },
                description="Model images for the direct Fourier transform.",
            )
        return img_xds
