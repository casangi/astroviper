"""Multi-term multi-frequency synthesis (MT-MFS) CLEAN of the Taylor terms of a continuum image."""

from astroviper.processing_functions.imaging.model_update.deconvolvers.deconvolver_base import (
    Deconvolver,
)


class DeconvolverMtmfs(Deconvolver):
    """MT-MFS CLEAN; its methods follow the contracts of :class:`Deconvolver`.

    Its model basis is ``"taylor_terms"``: :meth:`deconvolve` takes the image
    dataset.
    """

    name = "mtmfs"
    model_basis = "taylor_terms"
    imaging_kinds = ("continuum",)

    @staticmethod
    def deconvolve(
        img_xds,
        deconvolve_params,
        residual_name,
        psf_name,
        max_sidelobe_name,
        image_data_group_in_name,
        image_data_group_out_name,
        processing_function_threads=1,
        *,
        input_checked=False,
    ):
        """:meth:`Deconvolver.deconvolve` on the Taylor terms of ``img_xds``.

        The MT-MFS model update of
        :func:`~astroviper.processing_functions.imaging.image_continuum_single_field.model_update_mtmfs_single_field`,
        by ``_model_update_native_mtmfs`` of that module.

        Parameters
        ----------
        img_xds : xarray.Dataset
            Continuum image dataset; its residual Taylor terms are cleaned in
            place and the components are added to ``SKY_MODEL`` (zeros
            without one).
        deconvolve_params : dict
            Iteration controls; per-plane arrays have the shape
            ``(time, 1, polarization)``.
        residual_name, psf_name, max_sidelobe_name : str
            The variables of the residual, of the point spread function and
            of its largest sidelobe.
        image_data_group_in_name : str
            Input data group.
        image_data_group_out_name : str
            Data group of the model.
        processing_function_threads : int, optional
            Threads of the C++ kernel's FFTs.
        input_checked : bool, optional
            Skip the configuration check, which the caller has made.

        Returns
        -------
        imaging_dict : dict
            The record of the model update, keyed by ``Key(time, pol, chan)``.
        timing : pandas.DataFrame
            ``T_mtmfs`` and ``T_model_update_cycle``.
        """
        from astroviper.processing_functions.imaging import (
            image_continuum_single_field,
        )

        return image_continuum_single_field._model_update_native_mtmfs(
            img_xds,
            deconvolve_params,
            residual_name,
            psf_name,
            max_sidelobe_name,
            image_data_group_in_name,
            image_data_group_out_name,
            processing_function_threads,
            input_checked=input_checked,
        )

    @staticmethod
    def prepare_restoration(residual_da, psf_da):
        """:meth:`Deconvolver.prepare_restoration`: the principal solution of the Taylor residual.

        A copy, made by ``_mtmfs_principal_residual_for_restoration`` of
        :mod:`~astroviper.processing_functions.imaging.image_continuum_single_field`.
        """
        from astroviper.processing_functions.imaging import (
            image_continuum_single_field,
        )

        return image_continuum_single_field._mtmfs_principal_residual_for_restoration(
            residual_da, psf_da
        )
