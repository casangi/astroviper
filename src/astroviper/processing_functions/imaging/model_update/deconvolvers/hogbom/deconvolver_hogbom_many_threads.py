"""Hogbom CLEAN, with threads across and within the image planes."""

from astroviper.processing_functions.imaging.model_update.deconvolvers.deconvolver_base import (
    Deconvolver,
)


class DeconvolverHogbomManyThreads(Deconvolver):
    """Hogbom CLEAN with threads within the planes; its methods follow the contracts of :class:`Deconvolver`."""

    name = "hogbom_many_threads"
    model_basis = "pixels"
    imaging_kinds = ("cube",)

    @staticmethod
    def deconvolve(
        residual_cube,
        psf_cube,
        model_cube,
        deconvolve_params=None,
        processing_function_threads=1,
        mask_cube=None,
        *,
        input_checked=False,
    ):
        """:meth:`Deconvolver.deconvolve` by :func:`~astroviper.processing_functions.imaging.deconvolution.hogbom_clean_many_threads`."""
        from astroviper.processing_functions.imaging import deconvolution

        return deconvolution.hogbom_clean_many_threads(
            residual_cube=residual_cube,
            psf_cube=psf_cube,
            model_cube=model_cube,
            deconvolve_params=deconvolve_params,
            processing_function_threads=processing_function_threads,
            mask_cube=mask_cube,
            input_checked=input_checked,
        )

    @staticmethod
    def prepare_restoration(residual_da, psf_da):
        """:meth:`Deconvolver.prepare_restoration`: the residual itself."""
        return residual_da
