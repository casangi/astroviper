"""Adaptive Scale Pixel (Asp) CLEAN."""

from astroviper.processing_functions.imaging.model_update.deconvolvers.deconvolver_base import (
    Deconvolver,
)


class DeconvolverAsp(Deconvolver):
    """Asp CLEAN; its methods follow the contracts of :class:`Deconvolver`."""

    name = "asp"
    model_basis = "pixels"
    imaging_kinds = ("cube", "continuum")

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
        """:meth:`Deconvolver.deconvolve` by :func:`~astroviper.processing_functions.imaging.deconvolution.asp_clean`."""
        from astroviper.processing_functions.imaging import deconvolution

        return deconvolution.asp_clean(
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
