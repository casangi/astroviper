"""The interface of the deconvolvers.

A deconvolver runs the model update, which cleans the residual into the model
in place, and gives the residual that the restore adds to the restored model.
"""

from astroviper.utils.interface import Interface


class Deconvolver(metaclass=Interface):
    """Base class of the deconvolvers.

    An interface class (:class:`~astroviper.utils.interface.Interface`): the
    constants declare a deconvolver, and its static methods are called through
    the class. Here the methods raise ``NotImplementedError``.

    Attributes
    ----------
    name : str or None
        Registry key of the deconvolver.
    model_basis : str or None
        What the model holds: ``"pixels"`` (an image per plane) or
        ``"taylor_terms"`` (the Taylor terms of a continuum image).
    imaging_kinds : tuple of str
        The imaging kinds that use the deconvolver: ``"cube"``,
        ``"continuum"`` or both.
    """

    name = None
    model_basis = None
    imaging_kinds = ()

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
        """Run one model update: clean the residual into the model, in place.

        The arguments and the return value follow :attr:`model_basis`: those
        listed here are for ``"pixels"``. A deconvolver of ``"taylor_terms"``
        takes the image dataset instead and returns its record and timing
        frame (see
        :meth:`~astroviper.processing_functions.imaging.deconvolvers.mtmfs.deconvolver_mtmfs.DeconvolverMtmfs.deconvolve`).

        Parameters
        ----------
        residual_cube : numpy.ndarray
            Residual ``(time, frequency, polarization, l, m)``, updated in
            place.
        psf_cube : numpy.ndarray
            Point spread function of the residual's ``(time, frequency)``
            planes, with the residual's polarizations or one.
        model_cube : numpy.ndarray
            Model, shaped like the residual; the components are added to it
            in place.
        deconvolve_params : dict or None, optional
            Iteration controls.
        processing_function_threads : int, optional
            Threads of the C++ kernel.
        mask_cube : numpy.ndarray or None, optional
            Clean mask, shaped like the residual; ``None`` for none.
        input_checked : bool, optional
            Skip the configuration check, which the caller has made.

        Returns
        -------
        dict
            The results of the kernel, arrays over
            ``(time, frequency, polarization)``.
        """
        raise NotImplementedError

    @staticmethod
    def prepare_restoration(residual_da, psf_da):
        """Give the residual that the restore adds to the restored model.

        Parameters
        ----------
        residual_da : xarray.DataArray
            The residual of the model updates; not changed.
        psf_da : xarray.DataArray
            Its point spread function.

        Returns
        -------
        xarray.DataArray
            The residual to restore.
        """
        raise NotImplementedError
