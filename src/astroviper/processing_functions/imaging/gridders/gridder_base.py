"""The interface of the gridders.

A gridder grids the visibilities of one measurement set, degrids model
visibilities from a model and transforms between the uv grid and the image.
The imaging modes stay with the callers: they choose the weights and the
frequency map, and own the arrays the gridder accumulates into.
"""

from types import MappingProxyType

from astroviper.utils.interface import Interface
from astroviper.utils.param_docs import shares_param_docs


class Gridder(metaclass=Interface):
    """Base class of the gridders.

    An interface class (:class:`~astroviper.utils.interface.Interface`): the
    constants declare a gridder, and its static methods are called through
    the class. Here the methods raise ``NotImplementedError``.

    Attributes
    ----------
    name : str or None
        Registry key of the gridder.
    normalization : str or None
        What the image is divided by: ``"sum_of_weights"`` (the
        normalization array that :meth:`grid` accumulates) or
        ``"weight_image"`` (a weight image).
    gcf_terms : tuple of str
        The terms of ``gcf_xds`` the gridder applies.
    grid_roles : mapping of str to str
        The image data-group roles that :meth:`grid` writes, each with its
        default data-variable name: the accumulated array, then its
        normalization. Read-only.
    grid_uv_sampling_roles : mapping of str to str
        The same for :meth:`grid_uv_sampling`. Read-only.
    degrid_role : str or None
        The image data-group role of the model that :meth:`degrid` reads.
    """

    name = None
    normalization = None
    gcf_terms = ()
    grid_roles = MappingProxyType({})
    grid_uv_sampling_roles = MappingProxyType({})
    degrid_role = None

    @staticmethod
    def new_grid(img_xds, leading_shape, uv_sampling, fft_padding, complex_dtype):
        """The zero array that :meth:`grid` or :meth:`grid_uv_sampling` accumulates into.

        The caller adds it to ``img_xds`` with its leading dimensions (time,
        plane, polarization) followed by the two spatial dimensions returned
        here, gives it the ``type`` attribute returned here if there is one,
        and allocates the float64 normalization array of the leading shape.

        Parameters
        ----------
        img_xds : xarray.Dataset
            Image dataset; its ``l`` and ``m`` sizes are the image size.
        leading_shape : tuple of int
            Sizes of the leading dimensions (time, plane, polarization).
        uv_sampling : bool
            ``True`` for the array of :meth:`grid_uv_sampling`, ``False`` for
            that of :meth:`grid`.
        fft_padding : float
            Padding factor of the uv grid.
        complex_dtype : numpy.dtype
            Complex precision of the image (``numpy.complex64`` or
            ``numpy.complex128``).

        Returns
        -------
        values : numpy.ndarray
            Zeros, of shape ``leading_shape`` followed by the two spatial
            sizes.
        spatial_dims : tuple of str
            Names of the two spatial dimensions.
        variable_type : str or None
            The ``type`` attribute of the variable, or ``None`` for none.
        """
        raise NotImplementedError

    @staticmethod
    @shares_param_docs
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
        """Accumulate the weighted visibilities of one measurement set.

        Adds the visibilities, weighted by ``weights``, to ``grid`` and the
        sums that normalize the image to ``normalization``, both in place.
        The caller owns the two arrays (from :meth:`new_grid`, or views of
        them), and the weights and the frequency map, which are only read.
        Repeated calls accumulate.

        Parameters
        ----------
        ms_xds : xarray.Dataset
            Measurement set; the ``correlated_data`` and ``uvw`` roles of
            ``ms_data_group_in_name`` and the ``frequency`` coordinate are
            read.
        img_xds : xarray.Dataset
            Image dataset; its ``l`` and ``m`` coordinates give the image
            size and the cell size.
        gcf_xds : xarray.Dataset
            The ``gcf_xds`` of the gridding kernel, from
            :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.make_gridding_kernel`,
            or from
            :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.as_gridding_kernel`
            for a bare table of CASA's prolate spheroidal kernel of support 7.
        weights : numpy.ndarray
            float64, C-contiguous, shaped like the visibilities
            ``(time, baseline, frequency, polarization)``.
        frequency_map : numpy.ndarray
            Integer plane of ``grid`` of every visibility channel.
        grid : numpy.ndarray
            C-contiguous accumulation array ``(time, plane, polarization)``
            followed by the spatial dimensions of :meth:`new_grid`.
        normalization : numpy.ndarray
            float64, C-contiguous accumulation array
            ``(time, plane, polarization)``.
        ms_data_group_in_name : str
            Measurement-set data group whose roles are read.
        fft_padding : float
            Padding factor of the uv grid.
        processing_function_threads : int
            Number of threads handed to the per-processing-function (C++ / FFT)
            kernels. A distributed application resolves ``0`` (auto) once, before
            its checks, to the CPUs per Dask worker thread of the host with the
            fewest, at least 1 (1 where these are not known, as under MPI), and
            warns when a host's worker threads times the count exceed its CPUs
            (see
            :func:`~astroviper.utils.data_partitioning.resolve_processing_function_threads`).
            Node tasks and processing functions do not resolve ``0``; those that
            check the count refuse it. Given a count below 1, a C++ kernel and the
            PSF fit run one thread, and the Python FFTs pass the count to scipy.fft
            and pyfftw, whose scipy interfaces refuse ``0`` and read a negative
            ``n`` (down to ``-os.cpu_count()``) as ``os.cpu_count() + 1 + n``.
            numpy.fft takes no count: under ``"numpy"`` the Python FFTs run one
            thread.

        Returns
        -------
        None
        """
        raise NotImplementedError

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
        """Accumulate the weights of one measurement set: the uv sampling.

        As :meth:`grid`, with the same parameters, but the weights are
        accumulated in place of the weighted visibilities and the
        ``correlated_data`` role is not read; the image of ``grid`` is the
        point spread function.
        """
        raise NotImplementedError

    @staticmethod
    @shares_param_docs
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
        """Predict the model visibilities of one measurement set.

        Writes them in place into the variable that the ``correlated_data``
        role of the output data group names. When that variable is not in
        ``ms_xds`` it is allocated and the output data group registered;
        otherwise its samples that are not predicted keep their values. The
        caller owns ``grid`` and the frequency map, which are only read.

        Parameters
        ----------
        ms_xds : xarray.Dataset
            Measurement set, modified in place; the ``uvw`` role of
            ``ms_data_group_in_name`` and the ``frequency`` coordinate are
            read.
        img_xds : xarray.Dataset
            Image dataset; its ``l`` and ``m`` coordinates give the image
            size and the cell size.
        gcf_xds : xarray.Dataset
            The ``gcf_xds`` of the gridding kernel, from
            :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.make_gridding_kernel`,
            or from
            :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.as_gridding_kernel`
            for a bare table of CASA's prolate spheroidal kernel of support 7.
        grid : numpy.ndarray
            The model, ``(time, plane, polarization)`` followed by two
            spatial dimensions: the array of the role :attr:`degrid_role`.
        frequency_map : numpy.ndarray
            Integer plane of ``grid`` of every visibility channel.
        ms_data_group_in_name : str
            Measurement-set data group whose roles are read.
        ms_data_group_out_name : str
            Measurement-set data group of the model visibilities.
        ms_data_group_out_modified : dict or None
            Output role-to-variable mapping; ``None`` is
            ``{"correlated_data": "VISIBILITY_MODEL"}``.
        overwrite : bool
            Whether an existing output data group may be replaced.
        fft_padding : float
            Padding factor of the uv grid.
        processing_function_threads : int
            Number of threads handed to the per-processing-function (C++ / FFT)
            kernels. A distributed application resolves ``0`` (auto) once, before
            its checks, to the CPUs per Dask worker thread of the host with the
            fewest, at least 1 (1 where these are not known, as under MPI), and
            warns when a host's worker threads times the count exceed its CPUs
            (see
            :func:`~astroviper.utils.data_partitioning.resolve_processing_function_threads`).
            Node tasks and processing functions do not resolve ``0``; those that
            check the count refuse it. Given a count below 1, a C++ kernel and the
            PSF fit run one thread, and the Python FFTs pass the count to scipy.fft
            and pyfftw, whose scipy interfaces refuse ``0`` and read a negative
            ``n`` (down to ``-os.cpu_count()``) as ``os.cpu_count() + 1 + n``.
            numpy.fft takes no count: under ``"numpy"`` the Python FFTs run one
            thread.
        description : str
            Description of a newly registered output data group.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @staticmethod
    @shares_param_docs
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
        """Turn the accumulated arrays of ``img_xds`` into normalized images.

        Makes the images that ``image_data_group_out_modified`` names (roles
        ``"sky"`` and ``"point_spread_function"``) from the arrays of
        :meth:`grid` and :meth:`grid_uv_sampling` in the input data group, and
        registers the output data group. A gridder that accumulates into the
        images themselves (``"sky"`` in :attr:`grid_roles`) instead divides
        the images of those roles in the output data group by their
        normalization, in place, and registers no group.

        Parameters
        ----------
        img_xds : xarray.Dataset
            Image dataset, modified in place.
        image_params : dict
            Image geometry and output coordinates: ``image_size``, ``cell_size``,
            ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
            ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
            ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
            the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
            correlations (see ``instrument_polarization_basis``). The labels must be a
            nonempty one-dimensional vector. An xarray DataArray must have exactly
            the dimension ``polarization``; other dimension names are rejected.
            ``time_coords`` accepts XRADIO image-factory time formats: numeric MJD
            (UTC) days, numeric strings, ISO time strings, datetime64 values, Python
            datetime objects, or Astropy Time, as scalars or one-dimensional sequences.
            Times must be finite and nonempty; validation preserves the supplied values.
            Distributed single-field imagers require exactly one image time plane and
            an explicit ``fft_padding``. Designed kernels, including the default,
            require padding at least 1.05; CASA's fixed kernel permits 1.0.
        gcf_xds : xarray.Dataset
            The ``gcf_xds`` of the gridding kernel, from
            :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.make_gridding_kernel`,
            or from
            :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.as_gridding_kernel`
            for a bare table of CASA's prolate spheroidal kernel of support 7.
        image_data_group_in_name : str
            Image data group of the accumulated arrays; not read by a gridder
            that accumulates into the images.
        image_data_group_out_name : str
            Image data group of the images.
        image_data_group_out_modified : dict
            Role-to-variable mapping of the images to make.
        overwrite : bool
            Whether an existing output data group or variable may be replaced.
        image_data_variables_keep : list of str or None
            Roles of the uv grids kept in ``img_xds`` after the transform
            (``"visibility"``, ``"uv_sampling"``); the other grids transformed
            in this call are deleted from it. ``None`` keeps none.
        processing_function_threads : int
            Number of threads handed to the per-processing-function (C++ / FFT)
            kernels. A distributed application resolves ``0`` (auto) once, before
            its checks, to the CPUs per Dask worker thread of the host with the
            fewest, at least 1 (1 where these are not known, as under MPI), and
            warns when a host's worker threads times the count exceed its CPUs
            (see
            :func:`~astroviper.utils.data_partitioning.resolve_processing_function_threads`).
            Node tasks and processing functions do not resolve ``0``; those that
            check the count refuse it. Given a count below 1, a C++ kernel and the
            PSF fit run one thread, and the Python FFTs pass the count to scipy.fft
            and pyfftw, whose scipy interfaces refuse ``0`` and read a negative
            ``n`` (down to ``-os.cpu_count()``) as ``os.cpu_count() + 1 + n``.
            numpy.fft takes no count: under ``"numpy"`` the Python FFTs run one
            thread.
        fft_backend : str
            FFT library: ``"numpy"`` (``numpy.fft``), ``"scipy"`` (``scipy.fft``) or
            ``"pyfftw"`` (FFTW through pyFFTW). The three round differently; the C++
            kernels' FFTs do not use it.
        complex_dtype : numpy.dtype
            Complex precision of the image (``numpy.complex64`` or
            ``numpy.complex128``).

        Returns
        -------
        xarray.Dataset
            ``img_xds``.
        """
        raise NotImplementedError

    @staticmethod
    @shares_param_docs
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
        """Turn the model images of ``img_xds`` into the models :meth:`degrid` reads.

        Makes the arrays that ``image_data_group_out_modified`` names from the
        images of the roles in ``data_variables_to_process``, and registers
        the output data group. A gridder that degrids from the images
        themselves (:attr:`degrid_role` ``"sky"``) makes no arrays: it
        registers those images under the output data group when its name
        differs from that of the input data group.

        Parameters
        ----------
        img_xds : xarray.Dataset
            Image dataset, modified in place.
        image_params : dict
            Image geometry and output coordinates: ``image_size``, ``cell_size``,
            ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
            ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
            ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
            the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
            correlations (see ``instrument_polarization_basis``). The labels must be a
            nonempty one-dimensional vector. An xarray DataArray must have exactly
            the dimension ``polarization``; other dimension names are rejected.
            ``time_coords`` accepts XRADIO image-factory time formats: numeric MJD
            (UTC) days, numeric strings, ISO time strings, datetime64 values, Python
            datetime objects, or Astropy Time, as scalars or one-dimensional sequences.
            Times must be finite and nonempty; validation preserves the supplied values.
            Distributed single-field imagers require exactly one image time plane and
            an explicit ``fft_padding``. Designed kernels, including the default,
            require padding at least 1.05; CASA's fixed kernel permits 1.0.
        gcf_xds : xarray.Dataset
            The ``gcf_xds`` of the gridding kernel, from
            :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.make_gridding_kernel`,
            or from
            :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.as_gridding_kernel`
            for a bare table of CASA's prolate spheroidal kernel of support 7.
        plane_dim : {"frequency", "taylor_term"}
            The plane dimension of the images.
        image_data_group_in_name : str
            Image data group of the model images.
        image_data_group_out_name : str
            Image data group the models are registered under.
        image_data_group_out_modified : dict or None
            Role-to-variable mapping of the models to make.
        overwrite : bool
            Whether an existing output data group or variable may be replaced.
        image_data_variables_keep : list of str or None
            Roles of the images kept in ``img_xds`` after the transform
            (``"sky"``); the other images transformed in this call are deleted
            from it. ``None`` keeps none.
        processing_function_threads : int
            Number of threads handed to the per-processing-function (C++ / FFT)
            kernels. A distributed application resolves ``0`` (auto) once, before
            its checks, to the CPUs per Dask worker thread of the host with the
            fewest, at least 1 (1 where these are not known, as under MPI), and
            warns when a host's worker threads times the count exceed its CPUs
            (see
            :func:`~astroviper.utils.data_partitioning.resolve_processing_function_threads`).
            Node tasks and processing functions do not resolve ``0``; those that
            check the count refuse it. Given a count below 1, a C++ kernel and the
            PSF fit run one thread, and the Python FFTs pass the count to scipy.fft
            and pyfftw, whose scipy interfaces refuse ``0`` and read a negative
            ``n`` (down to ``-os.cpu_count()``) as ``os.cpu_count() + 1 + n``.
            numpy.fft takes no count: under ``"numpy"`` the Python FFTs run one
            thread.
        fft_backend : str
            FFT library: ``"numpy"`` (``numpy.fft``), ``"scipy"`` (``scipy.fft``) or
            ``"pyfftw"`` (FFTW through pyFFTW). The three round differently; the C++
            kernels' FFTs do not use it.
        data_variables_to_process : list of str or None
            Roles of the images to transform; ``None`` is ``["sky"]``.
        complex_dtype : numpy.dtype
            Complex precision of the image (``numpy.complex64`` or
            ``numpy.complex128``).
        new_polarization_basis : {"linear", "circular"} or None
            The basis whose correlations the Stokes planes of the images are
            combined into; ``None`` transforms the planes as they are.

        Returns
        -------
        xarray.Dataset
            ``img_xds``.
        """
        raise NotImplementedError
