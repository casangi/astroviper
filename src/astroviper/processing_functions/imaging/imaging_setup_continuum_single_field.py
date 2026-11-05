import time

import numpy as np

from astroviper.utils.param_docs import shares_param_docs


def _get_reference_frequency(image_params):
    """Return ``image_params["reference_frequency"]`` in Hz. Distributed
    callers must supply one common reference frequency before partitioning.
    """
    from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
        check_reference_frequency,
    )

    check_reference_frequency(
        image_params["reference_frequency"], allow_one_element=True
    )
    return float(np.asarray(image_params["reference_frequency"]).reshape(()))


def _validate_n_taylor_terms(n_taylor_terms):
    """Return ``n_taylor_terms``, a Python or NumPy integer of at least 1, as an int."""
    from astroviper.processing_functions.imaging.config_check._common import (
        require_integer,
    )

    require_integer("n_taylor_terms", n_taylor_terms, minimum=1)
    return int(n_taylor_terms)


def _validate_continuum_parameters(image_params):
    """Validate and return ``(n_taylor_terms, reference_frequency)``."""
    from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
        check_continuum_basis,
    )

    check_continuum_basis(image_params)
    return int(image_params["n_taylor_terms"]), float(
        np.asarray(image_params["reference_frequency"]).reshape(())
    )


def _imaging_weights_are_available(
    ps_xdt,
    processing_set_data_group_name,
):
    """Return True when every selected dataset has accessible imaging weights."""
    datasets_checked = 0

    for ms_name, ms_xdt in ps_xdt.items():
        data_groups = ms_xdt.attrs.get("data_groups", {})

        if processing_set_data_group_name not in data_groups:
            raise KeyError(
                f"Data group {processing_set_data_group_name!r} is missing "
                f"from processing-set child {ms_name!r}."
            )

        datasets_checked += 1

        data_group = data_groups[processing_set_data_group_name]
        weight_name = data_group.get("weight_imaging")

        if weight_name is None or weight_name not in ms_xdt:
            return False

    if datasets_checked == 0:
        raise RuntimeError("No processing-set datasets were available for imaging.")

    return True


def _attach_continuum_metadata(
    img_xds,
    *,
    n_taylor_terms,
    reference_frequency,
    continuum_mode,
):
    """Attach validated Taylor-expansion metadata to the image dataset."""
    img_xds.attrs["continuum_imaging"] = {
        "continuum_mode": continuum_mode,
        "n_taylor_terms": int(n_taylor_terms),
        "reference_frequency": float(reference_frequency),
        "n_residual_taylor_terms": int(n_taylor_terms),
        "n_psf_taylor_terms": int(2 * n_taylor_terms - 1),
    }
    return img_xds


def _mfs_channel_imaging_weight_sums(
    ps_xdt,
    img_xds,
    processing_set_data_group_name,
):
    """Sum the imaging weights that contribute to each MFS image channel.

    The MFS primary-beam polynomial must use the same spectral weights as the
    apparent-sky Taylor fit. Imaging weights already encode visibility flags as
    ``NaN``; summing their finite values therefore produces CASA's ``s_nu``
    (``weightlist``) directly from the samples used by the gridder. Processing
    children and visibility channels that map to the same image channel are
    accumulated into one value per correlation.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Prepared processing-set partition. Auto-correlations must already have
        been removed and every child must contain registered imaging weights.
    img_xds : xarray.Dataset
        MFS image whose frequency and correlation coordinates define the output.
    processing_set_data_group_name : str
        Data group that registers ``weight_imaging`` in every processing child.

    Returns
    -------
    xarray.DataArray
        Imaging-weight sums with dimensions ``(frequency, polarization)``.
    """
    import xarray as xr

    from astroviper.processing_functions.imaging.utils.frequency_mapping import (
        map_visibility_frequencies_to_image,
    )

    image_frequencies = np.asarray(img_xds.frequency.values, dtype=np.float64)
    image_polarizations = np.asarray(img_xds.polarization.values).astype(str)
    weight_sums = np.zeros(
        (image_frequencies.size, image_polarizations.size), dtype=np.float64
    )

    for ms_name, ms_xdt in ps_xdt.items():
        data_groups = ms_xdt.attrs.get("data_groups", {})
        if processing_set_data_group_name not in data_groups:
            raise KeyError(
                f"Data group {processing_set_data_group_name!r} is missing "
                f"from processing-set child {ms_name!r}."
            )
        data_group = data_groups[processing_set_data_group_name]
        weight_name = data_group.get("weight_imaging")
        if weight_name is None or weight_name not in ms_xdt:
            raise KeyError(
                f"Imaging weights are missing from processing-set child {ms_name!r}."
            )

        weight = ms_xdt[weight_name]
        required_dims = {"time", "frequency", "polarization"}
        if weight.ndim != 4 or not required_dims.issubset(weight.dims):
            raise ValueError(
                f"Imaging weights in child {ms_name!r} must have time, baseline, "
                "frequency, and polarization dimensions; received "
                f"{weight.dims}."
            )
        baseline_dims = [dim for dim in weight.dims if dim not in required_dims]
        if len(baseline_dims) != 1:
            raise ValueError(
                f"Imaging weights in child {ms_name!r} must have exactly one "
                f"baseline dimension; received {weight.dims}."
            )
        weight = weight.transpose("time", baseline_dims[0], "frequency", "polarization")

        child_polarizations = np.asarray(weight.polarization.values).astype(str)
        polarization_map = []
        for polarization in child_polarizations:
            matches = np.flatnonzero(image_polarizations == polarization)
            if matches.size != 1:
                raise ValueError(
                    f"Polarization {polarization!r} from processing-set child "
                    f"{ms_name!r} does not match exactly one image polarization."
                )
            polarization_map.append(int(matches[0]))

        frequency_map = map_visibility_frequencies_to_image(
            weight.frequency.values,
            image_frequencies,
            matching="nearest",
        )
        values = np.asarray(weight.values)
        if np.any(np.isinf(values)):
            raise ValueError(
                f"Imaging weights in processing-set child {ms_name!r} contain "
                "infinite values."
            )
        per_visibility_channel = np.nansum(values, axis=(0, 1), dtype=np.float64)
        for child_pol, image_pol in enumerate(polarization_map):
            np.add.at(
                weight_sums[:, image_pol],
                frequency_map,
                per_visibility_channel[:, child_pol],
            )

    if not np.all(np.isfinite(weight_sums)):
        raise ValueError("MFS channel imaging-weight sums must be finite.")

    return xr.DataArray(
        weight_sums,
        dims=("frequency", "polarization"),
        coords={
            "frequency": img_xds.frequency,
            "polarization": img_xds.polarization,
        },
        name="MFS_CHANNEL_IMAGING_WEIGHT_SUM",
        attrs={
            "description": (
                "Post-flagging imaging-weight sum used by the MFS primary-beam "
                "Taylor fit."
            )
        },
    )


def _convert_primary_beam_to_reference_frequency(
    img_xds,
    *,
    image_params,
    reference_frequency,
    n_taylor_terms,
    float_dtype,
    image_data_group_in_name="residual",
    make_primary_beam_taylor=True,
    channel_imaging_weight_sums=None,
):
    """Build the static and polynomial primary-beam products needed by MFS.

    CASA's standard-gridder MFS path evaluates ``pb.tt0`` on the singleton
    Taylor image at its spectral coordinate, i.e. at the MT-MFS reference
    frequency.  The reference PB is independent of visibility partitioning and
    imaging weights, so it is retained as a non-additive static product.

    When ``make_primary_beam_taylor`` is true, the frequency-dependent beam is
    additionally projected onto the same Taylor basis and with the same
    post-flagging imaging-weight sums ``s_nu`` as the apparent sky. This is the
    CASA ``widebandpbcor`` weighted polynomial fit, with AstroViper deriving its
    ``weightlist`` directly from ``WEIGHT_IMAGING``. This task-local function
    stores only additive polynomial-fit numerators and spectral moments; the
    global append node solves the fit after all frequency partitions have been
    reduced. No channel-sized beam cube is carried through the distributed
    reduction. Runs without restoration skip that projection.

    ``n_taylor_terms`` and ``reference_frequency`` are the supplied Taylor basis
    already validated at the setup boundary. Do not infer a different order or
    repeat configuration validation while constructing these runtime products.
    """
    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam import (
        evaluate_primary_beam,
    )

    data_groups = img_xds.attrs.get("data_groups", {})

    if image_data_group_in_name not in data_groups:
        raise KeyError(f"Image data group {image_data_group_in_name!r} is missing.")

    image_data_group = data_groups[image_data_group_in_name]

    reference_frequency = float(reference_frequency)
    dish_diameters = image_params.get("list_dish_diameters")
    blockage_diameters = image_params.get("list_blockage_diameters")
    if dish_diameters is None:
        dish_diameters = np.asarray([10.7])
    if blockage_diameters is None:
        blockage_diameters = np.asarray([0.75])

    pb_image_params = {
        **image_params,
        "image_center": (np.asarray(image_params["image_size"]) // 2).tolist(),
    }
    pb_params = {
        "list_dish_diameters": np.asarray(dish_diameters),
        "list_blockage_diameters": np.asarray(blockage_diameters),
        "ipower": image_params.get("primary_beam_ipower", 2),
    }
    common_coords = {
        dim: img_xds.coords[dim]
        for dim in ("time", "polarization", "l", "m")
        if dim in img_xds.coords
    }
    if make_primary_beam_taylor:
        channel_frequencies = np.asarray(img_xds.frequency.values, dtype=np.float64)
        if channel_frequencies.ndim != 1 or channel_frequencies.size == 0:
            raise ValueError("MFS primary-beam fitting requires channel frequencies.")
        if not np.all(np.isfinite(channel_frequencies)):
            raise ValueError("MFS primary-beam frequencies must be finite.")
        if channel_imaging_weight_sums is None:
            raise ValueError(
                "channel_imaging_weight_sums is required when constructing the "
                "MFS primary-beam Taylor fit."
            )
        if set(channel_imaging_weight_sums.dims) != {
            "frequency",
            "polarization",
        }:
            raise ValueError(
                "channel_imaging_weight_sums must have frequency and polarization "
                "dimensions."
            )
        channel_imaging_weight_sums = channel_imaging_weight_sums.transpose(
            "frequency", "polarization"
        )
        if not np.array_equal(
            channel_imaging_weight_sums.frequency.values,
            img_xds.frequency.values,
        ):
            raise ValueError(
                "channel_imaging_weight_sums frequency coordinates must match the "
                "MFS image."
            )
        if not np.array_equal(
            channel_imaging_weight_sums.polarization.values,
            img_xds.polarization.values,
        ):
            raise ValueError(
                "channel_imaging_weight_sums polarization coordinates must match "
                "the MFS image."
            )
        spectral_weights = np.asarray(
            channel_imaging_weight_sums.values, dtype=np.float64
        )
        if not np.all(np.isfinite(spectral_weights)) or np.any(spectral_weights < 0):
            raise ValueError(
                "MFS channel imaging-weight sums must be finite and non-negative."
            )

        channel_data = evaluate_primary_beam(
            channel_frequencies,
            img_xds.polarization.values,
            pb_params,
            pb_image_params,
            dtype=float_dtype,
        )[0]
        channel_data = np.broadcast_to(
            channel_data,
            (img_xds.sizes["time"], *channel_data.shape),
        )
        x_frequency = (channel_frequencies - reference_frequency) / (
            reference_frequency
        )
        n_moments = 2 * n_taylor_terms - 1
        numerator = np.stack(
            [
                np.sum(
                    channel_data
                    * spectral_weights[None, :, :, None, None]
                    * x_frequency[None, :, None, None, None] ** order,
                    axis=1,
                )
                for order in range(n_taylor_terms)
            ],
            axis=1,
        )
        moments = np.stack(
            [
                np.sum(
                    spectral_weights * x_frequency[:, None] ** order,
                    axis=0,
                )
                for order in range(n_moments)
            ],
            axis=0,
        )
        moments = np.broadcast_to(
            moments[None, :, :],
            (img_xds.sizes["time"], n_moments, img_xds.sizes["polarization"]),
        ).copy()
        img_xds["MFS_PRIMARY_BEAM_TAYLOR_NUMERATOR"] = xr.DataArray(
            numerator,
            dims=("time", "taylor_term", "polarization", "l", "m"),
            coords={
                **common_coords,
                "taylor_term": np.arange(n_taylor_terms, dtype=np.int64),
            },
            attrs={
                "description": (
                    "Additive imaging-weighted MFS primary-beam Taylor-fit numerator."
                ),
                "reference_frequency": reference_frequency,
                "spectral_weighting": "post_flagging_imaging_weight_sum",
            },
        )
        img_xds["MFS_PRIMARY_BEAM_FIT_MOMENTS"] = xr.DataArray(
            moments,
            dims=("time", "psf_taylor_order", "polarization"),
            coords={
                key: value
                for key, value in common_coords.items()
                if key in ("time", "polarization")
            }
            | {"psf_taylor_order": np.arange(n_moments, dtype=np.int64)},
            attrs={
                "description": (
                    "Additive imaging-weighted MFS primary-beam Taylor-fit moments."
                ),
                "reference_frequency": reference_frequency,
                "spectral_weighting": "post_flagging_imaging_weight_sum",
            },
        )
    reference_data = evaluate_primary_beam(
        np.asarray([reference_frequency]),
        img_xds.polarization.values,
        pb_params,
        pb_image_params,
        dtype=float_dtype,
    )[0, 0, ...][None, ...]
    reference_data = np.broadcast_to(
        reference_data,
        (img_xds.sizes["time"], *reference_data.shape[1:]),
    )
    reference_primary_beam = xr.DataArray(
        reference_data,
        dims=("time", "polarization", "l", "m"),
        coords=common_coords,
        name="PRIMARY_BEAM_REFERENCE",
        attrs={
            "description": "MFS primary beam evaluated at the reference frequency.",
            "beam_model": image_params.get("primary_beam_model", "airy"),
            "dish_diameters_m": np.asarray(dish_diameters).tolist(),
            "blockage_diameters_m": np.asarray(blockage_diameters).tolist(),
            "type": "primary_beam",
            "method": "airy_disk",
            "continuum_mode": "mfs",
            "continuum_pb_order": 0,
            "reference_frequency": reference_frequency,
            "frequency_interpretation": "direct_reference_frequency_evaluation",
        },
    )

    img_xds["PRIMARY_BEAM_REFERENCE"] = reference_primary_beam

    image_data_group["primary_beam_reference"] = "PRIMARY_BEAM_REFERENCE"

    return img_xds


@shares_param_docs
def imaging_setup_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    continuum_mode="mfs",
    processing_set_data_group_name="corrected",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    image_data_group_out_name="residual",
    make_primary_beam_taylor=True,
    gridding_kernel_params=None,
    *,
    input_checked: bool = False,
):
    """Prepare static products for one continuum partition.

    Reuse prepared imaging weights or calculate local weights when absent,
    attach continuum metadata, and build the primary beam and UV sampling
    grids. MFS produces ``2*n_taylor_terms-1`` Taylor sampling planes; MVC
    retains frequency planes for map-local FFT and Taylor conversion. Residual
    calculation and global beam fitting follow in later stages.

    Parameters
    ----------
    continuum_mode : {"mfs", "mvc"}
        Continuum construction mode. ``"mfs"`` grids Taylor-weighted
        visibilities directly. Its ``SKY_MODEL`` Taylor coefficients describe
        the apparent sky, ``A_nu * I_nu``; consequently ``tt1 / tt0`` contains
        both the intrinsic source spectral index and the primary-beam spectral
        index. Wideband primary-beam correction fits the beam in the same Taylor
        basis and divides the apparent-sky polynomial by the beam polynomial
        before restoration, removing that contribution from corrected spectral
        products.
        ``"mvc"`` forms
        channel images and common-beam Taylor equations, and predicts
        ``M_nu * A_nu / Abar``. Its ``SKY_MODEL`` coefficients therefore describe
        the intrinsic spectrum under the common-beam convention, so
        ``tt1 / tt0`` excludes the primary-beam spectral index. These conventions
        match CASA's standard-gridder MT-MFS and MVC behavior. MVC images the two
        parallel hands only; ``["I", "Q", "U", "V"]`` is refused.
    ps_xdt : xarray.DataTree or mapping
        Processing-set partition containing the visibility data and previously
        prepared imaging weights.
    img_xds : xarray.Dataset
        Empty continuum image dataset. On return, the dataset contains the
        partition-local static imaging products.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``). The labels must be a
        nonempty one-dimensional vector. An xarray DataArray must have exactly
        the dimension ``polarization``; other dimension names are rejected.
        Continuum imaging also requires ``frequency_coords``,
        ``n_taylor_terms`` (CASA ``nterms``), and ``reference_frequency`` in Hz.
        ``spectral_reference``
        sets the spectral frame (default ``"lsrk"``). ``primary_beam_model``
        selects ``auto``, ``airy`` or ``casa_airy``; ``list_dish_diameters`` and
        ``list_blockage_diameters`` are arrays in metres,
        ``primary_beam_max_radius_1ghz`` is a radius in radians at 1 GHz, and
        ``primary_beam_ipower`` is the exponent applied to the voltage pattern
        (default 2, the power beam). These primary-beam keys override the inferred
        beam prescription.
    imaging_weights_params : dict
        Weighting configuration: ``weighting`` is ``"natural"``, ``"uniform"``,
        ``"briggs"`` or ``"briggs_abs"``. Both Briggs variants require an
        explicit finite real ``robust`` in ``[-2, 2]``; no default is inferred.
        ``weighting_scope`` is ``"local"`` for a density per
        frequency channel or ``"global"`` for one density across all selected
        channels in distributed continuum imaging. Natural weighting is
        independent of this scope.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the model
        update runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    image_data_group_out_name : str, optional
        Name of the continuum image data group. Defaults to ``"residual"``.
    make_primary_beam_taylor : bool, optional
        Whether MFS should evaluate the channel-dependent primary beam and form
        the additive Taylor-fit contributions used by restoration. Default true.
    gridding_kernel_params : dict, optional
        The gridding convolution kernel the gridder and degridder use:
        ``type`` (``"prolate_spheroidal_casa"``, Schwab's rational
        approximation of the ``m = 6``, ``alpha = 1`` function with its
        analytic correction, the kernel CASA uses; ``"prolate_spheroidal"``,
        the generalised prolate spheroidal function of Schwab 1984 with its
        bandwidth and weight exponent chosen for the image region the
        ``fft_padding`` keeps; ``"exponential_semicircle"``, the modified
        exponential of a semicircle of Barnett et al. 2019 and Arras et al.
        2021 shaped for that region; or
        ``"least_misfit"``, the table that minimises the map error over that
        region for the lookup used, after Ye et al. 2020), ``support`` (the
        odd number of grid points the kernel spans along one axis; 7 for the
        ``prolate_spheroidal_casa`` kernel, any odd number from 3 for the others),
        ``oversampling`` (samples of the kernel table per uv cell) and
        ``interpolation`` (how the gridders read the table at a stencil
        point: ``"nearest"`` takes the nearest sample, so the lookup error,
        and with it the floor the imaging cycle converges to, falls as 1 over
        the oversampling; ``"linear"`` interpolates the two neighbouring
        samples and the error falls as 1 over its square). Missing keys take
        the defaults ``{"type": "least_misfit", "support": 9,
        "oversampling": 300, "interpolation": "linear"}``, the kernel RADPS
        Memo 13a selects for kernel errors of at most 0.1 sigma at a dynamic
        range of 1e4 (its table for ``fft_padding`` 1.2 ships with the
        package, other paddings solve it at run time); ``None`` is all
        defaults. With the type ``"prolate_spheroidal_casa"`` missing keys
        take CASA's values instead, so that type alone is CASA's kernel and
        sampling, ``{"type": "prolate_spheroidal_casa", "support": 7,
        "oversampling": 100, "interpolation": "nearest"}``, with Schwab's
        analytic correction. A ``prolate_spheroidal``,
        ``exponential_semicircle`` or ``least_misfit`` kernel whose image
        plane correction falls below 1e-6 of its centre value at the edge of
        the image kept (a support too large for the ``fft_padding``, e.g. 17
        at 1.05) is refused. The kernel of the
        former ``gridder="prolate_spheroidal"`` is ``"prolate_spheroidal_casa"``,
        not ``"prolate_spheroidal"``. ``{"type": "dft"}`` (no other key) replaces
        the gridding and the FFTs by the exact direct Fourier transform over the
        unpadded image, w term included: with the phase
        ``2 pi (u l + v m + w (n - 1))`` and ``n = sqrt(1 - l**2 - m**2)``, the
        model visibilities are ``sum M exp(+i phase) / n`` over the model pixels
        and the image is ``n sum W V exp(-i phase) / sum W`` over the samples,
        the measurement equation of ``simulate_processing_set``. Its cost is
        proportional to the samples times the image pixels (30 to 40 s per
        residual update of 451,500 samples, four correlations and 160 x 160
        pixels on one core), so give it several
        ``processing_function_threads``; its results do not depend on their
        number. Its ``VISIBILITY_NORMALIZATION`` and
        ``UV_SAMPLING_NORMALIZATION`` hold the sum of the imaging weights, the
        gridders' that sum times the kernel's stencil sums.

    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

    Returns
    -------
    img_xds : xarray.Dataset
        Dataset containing MFS Taylor or MVC frequency UV-sampling products,
        the primary beam, and continuum metadata.
    return_df : pandas.DataFrame
        One-row timing dataframe summarizing the setup stage.

    """
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check import (
            CONTINUUM_SINGLE_FIELD,
            check_residual_update,
        )

        check_residual_update(locals(), mode=CONTINUUM_SINGLE_FIELD)

    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field import (
        make_point_spread_function_continuum_single_field,
        make_point_spread_function_mvc_single_field,
    )
    from astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam import (
        make_continuum_primary_beam_single_field,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    if single_precision_image:
        float_dtype = np.float32
        complex_dtype = np.complex64
    else:
        float_dtype = np.float64
        complex_dtype = np.complex128

    ps_data_group_name = processing_set_data_group_name

    n_taylor_terms = image_params["n_taylor_terms"]
    reference_frequency = image_params["reference_frequency"]

    img_xds = _attach_continuum_metadata(
        img_xds,
        n_taylor_terms=n_taylor_terms,
        reference_frequency=reference_frequency,
        continuum_mode=continuum_mode,
    )

    # Create the residual group up front. The local Taylor PSFs, primary beam,
    # and later residual Taylor products are registered under this group.
    data_groups = img_xds.attrs.setdefault("data_groups", {})

    if image_data_group_out_name not in data_groups:
        img_xds.attrs["type"] = "image_dataset"
        img_xds = img_xds.xr_img.add_data_group(
            new_data_group_name=image_data_group_out_name,
            new_data_group={
                "description": (
                    "Chunk-local continuum Taylor PSF and residual products."
                ),
            },
        )

        logger.debug("continuum img_xds size " + str(img_xds.nbytes / 1.0e9) + " GB")

    # -------------------------------------------------------------
    # Load imaging weights from memory or calculate them
    # -------------------------------------------------------------

    start = time.time()

    # Determine whether the weights are already available
    weights_available = _imaging_weights_are_available(
        ps_xdt,
        processing_set_data_group_name,
    )

    weights_were_calculated = False

    # if weights are not available, we compute them locally
    if not weights_available:
        from astroviper.processing_functions.imaging.calculate_imaging_weights import (
            calculate_imaging_weights,
        )

        # calculate weights per channel
        calculate_imaging_weights(
            ps_xdt,
            img_xds,
            imaging_weights_params=imaging_weights_params,
            return_weight_density_grid=False,
            ms_data_group_in_name=processing_set_data_group_name,
            ms_data_group_out_name=processing_set_data_group_name,
            ms_data_group_out_modified={
                "weight_imaging": "WEIGHT_IMAGING",
            },
            overwrite=True,
            processing_function_threads=processing_function_threads,
            frequency_matching="nearest",
            input_checked=True,
        )

        weights_were_calculated = True

    # if they are available, we just load the weights
    else:
        for ms_name, ms_xdt in ps_xdt.items():
            data_groups = ms_xdt.attrs.get("data_groups", {})

            if processing_set_data_group_name not in data_groups:
                raise KeyError(
                    f"Data group {processing_set_data_group_name!r} is missing "
                    f"from processing-set child {ms_name!r}."
                )

            weight_name = data_groups[processing_set_data_group_name].get(
                "weight_imaging"
            )

            if weight_name is None:
                raise KeyError(
                    f"Imaging weights have not been registered for "
                    f"processing-set child {ms_name!r}."
                )

            if weight_name not in ms_xdt:
                raise KeyError(
                    f"Registered imaging-weight variable {weight_name!r} is absent "
                    f"from processing-set child {ms_name!r}."
                )

    T_weights = time.time() - start

    # -------------------------------------------------------------
    # Chunk-local Taylor PSF/Hessian products
    # -------------------------------------------------------------
    start = time.time()

    if continuum_mode == "mfs":
        (
            img_xds,
            point_spread_function_return_df,
        ) = make_point_spread_function_continuum_single_field(
            ps_xdt,
            img_xds,
            image_params,
            n_taylor_terms=n_taylor_terms,
            reference_frequency=reference_frequency,
            ms_data_group_in_name=ps_data_group_name,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            image_data_variables_keep=image_data_variables_keep,
            gridding_kernel_params=gridding_kernel_params,
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            complex_dtype=complex_dtype,
            input_checked=True,
        )
    elif continuum_mode == "mvc":
        (
            img_xds,
            point_spread_function_return_df,
        ) = make_point_spread_function_mvc_single_field(
            ps_xdt,
            img_xds,
            image_params,
            ms_data_group_in_name=ps_data_group_name,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            gridding_kernel_params=gridding_kernel_params,
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
            input_checked=True,
        )

    T_make_point_spread_function = time.time() - start

    # -------------------------------------------------------------
    # Primary beam
    # -------------------------------------------------------------
    #
    if continuum_mode == "mfs":
        start = time.time()
        channel_imaging_weight_sums = None
        if make_primary_beam_taylor:
            channel_imaging_weight_sums = _mfs_channel_imaging_weight_sums(
                ps_xdt,
                img_xds,
                processing_set_data_group_name,
            )
        img_xds = _convert_primary_beam_to_reference_frequency(
            img_xds,
            image_params=image_params,
            reference_frequency=reference_frequency,
            n_taylor_terms=n_taylor_terms,
            float_dtype=float_dtype,
            image_data_group_in_name=image_data_group_out_name,
            make_primary_beam_taylor=make_primary_beam_taylor,
            channel_imaging_weight_sums=channel_imaging_weight_sums,
        )
        primary_beam_return_df = pd.DataFrame({"T_primary_beam": [time.time() - start]})
    else:
        (
            img_xds,
            primary_beam_return_df,
        ) = make_continuum_primary_beam_single_field(
            img_xds,
            image_params,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            list_dish_diameters=image_params.get("list_dish_diameters"),
            list_blockage_diameters=image_params.get("list_blockage_diameters"),
            ipower=image_params.get("primary_beam_ipower", 2),
            float_dtype=float_dtype,
        )
        # MVC requires the full channel-dependent PB cube.
        primary_beam_name = img_xds.attrs["data_groups"][image_data_group_out_name][
            "primary_beam"
        ]

        img_xds[primary_beam_name].attrs.update(
            {
                "description": (
                    "Partition-local frequency-dependent primary beam used by MVC."
                ),
                "continuum_mode": "mvc",
            }
        )

    T_primary_beam = float(primary_beam_return_df["T_primary_beam"].iloc[0])

    # -------------------------------------------------------------
    # Correlation -> Stokes
    # -------------------------------------------------------------

    # Deliberately no point_spread_function_gaussian_fit here. The correct beam
    # is fitted from the globally reduced zeroth-order Taylor PSF.

    return_df = pd.DataFrame(
        {
            "T_weights": [T_weights],
            "T_make_point_spread_function": [T_make_point_spread_function],
            "T_primary_beam": [T_primary_beam],
            "T_psf_fit": [0.0],
            "n_taylor_terms": [n_taylor_terms],
            "n_psf_taylor_terms": [2 * n_taylor_terms - 1],
            "reference_frequency": [reference_frequency],
            "imaging_weights_calculated": [weights_were_calculated],
        }
    )

    # Avoid duplicate columns between the setup-level summary and the
    # fine-grained PSF timing dataframe. Duplicate names cause
    # return_df[column] to return a DataFrame instead of a Series.
    overlapping_columns = return_df.columns.intersection(
        point_spread_function_return_df.columns
    )

    if len(overlapping_columns) > 0:
        point_spread_function_return_df = point_spread_function_return_df.drop(
            columns=list(overlapping_columns)
        )

    return_df = pd.concat(
        [return_df, point_spread_function_return_df],
        axis=1,
    )

    return img_xds, return_df
