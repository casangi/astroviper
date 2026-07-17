import copy
import logging
from typing import Optional, Tuple

import numpy as np
import toolviper.utils.logger as logger
import xarray as xr

from astroviper.processing_functions.image_analysis import image_statistics as imgstats
from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
    extract_main_lobe,
)
from astroviper.processing_functions.imaging.deconvolvers import aspclean, hogbom
from astroviper.processing_functions.imaging.utils.return_dict import ReturnDict
from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)

# lg = logger.get_logger()
# lg.setLevel(logging.DEBUG)

# XXX : TODO: As of 2025-10-07 there is no way to supply an initial model image to the deconvolver


def get_phase_center(residual_image_xds):
    """
    Get the phase center from the residual image coordinates.

    Parameters
    ----------
    residual_image_xds : xarray.Dataset
        Residual image dataset with 2-D ``right_ascension`` and
        ``declination`` coordinate arrays.

    Returns
    -------
    phase_center : str
        The phase center in the format ``"RA,Dec"`` (e.g.
        ``"12.345,-67.890"``), taken from the central pixel of the
        coordinate grids.
    """
    ra_shape = residual_image_xds.coords["right_ascension"].shape
    dec_shape = residual_image_xds.coords["declination"].shape

    cx_ra, cy_ra = ra_shape[0] // 2, ra_shape[1] // 2
    cx_dec, cy_dec = dec_shape[0] // 2, dec_shape[1] // 2

    ra0 = residual_image_xds.coords["right_ascension"].values[cx_ra, cy_ra]
    dec0 = residual_image_xds.coords["declination"].values[cx_dec, cy_dec]
    return f"{ra0},{dec0}"


def progress_callback(
    iter_num: int,
    px: int,
    py: int,
    peak: float,
    niter_log: int = 100,
):
    """
    Log CLEAN progress at a fixed iteration cadence.

    Parameters
    ----------
    iter_num : int
        Current iteration number.
    px : int
        X-coordinate of the current peak.
    py : int
        Y-coordinate of the current peak.
    peak : float
        Value of the current peak.
    niter_log : int, optional
        Frequency (in iterations) at which to emit log messages.
        Default is 100.
    """
    if iter_num % niter_log == 0:
        logger.debug(f"  Iteration {iter_num}, peak at ({px}, {py}): {peak:.6f}")


def _validate_deconvolve_params(deconvolve_params):
    """
    Validate a deconvolution parameter dict and fill in defaults.

    Parameters
    ----------
    deconvolve_params : dict or None
        Parameter dict; missing keys are populated with defaults. If
        ``None``, a fresh dict of defaults is returned.

        Supported keys

        - ``gain`` : float, CLEAN loop gain in ``(0, 1]``. Default 0.1.
        - ``niter`` : int, maximum number of iterations. Default 1000.
        - ``threshold`` : float, stopping threshold, non-negative.
          Default 0.0.
        - ``primary_beam_limit`` : float in ``[0, 1]``, primary-beam mask
          cutoff as a fraction of the peak primary beam (used to build the
          deconvolution mask, *not* the deconvolver stopping threshold).
          Default 0.0.
        - ``clean_box`` : 4-tuple ``(xmin, xmax, ymin, ymax)``. Default
          ``(-1, -1, -1, -1)`` meaning the full image.

    Returns
    -------
    dict
        The validated parameter dict, with defaults applied.

    Raises
    ------
    ValueError
        If any supplied value is outside its allowed range.
    """
    # NOTE : XXX : This should probably not live here. This validation
    # function should live in utils or something.

    if deconvolve_params is None:
        deconvolve_params = {}

    default_params = {
        "gain": 0.1,
        "niter": 1000,
        "threshold": 0.0,
        "primary_beam_limit": 0.0,
        "clean_box": (-1, -1, -1, -1),
        "minpsffraction": 0.05,
        "maxpsffraction": 0.8,
    }

    for key, default_value in default_params.items():
        if key not in deconvolve_params:
            logger.debug(
                f"Deconvolution parameter '{key}' not specified. Using default: {default_value}"
            )
            deconvolve_params[key] = default_value
            continue

        value = deconvolve_params[key]
        if key == "gain":
            if not (0 < value <= 1):
                raise ValueError("CLEAN gain must be between 0 and 1.")
        elif key == "niter":
            if not (isinstance(value, int) and value > 0):
                raise ValueError(
                    "Maximum number of iterations must be a positive integer."
                )
        elif key == "threshold":
            if value is not None and value < 0:
                raise ValueError("Threshold must be non-negative or None.")
        elif key == "clean_box":
            if value is not None and not (isinstance(value, tuple) and len(value) == 4):
                raise ValueError(
                    "Clean box must be a 4-tuple (xmin, xmax, ymin, ymax) or None."
                )
        elif key in ("minpsffraction", "maxpsffraction", "primary_beam_limit"):
            if not (0 <= value <= 1):
                raise ValueError(f"{key} must be between 0 and 1.")

    return deconvolve_params


def _per_plane_iteration_controls(deconvolve_params, nt, nf, npol, threshold_dtype):
    """Build per-plane ``niter`` and ``cyclethreshold`` cubes of shape ``(nt, nf, npol)``.

    Iteration control is performed independently for every
    ``(time, frequency, polarization)`` plane, so the deconvolvers are driven
    with a per-plane iteration limit and a per-plane cyclethreshold rather than
    single scalars. The per-plane values are taken from
    ``deconvolve_params['niter_per_plane']`` and
    ``deconvolve_params['cyclethreshold_per_plane']`` when present (these are
    supplied by the :class:`IterationController`, indexed
    ``(time, frequency=chan, polarization)``). Otherwise the scalar fallbacks
    are broadcast across all planes for standalone / backward-compatible calls:
    ``niter`` for the iteration limit, and the representative ``cyclethreshold``
    (or, failing that, the absolute ``threshold``) for the stopping threshold.

    Parameters
    ----------
    deconvolve_params : dict
        Validated parameter dict. May carry ``niter_per_plane`` /
        ``cyclethreshold_per_plane`` ``(nt, nf, npol)`` arrays.
    nt, nf, npol : int
        Cube dimensions (time, frequency, polarization).
    threshold_dtype : numpy dtype
        Dtype for the returned cyclethreshold cube (image dtype for Hogbom,
        ``float64`` for Asp).

    Returns
    -------
    niter_cube : numpy.ndarray
        ``(nt, nf, npol)`` int32 per-plane iteration limits.
    cyclethreshold_cube : numpy.ndarray
        ``(nt, nf, npol)`` per-plane cyclethresholds in ``threshold_dtype``.
    """
    shape = (nt, nf, npol)

    niter_pp = deconvolve_params.get("niter_per_plane", None)
    if niter_pp is None:
        niter_cube = np.full(shape, int(deconvolve_params["niter"]), dtype=np.int32)
    else:
        niter_cube = np.ascontiguousarray(niter_pp, dtype=np.int32)
        if niter_cube.shape != shape:
            raise ValueError(
                f"niter_per_plane shape {niter_cube.shape} does not match the "
                f"image plane grid {shape}"
            )

    cyclethreshold_pp = deconvolve_params.get("cyclethreshold_per_plane", None)
    if cyclethreshold_pp is None:
        scalar = deconvolve_params.get(
            "cyclethreshold", deconvolve_params.get("threshold", 0.0)
        )
        scalar = 0.0 if scalar is None else float(scalar)
        cyclethreshold_cube = np.full(shape, scalar, dtype=threshold_dtype)
    else:
        cyclethreshold_cube = np.ascontiguousarray(
            cyclethreshold_pp, dtype=threshold_dtype
        )
        if cyclethreshold_cube.shape != shape:
            raise ValueError(
                f"cyclethreshold_per_plane shape {cyclethreshold_cube.shape} does "
                f"not match the image plane grid {shape}"
            )

    return niter_cube, cyclethreshold_cube


def _plane_peak_abs_signed(arr, mask=None):
    """
    Return the signed value at the absolute-value peak of a 2-D array,
    optionally restricted to pixels where ``mask > 0.5``.

    Parameters
    ----------
    arr : numpy.ndarray
        2-D image plane.
    mask : numpy.ndarray, optional
        2-D mask of the same shape as ``arr``. If provided, only pixels
        where ``mask > 0.5`` are considered.

    Returns
    -------
    float
        Signed value of ``arr`` at its absolute-value maximum within the
        mask. NaN if every pixel is masked.
    """
    if mask is None:
        absvals = np.abs(arr)
    else:
        valid = mask > 0.5
        if not np.any(valid):
            return float("nan")
        absvals = np.where(valid, np.abs(arr), np.nan)
    if np.all(np.isnan(absvals)):
        return float("nan")
    # Return the signed value at the absolute-value peak: locate the largest
    # magnitude with nanargmax, then return arr at that index so a strong
    # negative residual keeps its sign (as the name/docstring promise).
    idx = np.unravel_index(np.nanargmax(absvals), absvals.shape)
    return float(arr[idx])


def starting_statistics(
    img_xds: xr.Dataset,
    image_data_group_in_name: str,
    image_data_group_out_name: str,
    zero_model: bool,
):
    """
    Compute per-plane starting peak residuals and model fluxes before
    CLEAN runs.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset providing the residual via the input data group
        and the (possibly zero) starting model via the modified output
        data group.
    image_data_group_in_name : str
        Name of the input data group whose ``"sky"`` key resolves to the
        residual variable and ``"mask"`` to the optional mask variable.
    image_data_group_out_name : str
        Name of the modified output data group whose ``"sky"`` key
        resolves to the starting model variable.
    zero_model : bool
        If ``True``, the starting model is assumed to be all zeros and
        ``start_model_flux`` is filled with zeros without reading
        ``model_arr``.

    Returns
    -------
    dict
        Dictionary with keys ``start_peakres``, ``start_peakres_nomask``,
        and ``start_model_flux`` — each a ``(nt, nf, np)`` float64 array.
        Suitable for merging into the ``selected_calculations`` argument
        of :func:`create_deconvolution_return_dict`.
    """
    residual_data_group = img_xds.attrs["data_groups"][image_data_group_in_name]
    model_data_group = img_xds.attrs["data_groups"][image_data_group_out_name]

    residual_arr = img_xds[residual_data_group["sky"]].values
    model_arr = img_xds[model_data_group["sky"]].values

    mask_name = residual_data_group.get("mask", None)
    mask_arr = img_xds[mask_name].values if mask_name in img_xds else None

    ntime = img_xds.sizes["time"]
    nchan = img_xds.sizes["frequency"]
    npol = img_xds.sizes["polarization"]

    start_peakres = np.empty((ntime, nchan, npol), dtype=np.float64)
    start_peakres_nomask = np.empty((ntime, nchan, npol), dtype=np.float64)
    start_model_flux = np.empty((ntime, nchan, npol), dtype=np.float64)

    for tt in range(ntime):
        for nn in range(nchan):
            for pp in range(npol):
                rp = residual_arr[tt, nn, pp]
                mp = mask_arr[tt, nn, pp] if mask_arr is not None else None
                start_peakres[tt, nn, pp] = _plane_peak_abs_signed(rp, mask=mp)
                start_peakres_nomask[tt, nn, pp] = _plane_peak_abs_signed(rp)
                start_model_flux[tt, nn, pp] = (
                    0.0 if zero_model else float(model_arr[tt, nn, pp].sum())
                )

    return {
        "start_peakres": start_peakres,
        "start_peakres_nomask": start_peakres_nomask,
        "start_model_flux": start_model_flux,
    }


def create_deconvolution_return_dict(
    img_xds: xr.Dataset,
    image_data_group_in_name: str,
    image_data_group_out_name: str,
    deconvolve_params: dict,
    selected_calculations: dict,
):
    """
    Build the per-plane :class:`ReturnDict` summarizing a CLEAN run.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset providing the residual via the input data group
        and the model and (optional) mask via the modified output data
        group, plus the ``time``, ``frequency``, and ``polarization``
        coordinates.
    image_data_group_in_name : str
        Name of the input data group whose ``"sky"`` key resolves to the
        post-CLEAN residual variable and ``"mask"`` to the optional mask.
    image_data_group_out_name : str
        Name of the modified output data group whose ``"sky"`` key
        resolves to the post-CLEAN model variable.
    deconvolve_params : dict
        Validated deconvolution parameter dict; ``niter`` and ``gain`` are
        recorded per plane, along with the per-plane ``cyclethreshold`` that
        each plane was cleaned to (from ``cyclethreshold_per_plane`` when
        present, else the scalar ``cyclethreshold`` / ``threshold``).
    selected_calculations : dict
        Precomputed inputs for the return dict. Required keys:

        - ``iters`` : ``(nt, nf, np)`` int array of iterations performed.
        - ``start_peakres`` : ``(nt, nf, np)`` masked starting peaks.
        - ``start_peakres_nomask`` : ``(nt, nf, np)`` unmasked starting peaks.
        - ``start_model_flux`` : ``(nt, nf, np)`` starting model fluxes.
        - ``min_psf_fraction``, ``max_psf_fraction``, ``max_psf_sidelobe``
        - ``masksum``

    Returns
    -------
    ReturnDict
        Per-plane deconvolution statistics indexed by
        ``(time, chan, pol)``.
    """
    residual_data_group = img_xds.attrs["data_groups"][image_data_group_in_name]
    model_data_group = img_xds.attrs["data_groups"][image_data_group_out_name]

    residual_arr = img_xds[residual_data_group["sky"]].values
    model_arr = img_xds[model_data_group["sky"]].values

    mask_name = residual_data_group.get("mask", None)
    mask_arr = img_xds[mask_name].values if mask_name in img_xds else None

    ntime = img_xds.sizes["time"]
    nchan = img_xds.sizes["frequency"]
    npol = img_xds.sizes["polarization"]

    pol_vals = img_xds.coords["polarization"].values
    freq_vals = img_xds.coords["frequency"].values
    time_vals = img_xds.coords["time"].values

    iters = selected_calculations["iters"]
    start_peakres = selected_calculations["start_peakres"]
    start_peakres_nomask = selected_calculations["start_peakres_nomask"]
    start_model_flux = selected_calculations["start_model_flux"]
    min_psf_fraction = selected_calculations["min_psf_fraction"]
    max_psf_fraction = selected_calculations["max_psf_fraction"]
    max_psf_sidelobe = selected_calculations["max_psf_sidelobe"]
    masksum = selected_calculations["masksum"]

    # Record the per-plane cyclethreshold actually used to clean each plane.
    # When driven by the iteration controller this is the
    # ``cyclethreshold_per_plane`` array; standalone callers fall back to the
    # representative scalar ``cyclethreshold`` (or the absolute ``threshold``).
    cyclethreshold_pp = deconvolve_params.get("cyclethreshold_per_plane", None)
    cyclethreshold_scalar = deconvolve_params.get(
        "cyclethreshold", deconvolve_params.get("threshold", None)
    )

    returndict = ReturnDict()

    for tt in range(ntime):
        for nn in range(nchan):
            for pp in range(npol):
                rp = residual_arr[tt, nn, pp]
                mp = mask_arr[tt, nn, pp] if mask_arr is not None else None
                peakres = _plane_peak_abs_signed(rp, mask=mp)
                peakres_nomask = _plane_peak_abs_signed(rp)
                model_flux = float(model_arr[tt, nn, pp].sum())
                cyclethreshold = (
                    float(cyclethreshold_pp[tt, nn, pp])
                    if cyclethreshold_pp is not None
                    else cyclethreshold_scalar
                )

                returnvals = {
                    "niter": deconvolve_params.get("niter", None),
                    "cyclethreshold": cyclethreshold,
                    "iter_done": int(iters[tt, nn, pp]),
                    "loop_gain": deconvolve_params.get("gain", None),
                    "min_psf_fraction": min_psf_fraction,
                    "max_psf_fraction": max_psf_fraction,
                    "max_psf_sidelobe": max_psf_sidelobe[tt, nn, pp],
                    "stop_code": None,
                    "stokes": pol_vals[pp],
                    "frequency": freq_vals[nn],
                    "time": time_vals[tt],
                    "start_model_flux": start_model_flux[tt, nn, pp],
                    "model_flux": model_flux,
                    "start_peakres": start_peakres[tt, nn, pp],
                    "start_peakres_nomask": start_peakres_nomask[tt, nn, pp],
                    "peakres": peakres,
                    "peakres_nomask": peakres_nomask,
                    "masksum": masksum,
                }

                returndict.add(returnvals, time=tt, pol=pp, chan=nn)

    return returndict


def deconvolve(
    img_xds: xr.Dataset = None,
    algorithm: str = "hogbom",
    deconvolve_params: dict | None = None,
    processing_function_threads: int = 1,
    image_data_group_in_name: str = "residual",
    image_data_group_out_name: str = "model",
    image_data_group_out_modified: dict | None = None,
):
    """
    Deconvolve a residual image cube, in place, with planes parallelized
    in C++.

    The residual and model image variables of ``img_xds`` are mutated in
    place: the residual has CLEAN components subtracted from it, and the
    model has the same components added. The per-plane ``(time,
    frequency, polarization)`` loop is delegated to the C++ binding,
    which dispatches planes across ``processing_function_threads`` ``std::thread``
    workers. No per-plane copies of the ``(ny, nx)`` arrays are retained
    on the Python side; the full ``(nt, nf, np, ny, nx)`` numpy buffers
    are handed directly to the CLEAN routine.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset containing the residual, PSF, and (optionally)
        model variables with dimensions
        ``(time, frequency, polarization, y, x)``. The residual and
        model variables are updated in place.
    algorithm : str, optional
        Deconvolution algorithm to use. ``"hogbom"`` (default) for Hogbom
        CLEAN or ``"asp"`` (aliases ``"aspclean"``, ``"asp_clean"``) for
        Adaptive Scale Pixel CLEAN.
    deconvolve_params : dict, optional
        Algorithm-specific parameters. See
        :func:`_validate_deconvolve_params` for the common keys and
        defaults. The ``"asp"`` algorithm additionally honours
        ``fusedthreshold``, ``psf_width``, ``largestscale``,
        ``stoppointmode``, and ``norm_method`` (see :func:`asp_clean`).
    processing_function_threads : int, optional
        Number of ``std::thread`` workers to use for per-plane CLEAN
        execution. Values ``<= 1`` disable threading and run in the
        calling thread. Values larger than the number of planes are
        clamped to the plane count. Default 1.
    image_data_group_in_name : str, optional
        Name of the input data group in ``img_xds`` that provides the
        residual sky image and PSF. Default is ``"residual"``.
    image_data_group_out_name : str, optional
        Name of the output data group to create or update for the model
        image. Default is ``"model"``.
    image_data_group_out_modified : dict, optional
        Mapping from logical names to data-variable names for the output
        data group. Defaults to ``{"sky": "SKY_MODEL"}``.

    Returns
    -------
    returndict : ReturnDict
        Per-plane deconvolution statistics, indexed by
        ``(time, chan, pol)``. Each entry contains iteration count, peak
        residuals before and after, model fluxes, and PSF bookkeeping.

    Raises
    ------
    ValueError
        If ``algorithm`` is not recognized, or if the PSF polarization
        dimension is incompatible with the image polarization dimension.

    Notes
    -----
    - The PSF may be single-polarization (Stokes I only), in which case
      it is broadcast across all image polarizations.
    - Deconvolution is performed independently on each
      ``(time, frequency, polarization)`` plane; parallelization across
      planes happens inside the C++ binding.
    - The underlying CLEAN C++ binding operates directly on the
      Python-owned numpy buffers: the full ``(nt, nf, np, ny, nx)``
      residual and model cubes are the only per-plane allocations,
      and they are updated in place.
    - Because the update is in place, the residual and model data
      variables must be **materialised** (real numpy arrays), not lazy
      (dask-backed). Images opened lazily -- e.g. via
      ``xradio.image.open_image`` / ``load_image`` -- must be realised
      first with ``img_xds = img_xds.load()`` (or ``.compute()``);
      otherwise ``.values`` returns throwaway copies, the in-place CLEAN
      writes are lost, and the model stays zero. ``deconvolve`` raises a
      ``ValueError`` if handed lazy residual/model variables.
    """
    algorithm_l = algorithm.lower()
    if algorithm_l not in (
        "hogbom",
        "hogbom_many_threads",
        "asp",
        "aspclean",
        "asp_clean",
    ):
        raise ValueError(f"Deconvolution algorithm '{algorithm}' not recognized.")

    if image_data_group_out_modified is None:
        image_data_group_out_modified = {"sky": "SKY_MODEL"}

    data_group_in, data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=True,
    )

    residual_name = data_group_in["sky"]
    psf_name = data_group_in["point_spread_function"]
    model_name = data_group_out["sky"]
    max_sidelobe_point_spread_function_name = data_group_out.get(
        "max_sidelobe_point_spread_function", None
    )

    if model_name not in img_xds.data_vars:
        img_xds[model_name] = xr.zeros_like(img_xds[residual_name])
        zero_model = True
        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            data_group_out,
            description="Created model.",
        )
    else:
        zero_model = False

    ntime = img_xds.sizes["time"]
    nchan = img_xds.sizes["frequency"]
    npol = img_xds.sizes["polarization"]
    npol_psf = img_xds[psf_name].sizes["polarization"]

    broadcast_psf = npol_psf == 1 and npol > 1
    if broadcast_psf:
        logger.debug(f"PSF is single-plane, will broadcast to {npol} polarizations")
    elif npol_psf != npol:
        raise ValueError(
            "PSF should have same number of polarizations as the input image, "
            "or be Stokes I only. "
            f"(npol_psf = {npol_psf}, npol_image = {npol})"
        )

    masksum = imgstats.get_image_masksum(img_xds, dv=residual_name)

    deconvolve_params = _validate_deconvolve_params(deconvolve_params)

    max_psf_fraction = deconvolve_params["maxpsffraction"]
    min_psf_fraction = deconvolve_params["minpsffraction"]

    # CLEAN mutates the residual/model numpy buffers in place. A lazy
    # (dask-backed) DataArray returns a *fresh* array from ``.values`` on every
    # access, so the C++ writes would land in a throwaway copy and the model
    # would silently stay unchanged. Fail loudly instead: materialise first.
    lazy_vars = [
        name for name in (residual_name, model_name) if img_xds[name].chunks is not None
    ]
    if lazy_vars:
        raise ValueError(
            f"deconvolve() modifies the residual/model in place, but these image "
            f"data variables are lazy (dask-backed): {lazy_vars}. Materialise the "
            f"image before calling, e.g. `img_xds = img_xds.load()` (or "
            f"`.compute()`); otherwise the in-place CLEAN updates are lost."
        )

    residual_arr = img_xds[residual_name].values
    psf_arr = img_xds[psf_name].values
    model_arr = img_xds[model_name].values
    max_psf_sidelobe = img_xds[
        max_sidelobe_point_spread_function_name
    ].values  # time, frequency, polarization

    # Optional mask cube, matching the residual's shape. The C++ binding
    # accepts a bool mask directly; Python keeps ownership of the buffer.
    mask_name = img_xds.attrs["data_groups"][image_data_group_in_name].get("mask", None)
    mask_arr = None
    # if mask_name in img_xds:
    #    mask_arr = img_xds[mask_name].values
    if mask_name in img_xds:
        mask_arr = np.ascontiguousarray(
            img_xds[mask_name].values,
            dtype=np.bool_,
        )

    # Collect per-plane starting statistics in-place (no copies, just
    # reads). The actual CLEAN iteration is driven in C++.
    start_stats = starting_statistics(
        img_xds=img_xds,
        image_data_group_in_name=image_data_group_in_name,
        image_data_group_out_name=image_data_group_out_name,
        zero_model=zero_model,
    )

    # Drive the CLEAN loop in C++. The helper owns the full
    # (time, frequency, polarization) iteration and the parallel worker
    # pool.
    # print("The start_peakres is ", start_peakres)
    # print("The start_peakres_nomask is ", start_peakres_nomask)
    # print("The start_model_flux is ", start_model_flux)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20,10))
    # plt.imshow(psf_arr[0,0,1])
    # plt.colorbar()
    # plt.title("PSF ")

    # plt.figure(figsize=(20,10))
    # plt.imshow(residual_arr[0,0,1])
    # plt.colorbar()
    # plt.title("REsidual ")

    # plt.figure(figsize=(20,10))
    # plt.imshow(mask_arr[0,0,1])
    # plt.colorbar()
    # plt.title("Mask ")

    # plt.figure(figsize=(20,10))
    # plt.imshow(mask_arr[0,0,0])
    # plt.colorbar()
    # plt.title("Mask ")

    #     idx = np.unravel_index(np.abs(arr).argmax(), arr.shape)
    # return float(arr[idx])
    # print("The max abs value in PSF:", np.max(np.abs(psf_arr[0,0,1])))
    # print("The max abs value in Residual:", np.max(np.abs(residual_arr[0,0,1])))

    # print(np.unravel_index(np.abs(residual_arr[0,0,1]).argmax(), residual_arr.shape))

    # plt.show()

    if algorithm_l == "hogbom":
        results = hogbom_clean(
            residual_cube=residual_arr,
            psf_cube=psf_arr,
            model_cube=model_arr,
            deconvolve_params=deconvolve_params,
            processing_function_threads=processing_function_threads,
            mask_cube=mask_arr,
        )
    elif algorithm_l == "hogbom_many_threads":
        results = hogbom_clean_many_threads(
            residual_cube=residual_arr,
            psf_cube=psf_arr,
            model_cube=model_arr,
            deconvolve_params=deconvolve_params,
            processing_function_threads=processing_function_threads,
            mask_cube=mask_arr,
        )
    else:  # asp / aspclean / asp_clean
        results = asp_clean(
            residual_cube=residual_arr,
            psf_cube=psf_arr,
            model_cube=model_arr,
            deconvolve_params=deconvolve_params,
            processing_function_threads=processing_function_threads,
            mask_cube=mask_arr,
        )

    iters = np.asarray(results["iterations_performed"])

    returndict = create_deconvolution_return_dict(
        img_xds=img_xds,
        image_data_group_in_name=image_data_group_in_name,
        image_data_group_out_name=image_data_group_out_name,
        deconvolve_params=deconvolve_params,
        selected_calculations={
            **start_stats,
            "iters": iters,
            "min_psf_fraction": min_psf_fraction,
            "max_psf_fraction": max_psf_fraction,
            "max_psf_sidelobe": max_psf_sidelobe,
            "masksum": masksum,
        },
    )
    return returndict


def hogbom_clean(
    residual_cube: np.ndarray,
    psf_cube: np.ndarray,
    model_cube: np.ndarray,
    deconvolve_params: dict | None = None,
    processing_function_threads: int = 1,
    mask_cube: np.ndarray | None = None,
):
    """
    Run Hogbom CLEAN over an entire ``(time, frequency, polarization,
    y, x)`` image cube, parallelized across planes in C++.

    The residual and model cubes are updated in place: the residual has
    CLEAN components subtracted from it, and the model accumulates the
    same components. The per-plane loop and the ``std::thread`` worker
    pool live entirely in the C++ binding
    (:func:`hogbom.clean_cube`); this Python layer only validates inputs
    and forwards them.

    Parameters
    ----------
    residual_cube : numpy.ndarray
        5-D residual array with shape ``(nt, nf, np, ny, nx)``. Must be
        C-contiguous, writeable, and float32 or float64. Updated in
        place with the post-CLEAN residual.
    psf_cube : numpy.ndarray
        5-D PSF array with shape ``(nt, nf, np_psf, ny, nx)`` where
        ``np_psf`` equals ``np`` or is ``1`` (Stokes-I broadcast across
        all image polarizations). Must be C-contiguous and share the
        dtype of ``residual_cube``. Not modified.
    model_cube : numpy.ndarray
        5-D model array with shape ``(nt, nf, np, ny, nx)``. CLEAN
        components are **added** into this array in place, allowing
        callers to start from a non-zero initial model or accumulate
        across calls. Must be C-contiguous, writeable, and share the
        dtype of ``residual_cube``.
    deconvolve_params : dict, optional
        Algorithm parameters. See :func:`_validate_deconvolve_params`
        for supported keys and defaults.
    processing_function_threads : int, optional
        Number of ``std::thread`` workers used to run per-plane CLEAN in
        parallel. ``processing_function_threads <= 1`` disables threading; values larger
        than the number of planes are clamped to the plane count.
        Default 1.
    mask_cube : numpy.ndarray, optional
        5-D bool mask array with the same shape as ``residual_cube``.
        Must be C-contiguous and of dtype ``bool`` (the C++ binding
        accepts the buffer directly without a copy). Pixels with mask
        value ``True`` are considered in the peak search; others are
        ignored. ``None`` disables masking.

    Returns
    -------
    dict
        Per-plane summary arrays with shape ``(nt, nf, np)`` and keys
        ``iterations_performed`` (int), ``final_peak`` (float),
        ``total_flux_cleaned`` (float), and ``converged`` (bool). The
        final residual and model cubes are the caller-supplied arrays,
        updated in place.

    Raises
    ------
    ValueError
        If the input arrays are not 5-D, do not share a floating-point
        dtype, or have incompatible shapes.

    Notes
    -----
    The 5-D layout is the same one used by ``xarray`` image datasets in
    astroviper, so a plain ``img_xds[name].values`` call produces an
    array that can be passed to this function without any copy.
    """
    deconvolve_params = _validate_deconvolve_params(deconvolve_params)

    for name, arr in (
        ("residual_cube", residual_cube),
        ("psf_cube", psf_cube),
        ("model_cube", model_cube),
    ):
        if not isinstance(arr, np.ndarray) or arr.ndim != 5:
            raise ValueError(
                f"{name} must be a 5D numpy array with shape " "(nt, nf, np, ny, nx)"
            )

    if residual_cube.shape != model_cube.shape:
        raise ValueError(
            "residual_cube and model_cube must have same shape. "
            f"Got {residual_cube.shape} and {model_cube.shape}"
        )

    # PSF cube may broadcast on the polarization axis.
    nt, nf, npol_img, ny, nx = residual_cube.shape
    nt_p, nf_p, npol_psf, ny_p, nx_p = psf_cube.shape
    if (nt_p, nf_p, ny_p, nx_p) != (nt, nf, ny, nx):
        raise ValueError(
            "psf_cube must match residual_cube on (time, frequency, y, x). "
            f"Got psf shape {psf_cube.shape} vs residual shape "
            f"{residual_cube.shape}"
        )
    if npol_psf not in (npol_img, 1):
        raise ValueError(
            "psf_cube polarization axis must equal residual_cube "
            "polarization axis or be 1 (Stokes I broadcast). "
            f"Got np_psf={npol_psf}, np_img={npol_img}"
        )

    if mask_cube is not None:
        if mask_cube.shape != residual_cube.shape:
            raise ValueError(
                "mask_cube must have the same shape as residual_cube. "
                f"Got {mask_cube.shape} and {residual_cube.shape}"
            )

    logger.debug(f"Residual cube shape: {residual_cube.shape}")
    logger.debug(f"PSF cube shape: {psf_cube.shape}")
    logger.debug(
        "Running Hogbom CLEAN on cube with processing_function_threads=%d"
        % processing_function_threads
    )

    clean_box = deconvolve_params["clean_box"]
    if clean_box is None:
        clean_box = (-1, -1, -1, -1)

    mask_arg = (
        mask_cube if mask_cube is not None else np.array([], dtype=residual_cube.dtype)
    )

    # Per-plane iteration control: each (time, frequency, polarization) plane
    # is cleaned with its own iteration limit and cyclethreshold.
    niter_cube, cyclethreshold_cube = _per_plane_iteration_controls(
        deconvolve_params, nt, nf, npol_img, residual_cube.dtype
    )

    return hogbom.clean_cube(
        residual_cube=residual_cube,
        psf_cube=psf_cube,
        model_cube=model_cube,
        mask_cube=mask_arg,
        clean_box=clean_box,
        max_iter=niter_cube,
        gain=deconvolve_params["gain"],
        threshold=cyclethreshold_cube,
        processing_function_threads=int(processing_function_threads),
    )


def hogbom_clean_many_threads(
    residual_cube: np.ndarray,
    psf_cube: np.ndarray,
    model_cube: np.ndarray,
    deconvolve_params: dict | None = None,
    processing_function_threads: int = 1,
    mask_cube: np.ndarray | None = None,
):
    """Hogbom CLEAN over a 5-D cube, threaded across AND within planes (numba).

    A drop-in alternative to :func:`hogbom_clean`. The C++ ``hogbom`` deconvolver
    parallelizes only *across* the ``(time, frequency, polarization)`` planes, so
    for single-channel imaging (a few planes) it leaves most threads idle. This
    version additionally parallelizes each iteration's peak search and PSF
    subtraction *within* every plane (the parallel domain is ``plane x row``), so
    all ``processing_function_threads`` stay busy regardless of the plane count. The per-plane
    algorithm and tie-breaking match :func:`hogbom_clean`, so the result is
    bit-identical on tie-free data; the residual and model cubes are updated in
    place. Parameters and return value match :func:`hogbom_clean`.

    Threads are set from ``processing_function_threads`` (the imaging
    ``processing_function_threads``).
    """
    deconvolve_params = _validate_deconvolve_params(deconvolve_params)

    for name, arr in (
        ("residual_cube", residual_cube),
        ("psf_cube", psf_cube),
        ("model_cube", model_cube),
    ):
        if not isinstance(arr, np.ndarray) or arr.ndim != 5:
            raise ValueError(
                f"{name} must be a 5D numpy array with shape (nt, nf, np, ny, nx)"
            )
    if residual_cube.shape != model_cube.shape:
        raise ValueError(
            "residual_cube and model_cube must have same shape. "
            f"Got {residual_cube.shape} and {model_cube.shape}"
        )

    nt, nf, npol_img, ny, nx = residual_cube.shape
    nt_p, nf_p, npol_psf, ny_p, nx_p = psf_cube.shape
    if (nt_p, nf_p, ny_p, nx_p) != (nt, nf, ny, nx):
        raise ValueError(
            "psf_cube must match residual_cube on (time, frequency, y, x). "
            f"Got psf shape {psf_cube.shape} vs residual shape {residual_cube.shape}"
        )
    if npol_psf not in (npol_img, 1):
        raise ValueError(
            "psf_cube polarization axis must equal residual_cube polarization "
            f"axis or be 1 (Stokes I broadcast). Got np_psf={npol_psf}, np_img={npol_img}"
        )
    if mask_cube is not None and mask_cube.shape != residual_cube.shape:
        raise ValueError(
            "mask_cube must have the same shape as residual_cube. "
            f"Got {mask_cube.shape} and {residual_cube.shape}"
        )

    logger.debug(
        "Running many-threads Hogbom CLEAN on cube with processing_function_threads=%d"
        % processing_function_threads
    )

    clean_box = deconvolve_params["clean_box"]
    if clean_box is None:
        clean_box = (-1, -1, -1, -1)

    mask_arg = (
        mask_cube if mask_cube is not None else np.array([], dtype=residual_cube.dtype)
    )

    # Per-plane iteration control: each plane has its own iteration limit and
    # cyclethreshold (shared with the C++ path via _per_plane_iteration_controls).
    niter_cube, cyclethreshold_cube = _per_plane_iteration_controls(
        deconvolve_params, nt, nf, npol_img, residual_cube.dtype
    )

    return hogbom.clean_cube_many_threads(
        residual_cube=residual_cube,
        psf_cube=psf_cube,
        model_cube=model_cube,
        mask_cube=mask_arg,
        clean_box=clean_box,
        max_iter=niter_cube,
        gain=deconvolve_params["gain"],
        threshold=cyclethreshold_cube,
        processing_function_threads=int(processing_function_threads),
    )


def asp_clean(
    residual_cube: np.ndarray,
    psf_cube: np.ndarray,
    model_cube: np.ndarray,
    deconvolve_params: dict | None = None,
    processing_function_threads: int = 1,
    mask_cube: np.ndarray | None = None,
):
    """
    Run Adaptive Scale Pixel (Asp / AAspClean) CLEAN over an entire
    ``(time, frequency, polarization, y, x)`` image cube, parallelized
    across planes in C++.

    Like :func:`hogbom_clean`, the residual and model cubes are updated
    in place: the residual has the PSF response of each found component
    subtracted from it, and the model accumulates the components. The
    per-plane loop and the ``std::thread`` worker pool live entirely in
    the C++ binding (:func:`aspclean.clean_cube`); this Python layer only
    validates inputs and forwards them.

    Parameters
    ----------
    residual_cube : numpy.ndarray
        5-D residual array with shape ``(nt, nf, np, ny, nx)``. Must be
        C-contiguous, writeable, and float32 or float64. Updated in
        place with the post-CLEAN residual.
    psf_cube : numpy.ndarray
        5-D PSF array with shape ``(nt, nf, np_psf, ny, nx)`` where
        ``np_psf`` equals ``np`` or is ``1`` (Stokes-I broadcast). Must
        be C-contiguous and share the dtype of ``residual_cube``. Not
        modified.
    model_cube : numpy.ndarray
        5-D model array with shape ``(nt, nf, np, ny, nx)``. Components
        are **added** into this array in place. Must be C-contiguous,
        writeable, and share the dtype of ``residual_cube``.
    deconvolve_params : dict, optional
        Algorithm parameters. The common keys (``gain``, ``niter``,
        ``threshold``) are validated by
        :func:`_validate_deconvolve_params`. The following Asp-specific
        keys are read with sensible defaults if present:

        - ``fusedthreshold`` : float, residual/strength level below which
          the algorithm fuses to Hogbom-style point cleaning for speed.
          A negative value disables the switch. Default 0.0.
        - ``psf_width`` : float, PSF Gaussian width in pixels used to
          seed the initial scales. ``<= 0`` estimates it internally.
          Default 0.0.
        - ``largestscale`` : int, largest initial scale in pixels.
          ``-1`` uses the default (0, w, 2w, 4w, 8w). Default -1.
        - ``stoppointmode`` : int, stop after this many consecutive
          smallest-scale (point) components. ``-1`` disables. Default -1.
        - ``norm_method`` : int, scale-normalization method (1 is the
          preferred CASA default). Default 1.

        Note: ``clean_box`` is ignored by the Asp algorithm (use a mask
        instead).
    processing_function_threads : int, optional
        Number of ``std::thread`` workers used to run per-plane CLEAN in
        parallel. ``processing_function_threads <= 1`` disables threading; values larger
        than the number of planes are clamped to the plane count.
        Default 1.
    mask_cube : numpy.ndarray, optional
        5-D mask array with the same shape as ``residual_cube``. Unlike
        the Hogbom binding (which takes a ``bool`` mask), the Asp binding
        takes a numeric mask of the same dtype as the images (it is
        convolved with each scale internally). A ``bool`` mask is
        converted to that dtype (the only place a copy may be made).
        ``None`` cleans the whole image.

    Returns
    -------
    dict
        Per-plane summary arrays with shape ``(nt, nf, np)`` and keys
        ``iterations_performed`` (int), ``retval`` (int), ``converged``
        (bool), ``peak_residual`` (float), and ``model_flux`` (float).
        The final residual and model cubes are the caller-supplied
        arrays, updated in place.

    Raises
    ------
    ValueError
        If the input arrays are not 5-D, do not share a floating-point
        dtype, or have incompatible shapes.
    """
    deconvolve_params = _validate_deconvolve_params(deconvolve_params)

    for name, arr in (
        ("residual_cube", residual_cube),
        ("psf_cube", psf_cube),
        ("model_cube", model_cube),
    ):
        if not isinstance(arr, np.ndarray) or arr.ndim != 5:
            raise ValueError(
                f"{name} must be a 5D numpy array with shape " "(nt, nf, np, ny, nx)"
            )

    if residual_cube.shape != model_cube.shape:
        raise ValueError(
            "residual_cube and model_cube must have same shape. "
            f"Got {residual_cube.shape} and {model_cube.shape}"
        )

    # PSF cube may broadcast on the polarization axis.
    nt, nf, npol_img, ny, nx = residual_cube.shape
    nt_p, nf_p, npol_psf, ny_p, nx_p = psf_cube.shape
    if (nt_p, nf_p, ny_p, nx_p) != (nt, nf, ny, nx):
        raise ValueError(
            "psf_cube must match residual_cube on (time, frequency, y, x). "
            f"Got psf shape {psf_cube.shape} vs residual shape "
            f"{residual_cube.shape}"
        )
    if npol_psf not in (npol_img, 1):
        raise ValueError(
            "psf_cube polarization axis must equal residual_cube "
            "polarization axis or be 1 (Stokes I broadcast). "
            f"Got np_psf={npol_psf}, np_img={npol_img}"
        )

    if mask_cube is not None and mask_cube.shape != residual_cube.shape:
        raise ValueError(
            "mask_cube must have the same shape as residual_cube. "
            f"Got {mask_cube.shape} and {residual_cube.shape}"
        )

    logger.debug(f"Residual cube shape: {residual_cube.shape}")
    logger.debug(f"PSF cube shape: {psf_cube.shape}")
    logger.debug(
        "Running Asp CLEAN on cube with processing_function_threads=%d"
        % processing_function_threads
    )

    # The Asp binding takes a numeric mask matching the image dtype (it is
    # convolved with each scale). Convert if needed; an empty array means
    # "clean everywhere". ``ascontiguousarray`` is a no-op when the mask is
    # already the right dtype and layout.
    if mask_cube is not None:
        mask_arg = np.ascontiguousarray(mask_cube, dtype=residual_cube.dtype)
    else:
        mask_arg = np.array([], dtype=residual_cube.dtype)

    # Per-plane iteration control: each (time, frequency, polarization) plane
    # is cleaned with its own iteration limit and cyclethreshold. Asp uses a
    # float64 cyclethreshold regardless of the image dtype.
    niter_cube, cyclethreshold_cube = _per_plane_iteration_controls(
        deconvolve_params, nt, nf, npol_img, np.float64
    )

    return aspclean.clean_cube(
        residual=residual_cube,
        psf=psf_cube,
        model=model_cube,
        mask=mask_arg,
        gain=deconvolve_params["gain"],
        threshold=cyclethreshold_cube,
        niter=niter_cube,
        fusedthreshold=deconvolve_params.get("fusedthreshold", 0.0),
        psf_width=deconvolve_params.get("psf_width", 0.0),
        largestscale=deconvolve_params.get("largestscale", -1),
        stoppointmode=deconvolve_params.get("stoppointmode", -1),
        norm_method=deconvolve_params.get("norm_method", 1),
        processing_function_threads=int(processing_function_threads),
    )
