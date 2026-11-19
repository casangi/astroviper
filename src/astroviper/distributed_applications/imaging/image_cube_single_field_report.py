"""Report of an :func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field` run."""

from __future__ import annotations

import os

import numpy as np

# The maximum over frequency of these image variables is drawn, in this order:
# (image variable key, description).
_MOMENT_MAXIMUM_VARIABLES = (
    ("sky_residual", "residual"),
    ("sky_model", "model"),
    ("sky_restored_primary_beam_corrected", "primary beam corrected restored image"),
)


def image_cube_single_field_report(
    return_dict,
    image_store,
    report="html",
    output_image_format="zarr",
    image_data_variables_keep=None,
    thread_info=None,
    restoring_beam=None,
    planes_without_fit=None,
):
    """Make the report of an ``image_cube_single_field`` run.

    The processing section is the one of every distributed application
    (:func:`~astroviper.utils.report.add_processing_section`), its node task
    timing and phase times grouped by the phases of the imaging node task
    (``IMAGING_TIMING_PHASES``). The results section holds the cross sections of the point spread function (Stokes I,
    centre channel) along the major axis, the minor axis and the diagonal
    between them with the fitted Gaussian; the restoring beam of a run with
    ``restoring_beam``, the axes of the PSF fit of every channel against it,
    the ratios of its axes to them and the channels whose residual was left
    unconvolved and unscaled; the maximum over frequency (the
    ``"maximum"`` moment of
    :func:`~astroviper.distributed_applications.image_analysis.moments.moments`)
    of the residual, the model and the primary beam corrected restored image
    per Stokes plane; every per-plane statistic of the residual and of the
    restored image; and per channel the cycle control stop codes, the
    occurrences of the model update stop codes, the iterations, the imaging
    cycles, the peak residual at the start of the first model update and at
    the end, the final residual entropy and the residual RMS. What the run
    did not write or record is left out with a note.

    Parameters
    ----------
    return_dict : dict
        The return dict of ``image_cube_single_field``.
    image_store : str
        The image store it wrote (a Zarr store, or the directory of FITS
        files of ``output_image_format="fits"``).
    report : str, optional
        ``"html"`` (default), ``"pdf"`` or ``"notebook"``.
    output_image_format : str, optional
        ``"zarr"`` (default) or ``"fits"``, as in the run.
    image_data_variables_keep : list of str, optional
        The image variables the run kept; ``None`` takes the variables found
        in ``image_store``. FITS output has no file of a beam-fit-params
        variable, so the PSF beams of its restoring-beam section are found
        only with the list; the restoring beam is the header beam of the
        restored images.
    thread_info : dict, optional
        Passed to ``moments``.
    restoring_beam : list of float, optional
        The run's ``restoring_beam``; ``None`` (default) for a run without one,
        whose planes were each restored with their own PSF fit.
    planes_without_fit : dict, optional
        FITS output: which planes ``(frequency, polarization)`` of the file of
        each image variable had no beam fit before the run filled its
        ``BEAMS`` rows, as
        :func:`~astroviper.utils.chunk_io.fill_fits_beams_without_fit`
        returns them. ``None`` (default) takes the rows as they are.

    Returns
    -------
    str or None
        The path of the report written (``<name>_report.html`` or ``.pdf``
        next to ``image_store``); ``None`` for ``"notebook"``. The moment maps
        are kept in ``<name>_report_files`` next to ``image_store``.
    """
    import datetime

    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
    )
    from astroviper.processing_functions.imaging.utils import IMAGING_TIMING_PHASES
    from astroviper.utils.report import (
        Report,
        add_processing_section,
        report_files_directory,
        report_path,
    )

    path = report_path(image_store, report)
    files_directory = report_files_directory(image_store)
    available = _available_variables(
        image_store, output_image_format, image_data_variables_keep
    )
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with Report(
        "image_cube_single_field report",
        report,
        path,
        subtitle=f"Image store {image_store}, report made {stamp}.",
    ) as document:
        add_processing_section(
            document,
            return_dict,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            task_name="frequency chunk",
            node_task_timing_phases=IMAGING_TIMING_PHASES,
        )
        document.section("Results")
        _add_point_spread_function(
            document, image_store, output_image_format, available, planes_without_fit
        )
        _add_restoring_beam(
            document,
            image_store,
            output_image_format,
            available,
            restoring_beam,
            planes_without_fit,
        )
        _add_moment_maximum(
            document,
            image_store,
            output_image_format,
            available,
            files_directory,
            thread_info,
        )
        _add_image_statistics(document, return_dict)
        _add_deconvolution(document, return_dict)
    return path


def _available_variables(image_store, output_image_format, image_data_variables_keep):
    """The image variable keys written to ``image_store``."""
    if image_data_variables_keep is not None:
        return {str(key).lower() for key in image_data_variables_keep}
    if output_image_format == "fits":
        return {
            name[: -len(".fits")].lower()
            for name in os.listdir(image_store)
            if name.endswith(".fits")
        }
    import xarray as xr

    return {str(name).lower() for name in xr.open_zarr(image_store).data_vars}


def _open_variable(image_store, output_image_format, key, planes_without_fit=None):
    """The image dataset holding the variable ``key`` (lazily opened); the
    beams of a FITS file NaN in its planes without a fit."""
    import xarray as xr

    if output_image_format == "fits":
        from xradio.image import open_image

        img_xds = open_image({key: os.path.join(image_store, key.upper() + ".fits")})
        if planes_without_fit is not None and key in planes_without_fit:
            beam_name = "BEAM_FIT_PARAMS_" + key.upper()
            img_xds[beam_name] = img_xds[beam_name].where(
                ~xr.DataArray(
                    planes_without_fit[key], dims=("frequency", "polarization")
                )
            )
        return img_xds
    return xr.open_zarr(image_store)


def _add_point_spread_function(
    document, image_store, output_image_format, available, planes_without_fit=None
):
    from astroviper.utils.report_plots import plot_point_spread_function_cross_sections

    document.subsection("Point spread function")
    if "point_spread_function" not in available:
        document.note("The point spread function was not kept (point_spread_function).")
        return
    img_xds = _open_variable(
        image_store, output_image_format, "point_spread_function", planes_without_fit
    )
    beam_name = "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
    if beam_name not in img_xds:
        document.note(
            "The beam fit of the point spread function was not kept "
            "(beam_fit_params_point_spread_function)."
        )
        return
    channel = img_xds.sizes["frequency"] // 2
    polarization = [str(stokes) for stokes in img_xds["polarization"].values]
    plane = {
        "time": 0,
        "frequency": channel,
        "polarization": polarization.index("I"),
    }
    psf = img_xds["POINT_SPREAD_FUNCTION"].isel(plane).transpose("l", "m")
    beam = np.asarray(img_xds[beam_name].isel(plane).values, dtype=np.float64).ravel()
    if beam.size != 3 or not np.all(np.isfinite(beam)) or np.any(beam[:2] <= 0):
        document.note(f"The beam fit of the centre channel failed ({beam}).")
        return
    cell_size = (
        float(psf["l"].values[1] - psf["l"].values[0]),
        float(psf["m"].values[1] - psf["m"].values[0]),
    )
    frequency = float(img_xds["frequency"].values[channel])
    arcsec = 180.0 / np.pi * 3600.0
    document.figure(
        plot_point_spread_function_cross_sections(
            psf.values,
            beam,
            cell_size,
            title=(
                f"Stokes I, channel {channel} ({frequency / 1e9:.6g} GHz): beam "
                f'{beam[0] * arcsec:.4g}" x {beam[1] * arcsec:.4g}", position '
                f"angle {np.degrees(beam[2]):.1f} deg"
            ),
        ),
        caption="The point spread function through its peak (solid) and the "
        "fitted Gaussian (dashed) along the major axis, the minor axis and the "
        "diagonal between them.",
    )


def _open_beam_fit_params(
    image_store, output_image_format, available, key, planes_without_fit=None
):
    """The beam-fit-params variable ``key`` (lazily opened), ``None`` when it
    was not written. FITS output holds the restoring beam in the header of the
    restored images, and the others in the ``BEAMS`` tables of the image files
    they label."""
    if output_image_format != "fits":
        if key not in available:
            return None
        return _open_variable(image_store, output_image_format, key)[key.upper()]
    from astropy.io import fits

    from astroviper.utils.chunk_io.fits_blocks import (
        _RESTORED_VARIABLES,
        _beams_table_variable,
    )

    if key == "beam_fit_params_sky_restored":
        image_keys = [
            image_key
            for image_key in sorted(_RESTORED_VARIABLES & available)
            if "BMAJ"
            in fits.getheader(os.path.join(image_store, image_key.upper() + ".fits"))
        ]
    elif key in available:
        beam_variables = [
            name for name in available if name.startswith("beam_fit_params")
        ]
        image_keys = [
            image_key
            for image_key in sorted(available)
            if _beams_table_variable(image_key, beam_variables) == key
        ]
    else:
        image_keys = []
    if not image_keys:
        return None
    img_xds = _open_variable(
        image_store, output_image_format, image_keys[0], planes_without_fit
    )
    return img_xds["BEAM_FIT_PARAMS_" + image_keys[0].upper()]


def _add_restoring_beam(
    document,
    image_store,
    output_image_format,
    available,
    restoring_beam,
    planes_without_fit=None,
):
    import pandas as pd

    from astroviper.processing_functions.image_analysis.common_beam import (
        _correcting_covariance,
        _growth_factor,
    )
    from astroviper.utils.report_plots import plot_restoring_beam_axes

    document.subsection("Restoring beam")
    restored_beam = _open_beam_fit_params(
        image_store, output_image_format, available, "beam_fit_params_sky_restored"
    )
    if restored_beam is None:
        document.note(
            "No restoring beam: "
            + (
                "no restored FITS image has a header beam (it needs restore=True "
                "and a restoring_beam)."
                if output_image_format == "fits"
                else "beam_fit_params_sky_restored was not written (it needs "
                "restore=True)."
            )
        )
        return
    # A target is in every plane, but a plane of a skipped task holds none.
    planes = np.asarray(restored_beam.values, dtype=np.float64).reshape(-1, 3)
    planes = planes[
        np.isfinite(planes).all(axis=1) & (planes[:, 0] > 0.0) & (planes[:, 1] > 0.0)
    ]
    if not len(planes):
        document.note(
            "No restoring beam: beam_fit_params_sky_restored holds no beam (no "
            "task wrote it, or no plane has a PSF fit)."
        )
        return
    if restoring_beam is None:
        document.note(
            "No restoring beam: every plane was restored with the PSF fit of its "
            "first Stokes plane, as beam_fit_params_sky_restored records."
        )
        return
    target = planes[0]
    arcsec = 180.0 / np.pi * 3600.0
    document.text(
        f'Every plane was restored to the beam {target[0] * arcsec:.4g}" x '
        f'{target[1] * arcsec:.4g}", position angle {np.degrees(target[2]):.1f} deg.'
    )
    psf_beam = _open_beam_fit_params(
        image_store,
        output_image_format,
        available,
        "beam_fit_params_point_spread_function",
        planes_without_fit,
    )
    if psf_beam is None:
        document.note(
            "The beam fit of the point spread function was not kept "
            "(beam_fit_params_point_spread_function)"
            + (
                ", or no FITS file carries it: FITS output holds it in the BEAMS "
                "tables of the point spread function, residual, model and "
                "deconvolved images, found with the list of the kept variables."
                if output_image_format == "fits"
                else "."
            )
        )
        return
    # The restore applies the beam of the first polarization to every plane.
    psf_beam = psf_beam.isel(time=0, polarization=0).transpose(
        "frequency", "beam_params_label"
    )
    beams = np.asarray(psf_beam.values, dtype=np.float64)
    frequency = psf_beam["frequency"].values
    _, _, state = _correcting_covariance(beams, target)
    no_fit = state == "no fit"
    not_contained = state == "not contained"
    document.figure(
        plot_restoring_beam_axes(frequency, beams, target, not_contained),
        caption="The major and the minor axis of the PSF fit of every channel "
        "(of the first Stokes plane, whose beam the restore applies to all of "
        "them) and of the restoring beam (dashed); crosses mark the channels "
        "whose PSF beam the restoring beam does not contain.",
    )
    if not no_fit.all():
        ratios = target[:2] / beams[~no_fit, :2]
        document.text(
            "The restoring beam's axes are "
            f"{ratios[:, 0].min():.4f} to {ratios[:, 0].max():.4f} times the "
            "major axes and "
            f"{ratios[:, 1].min():.4f} to {ratios[:, 1].max():.4f} times the "
            "minor axes of the PSF fits."
        )
    uncorrected = no_fit | not_contained
    if not uncorrected.any():
        document.text("The restoring beam contains the PSF beam of every channel.")
        return
    growth = np.full(len(beams), np.nan)
    growth[not_contained] = _growth_factor(beams[not_contained], target)
    document.table(
        pd.DataFrame(
            {
                "frequency (GHz)": frequency[uncorrected] / 1e9,
                "PSF beam": np.where(no_fit[uncorrected], "no fit", "not contained"),
                "growth factor": growth[uncorrected],
            },
            index=pd.Index(np.flatnonzero(uncorrected), name="channel"),
        ),
        caption="The channels whose residual was left unconvolved and "
        "unscaled: without a PSF fit, or with a PSF beam the restoring beam "
        "does not contain; the restoring beam would have to grow by the growth "
        "factor to contain it.",
        float_format="{:.10g}",
    )


def _moment_input(image_store, output_image_format, key):
    """``(input_image_store, image_data_group_in_name, image_data_group_in_role)``
    of ``moments`` for the image variable ``key``."""
    if output_image_format == "fits":
        return os.path.join(image_store, key.upper() + ".fits"), "base", "sky"
    from astroviper.utils.io import imaging_data_variable_data_group_roles

    group, role = imaging_data_variable_data_group_roles[key]
    return image_store, group, role


def _add_moment_maximum(
    document, image_store, output_image_format, available, files_directory, thread_info
):
    import xarray as xr

    from astroviper.distributed_applications.image_analysis.moments import moments
    from astroviper.utils.report_plots import plot_stokes_images

    document.subsection("Maximum over frequency")
    for key, description in _MOMENT_MAXIMUM_VARIABLES:
        if key not in available:
            document.note(
                f"No {description}: {key} was not written"
                + (
                    " (it needs restore=True and restoration_output='primary_beam_corrected' or 'both')."
                    if key == "sky_restored_primary_beam_corrected"
                    else "."
                )
            )
            continue
        input_image_store, group, role = _moment_input(
            image_store, output_image_format, key
        )
        # A store name ending in .img.zarr is the store written.
        moments_image_store = os.path.join(
            files_directory, f"moment_maximum_{key}.img.zarr"
        )
        moments(
            input_image_store=input_image_store,
            moments_image_store=moments_image_store,
            moments=["maximum"],
            moment_axis="frequency",
            image_data_group_in_name=group,
            image_data_group_in_role=role,
            thread_info=thread_info,
            overwrite=True,
        )
        maximum = (
            xr.open_zarr(moments_image_store)["SKY_MOMENT_MAXIMUM"]
            .isel(time=0, frequency=0)
            .load()
        )
        document.figure(
            plot_stokes_images(maximum, title=f"Maximum over frequency: {description}"),
            caption=f"The largest value of the {description} over the channels "
            f"of every pixel ({moments_image_store}).",
        )


# The image statistics drawn, in this order: (image variable key, description).
_IMAGE_STATISTICS_VARIABLES = (
    ("sky_residual", "residual"),
    ("sky_restored", "restored image"),
)


def _add_image_statistics(document, return_dict):
    from astroviper.utils.report_plots import plot_image_statistics

    document.subsection("Image statistics")
    image_statistics = return_dict.get("image_statistics") or {}
    drawn = [key for key, _ in _IMAGE_STATISTICS_VARIABLES if key in image_statistics]
    if not drawn:
        document.note(
            "No image statistics of the residual or the restored image were recorded."
        )
    for key, description in _IMAGE_STATISTICS_VARIABLES:
        if key in drawn:
            document.figure(
                plot_image_statistics(
                    return_dict,
                    image_variable=key,
                    title=f"Image statistics of the {description}",
                ),
                caption=f"Every per-plane statistic of the {description} over (l, m), "
                "per channel and Stokes plane: over the image (solid) and inside the "
                "clean mask (dashed; the primary beam mask without one), from the "
                "image statistics of the run (time 0).",
            )
    others = [key for key in image_statistics if key not in drawn]
    if others:
        document.note(
            "The statistics of "
            + ", ".join(others)
            + " are in the return dict (image_statistics; plot_image_statistics "
            "draws them)."
        )


def _add_deconvolution(document, return_dict):
    from astroviper.processing_functions.imaging.utils import (
        deconvolution_per_channel,
    )
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_STOP_DESCRIPTIONS,
        MODEL_UPDATE_STOP_DESCRIPTIONS,
    )
    from astroviper.utils.report_plots import plot_deconvolution_summary

    document.subsection("Deconvolution")
    if (return_dict.get("image_statistics") or {}).get("sky_residual") is None:
        document.note("No image statistics of the residual were recorded.")
        return
    per_channel = deconvolution_per_channel(return_dict)
    if per_channel is None:
        document.note(
            "No deconvolution (max_iter=0 or max_cycles=0): no stop codes, "
            "iterations or imaging cycles; the RMS of the residual (the dirty "
            "image) is in the image statistics."
        )
        return
    document.figure(
        plot_deconvolution_summary(
            **per_channel,
            imaging_stop_descriptions=IMAGING_STOP_DESCRIPTIONS,
            model_update_stop_descriptions=MODEL_UPDATE_STOP_DESCRIPTIONS,
        ),
        caption="Per channel: the cycle control stop code of every Stokes plane, "
        "the number of model updates (over the Stokes planes) that ended with "
        "each model update stop code (a model update that did not run is not "
        "counted), the total iterations, the imaging cycles, the absolute peak "
        "residual inside the clean mask at the start of the first model update "
        "and of the final residual (with the threshold when the record holds "
        "one), and the entropy and RMS of the final residual. Shaded: channels "
        "never deconvolved.",
    )
    if not any(
        "model_update_stop_code" in fields
        for fields in return_dict["deconvolution"].values()
    ):
        document.note("The deconvolver recorded no model update stop codes.")
    if not np.any(np.isfinite(per_channel["entropy"])):
        document.note(
            "The final residual entropy is not a number for every plane: it is "
            "worked out only for a residual whose peak is at most entropy_max_snr "
            "times its RMS (and only for a run made with a report)."
        )
