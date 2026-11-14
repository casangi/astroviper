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
):
    """Make the report of an ``image_cube_single_field`` run.

    The processing section is the one of every distributed application
    (:func:`~astroviper.utils.report.add_processing_section`), its node task
    timing and phase times grouped by the phases of the imaging node task
    (``IMAGING_TIMING_PHASES``). The results section holds the cross sections of the point spread function (Stokes I,
    centre channel) along the major axis, the minor axis and the diagonal
    between them with the fitted Gaussian; the maximum over frequency (the
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
        in ``image_store``.
    thread_info : dict, optional
        Passed to ``moments``.

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
            document, image_store, output_image_format, available
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


def _open_variable(image_store, output_image_format, key):
    """The image dataset holding the variable ``key`` (lazily opened)."""
    if output_image_format == "fits":
        from xradio.image import open_image

        return open_image({key: os.path.join(image_store, key.upper() + ".fits")})
    import xarray as xr

    return xr.open_zarr(image_store)


def _add_point_spread_function(document, image_store, output_image_format, available):
    from astroviper.utils.report_plots import plot_point_spread_function_cross_sections

    document.subsection("Point spread function")
    if "point_spread_function" not in available:
        document.note("The point spread function was not kept (point_spread_function).")
        return
    img_xds = _open_variable(image_store, output_image_format, "point_spread_function")
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
        for fields in return_dict["deconvolution"].data.values()
    ):
        document.note("The deconvolver recorded no model update stop codes.")
    if not np.any(np.isfinite(per_channel["entropy"])):
        document.note(
            "The final residual entropy is not a number for every plane: it is "
            "worked out only for a residual whose peak is at most entropy_max_snr "
            "times its RMS (and only for a run made with a report)."
        )
