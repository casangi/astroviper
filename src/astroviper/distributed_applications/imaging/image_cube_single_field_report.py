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
    (:func:`~astroviper.utils.report.add_processing_section`). The results
    section holds the cross sections of the point spread function (Stokes I,
    centre channel) along the major axis, the minor axis and the diagonal
    between them with the fitted Gaussian; the restoring beam of a run with
    ``restoring_beam``, the axes of the PSF fit of every channel against it,
    the ratios of its axes to them and the channels whose residual was left
    unconvolved and unscaled; the maximum over frequency (the
    ``"maximum"`` moment of
    :func:`~astroviper.distributed_applications.image_analysis.moments.moments`)
    of the residual, the model and the primary beam corrected restored image
    per Stokes plane; the spectrum of the peak inside the clean mask (of the
    restored image, else of the residual); and per channel the cycle control
    stop codes, the occurrences of the model update stop codes, the
    iterations, the imaging cycles, the final residual entropy and the
    residual RMS. What the run did not write or record is left out with a
    note.

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
        )
        document.section("Results")
        _add_point_spread_function(
            document, image_store, output_image_format, available
        )
        _add_restoring_beam(document, image_store, output_image_format, available)
        _add_moment_maximum(
            document,
            image_store,
            output_image_format,
            available,
            files_directory,
            thread_info,
        )
        _add_spectrum(document, return_dict)
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


def _open_beam_fit_params(image_store, output_image_format, available, key):
    """The beam-fit-params variable ``key`` (lazily opened), ``None`` when it
    was not written. FITS output holds the restoring beam in the header of the
    restored images, and the others in the ``BEAMS`` tables of the image files
    they label."""
    if output_image_format != "fits":
        if key not in available:
            return None
        return _open_variable(image_store, output_image_format, key)[key.upper()]
    from astropy.io import fits

    from astroviper.node_tasks.imaging.utils.skunk_works_fits import (
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
    img_xds = _open_variable(image_store, output_image_format, image_keys[0])
    return img_xds["BEAM_FIT_PARAMS_" + image_keys[0].upper()]


def _add_restoring_beam(document, image_store, output_image_format, available):
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
            "No restoring beam: beam_fit_params_sky_restored was not written (it "
            "needs restore=True and a restoring_beam)."
        )
        return
    target = np.asarray(
        restored_beam.isel(time=0, frequency=0, polarization=0).values,
        dtype=np.float64,
    )
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
    )
    if psf_beam is None:
        document.note(
            "The beam fit of the point spread function was not kept "
            "(beam_fit_params_point_spread_function)."
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


def _stokes_and_frequency(statistics):
    stokes = [str(label) for label in statistics["polarization"].values]
    return stokes, statistics["frequency"].values


def _add_spectrum(document, return_dict):
    from astroviper.utils.report_plots import plot_spectra

    document.subsection("Spectrum")
    image_statistics = return_dict.get("image_statistics") or {}
    descriptions = {"sky_restored": "restored image", "sky_residual": "residual"}
    key = next((key for key in descriptions if key in image_statistics), None)
    if key is None:
        document.note("No image statistics were recorded.")
        return
    description = descriptions[key]
    statistics = image_statistics[key].isel(time=0)
    stokes, frequency = _stokes_and_frequency(statistics)
    document.figure(
        plot_spectra(
            frequency,
            statistics["peak_masked"].transpose("frequency", "polarization").values,
            stokes,
            "peak inside the mask",
            title=f"Spectrum: peak of the {description} inside the clean mask",
        ),
        caption=f"The signed peak of the {description} inside the clean mask (the "
        "primary beam mask without one) of every channel and Stokes plane, from "
        "the image statistics of the run.",
    )


def _add_deconvolution(document, return_dict):
    from astroviper.processing_functions.imaging.utils import imaging_dict_to_dataframe
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_STOP_DESCRIPTIONS,
        MODEL_UPDATE_CONTINUE,
        MODEL_UPDATE_STOP_DESCRIPTIONS,
    )
    from astroviper.utils.report_plots import plot_deconvolution_summary, plot_spectra

    document.subsection("Deconvolution")
    residual = (return_dict.get("image_statistics") or {}).get("sky_residual")
    if residual is None:
        document.note("No image statistics of the residual were recorded.")
        return
    residual = residual.isel(time=0).transpose("frequency", "polarization")
    stokes, frequency = _stokes_and_frequency(residual)
    shape = (len(frequency), len(stokes))
    entropy = (
        residual["entropy"].values if "entropy" in residual else np.full(shape, np.nan)
    )
    record = imaging_dict_to_dataframe(return_dict.get("deconvolution"))
    if record.empty:
        document.note(
            "No deconvolution (max_iter=0 or max_cycles=0): no stop codes, "
            "iterations or imaging cycles."
        )
        document.figure(
            plot_spectra(frequency, residual["rms"].values, stokes, "RMS"),
            caption="The RMS of the residual (the dirty image) of every channel "
            "and Stokes plane.",
        )
        return

    record = record[record["time_index"] == 0]
    # The statistics are in frequency order: a plane goes to the row of its
    # frequency (its channel number for a record without one).
    row_of_frequency = {float(value): index for index, value in enumerate(frequency)}
    imaging_stop_code = np.full(shape, np.nan)
    iterations = np.zeros(shape)
    imaging_cycles = np.zeros(len(frequency))
    counts = {}
    has_model_update_codes = "model_update_stop_code" in record
    for row in record.itertuples(index=False):
        chan = row_of_frequency.get(
            float(getattr(row, "frequency", np.nan)), int(row.chan)
        )
        pol = int(row.pol)
        imaging_stop_code[chan, pol] = getattr(row, "stop_code_imaging", np.nan)
        iterations[chan, pol] = row.iter_total
        imaging_cycles[chan] = max(imaging_cycles[chan], row.n_cycles)
        if has_model_update_codes and isinstance(row.model_update_stop_code, list):
            for code in row.model_update_stop_code:
                if int(code) == MODEL_UPDATE_CONTINUE:
                    continue
                counts.setdefault(int(code), np.zeros(len(frequency)))[chan] += 1
    document.figure(
        plot_deconvolution_summary(
            frequency,
            stokes,
            imaging_stop_code,
            counts,
            iterations,
            imaging_cycles,
            entropy,
            residual["rms"].values,
            IMAGING_STOP_DESCRIPTIONS,
            MODEL_UPDATE_STOP_DESCRIPTIONS,
        ),
        caption="Per channel: the cycle control stop code of every Stokes plane, "
        "the number of model updates (over the Stokes planes) that ended with "
        "each model update stop code (a model update that did not run is not "
        "counted), the total iterations, the imaging cycles, and the entropy and "
        "RMS of the final residual.",
    )
    if not has_model_update_codes:
        document.note("The deconvolver recorded no model update stop codes.")
    if not np.any(np.isfinite(entropy)):
        document.note(
            "The final residual entropy is not a number for every plane: it is "
            "worked out only for a residual whose peak is at most entropy_max_snr "
            "times its RMS (and only for a run made with a report)."
        )
