"""Public flat-sky image products shared by cube, MFS, and MVC workflows."""

import copy
import time

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import modify_data_groups_xds

POLYNOMIAL = "frequency_polynomial"
MOMENT = "frequency_polynomial_moment"


def _role(image, group, role):
    from astroviper.processing_functions.imaging.restore import (
        resolve_restoration_product,
    )

    return resolve_restoration_product(
        image,
        group,
        role,
        fallback={
            "primary_beam": "PRIMARY_BEAM",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        }.get(role),
    )


def _divide_by_beam(image, beam, limit, outside):
    supported = np.isfinite(beam) & (beam > 0) & (beam >= limit)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = xr.where(supported, image / beam, outside).astype(image.dtype)
    result = result.where(np.isfinite(result), outside).transpose(*image.dims)
    result.attrs = dict(image.attrs)
    return result


def multiply_taylor_polynomials(image, beam):
    """Multiply coefficient stacks, retaining exactly the image's Taylor orders.

    Parameters
    ----------
    image, beam : xarray.DataArray
        Full image stacks with a common polynomial axis and matching time,
        polarization and spatial coordinates. The beam supplies at least as
        many coefficients as the image.

    Returns
    -------
    xarray.DataArray
        Coefficients of the product polynomial, truncated at the image order.
    """
    axis = POLYNOMIAL if POLYNOMIAL in image.dims else "taylor_term"
    if axis not in beam.dims or beam.sizes[axis] < image.sizes[axis]:
        raise ValueError("PRIMARY_BEAM must supply every image polynomial order")
    for dim in image.dims:
        if dim != axis and not image.coords[dim].equals(beam.coords[dim]):
            raise ValueError(
                f"PRIMARY_BEAM coordinate {dim!r} does not match the image"
            )
    planes = []
    for order in range(image.sizes[axis]):
        plane = xr.zeros_like(image.isel({axis: 0}, drop=True))
        for beam_order in range(order + 1):
            plane = plane + beam.isel({axis: beam_order}, drop=True) * image.isel(
                {axis: order - beam_order}, drop=True
            )
        planes.append(plane)
    result = (
        xr.concat(planes, dim=image[axis]).transpose(*image.dims).astype(image.dtype)
    )
    result.attrs = dict(image.attrs)
    return result


def finalize_image_products(
    img_xds,
    *,
    continuum_mode=None,
    deconvolver="hogbom",
    primary_beam_limit=0.2,
    image_data_group_in_residual_name="residual",
    image_data_group_in_model_name="model",
    apparent_inputs=None,
):
    """Convert internal image solutions to the common public flat-sky basis.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Internal cube or continuum products, with model and residual groups.
    continuum_mode : {"mfs", "mvc"} or None, optional
        Internal continuum convention; None selects a frequency cube.
    deconvolver : str, optional
        MT-MFS residual right-hand sides are converted to principal residuals
        before primary-beam correction.
    primary_beam_limit : float, optional
        Positive finite primary-beam response required for valid sky pixels.
    image_data_group_in_residual_name, image_data_group_in_model_name : str, optional
        Input product groups. Their public sky roles are updated in the result.
    apparent_inputs : dict or None, optional
        Empty temporary mapping supplied only when flat-noise restoration is
        requested. Receives apparent model and principal residual arrays before
        cutoff masking; restore_finalized_image consumes and clears the mapping.
        These arrays are never attached to the public dataset.

    Returns
    -------
    xarray.Dataset
        Flat-sky model and coefficient-space residual. Continuum PRIMARY_BEAM
        contains Taylor coefficients; MVC also retains COMMON_PRIMARY_BEAM.
        The input and its cycle-control residual are not modified.
    """
    if img_xds.attrs.get("public_image_convention") == "flat_sky":
        return img_xds
    if continuum_mode not in (None, "mfs", "mvc"):
        raise ValueError("continuum_mode must be None, 'mfs', or 'mvc'")
    from astroviper.processing_functions.imaging.config_check._predicates import (
        check_primary_beam_limit,
    )

    check_primary_beam_limit(primary_beam_limit)
    out = img_xds.copy(deep=False)
    out.attrs = copy.deepcopy(img_xds.attrs)
    residual_group = image_data_group_in_residual_name
    model_group = image_data_group_in_model_name
    model_name = _role(out, model_group, "sky")
    residual_name = _role(out, residual_group, "sky")
    model, residual = out[model_name], out[residual_name]
    axis = "frequency" if continuum_mode is None else "taylor_term"
    expected = ("time", axis, "polarization", "l", "m")
    for name, value in ((model_name, model), (residual_name, residual)):
        if axis not in value.dims:
            raise ValueError(f"{name!r} must contain a {axis!r} dimension")
        if value.sizes[axis] == 0:
            raise ValueError(f"{name!r} contains no Taylor terms")
        if set(value.dims) != set(expected):
            raise ValueError(f"{name!r} is missing restoration dimensions {expected}")
    if model.shape != residual.shape or model.dims != residual.dims:
        raise ValueError("Model and residual must have identical dimensions and shape")
    xr.align(model, residual, join="exact", copy=False)
    if continuum_mode is not None and (
        deconvolver == "mtmfs" or "POINT_SPREAD_FUNCTION" in out
    ):
        psf = out[_role(out, residual_group, "point_spread_function")]
        if any(
            psf.sizes.get(dim, 1) == 0
            for dim in ("psf_taylor_order", "taylor_term", "frequency")
        ):
            raise ValueError("PSF contains no Taylor-order planes")
    if continuum_mode is not None and deconvolver == "mtmfs":
        from astroviper.processing_functions.imaging.image_continuum_single_field import (
            _mtmfs_principal_residual_for_restoration,
        )

        residual = _mtmfs_principal_residual_for_restoration(
            residual, out[_role(out, residual_group, "point_spread_function")]
        )
    if apparent_inputs is not None:
        if apparent_inputs:
            raise ValueError("apparent_inputs must be empty before finalization")
        if continuum_mode == "mvc":
            common_beam = out[_role(out, residual_group, "primary_beam")]
            if "frequency" in common_beam.dims:
                common_beam = common_beam.squeeze("frequency", drop=True)
            polynomial_beam = out["PRIMARY_BEAM_TAYLOR"]
            apparent_inputs["model"] = multiply_taylor_polynomials(
                _divide_by_beam(model, common_beam, 0.0, 0.0), polynomial_beam
            )
            apparent_inputs["residual"] = multiply_taylor_polynomials(
                _divide_by_beam(residual, common_beam, 0.0, np.nan), polynomial_beam
            )
        else:
            # Keep references to existing arrays; finalization replaces the
            # public variables without modifying the solver's arrays in place.
            apparent_inputs.update(model=model, residual=residual)
    if continuum_mode == "mfs":
        from astroviper.processing_functions.imaging.restore import (
            _divide_taylor_polynomials,
        )

        polynomial_name = out.attrs["data_groups"][residual_group].get(
            "primary_beam_taylor", "PRIMARY_BEAM_TAYLOR"
        )
        if polynomial_name not in out:
            raise KeyError(
                "MFS finalization requires fitted PRIMARY_BEAM_TAYLOR coefficients"
            )
        beam = out[polynomial_name]
        model = _divide_taylor_polynomials(
            model, beam, primary_beam_limit, outside_value=0.0
        )
        residual = _divide_taylor_polynomials(residual, beam, primary_beam_limit)
    else:
        beam = out[_role(out, residual_group, "primary_beam")]
        if continuum_mode == "mvc":
            out["COMMON_PRIMARY_BEAM"] = beam.copy(deep=False)
            if "frequency" in beam.dims:
                if beam.sizes["frequency"] != 1:
                    raise ValueError(
                        "MVC finalization requires one common primary beam"
                    )
                beam = beam.isel(frequency=0, drop=True)
        model = _divide_by_beam(model, beam, primary_beam_limit, 0.0)
        residual = _divide_by_beam(residual, beam, primary_beam_limit, np.nan)
    for name, value in ((model_name, model), (residual_name, residual)):
        value.attrs = {
            **value.attrs,
            "sky_convention": "flat_sky",
            "primary_beam_convention": "flat_sky",
        }
        value.attrs["description"] = (
            "Intrinsic flat-sky model coefficients."
            if name == model_name
            else "Flat-sky coefficient-space residual added during restoration."
        )
        out[name] = value
    out[residual_name].attrs["residual_representation"] = "coefficient_space"
    if continuum_mode is not None:
        if "PRIMARY_BEAM_TAYLOR" not in out:
            raise KeyError(
                "Continuum finalization requires fitted PRIMARY_BEAM_TAYLOR coefficients"
            )
        # Remove the channel/common beam before installing the polynomial beam.
        polynomial_beam = out["PRIMARY_BEAM_TAYLOR"]
        out = out.drop_vars(["PRIMARY_BEAM", "PRIMARY_BEAM_TAYLOR"], errors="ignore")
        out["PRIMARY_BEAM"] = polynomial_beam
        for name, group in list(out.attrs["data_groups"].items()):
            group = dict(group)
            group.pop("primary_beam_taylor", None)
            if "primary_beam" in group or name in (residual_group, model_group):
                group["primary_beam"] = "PRIMARY_BEAM"
            modify_data_groups_xds(
                out, name, group, description="Finalized flat-sky image products."
            )
    if continuum_mode is not None and "frequency" not in out.coords:
        out = out.assign_coords(frequency=[float(out.attrs["reference_frequency"])])
    out.attrs["public_image_convention"] = "flat_sky"
    out.attrs["public_residual_representation"] = "coefficient_space"
    out.attrs["primary_beam_limit"] = float(primary_beam_limit)
    for coordinate in img_xds.coords:
        if coordinate in out.coords:
            out[coordinate].attrs = copy.deepcopy(img_xds[coordinate].attrs)
    return out


def _restoring_beams(
    image, residual_group, reference, beam_fit_params_key, beam_polarization_index
):
    group = image.attrs["data_groups"][residual_group]
    name = group.get("beam_fit_params_sky", group.get(beam_fit_params_key))
    if name is None:
        name = "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
    if name not in image:
        raise KeyError("Restoration requires registered residual beam-fit parameters")
    beam = image[name]
    missing = {"time", "polarization"} - set(beam.dims)
    if missing:
        raise ValueError(
            f"Beam variable {name!r}: Missing dimensions {sorted(missing)}"
        )
    param = "beam_params_label" if "beam_params_label" in beam.dims else "beam_params"
    axis = next(
        d for d in reference.dims if d not in ("time", "polarization", "l", "m")
    )
    for dim in ("frequency", "taylor_term", "psf_taylor_order", POLYNOMIAL, MOMENT):
        if dim in beam.dims:
            if axis == "frequency" and dim == axis:
                continue
            beam = beam.isel({dim: 0}, drop=True)
    beam = beam.isel(polarization=beam_polarization_index, drop=True).expand_dims(
        polarization=reference.polarization
    )
    if axis not in beam.dims:
        beam = beam.expand_dims({axis: reference[axis]}, axis=1)
    beam = beam.broadcast_like(reference.isel(l=0, m=0, drop=True)).transpose(
        "time", axis, "polarization", param
    )
    return beam, param, axis


def restore_finalized_image(
    img_xds,
    *,
    restoration_output="primary_beam_corrected",
    restoring_beam=None,
    processing_function_threads=1,
    beam_fit_params_key="beam_fit_params_point_spread_function",
    beam_polarization_index=0,
    image_data_group_in_residual_name="residual",
    image_data_group_in_model_name="model",
    image_data_group_out_restore_name="restored",
    apparent_inputs=None,
):
    """Restore requested public flat-sky and flat-noise products.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Finalized flat-sky model and residual, plus primary and clean beams.
    restoration_output : {"apparent", "primary_beam_corrected", "both"}, optional
        Requested flat-noise, flat-sky, or both restored products.
    restoring_beam : array_like or None, optional
        Optional target clean beam. The reported residual is taken to this
        beam too, preserving restored = convolved model + reported residual.
    processing_function_threads : int, optional
        Threads supplied to the restoration backend.
    beam_fit_params_key : str, optional
        Residual-group beam-fit role, used unless a transformed residual already
        registers its own beam_fit_params_sky role.
    beam_polarization_index : int, optional
        Polarization whose clean beam is applied to every polarization.
    image_data_group_in_residual_name, image_data_group_in_model_name : str, optional
        Groups containing the finalized inputs.
    image_data_group_out_restore_name : str, optional
        Group for the requested restored image. With both, a separate
        primary_beam_corrected_restored group registers the flat-sky product.
    apparent_inputs : dict or None, optional
        Temporary apparent inputs from finalize_image_products. Entries are
        consumed during restoration and are not retained in the result. When
        absent, apparent inputs are reconstructed on the valid flat-sky domain.

    Returns
    -------
    tuple
        Updated dataset and a one-row restoration timing dataframe.
    """
    import pandas as pd

    from astroviper.processing_functions.imaging.restore import (
        NO_FIT_BEAM,
        _correcting_transforms,
        _residual_to_beam,
        _restore_image_arrays,
        _restored_beam_variable,
        fill_planes_without_fit,
    )

    if img_xds.attrs.get("public_image_convention") != "flat_sky":
        raise ValueError("Restoration requires finalized flat-sky image products")
    if restoration_output not in ("apparent", "primary_beam_corrected", "both"):
        raise ValueError("Unsupported restoration_output")
    if restoring_beam is not None:
        from astroviper.processing_functions.imaging.config_check._predicates import (
            check_restoring_beam,
            check_restoring_beam_within_image,
        )

        check_restoring_beam(restoring_beam)
        check_restoring_beam_within_image(
            restoring_beam,
            [img_xds.sizes["l"], img_xds.sizes["m"]],
            [
                img_xds.l.values[1] - img_xds.l.values[0],
                img_xds.m.values[1] - img_xds.m.values[0],
            ],
        )
    start = time.time()
    out = img_xds.copy(deep=False)
    out.attrs = copy.deepcopy(img_xds.attrs)
    rg, mg = image_data_group_in_residual_name, image_data_group_in_model_name
    mn, rn = _role(out, mg, "sky"), _role(out, rg, "sky")
    reference = out[rn]
    beam, param, axis = _restoring_beams(
        out, rg, reference, beam_fit_params_key, beam_polarization_index
    )
    dims = ("time", axis, "polarization", "l", "m")
    model, residual = out[mn].transpose(*dims), reference.transpose(*dims)
    pixel = abs(float(out.l.values[1] - out.l.values[0]))
    workers = processing_function_threads if processing_function_threads > 0 else -1
    if restoring_beam is not None:
        transforms = _correcting_transforms(beam.values[:, :, 0], restoring_beam, pixel)
    requested = []
    if restoration_output in ("primary_beam_corrected", "both"):
        requested.append(("SKY_PRIMARY_BEAM_CORRECTED_RESTORED", model, residual))
    if restoration_output in ("apparent", "both"):
        if apparent_inputs:
            apparent_model = apparent_inputs.pop("model").transpose(*dims)
            apparent_residual = apparent_inputs.pop("residual").transpose(*dims)
        elif axis == "frequency":
            primary_beam = out[_role(out, rg, "primary_beam")]
            apparent_model, apparent_residual = (
                model * primary_beam,
                residual * primary_beam,
            )
        else:
            primary_beam = out[_role(out, rg, "primary_beam")]
            apparent_model = multiply_taylor_polynomials(model, primary_beam)
            apparent_residual = multiply_taylor_polynomials(residual, primary_beam)
        requested.append(("SKY_RESTORED", apparent_model, apparent_residual))
    for name, model_input, residual_input in requested:
        products, _ = _restore_image_arrays(
            residual_input.values,
            model_input.values,
            beam.values,
            pixel,
            processing_function_threads=processing_function_threads,
            output="apparent",
            restoring_beam=restoring_beam,
        )
        if "apparent" not in products:
            raise RuntimeError(
                "Restoration backend did not return the requested apparent product"
            )
        out[name] = residual.copy(data=products["apparent"])
        out[name].attrs = {
            **reference.attrs,
            "sky_convention": "flat_sky"
            if name.startswith("SKY_PRIMARY")
            else "flat_noise",
        }
        out[name].attrs.pop("residual_representation", None)
        out[name].attrs["primary_beam_convention"] = out[name].attrs["sky_convention"]
        out[name].attrs["description"] = (
            "Restored flat-sky image."
            if name == "SKY_PRIMARY_BEAM_CORRECTED_RESTORED"
            else "Restored flat-noise apparent-sky image."
        )

    restored_names = [name for name, _, _ in requested]
    # Release the temporary apparent arrays before any target-beam residual
    # allocation. Only the public products and flat-sky inputs remain live.
    del requested, model_input, residual_input
    if restoration_output in ("apparent", "both"):
        del apparent_model, apparent_residual

    restored_beams = (
        restoring_beam
        if restoring_beam is not None
        else fill_planes_without_fit(beam.values[:, :, :1], no_fit_beam=NO_FIT_BEAM)
    )
    # Beam-fit parameters are frequency/reference-frequency quantities, not Taylor coefficients.
    beam_for_metadata = (
        beam.rename({param: "beam_params_label"})
        if param != "beam_params_label"
        else beam
    )
    if axis != "frequency":
        beam_for_metadata = beam_for_metadata.isel({axis: 0}, drop=True).expand_dims(
            frequency=out.frequency.values, axis=1
        )
    if axis != "frequency" and restoring_beam is None:
        restored_beams = restored_beams[:, :1]
    out["BEAM_FIT_PARAMS_SKY_RESTORED"] = _restored_beam_variable(
        beam_for_metadata, restored_beams
    )
    if restoring_beam is not None:
        converted = residual.copy(deep=True)
        for (ti, pi), (covariance, ratio) in transforms.items():
            for pol in range(residual.sizes["polarization"]):
                converted.values[ti, pi, pol] = _residual_to_beam(
                    residual.values[ti, pi, pol], covariance, ratio, workers
                )
        out[rn] = converted
        out["BEAM_FIT_PARAMS_SKY_RESIDUAL"] = out.BEAM_FIT_PARAMS_SKY_RESTORED
        group = dict(out.attrs["data_groups"][rg])
        group["beam_fit_params_sky"] = "BEAM_FIT_PARAMS_SKY_RESIDUAL"
        modify_data_groups_xds(
            out,
            rg,
            group,
            description="Residual expressed at the requested restoring beam.",
        )
    shared = {
        k: v
        for k, v in out.attrs["data_groups"][rg].items()
        if k not in ("sky", "date", "description", "beam_fit_params_sky")
    }
    for name in restored_names:
        is_corrected = name == "SKY_PRIMARY_BEAM_CORRECTED_RESTORED"
        group_name = (
            "primary_beam_corrected_restored"
            if is_corrected and restoration_output == "both"
            else image_data_group_out_restore_name
        )
        group = {
            **shared,
            "sky": name,
            "beam_fit_params_sky": "BEAM_FIT_PARAMS_SKY_RESTORED",
        }
        modify_data_groups_xds(
            out,
            group_name,
            group,
            description="Restored flat-sky image."
            if is_corrected
            else "Restored flat-noise image.",
        )
    for coordinate in img_xds.coords:
        if coordinate in out.coords:
            out[coordinate].attrs = copy.deepcopy(img_xds[coordinate].attrs)
    return out, pd.DataFrame({"T_restore": [time.time() - start]})


def public_continuum_layout(img_xds):
    """Expose continuum coefficients and moments on the proposed XRADIO axes.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Finalized continuum image with internal Taylor coordinate names.

    Returns
    -------
    xarray.Dataset
        Public polynomial layout with reference-frequency metadata and no padding.
    """
    out = img_xds
    if "beam_params" in out.dims and "beam_params_label" in out.dims:
        out = out.copy(deep=False)
        for name in list(out.data_vars):
            if "beam_params" in out[name].dims:
                value = out[name].rename(beam_params="beam_params_label")
                out = out.drop_vars(name)
                out[name] = value
        out = out.drop_vars("beam_params", errors="ignore")
    out = out.rename(
        {
            old: new
            for old, new in (
                ("taylor_term", POLYNOMIAL),
                ("psf_taylor_order", MOMENT),
                ("beam_params", "beam_params_label"),
            )
            if old in out.dims
        }
    )
    out.attrs = copy.deepcopy(img_xds.attrs)
    if out.sizes.get("frequency") != 1:
        raise ValueError(
            "Public continuum products require one reference-frequency coordinate"
        )
    reference = float(out.attrs.get("reference_frequency", out.frequency.values[0]))
    spectral = dict(out.frequency.attrs)
    for name in list(out.data_vars):
        if name.startswith("BEAM_FIT_PARAMS_"):
            value = out[name]
            for axis in (POLYNOMIAL, MOMENT):
                if axis in value.dims:
                    value = value.isel({axis: 0}, drop=True)
            if "frequency" not in value.dims:
                value = value.expand_dims(frequency=out.frequency, axis=1)
            out[name] = value
    out.frequency.attrs = spectral

    reference_measure = copy.deepcopy(spectral.get("reference_frequency", {}))
    if not isinstance(reference_measure, dict):
        reference_measure = {}
    reference_measure.setdefault(
        "attrs", {"type": "spectral_coord", "units": "Hz", "observer": "lsrk"}
    )
    reference_measure["data"] = reference
    reference_measure.setdefault("dims", [])
    out.frequency.attrs["reference_frequency"] = copy.deepcopy(reference_measure)
    bandwidth = out.attrs.get(
        "continuum_bandwidth", spectral.get("channel_width", {"data": 0.0})
    )
    if isinstance(bandwidth, dict):
        bandwidth = bandwidth.get("data", 0.0)
    attrs = {
        "polynomial_type": "taylor",
        "reference_frequency": reference_measure,
        "channel_width": {
            "data": float(bandwidth),
            "dims": [],
            "attrs": {"type": "quantity", "units": "Hz"},
        },
    }
    for axis in (POLYNOMIAL, MOMENT):
        if axis in out.dims:
            if axis in img_xds.coords and not np.array_equal(
                img_xds[axis].values, np.arange(out.sizes[axis])
            ):
                raise ValueError(
                    "Public polynomial orders must start at zero and increase by one"
                )
            out = out.assign_coords({axis: np.arange(out.sizes[axis], dtype=np.int64)})
            out[axis].attrs = copy.deepcopy(attrs)
    from xradio.image.schema import IMAGE_SCHEMA_VERSION

    out.attrs["schema_version"] = IMAGE_SCHEMA_VERSION
    return out
