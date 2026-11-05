"""Primary-beam creation for the single-field cube imager."""

import functools

from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def make_primary_beam_single_field(
    img_xds,
    image_params,
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    list_dish_diameters=None,
    list_blockage_diameters=None,
    max_rad_1GHz=None,
    ipower=2,
    float_dtype=None,
    beam_model=None,
):
    """Add an azimuthally-symmetric primary beam to a single-field image dataset.

    Follows the CASA definition: the primary beam is the **power** sensitivity
    pattern of the antenna, the square of the absolute value of the voltage
    pattern (``P = |V| ** 2``).  The beam is an antenna beam model of the
    simulator, evaluated per frequency channel with
    :func:`~astroviper.processing_functions.simulation.antenna_beams.symmetric_beam_response`
    -- the function whose voltage pattern
    :func:`~astroviper.processing_functions.simulation.antenna_beams.sample_jones`
    applies to simulated visibilities -- so it is zero beyond
    ``max_rad_1GHz / (frequency / 1 GHz)`` when the model has a cut, as CASA's
    ``.pb`` image.  By default it is the textbook obscured Airy pattern
    (:func:`~astroviper.processing_functions.imaging.primary_beam.airy_disk.airy_disk_response`)
    without a cut; ``beam_model`` gives another (the CASA ``PBMath1DAiry``
    compatible one, ``func="casa_airy"``, matching the ``.pb`` image of
    ``tclean`` to float32 precision, or the EVLA beam polynomial).  It is
    written as the ``PRIMARY_BEAM``
    data variable, registered under ``image_data_group_out_name``, and
    broadcast across polarization (a single antenna beam is assumed).

    The beam is created in whatever polarization basis ``img_xds`` currently
    carries; because ``PRIMARY_BEAM`` is stamped with ``method="airy_disk"``
    (or ``"beam_polynomial"`` or ``"uniform"``) it is skipped by
    :func:`~astroviper.processing_functions.image_analysis.transform_polarization_basis.transform_polarization_basis`,
    so it is invariant to a later basis change.  If the data group already has a
    ``primary_beam`` entry the function is a no-op.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset with ``time``, ``frequency``, ``polarization``, ``l`` and
        ``m`` coordinates. Modified in place.
    image_params : dict
        Image geometry. Must contain ``"image_size"`` (``(nx, ny)``) and
        ``"cell_size"`` (l, m cell sizes in radians). The image centre is derived
        internally as ``image_size // 2``; the supplied dictionary is not mutated.
    image_data_group_in_name : str, optional
        Image data group whose existing entries are carried into the output
        group. Default ``"residual"``.
    image_data_group_out_name : str, optional
        Name of the data group under which ``PRIMARY_BEAM`` is registered.
        Default ``"residual"``.
    list_dish_diameters : array-like of float, optional
        Antenna dish diameters in metres. Defaults to ``numpy.array([10.7])``,
        the effective ALMA 12 m diameter used by the reference imaging tests.
    list_blockage_diameters : array-like of float, optional
        Sub-reflector blockage diameters in metres. Defaults to
        ``numpy.array([0.75])``.
    max_rad_1GHz : float, optional
        Radius at 1 GHz in radians (scaled to the observing frequency) beyond
        which the beam is zero.  Default ``None``, no cut.
    ipower : int, optional
        Voltage-pattern exponent: ``1`` returns voltage and ``2`` returns the
        power primary beam. Defaults to ``2``.
    float_dtype : numpy.dtype, optional
        Floating-point precision for the primary beam. Defaults to
        ``numpy.float64``.
    beam_model : dict or xarray.Dataset, optional
        Analytic (dict) or beam-polynomial (dataset) antenna beam model, as
        ``beam_models`` of the simulator takes it
        (:mod:`astroviper.utils.beam_models`), evaluated instead of the Airy
        disk (:func:`pack_primary_beam_model`); ``list_dish_diameters``,
        ``list_blockage_diameters`` and ``max_rad_1GHz`` are then not used.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with ``PRIMARY_BEAM`` added (or unchanged if it
        already existed).
    return_df : pandas.DataFrame
        One-row timing frame with the ``T_primary_beam`` column.

    See Also
    --------
    astroviper.processing_functions.simulation.antenna_beams.symmetric_beam_response
    astroviper.processing_functions.imaging.make_point_spread_function.make_point_spread_function_single_field
    astroviper.processing_functions.imaging.correct_sky_by_primary_beam.correct_sky_by_primary_beam
    """
    import time

    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.processing_functions.simulation.antenna_beams import (
        symmetric_beam_response,
    )
    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    if float_dtype is None:
        float_dtype = np.float64
    if list_dish_diameters is None:
        list_dish_diameters = np.array([10.7])
    if list_blockage_diameters is None:
        list_blockage_diameters = np.array([0.75])

    start = time.time()

    data_group = img_xds.attrs["data_groups"][image_data_group_in_name]
    if data_group.get("primary_beam", None) is not None:
        return img_xds, pd.DataFrame({"T_primary_beam": [0.0]})

    image_data_group_in, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified={"primary_beam": "PRIMARY_BEAM"},
        overwrite=False,
    )

    if beam_model is None:
        beam_model = {
            "func": "airy",
            "dish_diameter": float(np.asarray(list_dish_diameters)[0]),
            "blockage_diameter": float(np.asarray(list_blockage_diameters)[0]),
            "max_rad_1GHz": max_rad_1GHz,
        }
    frequency = img_xds.frequency.values
    packed_model = pack_primary_beam_model(beam_model, frequency)
    if packed_model["kind"] == "polynomial":
        method = "beam_polynomial"
    elif packed_model["func"] == "none":
        method = "uniform"
    else:
        method = "airy_disk"

    # Evaluate the beam on the image's (l, m) grid one channel at a time, then
    # broadcast over polarization and add a leading time axis.
    pb = None
    for i_chan, f in enumerate(frequency):
        response = symmetric_beam_response(
            packed_model,
            img_xds.l.values[:, None],
            img_xds.m.values[None, :],
            f,
            ipower,
        )
        if pb is None:
            pb = np.empty((frequency.size, *response.shape), dtype=float_dtype)
        pb[i_chan] = response
        del response
    img_xds["PRIMARY_BEAM"] = xr.DataArray(
        np.tile(pb[None, :, None, :, :], (1, 1, img_xds.sizes["polarization"], 1, 1)),
        dims=("time", "frequency", "polarization", "l", "m"),
    )
    img_xds["PRIMARY_BEAM"].attrs["type"] = "primary_beam"
    img_xds["PRIMARY_BEAM"].attrs["method"] = method

    modify_data_groups_xds(
        img_xds,
        data_group_out_name=image_data_group_out_name,
        data_group_out=image_data_group_out,
        description="Added primary beam to data group.",
    )

    return img_xds, pd.DataFrame({"T_primary_beam": [time.time() - start]})


def pack_primary_beam_model(beam_model, frequency):
    """The packed antenna beam model the primary beam is evaluated with.

    The model as
    :func:`~astroviper.processing_functions.simulation.antenna_beams.evaluate_beam_models`
    gives it to the simulator (a beam polynomial reduced to the row
    :func:`~astroviper.utils.beam_models.beam_polynomial_coefficient_index`
    picks at each frequency), packed.

    Parameters
    ----------
    beam_model : dict or xarray.Dataset
        Analytic or beam-polynomial antenna beam model.
    frequency : array_like
        Frequencies (Hz).

    Returns
    -------
    dict
        One element of
        :func:`~astroviper.processing_functions.simulation.antenna_beams.pack_beam_models`.

    Raises
    ------
    ValueError
        For another kind of model or a beam that is not physical.
    """
    import numpy as np

    from astroviper.processing_functions.simulation.antenna_beams import (
        evaluate_beam_models,
        pack_beam_models,
    )

    # time, phase centre and site are only used for Zernike models
    evaluated, _ = evaluate_beam_models(
        [_checked_primary_beam_model(beam_model)],
        np.zeros(1),
        np.atleast_1d(np.asarray(frequency, dtype=np.float64)),
        np.zeros((1, 2)),
        np.zeros(3),
    )
    return pack_beam_models(evaluated)[0]


def _checked_primary_beam_model(beam_model):
    from astroviper.processing_functions.simulation.antenna_beams import (
        beam_model_kind,
        normalize_beam_model,
    )

    kind = beam_model_kind(beam_model)
    if kind not in ("analytic", "beam_polynomial_coefficients"):
        raise ValueError(
            f"beam_model of kind {kind!r} is not supported: the primary beam is "
            "an analytic or beam-polynomial model, azimuthally symmetric and "
            "independent of the parallactic angle"
        )
    beam_model = normalize_beam_model(beam_model)
    if kind == "analytic":
        dish, blockage = beam_model["dish_diameter"], beam_model["blockage_diameter"]
        if beam_model["func"] not in ("casa_airy", "airy", "none") or not (
            dish > 0.0 and 0.0 <= blockage < dish
        ):
            raise ValueError(
                "an analytic beam_model needs func 'casa_airy', 'airy' or 'none', "
                "dish_diameter > 0 and 0 <= blockage_diameter < dish_diameter; "
                f"got {beam_model}"
            )
    max_rad = (beam_model if kind == "analytic" else beam_model.attrs)["max_rad_1GHz"]
    if max_rad is not None and not max_rad > 0.0:
        raise ValueError(f"beam_model needs max_rad_1GHz > 0; got {max_rad}")
    return beam_model


@functools.cache
def _evla_beam_polynomial_coefficients():
    from astroviper.utils.beam_models import read_beam_polynomial_coefficients

    return read_beam_polynomial_coefficients("EVLA_")


PRIMARY_BEAM_PARAMS_KEYS = (
    "telescope",
    "effective_dish_diameter",
    "blockage_diameter",
    "max_rad_1GHz",
    "beam_model",
)


def resolve_primary_beam_params(primary_beam_params=None):
    """Keyword arguments of :func:`make_primary_beam_single_field` for ``primary_beam_params``.

    Parameters
    ----------
    primary_beam_params : dict, optional
        ``beam_model`` (an analytic dict or a beam-polynomial dataset, as
        ``beam_models`` of the simulator takes it; alone), or ``telescope``
        (a key of :data:`astroviper.utils.beam_models.AIRY_DISK_MODELS`,
        default ``"alma"``, which names ``airy_disk_model(telescope)``, the
        textbook Airy disk without a cut; ``"vla"`` names
        ``read_beam_polynomial_coefficients("EVLA_")``, the EVLA beam
        polynomial at the exact radius) and the numbers
        ``effective_dish_diameter``, ``blockage_diameter`` (metres; refused for
        ``"vla"``) and ``max_rad_1GHz`` (radians, or ``None`` for no cut;
        ``"vla"`` needs one), which override the telescope model's values.

    Returns
    -------
    dict
        ``{}`` for ``None`` (the function's ALMA defaults), else
        ``beam_model``.

    Raises
    ------
    ValueError
        For an unknown key or telescope, ``beam_model`` with another key, a
        Zernike aperture or Jones image ``beam_model``, a dish or blockage
        diameter or no cut given with ``"vla"``, or a beam that is not physical.
    TypeError
        For a ``beam_model`` that is not a beam model.
    """
    from astroviper.utils.beam_models import AIRY_DISK_MODELS, airy_disk_model

    if primary_beam_params is None:
        return {}
    unknown = set(primary_beam_params) - set(PRIMARY_BEAM_PARAMS_KEYS)
    if unknown:
        raise ValueError(
            f"unknown primary_beam_params key(s) {sorted(unknown)}; expected "
            f"{list(PRIMARY_BEAM_PARAMS_KEYS)}"
        )
    if "beam_model" in primary_beam_params:
        refused = sorted(set(primary_beam_params) - {"beam_model"})
        if refused:
            raise ValueError(
                f"primary_beam_params {refused} do not apply with beam_model, "
                "which holds its own values"
            )
        return {
            "beam_model": _checked_primary_beam_model(primary_beam_params["beam_model"])
        }
    telescope = str(primary_beam_params.get("telescope", "alma"))
    if telescope.lower() == "vla":
        refused = sorted(
            {"effective_dish_diameter", "blockage_diameter"} & set(primary_beam_params)
        )
        if refused:
            raise ValueError(
                f"primary_beam_params {refused} do not apply to telescope "
                f"{telescope!r}, CASA's EVLA beam polynomial"
            )
        model = _evla_beam_polynomial_coefficients()
        max_rad = primary_beam_params.get("max_rad_1GHz", model.attrs["max_rad_1GHz"])
        if max_rad is None or not float(max_rad) > 0.0:
            raise ValueError(
                f"primary_beam_params need max_rad_1GHz > 0; got {max_rad}"
            )
        return {"beam_model": model.assign_attrs(max_rad_1GHz=float(max_rad))}
    if telescope.lower() not in AIRY_DISK_MODELS:
        raise ValueError(
            f"primary_beam_params['telescope'] {telescope!r} is not one of "
            f"{sorted(AIRY_DISK_MODELS)}"
        )
    model = airy_disk_model(telescope)
    dish = float(
        primary_beam_params.get("effective_dish_diameter", model["dish_diameter"])
    )
    blockage = float(
        primary_beam_params.get("blockage_diameter", model["blockage_diameter"])
    )
    max_rad = primary_beam_params.get("max_rad_1GHz", model["max_rad_1GHz"])
    if max_rad is not None:
        max_rad = float(max_rad)
    if not (
        dish > 0.0 and 0.0 <= blockage < dish and (max_rad is None or max_rad > 0.0)
    ):
        raise ValueError(
            "primary_beam_params need effective_dish_diameter > 0, "
            "0 <= blockage_diameter < effective_dish_diameter and max_rad_1GHz > 0; "
            f"got {dish}, {blockage}, {max_rad}"
        )
    return {
        "beam_model": {
            **model,
            "dish_diameter": dish,
            "blockage_diameter": blockage,
            "max_rad_1GHz": max_rad,
        }
    }
