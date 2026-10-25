"""Accelerated model steps for the continuum imaging cycle.

Each model update deconvolves the refreshed residual into a model increment.
The functions below turn that increment into the step that is added to the
model, following Müller, Hsieh and Bhatnagar (2026, A&A 706, A77): a
conjugate gradient step or a Nesterov momentum step. Neither needs another
residual update. The functions are stateless; the state of the preceding
imaging cycle is passed in and returned explicitly.

Every dimension other than ``taylor_term``, ``l`` and ``m`` is a plane
dimension. Inner products sum over the Taylor terms and pixels of a plane in
float64, so all Taylor terms of a plane share one step length and one
coefficient.
"""

from astroviper.utils.param_docs import shares_param_docs

_REDUCTION_DIMS = ("taylor_term", "l", "m")


def _plane_layout(model_increment, arrays):
    """Validate the step inputs and return their plane and reduction dimensions."""
    import numpy as np

    if not {"l", "m"} <= set(model_increment.dims):
        raise ValueError(
            "model_increment must contain the 'l' and 'm' dimensions; received "
            f"{model_increment.dims}."
        )
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(
                f"{name} must be real floating point; received {array.dtype}."
            )
    for name, array in arrays.items():
        if array.dtype != model_increment.dtype:
            raise TypeError(
                f"{name} has dtype {array.dtype}; expected {model_increment.dtype}."
            )
        if set(array.dims) != set(model_increment.dims) or any(
            array.sizes[dim] != size for dim, size in model_increment.sizes.items()
        ):
            raise ValueError(
                f"{name} has sizes {dict(array.sizes)}; expected "
                f"{dict(model_increment.sizes)}."
            )
    reduction_dims = tuple(
        dim for dim in _REDUCTION_DIMS if dim in model_increment.dims
    )
    plane_dims = tuple(dim for dim in model_increment.dims if dim not in reduction_dims)
    return plane_dims, reduction_dims


def _plain_step_mask(plain_step_planes, plane_dims, sizes):
    """Return the plain-step planes as a boolean array in plane-dimension order."""
    import numpy as np

    if plain_step_planes is None:
        return np.zeros(tuple(sizes[dim] for dim in plane_dims), dtype=bool)
    if (
        plain_step_planes.dtype != bool
        or set(plain_step_planes.dims) != set(plane_dims)
        or any(plain_step_planes.sizes[dim] != sizes[dim] for dim in plane_dims)
    ):
        raise ValueError(
            "plain_step_planes must be a boolean array over the plane dimensions "
            f"{plane_dims}; received {plain_step_planes.dtype} with sizes "
            f"{dict(plain_step_planes.sizes)}."
        )
    return plain_step_planes.transpose(*plane_dims).values


def _state_values(state_xds, cycle_acceleration, layouts, sizes):
    """Return the state arrays in their ``(dims, dtype)`` layouts, or ``None``."""
    import numpy as np

    if state_xds is None:
        return None
    if state_xds.attrs.get("cycle_acceleration") != cycle_acceleration:
        raise ValueError(
            "cycle_acceleration_state_xds belongs to "
            f"{state_xds.attrs.get('cycle_acceleration')!r}; expected "
            f"{cycle_acceleration!r}."
        )
    values = []
    for name, (dims, dtype) in layouts.items():
        variable = state_xds.get(name)
        expected = {dim: sizes[dim] for dim in dims}
        if (
            variable is None
            or variable.dtype != dtype
            or set(variable.dims) != set(dims)
            or any(variable.sizes[dim] != size for dim, size in expected.items())
        ):
            raise ValueError(
                f"cycle_acceleration_state_xds[{name!r}] is missing or does not "
                f"match {np.dtype(dtype)} {expected}."
            )
        values.append(variable.transpose(*dims).values)
    return values


def _inner_product(first, second, n_reduction_dims):
    """Sum ``first * second`` over the trailing reduction axes in float64."""
    import numpy as np

    axes = "abc"[:n_reduction_dims]
    return np.einsum(f"...{axes},...{axes}->...", first, second, dtype=np.float64)


def _ramp(momentum_coefficient, cycles):
    """Return the capped coefficient ``min(mu, (j - 1) / (j + 2))`` in float64."""
    import numpy as np

    return np.minimum(momentum_coefficient, (cycles - 1) / (cycles + 2))


@shares_param_docs
def conjugate_gradient_model_step(
    residual,
    residual_after_model_update,
    model_increment,
    plain_step_planes=None,
    cycle_acceleration_state_xds=None,
):
    """Turn a model increment into a conjugate gradient step.

    The search direction ``p = z + beta p_previous`` combines the increment
    ``z`` with the previous direction, and the step is ``alpha p`` with the
    step length ``alpha = <r, p> / <p, B p>`` of Müller, Hsieh and Bhatnagar
    (2026, Alg. 4). ``B z = r - r_after`` is the deconvolver's own residual
    change, so ``B p`` needs no residual update. ``beta`` is the
    Hestenes-Stiefel coefficient of the true residual change
    ``y = r - r_previous`` between imaging cycles, ``beta = <z, y> /
    (-<p_previous, y>)``, which makes consecutive directions conjugate with
    respect to the Hessian of the residual update.

    A plane restarts from ``p = z`` with ``alpha = <r, z> / <z, B z>`` when
    the numerator ``<z, y>`` or the denominator ``-<p_previous, y>`` of
    ``beta`` is not positive, when ``beta`` is not finite, or when ``<r, p>``
    or ``<p, B p>`` is not positive. It takes the plain increment
    (``alpha = 1``, ``p = z``) when it is listed in ``plain_step_planes``, or
    when that line search fails too, for example for a zero increment.

    Parameters
    ----------
    residual : xarray.DataArray
        Residual ``r`` before the model update, with dimensions including
        ``l`` and ``m`` and optionally ``taylor_term``.
    residual_after_model_update : xarray.DataArray
        The deconvolver's own residual after the model update.
    model_increment : xarray.DataArray
        Model increment ``z`` of the model update, which started from an
        empty model.
    plain_step_planes : xarray.DataArray, optional
        Boolean over the plane dimensions. ``True`` planes take the plain
        increment and restart the search direction. The continuum node task
        lists the planes whose model update brought the residual below the
        global threshold (within ``threshold_tolerance``) and the planes whose
        residual entropy the entropy stop follows.
    cycle_acceleration_state_xds : xarray.Dataset, optional
        State returned by the preceding call, or ``None`` in the first
        imaging cycle.

    Returns
    -------
    model_step : xarray.DataArray
        Step to add to the model, with the layout of ``model_increment``.
    cycle_acceleration_state_xds : xarray.Dataset
        ``SKY_MODEL_DIRECTION`` (``p``), ``SKY_MODEL_DIRECTION_CONVOLVED``
        (``B p``) and ``SKY_RESIDUAL_PREVIOUS`` (``r``, sharing memory with
        ``residual``) for the next imaging cycle.

    Raises
    ------
    TypeError
        If an input is not real floating point or the dtypes differ.
    ValueError
        If the dimensions, sizes, ``plain_step_planes`` or the state do not
        match.
    """
    import numpy as np
    import toolviper.utils.logger as logger
    import xarray as xr

    plane_dims, reduction_dims = _plane_layout(
        model_increment,
        {
            "residual": residual,
            "residual_after_model_update": residual_after_model_update,
            "model_increment": model_increment,
        },
    )
    template = model_increment.transpose(*plane_dims, *reduction_dims)
    residual_values = residual.transpose(*template.dims).values
    increment = template.values
    increment_convolved = (
        residual_values - residual_after_model_update.transpose(*template.dims).values
    )
    plain = _plain_step_mask(plain_step_planes, plane_dims, model_increment.sizes)
    state = _state_values(
        cycle_acceleration_state_xds,
        "conjugate_gradient",
        dict.fromkeys(
            (
                "SKY_MODEL_DIRECTION",
                "SKY_MODEL_DIRECTION_CONVOLVED",
                "SKY_RESIDUAL_PREVIOUS",
            ),
            (template.dims, template.dtype),
        ),
        template.sizes,
    )
    n_reduction_dims = len(reduction_dims)
    plane_shape = plain.shape + (1,) * n_reduction_dims

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r_z = _inner_product(residual_values, increment, n_reduction_dims)
        z_bz = _inner_product(increment, increment_convolved, n_reduction_dims)
        beta = np.zeros_like(r_z)
        delta, kappa = r_z, z_bz
        use_direction = np.zeros_like(plain)
        if state is not None:
            direction, direction_convolved, previous_residual = state
            r_p = _inner_product(residual_values, direction, n_reduction_dims)
            numerator = r_z - _inner_product(
                previous_residual, increment, n_reduction_dims
            )
            denominator = (
                _inner_product(previous_residual, direction, n_reduction_dims) - r_p
            )
            hestenes_stiefel = numerator / denominator
            use_direction = (
                ~plain
                & (denominator > 0)
                & (numerator > 0)
                & np.isfinite(hestenes_stiefel)
            )
            beta = np.where(use_direction, hestenes_stiefel, 0.0)
            delta = r_z + beta * r_p
            kappa = (
                z_bz
                + beta
                * (
                    _inner_product(increment, direction_convolved, n_reduction_dims)
                    + _inner_product(direction, increment_convolved, n_reduction_dims)
                )
                + beta**2
                * _inner_product(direction, direction_convolved, n_reduction_dims)
            )
            use_direction &= (delta > 0) & (kappa > 0) & np.isfinite(delta / kappa)
            beta = np.where(use_direction, beta, 0.0)
            delta = np.where(use_direction, delta, r_z)
            kappa = np.where(use_direction, kappa, z_bz)
        alpha = delta / kappa
        use_increment = (
            ~plain & ~use_direction & (r_z > 0) & (z_bz > 0) & np.isfinite(alpha)
        )
        line_search = use_direction | use_increment
        alpha = np.where(line_search, alpha, 1.0)

        if use_direction.any():
            use_direction_planes = use_direction.reshape(plane_shape)
            beta_planes = beta.astype(increment.dtype).reshape(plane_shape)
            new_direction = np.multiply(beta_planes, direction)
            new_direction += increment
            np.copyto(new_direction, increment, where=~use_direction_planes)
            new_direction_convolved = np.multiply(beta_planes, direction_convolved)
            new_direction_convolved += increment_convolved
            np.copyto(
                new_direction_convolved,
                increment_convolved,
                where=~use_direction_planes,
            )
        else:
            new_direction = increment.copy()
            new_direction_convolved = increment_convolved
        model_step = np.multiply(
            alpha.astype(increment.dtype).reshape(plane_shape), new_direction
        )
        np.copyto(model_step, increment, where=~line_search.reshape(plane_shape))

    case = np.where(
        use_direction,
        "conjugate",
        np.where(use_increment, "restart", np.where(plain, "plain", "fallback")),
    )
    logger.debug(
        f"Conjugate gradient model step over {plane_dims}: alpha {alpha.tolist()}, "
        f"beta {beta.tolist()}, delta {delta.tolist()}, kappa {kappa.tolist()}, "
        f"case {case.tolist()}"
    )

    def wrap(values):
        return template.copy(deep=False, data=values).transpose(*model_increment.dims)

    return wrap(model_step), xr.Dataset(
        {
            "SKY_MODEL_DIRECTION": wrap(new_direction),
            "SKY_MODEL_DIRECTION_CONVOLVED": wrap(new_direction_convolved),
            "SKY_RESIDUAL_PREVIOUS": wrap(residual_values),
        },
        attrs={"cycle_acceleration": "conjugate_gradient"},
    )


@shares_param_docs
def nesterov_momentum_model_step(
    residual,
    model_increment,
    plain_step_planes=None,
    cycle_acceleration_state_xds=None,
    momentum_coefficient=0.5,
):
    """Turn a model increment into a Nesterov momentum step.

    With the momentum ``v = mu_j v_previous + z`` of the increment ``z``, the
    step is ``z + mu_j v``, so the next residual update is made at an
    extrapolated model. With a constant coefficient this is Alg. 6 of
    Müller, Hsieh and Bhatnagar (2026). Here the coefficient follows the
    capped ramp ``mu_j = min(mu, (j - 1) / (j + 2))``, with ``mu`` the
    ``momentum_coefficient`` and ``j`` the imaging cycles since the plane's
    last start or restart, this one included: 0, 1/4, 2/5, 1/2, ... up to
    ``mu``. The first step after a start or restart is therefore the plain
    increment ``z``, which starts the momentum ``v = z``; Alg. 6 keeps ``mu``
    constant, so its first step is ``(1 + mu) z``.

    ``v`` is updated with the current ``mu_j``, not with the ``mu_(j-1)`` of
    the preceding step as in the look-ahead form of Nesterov momentum for a
    varying coefficient (Bengio, Boulanger-Lewandowski and Pascanu 2013), so
    ``v`` is not the displacement of the models before extrapolation. The
    first increment ``z_1`` enters the second step with the weight
    ``mu_2**2``: for ``mu >= 1/4`` the second step is ``1.25 z_2 + z_1 / 16``,
    where the look-ahead form takes ``1.25 z_2``. With deep MT-MFS model
    updates on simulated data, this form reached the threshold and the
    look-ahead form ended on ``max_iter``: here the second step raised the
    residual peak, the gradient restart fired in the third model update, and
    the later steps were plain (measurements in
    ``image_continuum_single_field``).

    A plane resets the momentum, takes the plain increment and restarts the
    ramp when it is listed in ``plain_step_planes``, when its increment is
    zero, or when ``<r, mu_j v_previous + z> <= 0`` (gradient restart). The
    momentum also restarts when the residual norm ``||r||`` exceeds that of
    the preceding call after a step with momentum, that is with ``mu_j > 0``:
    a growth after the plain first step, or with a zero
    ``momentum_coefficient``, is not counted. When that happens a second
    time, the plane takes the plain increment in this and every later call.
    Inner products and norms sum over the Taylor terms and pixels of a
    plane.

    Parameters
    ----------
    residual : xarray.DataArray
        Residual ``r`` before the model update, with dimensions including
        ``l`` and ``m`` and optionally ``taylor_term``.
    model_increment : xarray.DataArray
        Model increment ``z`` of the model update, which started from an
        empty model.
    plain_step_planes : xarray.DataArray, optional
        Boolean over the plane dimensions. ``True`` planes take the plain
        increment and reset the momentum. The continuum node task lists the
        planes that the cycle controller's stop test would stop on the
        residual the model update ended with (residual below the global
        threshold within ``threshold_tolerance``, ``max_iter`` or
        ``max_cycles`` used up, a zero mask, no progress or divergence) and
        the planes whose residual entropy the entropy stop follows.
    cycle_acceleration_state_xds : xarray.Dataset, optional
        State returned by the preceding call, or ``None`` in the first
        imaging cycle.
    momentum_coefficient : float, optional
        Cap ``mu`` of the momentum coefficient
        ``mu_j = min(mu, (j - 1) / (j + 2))`` in
        ``cycle_acceleration="nesterov_momentum"``, with ``j`` the imaging
        cycles since a plane's last start or restart; in ``[0, 1)``. 0 gives
        the plain imaging cycle. Default 0.5, the constant coefficient of
        Müller, Hsieh and Bhatnagar (2026), which the ramp reaches at ``j = 4``.
        A higher cap acts only on planes that run longer without a restart
        (0.9 at ``j = 28``). On simulated data with shallow MT-MFS model
        updates, caps of 0.7 and 0.9 needed 18 imaging cycles, 0.5 needed 22
        and ``"none"`` 33; the paper found models oscillating with constant
        coefficients of 0.7 to 0.9.

    Returns
    -------
    model_step : xarray.DataArray
        Step to add to the model, with the layout of ``model_increment``.
    cycle_acceleration_state_xds : xarray.Dataset
        ``SKY_MODEL_MOMENTUM`` (``v``) and, per plane,
        ``SKY_RESIDUAL_PREVIOUS_NORM`` (``||r||``), ``MOMENTUM_CYCLES`` (``j``,
        or 0 where the momentum was reset), ``RESIDUAL_NORM_GREW`` (the
        residual norm has grown after a step with momentum) and
        ``MOMENTUM_DISABLED`` for the next imaging cycle.

    Raises
    ------
    TypeError
        If an input is not real floating point or the dtypes differ.
    ValueError
        If the dimensions, sizes, ``plain_step_planes`` or the state do not
        match.
    """
    import numpy as np
    import toolviper.utils.logger as logger
    import xarray as xr

    plane_dims, reduction_dims = _plane_layout(
        model_increment,
        {"residual": residual, "model_increment": model_increment},
    )
    template = model_increment.transpose(*plane_dims, *reduction_dims)
    residual_values = residual.transpose(*template.dims).values
    increment = template.values
    plain = _plain_step_mask(plain_step_planes, plane_dims, model_increment.sizes)
    state = _state_values(
        cycle_acceleration_state_xds,
        "nesterov_momentum",
        {
            "SKY_MODEL_MOMENTUM": (template.dims, template.dtype),
            "SKY_RESIDUAL_PREVIOUS_NORM": (plane_dims, np.float64),
            "MOMENTUM_CYCLES": (plane_dims, np.int64),
            "RESIDUAL_NORM_GREW": (plane_dims, bool),
            "MOMENTUM_DISABLED": (plane_dims, bool),
        },
        template.sizes,
    )
    n_reduction_dims = len(reduction_dims)
    plane_shape = plain.shape + (1,) * n_reduction_dims
    reduction_axes = tuple(range(plain.ndim, increment.ndim))
    momentum_coefficient = float(momentum_coefficient)

    with np.errstate(invalid="ignore", over="ignore"):
        residual_norm = np.sqrt(
            _inner_product(residual_values, residual_values, n_reduction_dims)
        )
        descent = _inner_product(residual_values, increment, n_reduction_dims)
        if state is None:
            cycles = np.ones(plain.shape, dtype=np.int64)
            coefficients = _ramp(momentum_coefficient, cycles)
            momentum = increment.copy()
            grew = grew_before = disabled = np.zeros_like(plain)
        else:
            previous_momentum, previous_norm, previous_cycles, grew_before, disabled = (
                state
            )
            cycles = previous_cycles + 1
            coefficients = _ramp(momentum_coefficient, cycles)
            descent += coefficients * _inner_product(
                residual_values, previous_momentum, n_reduction_dims
            )
            momentum = np.multiply(
                coefficients.astype(increment.dtype).reshape(plane_shape),
                previous_momentum,
            )
            momentum += increment
            # Only a step with momentum (mu_j > 0) is held responsible for the
            # growth: not the plain first step after a start or restart, and
            # no step with a zero momentum_coefficient.
            with_momentum = (_ramp(momentum_coefficient, previous_cycles) > 0) & (
                np.any(previous_momentum, axis=reduction_axes)
            )
            grew = with_momentum & (residual_norm > previous_norm)
            disabled = disabled | (grew & grew_before)
        frozen = ~np.any(increment, axis=reduction_axes)
        accelerate = (
            ~plain & ~frozen & ~disabled & ~grew & (descent > 0) & np.isfinite(descent)
        )
        accelerate_planes = accelerate.reshape(plane_shape)
        np.copyto(momentum, 0, where=~accelerate_planes)
        model_step = np.multiply(
            coefficients.astype(increment.dtype).reshape(plane_shape), momentum
        )
        model_step += increment
        np.copyto(model_step, increment, where=~accelerate_planes)
        cycles = np.where(accelerate, cycles, 0)

    case = np.where(
        accelerate,
        "momentum",
        np.where(
            plain,
            "plain",
            np.where(
                frozen,
                "frozen",
                np.where(disabled, "disabled", np.where(grew, "grew", "restart")),
            ),
        ),
    )
    logger.debug(
        f"Nesterov momentum model step over {plane_dims} with coefficient "
        f"{momentum_coefficient}: mu_j {coefficients.tolist()}, <r, mu_j v + z> "
        f"{descent.tolist()}, ||r|| {residual_norm.tolist()}, case {case.tolist()}"
    )

    def wrap(values):
        return template.copy(deep=False, data=values).transpose(*model_increment.dims)

    return wrap(model_step), xr.Dataset(
        {
            "SKY_MODEL_MOMENTUM": wrap(momentum),
            "SKY_RESIDUAL_PREVIOUS_NORM": (plane_dims, residual_norm),
            "MOMENTUM_CYCLES": (plane_dims, cycles),
            "RESIDUAL_NORM_GREW": (plane_dims, grew_before | grew),
            "MOMENTUM_DISABLED": (plane_dims, disabled),
        },
        attrs={"cycle_acceleration": "nesterov_momentum"},
    )
