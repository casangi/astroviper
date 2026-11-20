"""Read-only checks of gridding recipes before allocating image grids."""

from astroviper.processing_functions.imaging.config_check._common import (
    require_finite_real,
    require_integer,
    require_mapping,
)


def check_gridder(gridder):
    """Require a name of the gridder registry.

    Parameters
    ----------
    gridder : str
        A key of
        :data:`~astroviper.processing_functions.imaging.gridders.gridder_registry.GRIDDERS`.

    Returns
    -------
    type
        The gridder's class.

    Raises
    ------
    ValueError
        For any other value, a gridder class included.
    """
    from astroviper.processing_functions.imaging.gridders.gridder_registry import (
        GRIDDERS,
    )

    if not isinstance(gridder, str) or gridder not in GRIDDERS:
        raise ValueError(f"gridder must be one of {tuple(GRIDDERS)}.")
    return GRIDDERS[gridder]


def check_gridder_kernel(gridder, params):
    """Require no kernel recipe for a gridder without the anti-alias term.

    Parameters
    ----------
    gridder : str
        A key of
        :data:`~astroviper.processing_functions.imaging.gridders.gridder_registry.GRIDDERS`.
    params : mapping or None
        Kernel recipe.

    Returns
    -------
    type
        The gridder's class.

    Raises
    ------
    ValueError
        For a gridder :func:`check_gridder` refuses, or a recipe other than
        ``None`` for a gridder that uses no kernel.
    """
    gridder_class = check_gridder(gridder)
    if "anti_alias" not in gridder_class.gcf_terms and params is not None:
        raise ValueError(
            f"gridder={gridder!r} has no gridding kernel: "
            "gridding_kernel_params must be None."
        )
    return gridder_class


def check_gridder_continuum_mode(gridder, continuum_mode):
    """Require MFS for a gridder that makes no visibility grid.

    Parameters
    ----------
    gridder : str
        A key of
        :data:`~astroviper.processing_functions.imaging.gridders.gridder_registry.GRIDDERS`.
    continuum_mode : str
        Continuum construction.

    Raises
    ------
    ValueError
        For a gridder :func:`check_gridder` refuses, or MVC, which is made of
        uv grids, with a gridder without the ``"visibility"`` role.
    """
    if (
        "visibility" not in check_gridder(gridder).grid_roles
        and continuum_mode != "mfs"
    ):
        raise ValueError("DFT continuum imaging requires continuum_mode='mfs'.")


def check_gridding_kernel_parameters(params):
    """Check the recipe schema without constructing a numerical kernel.

    Parameters
    ----------
    params : mapping or None
        Explicit kernel settings. Missing keys retain the builder's defaults.

    Notes
    -----
    The production builder caches kernels by their recipe, so this preflight
    and later gridding reuse the same design. No image or visibility is read.
    """
    # --- Recognized recipe ---
    # Misspellings and false-like non-mappings must not select the default.
    if params is None:
        return
    require_mapping("gridding_kernel_params", params)

    from astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term import (
        CASA_GRIDDING_KERNEL_PARAMS,
        DEFAULT_GRIDDING_KERNEL_PARAMS,
    )

    unknown = set(params) - set(DEFAULT_GRIDDING_KERNEL_PARAMS)
    if unknown:
        raise ValueError(f"Unknown gridding_kernel_params keys: {sorted(unknown)}.")
    # CASA's fixed support and sampling apply to omitted keys of that type.
    defaults = (
        CASA_GRIDDING_KERNEL_PARAMS
        if params.get("type") == "prolate_spheroidal_casa"
        else DEFAULT_GRIDDING_KERNEL_PARAMS
    )
    recipe = {**defaults, **params}
    check_kernel_options(recipe["type"], recipe["interpolation"])

    # --- Table sampling ---
    # Stencils need an odd support and a positive integral sampling density.
    require_integer("kernel support", recipe["support"], minimum=3)
    if recipe["support"] % 2 == 0:
        raise ValueError("kernel support must be odd.")
    require_integer("kernel oversampling", recipe["oversampling"], minimum=1)

    # --- CASA support ---
    # The tabulated CASA spheroidal approximation is defined for support 7.
    if recipe["type"] == "prolate_spheroidal_casa":
        check_casa_support(recipe["support"])
        return


def check_kernel_padding(fft_padding):
    """Require a padding that leaves a design region below the alias boundary.

    Parameters
    ----------
    fft_padding : float
        Finite scalar padding factor of at least 1.05.
    """
    from astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term import (
        MIN_DESIGN_PADDING,
    )

    if fft_padding is None:
        raise ValueError("This gridding kernel requires fft_padding >= 1.05.")
    padding = require_finite_real("fft_padding", fft_padding, scalar=True)
    if padding < MIN_DESIGN_PADDING:
        raise ValueError("This gridding kernel requires fft_padding >= 1.05.")


def check_gridding_kernel(params, *, fft_padding=None):
    """Check the recipe and cached numerical image-edge correction.

    Parameters
    ----------
    params : mapping or None
        Kernel recipe; None selects the production default.
    fft_padding : float or None, optional
        Padding factor supplied to the cached numerical kernel builder.
    """
    if params is not None:
        require_mapping("gridding_kernel_params", params)

    from astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term import (
        make_gridding_kernel,
    )

    # None selects the production default, which has the same design/padding
    # requirements as an explicit recipe. The cached builder owns those rules.
    make_gridding_kernel(params, fft_padding=fft_padding)


def check_kernel_options(kernel_type, interpolation):
    """Check options shared by recipe builders and explicitly supplied tables.

    Parameters
    ----------
    kernel_type : str
        Kernel type registered by the production builder.
    interpolation : {'nearest', 'linear'}
        Lookup interpolation used for the kernel table.
    """
    from astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term import (
        KERNEL_TYPES,
    )

    if kernel_type not in KERNEL_TYPES:
        raise ValueError(
            f"gridding_kernel_params['type'] must be one of {KERNEL_TYPES}."
        )
    if interpolation not in ("nearest", "linear"):
        raise ValueError("kernel interpolation must be 'nearest' or 'linear'.")


def check_casa_support(support):
    """Schwab's tabulated approximation is defined for exactly seven grid points.

    Parameters
    ----------
    support : int
        Stencil width, which must equal seven for the CASA kernel.
    """
    if support != 7:
        raise ValueError("prolate_spheroidal_casa requires kernel support=7.")
