"""Read-only checks of gridding recipes before allocating image grids."""

from astroviper.processing_functions.imaging.config_check._common import (
    require_finite_real,
    require_integer,
    require_mapping,
)


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
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        DEFAULT_GRIDDING_KERNEL_PARAMS,
    )

    # --- Recognized recipe ---
    # Misspellings and false-like non-mappings must not select the default.
    if params is None:
        return
    require_mapping("gridding_kernel_params", params)
    unknown = set(params) - set(DEFAULT_GRIDDING_KERNEL_PARAMS)
    if unknown:
        raise ValueError(f"Unknown gridding_kernel_params keys: {sorted(unknown)}.")
    recipe = {**DEFAULT_GRIDDING_KERNEL_PARAMS, **params}
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
    """Require a padding that leaves a design region below the alias boundary."""
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        MIN_DESIGN_PADDING,
    )

    if fft_padding is None:
        raise ValueError("This gridding kernel requires fft_padding >= 1.05.")
    padding = require_finite_real("fft_padding", fft_padding, scalar=True)
    if padding < MIN_DESIGN_PADDING:
        raise ValueError("This gridding kernel requires fft_padding >= 1.05.")


def check_gridding_kernel(params, *, fft_padding=None):
    """Check the recipe and cached numerical image-edge correction."""
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        make_gridding_kernel,
    )

    if params is not None:
        # The builder delegates its recipe and padding checks to this module.
        # Its numerical correction guard is also used by direct builder calls.
        make_gridding_kernel(params, fft_padding=fft_padding)


def check_kernel_options(kernel_type, interpolation):
    """Check options shared by recipe builders and explicitly supplied tables."""
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        KERNEL_TYPES,
    )

    if kernel_type not in KERNEL_TYPES:
        raise ValueError(
            f"gridding_kernel_params['type'] must be one of {KERNEL_TYPES}."
        )
    if interpolation not in ("nearest", "linear"):
        raise ValueError("kernel interpolation must be 'nearest' or 'linear'.")


def check_casa_support(support):
    """Schwab's tabulated approximation is defined for exactly seven grid points."""
    if support != 7:
        raise ValueError("prolate_spheroidal_casa requires kernel support=7.")
