"""Read-only checks of gridding recipes before allocating image grids."""

from astroviper.processing_functions.imaging.config_check._common import (
    require_finite_real,
    require_integer,
    require_mapping,
)


def check_gridding_kernel(params, *, fft_padding=None):
    """Check a kernel recipe and its parameter-derived edge correction.

    Parameters
    ----------
    params : mapping or None
        Explicit kernel settings. Missing keys retain the builder's defaults.
    fft_padding : float or None
        Padding used to design non-CASA kernels.

    Notes
    -----
    The production builder caches kernels by their recipe, so this preflight
    and later gridding reuse the same design. No image or visibility is read.
    """
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        DEFAULT_GRIDDING_KERNEL_PARAMS,
        KERNEL_TYPES,
        MIN_DESIGN_PADDING,
        make_gridding_kernel,
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
    if recipe["type"] not in KERNEL_TYPES:
        raise ValueError(
            f"gridding_kernel_params['type'] must be one of {KERNEL_TYPES}."
        )

    # --- Table sampling ---
    # Stencils need an odd support and a positive integral sampling density.
    require_integer("kernel support", recipe["support"], minimum=3)
    if recipe["support"] % 2 == 0:
        raise ValueError("kernel support must be odd.")
    require_integer("kernel oversampling", recipe["oversampling"], minimum=1)
    if recipe["interpolation"] not in ("nearest", "linear"):
        raise ValueError("kernel interpolation must be 'nearest' or 'linear'.")

    # --- CASA support ---
    # The tabulated CASA spheroidal approximation is defined for support 7.
    if recipe["type"] == "prolate_spheroidal_casa":
        if recipe["support"] != 7:
            raise ValueError("prolate_spheroidal_casa requires kernel support=7.")
        return

    # --- Designed image region ---
    # Other kernels need padding to suppress aliases over the retained image.
    if fft_padding is None:
        raise ValueError("This gridding kernel requires fft_padding >= 1.05.")
    padding = require_finite_real("fft_padding", fft_padding, scalar=True)
    if padding < MIN_DESIGN_PADDING:
        raise ValueError("This gridding kernel requires fft_padding >= 1.05.")

    # --- Image-edge correction ---
    # The builder rejects correction < 1e-6 before division would distort flux.
    # Keep that original guard and reuse its cached numerical construction.
    make_gridding_kernel(recipe, fft_padding=fft_padding)
