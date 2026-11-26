"""Pure checks for Feather configuration and opened input-image metadata.

File access, output existence, and chunk iteration belong to the workflow.
These checks neither open files nor insert configuration defaults.
"""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    check_fft_backend,
    require_boolean,
    require_finite_real,
    require_mapping,
)

# ========================================================================
# Supplied configuration, checked before opening either image
# ========================================================================


def check_feather_config(outim, highres, lowres):
    """Validate image paths and the output policy without modifying mappings.

    Parameters
    ----------
    outim : mapping
        Required nonempty output name and optional Boolean overwrite policy.
    highres, lowres : str
        Nonempty paths to the high- and low-resolution input images.
    """
    # --- Input locations ---
    # Each graph node reopens these paths; in-memory images are not supported.
    for name, path in (("highres", highres), ("lowres", lowres)):
        if not isinstance(path, str):
            raise TypeError(
                f"feather {name} must be a string path to an on-disk image."
            )
        if not path.strip():
            raise ValueError(f"feather {name} must be a nonempty image path.")

    # --- Output location and overwrite policy ---
    # A missing overwrite key means False to the caller; never insert that key.
    require_mapping("outim", outim)
    if "name" not in outim:
        raise ValueError("outim dict must have key 'name'.")
    if not isinstance(outim["name"], str):
        raise TypeError("outim['name'] must be a string path.")
    if not outim["name"].strip():
        raise ValueError("outim['name'] must be a nonempty image path.")
    if "overwrite" in outim:
        require_boolean("outim['overwrite']", outim["overwrite"])


def check_feather_combination(sdfactor, axes, fft_backend):
    """Validate the settings of the feather combination without modifying them.

    Parameters
    ----------
    sdfactor : float
        Finite real scalar scaling of the single-dish image.
    axes : tuple or list of str
        Two distinct names of the spatial axes.
    fft_backend : {"numpy", "scipy", "pyfftw"}
        An installed FFT library.
    """
    require_finite_real("sdfactor", sdfactor, scalar=True)
    if not isinstance(axes, (tuple, list)) or not all(
        isinstance(axis, str) for axis in axes
    ):
        raise TypeError("feather axes must be a tuple or list of axis names.")
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError("feather axes must name two distinct axes.")
    check_fft_backend(fft_backend)


def check_feather_output(output_image_format, io_engine):
    """Require a supported output format for the I/O engine.

    Parameters
    ----------
    output_image_format : {"zarr", "fits"}
        Format of the output image.
    io_engine : str
        I/O engine of the node tasks; FITS output needs ``"chunk"``.
    """
    if output_image_format not in ("zarr", "fits"):
        raise ValueError("output_image_format must be 'zarr' or 'fits'.")
    if output_image_format == "fits" and io_engine != "chunk":
        raise ValueError("FITS output needs io_engine='chunk'.")


# ========================================================================
# Opened image metadata, always checked even for prechecked configuration
# ========================================================================


def check_feather_inputs(highres, lowres):
    """Require compatible SKY shapes and supported real floating-point storage.

    Parameters
    ----------
    highres, lowres : xarray.Dataset
        Opened images whose SKY variables must have identical shapes and float32
        or float64 storage.

    Raises
    ------
    ValueError
        A SKY product is missing or has unsupported storage precision.
    RuntimeError
        The input SKY shapes differ.

    Notes
    -----
    Only metadata is inspected: no image pixels are loaded or converted. Mixed
    float32/float64 inputs remain supported with the existing precision policy.
    """
    # --- Required products and storage types ---
    # Validate both inputs, rather than allowing the output dtype choice to hide
    # an unsupported dtype in the other image.
    for name, image in (("highres", highres), ("lowres", lowres)):
        if "SKY" not in image:
            raise ValueError(f"feather {name} must contain a SKY data variable.")
        if image["SKY"].dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(
                f"Unsupported image dtype {image['SKY'].dtype} in {name}; "
                "expected float32 or float64."
            )

    # --- Pixel correspondence ---
    # The combination works elementwise and cannot broadcast unlike shapes.
    if highres["SKY"].shape != lowres["SKY"].shape:
        raise RuntimeError("Image shapes differ")
