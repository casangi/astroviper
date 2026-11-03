"""Pure checks for Feather configuration and opened input-image metadata.

File access, output existence, and chunk iteration belong to the workflow.
These checks neither open files nor insert configuration defaults.
"""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    require_boolean,
    require_mapping,
)

# ========================================================================
# Supplied configuration, checked before opening either image
# ========================================================================


def check_feather_config(outim, highres, lowres):
    """Validate image paths and the output policy without modifying mappings."""
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


# ========================================================================
# Opened image metadata, always checked even for prechecked configuration
# ========================================================================


def check_feather_inputs(highres, lowres):
    """Require compatible SKY shapes and supported real floating-point storage.

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
