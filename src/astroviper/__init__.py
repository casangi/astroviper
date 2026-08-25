"""AstroVIPER: radio-astronomy interferometric imaging."""

import warnings

# Silence noisy, non-actionable third-party warnings package-wide. Zarr's
# experimental V3 backend emits these on routine reads/writes of the data XRADIO
# stores, and neither can be acted on from AstroVIPER. They are matched by
# message (not category) so that ``import astroviper`` stays free of an eager
# ``import zarr`` -- important for graph-worker startup time -- and so only these
# specific messages are hidden rather than a whole warning base category. The
# test suite silences the same warnings via ``filterwarnings`` in pyproject.toml
# (pytest's per-test warning capture resets these import-time filters).

# Fixed-length UTF string dtypes (used throughout XRADIO's Zarr stores) have no
# stabilized Zarr V3 spec -> zarr.errors.UnstableSpecificationWarning.
warnings.filterwarnings(
    "ignore",
    message=r"The data type .* does not have a Zarr V3 specification",
)
# Consolidated metadata is not yet part of the Zarr V3 spec, emitted on every
# consolidated read/write -> zarr.errors.ZarrUserWarning.
warnings.filterwarnings(
    "ignore",
    message=r"Consolidated metadata is currently not part in the Zarr format 3 specification",
)
