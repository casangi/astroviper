"""Single source of truth for shared image-analysis parameter descriptions.

Every parameter below is spelled out in more than one layer of the
image-analysis stack (the distributed application, the node task and the
science processing functions).  Edit the canonical description **here** and
run ``python -m astroviper.utils.param_docs sync`` to propagate it into every
function decorated with
:func:`~astroviper.utils.param_docs.shares_param_docs` (CI verifies it with
``... param_docs check``).

Each value is the NumPy-style *description* body only (no ``name : type``
line): the per-function ``name : type`` line -- which encodes the type and
whether the parameter is optional -- is preserved by the codegen, so the same
description can be shared by functions that give the parameter different
defaults.
"""

IMAGE_ANALYSIS_PARAM_DOCS = {
    "input_image_store": (
        "Path to the on-disk input image (a ``.zarr`` store or any format\n"
        "``xradio.image.open_image`` understands)."
    ),
    "moments_image_store": (
        "Path of the output Zarr image store the moment maps are written to.\n"
        "Created up front by the distributed application and written\n"
        "chunk-by-chunk in parallel by the node tasks."
    ),
    "moments": (
        "The moments to compute, as canonical names and/or CASA ``immoments``\n"
        "integer codes:\n"
        "\n"
        '- ``"mean"`` (CASA ``-1``) : mean value of the profile.\n'
        '- ``"integrated"`` (``0``) : integrated value ``sum(I * delta_v)``\n'
        "  with ``delta_v`` the per-plane moment-axis coordinate width.\n"
        '- ``"weighted_coord"`` (``1``) : intensity-weighted coordinate\n'
        "  ``sum(I * v) / sum(I)`` (e.g. velocity field).  Use\n"
        "  ``include_pixel_range`` to restrict to positive flux for sensible\n"
        "  results.\n"
        '- ``"weighted_dispersion_coord"`` (``2``) : intensity-weighted\n'
        "  coordinate dispersion ``sqrt(sum(I * v^2)/sum(I) - m1^2)``.\n"
        '- ``"median"`` (``3``) : median value of the profile.\n'
        '- ``"median_coord"`` (``4``) : coordinate at which the cumulative\n'
        "  profile crosses 50% of its total (only meaningful for\n"
        "  predominantly positive profiles).\n"
        '- ``"standard_deviation"`` (``5``) : standard deviation about the\n'
        "  profile mean.\n"
        '- ``"rms"`` (``6``) : root mean square of the profile.\n'
        '- ``"abs_mean_dev"`` (``7``) : mean absolute deviation from the\n'
        "  profile mean.\n"
        '- ``"maximum"`` (``8``) / ``"maximum_coord"`` (``9``) : maximum of\n'
        "  the profile and the coordinate at which it occurs.\n"
        '- ``"minimum"`` (``10``) / ``"minimum_coord"`` (``11``) : minimum of\n'
        "  the profile and the coordinate at which it occurs."
    ),
    "moment_axis": (
        'Image dimension to collapse: ``"l"``, ``"m"``, ``"frequency"``,\n'
        '``"polarization"`` or ``"time"`` (AstroVIPER names for CASA\'s *ra*,\n'
        "*dec*, *spectral*, *stokes*).  The moment axis is never used for\n"
        "parallelism.  Coordinate-valued moments are expressed in the native\n"
        "coordinate units of this axis (Hz for ``frequency``, rad for\n"
        "``l``/``m``); a non-numeric axis (``polarization``) uses the plane\n"
        "index."
    ),
    "image_data_group_in_name": (
        'Key in the image\'s ``data_groups`` whose ``"sky"`` (and, optionally,\n'
        '``"mask"``) roles name the input data variables.  Datasets without\n'
        'data groups fall back to the conventional ``"SKY"`` variable.'
    ),
    "include_pixel_range": (
        "Only pixel values inside ``[low, high]`` contribute.  A single value\n"
        "``b`` means ``[-abs(b), abs(b)]`` (CASA convention).  Mutually\n"
        "exclusive with ``exclude_pixel_range``."
    ),
    "exclude_pixel_range": (
        "Pixel values inside ``[low, high]`` do not contribute.  Same\n"
        "conventions as ``include_pixel_range``."
    ),
    "use_mask": (
        "If ``True``, pixels where the input data group's mask variable is\n"
        "``False`` are excluded (XRADIO convention: mask ``True`` = include)."
    ),
    "selection": (
        "``xarray`` ``isel`` selection applied to the input image (e.g. a\n"
        "``frequency`` channel range or an ``l``/``m`` sub-window), the\n"
        "AstroVIPER analogue of CASA's ``chans``/``stokes``/``box``.  Must not\n"
        "select along the parallel axis (chunking happens on the selected\n"
        "image)."
    ),
}
