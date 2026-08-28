"""Frequency-resolved visibility gridding for MVC continuum imaging."""

from __future__ import annotations

import copy

import numpy as np
import xarray as xr


def add_visibility_grid_mvc_single_field(
    ms_xds: xr.Dataset,
    cgk_1D: np.ndarray,
    img_xds: xr.Dataset,
    ms_data_group_in_name: str = "base",
    image_data_group_in_name: str = "residual",
    image_data_group_out_name: str = "residual",
    image_data_group_out_modified: dict | None = None,
    overwrite: bool = True,
    fft_padding: float = 1.2,
    processing_function_threads: int = 1,
    complex_dtype=None,
) -> None:
    """Grid visibilities into a frequency-resolved MVC UV cube.

    MVC retains the frequency axis through gridding and image formation.
    Therefore, unlike direct MT-MFS gridding, this function applies no Taylor
    weights and does not collapse input channels onto Taylor UV planes. Input
    channel ``i`` is mapped directly to output frequency plane ``i``.

    The generated arrays have dimensions

    ``VISIBILITY(time, frequency, polarization, u, v)``

    and

    ``VISIBILITY_NORMALIZATION(time, frequency, polarization)``.

    Repeated calls accumulate contributions from multiple measurement sets into
    the existing output arrays. All measurement sets accumulated into the same
    image dataset must have frequency coordinates matching the image frequency
    coordinate. A measurement-set child may cover a unique subset of the image
    frequency axis; repeated calls accumulate different child subsets into the
    shared image cube.

    Parameters
    ----------
    ms_xds : xarray.Dataset
        Measurement-set dataset containing correlated visibility data, UVW
        coordinates, imaging weights, and a one-dimensional frequency
        coordinate.

    cgk_1D : numpy.ndarray
        One-dimensional prolate-spheroidal gridding convolution kernel.

    img_xds : xarray.Dataset
        Image dataset receiving the frequency-resolved MVC UV grids. Its
        ``frequency`` coordinate must contain every frequency in ``ms_xds``.

    ms_data_group_in_name : str, optional
        Measurement-set data group containing the logical roles
        ``"correlated_data"``, ``"uvw"``, and ``"weight_imaging"``.

    image_data_group_in_name : str, optional
        Existing image data group used as the input group.

    image_data_group_out_name : str, optional
        Image data group under which the MVC UV-grid products are registered.

    image_data_group_out_modified : dict, optional
        Output role-to-variable mapping. By default, the function writes
        ``VISIBILITY`` and ``VISIBILITY_NORMALIZATION``.

    overwrite : bool, optional
        Passed to the image data-group helper.

    fft_padding : float, optional
        FFT padding factor used to determine the UV-grid size.

    processing_function_threads : int, optional
        Number of threads supplied to the C++ gridder.

    complex_dtype : numpy dtype, optional
        Complex dtype used for the visibility grid. Defaults to
        ``numpy.complex128``.

    Returns
    -------
    None
        The function modifies ``img_xds`` in place.

    Notes
    -----
    This function only performs the frequency-resolved gridding stage of MVC.
    The global workflow must subsequently

    1. reduce matching frequency planes across map tasks;
    2. inverse Fourier-transform and normalize the frequency cube;
    3. divide each channel image by its corresponding primary beam;
    4. fit the corrected image cube into Taylor terms.

    Each measurement-set frequency must map one-to-one onto a channel in the
    image frequency axis.
    """
    from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp import (
        prolate_spheroidal_grid,
    )
    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )
    from astroviper.processing_functions.imaging.utils.frequency_mapping import (
        map_visibility_frequencies_to_image,
    )
    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    if image_data_group_out_modified is None:
        image_data_group_out_modified = {
            "visibility": "VISIBILITY",
            "visibility_normalization": ("VISIBILITY_NORMALIZATION"),
        }

    if complex_dtype is None:
        complex_dtype = np.complex128

    if not np.issubdtype(
        np.dtype(complex_dtype),
        np.complexfloating,
    ):
        raise TypeError(
            "complex_dtype must be a complex floating-point dtype; "
            f"received {complex_dtype!r}."
        )

    if not np.isfinite(fft_padding) or fft_padding < 1.0:
        raise ValueError(
            f"fft_padding must be finite and at least 1.0; received {fft_padding}."
        )

    if int(processing_function_threads) < 1:
        raise ValueError(
            "processing_function_threads must be at least 1; "
            f"received {processing_function_threads}."
        )

    processing_function_threads = int(processing_function_threads)

    # ------------------------------------------------------------------
    # Resolve input and output data groups.
    # ------------------------------------------------------------------
    data_groups = ms_xds.attrs.get("data_groups", {})

    if ms_data_group_in_name not in data_groups:
        raise KeyError(
            f"Measurement-set data group {ms_data_group_in_name!r} is missing."
        )

    ms_data_group_in = data_groups[ms_data_group_in_name]

    required_roles = (
        "correlated_data",
        "uvw",
        "weight_imaging",
    )

    missing_roles = [role for role in required_roles if role not in ms_data_group_in]

    if missing_roles:
        raise KeyError(
            f"Measurement-set data group "
            f"{ms_data_group_in_name!r} is missing roles "
            f"{missing_roles}."
        )

    for role in required_roles:
        variable_name = ms_data_group_in[role]

        if variable_name not in ms_xds:
            raise KeyError(
                f"Measurement-set data group role {role!r} "
                f"registers variable {variable_name!r}, but that "
                "variable is absent."
            )

    output_mapping = copy.deepcopy(image_data_group_out_modified)

    _, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=output_mapping,
        overwrite=overwrite,
    )

    visibility_name = image_data_group_out["visibility"]
    normalization_name = image_data_group_out["visibility_normalization"]

    # ------------------------------------------------------------------
    # Validate visibility-domain arrays.
    # ------------------------------------------------------------------
    visibility_data = np.asarray(ms_xds[ms_data_group_in["correlated_data"]].values)
    uvw = np.asarray(ms_xds[ms_data_group_in["uvw"]].values)
    weight_imaging = np.asarray(ms_xds[ms_data_group_in["weight_imaging"]].values)

    if weight_imaging.ndim != 4:
        raise ValueError(
            "The imaging-weight array must have dimensions "
            "(time, baseline, frequency, polarization); "
            f"received shape {weight_imaging.shape}."
        )

    if visibility_data.shape != weight_imaging.shape:
        raise ValueError(
            "The correlated-data and imaging-weight arrays must "
            "have identical shapes; received "
            f"{visibility_data.shape} and "
            f"{weight_imaging.shape}."
        )

    n_time = weight_imaging.shape[0]
    n_chan = weight_imaging.shape[2]
    n_pol = weight_imaging.shape[3]

    if "frequency" not in ms_xds.coords:
        raise KeyError(
            "The measurement-set dataset does not contain a frequency coordinate."
        )

    frequency_coord = np.asarray(
        ms_xds.coords["frequency"].values,
        dtype=np.float64,
    )

    if frequency_coord.ndim != 1:
        raise ValueError(
            "The measurement-set frequency coordinate must be "
            "one-dimensional; received shape "
            f"{frequency_coord.shape}."
        )

    if frequency_coord.size != n_chan:
        raise ValueError(
            "The measurement-set frequency coordinate length "
            "does not match the visibility channel axis: "
            f"{frequency_coord.size} != {n_chan}."
        )

    if not np.all(np.isfinite(frequency_coord)):
        raise ValueError(
            "The measurement-set frequency coordinate contains non-finite values."
        )

    if "frequency" not in img_xds.coords:
        raise KeyError("The MVC image dataset must contain a frequency coordinate.")

    image_frequency_coord = np.asarray(
        img_xds.coords["frequency"].values,
        dtype=np.float64,
    )

    if image_frequency_coord.ndim != 1:
        raise ValueError(
            "The image frequency coordinate must be "
            "one-dimensional; received shape "
            f"{image_frequency_coord.shape}."
        )

    frequency_map = map_visibility_frequencies_to_image(
        frequency_coord,
        image_frequency_coord,
    )
    n_image_chan = image_frequency_coord.size

    if "polarization" not in img_xds.coords:
        raise KeyError("The MVC image dataset must contain a polarization coordinate.")

    if img_xds.sizes["polarization"] != n_pol:
        raise ValueError(
            "The image and visibility polarization axes have "
            "different lengths: "
            f"{img_xds.sizes['polarization']} != {n_pol}."
        )

    if "time" not in img_xds.coords:
        raise KeyError("The MVC image dataset must contain a time coordinate.")

    # ------------------------------------------------------------------
    # Construct grid maps.
    # ------------------------------------------------------------------
    n_image_time = 1

    if img_xds.sizes["time"] != n_image_time:
        raise ValueError(
            "The current single-field MVC gridder expects exactly "
            "one image time plane; received "
            f"{img_xds.sizes['time']}."
        )

    # All input integrations contribute to one image time plane.
    time_map = np.zeros(
        n_time,
        dtype=np.int64,
    )

    pol_map = np.arange(
        n_pol,
        dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # Determine UV-grid geometry.
    # ------------------------------------------------------------------
    n_uv = padded_grid_size(
        [
            img_xds.sizes["l"],
            img_xds.sizes["m"],
        ],
        fft_padding,
    )

    delta_lm = img_xds.xr_img.get_lm_cell_size()

    expected_grid_shape = (
        n_image_time,
        n_image_chan,
        n_pol,
        n_uv[0],
        n_uv[1],
    )

    expected_normalization_shape = (
        n_image_time,
        n_image_chan,
        n_pol,
    )

    # ------------------------------------------------------------------
    # Create or validate output arrays.
    # ------------------------------------------------------------------
    if visibility_name not in img_xds:
        visibility_coords = {
            "time": img_xds.coords["time"],
            "frequency": img_xds.coords["frequency"],
            "polarization": img_xds.coords["polarization"],
        }

        if "u" in img_xds.coords:
            visibility_coords["u"] = img_xds.coords["u"]

        if "v" in img_xds.coords:
            visibility_coords["v"] = img_xds.coords["v"]

        img_xds[visibility_name] = xr.DataArray(
            np.zeros(
                expected_grid_shape,
                dtype=complex_dtype,
            ),
            dims=(
                "time",
                "frequency",
                "polarization",
                "u",
                "v",
            ),
            coords={
                "time": img_xds.coords["time"],
                "frequency": (
                    "frequency",
                    image_frequency_coord,
                ),
                "polarization": img_xds.coords["polarization"],
            },
        )

        img_xds[normalization_name] = xr.DataArray(
            np.zeros(
                expected_normalization_shape,
                dtype=np.float64,
            ),
            dims=(
                "time",
                "frequency",
                "polarization",
            ),
            coords={
                "time": img_xds.coords["time"],
                "frequency": (
                    "frequency",
                    image_frequency_coord,
                ),
                "polarization": img_xds.coords["polarization"],
            },
        )

        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            image_data_group_out,
            description=(
                "Added frequency-resolved MVC visibility grids "
                "with add_visibility_grid_mvc_single_field."
            ),
        )

    else:
        if normalization_name not in img_xds:
            raise KeyError(
                f"{visibility_name!r} exists, but corresponding "
                f"normalization variable "
                f"{normalization_name!r} is missing."
            )

        if img_xds[visibility_name].shape != expected_grid_shape:
            raise ValueError(
                f"Existing {visibility_name!r} has shape "
                f"{img_xds[visibility_name].shape}; expected "
                f"{expected_grid_shape}."
            )

        if img_xds[normalization_name].shape != expected_normalization_shape:
            raise ValueError(
                f"Existing {normalization_name!r} has shape "
                f"{img_xds[normalization_name].shape}; expected "
                f"{expected_normalization_shape}."
            )

        if img_xds[visibility_name].dims != (
            "time",
            "frequency",
            "polarization",
            "u",
            "v",
        ):
            raise ValueError(
                f"Existing {visibility_name!r} has dimensions "
                f"{img_xds[visibility_name].dims}; expected "
                "(time, frequency, polarization, u, v)."
            )

        if img_xds[normalization_name].dims != (
            "time",
            "frequency",
            "polarization",
        ):
            raise ValueError(
                f"Existing {normalization_name!r} has dimensions "
                f"{img_xds[normalization_name].dims}; expected "
                "(time, frequency, polarization)."
            )

    # ------------------------------------------------------------------
    # Grid directly into the output frequency cube.
    # ------------------------------------------------------------------
    grid = img_xds[visibility_name].values
    normalization = img_xds[normalization_name].values

    prolate_spheroidal_grid(
        grid,
        normalization,
        visibility_data,
        uvw,
        frequency_coord,
        frequency_map,
        time_map,
        pol_map,
        weight_imaging,
        cgk_1D,
        n_uv,
        delta_lm,
        support=7,
        oversampling=100,
        processing_function_threads=(processing_function_threads),
    )

    # ------------------------------------------------------------------
    # Record output semantics.
    # ------------------------------------------------------------------
    img_xds[visibility_name].attrs.update(
        {
            "description": ("Frequency-resolved MVC visibility UV grids."),
            "specmode": "mvc",
            "channel_mapping": "one_to_one",
            "n_frequency_planes": n_image_chan,
        }
    )

    img_xds[normalization_name].attrs.update(
        {
            "description": ("Per-frequency MVC visibility-grid normalization sums."),
            "specmode": "mvc",
            "channel_mapping": "one_to_one",
            "n_frequency_planes": n_image_chan,
        }
    )
