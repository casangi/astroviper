"""User-supplied CLEAN masks of the imaging applications."""


def load_clean_mask(clean_mask, image_size):
    """Load and validate an optional two-dimensional CLEAN mask.

    Parameters
    ----------
    clean_mask : str or os.PathLike or None
        Path of a NumPy ``.npy`` file holding one image plane.
    image_size : sequence of int
        Expected ``(l, m)`` shape of the plane.

    Returns
    -------
    numpy.ndarray or None
        C-contiguous bool plane, ``True`` where the file holds a value greater
        than 0.5; ``None`` when ``clean_mask`` is ``None``.

    Raises
    ------
    ValueError
        If the plane does not have shape ``image_size``, holds non-finite values
        or selects no pixel.
    """
    if clean_mask is None:
        return None

    import numpy as np

    clean_mask_array = np.squeeze(np.asarray(np.load(clean_mask)))
    expected_shape = tuple(int(value) for value in image_size)
    if clean_mask_array.ndim != 2 or clean_mask_array.shape != expected_shape:
        raise ValueError(
            "clean_mask must contain one 2-D image plane with shape "
            f"{expected_shape}; received {clean_mask_array.shape}."
        )
    if not np.all(np.isfinite(clean_mask_array)):
        raise ValueError("clean_mask contains non-finite values.")

    clean_mask_array = np.ascontiguousarray(clean_mask_array > 0.5)
    if not np.any(clean_mask_array):
        raise ValueError("clean_mask does not select any pixels.")
    return clean_mask_array


def install_clean_mask(img_xds, clean_mask, image_data_group_name="residual"):
    """Install a 2-D CLEAN mask on every residual plane of an image.

    The mask is written to ``MASK`` with the dimensions of ``SKY_RESIDUAL`` and
    registered as the ``mask`` of ``image_data_group_name``, where it takes the
    place of the primary-beam mask.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image with ``SKY_RESIDUAL`` ending in ``(l, m)``. Modified in place.
    clean_mask : numpy.ndarray or None
        Bool ``(l, m)`` plane (see :func:`load_clean_mask`); ``None`` is a no-op.
    image_data_group_name : str, optional
        Data group that gets the mask. Default ``"residual"``.

    Raises
    ------
    KeyError
        If ``SKY_RESIDUAL`` or the data group is missing.
    ValueError
        If ``SKY_RESIDUAL`` does not end in ``(l, m)`` or the mask has another
        shape.
    """
    if clean_mask is None:
        return

    import numpy as np
    import xarray as xr

    from astroviper.utils.data_group_tools import modify_data_groups_xds

    if "SKY_RESIDUAL" not in img_xds:
        raise KeyError("The image does not contain 'SKY_RESIDUAL'.")

    residual = img_xds["SKY_RESIDUAL"]
    if residual.dims[-2:] != ("l", "m"):
        raise ValueError("SKY_RESIDUAL must end in the ('l', 'm') dimensions.")

    mask_plane = np.asarray(clean_mask, dtype=bool)
    expected_shape = (residual.sizes["l"], residual.sizes["m"])
    if mask_plane.shape != expected_shape:
        raise ValueError(
            f"The clean mask has shape {mask_plane.shape}; expected {expected_shape}."
        )

    expanded = mask_plane.reshape((1,) * (residual.ndim - 2) + mask_plane.shape)
    mask_values = np.broadcast_to(expanded, residual.shape).copy()
    img_xds["MASK"] = xr.DataArray(
        mask_values,
        dims=residual.dims,
        coords={dimension: residual.coords[dimension] for dimension in residual.dims},
        attrs={"description": "User-supplied deconvolution mask.", "type": "mask"},
    )

    data_groups = img_xds.attrs.get("data_groups", {})
    if image_data_group_name not in data_groups:
        raise KeyError(
            f"The image does not contain image data group {image_data_group_name!r}."
        )
    residual_group = dict(data_groups[image_data_group_name])
    residual_group["mask"] = "MASK"
    modify_data_groups_xds(
        img_xds,
        data_group_out_name=image_data_group_name,
        data_group_out=residual_group,
        description="User-supplied deconvolution mask installed.",
    )
