"""Directly callable image-statistics node task.

The node owns the complete local control path: inspect metadata, parse CASA-like
selectors, merge them with an optional application partition, load only the
effective selection, apply masks, and call pure processing functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

_IMAGE_DIMS = ("time", "frequency", "polarization", "l", "m")
_DEFAULT_STATISTICS = ("max", "min", "sum", "mean", "npts")


@dataclass(frozen=True)
class ImageSelection:
    """Positional user, execution, and effective image selections.

    ``user_indexers`` describes the logical request. ``partition_indexers``
    describes the part assigned to this node, in source-image positions.  Only
    their intersection is exposed to storage through ``effective_indexers``.
    Keeping the two inputs separate prevents an execution partition from
    replacing or expanding the user's request.
    """

    user_indexers: dict[str, np.ndarray]
    partition_indexers: dict[str, np.ndarray]
    effective_indexers: dict[str, np.ndarray]
    boxes: tuple[tuple[int, int, int, int], ...]
    reduction_dims: tuple[str, ...]

    @property
    def indexers(self) -> dict[str, np.ndarray]:
        """Storage-ready indexers (compatibility alias)."""
        return self.effective_indexers


# Compatibility for callers which used the implementation name during the
# initial image-statistics work.  New code should use ImageSelection.
ImageSlicer = ImageSelection


def _positions(selector, size: int) -> np.ndarray:
    """Normalize one positional selector against an axis of ``size``."""
    positions = np.arange(size, dtype=np.int64)
    try:
        selected = positions[selector]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"Invalid positional selector {selector!r}") from exc
    selected = np.atleast_1d(selected)
    if selected.dtype == bool:
        if selected.size != size:
            raise ValueError("Boolean positional selector has the wrong length")
        selected = positions[selected]
    selected = np.asarray(selected, dtype=np.int64)
    selected = np.where(selected < 0, selected + size, selected)
    if np.any((selected < 0) | (selected >= size)):
        raise IndexError(f"Selection is outside axis length {size}")
    return selected


def _intersect_in_order(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Intersect positional arrays while preserving ``left`` ordering."""
    return left[np.isin(left, right)]


def _index_expression(expression: str, size: int, name: str) -> np.ndarray:
    if expression is None or str(expression).strip() == "":
        return np.arange(size, dtype=np.int64)
    selected: list[np.ndarray] = []
    for raw_token in re.split(r"[;,]", str(expression)):
        token = raw_token.strip().replace(" ", "")
        if not token:
            continue
        match = re.fullmatch(r"(-?\d+)~(-?\d+)(?:\^(\d+))?", token)
        if match:
            start, stop = int(match.group(1)), int(match.group(2))
            step = int(match.group(3) or 1)
            if step < 1:
                raise ValueError(f"{name} range step must be positive")
            direction = 1 if stop >= start else -1
            selected.append(np.arange(start, stop + direction, direction * step))
            continue
        match = re.fullmatch(r"(<=|>=|<|>)(-?\d+)", token)
        if match:
            op, boundary_text = match.groups()
            boundary = int(boundary_text)
            values = np.arange(size, dtype=np.int64)
            selected.append(
                values[
                    {
                        "<": values < boundary,
                        "<=": values <= boundary,
                        ">": values > boundary,
                        ">=": values >= boundary,
                    }[op]
                ]
            )
            continue
        if re.fullmatch(r"-?\d+", token):
            selected.append(np.asarray([int(token)], dtype=np.int64))
            continue
        raise ValueError(f"Unsupported {name} selector token {raw_token!r}")
    if not selected:
        raise ValueError(f"{name} selection is empty")
    indices = np.unique(np.concatenate(selected))
    indices = np.where(indices < 0, indices + size, indices)
    if np.any((indices < 0) | (indices >= size)):
        raise IndexError(f"{name} selection is outside axis length {size}")
    return indices.astype(np.int64)


def _parse_boxes(box: str | None, l_size: int, m_size: int):
    if box is None or str(box).strip() == "":
        return ()
    values = [int(value.strip()) for value in str(box).split(",") if value.strip()]
    if len(values) % 4:
        raise ValueError("box must contain groups of x0,y0,x1,y1")
    boxes = []
    for offset in range(0, len(values), 4):
        l0, m0, l1, m1 = values[offset : offset + 4]
        if l0 > l1 or m0 > m1:
            raise ValueError("box lower-left coordinate must precede upper-right")
        if l0 < 0 or m0 < 0 or l1 >= l_size or m1 >= m_size:
            raise IndexError("box lies outside the image direction plane")
        boxes.append((l0, m0, l1, m1))
    return tuple(boxes)


def _region_boxes(region: Any, l_size: int, m_size: int):
    if region is None or region == "":
        return ()
    if isinstance(region, dict):
        if not {"blc", "trc"} <= set(region):
            raise ValueError("A region record must contain 'blc' and 'trc'")
        blc, trc = region["blc"], region["trc"]
        return _parse_boxes(
            f"{int(blc[0])},{int(blc[1])},{int(trc[0])},{int(trc[1])}",
            l_size,
            m_size,
        )
    text = Path(region).read_text() if isinstance(region, Path) else str(region)
    possible_path = Path(text.strip("`"))
    try:
        is_file = "\n" not in text and possible_path.is_file()
    except OSError:
        is_file = False
    if is_file:
        text = possible_path.read_text()
    matches = re.findall(
        r"box\s*\[\s*\[\s*(-?\d+)\s*pix\s*,\s*(-?\d+)\s*pix\s*\]\s*,"
        r"\s*\[\s*(-?\d+)\s*pix\s*,\s*(-?\d+)\s*pix\s*\]\s*\]",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return _parse_boxes(
            ",".join(value for match in matches for value in match), l_size, m_size
        )
    if re.fullmatch(r"[\d\s,\-]+", text):
        return _parse_boxes(text, l_size, m_size)
    raise ValueError("Only pixel-coordinate box regions are currently supported")


def _polarization_indices(expression: str, coordinate: xr.DataArray) -> np.ndarray:
    labels = [str(value) for value in coordinate.values]
    if expression is None or str(expression).strip() == "":
        return np.arange(len(labels), dtype=np.int64)
    text = str(expression).replace(" ", "")
    if "," in text:
        requested = [value for value in text.split(",") if value]
    elif text in labels:
        requested = [text]
    elif all(len(label) == 1 for label in labels):
        requested = list(text)
    else:
        requested = []
        remaining = text
        candidates = sorted(labels, key=len, reverse=True)
        while remaining:
            match = next(
                (label for label in candidates if remaining.startswith(label)), None
            )
            if match is None:
                raise ValueError(f"Cannot parse stokes selection {expression!r}")
            requested.append(match)
            remaining = remaining[len(match) :]
    unknown = set(requested) - set(labels)
    if unknown:
        raise ValueError(f"Unknown polarization labels: {sorted(unknown)}")
    return np.asarray([labels.index(label) for label in requested], dtype=np.int64)


def _reduction_dims(axes, dims: tuple[str, ...]) -> tuple[str, ...]:
    if axes == -1 or axes is None:
        return dims
    values = [axes] if isinstance(axes, (str, int, np.integer)) else list(axes)
    result = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            value = dims[int(value)]
        if value not in dims:
            raise ValueError(f"Unknown statistics axis {value!r}")
        if value not in result:
            result.append(value)
    return tuple(result)


def _choose_data_array(
    image: xr.DataArray | xr.Dataset, data_variable: str | None
) -> tuple[xr.Dataset, str]:
    if isinstance(image, xr.DataArray):
        name = data_variable or image.name or "IMAGE"
        return image.rename(name).to_dataset(), name
    if not isinstance(image, xr.Dataset):
        raise TypeError("image must be an xarray DataArray, Dataset, or on-disk path")
    if data_variable is not None:
        if data_variable not in image:
            raise KeyError(f"Image has no data variable {data_variable!r}")
        return image, data_variable
    candidates = [
        name
        for name, value in image.data_vars.items()
        if set(_IMAGE_DIMS) <= set(value.dims)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "data_variable is required when the image does not have exactly one five-axis variable"
        )
    return image, candidates[0]


def _open_metadata(image, data_variable):
    if isinstance(image, (str, Path)):
        dataset = xr.open_zarr(str(image))
    else:
        dataset = image
    return _choose_data_array(dataset, data_variable)


def _required_variables(metadata: xr.Dataset, variable: str, mask) -> list[str]:
    """Return the only data variables that a local statistics task may load."""
    required = [variable]
    if isinstance(mask, str) and mask:
        if mask not in metadata:
            raise KeyError(f"Image has no mask variable {mask!r}")
        required.append(mask)
    return required


def build_image_selection(
    data: xr.DataArray,
    *,
    axes=-1,
    region="",
    box="",
    chans="",
    stokes="",
    timerange="",
    partition: dict[str, slice | np.ndarray] | None = None,
) -> ImageSelection:
    """Normalize user and relative partition selectors, then intersect them."""
    dims = tuple(data.dims)
    user_indexers = {dim: np.arange(data.sizes[dim], dtype=np.int64) for dim in dims}
    if "frequency" in dims:
        user_indexers["frequency"] = _index_expression(
            chans, data.sizes["frequency"], "chans"
        )
    if "time" in dims:
        user_indexers["time"] = _index_expression(
            timerange, data.sizes["time"], "timerange"
        )
    if "polarization" in dims:
        user_indexers["polarization"] = _polarization_indices(
            stokes, data["polarization"]
        )
    if box and region:
        raise ValueError("Specify either region or box, not both")
    boxes = _parse_boxes(box, data.sizes["l"], data.sizes["m"])
    if region:
        boxes = _region_boxes(region, data.sizes["l"], data.sizes["m"])
    if boxes:
        user_indexers["l"] = np.arange(
            min(item[0] for item in boxes), max(item[2] for item in boxes) + 1
        )
        user_indexers["m"] = np.arange(
            min(item[1] for item in boxes), max(item[3] for item in boxes) + 1
        )

    partition_indexers = {dim: values.copy() for dim, values in user_indexers.items()}
    for dim, relative_selector in (partition or {}).items():
        if dim not in dims:
            raise ValueError(f"Partition dimension {dim!r} is not present")
        relative_positions = _positions(relative_selector, len(user_indexers[dim]))
        partition_indexers[dim] = user_indexers[dim][relative_positions]

    effective_indexers = {
        dim: _intersect_in_order(user_indexers[dim], partition_indexers[dim])
        for dim in dims
    }
    if any(values.size == 0 for values in effective_indexers.values()):
        raise ValueError("The image selection is empty")
    return ImageSelection(
        user_indexers=user_indexers,
        partition_indexers=partition_indexers,
        effective_indexers=effective_indexers,
        boxes=boxes,
        reduction_dims=_reduction_dims(axes, dims),
    )


build_image_slicer = build_image_selection


def _box_mask(selected: xr.DataArray, selection: ImageSelection):
    if len(selection.boxes) <= 1:
        return None
    l_indices = selection.effective_indexers["l"]
    m_indices = selection.effective_indexers["m"]
    li, mi = np.meshgrid(l_indices, m_indices, indexing="ij")
    mask = np.zeros((selected.sizes["l"], selected.sizes["m"]), dtype=bool)
    for l0, m0, l1, m1 in selection.boxes:
        mask |= (li >= l0) & (li <= l1) & (mi >= m0) & (mi <= m1)
    return xr.DataArray(mask, dims=("l", "m"))


def _select_external_mask(
    mask: xr.DataArray, selection: ImageSelection
) -> xr.DataArray:
    applicable = {
        dim: value
        for dim, value in selection.effective_indexers.items()
        if dim in mask.dims
    }
    return mask.isel(applicable)


def _apply_masks(
    data: xr.DataArray,
    dataset: xr.Dataset,
    selection: ImageSelection,
    mask,
    stretch: bool,
    includepix,
    excludepix,
) -> xr.DataArray:
    spatial_mask = _box_mask(data, selection)
    if spatial_mask is not None:
        data = data.where(spatial_mask)
    if mask is not None and not (isinstance(mask, str) and mask == ""):
        if isinstance(mask, str):
            if mask not in dataset:
                raise KeyError(f"Image has no mask variable {mask!r}")
            mask_array = dataset[mask]
        elif isinstance(mask, xr.DataArray):
            mask_array = _select_external_mask(mask, selection)
        else:
            values = np.asarray(mask, dtype=bool)
            if values.ndim == data.ndim:
                mask_dims = data.dims
            elif (
                stretch
                and values.ndim == 2
                and values.shape
                == (
                    data.sizes["l"],
                    data.sizes["m"],
                )
            ):
                mask_dims = ("l", "m")
            else:
                raise ValueError(
                    "An array mask must match the selected image, or be a "
                    "two-dimensional l/m mask with stretch=True"
                )
            mask_array = xr.DataArray(values, dims=mask_dims)
        if not stretch and (
            mask_array.dims != data.dims or mask_array.sizes != data.sizes
        ):
            raise ValueError(
                "Mask does not match the selected image; use stretch=True to broadcast degenerate dimensions"
            )
        mask_array, _ = xr.broadcast(mask_array.astype(bool), data)
        data = data.where(mask_array)
    if includepix is not None:
        low, high = includepix
        data = data.where((data >= low) & (data <= high))
    if excludepix is not None:
        low, high = excludepix
        data = data.where((data < low) | (data > high))
    return data


def image_statistics(
    image: xr.DataArray | xr.Dataset | str | Path,
    *,
    data_variable: str | None = None,
    axes=-1,
    region="",
    box="",
    chans="",
    stokes="",
    timerange="",
    mask=None,
    stretch: bool = False,
    includepix=None,
    excludepix=None,
    statistics=_DEFAULT_STATISTICS,
    partition: dict[str, slice | np.ndarray] | None = None,
    data_selection: dict | None = None,
    finalize: bool = True,
) -> xr.Dataset:
    """Load one selected image block and calculate local image statistics.

    Parameters
    ----------
    image : xarray.DataArray, xarray.Dataset, str, or pathlib.Path
        An in-memory image or an on-disk XRADIO/Zarr image. On-disk pixels are
        not loaded until all selectors and the optional graph partition have
        been merged.
    data_variable : str, optional
        Image variable to analyze. Required when a Dataset contains more than
        one five-axis image variable.
    axes : int, sequence of int, str, or sequence of str, default -1
        Dimensions reduced by the statistics. ``-1`` reduces all dimensions.
        Named dimensions are preferred; integers follow ``DataArray.dims``.
    region : str, pathlib.Path, or dict, optional
        Pixel-box region. Supported forms are a CASA record containing ``blc``
        and ``trc``, CRTF ``box[[...pix],[...pix]]`` text, or a CRTF file
        containing pixel boxes. Other CRTF shapes are rejected.
    box : str, optional
        CASA inclusive pixel-box syntax ``x0,y0,x1,y1``. Additional groups of
        four integers form a union of disjoint boxes.
    chans, timerange : str, optional
        Zero-based index expressions supporting scalars, inclusive ``a~b``
        ranges, ``a~b^step``, comma/semicolon unions, and ``< <= > >=``.
    stokes : str, optional
        CASA polarization labels, either concatenated or comma-separated.
    mask : str, xarray.DataArray, or array-like, optional
        Named image mask or an aligned/broadcastable Boolean mask.
    stretch : bool, default False
        Allow lower-dimensional or degenerate masks to broadcast.
    includepix, excludepix : pair of float, optional
        Inclusive value range to retain or exclude, respectively.
    statistics : sequence of str
        Any of ``min``, ``max``, ``sum``, ``mean``, and ``npts``.
    partition : mapping, optional
        Structured application-generated selector relative to the user
        selection. It is intersected with, never substituted for, user input.
    finalize : bool, default True
        Return public statistics when true, otherwise a mergeable partial state.

    Returns
    -------
    xarray.Dataset
        Requested statistics over the dimensions not listed in ``axes``.

    Notes
    -----
    Only the requested image variable and an optional named mask are loaded.
    Parsing may iterate over short selector specifications; no Python loop
    iterates over image samples, planes, channels, or storage chunks.
    """
    from astroviper.processing_functions.image_analysis.statistics import (
        create_statistics_state,
        finalize_statistics_state,
    )

    metadata, variable = _open_metadata(image, data_variable)
    data = metadata[variable]
    graph_partition = None
    if data_selection:
        graph_partition = data_selection.get("image", data_selection)
    if partition and graph_partition:
        raise ValueError("partition and graph data_selection cannot both be specified")
    selection = build_image_selection(
        data,
        axes=axes,
        region=region,
        box=box,
        chans=chans,
        stokes=stokes,
        timerange=timerange,
        partition=partition or graph_partition,
    )

    required = _required_variables(metadata, variable, mask)

    if isinstance(image, (str, Path)):
        from xradio.image import load_image

        lazy_selection = load_image(str(image), block_des=selection.effective_indexers)
        missing = set(required) - set(lazy_selection.data_vars)
        if missing:
            raise KeyError(f"Loaded image is missing variables: {sorted(missing)}")
        selected_dataset = lazy_selection[required].load()
    else:
        selected_dataset = metadata[required].isel(selection.effective_indexers).load()
    selected = selected_dataset[variable]
    selected = _apply_masks(
        selected,
        selected_dataset,
        selection,
        mask,
        stretch,
        includepix,
        excludepix,
    )
    state = create_statistics_state(selected, selection.reduction_dims)
    return finalize_statistics_state(state, statistics) if finalize else state


imstatistics = image_statistics
