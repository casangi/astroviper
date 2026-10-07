"""Small generated images for continuum software tests; no external data."""

import numpy as np
import xarray as xr


def make_continuum_image(psf_axis="psf_taylor_order", beam_axis=None, legacy=False):
    coords = dict(
        time=[0.0],
        taylor_term=[0, 1],
        polarization=["I"],
        l=np.arange(8) * -1e-5,
        m=np.arange(8) * 1e-5,
    )
    dims = ("time", "taylor_term", "polarization", "l", "m")
    model = np.zeros((1, 2, 1, 8, 8))
    model[0, 0, 0, 4, 4] = 2.0
    model[0, 1, 0, 4, 4] = 17.0  # Higher Taylor terms must not be restored.
    image = xr.Dataset(
        {"SKY_MODEL": (dims, model), "SKY_RESIDUAL": (dims, np.full_like(model, 0.25))},
        coords=coords,
    )
    plane = xr.DataArray(
        np.ones((1, 1, 8, 8)),
        dims=("time", "polarization", "l", "m"),
        coords={k: image.coords[k] for k in ("time", "polarization", "l", "m")},
    )
    psf = plane
    beam = xr.DataArray(
        np.array([3e-5, 2e-5, 0.3]).reshape(1, 1, 3),
        dims=("time", "polarization", "beam_params"),
        coords={
            "time": [0.0],
            "polarization": ["I"],
            "beam_params": ["major", "minor", "pa"],
        },
    )
    for axis, kind in ((psf_axis, "psf"), (beam_axis, "beam")):
        if axis is not None:
            values = image[axis].values if axis in image.coords else [100.0]
            if kind == "psf":
                psf = psf.expand_dims({axis: values}, axis=1)
            else:
                beam = beam.expand_dims({axis: values}, axis=1)
    image["POINT_SPREAD_FUNCTION"] = psf
    image["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] = beam
    image["PRIMARY_BEAM"] = plane * 0.5
    residual = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
    if not legacy:
        residual.update(
            point_spread_function="POINT_SPREAD_FUNCTION",
            beam_fit_params_point_spread_function="BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
        )
    image.attrs = {
        "reference_frequency": 100.0,
        "data_groups": {"residual": residual, "model": {"sky": "SKY_MODEL"}},
    }
    return image
