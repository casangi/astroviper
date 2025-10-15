from astroviper.core.imaging.fft import fft_lm_to_uv
from astroviper.core.imaging.ifft import ifft_uv_to_lm
from astropy import units as u
import xarray as xr

_sky = "SKY"
_beam = "BEAM"
import numpy as np
from xradio.image import make_empty_aperture_image
from xradio.image import load_image
import toolviper.utils.logger as logger


def compute_u_v(xds):
    shape = [xds.dims["l"], xds.dims["m"]]
    # sics = np.abs(xds.attrs["direction"]["reference"]["cdelt"])
    sics = np.abs(2 * [xds.l[1] - xds.l[0]])
    w_xds = make_empty_aperture_image(
        phase_center=[0, 0],
        image_size=shape,
        sky_image_cell_size=sics,
        chan_coords=[1],
        pol_coords=["I"],
        time_coords=[0],
    )
    u = np.zeros(shape)
    v = np.zeros(shape)
    for i, ux in enumerate(w_xds.coords["u"]):
        u[i, :] = ux
    for i, vx in enumerate(w_xds.coords["v"]):
        v[:, i] = vx
    return (u, v)


def feather_core(input_params):
    # display(HTML(dict_to_html(input_params)))

    def compute_w_multiple_beams(xds, uv):
        """xds is the single dish xds"""
        beams = xds[_beam]
        # logger.debug("beams " + str(beams))
        w = np.zeros(xds[_sky].shape)
        bunit = beams.attrs["units"]
        bmaj = beams.sel(beam_param="major")
        # logger.debug("bmaj orig shape" + str(bmaj.shape))
        # add l and m dims
        bmaj = np.expand_dims(bmaj, -1)
        bmaj = np.expand_dims(bmaj, -1)
        # logger.debug("bmaj shape " + str(bmaj.shape))
        alpha = bmaj * u.Unit(bunit)
        alpha = alpha.to(u.rad).value
        bmin = beams.sel(beam_param="minor")
        bmin = np.expand_dims(bmin, -1)
        bmin = np.expand_dims(bmin, -1)
        beta = bmin * u.Unit(bunit)
        beta = beta.to(u.rad).value
        bpa = beams.sel(beam_param="pa")
        bpa = np.expand_dims(bpa, -1)
        bpa = np.expand_dims(bpa, -1)
        phi = bpa * u.Unit(bunit)
        phi = phi.to(u.rad).value

        alpha2 = alpha * alpha
        beta2 = beta * beta
        # u -> uu, v -> vv because we've already used
        # u for astropy.units
        uu, vv = uv
        uu = uu[np.newaxis, np.newaxis, np.newaxis, :, :]
        vv = vv[np.newaxis, np.newaxis, np.newaxis, :, :]
        aterm2 = (uu * np.sin(phi) - vv * np.cos(phi)) ** 2
        bterm2 = (uu * np.cos(phi) + vv * np.sin(phi)) ** 2
        w = np.exp(
            -np.pi * np.pi / 4.0 / np.log(2) * (alpha2 * aterm2 + beta2 * bterm2)
        )
        # w is an np.array
        return w

    # if input_params["input_data"] is None: #Load
    dtypes = {"sd": np.int32, "int": np.int32}
    for k in ["sd", "int"]:
        # the "data_selection" key is set in
        # interpolate_data_coords_onto_parallel_coords()
        # print("data store", input_params["input_data_store"][k])
        # print("block_des", input_params["data_selection"][k])
        xds = load_image(
            input_params["input_data_store"][k],
            block_des=input_params["data_selection"][k],
        )
        # print("load image for", k, "complete")
        # print("completed load_image()")
        fft_plane = (
            xds[_sky].dims.index(input_params["axes"][0]),
            xds[_sky].dims.index(input_params["axes"][1]),
        )
        # print("completed fft_plane")
        # else:
        #   img_xds = input_params["input_data"]['img'] #In memory
        aperture = fft_lm_to_uv(xds[_sky], fft_plane)
        # print("completed _fft_im_to_uv()")
        dtypes[k] = xds[_sky].dtype
        if k == "int":
            int_ap = aperture
            int_xds = xds
            # logger.debug("int_xds beam " + str(int_xds[_beam]))
        else:
            sd_ap = aperture
            sd_xds = xds
            # logger.debug("sd_xds beam " + str(sd_xds[_beam]))
    mytype = dtypes["sd"] if dtypes["sd"] < dtypes["int"] else dtypes["int"]

    uv = compute_u_v(sd_xds)
    w = compute_w_multiple_beams(sd_xds, uv)

    one_minus_w = 1 - w
    s = input_params["s"]
    if _beam in int_xds.data_vars:
        int_ba = int_xds[_beam].sel(beam_param="major") * int_xds[_beam].sel(
            beam_param="minor"
        )
    else:
        error_message = "Unable to find BEAM data variable in interferometer image."
        logger.error(error_message)
        raise Exception(error_message)

    if _beam in sd_xds.data_vars:
        sd_ba = sd_xds[_beam].sel(beam_param="major") * sd_xds[_beam].sel(
            beam_param="minor"
        )
    else:
        error_message = "Unable to find BEAM data variable in single dish image."
        logger.error(error_message)
        raise Exception(error_message)
    # need to use values becuase the obs times will in general be different
    # which would cause a resulting shape with the time dimension having
    # length 0
    beam_ratio_values = int_ba.values / sd_ba.values
    # use interferometer coords
    beam_ratio = xr.DataArray(
        beam_ratio_values, dims=int_ba.dims, coords=int_ba.coords.copy()
    )

    beam_ratio = np.expand_dims(beam_ratio, -1)
    beam_ratio = np.expand_dims(beam_ratio, -1)

    term = (one_minus_w * int_ap + s * beam_ratio * sd_ap) / (one_minus_w + s * w)
    feather_npary = ifft_uv_to_lm(term, fft_plane).astype(mytype)
    from xradio.image._util._zarr.zarr_low_level import write_chunk

    from xradio.image import make_empty_sky_image

    """
    # FIXME lon/latpole is not the phase center
    phase_center = [
        sd_xds.attrs["direction"]["longpole"]["value"],
        int_xds.attrs["direction"]["latpole"]["value"],
    ]

    featherd_img_chunk_xds = make_empty_sky_image(
        phase_center=phase_center,
        image_size=[int_xds.sizes["l"], int_xds.sizes["m"]],
        cell_size=int_xds.attrs["direction"]["reference"]["cdelt"],
        chan_coords=int_xds.frequency.values,
        pol_coords=int_xds.polarization.values,
        time_coords=[0],
    )
    """

    featherd_img_chunk_xds = xr.Dataset(coords=int_xds.coords)
    # we need an xradio function to return an ordered list of dimensions
    featherd_img_chunk_xds[_sky] = xr.DataArray(
        feather_npary, dims=["time", "frequency", "polarization", "l", "m"]
    )
    parallel_dims_chunk_id = dict(
        zip(input_params["parallel_dims"], input_params["chunk_indices"])
    )

    # print('input_params["zarr_meta"]',input_params["zarr_meta"])
    if input_params["to_disk"]:
        for data_variable, meta in input_params["zarr_meta"].items():
            write_chunk(
                featherd_img_chunk_xds,
                meta,
                parallel_dims_chunk_id,
                input_params["compressor"],
                input_params["image_file"],
            )

        results_dict = {}
        return results_dict
    else:
        results_dict = {"featherd_img_chunk_xds": featherd_img_chunk_xds}
        return featherd_img_chunk_xds
