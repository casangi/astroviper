import xarray as xr


def moment_max(input_data, input_parms):
    # print(len(input_data),input_data[0].dims,input_data[1].dims)
    # print(input_data)
    # print('********')

    for i, img_xds in enumerate(input_data):
        input_data[i] = img_xds.max(dim="frequency", keepdims=True)

    img_xds = xr.concat(input_data, dim="frequency")
    return img_xds.max(dim="frequency", keepdims=True)
