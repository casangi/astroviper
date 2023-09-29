import xarray as xr

def _moment_max(img_xds_1,img_xds_2,input_parms):
    img_xds = xr.concat([img_xds_1.max(dim='frequency',keepdims=True),img_xds_2.max(dim='frequency',keepdims=True)],dim='frequency')
    return img_xds
