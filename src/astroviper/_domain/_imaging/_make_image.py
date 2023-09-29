

def _make_image(input_parms):
    #print(input_parms['ps_name'])
    #print(input_parms.keys())
    from xradio.vis.load_processing_set import load_processing_set
    ps = load_processing_set(ps_name=input_parms['ps_name'],sel_parms=input_parms['data_sel'])
    

    #print(ps.keys())
   
    import numpy as np
    from astroviper._domain._visibility._phase_shift import _phase_shift_vis_ds
    from astroviper._domain._imaging._make_imaging_weights import _make_imaging_weights
    from astroviper._domain._imaging._make_gridding_convolution_function import _make_gridding_convolution_function
    from astroviper._domain._imaging._make_aperture_grid import _make_aperture_grid
    from astroviper._domain._imaging._make_uv_sampling_grid import _make_uv_sampling_grid
    from astroviper.image.make_empty_sky_image import make_empty_sky_image
    from astroviper._domain._imaging._make_visibility_grid import _make_visibility_grid
    from astroviper._domain._imaging._fft_norm_img_xds import _fft_norm_img_xds

    from xradio.vis.load_processing_set import load_processing_set
    import xarray as xr

    shift_parms={}
    shift_parms['new_phase_center'] = ps['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_12'].attrs['field_info']['phase_dir'] #should be 13?
    shift_parms["common_tangent_reprojection"] = True
    #print(shift_parms)

    grid_parms = {}
    grid_parms['chan_mode'] = 'cube'
    grid_parms['image_size'] = [500,500]
    grid_parms['cell_size'] = np.array([-0.13,0.13])*np.pi/(180*3600)
    grid_parms['fft_padding'] = 1.0
    grid_parms['phase_center'] = shift_parms['new_phase_center']
    #print('grid_parms[phase_center]',grid_parms['phase_center'])
    #print(ps['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_1'].attrs['field_info']['phase_dir'])

    #print(ps['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_1'])
    #print(ps['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_1'].frequency)

    freq_coords = ps['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_1'].frequency
    chan_width = ps['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_1'].frequency.values
    pol_coords = ps['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_1'].polarization
    time_coords = np.mean(ps['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_1'].time.values)

    #print(freq_coords.shape,input_parms['data_sel']['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_1'])

    #print(img_xds)
    img_xds = xr.Dataset()
    img_xds = make_empty_sky_image(img_xds,grid_parms['phase_center'],grid_parms['image_size'],grid_parms['cell_size'],freq_coords.values,chan_width,pol_coords,time_coords,data_group_name='mosaic')
    
    #for ms_xds in [list(ps.values())[0]]:
    for ms_xds in ps.values():

        #print(ms_xds)
        data_group_out = _phase_shift_vis_ds(ms_xds,shift_parms=shift_parms,sel_parms={})

        #data_group_out = _make_imaging_weights(ms_xds,grid_parms=grid_parms,imaging_weights_parms={'weighting':'briggs','robust':0.6},sel_parms={"data_group_in":data_group_out})
        data_group_out = _make_imaging_weights(ms_xds,grid_parms=grid_parms,imaging_weights_parms={'weighting':'natural','robust':0.6},sel_parms={"data_group_in":data_group_out})

        gcf_parms = {}
        gcf_parms['function'] = 'casa_airy'
        gcf_parms['list_dish_diameters'] = np.array([10.7])
        gcf_parms['list_blockage_diameters'] = np.array([0.75])
        
        #print(ms_xds.attrs['antenna_xds'])
        unique_ant_indx = ms_xds.attrs['antenna_xds'].dish_diameter.values
        unique_ant_indx[unique_ant_indx == 12.0] = 0
        
        gcf_parms['unique_ant_indx'] = unique_ant_indx.astype(int)
        gcf_parms['phase_center'] = grid_parms['phase_center']
        gcf_xds = _make_gridding_convolution_function(ms_xds,gcf_parms,grid_parms,sel_parms={"data_group_in":data_group_out})
        #print(gcf_xds)
        #print(ms_xds['WEIGHT_IMAGING'])

        _make_aperture_grid(ms_xds,gcf_xds,img_xds,vis_sel_parms={"data_group_in":data_group_out},img_sel_parms={"data_group_in":"mosaic"},grid_parms=grid_parms)

        _make_uv_sampling_grid(ms_xds,gcf_xds,img_xds,vis_sel_parms={"data_group_in":data_group_out},img_sel_parms={"data_group_in":"mosaic"},grid_parms=grid_parms)

        _make_visibility_grid(ms_xds,gcf_xds,img_xds,vis_sel_parms={"data_group_in":data_group_out},img_sel_parms={"data_group_in":"mosaic"},grid_parms=grid_parms)

    _fft_norm_img_xds(img_xds,gcf_xds,grid_parms,norm_parms={},sel_parms={"data_group_in":"mosaic","data_group_out":"mosaic"})
    

    #Tranform uv-space -> lm-space (sky)

    return img_xds


'''

    ps_name = '/Users/jsteeb/Dropbox/Data/Antennae_North.cal.lsrk.vis.zarr'
    sel_parms = {}
    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_1'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_2'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_3'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_13'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_4'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_5'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_6'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_7'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_8'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_9'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_10'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_11'] = {}
    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_12'] = {}

#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_14'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_15'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_16'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_17'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_18'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_19'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_20'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_21'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_22'] = {}
#    sel_parms['Antennae_North.cal.lsrk_ddi_0_intent_OBSERVE_TARGET#ON_SOURCE_field_id_23'] = {}

    ps = load_processing_set(ps_name,sel_parms)

'''
