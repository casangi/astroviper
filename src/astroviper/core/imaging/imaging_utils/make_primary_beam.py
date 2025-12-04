#generate primary beam image using make_pb_symmetric.casa_airy_disk_rorder
from astroviper.core.imaging.imaging_utils.make_pb_symmetric import casa_airy_disk_rorder
from xradio.image import make_empty_sky_image

def cube_single_field_primary_beam(im_params, field_id, telescope):
    """
    Generate a primary beam image for a single field in a measurement set.

    Parameters:
    im_params : dict
        Imaging parameters including 'cell', 'imsize', and 'freq'.
    field_id : int
        The field ID for which to generate the primary beam.
    telescope : str
        The name of the telescope (e.g., 'ALMA', 'ACA').

    Returns:
    pb_image : np.ndarray
        The generated primary beam image.
    """

    # ALMA parameters: 
    # ALMA 12m: effective diameter 10.7m, blockage 0.75m
    # ACA 7m: effective diameter 6.7m, blockage 0.75m
    match telescope:
        case 'ALMA':
            dish_diameter = 10.7
            blockage_diameter = 0.75
        case 'ACA':
            dish_diameter = 6.7
            blockage_diameter = 0.75
        case _:
            raise ValueError(f"Unsupported telescope: {telescope}")
    
    pb_params = {
        'list_dish_diameters': [10.7],  # in meters
        'list_blockage_diameter': [0.75],  # in meters
        'ipower': 2
    }
    cell = im_params['cell']
    imsize = im_params['imsize']
    freq_chan = im_params['freq']

    pb_image_data = casa_airy_disk_rorder(
        freq_chan,
        pol,
        pb_params,
        im_params

    )
    
    make_empty_sky_image
    return pb_image
)
            
        
