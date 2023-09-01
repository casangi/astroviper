

def _wrap_func_with_dio(mds, chunk_function,  chunk_function_input_parms, mds_sel_parms_chunk):

    if mds is Dataset:
        mds_chunk = mds
    else:
        #Load the data from cache or common storage. If data loads from common storage it can be cache.
        mds_chunk = transfer_mds_chunk(mds_name, mds_sel_parms_chunk)
    
    #Run the chunk function on the above selected mds_chunk.
    mds_result_chunk = chunk_function(mds_chunk, chunk_function_input_parms)
    
    
    if  chunk_function_input_parms['return_data']:
        return mds_result_chunk
    else:
        ## Save the results
        transfer_mds_chunk(mds_result_chunk)
        results_dict = mds_result_chunk['results_dict']
        return results_dict


