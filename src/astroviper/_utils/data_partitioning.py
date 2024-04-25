import graphviper.utils.logger as logger

bytes_in_dtype = {'float32':4,'float64':8,'double':8,'complex64':8,'complex128':16}

def get_thread_info(client=None):
    
    if client is None: 
        try:
            from distributed import Client
            client = Client.current()
        except: #Using default Dask schedular.
            import psutil
            cpu_cores = psutil.cpu_count()
            total_memory = psutil.virtual_memory().total/(1024 ** 3)
            #print(cpu_cores,total_memory)
            thread_info = {'n_threads':cpu_cores,'memory_per_thread':total_memory/cpu_cores}
            return thread_info

    memory_per_thread = -1
    n_threads = 0
    for worker_name, worker in client.cluster.scheduler_info['workers'].items():
        temp_memory_per_thread = (worker['memory_limit']/worker['nthreads'])/(1024**3)
        n_threads = n_threads  + worker['nthreads']
        if (memory_per_thread == -1) or (memory_per_thread > temp_memory_per_thread):
            memory_per_thread = temp_memory_per_thread

    thread_info = {'n_threads':n_threads,'memory_per_thread':memory_per_thread}    
    return  thread_info

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def calculate_data_chunking(memory_singleton_chunk,chunking_dims_sizes,thread_info,constant_memory=0,tasks_per_thread=4):
    
    import numpy as np
    n_total_dims = np.prod(np.array(list(chunking_dims_sizes.values())))

    n_mem_chunks = int(np.ceil(n_total_dims/((thread_info['memory_per_thread'] - constant_memory)/memory_singleton_chunk)))
    assert n_mem_chunks < n_total_dims, "Not enough cluster memory per thread. Need at least " + str(memory_singleton_chunk) + " GiB but only " + str(thread_info['memory_per_thread']) + " GiB available."
    
    n_graph_chunks = int(thread_info['n_threads']*tasks_per_thread)
    if  n_graph_chunks > n_total_dims:
        n_graph_chunks = n_total_dims


    if n_mem_chunks > n_graph_chunks:
        n_chunks = n_mem_chunks
    else:
        n_chunks = n_graph_chunks


    logger.debug('Suggest n_chunks: ' + str(n_chunks) + ', n_mem_chunks: ' + str(n_mem_chunks) + ', n_graph_chunks: ' + str(n_graph_chunks))

    dims_sizes = []
    dims_names = []

    for dim_name in sorted(chunking_dims_sizes, key=chunking_dims_sizes.get):
        dims_sizes.append(chunking_dims_sizes[dim_name])
        dims_names.append(dim_name)

    dims_sizes_arr = np.array(dims_sizes)
    dims_names_arr = np.array(dims_names)

    if len(dims_names_arr) == 1:
        n_chunks_dict = dict(zip([dims_names_arr[0]], [n_chunks]))
        return n_chunks_dict
    else:
        factors = prime_factors(n_chunks)

        found_factors=False

        while not found_factors:

            if len(factors) > len(chunking_dims_sizes):
                
                n_reduce = len(factors) - len(chunking_dims_sizes)

                for i in range(n_reduce):
                    if factors[0]*factors[-(i+1)] < dims_sizes_arr[-(i+1)]:
                        new_factors = factors[1:]
                        new_factors[-(i+1)] = factors[0]*factors[-(i+1)]
                        factors = new_factors
                
            elif len(factors) < len(chunking_dims_sizes):
                while len(factors) < len(chunking_dims_sizes):
                    n_chunks = n_chunks+1
                    factors = prime_factors(n_chunks_inc)
            else:
                found_factors=True


        #print(len(chunking_dims_sizes), len(factors), factors)

        n_chunks_dict = dict(zip(dims_names_arr, factors))
        return n_chunks_dict
