// single_cf_gridder.h
#ifndef SINGLE_CF_GRIDDER_H
#define SINGLE_CF_GRIDDER_H
#include "../data_io/zarr_reader.h"
#include <iostream>
#include <cmath>
#include <complex>
#include <algorithm>
#include <vector>
class single_cf_gridder
{
//private:
//  int gx;
//  int gy;

public:
  single_cf_gridder();
  void create_grid(long n_imag_chan, long n_imag_pol, long image_size);
  // pass_grid
    
  void set_grid(std::complex<double>* grid);
    
  void add_to_grid(long* grid_shape, double* sum_weight, std::complex<double>* vis_data, long* vis_shape, double* uvw, double* freq_chan, long* chan_map, long* pol_map, double* weight, double* cgk_1D, double* delta_lm, int support, int oversampling);
    
  std::vector<double> create_prolate_spheroidal_kernel_1d(int oversampling, int support);
  std::pair<std::vector<double>, std::vector<double>> prolate_spheroidal_function(std::vector<double> u);
    
  std::pair<int, int> grid(std::string vis_data_folder, int image_size, int n_time_chunks, int n_chan_chunks);

private:
    std::vector<std::complex<double>> internal_grid;
    std::complex<double>* grid_ptr;
    bool use_internal_grid;
    bool grid_set;
};

#endif
