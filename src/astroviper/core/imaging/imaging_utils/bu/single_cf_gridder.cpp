#include "single_cf_gridder.h"
#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

single_cf_gridder::single_cf_gridder()
{
    use_internal_grid = false;
}

void single_cf_gridder::set_grid(std::complex<double>* grid)
{
    grid_ptr = grid;
    use_internal_grid = false;
}


void single_cf_gridder::create_grid(long n_imag_chan,long n_imag_pol,long image_size)
{
    int grid_size = n_imag_chan*n_imag_pol*image_size*image_size;
    internal_grid.resize(grid_size,std::complex<double>(0.0, 0.0));
    use_internal_grid = true;
}

void single_cf_gridder::add_to_grid(long* grid_shape, double* sum_weight, std::complex<double>* vis_data, long* vis_shape, double* uvw, double* freq_chan, long* chan_map, long* pol_map, double* weight, double* cgk_1D, double* delta_lm, int support, int oversampling) {
    
    auto start = chrono::high_resolution_clock::now();
    std::complex<double>* grid;
    
    if(use_internal_grid){
        grid = internal_grid.data();
    }else{
        grid = grid_ptr;
    }

    int n_time = vis_shape[0];
    int n_baseline = vis_shape[1];
    int n_chan = vis_shape[2];
    int n_pol = vis_shape[3];

    int n_u = grid_shape[2];
    int n_v = grid_shape[3];
    double c = 299792458.0;
    
    
    support = 7;
    oversampling = 100;

    int support_center = support / 2;
    int u_center = n_u / 2;
    int v_center = n_v / 2;

    int start_support = -support_center;
    int end_support = support - support_center;

    for (int i_time = 0; i_time < n_time; i_time++) {
        for (int i_baseline = 0; i_baseline < n_baseline; i_baseline++) {
            for (int i_chan = 0; i_chan < n_chan; i_chan++) {
                int a_chan = chan_map[i_chan];
                double u = uvw[i_time * n_baseline * 3 + i_baseline * 3 + 0] * (-(freq_chan[i_chan] * delta_lm[0] * n_u) / c);
                double v = uvw[i_time * n_baseline * 3 + i_baseline * 3 + 1] * (-(freq_chan[i_chan] * delta_lm[1] * n_v) / c);

                if (!std::isnan(u) && !std::isnan(v)) {
                    double u_pos = u + u_center;
                    double v_pos = v + v_center;

                    int u_center_indx = static_cast<int>(u_pos + 0.5);
                    int v_center_indx = static_cast<int>(v_pos + 0.5);

                    if (u_center_indx + support_center < n_u && v_center_indx + support_center < n_v && u_center_indx - support_center >= 0 && v_center_indx - support_center >= 0) {
                        double u_offset = u_center_indx - u_pos;
                        int u_center_offset_indx = static_cast<int>(std::floor(u_offset * oversampling + 0.5));
                        double v_offset = v_center_indx - v_pos;
                        int v_center_offset_indx = static_cast<int>(std::floor(v_offset * oversampling + 0.5));

                        for (int i_pol = 0; i_pol < n_pol; i_pol++) {
                            double sel_weight = weight[i_time * n_baseline * n_chan * n_pol + i_baseline * n_chan * n_pol + i_chan * n_pol + i_pol];
                            std::complex<double> weighted_data = vis_data[i_time * n_baseline * n_chan * n_pol + i_baseline * n_chan * n_pol + i_chan * n_pol + i_pol] * weight[i_time * n_baseline * n_chan * n_pol + i_baseline * n_chan * n_pol + i_chan * n_pol + i_pol];

                            if (!std::isnan(weighted_data.real()) && !std::isnan(weighted_data.imag()) && weighted_data.real() != 0.0 && weighted_data.imag() != 0.0) {
                                int a_pol = pol_map[i_pol];
                                double norm = 0.0;

                                for (int i_v = start_support; i_v < end_support; i_v++) {
                                    int v_indx = v_center_indx + i_v;
                                    int v_offset_indx = std::abs(oversampling * i_v + v_center_offset_indx);
                                    double conv_v = cgk_1D[v_offset_indx];

                                    for (int i_u = start_support; i_u < end_support; i_u++) {
                                        int u_indx = u_center_indx + i_u;
                                        int u_offset_indx = std::abs(oversampling * i_u + u_center_offset_indx);
                                        double conv_u = cgk_1D[u_offset_indx];
                                        double conv = conv_u * conv_v;
                                        grid[a_chan * n_pol * n_u * n_v + a_pol * n_u * n_v + u_indx * n_v + v_indx] += conv * weighted_data;
                                        norm += conv;
                                    }
                                }

                                sum_weight[a_chan * n_pol + a_pol] += sel_weight * norm;
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    auto end = chrono::high_resolution_clock::now();
    cout << "@@@@@@ Grid time " <<  chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
}


std::pair<int, int> single_cf_gridder::grid(std::string vis_data_folder, int image_size, int n_time_chunks, int n_chan_chunks)
{
    
    double field_of_view = 60*M_PI/(180*3600);
    int oversampling = 100;
    int support = 7;
    int n_imag_pol = 2;

    auto cgk_1D = create_prolate_spheroidal_kernel_1d(oversampling, support);
    vector<double> delta_lm = {field_of_view/static_cast<double>(image_size), field_of_view/static_cast<double>(image_size)};
    
    int data_load_time = 0;
    int gridding_time = 0;


    complex<double>* vis;
    double* weight;
    double* uvw;
    double* chan;
    long* vis_shape;

    int chan_chunk_size = 2;
    int grid_size = chan_chunk_size*n_chan_chunks*n_imag_pol*image_size*image_size;
    cout << "grid_size " << grid_size << ",*," << chan_chunk_size << ",*," << n_chan_chunks << ",*," << n_imag_pol << ",*," << image_size <<  endl;
    //std::vector<std::complex<double>> grid(grid_size,std::complex<double>(0.0, 0.0));
    create_grid(chan_chunk_size*n_chan_chunks, n_imag_pol, image_size);
    
    vector<long> grid_shape = {chan_chunk_size*n_chan_chunks, n_imag_pol, image_size, image_size};
    //std::complex<double>* grid_ptr = grid.data();

    //std::vector<double> sum_weight;
    //sum_weight.resize(chan_chunk_size*n_chan_chunks*n_imag_pol, 0.0);
    std::vector<double> sum_weight(chan_chunk_size*n_chan_chunks*n_imag_pol, 0.0);
    double* sum_weight_ptr = sum_weight.data();

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    std::vector<double> chan_slice;
    std::vector<long> chan_map;
    std::vector<long> pol_map;

    int start_index;
    int end_index;
    int k;

    for (int i_time_chunk = 0; i_time_chunk < n_time_chunks; ++i_time_chunk) {
       for (int i_chan_chunk = 0; i_chan_chunk < n_chan_chunks; ++i_chan_chunk)
       {
           start = chrono::high_resolution_clock::now();
           open_no_dask_zarr(&vis,&weight,&uvw,&chan,&vis_shape,vis_data_folder, i_time_chunk, i_chan_chunk);
           
           chan_slice.resize(vis_shape[2]);
           chan_map.resize(vis_shape[2]);
           pol_map.resize(vis_shape[3]);

           start_index = i_chan_chunk * vis_shape[2];
           end_index = (i_chan_chunk + 1) * vis_shape[2];
           k = 0;
           for (long i = start_index; i < end_index; ++i) {
               chan_map[k] = i;
               chan_slice[k] = chan[i];
               //cout << "chan_map " << k << ",*," << chan_map[k] << ",*," << chan_slice[k] << endl;
               k++;
           }

           for (long i = 0; i < vis_shape[3]; ++i) {
                pol_map[i] = i;
                //cout << "pol_map " << i << ",*," << pol_map[i] << endl;
           }

           end = chrono::high_resolution_clock::now();
           data_load_time = data_load_time + chrono::duration_cast<chrono::milliseconds>(end - start).count();


           start = chrono::high_resolution_clock::now();
           add_to_grid(grid_shape.data(), sum_weight_ptr, vis, vis_shape, uvw, chan_slice.data(), chan_map.data(), pol_map.data(), weight, cgk_1D.data(), delta_lm.data(), support, oversampling);
           end = chrono::high_resolution_clock::now();
           cout << "*Grid time " <<  chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
           gridding_time = gridding_time + chrono::duration_cast<chrono::milliseconds>(end - start).count();

           delete[] vis;
           delete[] uvw;
           delete[] weight;
           delete[] chan;
           delete[] vis_shape;
       }
    }
    
//    cout << "the grid is " << grid[12502500] << ",*," << grid[37502500] << ",*," << grid[262502500] << ",*," << grid[287502500] << ",*," << grid[512502500] << ",*," << grid[537502500] << ",*," << grid[962502500] << ",*," << grid[987502500] << endl;
//
//    cout  << "****************" << endl;
//    for (int i_chan = 0; i_chan < chan_chunk_size*n_chan_chunks; ++i_chan) {
//        for (int i_pol = 0; i_pol < n_imag_pol; ++i_pol){
//            cout << "sum_weight " << i_chan*n_imag_pol + i_pol << " " << sum_weight[i_chan*n_imag_pol + i_pol] << endl;
//        }
//    }
//    cout  << "****************" << endl;
    cout << "Data load time " << data_load_time << endl;
    cout << "Data gridding time " << gridding_time << endl;
    cout << image_size << ",*," << n_time_chunks << ",*," << n_chan_chunks << endl;
    
    return std::make_pair(data_load_time, gridding_time);
}


std::vector<double> single_cf_gridder::create_prolate_spheroidal_kernel_1d(int oversampling, int support) {
    int support_center = support / 2;

    std::vector<double> u(oversampling * (support_center), 0.0);
    for (int i = 0; i < oversampling * (support_center); i++) {
        u[i] = static_cast<double>(i) / (support_center * oversampling);
    }
    
    std::vector<double> long_half_kernel_1d(oversampling * (support_center + 1), 0.0);
    std::pair<std::vector<double>, std::vector<double>> grdsf = prolate_spheroidal_function(u);

    for (int i = 0; i < oversampling * (support_center); i++) {
        //cout << u[i] << ",*," << grdsf.first[i] << ",*," << grdsf.second[i] << endl;
        long_half_kernel_1d[i] = grdsf.second[i];
    }
    return long_half_kernel_1d;
}



std::pair<std::vector<double>, std::vector<double>> single_cf_gridder::prolate_spheroidal_function(std::vector<double> u) {
    std::vector<std::vector<double>> p = {{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
                                          {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}};
    std::vector<std::vector<double>> q = {{1.0000000e0, 8.212018e-1, 2.078043e-1},
                                          {1.0000000e0, 9.599102e-1, 2.918724e-1}};

    int n_p = p[0].size();
    int n_q = q[0].size();

    std::vector<double> u_abs(u.size());
    std::transform(u.begin(), u.end(), u_abs.begin(), [](double val) { return std::abs(val); });

    std::vector<double> uend(u.size(), 0.0);
    std::vector<int> part(u.size(), 0);

    for (size_t i = 0; i < u.size(); ++i) {
        if (u_abs[i] >= 0.0 && u_abs[i] < 0.75) {
            part[i] = 0;
            uend[i] = 0.75;
        } else if (u_abs[i] >= 0.75 && u_abs[i] <= 1.0) {
            part[i] = 1;
            uend[i] = 1.0;
        }
    }

    std::vector<double> delusq(u.size());
    for (size_t i = 0; i < u.size(); ++i) {
        delusq[i] = std::pow(u_abs[i], 2) - std::pow(uend[i], 2);
    }

    std::vector<double> top(u.size(), 0.0);
    for (int k = 0; k < n_p; ++k) {
        for (size_t i = 0; i < u.size(); ++i) {
            top[i] += p[part[i]][k] * std::pow(delusq[i], k);
        }
    }

    std::vector<double> bot(u.size(), 0.0);
    for (int k = 0; k < n_q; ++k) {
        for (size_t i = 0; i < u.size(); ++i) {
            bot[i] += q[part[i]][k] * std::pow(delusq[i], k);
        }
    }

    std::vector<double> grdsf(u.size(), 0.0);
    for (size_t i = 0; i < u.size(); ++i) {
        if (bot[i] > 0.0) {
            grdsf[i] = top[i] / bot[i];
        }
        if (std::abs(u_abs[i]) > 1.0) {
            grdsf[i] = 0.0;
        }
    }

    std::vector<double> correcting_image(u.size());
    std::transform(u.begin(), u.end(), grdsf.begin(), correcting_image.begin(), [](double u_val, double grdsf_val) {
        return (1 - std::pow(u_val, 2)) * grdsf_val;
    });

    return std::make_pair(grdsf, correcting_image);
}




