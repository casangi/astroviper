import math
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def single_dish_gridder_jit(
    # Inputs
    samples_values=None,
    samples_weights=None,
    samples_coords=None,
    # Outputs
    grid_values=None,
    grid_weights=None,
    # Convolution Parameters
    cut_radius=None,
    sampling=None,
    convolution_array=None,
):
    """

    Parameters
    ----------
    samples_values : array
        with shape (n_samples, n_chan, n_pol)
    samples_weights : float array
        with shape (n_samples, n_chan, n_pol)
    samples_coords  : array
        with shape (n_samples, 2)
    grid_values : array
        Image with spatial dimensions width = n_x pixels, height = n_y pixels
        with shape (n_x, n_y, n_chan, n_pol)
    grid_weights : array
        Weight Image with spatial dimensions width = n_x pixels, height = n_y pixels
        with shape (n_x, n_y, n_chan, n_pol)
    cut_radius : int
        Cut radius of a 2D convolution function with circular symmetry, in pixels unit
    sampling : int
        Number of samples per pixel of the convolution function's 1D kernel
    kernel_1D : array
        1D array storing the 1D kernel of the 2D convolution function with circular
        symmetry, sampled over `cut_radius` pixels at rate `sampling` samples per pixel
        with shape (`cut_radius` * `sampling` + 1)

    Returns
    -------
    """
    n_samples, n_chan, n_pol = samples_values.shape
    n_x, n_y = grid_values.shape[0:2]

    conv_array_size = convolution_array.size

    r_2 = cut_radius * cut_radius

    for sample_index in range(n_samples):
        sample_x = samples_coords[sample_index, 0]
        sample_y = samples_coords[sample_index, 1]
        # How should we handle borders ?
        # For now, ignore contributions from samples
        # falling outside the imaged region
        if not ((0 <= sample_x < n_x) and (0 <= sample_y < n_y)):
            continue
        # Ignore flagged pointings
        if np.isnan(sample_x) or np.isnan(sample_y):
            continue
        # When there are 2 or 4 nearest grid points,
        # select one using the banker's rule:
        # round half-integers to the nearest even integer
        nearest_grid_point_x = round(sample_x)
        nearest_grid_point_y = round(sample_y)
        # Scatter sample's value over surrounding grid pixels
        for dx in range(-cut_radius, cut_radius + 1):
            neighbor_x = nearest_grid_point_x + dx
            if not (0 <= neighbor_x < n_x):
                continue
            neighbor_offset_x = neighbor_x - sample_x
            neighbor_offset_x_2 = neighbor_offset_x * neighbor_offset_x
            max_neighbor_offset_y_2 = r_2 - neighbor_offset_x_2
            for dy in range(-cut_radius, cut_radius + 1):
                neighbor_y = nearest_grid_point_y + dy
                if not (0 <= neighbor_y < n_y):
                    continue
                neighbor_offset_y = neighbor_y - sample_y
                neighbor_offset_y_2 = neighbor_offset_y * neighbor_offset_y
                if neighbor_offset_y_2 > max_neighbor_offset_y_2:
                    continue
                neighbor_offset_norm = math.sqrt(
                    neighbor_offset_x_2 + neighbor_offset_y_2
                )
                # Each neighbor pixel receives a contribution given by
                # the convolution kernel, evaluated at pixel's rescaled distance
                # to the sample location.
                neighbor_rescaled_distance = neighbor_offset_norm * sampling
                # Nearest interpolation, round half to even
                conv_index = round(neighbor_rescaled_distance)
                if conv_index >= conv_array_size:
                    continue
                convolution_weight = convolution_array[conv_index]
                for chan_index in range(n_chan):
                    for pol_index in range(n_pol):
                        sample_value = samples_values[
                            sample_index, chan_index, pol_index
                        ]
                        if np.isnan(sample_value):
                            continue
                        sample_weight = samples_weights[
                            sample_index, chan_index, pol_index
                        ]
                        if sample_weight == 0:
                            continue
                        weighted_value = np.conjugate(sample_value) * sample_weight
                        grid_values[neighbor_x, neighbor_y, chan_index, pol_index] += (
                            convolution_weight * weighted_value
                        )
                        grid_weights[neighbor_x, neighbor_y, chan_index, pol_index] += (
                            convolution_weight * sample_weight
                        )
