def calculate_briggs_params(grid_of_imaging_weights, sum_weight, imaging_weights_parms):
    import numpy as np

    if imaging_weights_parms["weighting"] == "briggs":
        robust = imaging_weights_parms["robust"]
        briggs_factors = np.ones(
            (2,) + sum_weight.shape
        )  # 2 x chan x pol. Briggs denominator: briggs_factor[0,...] * G + briggs_factor[1,...] where G is the degridded weight. See grid_imaging_weights function.

        squared_sum_weight = np.sum((grid_of_imaging_weights) ** 2, axis=(2, 3))

        with np.errstate(invalid="ignore", divide="ignore"):
            briggs_factors[0, :, :] = (
                np.square(5.0 * 10.0 ** (-robust)) / (squared_sum_weight / (sum_weight))
            )[None, None, :, :]

        # print("briggs_factors", briggs_factors)
    elif imaging_weights_parms["weighting"] == "briggs_abs":
        robust = imaging_weights_parms["robust"]
        briggs_factors = np.ones((2,) + sum_weight.shape)
        briggs_factors[0, :, :] = briggs_factors[0, :, :] * np.square(robust)
        briggs_factors[1, :, :] = (
            briggs_factors[1, :, :]
            * 2.0
            * np.square(imaging_weights_parms["briggs_abs_noise"])
        )
    else:
        briggs_factors = np.zeros((2, 1, 1) + sum_weight.shape)
        briggs_factors[0, :, :] = np.ones((1, 1, 1) + sum_weight.shape)

    return briggs_factors
