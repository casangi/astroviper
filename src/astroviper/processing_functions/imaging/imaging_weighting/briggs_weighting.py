def calculate_briggs_params(grid_of_imaging_weights, sum_weight, imaging_weights_parms):
    import numpy as np

    if imaging_weights_parms.get("weighting", "natural") in ("briggs", "uniform"):
        robust = (
            -2.0
            if imaging_weights_parms["weighting"] == "uniform"
            else imaging_weights_parms["robust"]
        )
        briggs_factors = np.ones(
            (2,) + sum_weight.shape
        )  # 2 x chan x pol. Briggs denominator: briggs_factor[0,...] * G + briggs_factor[1,...] where G is the degridded weight. See grid_imaging_weights function.

        squared_sum_weight = np.sum((grid_of_imaging_weights) ** 2, axis=(2, 3))

        density_ratio = np.divide(
            squared_sum_weight,
            sum_weight,
            out=np.zeros_like(sum_weight),
            where=sum_weight > 0,
        )
        briggs_factors[0] = np.divide(
            np.square(5.0 * 10.0 ** (-robust)),
            density_ratio,
            out=np.zeros_like(sum_weight),
            where=density_ratio > 0,
        )

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
