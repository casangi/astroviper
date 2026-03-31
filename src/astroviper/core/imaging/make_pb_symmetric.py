# Simple 1D Cases
# Airy Disk dish, blockage, freq
# Gaussian halfWidth
# Poly
# Cos Polyp
# Inverse Poly coeff


# Formula for obscured airy pattern found in https://en.wikipedia.org/wiki/Airy_disk (see Obscured Airy pattern section)
# If ipower is 1 the voltage pattern is returned if ipower is 2 the primary beam is returned.
def airy_disk(freq_chan, pol, pb_params, grid_params):
    """
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    """

    import numpy as np
    import scipy.constants
    from scipy.special import jn

    cell = grid_params["cell_size"]
    image_size = grid_params["image_size"]
    image_center = grid_params["image_center"]

    list_dish_diameters = pb_params["list_dish_diameters"]
    list_blockage_diameters = pb_params["list_blockage_diameters"]
    ipower = pb_params["ipower"]

    c = scipy.constants.c  # 299792458
    k = (2 * np.pi * freq_chan) / c

    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell[0]
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell[1]

    airy_disk_size = (
        image_size[0],
        image_size[1],
        len(freq_chan),
        1,
        len(list_blockage_diameters),
    )  # len(pol) is set initially to 1. For now, the PB is assumed the same. This will change.
    airy_disk = np.zeros(airy_disk_size)

    for i, (dish_diameter, blockage_diameter) in enumerate(
        zip(list_dish_diameters, list_blockage_diameters)
    ):
        aperture = dish_diameter / 2
        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

        # r_grid = (np.sin(np.sqrt(x_grid**2 + y_grid**2))[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid = (
            np.sqrt(x_grid**2 + y_grid**2)[:, :, None] * k * aperture
        )  # d0 x d1 x chan
        r_grid[image_center[0], image_center[1], :] = (
            1.0  # Avoid the 0/0 for the centre value.
        )

        if blockage_diameter == 0.0:
            airy_disk[:, :, :, 0, i] = (2.0 * jn(1, r_grid) / r_grid) ** ipower
        else:
            e = blockage_diameter / dish_diameter
            airy_disk[:, :, :, 0, i] = (
                (2.0 * jn(1, r_grid) / r_grid - 2.0 * e * jn(1, r_grid * e) / r_grid)
                / (1.0 - e**2)
            ) ** ipower

    airy_disk[image_center[0], image_center[1], :, 0, :] = 1.0  # Fix centre value
    airy_disk = np.tile(airy_disk, (1, 1, 1, len(pol), 1))

    return airy_disk


# Formula for obscured airy pattern found in casa6/casa5/code/synthesis/TransformMachines/PBMath1DAiry.cc/h
# If ipower is 1 the voltage pattern is returned if ipower is 2 the primary beam is returned.
def casa_airy_disk(freq_chan, pol, pb_params, grid_params):
    """
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    """

    import numpy as np
    import scipy.constants
    from scipy.special import jn

    cell = grid_params["cell_size"]
    image_size = grid_params["image_size"]
    image_center = grid_params["image_center"]

    list_dish_diameters = pb_params["list_dish_diameters"]
    list_blockage_diameters = pb_params["list_blockage_diameters"]
    ipower = pb_params["ipower"]

    c = scipy.constants.c  # 299792458
    k = (2 * np.pi * freq_chan) / c

    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell[0]
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell[1]

    airy_disk_size = (
        image_size[0],
        image_size[1],
        len(freq_chan),
        1,
        len(list_blockage_diameters),
    )  # len(pol) is set initially to 1. For now, the PB is assumed the same. This will change.
    airy_disk = np.zeros(airy_disk_size)

    for i, (dish_diameter, blockage_diameter) in enumerate(
        zip(list_dish_diameters, list_blockage_diameters)
    ):
        aperture = dish_diameter / 2
        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

        # r_grid = (np.sin(np.sqrt(x_grid**2 + y_grid**2))[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid = (
            np.sqrt(x_grid**2 + y_grid**2)[:, :, None] * k * aperture
        )  # d0 x d1 x chan
        r_grid[image_center[0], image_center[1], :] = (
            1.0  # Avoid the 0/0 for the centre value.
        )

        if blockage_diameter == 0.0:
            airy_disk[:, :, :, 0, i] = (2.0 * jn(1, r_grid) / r_grid) ** ipower
        else:
            area_ratio = (dish_diameter / blockage_diameter) ** 2
            length_ratio = dish_diameter / blockage_diameter
            airy_disk[:, :, :, 0, i] = (
                (
                    area_ratio * 2.0 * jn(1, r_grid) / r_grid
                    - 2.0 * jn(1, r_grid * length_ratio) / (r_grid * length_ratio)
                )
                / (area_ratio - 1.0)
            ) ** ipower

    airy_disk[image_center[0], image_center[1], :, 0, :] = 1.0  # Fix centre value
    airy_disk = np.tile(airy_disk, (1, 1, 1, len(pol), 1))

    return airy_disk


# Functions used during the creatiuon of the gridding convolution functions.
# Formula for obscured airy pattern found in https://en.wikipedia.org/wiki/Airy_disk (see Obscured Airy pattern section)
# If ipower is 1 the voltage pattern is returned if ipower is 2 the primary beam is returned.
def airy_disk_rorder(freq_chan, pol, pb_params, grid_params):
    """
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    """

    import numpy as np
    import scipy.constants
    from scipy.special import jn

    cell = grid_params["cell_size"]
    image_size = grid_params["image_size"]
    image_center = grid_params["image_center"]

    list_dish_diameters = pb_params["list_dish_diameters"]
    list_blockage_diameters = pb_params["list_blockage_diameters"]
    ipower = pb_params["ipower"]

    c = scipy.constants.c  # 299792458
    k = (2 * np.pi * freq_chan) / c

    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell[0]
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell[1]

    airy_disk_size = (
        len(list_blockage_diameters),
        len(freq_chan),
        1,
        image_size[0],
        image_size[1],
    )  # len(pol) is set initially to 1. For now, the PB is assumed the same. This will change.
    airy_disk = np.zeros(airy_disk_size)

    for i, (dish_diameter, blockage_diameter) in enumerate(
        zip(list_dish_diameters, list_blockage_diameters)
    ):
        aperture = dish_diameter / 2
        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

        # r_grid = (np.sin(np.sqrt(x_grid**2 + y_grid**2))[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid = np.moveaxis(
            (np.sqrt(x_grid**2 + y_grid**2)[:, :, None] * k * aperture), 2, 0
        )  # chan x d0 x d1
        r_grid[:, image_center[0], image_center[1]] = (
            1.0  # Avoid the 0/0 for the centre value.
        )

        if blockage_diameter == 0.0:
            airy_disk[i, :, 0, :, :] = (2.0 * jn(1, r_grid) / r_grid) ** ipower
        else:
            e = blockage_diameter / dish_diameter
            airy_disk[i, :, 0, :, :] = (
                (2.0 * jn(1, r_grid) / r_grid - 2.0 * e * jn(1, r_grid * e) / r_grid)
                / (1.0 - e**2)
            ) ** ipower

    airy_disk[:, :, 0, image_center[0], image_center[1]] = 1.0  # Fix centre value
    # airy_disk[airy_disk<pb_limit] = 0.0
    airy_disk = np.tile(airy_disk, (1, 1, len(pol), 1, 1))

    return airy_disk


# Fast version of airy_disk_rorder using a 1D lookup table + np.interp.
# jn is evaluated only on a compact 1D grid (~10k points) instead of the full
# (chan x N0 x N1) array, then linearly interpolated back onto the 2D image.
# The interpolation query is fully vectorised across channels via broadcasting.
def airy_disk_rorder_v2(freq_chan, pol, pb_params, grid_params):
    """
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    """

    import numpy as np
    import scipy.constants
    from scipy.special import jn

    cell = grid_params["cell_size"]
    image_size = grid_params["image_size"]
    image_center = grid_params["image_center"]

    list_dish_diameters = pb_params["list_dish_diameters"]
    list_blockage_diameters = pb_params["list_blockage_diameters"]
    ipower = pb_params["ipower"]

    c = scipy.constants.c  # 299792458
    k = (2 * np.pi * freq_chan) / c

    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell[0]
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell[1]

    airy_disk_size = (
        len(list_blockage_diameters),
        len(freq_chan),
        1,
        image_size[0],
        image_size[1],
    )
    airy_disk = np.zeros(airy_disk_size)

    # Compute 2D radius once — shared across all dishes and channels.
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    r_2d = np.sqrt(x_grid**2 + y_grid**2)  # (N0, N1)

    for i, (dish_diameter, blockage_diameter) in enumerate(
        zip(list_dish_diameters, list_blockage_diameters)
    ):
        aperture = dish_diameter / 2

        # u = r * k * aperture.  Build a 1D lookup table over the full u range.
        u_max = float(r_2d.max() * k.max() * aperture)
        # ~10 samples per π gives <0.1 % linear-interpolation error for J1(u)/u.
        n_samples = max(int(u_max * 10.0 / np.pi) + 1, 10000)
        u_1d = np.linspace(0.0, u_max * 1.001, n_samples)
        u_safe = u_1d.copy()
        u_safe[0] = 1.0  # avoid 0/0 at origin

        if blockage_diameter == 0.0:
            f_1d = 2.0 * jn(1, u_safe) / u_safe
        else:
            e = blockage_diameter / dish_diameter
            f_1d = (
                2.0 * jn(1, u_safe) / u_safe - 2.0 * e * jn(1, u_safe * e) / u_safe
            ) / (1.0 - e**2)
        f_1d[0] = 1.0  # limit at u = 0

        if ipower != 1:
            f_1d **= ipower

        # u_3d[ci, n0, n1] = r_2d[n0, n1] * k[ci] * aperture  — no Python loop.
        u_3d = r_2d[np.newaxis, :, :] * (k[:, np.newaxis, np.newaxis] * aperture)
        airy_disk[i, :, 0, :, :] = np.interp(u_3d, u_1d, f_1d)

    airy_disk[:, :, 0, image_center[0], image_center[1]] = 1.0  # Fix centre value
    airy_disk = np.tile(airy_disk, (1, 1, len(pol), 1, 1))

    return airy_disk


# Formula for obscured airy pattern found in casa6/casa5/code/synthesis/TransformMachines/PBMath1DAiry.cc/h
# If ipower is 1 the voltage pattern is returned if ipower is 2 the primary beam is returned.
def casa_airy_disk_rorder(freq_chan, pol, pb_params, grid_params):
    """
    Does not yet handle beam squint
    dish_diameters : list of int
    blockage_diameters : list of int
    frequencies : list of number
    """

    import numpy as np
    import scipy.constants
    from scipy.special import jn

    cell = grid_params["cell_size"]
    image_size = grid_params["image_size"]
    image_center = grid_params["image_center"]

    list_dish_diameters = pb_params["list_dish_diameters"]
    list_blockage_diameters = pb_params["list_blockage_diameters"]
    ipower = pb_params["ipower"]

    c = scipy.constants.c  # 299792458
    k = (2 * np.pi * freq_chan) / c

    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell[0]
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell[1]

    airy_disk_size = (
        len(list_blockage_diameters),
        len(freq_chan),
        1,
        image_size[0],
        image_size[1],
    )  # len(pol) is set initially to 1. For now, the PB is assumed the same. This will change.
    airy_disk = np.zeros(airy_disk_size)

    for i, (dish_diameter, blockage_diameter) in enumerate(
        zip(list_dish_diameters, list_blockage_diameters)
    ):
        aperture = dish_diameter / 2
        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

        # r_grid = (np.sin(np.sqrt(x_grid**2 + y_grid**2))[:,:,None]*k*aperture) #d0 x d1 x chan
        r_grid = np.moveaxis(
            (np.sqrt(x_grid**2 + y_grid**2)[:, :, None] * k * aperture), 2, 0
        )  # chan x d0 x d1
        r_grid[:, image_center[0], image_center[1]] = (
            1.0  # Avoid the 0/0 for the centre value.
        )

        if blockage_diameter == 0.0:
            airy_disk[i, :, 0, :, :] = (2.0 * jn(1, r_grid) / r_grid) ** ipower
        else:
            area_ratio = (dish_diameter / blockage_diameter) ** 2
            length_ratio = dish_diameter / blockage_diameter
            airy_disk[i, :, 0, :, :] = (
                (
                    area_ratio * 2.0 * jn(1, r_grid) / r_grid
                    - 2.0 * jn(1, r_grid * length_ratio) / (r_grid * length_ratio)
                )
                / (area_ratio - 1.0)
            ) ** ipower

    airy_disk[:, :, 0, image_center[0], image_center[1]] = 1.0  # Fix centre value
    airy_disk = np.tile(airy_disk, (1, 1, len(pol), 1, 1))

    return airy_disk
