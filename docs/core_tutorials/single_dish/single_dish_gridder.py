from typing import Annotated
from typing import Tuple
import numpy as np
from math import floor, ceil


def nint(x):
    """ Round to the nearest integer like Fortran nint

    Unlike Python round(), nint does not follow the banker rule
    for half integers

    0.30000001192092896    0       -0.30000001192092896      0
    0.50000000000000000    1       -0.50000000000000000     -1
    0.80000001192092896    1       -0.80000001192092896     -1
    1.2999999523162842     1       -1.2999999523162842      -1
    1.5000000000000000     2       -1.5000000000000000      -2
    2.5000000000000000     3       -2.5000000000000000      -3
    3.5000000000000000     4       -3.5000000000000000      -4
    4.5000000000000000     5       -4.5000000000000000      -5
    5.5000000000000000     6       -5.5000000000000000      -6
    6.5000000000000000     7       -6.5000000000000000      -7
    """
    if x >= 0:
        return floor(x+0.5)
    else:
        return ceil(x-0.5)


def test_nint():
    values = [0.3, 0.5, 0.8, 1.3, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    for x in values:
        print(f"{x} {nint(x)} {round(x)}   {-x} {nint(-x):2d} {round(-x):2d}")
    """ Test Result
    0.3 0 0   -0.3  0  0
    0.5 1 0   -0.5 -1  0
    0.8 1 1   -0.8 -1 -1
    1.3 1 1   -1.3 -1 -1
    1.5 2 2   -1.5 -2 -2
    2.5 3 2   -2.5 -3 -2
    3.5 4 4   -3.5 -4 -4
    4.5 5 4   -4.5 -5 -4
    5.5 6 6   -5.5 -6 -6
    6.5 7 6   -6.5 -7 -6
    """


def sgridsd(
        xy: Annotated[np.ndarray[np.float64], (2,)],
        sampling: int) -> Tuple[
            Annotated[np.ndarray[int], (2,)],
            Annotated[
                np.ndarray[int], (2,),
                "Values in [-sampling/2 ; sampling/2[ when sampling is even"
            ]
        ]:
    """ Calculate gridded coordinates

    Note
    ----
    Original Fortran code
    C
    C Calculate gridded coordinates
    C
      subroutine sgridsd (xy, sampling, pos, loc, off)
      implicit none
      integer sampling
      integer loc(2), off(2)
      double precision xy(2)
      real pos(2)
      integer idim

      do idim=1,2
         pos(idim)=xy(idim)+1.0
         loc(idim)=nint(pos(idim))
         off(idim)=nint((loc(idim)-pos(idim))*sampling)
      end do
      return 
      end

    """
    """ Note:
    loc(idim) = nint(pos(idim)) 
              = nint(xy(idim) + 1)
              = nint(xy(idim)) + 1
    off(idim) = nint( (loc(idim) - pos(idim))*sampling )
              = nint( (nint(xy(idim)) + 1) - (xy(idim)+1.0) )*sampling )
              = nint( nint(xy(idim) - xy(idim) )

    """
    # Python arrays are 0-based, so pos = xy
    nearest_grid_point = np.vectorize(nint)(xy)
    rounded_rescaled_offset = (
        np.vectorize(nint)((nearest_grid_point - xy)*sampling)
    )

    return nearest_grid_point, rounded_rescaled_offset


def is_on_grid(
        nx: int,
        ny: int,
        loc: Annotated[np.ndarray[np.int32], (2,)],
        support: int):
    """Is the given location on the grid ?

    Is the given location inside a grid of size (nx,ny) pixels,
    and at a distance of at least support pixels from its edges ?

    C
    C Is this on the grid?
    C
          logical function ogridsd (nx, ny, loc, support)
          implicit none
          integer nx, ny, loc(2), support
          ogridsd=(loc(1)-support.ge.1).and.(loc(1)+support.le.nx).and.
         $     (loc(2)-support.ge.1).and.(loc(2)+support.le.ny)
          return
          end
    """
    return (
        (loc[0] - support) >= 0 and
        (loc[0] + support) < nx and
        (loc[1] - support) >= 0 and
        (loc[1] + support) < ny
    )


def ggridsd(
            # In-outs
            # ---- Image data
            grid: Annotated[
                np.ndarray[np.complex64],
                "shape=(nx, ny, npol, nchan)"
            ] = None,
            # ---- Weight image data
            wgrid: Annotated[
                np.ndarray[np.float32],
                "shape=(nx, ny, npol, nchan)"
            ] = None,
            # ---- Sum of image data weights
            sumwt: Annotated[
                np.ndarray[np.float32],
                "shape=(npol, nchan)"
            ] = None,
            # Inputs
            # ---- Data and metadata
            xy: Annotated[
                np.ndarray[np.float64],
                "shape=(2, nrow)"
            ] = None,
            values: Annotated[
                np.ndarray[complex],
                "shape=(nvispol, nvischan, nrow)"
            ] = None,
            flag: Annotated[
                np.ndarray[bool],
                "shape=(nvispol, nvischan, nrow)"
            ] = None,
            irow: int = None,
            nrow: int = None,
            rflag: Annotated[np.ndarray[bool], "shape=(nrow, )"] = None,
            nvispol: int = None,
            nvischan: int = None,
            weight: Annotated[
                np.ndarray[float],
                "shape=(nvischan, nrow)"
            ] = None,
            # ---- Image array shape parameters
            nx: int = None,
            ny: int = None,
            npol: int = None,
            nchan: int = None,
            # ---- Functionality control
            grid_weight: bool = None,
            # ---- Data to image mappings
            chanmap: Annotated[np.ndarray[int], "shape=(nvischan, )"] = None,
            polmap: Annotated[np.ndarray[int], "shape=(nvispol, )"] = None,
            # ---- Convolution
            convFunc: Annotated[
                np.ndarray[np.float32],
                "shape=(unknown/runtime-defined, )"
            ] = None,
            support: int = None,
            sampling: int = None
        ):
    """Grid a number of visibility records: single dish gridding

    Parameters
    ----------
    Inputs
    xy : array of double, shape=(`nrow`, 2)
        Coordinates of antenna direction interpolated at data taking time,
        projected on image's spatial geometric plane.
    values : array of complex, shape=(`nrow`, )
        Values of data to be gridded
    irow : integer
         the data row to grid if positive, else grid all nrow rows of data
    rflag : array of bool, shape=(`nrow`)
         Row flag.

    chanmap : array of integers, shape=(`nchan`,)

    In-Out
    """

    # Local storage
    # Array storing the radii of circles
    # - centered on the coordinates of the data and passing by each point of
    # - a grid of (2**support +1)**2 points of cell side=sampling
    #   centered on the "convolution reference point"
    # - in units of 1/sampling pixel side
    # The "convolution reference point" is
    # - the nearest pixel on a grid of cell side 1/sampling centered on the data
    # - of the WCS pixel center nearest to the data
    #   (nearest point with integer coordinates)
    radius_to = np.zeros((2*support+1)**2, dtype=np.int32)

    # x and y coordinates of the corresponding (2**support+1)**2 points
    # on a grid of cell side 1, in units of pixel side,
    # centered on data's nearest pixel
    xloc = np.zeros(2*support+1, dtype=np.int32)
    yloc = np.zeros(2*support+1, dtype=np.int32)

    # Main loop
    rows = range(nrow) if irow < 0 else range(irow, irow + 1)
    for row in rows:
        if rflag[row]:
            continue
        nearest_grid_point, convolution_ref_point = (
            sgridsd(xy[:, row], sampling)
        )
        if not is_on_grid(nx, ny, nearest_grid_point, 0):
            continue
        # Initialize for distances computation
        # Coordinates here are with respect to a grid of cell side 1/sampling,
        # centered on the data
        # The grid cell size is (support*sampling)x(support*sampling)
        # Grid points coordinates offsets from the center are given by
        # c[k] = -(support+1)*sampling + (k+1)*sampling
        #      = (-support + k)*sampling, for k in [0; 2*support]
        # Offsets 
        # - start at:   c[0] = -support*sampling
        # - and end at: c[2*support] = support*sampling
        # So, the grid size is 2*support*sampling, 
        # its half-size is support*sampling, 

        point_index = 0
        left_to_grid_leftmost_x = (
            -(support + 1)*sampling + convolution_ref_point[0]
        )
        conv_point_y = -(support + 1)*sampling + convolution_ref_point[1]
        # Scan the grid of convolution points from its bottom left corner,
        # and compute the distance to each of them
        for iy in range(2*support + 1):
            conv_point_x = left_to_grid_leftmost_x
            conv_point_y += sampling
            for ix in range(2*support +1):
                conv_point_x += sampling
                radius_to[point_index] = np.linalg.norm(
                    [conv_point_x, conv_point_y]
                )
                point_index += 1
        # Compute xloc and yloc arrays, shape=(2*support +1,)
        # Coordinates here are with respect to the WCS grid of cell side 1
        xloc[0] = nearest_grid_point[0] - support
        yloc[0] = nearest_grid_point[1] - support
        for k in range(1, 2*support + 1):
            xloc[k] = xloc[k-1] + 1
            yloc[k] = yloc[k-1] + 1

        # Process array of data in current row
        for data_chan in range(nvischan):
            img_chan = chanmap[data_chan]
            if not (img_chan >= 0 and img_chan < nchan and
                    weight[data_chan, row] > 0
                    ):
                continue
            for data_pol in range(nvispol):
                img_pol = polmap[data_pol]
                if not ( not flag[data_pol, data_chan, row]
                         and img_pol >= 0
                         and img_pol < npol):
                    continue
                # Select data to grid
                if grid_weight:
                    data_val = complex(weight[data_chan, data_pol])
                else:
                    data_val = (
                        weight[data_chan, data_pol] *
                        values[data_pol, data_chan, row].conjugate()
                    )
                # Update gridded data and gridded weights:
                # distribute the value of current data
                # to surrounding grid points
                conv_weight_sum = 0.0
                for conv_point_y in range(2*support + 1):
                    img_y = yloc[conv_point_y]
                    if img_y < 0 or img_y >= ny:
                        continue
                    for conv_point_x in range(2*support + 1):
                        img_x = xloc[conv_point_x]
                        if img_x < 0 or img_x >= nx:
                            continue
                        conv_point_index = (
                            conv_point_y*(2*support+1) + conv_point_x
                        )
                        # Compute the convolution weight for this data,
                        # using the distance we computed at the nearby
                        # convolution point
                        convolution_weight = convFunc[
                            radius_to[conv_point_index]
                        ]
                        grid[img_x, img_y, img_pol, img_chan] += (
                            data_val * convolution_weight
                        )
                        wgrid[img_x, img_y, img_pol, img_chan] += (
                            weight[data_chan, row] * convolution_weight
                        )
                        conv_weight_sum += convolution_weight
                # Update sumwt
                sumwt[img_pol, img_chan] += (
                    weight[data_chan, row] * conv_weight_sum
                )
