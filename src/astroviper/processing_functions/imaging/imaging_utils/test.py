@jit(nopython=True, cache=True, nogil=True)
def dgrid_numba(
    uvw,
    dphase,
    values,
    flag,
    scale,
    offset,
    grid,
    freq,
    c,
    supporta,
    samplinga,
    convFunc,
    chanmap,
    polmap,
):

    support = 7
    sampling = 100
    nvispol = values.shape[3]
    nvischan = values.shape[2]
    nchan, npol, ny, nx = grid.shape
    nt, nb = values.shape[:2]
    supp_centre = support // 2
    supp_beg = -supp_centre
    supp_end = supp_beg + support

    pos = np.empty(2, dtype=np.float64)
    loc = np.empty(2, dtype=np.int32)
    off = np.empty(2, dtype=np.int32)

    for t in prange(nt):
        for b in range(nb):
            for ipol in range(nvispol):
                apol = polmap[ipol]
                if 0 <= apol < npol:
                    for ichan in range(nvischan):
                        achan = chanmap[ichan]
                        if 0 <= achan < nchan and not flag[t, b, ichan, ipol]:
                            # phasor, loc, off = sgrid_numba(
                            #     uvw[t, b],
                            #     dphase[t, b],
                            #     freq[ichan],
                            #     c,
                            #     scale,
                            #     offset,
                            #     sampling,
                            # )

                            #########################
                            pos[0] = (
                                scale[0] * uvw[t, b, 0] * freq[ichan] / c + offset[0]
                            )
                            pos[1] = (
                                scale[1] * uvw[t, b, 1] * freq[ichan] / c + offset[1]
                            )

                            # loc[0] = int(np.round(pos[0]))
                            # loc[1] = int(np.round(pos[1]))

                            # off[0] = int(np.round((float(loc[0]) - pos[0]) * float(sampling)))
                            # off[1] = int(np.round((float(loc[1]) - pos[1]) * float(sampling)))

                            loc[0] = int(math.floor(pos[0] + 0.5))
                            loc[1] = int(math.floor(pos[1] + 0.5))

                            off[0] = int(
                                math.floor(
                                    (float(loc[0]) - pos[0]) * float(sampling) + 0.5
                                )
                            )
                            off[1] = int(
                                math.floor(
                                    (float(loc[1]) - pos[1]) * float(sampling) + 0.5
                                )
                            )

                            phase = -2.0 * np.pi * dphase[t, b] * freq[ichan] / c
                            phasor = np.cos(phase) + 1j * np.sin(phase)
                            #########################

                            x0, y0 = loc
                            if (
                                (x0 + supp_beg >= 0)
                                and ((x0 + supp_end - 1) < nx)
                                and (y0 + supp_beg >= 0)
                                and ((y0 + supp_end - 1) < ny)
                            ):
                                nvalue = 0.0
                                norm = 0.0
                                # if b == 100 and t == 10 and ichan == 35:
                                #     print("grid pos", achan, apol, x0, y0)
                                #     print("vis pos", t, b, ichan, ipol)
                                #     print("uvw", uvw[t, b, 0], uvw[t, b, 1])

                                for ix in range(supp_beg, supp_end):
                                    iloc1 = abs(sampling * ix + off[0])
                                    wtx = convFunc[iloc1]
                                    for iy in range(supp_beg, supp_end):
                                        iloc2 = abs(sampling * iy + off[1])
                                        wty = convFunc[iloc2]
                                        wt = wtx * wty
                                        norm += wt
                                        nvalue += (
                                            wt
                                            * grid[
                                                achan,
                                                apol,
                                                x0 + ix,
                                                y0 + iy,
                                            ]
                                        )

                                if norm != 0.0:
                                    values[t, b, ichan, ipol] += (
                                        nvalue * np.conj(phasor)
                                    ) / norm
