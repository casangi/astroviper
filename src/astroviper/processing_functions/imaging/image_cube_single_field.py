

def image_cube_single_field(input_params, ps_xdt, img_xds):
    import toolviper.utils.logger as logger
    from astroviper.processing_functions.imaging.residual_cycle import residual_cycle_cube_single_field
    from astroviper.processing_functions.imaging.model_update_cycle import model_update_cycle_cube_single_field
    import time
    from astroviper.processing_functions.imaging.iteration_control import (
        IterationController,
        ReturnDict,
        merge_return_dicts,
    )


    logger.debug("Processing chunk " + str(input_params["task_id"]))
    
    controller = IterationController(
        niter=input_params["iteration_control_params"]["niter"],
        nmajor=input_params["iteration_control_params"]["nmajor"],
        threshold=input_params["iteration_control_params"]["threshold"],
        gain=input_params["iteration_control_params"]["gain"],
        cyclefactor=input_params["iteration_control_params"]["cyclefactor"],
        minpsffraction=input_params["iteration_control_params"]["minpsffraction"],
        maxpsffraction=input_params["iteration_control_params"]["maxpsffraction"],
        cycleniter=input_params["iteration_control_params"]["cycleniter"],
    )
    combined_deconvolve_dict = ReturnDict()

    is_n_iter_0 = True
    T_residual_cycle = 0.0
    T_model_update_cycle = 0.0
    i_cycles = 0
    while controller.stopcode.major == 0:
        i_cycles += 1
        print("$$$$************" * 10, i_cycles)
        start = time.time()
        img_xds, return_df = residual_cycle_cube_single_field(
            ps_xdt, img_xds, input_params, is_n_iter_0=is_n_iter_0
        )
        T_residual_cycle = T_residual_cycle + time.time() - start

        if input_params["iteration_control_params"]["niter"] > 0:
            logger.debug("Doing model update")
            cycle_niter, cyclethresh = get_calculate_cycle_controls(controller, combined_deconvolve_dict, img_xds, is_n_iter_0, iteration_control_params=input_params["iteration_control_params"])
            
            input_params["iteration_control_params"]["cycleniter"] = cycle_niter
            input_params["iteration_control_params"]["threshold"] = cyclethresh
            start = time.time()
            deconvolve_dict = model_update_cycle_cube_single_field(img_xds, input_params, is_n_iter_0=is_n_iter_0, num_threads=input_params["processing_function_threads"], img_data_group_in_name = "residual", img_data_group_out_name = "model")
            T_model_update_cycle = T_model_update_cycle + time.time() - start
        else:
            deconvolve_dict = ReturnDict()
            
        is_n_iter_0 = False
        
        controller.update_counts(deconvolve_dict)

        # check_convergence stamps the stop code into deconvolve_dict, so run
        # it before the merge to carry that stop code into the combined dict.
        stopcode, stopdesc = controller.check_convergence(deconvolve_dict)
        combined_deconvolve_dict = merge_return_dicts([combined_deconvolve_dict, deconvolve_dict])

        if stopcode.major != 0:
            logger.info(f"  *** CONVERGED: {stopdesc} ***")
            break
                
    #Last residual cycle to calcultate final residual image after last model update cycle.
    if input_params["iteration_control_params"]["niter"] > 0:       
        start = time.time()
        img_xds, return_df = residual_cycle_cube_single_field(
            ps_xdt, img_xds, input_params, is_n_iter_0=is_n_iter_0
        )
        T_residual_cycle = T_residual_cycle + time.time() - start
        

    return_df["task_id"] = input_params["task_id"]
    return_df["n_channels"] = len(input_params["task_coords"]["frequency"]["data"])
    return_df["T_residual_cycle"] = T_residual_cycle
    return_df["T_model_update_cycle"] = T_model_update_cycle
    
    print("@@@@@@@@@ Combined Deconvolve Dict:")
    print_deconvolve_dict(combined_deconvolve_dict)
    print("***************")

    # #Write Data chunk to disk
    return img_xds, return_df

def get_calculate_cycle_controls(controller,combined_deconvolve_dict, img_xds, is_n_iter_0, iteration_control_params, residual_data_group_name = "residual",):
    from astroviper.processing_functions.imaging.iteration_control import (
        IterationController,
        ReturnDict,
        merge_return_dicts,
    )
    import numpy as np
    residual_data_group = img_xds.attrs["data_groups"][residual_data_group_name]
    if is_n_iter_0:
        peak_res = np.max(np.abs(img_xds[residual_data_group["sky"]].values))
        temp_rd = ReturnDict()
        temp_rd.add(
            {
                "peakres": peak_res,
                "peakres_nomask": peak_res,
                "masksum": img_xds.sizes["l"] * img_xds.sizes["m"],
                "iter_done": 0,
                "max_psf_sidelobe": iteration_control_params["maxpsffraction"],
                "loop_gain": iteration_control_params["gain"],
            },
            time=0,
            pol=0,
            chan=0,
        )
        cycle_niter, cyclethresh = controller.calculate_cycle_controls(temp_rd)
    else:
        cycle_niter, cyclethresh = controller.calculate_cycle_controls(
            combined_deconvolve_dict
        )

    return cycle_niter, cyclethresh


def format_deconvolve_dict(combined_deconvolve_dict, float_format="{:.6g}"):
    """Return a human-readable string representation of a deconvolution ReturnDict.

    A deconvolution ReturnDict maps ``Key(time, pol, chan)`` planes to field
    dicts that mix constant parameters (``niter``, ``threshold``, ...) with
    per-major-cycle history lists (``peakres``, ``iter_done``, ``model_flux``,
    ...). The default ``repr`` dumps each plane on one very long line, which is
    hard to read. This formatter groups each plane, separates scalar parameters
    from the per-cycle history, and lays the history out as an aligned table
    with one column per major cycle. Numpy scalars (``np.float64``, ``np.str_``)
    are unwrapped so they print as plain values.

    Parameters
    ----------
    combined_deconvolve_dict : ReturnDict or dict
        Either a ReturnDict instance or its underlying ``.data`` mapping of
        ``Key(time, pol, chan)`` -> field dict.
    float_format : str, optional
        Format string applied to floating point values (default ``"{:.6g}"``).

    Returns
    -------
    str
        The formatted, multi-line representation.
    """
    # Accept either a ReturnDict (exposes .data) or a plain mapping.
    data = getattr(combined_deconvolve_dict, "data", combined_deconvolve_dict)

    def to_py(v):
        # Unwrap numpy scalars (np.float64, np.str_, ...) to native Python types.
        return v.item() if hasattr(v, "item") else v

    def fmt(v):
        v = to_py(v)
        if v is None:
            return "None"
        if isinstance(v, float):
            return float_format.format(v)
        return str(v)

    if not data:
        return "<empty deconvolve dict>"

    lines = []
    for key, fields in data.items():
        # Split fields into per-cycle history (lists) and scalar parameters.
        history = {f: list(v) for f, v in fields.items() if isinstance(v, list)}
        scalars = {f: v for f, v in fields.items() if not isinstance(v, list)}

        time = getattr(key, "time", key[0] if len(key) > 0 else "?")
        pol = getattr(key, "pol", key[1] if len(key) > 1 else "?")
        chan = getattr(key, "chan", key[2] if len(key) > 2 else "?")
        stokes = scalars.get("stokes")
        header = f"Key(time={time}, pol={pol}, chan={chan})"
        if stokes is not None:
            header += f"  -  Stokes {to_py(stokes)}"

        lines.append("=" * 78)
        lines.append(header)
        lines.append("-" * 78)

        # Scalar parameters (skip stokes, already shown in the header).
        scalar_items = [(f, v) for f, v in scalars.items() if f != "stokes"]
        if scalar_items:
            label_w = max(len(f) for f, _ in scalar_items)
            lines.append("Parameters:")
            for f, v in scalar_items:
                lines.append(f"  {f:<{label_w}} : {fmt(v)}")

        # Per-cycle history table (one column per major cycle).
        if history:
            n_cycles = max(len(v) for v in history.values())
            label_w = max([len(f) for f in history] + [len("cycle")])
            # Format every cell, then right-justify to a uniform width.
            formatted = {f: [fmt(x) for x in v] for f, v in history.items()}
            cycle_labels = [str(i) for i in range(n_cycles)]
            cell_w = max(
                [len(c) for cells in formatted.values() for c in cells]
                + [len(c) for c in cycle_labels]
            )

            def row(label, cells):
                padded = " ".join(f"{c:>{cell_w}}" for c in cells)
                return f"  {label:<{label_w}} : {padded}"

            plural = "s" if n_cycles != 1 else ""
            lines.append(f"Per-cycle history ({n_cycles} cycle{plural}):")
            lines.append(row("cycle", cycle_labels))
            for f, cells in formatted.items():
                lines.append(row(f, cells))
        lines.append("")

    return "\n".join(lines)


def print_deconvolve_dict(combined_deconvolve_dict, float_format="{:.6g}"):
    """Pretty-print a deconvolution ReturnDict. See :func:`format_deconvolve_dict`."""
    print(format_deconvolve_dict(combined_deconvolve_dict, float_format=float_format))

