def plot_astroviper_vs_casa_weights_interactive(
    ps_xdt, ms_xdt_name, robust, freq_idx, bl_idx
):
    import matplotlib.pyplot as plt
    import numpy as np

    ms_xdt = ps_xdt[ms_xdt_name]
    av = ms_xdt[f"WEIGHT_AV_R_{robust}"].isel(
        baseline_id=bl_idx, frequency=freq_idx, polarization=0
    )
    casa = ms_xdt[f"WEIGHT_CASA_R_{robust}"].isel(
        baseline_id=bl_idx, frequency=freq_idx
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(av, label="AstroVIPER WEIGHT")
    ax1.plot(casa, label="CASA WEIGHT")

    if robust == 2:
        natural_weights = ms_xdt[f"WEIGHT"].isel(baseline_id=bl_idx, frequency=freq_idx)
        ax1.plot(natural_weights, label="Natural WEIGHT", linestyle="--")

    ax1.set(
        ylabel="Imaging Weight",
        title=f"R={robust} | freq={freq_idx} | baseline={bl_idx}",
    )

    per_dif = 100 * (av - casa) / (casa.max())
    ax2.plot(per_dif, label="Percentage Difference (AstroVIPER - CASA)")
    ax2.set(xlabel="Time Index", ylabel="Percentage Difference")

    print("The max difference is:", np.nanmax(per_dif))

    ax1.grid(True)
    ax1.legend()
    plt.show()
    plt.close(fig)


def plot_astroviper_vs_casa_weights_imshow_interactive(
    ps_xdt, ms_xdt_name, robust, bl_idx
):
    import matplotlib.pyplot as plt
    import numpy as np

    ms_xdt = ps_xdt[ms_xdt_name]
    av = ms_xdt[f"WEIGHT_AV_R_{robust}"].isel(baseline_id=bl_idx, polarization=0).values
    casa = ms_xdt[f"WEIGHT_CASA_R_{robust}"].isel(baseline_id=bl_idx).values
    av[av == 0] = np.nan
    casa[casa == 0] = np.nan

    per_dif = 100 * (av - casa) / (np.nanmax(casa))

    # Wide figure for horizontal layout
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

    im_kwargs = dict(aspect="auto", interpolation="nearest")

    # Plot AstroVIPER
    im1 = ax1.imshow(av, **im_kwargs)
    ax1.set_box_aspect(1)
    ax1.set_title(f"AstroVIPER | R={robust} | baseline={bl_idx}")
    ax1.set_ylabel("Imaging Weight")
    ax1.set_xlabel("Frequency Index")
    ax1.set_ylabel("Time Index")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot CASA
    im2 = ax2.imshow(casa, **im_kwargs)
    ax2.set_box_aspect(1)
    ax2.set_title(f"CASA | R={robust} | baseline={bl_idx}")
    ax2.set_ylabel("Imaging Weight")
    ax2.set_xlabel("Frequency Index")
    ax2.set_ylabel("Time Index")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Plot % difference
    im3 = ax3.imshow(per_dif, **im_kwargs)
    ax3.set_box_aspect(1)
    ax3.set_title("Percentage Difference (AstroVIPER - CASA)")
    ax3.set_xlabel("Frequency Index")
    ax3.set_ylabel("Time Index")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    print("The max difference is:", float(np.nanmax(per_dif)))

    plt.show()
    plt.close(fig)
