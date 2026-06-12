def generate_data():
    ps_store = "twhya_selfcal_lsrk_5chans.ps.zarr"
    import dask

    dask.config.set(scheduler="synchronous")

    import os
    import numpy as np
    from xradio.measurement_set import open_processing_set

    # from astroviper.distributed_graphs.imaging.image_cube_single_field import (
    #     image_cube_single_field,
    # )

    ps_xdt = open_processing_set(ps_store)
    combined_field_and_source_xds = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
    phase_direction = combined_field_and_source_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=center_field_name
    )
    image_params = {
        "image_size": [250, 250],
        "cell_size": np.array([-0.1, 0.1]) * np.pi / (180 * 3600),
        "phase_direction": phase_direction.values,
        "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }

    from xradio.image import load_image
    import xarray as xr

    weighting = ["briggs", "natural"]
    robust = ["0.5", ""]
    specmode = ["cube", "cubedata"]
    perchanweightdensity = [True, False]
    interpolation = ["nearest", "linear"]

    weighting_setup = [
        {"weighting": "briggs", "robust": 0.5, "casa_weighting_implementation": True},
        {"weighting": "natural", "robust": 0.0, "casa_weighting_implementation": True},
        {"weighting": "briggs", "robust": 0.5, "casa_weighting_implementation": False},
        {"weighting": "natural", "robust": 0.0, "casa_weighting_implementation": False},
    ]

    for i, w in enumerate(weighting_setup):

        imaging_weights_params = w

        print("imaging_weights_params: ", imaging_weights_params)

        image_store = (
            "test_image_"
            + w["weighting"]
            + "_robust_"
            + str(w["robust"])
            + "_casa_"
            + str(w["casa_weighting_implementation"])
            + ".zarr"
        )

        import astroviper.distributed_graphs as distributed_graphs

        imaging_metadata_pd = distributed_graphs.imaging.image_cube_single_field(
            ps_store=ps_store,
            image_store=image_store,
            image_params=image_params,
            imaging_weights_params=imaging_weights_params,
            iteration_control_params={
                "niter": 0,
                "nmajor": 0,
                "threshold": 0.0,
                "gain": 0.1,
                "cyclefactor": 1.5,
                "cycleniter": 1,
            },
            gridder="prolate_spheroidal",
            deconvolver="hogbom",
            scan_intents="OBSERVE_TARGET#ON_SOURCE",
            image_data_variables_keep=[
                "sky_residual",
                "point_spread_function",
                "primary_beam",
                "beam_fit_params_point_spread_function",
                "visibility_normalization",
                "uv_sampling_normalization",
            ],
            # image_data_variables_keep=[ "sky", "point_spread_function", "primary_beam"],
            processing_set_data_group_name="base",
            single_precision_image=False,
            thread_info=None,
            processing_function_threads=1,
            n_chunks=None,
            overwrite=True,
            disk_chunk_sizes={"frequency": 2},
            vizualize_graph=True,
        )
        img_av_xds = xr.open_zarr(image_store)


def plot_data():
    import xarray as xr
    import matplotlib.pyplot as plt

    weighting_setup = [
        {"weighting": "briggs", "robust": 0.5, "casa_weighting_implementation": True},
        {"weighting": "natural", "robust": 0.0, "casa_weighting_implementation": True},
        {"weighting": "briggs", "robust": 0.5, "casa_weighting_implementation": False},
        {"weighting": "natural", "robust": 0.0, "casa_weighting_implementation": False},
    ]

    psfs = []
    labels = []
    for w in weighting_setup:
        image_store = (
            "test_image_"
            + w["weighting"]
            + "_robust_"
            + str(w["robust"])
            + "_casa_"
            + str(w["casa_weighting_implementation"])
            + ".zarr"
        )
        img_xds = xr.open_zarr(image_store)
        psf = img_xds["POINT_SPREAD_FUNCTION"].isel(frequency=0, polarization=0, time=0)
        psfs.append(psf)
        labels.append(
            f"{w['weighting']} r={w['robust']} casa={w['casa_weighting_implementation']}"
        )

    nl, nm = psfs[0].shape
    center_l, center_m = nl // 2, nm // 2

    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    for psf, label in zip(psfs, labels):
        axes[0].plot(psf.isel(m=center_m).values, label=label)
        axes[1].plot(psf.isel(l=center_l).values, label=label)

    axes[0].set_title("PSF cross section along l (m = center)")
    axes[0].set_xlabel("l pixel")
    axes[0].set_ylabel("PSF")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("PSF cross section along m (l = center)")
    axes[1].set_xlabel("m pixel")
    axes[1].set_ylabel("PSF")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("psf_weighting_comparison.png", dpi=150)

    n = len(psfs)
    diffs = [[psfs[i].values - psfs[j].values for j in range(n)] for i in range(n)]
    vmax = max(
        abs(diffs[i][j][:, center_m]).max() if i != j else 0.0
        for i in range(n)
        for j in range(n)
    )
    vmax = max(
        vmax,
        max(
            abs(diffs[i][j][center_l, :]).max() if i != j else 0.0
            for i in range(n)
            for j in range(n)
        ),
    )
    ylim = (-vmax * 1.05, vmax * 1.05)

    fig_diff, axes_diff = plt.subplots(n, n, figsize=(18, 16), sharex=True, sharey=True)
    for i in range(n):
        for j in range(n):
            ax = axes_diff[i, j]
            diff = diffs[i][j]
            ax.plot(diff[:, center_m], label="l-cut (m=center)")
            ax.plot(diff[center_l, :], label="m-cut (l=center)")
            ax.set_title(f"{labels[i]}\n  -  {labels[j]}", fontsize=8)
            ax.grid(True)
            ax.set_ylim(ylim)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
            if i == n - 1:
                ax.set_xlabel("pixel")
            if j == 0:
                ax.set_ylabel("PSF difference")

    fig_diff.suptitle("PSF cross-section differences (row - column)", fontsize=12)
    fig_diff.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig_diff.savefig("psf_weighting_difference_grid.png", dpi=150)

    pairs = []
    for i, wi in enumerate(weighting_setup):
        if not wi["casa_weighting_implementation"]:
            continue
        for j, wj in enumerate(weighting_setup):
            if (
                wj["casa_weighting_implementation"]
                or wj["weighting"] != wi["weighting"]
                or wj["robust"] != wi["robust"]
            ):
                continue
            pairs.append((i, j))

    casa_diffs = [psfs[i].values - psfs[j].values for i, j in pairs]
    vmax_casa = max(
        max(abs(d[:, center_m]).max(), abs(d[center_l, :]).max()) for d in casa_diffs
    )
    ylim_casa = (-vmax_casa * 1.05, vmax_casa * 1.05)

    fig_casa, axes_casa = plt.subplots(
        len(pairs), 2, figsize=(14, 5 * len(pairs)), sharex=True, sharey=True
    )
    for ri, ((i, j), diff) in enumerate(zip(pairs, casa_diffs)):
        ax_l = axes_casa[ri, 0]
        ax_m = axes_casa[ri, 1]
        ax_l.plot(diff[:, center_m])
        ax_m.plot(diff[center_l, :])
        title = f"{labels[i]}\n  -  {labels[j]}"
        ax_l.set_title(title + "    (l-cut, m=center)", fontsize=9)
        ax_m.set_title(title + "    (m-cut, l=center)", fontsize=9)
        ax_l.grid(True)
        ax_m.grid(True)
        ax_l.set_ylim(ylim_casa)
        ax_m.set_ylim(ylim_casa)
        ax_l.set_ylabel("PSF difference")
        if ri == len(pairs) - 1:
            ax_l.set_xlabel("l pixel")
            ax_m.set_xlabel("m pixel")

    fig_casa.suptitle(
        "PSF cross-section differences: casa=True - casa=False "
        "(same weighting and robust)",
        fontsize=12,
    )
    fig_casa.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig_casa.savefig("psf_weighting_casa_difference.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    # generate_data()
    plot_data()
