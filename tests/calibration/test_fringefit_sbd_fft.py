from xradio.measurement_set.open_processing_set import open_processing_set
from astroviper.calibration.fringefit import fringefit_ps, apply_cal_ps
from xradio.measurement_set import convert_msv2_to_processing_set
import toolviper

import xarray as xa
import astropy.time
import numpy as np
import matplotlib.pyplot as plt

msv2_name = "global_vlbi_gg084b_reduced.ms"
toolviper.utils.data.download(file=msv2_name)

convert_out = "gg084b"
partition_scheme = ["FIELD_ID", "SCAN_NUMBER", "SPW"]

convert_msv2_to_processing_set(
    in_file=msv2_name,
    out_file=convert_out,
    partition_scheme=partition_scheme,
    persistence_mode="w",
    parallel_mode="partition",
)

# This currently fails with a very long error message ending with
# "TypeError: Expected a BytesBytesCodec. Got <class 'numcodecs.zstd.Zstd'> instead."

# The conversion appends ".ps.zarr" to the out_file it was given,
# But currently the conversion doesn't output a valid .zarr dataset anyway

ps = open_processing_set("gg084b.ps.zarr")

s = ps.xr_ps.summary()  # => Pandas table

ff_start = astropy.time.Time("2018-05-27 07:15:00", format="iso").unix
ff_interval = 90

ref_ant = "MK"

cal_tree = fringefit_ps(ps, ref_ant, ff_start, ff_interval)
ps2 = apply_cal_ps(ps, cal_tree, ff_start, ff_interval)


def bl_inds(xds, a1, a2):
    return (
        (xds.baseline_antenna1_name == a1) & (xds.baseline_antenna2_name == a2)
        | (xds.baseline_antenna1_name == a2) & (xds.baseline_antenna2_name == a1)
    ).compute()


# We need an xds to get baselines from; I happen to know this one contains the interval we chose above
xds = ps2["global_vlbi_gg084b_reduced_8"]

ref_ant = "MK"
target_ant = "YY"
[fdmk_index] = xds.baseline_id[bl_inds(xds, ref_ant, target_ant)].values
ant1 = xds.baseline_antenna1_name[fdmk_index].values
ant2 = xds.baseline_antenna2_name[fdmk_index].values
vis0 = np.angle(np.mean(xds["VISIBILITY"][:, fdmk_index, :, 0].squeeze(), axis=0))
vis1 = np.angle(
    np.mean(xds["VISIBILITY_CORRECTED"][:, fdmk_index, :, 0].squeeze(), axis=0)
)
plt.figure()
plt.scatter(xds.frequency, vis0, label="Raw")
plt.scatter(xds.frequency, vis1, label="Corrected")
plt.title(f"Baseline {ant1}-{ant2}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.legend()
plt.savefig("fringe.png")
