import make_lightcurves
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as it
import time
import os
from astropy.stats import sigma_clipped_stats
import argparse
import subprocess as subp
import glob


def get_loc_stats(chunk):
    chunk = chunk[chunk > 0]
    med = np.median(chunk)
    std = np.std(chunk)
    return med, std


script_start_time = time.time()

parser = argparse.ArgumentParser(
    description="""
    Generate the coinc fits table
    """
)

parser.add_argument(
    "--orbit",
    type=str,
    help="Name of the orbit for which you want to process",
)

parser.add_argument(
    "--tbin",
    type=str,
    help="Bin size that you want to process",
)

args = parser.parse_args()
orbit = args.orbit
tbin = float(args.tbin)
window = 100 / tbin  # The window that is used for finding the statistics
directory = "/mnt/nas2_czti/czti/local_level2"
out_dir = "/home/czti/user_area/ashwin/sem_5_content/tables"

# if len(glob.glob(f"{out_dir}/{orbit}*{tbin}*V2.fits")) > 0:
#     print(f"\nTable for {orbit} at {tbin} is already processed, skipping it.")
#     raise SystemExit

orbitinfo = pd.read_csv("orbitinfo.csv")
obsname = orbitinfo["obsname"]
orbitsdqr = []
obs = []
for obsn in obsname:
    spl = obsn.split("_")
    if len(spl) > 5:
        obs.append(obsn.split("_level2")[0])
        orbitsdqr.append(spl[-1])

orbitsdqr = np.array(orbitsdqr)
print(orbitsdqr[0])
print(obs[0])

# exit()

orbit_count = 0
total_computation_time = 0

orbit_start_time = time.time()
print("orbit:", orbit)
# path_internal = path + orbit + "/"
try:
    k = np.where(orbitsdqr == f"{orbit}")[0][0]
except IndexError:
    print("File not found")
    raise SystemExit
path_internal = (
    f"/mnt/nas2_czti/czti/local_level2/{obs[k]}_level2/czti/orbit/{orbit}_V1.0/"
)
lc_all = []
times_all = []
tot_stats = []

for quad in np.arange(0, 4):
    try:
        light_curve_file = path_internal + f"{orbit}_V1.0_{tbin}_3_Q{quad}.lc"
        print("Light Curve File -", light_curve_file)
        subp.call(
            f"rsync -vzrutP {light_curve_file} /home/czti/user_area/ashwin/sem_5_content/lightcurves",
            shell=True,
        )
        print(
            f"Rsync done, starting cleaning amd stats calculation now (in a window of {window} bins)"
        )

        lc_file = (
            f"/home/czti/user_area/ashwin/sem_5_content/lightcurves/{orbit}_V1.0_{tbin}_3_Q{quad}.lc"
        )
        lc_data_q = fits.getdata(lc_file, 1)

        rate = lc_data_q["RATE"]
        times = lc_data_q["TIME"]
        mask = np.where(times > 0)[0]
        times = times[mask]
        rate = rate[mask]
        mask = np.where(rate > 0)[0]
        times = times[mask]
        rate = rate[mask]

        stats = []
        plt.plot(times,rate)
        plt.savefig(f"plots/{orbit}_{tbin}_{quad}.png")
        plt.clf()
        for i, counts in enumerate(rate):
            if i - int(window / 2) < 0:
                k = 0
            else:
                k = i - int(window / 2)
            if i + int(window / 2) > len(rate):
                j = len(rate)
            else:
                j = i + int(window / 2)
            med, std = get_loc_stats(rate[k:j])
            stats.append(np.array([med, std]))
        stats = np.array(stats, dtype=float)
        tot_stats.append(stats)
        lc_all.append(rate)
        times_all.append(times)
    except FileNotFoundError:
        print("File not Found")
    print(f"Quad {quad} done!")

tot_stats = np.array(tot_stats, dtype=object)
times_all = np.array(times_all, dtype=object)

coinc_arr = np.arange(1, 5)
sigma_arr = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
coinc_sigma_table = np.zeros((len(sigma_arr), 15))
arr_len = max(len(lc_all[0]), len(lc_all[1]), len(lc_all[2]), len(lc_all[3]))
cutoffs = np.zeros((4, len(sigma_arr), arr_len))

computation_start_time = time.time()

print(f"Finding all {len(sigma_arr)} number of sigmas stats now")
for quadno in range(4):
    for i, sigma_id in enumerate(sigma_arr):
        for j, arr in enumerate(tot_stats[quadno]):
            med, std = arr
            cutrate = med + sigma_id * std
            cutoffs[quadno][i][j] = cutrate

quad_ind_max = np.argmax(
    [len(lc_all[0]), len(lc_all[1]), len(lc_all[2]), len(lc_all[3])]
)
max_time_lc = times_all[quad_ind_max]
peak_map = np.zeros((4, len(sigma_arr), arr_len))

print(f"Finding the outliers")
for i, t in enumerate(max_time_lc):
    for quad in range(4):
        for s, sigma_id in enumerate(sigma_arr):
            tind = np.where(times_all[quad] == t)[0]
            if lc_all[quad][tind] >= cutoffs[quad][s][tind]:
                peak_map[quad][s][i] = +1


print(f"Finding the different combinations of outliers")
for coinc in coinc_arr:
    for sid in range(len(sigma_arr)):
        if coinc == 1:
            for n in range(4):
                peak_ind_bin = np.where(peak_map[n][sid] == coinc)
                coinc_sigma_table[sid, n] = len(peak_ind_bin[0])

        elif coinc == 2:
            arr = pd.Series(list(it.combinations(np.unique(range(4)), coinc)))
            for m in range(len(arr)):
                k = arr[m]
                peak_ind_bin = np.where(
                    peak_map[k[0]][sid] + peak_map[k[1]][sid] == coinc
                )
                coinc_sigma_table[sid, n + 1 + m] = len(peak_ind_bin[0])

        elif coinc == 3:
            arr = pd.Series(list(it.combinations(np.unique(range(4)), coinc)))
            for j in range(len(arr)):
                k = arr[j]
                peak_ind_bin = np.where(
                    peak_map[k[0]][sid] + peak_map[k[1]][sid] + peak_map[k[2]][sid]
                    == coinc
                )
                coinc_sigma_table[sid, n + 2 + m + j] = len(peak_ind_bin[0])

        elif coinc == 4:
            peak_ind_bin = np.where(
                peak_map[0][sid]
                + peak_map[1][sid]
                + peak_map[2][sid]
                + peak_map[3][sid]
                == coinc
            )
            coinc_sigma_table[sid, -1] = len(peak_ind_bin[0])

print(coinc_sigma_table.transpose())
col_arr = []
t = 0
for sigma in sigma_arr:
    col_arr.append(fits.Column(name=str(sigma), format="J", array=coinc_sigma_table[t]))
    t += 1

hdu = fits.BinTableHDU.from_columns(col_arr)
hdu.writeto(
    f"{out_dir}/{orbit}_V1.0_{tbin}_3_CoincSigmaTable_V2.fits",
    overwrite=True,
)
print(f"File saved - {out_dir}/{orbit}_V1.0_{tbin}_3_CoincSigmaTable_V2.fits")
print("time on orbit:", time.time() - orbit_start_time)
computation_time = time.time() - computation_start_time
print("time_for_my_computation:", computation_time)
total_computation_time += computation_time

total_time = time.time() - script_start_time
print("Total time -", total_computation_time)
