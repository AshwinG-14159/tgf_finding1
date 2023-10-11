import csv
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

import time
from scipy.ndimage import label








t_script_start=time.time()

set_of_rows_occ = []
set_of_rows_polar = []
# names_of_regions=['occ', 'occ_free', 'free', 'free_occ']
binning = 0.001
gap = 8
total_attempts = 100
version = 1




with open('../data/orbitinfo.csv', 'r') as f:
    r = csv.reader(f)
    count=0
    for row in r:

        if(count<1):
            count+=1
            continue

        if(float(row[5])<-65 or float(row[5])>65) and row[0][-6:]!='level2':
            if(count%gap!=0):
                count+=1
                continue
            set_of_rows_polar.append(row)
            count+=1
            if(count/gap>total_attempts): # leave 10 between any 2 and take a total of 200. Binning = 0.001
                break
        # if(float(row[5])<-60 or float(row[5])>60) and row[0][-6:]!='level2':
        #     set_of_rows_polar.append(row)



print("file reading time:",time.time()-t_script_start)
print("number of orbits to work with: ", len(set_of_rows_polar))

path = "/mnt/nas2_czti/czti/"
path_tables = "../data/tables2"

t_comp_start = time.time()

Coinc_Tables = 0
# Analysed_Durations = 0

skip_till = ['20230329_A12_054T02_9000005550_level2_40553', 'A12_054T02_9000005550', 'priyanka_iucaa', 'SBS 0846+513', '132.4916', '51.1414', '5100.90680462', '2023-03-29T16:16:53', '2023-03-29T18:16:54']
skipped = 1


err_counts = {"mkf file error":0, 
"coordinates error":0, 
"evt file header messup": 0, 
"evt file does not open":0,
"histograms not present":0,
"unable to make hists by region": 0}


orbit_num=0
for row in set_of_rows_polar:
    orbit_num+=1
    if(row!=skip_till and not skipped):
        continue
    else:
        if(row==skip_till):
            skipped = 1
    t_row_start = time.time()
    print("orbit num: ", orbit_num)
    print('writing data to: ', f"{path_tables}/{binning}/{row[0][:-6]}_{row[0][-5:]}_V1.0_{binning}_3_CoincSigmaTable_V2_polar_ver{version}.fits")
    exit(0)
    print('trying to access row:', row)
    target_folder = path+f'level2/{row[0][:-6]}/czti/orbit/{row[0][-5:]}_V1.0'
    mkf_file = target_folder+f'/AS1{row[0][9:-13]}_{row[0][-5:]}czt_level2.mkf'
    evt_file = target_folder+f'/modeM0/AS1{row[0][9:-13]}_{row[0][-5:]}cztM0_level2_bc.evt'
    


    t_row_sep_ready = time.time()
    print("time to get sep array: ", t_row_sep_ready-t_row_start)


    time_stamps = []
    earliest = []
    latest = []
    error_check = 0

    try:
        with fits.open(evt_file) as hdul:
            for quarter in range(1,5):
                try:
                    q_data = hdul[quarter].data['Time']
                    start = hdul[quarter].header['TSTARTI']
                    end = hdul[quarter].header['TSTOPI']
                except:
                    err_counts["evt file header messup"]+=1
                    print('orbit data not extractable')
                    error_check = 1
                    break
                q_data = q_data[(q_data>start) & (q_data<end)]
                time_stamps.append(q_data)
                earliest.append(start)
                latest.append(end)
    except:
        err_counts["evt file does not open"]+=1
        print("Evt files not found")
        continue
    
    t_row_data_ready = time.time()
    print("time to get data: ", t_row_data_ready-t_row_sep_ready)

    earliest = max(earliest)
    latest = min(latest)
    bins = np.linspace(earliest, latest, int((latest-earliest)/binning))
    try:
        hist1, _ = np.histogram(time_stamps[0], bins)
        print('hist1')
        hist2, _ = np.histogram(time_stamps[1], bins)
        print('hist2')
        hist3, _ = np.histogram(time_stamps[2], bins)
        print('hist3')
        hist4, _ = np.histogram(time_stamps[3], bins)
        print('hist4')
        hist = np.vstack((hist1, hist2, hist3, hist4))
        bins = bins[:-1]
        print(hist.shape)
    except:
        print("4 histograms not present here")
        err_counts["histograms not present"]+=1
        continue


    

    t_row_hist_ready = time.time()
    print("time to get histograms: ", t_row_hist_ready-t_row_data_ready)

    k_net = hist
    if(k_net.shape[1]==0):
        continue
    cutoffs = [5,6,7,8,9,10]
    coinc_arr = [1,2,3,4]
    coinc_sigma_table =  np.zeros((len(cutoffs), len(coinc_arr)))
        
        #k_net is net array.. coinc_sigma_table is table to store number of occurences for all combinations.
    
    for coinc in range(len(coinc_arr)):
        peak_ind_bin = np.where(k_net[0] - k_net[0] == 0)
        complete_arr2 = k_net[:,peak_ind_bin[0]]
        for cutoff_set in range(len(cutoffs)):
            peak_map = np.zeros((4,len(peak_ind_bin[0])))
            complete_arr2 = complete_arr2[:,peak_ind_bin[0]]
            if len(complete_arr2[0,:])==0:      
                break
            for i in range(4):
                peak_map[i][complete_arr2[i] > cutoffs[cutoff_set]] = 1
                
            peak_ind_bin =  np.where(np.sum(peak_map, axis=0) >= coinc_arr[coinc])
            coinc_sigma_table[cutoff_set, coinc] =  len(peak_ind_bin[0])

    if(type(Coinc_Tables)==int):
        Coinc_Tables=coinc_sigma_table
    else:
        Coinc_Tables+=coinc_sigma_table


    col_arr = []
    t=0
    for cutoff in cutoffs:
        col_arr.append(fits.Column(name=str(cutoff), format="J", array=coinc_sigma_table[t]))
        t+=1
    header = fits.Header()
    header['DURATION'] = k_net.shape[1]
    hdu = fits.BinTableHDU.from_columns(col_arr, header=header)
    hdu.writeto(
        f"{path_tables}/{binning}/{row[0][:-6]}_{row[0][-5:]}_V1.0_{binning}_3_CoincSigmaTable_V2_polar_ver{version}.fits",
        overwrite=True,
    )
    # print(names_of_regions[mark], Coinc_Tables[mark]/Analysed_Durations[mark])
    print('writing data to: ', f"{path_tables}/{binning}/{row[0][:-6]}_{row[0][-5:]}_V1.0_{binning}_3_CoincSigmaTable_V2_polar_ver{version}.fits")


    t_row_complete = time.time()

    print("time to get tables: ", t_row_complete-t_row_hist_ready)

print(err_counts)
print("total time to run code:", time.time() - t_script_start)




