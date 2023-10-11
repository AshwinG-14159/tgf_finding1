import csv
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import astropy.units as u
from astropy.coordinates import SkyCoord

import time
from scipy.ndimage import label


# def debug(context, data):
#     f = open(f'Debug/{context}', 'a')
#     f.write(data)
#     f.close()


t_script_start=time.time()

set_of_rows_occ = []
set_of_rows_polar = []
names_of_regions=['occ', 'occ_free', 'free', 'free_occ']
gap = 30
total_attempts = 400
version=3

gap2 = 26
attempts2 = 400
version2 = 3

plot_version = 7


with open('../data/orbitinfo.csv', 'r') as f:
    r = csv.reader(f)
    count=0
    for row in r:

        if(count<1):
            count+=1
            continue
        
        if(float(row[5])>-50 and float(row[5])<50) and row[0][-6:]!='level2':
            if(count%gap!=0):
                count+=1
                continue
            set_of_rows_occ.append(row)
            count+=1
            if(count/gap>total_attempts):
                break
        # if(float(row[5])<-60 or float(row[5])>60) and row[0][-6:]!='level2':
        #     set_of_rows_polar.append(row)
    else:
        print("file ended")
# exit(0)
print("file reading time:",time.time()-t_script_start)
print("number of orbits to work with: ", len(set_of_rows_occ))


names_of_regions=['occ', 'occ_free', 'free', 'free_occ']

binning = 0.001
mark=0
path_to_tables = "../data/tables2"
path_to_plots = "../data/final_plots2"
total_duration_arr = [0,0,0,0]
total_table = [0,0,0,0]

for row in set_of_rows_occ:
    # print('row: ', row)
    # print('name: ', row[0])
    found = 0
    for k_index in range(4):
        table = f"{path_to_tables}/{binning}/{row[0][:-6]}_{row[0][-5:]}_V1.0_{binning}_3_CoincSigmaTable_V2_{names_of_regions[k_index]}_ver{version}.fits"
        # print('table to find: ', table)
        try:
            with fits.open(table) as hdul:
                table=Table(hdul[1].data)
                duration = hdul[1].header['DURATION']
        except:
            continue

        if(type(total_table[k_index])==int):
            total_table[k_index]=table
        else:
            for col in total_table[k_index].colnames:
                total_table[k_index][col]+= table[col]


        total_duration_arr[k_index]+=duration
        found=1

    
    if(found):
        mark+=1
    else:
        continue


print(f'count of orbits seen: {mark}')

x_vals = [1,2,3,4,5]

frequencies = []

for k_index in range(4):
    new_arr = np.zeros((4,6)) #number of quadrants x threshold
    for col in total_table[k_index].colnames:
        for element in range(len(total_table[k_index][col])):
            new_arr[element][(int(col)-5)] = total_table[k_index][col][element]/total_duration_arr[k_index]
    frequencies.append(new_arr)

# frequencies = np.array(frequencies)

# print(frequencies)
print("total duration arr: ", total_duration_arr)





# exit()



with open('../data/orbitinfo.csv', 'r') as f:
    r = csv.reader(f)
    count=0
    for row in r:

        if(count<1):
            count+=1
            continue

        if(float(row[5])<-65 or float(row[5])>65) and row[0][-6:]!='level2':
            if(count%gap2!=0):
                count+=1
                continue
            set_of_rows_polar.append(row)
            count+=1
            if(count/gap2>attempts2): # leave 10 between any 2 and take a total of 200. Binning = 0.001
                break
    else:
        print('csv ended')

print('num of polar attempts: ', len(set_of_rows_polar))
total_duration_arr = 0
total_table = 0
mark2=0
for row in set_of_rows_polar:
    found = 0
    table = f"{path_to_tables}/{binning}/{row[0][:-6]}_{row[0][-5:]}_V1.0_{binning}_3_CoincSigmaTable_V2_polar_ver{version2}.fits"

    try:
        with fits.open(table) as hdul:
            table=Table(hdul[1].data)
            duration = hdul[1].header['DURATION']
    except:
        # print(table, "not found")
        continue

    if(type(total_table)==int):
        total_table=table
    else:
        for col in total_table.colnames:
            total_table[col]+= table[col]
    found = 1
    if(found):
        total_duration_arr+=duration
        mark2+=1
print(f'count of orbits seen: {mark2}')
print(f'total duration of polar: {total_duration_arr}')

new_arr = np.zeros((4,6)) #number of quadrants x threshold
for col in total_table.colnames:
    for element in range(len(total_table[col])):
        new_arr[element][(int(col)-5)] = total_table[col][element]/total_duration_arr


frequencies.append(new_arr)


frequencies = np.array(frequencies)

names_of_regions.append("polar")


plt.figure(figsize=(8,12))

for quads in range(len(frequencies[0])):
    for threshold in range(len(frequencies[0,0])):
        plt.plot(x_vals, frequencies[:,quads, threshold], label = str(threshold+5))
    plt.title(f'Variation of frequency, Num_orbits = {mark},{mark2}, binning = {binning}, quads = {quads+1}')
    plt.xticks(x_vals, names_of_regions)
    plt.yscale('log')
    plt.xlabel("region")
    plt.ylabel("frequency of detections")
    plt.legend()
    plt.savefig(f'{path_to_plots}/variation_{quads+1}_ver{plot_version}.png')
    plt.cla()





exit(0)
