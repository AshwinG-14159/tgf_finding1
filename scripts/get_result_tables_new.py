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

with open('../data/orbitinfo.csv', 'r') as f:
    r = csv.reader(f)
    count=0
    for row in r:

        if(count<1):
            count+=1
            continue
        
        if(float(row[5])>-50 and float(row[5])<50) and row[0][-6:]!='level2':
            if(count%5!=0):
                count+=1
                continue
            set_of_rows_occ.append(row)
            count+=1
            if(count/5>100):
                break
        # if(float(row[5])<-60 or float(row[5])>60) and row[0][-6:]!='level2':
        #     set_of_rows_polar.append(row)


print("file reading time:",time.time()-t_script_start)
print("number of orbits to work with: ", len(set_of_rows_occ))


names_of_regions=['occ', 'occ_free', 'free', 'free_occ']

binning = 0.001
mark=0
path_to_tables = "../data/tables"
path_to_plots = "../data/final_plots"
total_duration_arr = [0,0,0,0]
total_table = [0,0,0,0]

for row in set_of_rows_occ:
    print('row: ', row)
    print('name: ', row[0])
    found = 0
    for k_index in range(4):
        table = f"{path_to_tables}/{binning}/{row[0][:-6]}_{row[0][-5:]}_V1.0_{binning}_3_CoincSigmaTable_V2_{names_of_regions[k_index]}_ver1.fits"
        print('table to find: ', table)
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

x_vals = [1,2,3,4]

frequencies = []

for k_index in range(4):
    new_arr = np.zeros((4,6)) #number of quadrants x threshold
    for col in total_table[k_index].colnames:
        for element in range(len(total_table[k_index][col])):
            new_arr[element][(int(col)-5)] = total_table[k_index][col][element]/total_duration_arr[k_index]
    frequencies.append(new_arr)

frequencies = np.array(frequencies)

print(frequencies)
print("total duration arr: ", total_duration_arr)



plt.figure(figsize=(8,12))

for quads in range(len(frequencies[0])):
    for threshold in range(len(frequencies[0,0])):
        plt.plot(x_vals, frequencies[:,quads, threshold], label = str(threshold+5))
    plt.title(f'Variation of frequency, Num_orbits = {mark}, binning = {binning}, quads = {quads+1}')
    plt.xticks(x_vals, names_of_regions)
    plt.yscale('log')
    plt.xlabel("region")
    plt.ylabel("frequency of detections")
    plt.legend()
    plt.savefig(f'{path_to_plots}/variation_{quads+1}_ver1.png')
    plt.cla()


exit()


f_timing = open('TGFs_Analysed_durations_polar.txt', 'r')
data = f_timing.read().split('\n\n')
# print(data)
f_timing.close()

mark = 0

total_duration_arr = 0
total_table = 0

for row in set_of_rows_polar:
    # print('row: ', row)
    # print('name: ', row[0])
    found = 0
    # for k_index in range(4):
    table = f"tables/0.001/{row[0][:-6]}_{row[0][-5:]}_V1.0_0.001_3_CoincSigmaTable_V2_polar.fits"
    # print('table to find: ', table)
    try:
        with fits.open(table) as hdul:
            table=Table(hdul[1].data)
    except:
        # print('table not found')
        continue

    # print(table)
    if(type(total_table)==int):
        total_table=table
    else:
        for col in total_table.colnames:
            total_table[col]+= table[col]
    # print(table)
    found = 1
    
    duration = data[mark].split('\n')
    while(len(duration)==1):
        mark+=1
        duration = data[mark].split('\n')
    print('file name in txt: ', duration[0][34:71], duration[0][83:88])
    if(duration[0][34:71]+'_'+ duration[0][83:88] != row[0]):
        print('something is wrong here')
        exit(0)
    if(len(duration)>1):
        d_val = int(duration[1][duration[1].find(':')+1:-1])
        total_duration_arr+=d_val
    else:
        print('number of durations here: ', len(duration), duration)
        print('duration: ', d_val)
    
    if(found):
        mark+=1
    else:
        # print("This row's data was never saved")
        continue
    # if(mark>10):
    #     break

print(f'count of orbits seen: {mark-1}')


print('total duration: ',total_duration_arr)


# for k_index in range(4):
new_arr = np.zeros((4,6)) #number of quadrants x threshold
for col in total_table.colnames:
    # total_table[k_index][col] = total_table[k_index][col].astype(float)
    for element in range(len(total_table[col])):
        new_arr[element][(int(col)-5)] = total_table[col][element]/total_duration_arr
# frequencies.append(new_arr)
    # total_table[k_index][col]/=int(total_duration_arr[k_index])
# frequencies = np.array()
print(new_arr)


frequencies.append(new_arr)


frequencies = np.array(frequencies)
print('fff', frequencies)



names_of_regions.append("polar")

plt.figure(figsize=(8,12))

for quads in range(len(frequencies[0])):
    for threshold in range(len(frequencies[0,0])):
        plt.plot(x_vals, frequencies[:,quads, threshold], label = str(threshold+5))
        plt.title(f'quads = {quads+1}')
        plt.xticks(x_vals, names_of_regions)
        plt.yscale('log')
    plt.legend()
    plt.savefig(f'{path_to_plots}/variation_{quads+1}_ver1.png')
    plt.cla()




# print(names_of_regions[k_index], total_table[k_index], total_duration_arr[k_index])




