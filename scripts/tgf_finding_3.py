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


def debug(context, data):
    f = open(f'Debug/{context}', 'a')
    f.write(data)
    f.close()


t_script_start=time.time()

set_of_rows_occ = []
set_of_rows_polar = []
names_of_regions=['occ', 'occ_free', 'free', 'free_occ']
binning = 0.1


with open('orbitinfo.csv', 'r') as f:
    r = csv.reader(f)
    t=0
    for row in r:
        if(t<1):
            t+=1
            continue
        # print(float(row[5]))
        if(float(row[5])>-65 and float(row[5])<65) and row[0][-6:]!='level2':
            set_of_rows_occ.append(row)
        # if(float(row[5])<-60 or float(row[5])>60) and row[0][-6:]!='level2':
        #     set_of_rows_polar.append(row)
        if(t>1500):
            break
        t+=1


print("file reading time:",time.time()-t_script_start)
print("number of orbits to work with: ", len(set_of_rows_occ))

path = "/mnt/nas2_czti/czti/"

f = open('TGFs_Analysed_durations_ver4.txt','w' )

t_comp_start = time.time()

Coinc_Tables = []
Analysed_Durations = []

skip_till = ['20230329_A12_054T02_9000005550_level2_40553', 'A12_054T02_9000005550', 'priyanka_iucaa', 'SBS 0846+513', '132.4916', '51.1414', '5100.90680462', '2023-03-29T16:16:53', '2023-03-29T18:16:54']
skipped = 1

for row in set_of_rows_occ:
    if(row!=skip_till and not skipped):
        continue
    else:
        if(row==skip_till):
            skipped = 1
    t_row_start = time.time()

    # row = ['20230707_A12_037T01_9000005736_level2_42028', 'A12_037T01_9000005736', 'sreeja', 'VCC43', '183.0228', '13.20564', '814.751390755', '2023-07-07T06:50:08', '2023-07-07T07:31:34']

    print('trying to access row:', row)
    target_folder = path+f'level2/{row[0][:-6]}/czti/orbit/{row[0][-5:]}_V1.0'
    mkf_file = target_folder+f'/AS1{row[0][9:-13]}_{row[0][-5:]}czt_level2.mkf'
    evt_file = target_folder+f'/modeM0/AS1{row[0][9:-13]}_{row[0][-5:]}cztM0_level2_bc.evt'
    try:
        tab = Table.read(mkf_file,1)
    except:
        continue
    x_arr,y_arr,z_arr = np.array(tab['POSX']),np.array(tab['POSY']),np.array(tab['POSZ'])
    time_arr_og = tab['TIME']




#trying to mask the values
    mask_x = x_arr!=0
    mask_y = y_arr!=0
    mask_z = z_arr!=0
    mask_time = time_arr_og>0
    mask_coord = mask_x & mask_y & mask_z & mask_time #use this mask later




    r_arr = np.sqrt(np.square(x_arr)+np.square(y_arr)+np.square(z_arr))

    dec_arr = -(90-np.degrees(np.arccos(z_arr/r_arr)))
    RA_arr = -(np.degrees(np.arctan2(y_arr,x_arr)))
    with fits.open(mkf_file) as hdul: #get the pointing directions
        RA_pnt = hdul[0].header['RA_PNT']
        dec_pnt = hdul[0].header['DEC_PNT']
    try:
        pointing = SkyCoord(RA_pnt*u.deg, dec_pnt*u.deg, frame='icrs')
        Earth_center = SkyCoord(RA_arr*u.deg, dec_arr*u.deg, frame='icrs')
        sep_arr = Earth_center.separation(pointing).degree
    except:
        print('Something wrong with the coordinates') #trying to access row: ['20230329_A12_054T02_9000005550_level2_40553', 'A12_054T02_9000005550', 'priyanka_iucaa', 'SBS 0846+513', '132.4916', '51.1414', '5100.90680462', '2023-03-29T16:16:53', '2023-03-29T18:16:54']
        continue



    t_row_sep_ready = time.time()
    print("time to get sep array: ", t_row_sep_ready-t_row_start)

    regs_across_quarters = []

    time_stamps = []
    earliest = []
    latest = []
    error_check = 0

    with fits.open(evt_file) as hdul:
        for quarter in range(1,5):
            # print('file: ', evt_file)
            try:
                q_data = hdul[quarter].data['Time']
                start = hdul[quarter].header['TSTARTI']
                end = hdul[quarter].header['TSTOPI']
            except:
                print('orbit data not extractable')
                error_check = 1
                break
            # print(f"quadrant:{quarter}. Len before = {len(q_data)}. ", end='')
            q_data = q_data[(q_data>start) & (q_data<end)]
            # print(f"Len after = {len(q_data)}")
            time_stamps.append(q_data)
            earliest.append(start)
            latest.append(end)
            # num_bins = int((latest[-1]-earliest[-1])/binning)
            # # print(f"Earliest: {earliest[-1]}, Latest: {latest[-1]}, First element: {q_data[0]}, Last element: {q_data[-1]}")


            # hist, bins = np.histogram(q_data, num_bins)
            # bins = bins[:-1]



            # counts.append(hist)
            # times.append(bins)


            match_with_time_cond = (time_arr_og>=start) & (time_arr_og<=end) & mask_coord

            time_arr_here = time_arr_og[match_with_time_cond]
            sep_arr_here = sep_arr[match_with_time_cond]
            # plt.plot(time_arr_here, sep_arr_here)
            # plt.savefig(f"plots/{row[0]}_{quarter}_separation.png")
            # plt.cla()


            labels, num_features = label(sep_arr_here>105)
            free = [np.where(labels == label_id)[0] for label_id in range(1, num_features + 1)]

            labels, num_features = label(sep_arr_here<75)
            occ = [np.where(labels == label_id)[0] for label_id in range(1, num_features + 1)]

            if(len(occ)==0 and len(free)==0):
                print('This orbit has no free or occ regions')
                continue


            # 4 regions named as occ, occ_free, free and free_occ
            reg_occ = []
            reg_occ_free = []
            reg_free = []
            reg_free_occ = []

            mark1=mark2=0
            # print('occ[0][0]:', occ[0][0])
            # print('free[0][0]:', free[0][0])
            # print(occ)
            # print(free)
            if(len(occ)>0 and len(free)>0):
                if(0<occ[0][0] and 0 < free[0][0]):
                    if(occ[0][0]<free[0][0]):
                        reg_free_occ.append([start,start+occ[0][0]-1])
                    else:
                        reg_occ_free.append([start,start+free[0][0]-1])
                
                while(True):
                    if(mark1>=len(occ) and mark2<len(free)):
                        reg_free.append([start+free[mark2][0],start+free[mark2][-1]])
                        if(start+free[mark2][-1]+1!= end):
                            reg_free_occ.append([start+free[mark2][-1]+1, end])
                        mark2+=1
                        continue

                    if(mark2>=len(free) and mark1<len(occ)):
                        reg_occ.append([start+occ[mark1][0],start+occ[mark1][-1]])
                        if(start+occ[mark1][-1]+1!= end):
                            reg_occ_free.append([start+occ[mark1][-1]+1, end])
                        mark1+=1
                        continue
                    if(mark1>=len(occ) and mark2>=len(free)):
                        break
                    

                    if(occ[mark1][0]<free[mark2][0]):
                        reg_occ.append([start+occ[mark1][0],start+occ[mark1][-1]])
                        reg_occ_free.append([start+occ[mark1][-1]+1, start+free[mark2][0]-1])
                        mark1+=1
                    else:
                        reg_free.append([start+free[mark2][0],start+free[mark2][-1]])
                        reg_free_occ.append([start+free[mark2][-1]+1,start+occ[mark1][0]-1])   
                        mark2+=1

            elif(len(free)>0):
                if(free[0][0]>0):
                    reg_occ_free.append([start, start+free[0][0]-1])
                reg_free.append([start+free[0][0], start+free[0][-1]])
                if(start+free[0][-1]+1 != end):
                    reg_free_occ.append([start+free[0][-1]+1, end])
            elif(len(occ)>0):
                if(occ[0][0]>0):
                    reg_free_occ.append([start, start+occ[0][0]-1])
                reg_occ.append([start+occ[0][0], start+occ[0][-1]])
                if(start+occ[0][-1]+1 != end):
                    reg_occ_free.append([start+occ[0][-1]+1, end])


            regions = [reg_occ, reg_occ_free, reg_free, reg_free_occ]

            debug(f"{row[0]}_{quarter}_debug_ver4.txt", f"{names_of_regions[0]}:{regions[0]}, {names_of_regions[1]}:{regions[1]}, {names_of_regions[2]}:{regions[2]}, {names_of_regions[3]}:{regions[3]}\n")
            # print(regions)
            # print('iiiiiiiiiiiiiiiiiii')
            if(len(regs_across_quarters)==0):
                regs_across_quarters = regions
                continue
        
            for reg_id in range(len(regs_across_quarters)):
                try:
                    for reg_sec_id in range(len(regs_across_quarters[reg_id])):
                        regs_across_quarters[reg_id][reg_sec_id][0] = max(regs_across_quarters[reg_id][reg_sec_id][0], regions[reg_id][reg_sec_id][0])
                        regs_across_quarters[reg_id][reg_sec_id][1] = min(regs_across_quarters[reg_id][reg_sec_id][1], regions[reg_id][reg_sec_id][1])
                except:
                    # print("ERROR!")
                    # print(regs_across_quarters[reg_id])
                    # print(regions[reg_id])
                    regs_across_quarters[reg_id] = regs_across_quarters[reg_id][:-1]

            # print('-------------')
            # print(regs_across_quarters)
            
    if(error_check):
        continue

    t_row_data_ready = time.time()
    print("time to get data and regions: ", t_row_data_ready-t_row_sep_ready)


    earliest = max(earliest)
    latest = min(latest)
    bins = np.linspace(earliest, latest, int((latest-earliest)/binning))
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

    hists_by_reg = []
    for reg in regs_across_quarters:

        for i in range(len(reg)):
            if(i==0):
                condition = (bins>=reg[i][0]) & (bins<=reg[i][1])
            else:
                condition = condition | ((bins>=reg[i][0]) & (bins<=reg[i][1]))
        try:
            hists_by_reg.append(hist[:, np.where(condition)[0]])
        except:
            continue
    
    f.write(f'orbit: {evt_file[:-4]}\n')
    for i in range(len(hists_by_reg)):
        if(len(Analysed_Durations)<4):
            Analysed_Durations.append(hists_by_reg[i].shape[1])
        else:
            Analysed_Durations[i] += hists_by_reg[i].shape[1]
        
        f.write(f'{names_of_regions[i]}, Analysed duration: {hists_by_reg[i].shape[1]};\n')

    mark = 0


    t_row_hist_ready = time.time()
    print("time to get histograms: ", t_row_hist_ready-t_row_data_ready)

    for k_index in range(len(hists_by_reg)):
        k_net = hists_by_reg[k_index]
        if(k_net.shape[1]==0):
            continue
        cutoffs = [90,100,110,120,130,140]
        coinc_arr = [1,2,3,4]
        coinc_sigma_table =  np.zeros((len(cutoffs), len(coinc_arr)))
        
        #k_net is net array.. coinc_sigma_table is table to store number of occurences for all combinations.
        
        for coinc in range(len(coinc_arr)):
            peak_ind_bin = np.where(k_net[0] - k_net[0] == 0)
            complete_arr2 = k_net[:,peak_ind_bin[0]]
            for cutoff_set in range(len(cutoffs)):
                        # print('peak_ind_bin: ', peak_ind_bin[0][:10])
                peak_map = np.zeros((4,len(peak_ind_bin[0])))
                        # peak_map2 = np.zeros((4,arr_len))
                complete_arr2 = complete_arr2[:,peak_ind_bin[0]]
                if len(complete_arr2[0,:])==0:      
                    break
                        # print(np.shape(complete_arr2))
                        # print(np.shape(complete_arr))
                        # cutoffs2 = cutoffs[:, peak_ind_bin]
                for i in range(4):
                    peak_map[i][complete_arr2[i] > cutoffs[cutoff_set]] = 1
                    
                peak_ind_bin =  np.where(np.sum(peak_map, axis=0) >= coinc_arr[coinc])
                coinc_sigma_table[cutoff_set, coinc] =  len(peak_ind_bin[0])
        # print(coinc_sigma_table)

        # coinc_sigma_table = coinc_sigma_table/(k_net.shape[1])
        # print('[DEBUG]', k_net.shape[1])

        if(len(Coinc_Tables)<4):
            Coinc_Tables.append(coinc_sigma_table)
        else:
            Coinc_Tables[mark]+=coinc_sigma_table

        col_arr = []
        t=0
        for cutoff in cutoffs:
            col_arr.append(fits.Column(name=str(cutoff), format="J", array=coinc_sigma_table[t]))
            t+=1
        hdu = fits.BinTableHDU.from_columns(col_arr)
        hdu.writeto(
            f"tables/{binning}/{row[0][:-6]}_{row[0][-5:]}_V1.0_{binning}_3_CoincSigmaTable_V2_{names_of_regions[k_index]}_ver4.fits",
            overwrite=True,
        )
        print(names_of_regions[mark], Coinc_Tables[mark]*1000/Analysed_Durations[mark])
        # print('writing: ', coinc_sigma_table)
        print('writing data to: ', f"tables/{binning}/{row[0][:-6]}_{row[0][-5:]}_V1.0_{binning}_3_CoincSigmaTable_V2_{names_of_regions[k_index]}_ver4.fits")
        # print(names_of_regions[mark], Coinc_Tables[mark])

        # f.write(f'{names_of_regions[mark]}, {coinc_sigma_table}\n')
        mark+=1


    # print(Coinc_Tables)
    f.write('\n')
    f.flush()
    t_row_complete = time.time()

    print("time to get tables: ", t_row_complete-t_row_hist_ready)


# for i in range(len(Coinc_Tables)):
#     Coinc_Table_rate = Coinc_Tables[i]/Analysed_Durations[i]
#     col_arr = []
#     t=0
#     for cutoff in cutoffs:
#         col_arr.append(fits.Column(name=str(cutoff), format="J", array=Coinc_Table_rate[t]))
#         t+=1
#     hdu = fits.BinTableHDU.from_columns(col_arr)
#     hdu.writeto(
#         f"tables/{binning}/Total_V1.0_{binning}_3_CoincSigmaTable_V2_{names_of_regions[k_index]}.fits",
#         overwrite=True,
#     )

# f.close()






