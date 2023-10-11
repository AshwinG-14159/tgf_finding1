import os
import copy
import subprocess
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.signal import savgol_filter
import datetime
import logging
from astropy.table import Table, Column, vstack
import json

t_buffer = 50.0

def check_and_edit_column(mkffile):
        mkf = fits.open(mkffile)
        hdr = mkf[1].header
        sunelv_bool = False
        for key in hdr.keys():
                if(hdr[key] == 'SUNELV'):
                        sunelv_bool = True
                        break
        if(sunelv_bool == False):
                temppath = os.path.dirname(mkffile) + '/temp.mkf'
                subprocess.call('mv '+mkffile+' '+temppath,shell=True)
                subprocess.call('ftcalc '+temppath+' '+mkffile+' SUNELV 0*TIME',shell=True)
                subprocess.call('rm '+temppath,shell=True)

def gtigen(bcevt, mkffile, outpath, mkf_thresholds):
        """Generate good time intervals.

        Inputs:
        bcevt  - Bunch clean event file
        mkffile - .mkf file of input folder
        mkf_thresholds - File containing the threshold values
        All above files must include the path
        outpath - Folder in which the new files are to be written

        Returns:
        outgti - Path to the output gti file.

        Local variables:
        bcevt_base - Filename excluding path

        """
        bcevt_base = os.path.basename(bcevt)
        # Conventional naming of output gti file
        outgti = outpath + bcevt_base.replace('_bc.evt', '.gti')
        print('\r\n' , '-------------------------------------------')
        print('cztgtigen', 'eventfile=' + bcevt, 'mkffile=' + mkffile,'thresholdfile=' + mkf_thresholds, 'outfile=' + outgti)
        print('-------------------------------------------', '\r\n')
        # Run cztgtigen command in the terminal
        # hdu= fits.open(bcevt)
        # print('hdu',hdu[1].data , hdu[2].data)
        subprocess.call(['cztgtigen', 'eventfile=' + bcevt, 'mkffile=' + mkffile,
                                         'thresholdfile=' + mkf_thresholds, 'outfile=' + outgti,
                                         'usergtifile=-', 'history=y', 'clobber=y'])
        return outgti


def datasel(bcevt, gtifile, outpath):
        """Select data corresponding to GTI's.

        gti_type - QUAD
        Inputs:
        bcevt - Bunch clean event file to be processed
        gtifile - File containg the GTIs for each quadrant
        All the above files must include the path
        outpath - Folder in which the new files are to be written

        Returns:
        quad_dsevt - path to the data selected file

        Local Variables:
        bcevt_base - Filename excluding path

        """
        bcevt_base = os.path.basename(bcevt)
        # Conventional naming of data select file
        quad_dsevt = outpath + bcevt_base.replace('bc', 'quad_bc_ds')
        print('\r\n', '------------------------------------------')
        print('cztdatasel', 'infile=' + bcevt, 'gtifile=' + gtifile,'gtitype=QUAD', 'outfile=' + quad_dsevt)
        print('------------------------------------------','\r\n')
        # Run cztdatasel
        subprocess.call(['cztdatasel', 'infile=' + bcevt, 'gtifile=' + gtifile,
                                         'gtitype=QUAD', 'outfile=' + quad_dsevt, 'clobber=y',
                                         'history=y'])

        return quad_dsevt


def pixclean(quad_dsevt, bclive_tfile, maxpixcount, maxdetcount, outpath):
        """Clean pixels. Run cztpixclean.

        Inputs:
        quad_dsevt - Data-selected file (using gtitype as quad)
        bclive_tfile - Livetime file needed to be processed(doesn't include badpix)
        Abobe files must include the path
        maxpixcount - Threshold pixel count above which it shall be flagged bad
        maxdetcount - Threshold detector count.
        outpath - Folder in which output files must be written

        Returns:
        quad_pcevt - Pixel cleaned file
        quad_dblevt - Double event file
        quad_livet - New livetime file
        quad_badpix - New badpix file

        Local variables:
        livet_base - Livetime fits filename excluding path

        """
        livet_base = os.path.basename(bclive_tfile)
        # Conventional naming
        quad_pcevt = quad_dsevt.replace('quad_bc_ds', 'quad_bc_ds_pc')
        quad_dblevt = quad_dsevt.replace('quad_bc_ds.evt', 'quad.dblevt')
        quad_livet = outpath + livet_base.replace('bc_livetime', 'quad_livetime')
        quad_badpix = outpath + livet_base.replace('bc_livetime', 'quad_pc_badpix')
        print('\r\n','------------------------------------------------')
        print('cztpixclean', 'par_infile=' + quad_dsevt,'par_inlivetimefile=' + bclive_tfile,'par_outfile1=' + quad_pcevt,
                        'par_outfile2=' + quad_dblevt,'par_outlivetimefile=' + quad_livet,'par_badpixfile=' + quad_badpix)
        print('------------------------------------------------','\r\n')
        # Run cztpixclean
        subprocess.call(['cztpixclean', 'par_infile=' + quad_dsevt,
                                         'par_inlivetimefile=' + bclive_tfile,
                                         'par_outfile1=' + quad_pcevt,
                                         'par_outfile2=' + quad_dblevt,
                                         'par_outlivetimefile=' + quad_livet,
                                         'par_badpixfile=' + quad_badpix, 'par_nsigma=5',
                                         'par_det_tbinsize=1', 'par_pix_tbinsize=1',
                                         'par_det_count_thresh=' + str(maxdetcount),
                                         'par_pix_count_thresh=' + str(maxpixcount),
                                         'par_writedblevt=yes'])
        return quad_pcevt, quad_livet, quad_badpix,quad_dblevt


def evtclean(quad_pcevt):
        """Clean alpha tagged events.

        Inputs:
        quad_pcevt - Pic clean file

        Returns:
        quad_cleanevt - Cleaned event file

        """
        quad_cleanevt = quad_pcevt.replace('bc_ds_pc', 'clean')
        print('\r\n','------------------------------------------------')
        print('cztevtclean', 'infile=' + quad_pcevt,'outfile=' + quad_cleanevt)
        print('------------------------------------------------','\r\n')
        # Run cztevtclean
        subprocess.call(['cztevtclean', 'infile=' + quad_pcevt,
                                         'outfile=' + quad_cleanevt, 'alphaval=0',
                                         'vetorange=0', 'clobber=y', 'isdoubleEvent=n',
                                         'history=y'])
        return quad_cleanevt


def flag_bad(quad_badpix):
        """Generate a flagged bad pix file.

        Inputs:
        quad_badpix - Badpix file generated by cztpixclean

        Returns:
        flag_badpix - Flagged badpix file

        """
        flag_badpix = quad_badpix.replace('quad_pc_badpix', 'quad_badpix')
        print('\r\n','-----------------------------------------------')
        print('cztflagbadpix', 'nbadpixFiles=1','badpixfile=' + quad_badpix, 'outfile=' + flag_badpix)
        print('-----------------------------------------------','\r\n')
        # Run cztflagbadpix
        subprocess.call(['cztflagbadpix', 'nbadpixFiles=1',
                                         'badpixfile=' + quad_badpix, 'outfile=' + flag_badpix,
                                         'clobber=y', 'history=y', 'debug=no'])
        return flag_badpix


def bindata(quad_cleanevt, mkffile, flag_badpix, quad_livet, tbin, outbase, band):
        """Bin the event file.

        Inputs:
        quad_cleanevt - Cleaned event file (output of cztevtclean)
        mkffile - .mkf file
        flag_badpix - Flagged bad pixel file (output of cztflagbadpix)
        quad_livet - Livetime file (output of pixclean)
        tbin - Binning time
        band - energy range indicator

        Outputs:
        quad_lc - Path and prefix to the light curve files generated
        quad_weights - Weights applied

        """
        clean_evtbase = os.path.basename(quad_cleanevt)
        cleanevt_path = quad_cleanevt.replace(clean_evtbase, '')
        quad_lc = cleanevt_path + outbase + '_' + str(tbin)+'_'+str(band)
        quad_weights = cleanevt_path + outbase + '_' + str(tbin)+'_'+str(band) + '_weights.evt'
        #Set energy range according to band index
        if(band==0):
                erange="20-50"
        elif(band==1):
                erange="50-100"
        elif(band==2):
                erange="100-200"
        elif(band==3):
                erange="20-200"
        print('\r\n','--------------------------------------------')
        print('cztbindata', 'inevtfile=' + quad_cleanevt,'mkffile=' + mkffile, 'badpixfile=' + flag_badpix)
        print('--------------------------------------------','\r\n')
        # Run cztbindata
        subprocess.call(['cztbindata', 'inevtfile=' + quad_cleanevt,
                                         'mkffile=' + mkffile, 'badpixfile=' + flag_badpix,
                                         'quadsToProcess=-', 'badpixThreshold=0',
                                         'livetimefile=' + quad_livet, 'outputtype=lc',
                                         'energyrange=' + erange, 'generate_eventfile=yes',
                                         'timebinsize=' + str(tbin), 'outfile=' + quad_lc,
                                         'outevtfile=' + quad_weights, 'maskWeight=no',
                                         'rasrc=0.0', 'decsrc=0.0', 'clobber=yes',
                                         'history=yes', 'debug=no'])
        return quad_lc


def pipeline(infolder, outpath, outbase, tbin, mkf_thresholds):
        """Run the czti pipeline.

        For more information on the pipeline refer
        http://astrosat-ssc.iucaa.in/uploads/czti/CZTI_level2_software_userguide_v1.1.pdf

        Inputs:
        infolder - Input folder (containing the relevant files)
        outpath - Locaton of creation of output files
        outbase - Prefix for the output light
        tbin - array containg all the required bin times
        mkf_thresholds - File containg thresholds to select events

        Ouputs:
        quad_lcs - List of prefixes to the names of light curves
        quad_weights - List of Evt files containing the weights applied

        Local variables:
        bcevt - Bunch clean event file name
        mkffile = .mkf file name
        outgti - GTI file name
        quad_dsevt - Data selected file with QUAD gtitype
        bclive_tfile - live time file
        quad_pcevt - Pix clean file name
        quad_livet - Livetime file generated by cztpixclean
        quad_badpix - Badpix file generated by cztpixclean
        quad_cleanevt - Cleaned event file
        flag_badpix - File containing the flagged bad pixels
        elem - (i, tbin[i])
        quad_lc = Prefix of light curves for given tbin
        quad_lc_band = Prefix of light curves for given band in given tbin
        quad_weight - Event file containing weights for given tbin

        """
        infolder = infolder+'/'
        outpath = outpath+'/'
        
        try:
                bcevt = glob.glob(infolder + '**/*_bc.evt')[0]
                mkffile = glob.glob(infolder + '*.mkf')[0]
                check_and_edit_column(mkffile)

                print('\nLook here', bcevt, mkffile , outpath, mkf_thresholds)
                outgti = gtigen(bcevt, mkffile, outpath, mkf_thresholds)
                quad_dsevt = datasel(bcevt, outgti, outpath)
                bclive_tfile = glob.glob(infolder + '**/*bc_livetime.fits')[0]
                quad_pcevt, quad_livet, quad_badpix,quad_dblevt = pixclean(
                        quad_dsevt, bclive_tfile, 100, 1000, outpath)
                quad_cleanevt = evtclean(quad_pcevt)
                flag_badpix = flag_bad(quad_badpix)
                quad_lcs = []
                for t_bin in tbin:
                        quad_lc_band=[]

                        for band in range(4):
                                print(f'Starting bindata for {tbin} binsize, and Band {band}')
                                quad_lc = bindata(quad_cleanevt, mkffile, flag_badpix, quad_livet, t_bin, outbase, band)
                                quad_lc_band.append(quad_lc)
                        quad_lcs.append(quad_lc_band)
        except:
                print('Files missing')
                quad_lcs = []
        return quad_lcs
'''-------------------------------------------------------------------------------------------------------------------------'''

def median_filter(array, filterwidth):
        """Apply median filter to the array.

        Inputs:
        array - Input array to be filtered
        filterwidth - Width of filter

        Output:
        arr_filt - Filtered array
        """
        halfwidth = (int)(filterwidth/2)
        arr_filt = np.zeros_like(array)
        for index in range(len(array[:halfwidth])):
                arr_filt[index] = np.median(array[:index+halfwidth+1])
        for index in range(halfwidth, len(array[:-1*halfwidth])):
                arr_filt[index] = np.median(array[index-halfwidth:index+halfwidth+1])
        for index in range(-1*halfwidth, 0):
                arr_filt[index] = np.median(array[index-halfwidth:])
        return arr_filt


def filterwindows(startbins, stopbins, tbin):
        """Filter windows is length is below 0.5s.

        Inputs:
        startbins - raw startbins
        stopbins - raw stopbins

        Outups
        startbins2 - filtered startbins
        stopbins2 - siltered stopbins
        """
        startbins2 = copy.copy(startbins)
        stopbins2 = copy.copy(stopbins)
        for bin_i, stopbin in enumerate(stopbins[:-1]):
                if startbins[bin_i+1] - stopbin < 0.5/tbin:
                        startbins2.remove(startbins[bin_i+1])
                        stopbins2.remove(stopbin)
        return startbins2, stopbins2


def clipwindows(startbins, stopbins, clipbins, minwidth):
        """Clip windows by given tclip.

        Minimum width of windows after clipping should be minwidth.

        Inputs:
        startbins - filtered startbins
        stopbins - filtered stopbins
        clipbins - number of bins to be clipped
        minwidth - minimum width of windows after clipping

        Ouputs:
        startbins2 - clipped startbins
        stopbins - clipped stopbins
        """
        startbins2 = []
        stopbins2 = []
        for bin_i, startbin in enumerate(startbins):
                str_bin = startbin + clipbins
                stp_bin = stopbins[bin_i] - clipbins
                if stp_bin - str_bin > minwidth:
                        startbins2.append(str_bin)
                        stopbins2.append(stp_bin)
        return startbins2, stopbins2


def getwindows(l_curve, threshold, bin_t, tclip, minwidth=0):
        """Return windows where rate is above threshold.

        Inputs:
        l_curve - Input light curve
        threshold - Threshold rate
        bin_t - Bin length of time
        tclip - time in seconds to be clipped
        minwidth - Minimum width of each window

        Outputs:
        windows - Windows of light curves above thresold rate
        mask_lc - Mask to be used for further analysic of lc

        Local variables:
        lc_mask - raw mask of l_curve
        mask_temp - raw mask
        clipbins - number of bins to be clipped
        startbins = start bins for windows
        stopbins = stopbins for windows
        """
        lc_mask = np.ma.masked_less_equal(l_curve, threshold)
        mask_temp = np.atleast_1d(lc_mask.mask)
        #print mask_temp
        clipbins = (int)(tclip/bin_t) + 1
        windows = []
        startbins = []
        stopbins = []
        if not mask_temp[0]:
                startbins.append(0)
        for tbin, mask in enumerate(mask_temp[1:]):
                if mask_temp[tbin] and not mask:
                        startbins.append(tbin+1)
                if not mask_temp[tbin] and mask:
                        stopbins.append(tbin+1)
        if not mask_temp[-1]:
                stopbins.append(len(l_curve))
        startbins2, stopbins2 = filterwindows(startbins, stopbins, bin_t)
        startbins3, stopbins3 = clipwindows(startbins2, stopbins2, clipbins,
                                                                                minwidth)
        for bin_i, startbin in enumerate(startbins3):
                windows.append(l_curve[startbin:stopbins3[bin_i]])

        mask_lc = np.ones(len(l_curve), dtype=bool)
        for bin_i, startbin in enumerate(startbins3):
                mask_lc[startbin:stopbins3[bin_i]] = 0
        return windows, mask_lc, startbins3, stopbins3 #Changed SAA bin output from startbins3 to startbins

def get_trend(tbin, l_curve, threshold, filtertype, filterorder, filterwidth,
                          tclip):
        """Return the trend of the light curve.

        Returned trend will be used for detrending. The values below threshold
        are masked and the trend is calculated using required filter.
        Then the values corresponding to masked indices are replaced by zeros.

        Inputs:
        l_curve - Iinitial light curve
        threshold - Threshold below which the values shouldn't be considered
        filtertype - savgol or median
        filterorder - order of the savgol filter
        filterwidth - width of the filter
        tclip - Time in seconds to be clipped

        Outputs:
        trend - Trend of the light curve
        lc_mask.mask - Mask to be applied to filter out data below threshold

        Local variables:
        lc_mask - Masked light curves
        """
        error_flag = 0
        filterbins = (int)(filterwidth/tbin) + 1
        windows, mask_lc, startbins3, stopbins3 = getwindows(l_curve, threshold, tbin, tclip, 2)
        trend = np.zeros_like(l_curve)
        win_trends = []
        for window in windows:
                if filtertype == 'savgol':
                        if len(window) <= filterbins:
                                trend_win = savgol_filter(window, ((len(window)+1)//2)*2 - 1,
                                                                                  filterorder)
                        else:
                                trend_win = savgol_filter(window, filterbins,
                                                                                  filterorder)
                elif filtertype == 'median':
                        if len(window) <= filterbins:
                                trend_win = median_filter(window, len(window)//2)
                        else:
                                trend_win = median_filter(window, filterbins)
                win_trends.append(trend_win)
        #print win_trends
        if(win_trends == []):
                error_flag=1
        else:
                trend[~mask_lc] = np.concatenate(win_trends)

        return trend, mask_lc, startbins3, stopbins3, error_flag


def getlc_clean(tbin, quad, threshold, filtertype, filterorder, filterwidth,
                                tclip):
        """Return detrended light curves.

        Inputs:
        tbin - bin time
        quad - Light curve file of quadrant
        threshold - threshold count below which count rate is considered invalid
        filterorder - Degree of polynomial used in the savgol_filter
        filterwidth - Length of filter window. Must be odd positive
        filtertype - 'median' or 'savgol'
        tclip - Time in seconds to be clipped

        Outputs:
        bin_times - Time bins of the light curve
        lc_detrend - Detrended light curve
        mask_lc - Mask to be applied to filter out data below threshold

        Local variables:
        quad_lc - original light curve
        rate - Count rate of the original light curve after livetime corrections
        trend - Smoothened light curve

        """
        quad_lc = fits.getdata(quad)
        rate = quad_lc['rate']
        trend, mask_lc,startbins3,stopbins3, error_flag = get_trend(tbin, rate, threshold, filtertype, filterorder,
                                                           filterwidth, tclip)
        lc_detrend = np.zeros_like(rate)
        lc_detrend[~mask_lc] = rate[~mask_lc] - trend[~mask_lc]
        return lc_detrend, mask_lc,startbins3,stopbins3, error_flag

def zero_runs(a):
                # Create an array that is 1 where a is 0, and pad each end with an extra 0.
                iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
                absdiff = np.abs(np.diff(iszero))
                # Runs start and end where absdiff is 1.
                ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
                return ranges

def getbin_lcs(tbin, quad_lc, threshold, filtertype, filterorder,
                           filterwidth, tclip):
        """Return light curves for the 4 quandrants.

        Inputs:
        tbin - bin time
        quad - Light curve file of quadrant
        threshold - Threshold count below which count rate is considered invalid
        filtertype - 'median' or 'savgol'
        filterorder - Degree of polynomial used in the savgol_filter
        filterwidth - Length of filter window. Must be odd positive
        tclip = Time inseconds to be clipped

        Outputs:
        bins_all - List of bin times of all quadrants
        lc_all - List of light curves of all quadrants
        masks_all - List of all masks

        Local variables:
        quadname - Name of the individual .lc file
        bins, l_curves, mask_lc - values corresponding to one quadrant
        """

        bins_all = []
        lc_all = []
        masks_all = []
        for quadnum in range(4):
                quadname = quad_lc + '_Q' + str(quadnum) + '.lc'
                bins = fits.getdata(quadname)['time']
                l_curve, mask_lc,startbins3,stopbins3, error_flag = getlc_clean(tbin, quadname, threshold,
                                                                           filtertype, filterorder,
                                                                           filterwidth, tclip)
                bins_all.append(bins)
                lc_all.append(l_curve)
                masks_all.append(mask_lc)
        zeros = []
        indices_l1 = []
        indices_r1 = []
        indices_l2 = []
        indices_r2 = []
        indices = []
        for i in range(4):
                loc = np.where(np.diff(zero_runs(lc_all[i])) >= 900/tbin)[0]
                locc = np.diff(zero_runs(lc_all[i]))[loc]
                if len(loc) == 2:
                        indices_l1.append(zero_runs(lc_all[i])[loc][0,0])
                        indices_r1.append(zero_runs(lc_all[i])[loc][0,1])
                        indices_l2.append(zero_runs(lc_all[i])[loc][1,0])
                        indices_r2.append(zero_runs(lc_all[i])[loc][1,1])
                elif len(loc) == 1:
                        indices_l1.append(zero_runs(lc_all[i])[loc][0,0])
                        indices_r1.append(zero_runs(lc_all[i])[loc][0,1])
                # plt.plot(bins_all[i], lc_all[i], label='Inline label')
                # plt.savefig(f'data/test_img_{tbin}_{i}.png')
                # plt.legend()
                # print('plotted')
        lc_all = np.array(lc_all)
        bins_all = np.array(bins_all)
        masks_all = np.array(masks_all)

        ## if no SAA in orbit (unlikely)/ Cutoff is somehow not crossed
        if len(indices_l1) == 0:
                indices_l1 = [0]

        if len(indices_r1) == 0 and bins_all.size > 4:
                right_index1 = bins_all.shape[1]-1
        elif len(indices_r1) == 0:
                right_index1 = 0
        else:
                right_index1 = np.max(indices_r1)

        if len(indices_r2) == 0 and bins_all.size > 4:
                right_index2 = bins_all.shape[1]-1
        elif len(indices_r2) == 0:
                right_index2 = 0
        else:
                right_index2 = np.max(indices_r2)

        saa_start = []
        saa_end = []
        if indices_l2:
                left_index1 = np.max(indices_l1)
                left_index2 = np.max(indices_l2)
                ## Correcting for the index
                if bins_all.size > 4:
                        if right_index1 == bins_all.shape[1]:
                                right_index1 = right_index1 - 1
                        if right_index2 == bins_all.shape[1]:
                                right_index2 = right_index2 - 1
                
                saa_start.append(bins_all[0,left_index1] - t_buffer)
                saa_start.append(bins_all[0,left_index2] - t_buffer)
                saa_end.append(bins_all[0,right_index1] + t_buffer) 
                saa_end.append(bins_all[0,right_index2] + t_buffer)
        else:
                left_index1 = np.max(indices_l1)
                ## Correcting for the index
                if bins_all.size > 4:
                        if right_index1 == bins_all.shape[1]:
                                right_index1 = right_index1 - 1
                saa_start.append(bins_all[0][left_index1] - t_buffer)
                saa_end.append(bins_all[0][right_index1] + t_buffer) 
        
        tmin = max(bins_all[0][0],bins_all[1][0],bins_all[2][0],bins_all[3][0])
        # diff_tmin = [int((-bins_all[0][0]+tmin)/tbin),int((-bins_all[1][0]+tmin)/tbin),int((-bins_all[2][0]+tmin)/tbin),int((-bins_all[3][0]+tmin)/tbin)]
        tmax = max(bins_all[0][-1],bins_all[1][-1],bins_all[2][-1],bins_all[3][-1])
        # print('OBSID Start:', tmin,', OBSID End:', tmax)
        new_bins_all=[]
        new_lc_all=[]
        new_masks_all=[]
        for i in range(4):
                new_bins_all.append(bins_all[i])
                new_lc_all.append(lc_all[i])
                new_masks_all.append(masks_all[i])
        stopbins3 = np.array(stopbins3)
        startbins3 = np.array(startbins3)
        stopbins3 = stopbins3[stopbins3<(len(new_bins_all[0])-1)]
        startbins3 = startbins3[startbins3<(len(new_bins_all[0])-1)]
        for i in range(4):
                if(len(new_bins_all[i])!=len(new_bins_all[0])):
                        error_flag=1
                else:
                        error_flag = 0

        # print(len(new_bins_all[0]),startbins3,stopbins3)
        return new_bins_all[0], new_lc_all, new_masks_all,startbins3,stopbins3, error_flag, saa_start, saa_end
'''------------------------------------------------------------------------------------------------------------------------------'''

def get_vetocount(orbit):
        '''
        For a given orbit, get veto spectrum data column from quad cleaned file(pipeline processed) and extract veto counts
        Input: Orbit path
        Output: time - time bin array for four quadrants, shape(4,length of orbit)
                        veto_lc - veto counts array for four quadrants, shape(4, length of orbit)
        Local variables:
                        quad_clean = path to quad cleaned event file
                        hdu = fits data array from quad_clean
                        data = veto spectrum data
                        table = Tabulated form of data
                        vetocount = veto count array for each row of table
        '''
        quad_clean = glob.glob(orbit+'/*quad_clean.evt')[0]
        if(quad_clean==[]):
                print("Quad clean file doesn't exist for "+orbit)
                return [],[]
        else:
                hdu = fits.open(quad_clean)
                d = hdu[5].data
                table = Table(d)
                vetocount = np.sum(table['VetoSpec'][:,0:128], axis=1)
                table.add_column(Column(vetocount, name='VetoCount'))
                veto_lc=[]
                time = []
                for quad in range(4):
                        lc = table[table['QuadID']==quad]
                        veto_lc.append(lc['VetoCount'])
                        time.append(lc['Time'])
                time = np.array(time)
                veto_lc = np.array(veto_lc)
                zeros = []
                new_time=[]
                new_veto_lc_all=[]
                if len(zero_runs(veto_lc[0])) != 0:
                        for i in range(4):
                                loc = np.argmax(np.diff(zero_runs(veto_lc[i])))
                                index = zero_runs(veto_lc[i])[loc][0]
                                zeros.append(time[i][index])
                        veto_lc = np.array(veto_lc)
                        time = np.array(time)
                        veto_lc = veto_lc[:,:index]
                        time = time[:,:index]
                        tmin = max(time[0][0],time[1][0],time[2][0],time[3][0])
                        tmax = max(time[0][-1],time[1][-1],time[2][-1],time[3][-1])
                        for i in range(4):
                                new_time.append(time[i])
                                new_veto_lc_all.append(veto_lc[i])
                else:      
                        tmin = max(time[0][0],time[1][0],time[2][0],time[3][0])
                        diff_tmin = [int((-time[0][0]+tmin)),int((-time[1][0]+tmin)),int((-time[2][0]+tmin)),int((-time[3][0]+tmin))]
                        tmax = min(len(time[0])-diff_tmin[0],len(time[1])-diff_tmin[1],len(time[2])-diff_tmin[2],len(time[3])-diff_tmin[3])
                        for i in range(4):
                                new_time.append(time[i][diff_tmin[i]:(diff_tmin[i]+tmax)])
                                new_veto_lc_all.append(veto_lc[i][diff_tmin[i]:(diff_tmin[i]+tmax)])
                
        return new_time, new_veto_lc_all

def veto_lc_binning(time, veto_lc, t_bin, orbit):
        tbins=[]
        quad_lcs=[]
        time=np.array(time)
        veto_lc=np.array(veto_lc)
        for num, binval in enumerate(t_bin):
                # if(binval == 1.0):
                #       continue
                # else:
                lctable = Table()
                lctable['time']=time[0]
                for quad in range(4):
                        lctable['counts'+str(quad)]=veto_lc[quad]
                tbin = np.trunc(time[0]/binval)
                lctable_grouped = lctable.group_by(tbin)
                quad_lc=[]
                for quad in range(4):
                        lcbinned = lctable_grouped['counts'+str(quad)].groups.aggregate(np.sum)
                        quad_lc.append(np.array(lcbinned))
                tbins.append(np.array(lctable_grouped['time'].groups.aggregate(np.mean)))
                quad_lcs.append(quad_lc)
        tbins = np.array(tbins)
        quad_lcs = np.array(quad_lcs)
        return tbins, quad_lcs

def getvetowindows(orbit, veto_time, veto_lc, threshold, bin_t, tclip, minwidth=0):
        """Return windows where rate is above threshold.

        Inputs:
        l_curve - Input light curve
        threshold - Threshold rate
        bin_t - Bin length of time
        tclip - time in seconds to be clipped
        minwidth - Minimum width of each window

        Outputs:
        windows - Windows of light curves above thresold rate
        mask_lc - Mask to be used for further analysic of lc

        Local variables:
        lc_mask - raw mask of l_curve
        mask_temp - raw mask
        clipbins - number of bins to be clipped
        startbins = start bins for windows
        stopbins = stopbins for windows
        """
        mkf = glob.glob(orbit+'/*mkf')[0]
        hdu = fits.open(mkf)
        d = hdu[1].data
        tbl = Table(d)
        cpm_rate = np.array(tbl['CPM_Rate'])
        time = np.array(tbl['TIME'])
        time = np.round(time,0)
        tstart=[]
        if(cpm_rate[0]<threshold):
                tstart.append(time[0])
        tend=[]
        for i in range(len(time)-1):
                if(cpm_rate[i+1]<threshold and cpm_rate[i]>=threshold):
                        tstart.append(time[i+1])
                if(cpm_rate[i+1]>=threshold and cpm_rate[i]<threshold):
                        tend.append(time[i])
        if(cpm_rate[-1]<threshold):
                tend.append(time[-1])
        startbins=[]
        stopbins=[]
        for i,t in enumerate(tstart):
                startbins.append((np.abs(veto_time-t)).argmin())
                stopbins.append((np.abs(veto_time-tend[i])).argmin())
        clipbins = (int)(tclip/bin_t) + 1
        windows = []
        startbins2, stopbins2 = filterwindows(startbins, stopbins, bin_t)
        startbins3, stopbins3 = clipwindows(startbins2, stopbins2, clipbins,
                                                                                minwidth)
        for bin_i, startbin in enumerate(startbins3):
                windows.append(veto_lc[startbin:stopbins3[bin_i]])

        mask_lc = np.ones(len(veto_lc), dtype=bool)
        for bin_i, startbin in enumerate(startbins3):
                mask_lc[startbin:stopbins3[bin_i]] = 0
        # plt.figure()
        # plt.plot(veto_time,mask_lc,'-')
        # plt.savefig(os.path.basename(orbit)+'test'+str(bin_t)+'.png')
        return windows, mask_lc, startbins3, stopbins3

def get_trend_veto(tbin, l_curve, windows, mask_lc, filtertype, filterorder, filterwidth,orbit,veto_time):
        """Return the trend of the light curve.

        Returned trend will be used for detrending. The values below threshold
        are masked and the trend is calculated using required filter.
        Then the values corresponding to masked indices are replaced by zeros.

        Inputs:
        l_curve - Iinitial veto light curve
        windows - Orbit windows obtained from ordinary light curve
        mask_lc - masks obtained from ordinary light curve
        filtertype - savgol or median
        filterorder - order of the savgol filter
        filterwidth - width of the filter
        tclip - Time in seconds to be clipped

        Outputs:
        trend - Trend of the light curve
        lc_mask.mask - Mask to be applied to filter out data below threshold
        error_flag - something to track error in masking windows

        Local variables:
        lc_mask - Masked light curves
        """
        error_flag = 0
        filterbins = (int)(filterwidth/tbin) + 1
        trend = np.zeros_like(l_curve)
        win_trends = []
        for window in windows:
                if(len(window)==0):
                        continue
                if filtertype == 'savgol':
                        if len(window) <= filterbins:
                                trend_win = savgol_filter(window, ((len(window)+1)//2)*2 - 1, filterorder)
                        else:
                                trend_win = savgol_filter(window, filterbins, filterorder)
                elif filtertype == 'median':
                        if len(window) <= filterbins:
                                trend_win = median_filter(window, len(window)//2)
                        else:
                                trend_win = median_filter(window, filterbins)
                win_trends.append(trend_win)
        #print win_trends
        if(win_trends == []):
                error_flag=1
        else:
                trend[~mask_lc] = np.concatenate(win_trends)
        # plt.figure()
        # plt.plot(veto_time,l_curve,'-')
        # plt.plot(veto_time,trend,'r-')
        # plt.savefig('test'+str(tbin)+'.png')

        return trend, mask_lc, error_flag

def getvetobin_lcs(tbins, veto_lc, outpath, outbase, args, orbit):
        ## Get detrended veto light curves
        error_loc = []
        flags = []
        args.tbin = np.array(args.tbin)
        veto_tbin = args.tbin[args.tbin != 0.1] 
        for num,tbin in enumerate(veto_tbin):
                for quad in range(4):
                        #get_SAA(orbit,veto_lc[num][quad],tbins[num])
                        windows, mask, startbins, stopbins = getvetowindows(orbit, tbins[num], veto_lc[num,quad], args.threshold, tbin, args.tclip, 2)
                        veto_trend, veto_mask, error_flag = get_trend_veto(tbin, veto_lc[num,quad], windows, mask, 
                                                                                                        args.filtertype, args.filter_order, args.filterwidth,orbit,tbins[num])
                        if(error_flag==1):
                                print("error veto in "+outbase+" tbin "+str(tbin))
                                error_loc.append(str(tbin)+"_"+str(quad))
                                flags.append(num)
                                break
                        lc = np.array(veto_lc[num][quad])
                        veto_detrend = np.zeros(len(lc))
                        opp_mask = ~veto_mask
                        lc = lc.astype(float)
                        veto_trend = veto_trend.astype(float)
                        veto_detrend = np.subtract(lc,veto_trend,where=opp_mask)
                        # plt.figure()
                        # #plt.plot(tbins[num],lc,'r-')
                        # #plt.plot(tbins[num],veto_trend,'-')
                        # plt.plot(tbins[num],veto_detrend)
                        # #plt.xlim(266581780,266581900)
                        # plt.savefig('test'+str(num)+'.png')
                        c1 = fits.Column(name='time',array=tbins[num],format='D')
                        c2 = fits.Column(name='vetocounts',array=veto_detrend,format='D')
                        c3 = fits.Column(name='vetomask',array=veto_mask,format='L')
                        c4 = fits.Column(name='orig_vetocts',array=lc,format='D')
                        c5 = fits.Column(name='trend',array=veto_trend,format='D')
                        newtbl = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5])
                        #print outpath+'/'+outbase+'_veto_'+str(tbin)+'_Q'+str(quad)+'_detrended.fits'
                        newtbl.writeto(outpath+'/'+outbase+'_veto_'+str(tbin)+'_Q'+str(quad)+'_detrended.fits',overwrite=True)
                if(len(startbins) !=0):
                        startsatime = tbins[num][startbins]
                else:
                        startsatime = []
                if(len(stopbins) != 0):
                        stopsatime = tbins[num][stopbins]
                else:
                        stopsatime = []

        return flags,error_loc,startsatime,stopsatime

def create_orbit_tables(imdb):
    #imdb = sqlite3.connect('interface/orbitinfo')
    cur = imdb.cursor()
    # create orbitinfo table
    cur.execute(""" 
        CREATE TABLE IF NOT EXISTS orbitinfo (
        outbase text PRIMARY KEY,
        start_of_orbit real,
        end_of_orbit real,
        veto_start real,
        veto_end real
        );
        """)

def add_orbitinfo(imdb, orbitinfo):
	orbitinfo = orbitinfo[["outbase", "start_of_orbit", "end_of_orbit", "veto_start", "veto_end"]] 
	orbitinfo.to_sql('orbitinfo', imdb, if_exists = 'append', index = False)

def add_bins(startbins, stopbins, veto_startbins, veto_stopbins):
	#file = 'interface/comb_bins.json'
	if(glob.glob('interface/comb_bins.json')==[]):
		with open('interface/comb_bins.json', 'w') as file:
			file.write('{"startbins": ')
			json.dump(startbins.tolist(), file)
			file.write(', ')

			file.write('"stopbins": ')
			json.dump(stopbins.tolist(), file)
			file.write(', ')

			file.write('"veto_startbins": ')
			json.dump(veto_startbins.tolist(), file)
			file.write(', ')

			file.write('"veto_stopbins": ')
			json.dump(veto_stopbins.tolist(), file)
			file.write('}')
	else:
		with open('interface/comb_bins.json', 'r') as file:
			data = json.load(file)
			data['startbins'] = np.concatenate([data['startbins'], startbins]).tolist()
			data['stopbins'] = np.concatenate([data['stopbins'], stopbins]).tolist()
			data['veto_startbins'] = np.concatenate([data['veto_startbins'], veto_startbins]).tolist()
			data['veto_stopbins'] = np.concatenate([data['veto_stopbins'], veto_stopbins]).tolist()
		
		with open('interface/comb_bins.json', 'w') as file:
			json.dump(data, file)

def store_df(orbitinfo):
	orbitinfo.to_pickle('interface/orbitinfo.pkl')

'''---------------------------------------------------------------------------------------------------------------------------------'''

def make_detrended_lc(orbits,outpaths,args,dirname):
        orbitinfo = pd.DataFrame(columns=['outbase','start_of_orbit','end_of_orbit','error_loc',
                                                        'error_flag','veto_start','veto_end','veto_error_loc','veto_error_flag'],index=range(len(outpaths)))
        orbitinfo.apply(pd.to_numeric, errors='ignore')
        ## save detrended czti lc to fits file for all orbits
        #print orbitinfo
        comb_startbins, comb_stopbins = [],[]
        print('outpaths:', outpaths)
        for o,orbit in enumerate(outpaths):
                outbase = os.path.basename(orbit)
                orbitinfo.loc[o]['outbase']= outbase
                orbitinfo.loc[o]['error_loc'] = []
                orbitinfo.loc[o]['error_flag'] = []
                print(args.tbin)
                for c_tbin,n_tbin in enumerate(args.tbin):
                        for band in range(3):
                                quad_lc = "{orbit}/{outbase}_{tbin}_{band}".format(orbit=outpaths[o],outbase=outbase,tbin=n_tbin,band=band)
                                # bins_band, lc_band, masks_band,startbins3,stopbins3, error_flag = getbin_lcs(n_tbin, quad_lc, args.threshold, args.filtertype, args.filter_order,args.filterwidth, args.tclip)
                                bins_band, lc_band, masks_band,startbins3,stopbins3, error_flag, saa_start, saa_end = getbin_lcs(n_tbin, quad_lc, args.threshold, args.filtertype, args.filter_order,args.filterwidth, args.tclip)
                                print('saa_start', saa_start, 'saa_end', saa_end)
                                if(error_flag==1 or len(bins_band) == 0):
                                        ##########log orbit,tbin and band in which error occurred
                                        orbitinfo.loc[o]['error_loc'].append(str(n_tbin)+"_"+str(band))
                                        orbitinfo.loc[o]['error_flag'].append(c_tbin)
                                        startsatime, stopsatime = [],[]
                                        orbitinfo.loc[o]['start_of_orbit'] = 0
                                        orbitinfo.loc[o]['end_of_orbit'] = 0
                                        break
                                else:
                                        if(c_tbin == 0 and band == 0):
                                                if(len(startbins3) !=0):
                                                        startsatime = bins_band[startbins3]
                                                else:
                                                        startsatime = []
                                                if(len(stopbins3)!=0):
                                                        stopsatime = bins_band[stopbins3]
                                                else:
                                                        stopsatime = []
                                        orbitinfo.loc[o]['start_of_orbit'] = bins_band[0]
                                        orbitinfo.loc[o]['end_of_orbit'] = bins_band[-1]
                                        print("bins_band", bins_band[0], bins_band[-1])
                                        for quad in range(4):
                                                c1 = fits.Column(name='time', array=bins_band,format='D')
                                                c2 = fits.Column(name='countrate', array=lc_band[quad],format = 'D')
                                                c3 = fits.Column(name='mask',array=masks_band[quad],format='L')
                                                tbl = fits.BinTableHDU.from_columns([c1,c2,c3])
                                                tbl.writeto(quad_lc+'_Q'+str(quad)+'_detrended.fits',overwrite = True)
                # if(error_flag==1):
                #       orbit_range.append([0,0])
                #       continue
                # else:
                        #print len(bins_band)

                comb_startbins = np.concatenate([comb_startbins,startsatime])
                comb_stopbins = np.concatenate([comb_stopbins,stopsatime])
        
        ##### Combine detrended czti lc
        for c_tbin,n_tbin in enumerate(args.tbin):
                for band in range(3):
                        for quad in range(4):
                                tbl = Table()
                                tbl.add_column(Column(name='time',data=[],dtype='float64'))
                                tbl.add_column(Column(name='countrate',data=[],dtype=float))
                                tbl.add_column(Column(name='mask',data=[],dtype=bool))
                                for o,orbit in enumerate(outpaths):
                                        if(c_tbin in np.array(orbitinfo.loc[o]['error_flag'])):
                                                continue
                                        else:
                                                try:
                                                        quad_lc = outpaths[o]+'/'+os.path.basename(orbit)+'_'+str(n_tbin)+'_'+str(band)+'_Q'+str(quad)+'_detrended.fits'
                                                        t = Table.read(quad_lc,format='fits')
                                                        tbl = vstack([tbl,t])
                                                except:
                                                        continue
                                        #print len(tbl)
                                tbl.write('data/local_level2/'+dirname+'/czti/combined_'+str(n_tbin)+'_'+str(band)+'_Q'+str(quad)+'_detrended.fits',overwrite = True)

        ### Make veto detrended lc
        args.tbin = np.array(args.tbin)
        veto_tbin = args.tbin[args.tbin != 0.1]
        args.threshold = 15
        args.tclip = 80
        comb_veto_startbins,comb_veto_stopbins = [],[]
        for o,orbit in enumerate(outpaths):
                outbase = os.path.basename(orbit)
                time, veto_lc = get_vetocount(orbit)
                orbitinfo.loc[o]['veto_error_loc'] = []
                orbitinfo.loc[o]['veto_error_flag'] = []
                #print len(time[0]), np.shape(veto_lc)
                if(len(time[0])==0 or len(veto_lc[0])==0 or len(veto_lc[1])==0 or len(veto_lc[2])==0 or len(veto_lc[3])==0 ):
                        print("error : veto lightcurve is zero length array for orbit "+orbit)
                        orbitinfo.loc[o]['veto_start'] = 0
                        orbitinfo.loc[o]['veto_end'] = 0
                        orbitinfo.loc[o]['veto_error_loc'].append("LC length zero")
                        orbitinfo.loc[o]['veto_error_flag'].append(3)
                        continue
                else:
                        orbitinfo.loc[o]['veto_start'] = time[0][0]
                        orbitinfo.loc[o]['veto_end'] = time[0][-1]
                        tbins, quad_lcs = veto_lc_binning(time,veto_lc,veto_tbin,orbit)
                        error_flag,error_loc,startsatime,stopsatime = getvetobin_lcs(tbins, quad_lcs, outpaths[o], outbase, args, orbits[o])
                        orbitinfo.loc[o]['veto_error_loc'] = error_loc
                        orbitinfo.loc[o]['veto_error_flag'] = error_flag
                        comb_veto_startbins = np.concatenate([comb_veto_startbins,startsatime])
                        comb_veto_stopbins = np.concatenate([comb_veto_stopbins,stopsatime])

        ## Combine veto lcs
        for num,t_bin in enumerate(veto_tbin):
                for quad in range(4):
                        tbl = Table()
                        tbl.add_column(Column(name='time',data=[],dtype='float64'))
                        tbl.add_column(Column(name='vetocounts',data=[],dtype=float))
                        tbl.add_column(Column(name='vetomask',data=[],dtype=bool))
                        tbl.add_column(Column(name='orig_vetocts',data=[],dtype=float))
                        tbl.add_column(Column(name='trend',data=[],dtype=float))                
                        for o, orbit in enumerate(outpaths):
                                if(num in orbitinfo.loc[o]['veto_error_flag']):
                                        continue
                                else:
                                        try:
                                                quad_lc = outpaths[o]+'/'+os.path.basename(orbit)+'_veto_'+str(t_bin)+'_Q'+str(quad)+'_detrended.fits'
                                                t = Table.read(quad_lc,format='fits')
                                                tbl = vstack([tbl,t])
                                        except:
                                                continue
                        tbl.write('data/local_level2/'+dirname+'/czti/combined_veto_'+str(t_bin)+'_Q'+str(quad)+'_detrended.fits',overwrite=True)

        #store_df(orbitinfo)
        #add_bins(comb_startbins, comb_stopbins, comb_veto_startbins, comb_veto_stopbins)

        return orbitinfo, comb_startbins, comb_stopbins, comb_veto_startbins, comb_veto_stopbins, saa_start, saa_end

