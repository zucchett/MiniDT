#!/usr/bin/env python
# coding: utf-8

import os, sys, math
import pandas as pd
import numpy as np
from datetime import datetime

from modules.mapping.config import TDRIFT, VDRIFT, XCELL, ZCELL, DURATION
from modules.mapping import *
from modules.analysis.patterns import PATTERNS, PATTERN_NAMES, ACCEPTANCE_CHANNELS, MEAN_TZERO_DIFF, MEANTIMER_ANGLES, meantimereq, mean_tzero, tzero_clusters
from modules.reco.functions import *
from modules.reco import plot
from modules.reco import config_3_1 as config

import argparse
parser = argparse.ArgumentParser(description='Command line arguments')
parser.add_argument("-d", "--display", action="store", type=int, default=0, dest="display", help="Number of events to display")
parser.add_argument("-e", "--event", action="store", type=int, default=0, dest="event", help="Inspect a single event")
parser.add_argument("-f", "--flush", action="store_true", default=False, dest="flush", help="Discard first 128 words")
parser.add_argument("-i", "--inputfile", nargs='+', dest="filenames", default="data/Run000966/output_raw.dat", help="Provide input files (either binary or txt)")
parser.add_argument("-o", "--outputdir", action="store", type=str, dest="outputdir", default="output/", help="Specify output directory, if empty no csv output is produced")
parser.add_argument("-m", "--max", action="store", type=int, default=-1, dest="max", help="Maximum number of words to be read")
parser.add_argument("--noreco", action="store_true", default=False, dest="noreco", help="Do not run reconstruction")
parser.add_argument("--nosave", action="store_true", default=False, dest="nosave", help="Do not save output")
parser.add_argument("-p", "--parallel", action="store_true", default=False, dest="parallel", help="Enable CPU parallelization")
parser.add_argument("-t", "--tzero", action="store", type=str, default=False, dest="tzero", help="Specify the algorithm to be used to determine the T0. M : meantimer, T : HT trigger, S : SiPM scintillators, P : PMT scintillators, R : scintillators + time refit")
parser.add_argument("-s", "--suffix", action="store", type=str, default="", dest="suffix", help="Specify the suffix of the output files")
parser.add_argument("-v", "--verbose", action="store", type=int, default=0, dest="verbose", help="Specify verbosity level")
args = parser.parse_args()

runname = [x for x in args.filenames[0].split('/') if 'Run' in x][0] if "Run" in args.filenames[0] else "Run000000"

if len(args.outputdir) > 0:
    if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)
    for d in ["csv", "display", "plots"]:
        if not os.path.exists(args.outputdir + runname + "_" + d + "/"): os.makedirs(args.outputdir + runname + "_" + d + "/")

# Copy data from LNL to CERN
# scp /data/Run001089/*.dat zucchett@lxplus.cern.ch:/afs/cern.ch/work/z/zucchett/public/FortyMHz/Run001089/
# Copy data from CERN to local
# mkdir data/Run001089
# scp lxplus.cern.ch:/afs/cern.ch/work/z/zucchett/public/FortyMHz/Run001089/* data/Run001089/


mapconverter = Mapping()

# Layer    # Parameters

#          +--+--+--+--+--+--+--+--+
# 4        |  1  |  5  |  9  |  13 | 17 ...
#          +--+--+--+--+--+--+--+--+
# 3           |  3  |  7  |  11 | 15 ...
#          +--+--+--+--+--+--+--+--+
# 2        |  2  |  6  |  10 |  14 | 18 ...
#          +--+--+--+--+--+--+--+--+
# 1           |  4  |  8  |  12 | 16 ...
#          +--+--+--+--+--+--+--+--+

### -------------------------------------------

def meantimer(adf):
    global nEvSl, iEvSl
    iEvSl += 1
    if args.verbose == 1 and iEvSl % 100 == 0: print("Running meantimer [%.2f %%]" % (100.*iEvSl/nEvSl), end='\r')
    
    adf = adf.drop_duplicates()
    tzeros, angles = meantimer_results(adf)
    adf['TM0'] = np.mean(tzeros) if len(tzeros) > 0 else np.nan
    
    '''
    df = df.loc[df['SL']==sl+1]
    adf = df.drop_duplicates()
    adf = adf.loc[(adf['TDC_CHANNEL']!=139)]
    if adf.shape[0] < CHAN_PER_DF:
    out[event+sl/10.] = []
    return out[event+sl/10.]
    tzeros = meantimer_results(adf)[0]
    out[event+sl/10.] = tzeros
    return out[event+sl/10.]
    '''
    return adf

#from numba import jit
#@jit
def meantimer_results(df_hits):
    """Run meantimer over the group of hits"""
    sl = df_hits['SL'].iloc[0]
    # Getting a TIME column as a Series with TDC_CHANNEL_NORM as index
    df_time = df_hits.loc[:, ['TDC_CHANNEL_NORM', 'TIME_ABS', 'LAYER']]
    df_time.sort_values('TIME_ABS', inplace=True)
    # Split hits in groups where time difference is larger than maximum event duration
    grp = df_time['TIME_ABS'].diff().fillna(0)
    event_width_max = 1.1*TDRIFT
    grp[grp <= event_width_max] = 0
    grp[grp > 0] = 1
    grp = grp.cumsum().astype(np.int32)
    df_time['grp'] = grp
    # Removing groups with less than 3 unique hits
    df_time = df_time[df_time.groupby('grp')['TDC_CHANNEL_NORM'].transform('nunique') >= 3]
    # Determining the TIME0 using triplets [no external trigger]
    tzeros = []
    angles = []
    # Processing each group of hits
    patterns = PATTERN_NAMES.keys()
    for grp, df_grp in df_time.groupby('grp'):
        df_grp.set_index('TDC_CHANNEL_NORM', inplace=True)
        # Selecting only triplets present among physically meaningful hit patterns
        channels = set(df_grp.index.astype(np.int16))
        triplets = set(itertools.permutations(channels, 3))
        triplets = triplets.intersection(patterns)
        # Grouping hits by the channel for quick triplet retrieval
        times = df_grp.groupby(df_grp.index)['TIME_ABS']
        # Analysing each triplet
        for triplet in triplets:
            triplet_times = [times.get_group(ch).values for ch in triplet]
            for t1 in triplet_times[0]:
                for t2 in triplet_times[1]:
                    for t3 in triplet_times[2]:
                        timetriplet = (t1, t2, t3)
                        if max(timetriplet) - min(timetriplet) > 1.1*TDRIFT:
                            continue
                        pattern = PATTERN_NAMES[triplet]
                        mean_time, angle = meantimereq(pattern, timetriplet)
                        if args.verbose >= 2:
                            print('{4:d} {0:s}: {1:.0f}  {2:+.2f}  {3}'.format(pattern, mean_time, angle, triplet, sl))
                        # print(triplet, pattern, mean_time, angle)
                        #if not MEANTIMER_ANGLES[sl][0] < angle < MEANTIMER_ANGLES[sl][1]: # Override requirement as long as SL are swapped
                        if not config.FIT_ANGLES[0] < angle < config.FIT_ANGLES[1]: # Between +- 45 degrees from the vertical
                            continue
                        tzeros.append(mean_time)
                        angles.append(angle)

    return tzeros, angles

### -------------------------------------------


def recoSegments(hitlist):
    global nEvSl, iEvSl
    iEvSl += 1
    if args.verbose == 1 and iEvSl % 100 == 0: print("Running segment reconstruction [%.2f %%]" % (100.*iEvSl/nEvSl), end='\r')

    global segments, missinghits
    iorbit, isl, itime, itimePMT, itimeAND = hitlist['ORBIT'].values[0], hitlist['CHAMBER'].values[0], hitlist['T0'].values[0], hitlist['T_SCINTPMT'].values[0], hitlist['T_SCINTAND'].values[0]
    nhits = len(hitlist)
    
    if nhits < config.NHITS_LOCAL_MIN or nhits > config.NHITS_LOCAL_MAX:
        if args.verbose >= 2: print("Skipping       event", iorbit, ", chamber", isl, ", exceeds the maximum/minimum number of hits (", nhits, ")")
        return hitlist
    
    # Explicitly introduce left/right ambiguity
    lhits, rhits = hitlist.copy(), hitlist.copy()
    lhits['HIT_INDEX'] = hitlist.index
    lhits['X_LABEL'] = 1
    lhits['X'] = lhits['X_LEFT']
    rhits['HIT_INDEX'] = hitlist.index
    rhits['X_LABEL'] = 2
    rhits['X'] = rhits['X_RIGHT']
    lrhits = lhits.append(rhits, ignore_index=True) # Join the left and right hits
    
    # Compute all possible combinations in the most efficient way
    #layer_list = [list(lrhits[lrhits['LAYER'] == x + 1].index) for x in range(4)]
    layers = list(lrhits['LAYER'].unique())
    layer_list = [list(lrhits[lrhits['LAYER'] == x].index) for x in layers]
    all_combs = list(itertools.product(*layer_list))
    if args.verbose >= 2: print("Reconstructing event", iorbit, ", chamber", isl, ", has", nhits, "hits ->", len(all_combs), "combinations [%.2f %%]" % (100.*iEvSl/nEvSl)) #, end='\r'
    fitRange, fitResults = (hitlist['Z'].min() - 0.5*ZCELL, hitlist['Z'].max() + 0.5*ZCELL), []
    
    # Fitting each combination
    for comb in all_combs:
        lrcomb = lrhits.iloc[list(comb)]
        # Try to reject improbable combinations: difference between adjacent wires should be 2 or smaller
        if len(lrcomb) >= 3 and max(abs(np.diff(lrcomb['WIRE'].astype(np.int16)))) > 2: continue
        posx, posz, posl = lrcomb['X'], lrcomb['Z'], lrcomb['X_LABEL']
        seg_layer, seg_wire, seg_bx, seg_label, seg_idx = lrcomb['LAYER'].values, lrcomb['WIRE'].values, lrcomb['BX'].values, lrcomb['X_LABEL'].values, lrcomb.index
        m, q, chi2 = fitFast(posx.to_numpy(), posz.to_numpy())
        #m, q, chi2, dx = fit3D(posx.to_numpy(), posz.to_numpy(), posl.to_numpy())
        if chi2 < config.FIT_CHI2NDF_MAX and abs(m) < config.FIT_M_MAX: fitResults.append({"chi2" : chi2, "label" : seg_label, "layer" : seg_layer, "wire" : seg_wire, "bx" : seg_bx, "nhits" : len(seg_idx), "pars" : [m, q], "idx" : seg_idx})
        
    fitResults.sort(key=lambda x: x["chi2"])

    if len(fitResults) > 0:
        f_nhits, f_m, f_q, f_chi2, f_dt, f_dx = fitResults[0]["nhits"], fitResults[0]["pars"][0], fitResults[0]["pars"][1], fitResults[0]["chi2"], 0., 0.
        
        if args.tzero == 'R':
            # Refit with time
            index, labels = fitResults[0]["idx"], fitResults[0]["label"]
            selectedhits = lrhits.iloc[index]
            f_m, f_q, f_chi2, f_dx = fit3D(selectedhits['X_WIRE'].to_numpy(), selectedhits['X_DRIFT'].to_numpy(), selectedhits['X_LABEL'].to_numpy(), selectedhits['Z'].to_numpy())
            f_dt = f_dx / VDRIFT
        
        # Write results to dataframe
        segments = segments.append(pd.DataFrame.from_dict({'VIEW' : ['0'], 'ORBIT' : [iorbit], 'CHAMBER' : [(isl)], 'NHITS' : [f_nhits], 'M' : [f_m], 'SIGMAM' : [np.nan], 'Q' : [f_q], 'SIGMAQ' : [np.nan], 'CHI2' : [f_chi2], 'HIT_INDEX' : [seg_idx], 'T0' : [itime], 'T_REFIT' : [itime + f_dt], 'T_SCINTPMT' : [itimePMT], 'T_SCINTAND' : [itimeAND]}), ignore_index=True)
        for ilabel, ilayer, iwire, ibx in zip(fitResults[0]["label"], fitResults[0]["layer"], fitResults[0]["wire"], fitResults[0]["bx"]):
            x_seg = lrhits.loc[(lrhits['BX'] == ibx) & (lrhits['LAYER'] == ilayer) & (lrhits['WIRE'] == iwire) & (lrhits['X_LABEL'] == ilabel), 'Z'].values[0] * fitResults[0]["pars"][0] + fitResults[0]["pars"][1]
            hitlist.loc[(hitlist['ORBIT'] == iorbit) & (hitlist['BX'] == ibx) & (hitlist['CHAMBER'] == isl) & (hitlist['LAYER'] == ilayer) & (hitlist['WIRE'] == iwire), ['X_LABEL', 'X_SEG', 'X_SEGFITSHIFT', 'T_REFIT']] = ilabel, x_seg, f_dx, itime + f_dt

        # Missing hit interpolation
        if(fitResults[0]["nhits"] == 3):
            layers = list(hitlist['LAYER'])
            m_layer = [x for x in np.arange(1, 4+1) if not x in layers][0]
            m_zhit = mapconverter.getZlayer(m_layer)
            m_xhit =  m_zhit * fitResults[0]["pars"][0] + fitResults[0]["pars"][1]
            m_wire_num = mapconverter.getWireNumber(m_xhit, m_layer)
            missinghits = missinghits.append(pd.DataFrame.from_dict({'ORBIT' : [iorbit], 'BX' : [np.nan], 'CHAMBER' : [isl], 'LAYER' : [m_layer], 'WIRE' : [m_wire_num], 'X' : [m_xhit], 'Y' : [0.], 'Z' : [m_zhit]}), ignore_index=True)
    
        hitlist['NHITS_SEG'] = fitResults[0]["nhits"]
    
    # Do not update positions for now
    #hitlist.loc[hitlist['X_LABEL'] == 1, 'X_LEFT']  = hitlist['X_LEFT']  - hitlist['X_SEGFITSHIFT']
    #hitlist.loc[hitlist['X_LABEL'] == 2, 'X_RIGHT'] = hitlist['X_RIGHT'] + hitlist['X_SEGFITSHIFT']

    return hitlist

### -------------------------------------------

def recoTracks(hitlist):
    global nEvSl, iEvSl
    iEvSl += 1
    if args.verbose == 1 and iEvSl % 100 == 0: print("Running track reconstruction [%.2f %%]" % (100.*iEvSl/nEvSl), end='\r')

    global segments
    iorbit, itime, itimePMT, itimeAND = hitlist['ORBIT'].values[0], hitlist['T0'].values[0], hitlist['T_SCINTPMT'].values[0], hitlist['T_SCINTAND'].values[0]

    # Loop on the views (xz, yz)
    for view, sls in config.SL_FITS.items():
        #sl_ids = [sl.id for sl in sls]
        #for sl_idx in ([[str(x) for x in sl_ids]] + [[str(x)] for x in config.SL_VIEW[view] if not len(sl_ids) == 1 or not x in sl_ids]):
        for sl_idx in sls:
            f_dt, f_dx = 0., 0.
            #seg_t0 = segments.loc[(segments['ORBIT'] == iorbit) & (segments['CHAMBER'].isin(sl_idx)), 'T0'].mean() # & (np.isfinite(segments['T0'])) FIXME
            viewhits = hitlist[(hitlist['CHAMBER'].isin(sl_idx)) & (hitlist['X'].notnull())]
            nhits = len(viewhits)
            if nhits < 3: continue #*len(sl_idx)
            posxy, posz, poswire = viewhits[view[0].upper() + '_GLOB'], viewhits[view[1].upper() + '_GLOB'], viewhits[view[0].upper() + '_WIRE_GLOB']
            if args.tzero == 'R' and len(sl_idx) == max([len(x) for x in sls]): # Refit with time only the "most global" fit in each view
                # Refit with time
                m, q, chi2, f_dx = fit3D(poswire.to_numpy(), viewhits['X_DRIFT'].to_numpy(), viewhits['X_LABEL'].to_numpy(), posz.to_numpy())
                f_dt = f_dx / VDRIFT
                hitlist[[view[0].upper() + '_TRKFITSHIFT', 'T0']] = f_dx, itime + f_dt
            else:
                m, q, chi2 = fitFast(posxy.to_numpy(), posz.to_numpy())
            
            if chi2 < config.FIT_GLOBAL_CHI2NDF_MAX and abs(m) < config.FIT_M_MAX:
                segments = segments.append(pd.DataFrame.from_dict({'VIEW' : [view.upper()], 'ORBIT' : [iorbit], 'CHAMBER' : [','.join([str(x) for x in sl_idx])], 'NHITS' : [nhits], 'M' : [m], 'SIGMAM' : [np.nan], 'Q' : [q], 'SIGMAQ' : [np.nan], 'CHI2' : [chi2], 'HIT_INDEX': [list(viewhits.index)], 'T0' : [itime + f_dt], 'T_REFIT' : [itime + f_dt], 'T_SCINTPMT' : [itimePMT], 'T_SCINTAND' : [itimeAND]}), ignore_index=True)
                hitlist['NHITS_TRACK'] = nhits
                
                if len(sl_idx) > 1: # Avoid overwriting track residues in case only one SL is fitted
                    sl_idxr = config.SL_VIEW[view]
                    for isl in sl_idxr:
                        for ilayer in range(1, 4+1):
                            mask = (hitlist['CHAMBER'] == isl) & (hitlist['LAYER'] == ilayer)
                            hitlist.loc[mask, ['X_TRACK_GLOB', 'Y_TRACK_GLOB']] = np.array([0., 0.])
                            hitlist.loc[mask, view[0].upper() + '_TRACK_GLOB'] = m * hitlist.loc[mask, view[1].upper() + '_GLOB'].values + q
                                
        # Update positions according to new refitted time
        hitlist.loc[hitlist['X_LABEL'] == 1, view[0].upper() + '_GLOB'] = hitlist[view[0].upper() + '_GLOB'] - hitlist[view[0].upper() + '_TRKFITSHIFT']
        hitlist.loc[hitlist['X_LABEL'] == 2, view[0].upper() + '_GLOB'] = hitlist[view[0].upper() + '_GLOB'] + hitlist[view[0].upper() + '_TRKFITSHIFT']
                            
    return hitlist

### -------------------------------------------

def fix_orbit(orbit_arr):
    new_col = np.zeros_like(orbit_arr, dtype=np.int64)
    for i in range(len(new_col)):
        orbit = orbit_arr[i]
        if (np.isnan(orbit) or orbit==0) and i!=0:
            new_col[i] = new_col[i-1]
        elif (np.isnan(orbit) or orbit==0) and i==0:
            new_col[i] = 0
        else:
            new_col[i] = orbit
    return new_col

### -------------------------------------------


itime = datetime.now()
if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Starting script, importing dataset...")


################################################## PART 1: Read file(s) and map channels

df = pd.DataFrame()

for i, filename in enumerate(args.filenames):
    if args.verbose == 1: print("Reading dataset(s) [%.2f %%]" % (100.*i/len(args.filenames)), end='\r')

    if filename.endswith('.dat'):
        from modules.unpacker import *
        unpk = Unpacker()
        inputFile = open(filename, 'rb')
        dt = unpk.unpack(inputFile, args.max, args.flush if i == 0 else False)
        if args.verbose >= 1 and i == 0 and args.flush: print("[ INFO ] Skipping DMA flush at the beginning of the run")
        df = df.append(pd.DataFrame.from_dict(dt), ignore_index=True)
        
    elif filename.endswith('.txt') or filename.endswith('.csv'):
        # force 7 fields
        # in case of forcing, for some reason, the first raw is not interpreted as columns names
        dt = pd.read_csv(filename, \
            names=['HEAD', 'FPGA', 'TDC_CHANNEL', 'ORBIT_CNT', 'BX_COUNTER', 'TDC_MEAS'], \
            dtype={'HEAD' : 'int32', 'FPGA' : 'int32', 'TDC_CHANNEL' : 'int32', 'ORBIT_CNT' : 'int32', 'BX_COUNTER' : 'int32', 'TDC_MEAS' : 'float64'}, \
            low_memory=False, \
            skiprows=1, \
            nrows=args.max*1024 + 1 if args.max > 0 else 1e9, \
        )
        df = df.append(dt, ignore_index=True)
        df.loc[(df['HEAD'] == 0) | (df['HEAD'] == 2), 'TDC_MEAS'] = df.loc[(df['HEAD'] == 0) | (df['HEAD'] == 2), 'TDC_MEAS'] - 1 # Correct TDC as the input is in the [1, 30] range

    else:
        print("File format not recognized, skipping file...")

# Determine length of the run
runTime = (df['ORBIT_CNT'].max() - df['ORBIT_CNT'].min()) * DURATION['orbit'] * 1.e-9 # Approximate acquisition time in seconds

if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Read %d lines from %d file(s)." % (len(df), len(args.filenames)), "Duration of the run:\t%d s" % runTime)



# Swap channels
#df.loc[(df['FPGA'] == 0) & (df['TDC_CHANNEL'] == 109), 'TDC_CHANNEL'] = -9
#df.loc[(df['FPGA'] == 0) & (df['TDC_CHANNEL'] == 110), 'TDC_CHANNEL'] = 109
#df.loc[(df['FPGA'] == 0) & (df['TDC_CHANNEL'] == -9), 'TDC_CHANNEL'] = 110
# FIXME: SL 0 and 1 are swapped
#pd.options.mode.chained_assignment = None
#df.loc[df['SL'] == 0, 'SL'] = -1
#df.loc[df['SL'] == 1, 'SL'] = 0
#df.loc[df['SL'] == -1, 'SL'] = 1
df = df[df['TDC_CHANNEL'] < 135] # Remove tdc_channel = 139 since they are not physical events
df = df[df['BX_COUNTER'] < 4095] # Remove BX == 4095 as they do not have meaningful info and spoil the trigger T
if args.event > 0: df = df[df['ORBIT_CNT'] == args.event]

if len(df) == 0:
    print("Empty dataframe, exiting...")
    exit()

if args.verbose >= 2: print("DF:\n", df.head(50), "\n")

### Split dataframes according to their content
triggers = df[(df['HEAD'] == 4) | (df['HEAD'] == 5)].copy() # Trigger parameters
df = df.astype({'TDC_MEAS' : int}, errors='ignore') # Cast TDC_MEAS to int, as they are used in trigger t0, scrint t0, and hits
df['TIME'] = df['BX_COUNTER'] * DURATION['bx'] + df['TDC_MEAS'] / 30 * DURATION['bx'] # Use time instead of BX. This is valid for both trigger and scintillator, as they use the same columns
trigger_time = df[df['HEAD']==0].copy() # Trigger times
scintAND_time = df[(df['FPGA']==1) & (df['TDC_CHANNEL']==128)].copy() # Scintillator SiPM AND VME
scintEXT_time = df[(df['FPGA']==1) & (df['TDC_CHANNEL']==129)].copy() # Scintillator SiPM external times
scintINT_time = df[(df['FPGA']==0) & (df['TDC_CHANNEL']==128)].copy() # Scintillator SiPM AND NIM
scintPMT_time = df[(df['FPGA']==0) & (df['TDC_CHANNEL']==129)].copy() # Scintillator PMT AND times
df = df[(df['HEAD'] == 2) & (df['TDC_CHANNEL'] < 128)] # Physics hits

# Map TDC_CHANNEL, FPGA to SL, LAYER, WIRE_NUM, WIRE_POS
mapconverter = Mapping()
df = mapconverter.virtex7(df)
#df = mapconverter.virtex7obdt(df)

if args.verbose >= 2: print("DF:\n", df.head(50), "\n")

# Save occupancy numbers before any further selection
occupancy = df.groupby(['SL', 'LAYER', 'WIRE_NUM'])['HEAD'].count() # count performed a random column
occupancy = occupancy.reset_index().rename(columns={'HEAD' : 'COUNTS'}) # reset the indices and make new ones, because the old indices are needed for selection
occupancy['RATE'] = occupancy['COUNTS'] / runTime

if args.verbose >= 2:
    print("Occupancy:\n", occupancy.head(10), "\n")

if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Channel mapping completed")

################################################## PART 2: Trigger parameters

triggers = triggers[(triggers['FPGA'] == 0) & (triggers['TDC_CHANNEL'] >= 5)]
triggers['CHAMBER'] = triggers['FPGA'] * 2 + triggers['TDC_CHANNEL'] // 5
# Remove consecutive words
triggers['TDC_MEAS'] = triggers['TDC_MEAS'].loc[triggers['TDC_MEAS'].shift() != triggers['TDC_MEAS']]
triggers['HEAD'] = triggers['HEAD'].loc[triggers['HEAD'].shift() != triggers['HEAD']]
triggers = triggers[(triggers['HEAD'].notna()) | (triggers['TDC_MEAS'].notna())]
# Make parameters one-liner
triggers.loc[triggers['HEAD'] == 4, 'M'] = triggers[triggers['HEAD'] == 4]['TDC_MEAS']
triggers.loc[triggers['HEAD'] == 5, 'Q'] = triggers[triggers['HEAD'] == 5]['TDC_MEAS']
triggers[['M', 'Q']] = triggers.groupby('ORBIT_CNT')[['M', 'Q']].transform(np.max)
triggers = triggers.drop_duplicates(subset=['ORBIT_CNT', 'M', 'Q'], keep='first')
# Add macrocell offset
triggers['X0'] = - triggers['Q'] / triggers['M'] # local q coordinate in the standard x-z reference frame
triggers[['X0_GLOB', 'Q_GLOB']] = [np.nan, np.nan]
#triggers['Q'] = triggers['Q'] #+ XCELL * (triggers['TDC_CHANNEL'] % 5 - 2 + config.TRIGGER_CELL_OFFSET)
# Proper trigger dataframe formatting
triggers = triggers.rename(columns={'ORBIT_CNT' : 'ORBIT'})
triggers[['VIEW', 'NHITS', 'SIGMAM', 'SIGMAQ', 'CHI2', 'HIT_INDEX', 'T0']] = [config.PHI_VIEW.upper(), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
triggers = triggers[['VIEW', 'ORBIT', 'CHAMBER', 'NHITS', 'M', 'SIGMAM', 'Q', 'SIGMAQ', 'Q_GLOB', 'X0', 'X0_GLOB', 'CHI2', 'HIT_INDEX', 'T0']]
if args.verbose >= 2: print("Triggers:\n", triggers.head(50), "\n")

# remove double hits
##df['TDC_MEAS'] = df.groupby(['HEAD', 'FPGA', 'ORBIT_CNT', 'TDC_CHANNEL'])['TDC_MEAS'].transform(np.max)
##df = df.drop_duplicates(subset=['HEAD', 'FPGA', 'ORBIT_CNT', 'TDC_CHANNEL'], keep='last')
#df = df.drop_duplicates(subset=['HEAD', 'FPGA', 'ORBIT_CNT', 'TDC_CHANNEL'], keep='first')

if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Trigger parameters completed")

################################################## PART 3: Times

# Calibrate hit times
df['TIME'] = df['TIME'] + df['SL'].map(config.TIME_OFFSET_SL) + df['LAYER'].map(config.TIME_OFFSET_LAYER)

# Calibrate scintillator times
scintAND_time['TIME'] = scintAND_time['TIME'] + config.TIME_OFFSET_SCINTAND
scintEXT_time['TIME'] = scintEXT_time['TIME'] + config.TIME_OFFSET_SCINTEXT
scintINT_time['TIME'] = scintINT_time['TIME'] + config.TIME_OFFSET_SCINTINT
scintPMT_time['TIME'] = scintPMT_time['TIME'] + config.TIME_OFFSET_SCINTPMT

# Create scintillator AND time map
scintAND_time['TIME'] = scintAND_time.groupby('ORBIT_CNT')['TIME'].transform(np.nanmin)
scintAND_time_map = dict(scintAND_time.loc[scintAND_time['TIME'].notna(), ['ORBIT_CNT', 'TIME']].to_records(index=False))

# Create scintillator EXT time map
scintEXT_time['TIME'] = scintEXT_time.groupby('ORBIT_CNT')['TIME'].transform(np.nanmin)
scintEXT_time_map = dict(scintEXT_time.loc[scintEXT_time['TIME'].notna(), ['ORBIT_CNT', 'TIME']].to_records(index=False))

# Create scintillator INT time map
scintINT_time['TIME'] = scintINT_time.groupby('ORBIT_CNT')['TIME'].transform(np.nanmin)
scintINT_time_map = dict(scintINT_time.loc[scintINT_time['TIME'].notna(), ['ORBIT_CNT', 'TIME']].to_records(index=False))

# Create scintillator PMT time map
scintPMT_time['TIME'] = scintPMT_time.groupby('ORBIT_CNT')['TIME'].transform(np.nanmin)
scintPMT_time_map = dict(scintPMT_time.loc[scintPMT_time['TIME'].notna(), ['ORBIT_CNT', 'TIME']].to_records(index=False))

# Create trigger map and update trigger dataframe with trigger times
trigger_time['TIME'] = trigger_time.groupby('ORBIT_CNT')['TIME'].transform(np.nanmin)
trigger_time_map = dict(trigger_time.loc[trigger_time['TIME'].notna(), ['ORBIT_CNT', 'TIME']].to_records(index=False))

# Update hits and trigger dataframes
df['T_SCINTAND'] = df['ORBIT_CNT'].map(scintAND_time_map)
df['T_SCINTEXT'] = df['ORBIT_CNT'].map(scintEXT_time_map)
df['T_SCINTINT'] = df['ORBIT_CNT'].map(scintINT_time_map)
df['T_SCINTPMT'] = df['ORBIT_CNT'].map(scintPMT_time_map)
df['T_TRIGGER'] = df['ORBIT_CNT'].map(trigger_time_map)
triggers['T0'] = triggers['ORBIT'].map(trigger_time_map)
df['T_MEANT'] = np.nan

# Estimate scintillator trigger rate
if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Scintillator AND rate:\t%.4f Hz" % (len(scintAND_time)/runTime))
if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Scintillator EXT rate:\t%.4f Hz" % (len(scintEXT_time)/runTime))
if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Scintillator INT rate:\t%.4f Hz" % (len(scintINT_time)/runTime))
if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Scintillator PMT rate:\t%.4f Hz" % (len(scintPMT_time)/runTime))
if args.verbose >= 1: print("[", datetime.now() - itime, "]", "FPGA trigger     rate:\t%.4f Hz" % (len(trigger_time_map)/runTime))
if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Time mapping completed, found", len(triggers['ORBIT'].unique()), "triggers")

# Step 3: T0 is the one that will be effectively used to determine TDRIFT
if args.tzero == 'T': df['T0'] = df['T_TRIGGER']
elif args.tzero in ['S']: df['T0'] = df['T_SCINTAND']
elif args.tzero in ['P', 'R']: df['T0'] = df['T_SCINTPMT']
else:
    print("T0 determination method not recognized, exiting...")
    exit()
#df[['T0', 'T_TRIGGER', 'T_SCINT']] = df.groupby('ORBIT_CNT')[['T0', 'T_TRIGGER', 'T_SCINT']].transform(np.max)

# Determine BX0 either using meantimer or the trigger BX assignment
if args.tzero == 'M':
    mtime = datetime.now()

    # Add necessary columns
    #df['TDC_CHANNEL_NORM'] = (df['TDC_CHANNEL'] - 64 * (df['SL']%2)).astype(np.uint8)
    df['TIME_ABS'] = ((df['ORBIT_CNT'] - df['ORBIT_CNT'].min()).astype(np.float64)*DURATION['orbit'] + df['BX_COUNTER'].astype(np.float64)*DURATION['bx'] + df['TDC_MEAS'].astype(np.float64)*DURATION['tdc']).astype(np.float64)

    # Group by orbit counter (event) and SL    
    df = df.groupby(['ORBIT_CNT', 'SL'], as_index=False).apply(meantimer)

    # Overwrite BX assignment
    #df['BX_MEANT'] = (df['TIME_ABS'] - (df['TIME_ABS'] // DURATION['orbit']).astype(np.int32)*DURATION['orbit']) / DURATION['bx']
    df['T_MEANT'] = df['TM0'] - (df['ORBIT_CNT'] - df['ORBIT_CNT'].min())*DURATION['orbit']
    df.loc[df['T0'].isna(), 'T0'] = df.loc[df['T0'].isna(), 'T_MEANT'].astype(np.float64) # Use meantimer only if not trigger BX0 assignemnt
    df.drop(columns=['TM0', 'TIME_ABS'], inplace=True)
    df.reset_index(inplace=True)
    if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Meantimer completed")

# Step 4: create dataframe of times

times = pd.DataFrame.from_dict(trigger_time_map.items()).rename(columns={0 : 'ORBIT', 1 : 'T_TRIG'})
times['T_SCINTAND'] = times['ORBIT'].map(scintAND_time_map)
times['T_SCINTEXT'] = times['ORBIT'].map(scintEXT_time_map)
times['T_SCINTINT'] = times['ORBIT'].map(scintINT_time_map)
times['T_SCINTPMT'] = times['ORBIT'].map(scintPMT_time_map)

if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Time assignment completed")

################################################## PART 4: Cleaning and position assignment

# Select only hits with time information
losthits = df.loc[df['T0'].isnull()].copy()
hits = df.loc[df['T0'].notna()].copy()

if len(hits) == 0:
    print("No entry with non-NaN T0, exiting...")
    exit()

# Create column TDRIFT
hits['TDRIFT'] = hits['TIME'] - hits['T0']

# Find events
hits = hits[(hits['TDRIFT']>config.TIMEBOX[0]) & (hits['TDRIFT']<config.TIMEBOX[1])]

# Count hits in each event
hits['NHITS'] = hits.groupby('ORBIT_CNT')['HEAD'].transform(np.size)

# Conversion from time to position
#mapconverter.addXleftright(hits)
hits['X_DRIFT'] = np.minimum(np.maximum(hits['TDRIFT'], 0) * VDRIFT, XCELL/2.)
hits['X_LEFT']  = hits['WIRE_POS'] - hits['X_DRIFT']
hits['X_RIGHT'] = hits['WIRE_POS'] + hits['X_DRIFT']

# Cosmetic changes to be compliant with common format
hits.rename(columns={'ORBIT_CNT': 'ORBIT', 'BX_COUNTER': 'BX', 'SL' : 'CHAMBER', 'WIRE_NUM' : 'WIRE', 'Z_POS' : 'Z', 'WIRE_POS' : 'X_WIRE', 'TDRIFT' : 'TIMENS'}, inplace=True)

#if args.verbose >= 2: print(hits[hits['TDC_CHANNEL'] >= -128].head(50))

if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Cleaning completed")


################################################## PART 5: Reconstruction

# Reconstruction
events = hits[['ORBIT', 'BX', 'NHITS', 'CHAMBER', 'LAYER', 'WIRE', 'X_WIRE', 'X_DRIFT', 'X_LEFT', 'X_RIGHT', 'Z', 'TIMENS', 'TDC_MEAS', 'T0', 'T_TRIGGER', 'T_SCINTAND', 'T_SCINTPMT', 'T_MEANT']].copy()

events[['X_LABEL', 'NHITS_SEG', 'NHITS_TRACK', 'Y', 'X_SEGFITSHIFT', 'X_TRKFITSHIFT', 'Y_TRKFITSHIFT', 'T_REFIT']] = [0, 0, 0,  0., 0., 0., 0., np.nan]
events[['X', 'X_SEG', 'X_TRACK']] = [np.nan, np.nan, np.nan]
events[['X_TRACK_GLOB', 'Y_TRACK_GLOB']] = [np.nan, np.nan]
events[['X_GLOB', 'Y_GLOB', 'Z_GLOB']] = [np.nan, np.nan, np.nan]

missinghits = pd.DataFrame(columns=['ORBIT', 'BX', 'CHAMBER', 'LAYER', 'WIRE', 'X', 'Y', 'Z'])

# Add number of tappino (c)
#mapconverter.addTappinoNum(events)

# Segments reference frame is not the usual x-z frame, instead the angle is measured from the vertical, rotation clockwise is +, anticlockwise is -
segments = pd.DataFrame(columns=['VIEW', 'ORBIT', 'CHAMBER', 'NHITS', 'M', 'SIGMAM', 'Q', 'SIGMAQ', 'CHI2', 'HIT_INDEX', 'T0', 'T_REFIT', 'T_SCINTAND', 'T_SCINTPMT'])

# Reconstruction
from modules.geometry.sl import SL
from modules.geometry.segment import Segment
from modules.geometry import Geometry, COOR_ID
from modules.analysis import config as CONFIGURATION

# Initialize geometry
G = Geometry(CONFIGURATION)
SLs = {}
for iSL in config.SL_SHIFT.keys():
    SLs[iSL] = SL(iSL, config.SL_SHIFT[iSL], config.SL_ROTATION[iSL])

# Defining which SLs should be plotted (and fitted) in which global view
GLOBAL_VIEW_SLs = {}
for view in ['xz', 'yz']: GLOBAL_VIEW_SLs[view] = [SLs[x] for x in config.SL_VIEW[view]]
#    'xz': [SLs[0], SLs[2]],
#    'yz': [SLs[1], SLs[3]]

if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Reconstructing segments")

# Reset counters
nEvSl, iEvSl = len(events.groupby(['ORBIT', 'CHAMBER'])), 0

if not args.noreco:
    if False: #args.parallel:
        from pandarallel import pandarallel
        pandarallel.initialize()
        events = events.groupby(['ORBIT', 'CHAMBER'], as_index=False).parallel_apply(recoSegments)
        del pandarallel
    else:
        events = events.groupby(['ORBIT', 'CHAMBER'], as_index=False).apply(recoSegments)

events.loc[events['X_LABEL'] == 1, 'X'] = events['X_LEFT']
events.loc[events['X_LABEL'] == 2, 'X'] = events['X_RIGHT']

if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Adding global positions")

# Updating global positions
for iSL, sl in SLs.items():
    slmask = events['CHAMBER'] == iSL
    events.loc[slmask, ['X_GLOB', 'Y_GLOB', 'Z_GLOB']] = sl.coor_to_global(events.loc[slmask, ['X', 'Y', 'Z']].values)
    events.loc[slmask, ['X_LEFT_GLOB', 'Y_LEFT_GLOB', 'Z_LEFT_GLOB']] = sl.coor_to_global(events.loc[slmask, ['X_LEFT', 'Y', 'Z']].values)
    events.loc[slmask, ['X_RIGHT_GLOB', 'Y_RIGHT_GLOB', 'Z_RIGHT_GLOB']] = sl.coor_to_global(events.loc[slmask, ['X_RIGHT', 'Y', 'Z']].values)
    events.loc[slmask, ['X_WIRE_GLOB', 'Y_WIRE_GLOB', 'Z_WIRE_GLOB']] = sl.coor_to_global(events.loc[slmask, ['X_WIRE', 'Y', 'Z']].values)
    # missing hits
    slmask = missinghits['CHAMBER'] == iSL
    missinghits.loc[slmask, ['X_GLOB', 'Y_GLOB', 'Z_GLOB']] = sl.coor_to_global(missinghits.loc[slmask, ['X', 'Y', 'Z']].values)
    # triggers
    trmask = triggers['CHAMBER'] == iSL
    ntrmask = len(triggers[trmask])
    triggers.loc[trmask, 'X0_GLOB'] = sl.coor_to_global(np.vstack((np.zeros(ntrmask), np.zeros(ntrmask), triggers.loc[trmask, 'X0'])).T)[:,2] # transform to global standard reference frame
# Transform back to segment reference frame
triggers['Q_GLOB'] = -triggers['M'] * triggers['X0_GLOB']


if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Reconstructing tracks")

# Reset counters
nEvSl, iEvSl = len(events.groupby(['ORBIT'])), 0

if not args.noreco:
    if False: #args.parallel:
        from pandarallel import pandarallel
        pandarallel.initialize()
        events = events.groupby('ORBIT', as_index=False).parallel_apply(recoTracks)
        del pandarallel
    else:
        events = events.groupby('ORBIT', as_index=False).apply(recoTracks)

# Updating global positions for track reconstructed positions
for iSL, sl in SLs.items():
    slmask = events['CHAMBER'] == iSL
    events.loc[slmask, 'X_TRACK'] = sl.coor_to_local(events.loc[slmask, ['X_TRACK_GLOB', 'Y_TRACK_GLOB', 'Z_GLOB']].values)[:,0]

# Update times dataframe with refitted time
refit_time_map = dict(segments.loc[(segments['VIEW'] == config.PHI_VIEW.upper()) & (segments['CHAMBER'] == str(config.SL_AUX_STR)), ['ORBIT', 'T_REFIT']].to_records(index=False))
times['T_REFIT'] = times['ORBIT'].map(refit_time_map)

rtime = datetime.now()
if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Reconstruction completed")



################################################## PART 6: Trigger-segment matching

# Match times
#fit_time['TIME'] = trigger_time.groupby('ORBIT_CNT')['TIME'].transform(np.nanmin)
#print(events[['ORBIT', 'T0']].head(60))
#fit_time_map = dict(events.loc[events['T0'].notna(), ['ORBIT', 'T0']].to_records(index=False))

# Get trigger dict for M and Q and add them to segments
trigger_m_map = dict(triggers.loc[triggers['M'].notna(), ['ORBIT', 'M']].to_records(index=False))
trigger_qloc_map = dict(triggers.loc[triggers['Q'].notna(), ['ORBIT', 'Q']].to_records(index=False))
trigger_qglo_map = dict(triggers.loc[triggers['Q_GLOB'].notna(), ['ORBIT', 'Q_GLOB']].to_records(index=False))
# Add trigger time, m, q to segments
segments['T_TRIG'] = segments['ORBIT'].map(trigger_time_map)
segments['M_TRIG'] = segments['ORBIT'].map(trigger_m_map)
segments['Q_TRIG'] = segments['ORBIT'].map(trigger_qloc_map)
segments['Q_TRIG_GLOB'] = segments['ORBIT'].map(trigger_qglo_map)

# Use the convention where 0 is vertical
triggers['M_RAD'] = np.arctan(triggers['M'])
segments['M_RAD'] = np.arctan(segments['M'])
segments['M_TRIG_RAD'] = np.arctan(segments['M_TRIG'])

#triggers['M_RAD'] = np.where(triggers['M_RAD'] < 0, triggers['M_RAD'] + math.pi/2., triggers['M_RAD'] - math.pi/2.)
#segments['M_RAD'] = np.where(segments['M_RAD'] < 0, segments['M_RAD'] + math.pi/2., segments['M_RAD'] - math.pi/2.)
#segments['M_TRIG_RAD'] = np.where(segments['M_TRIG_RAD'] < 0, segments['M_TRIG_RAD'] + math.pi/2., segments['M_TRIG_RAD'] - math.pi/2.)

triggers['M_DEG'] = np.degrees(triggers['M_RAD'])
segments['M_DEG'] = np.degrees(segments['M_RAD'])
segments['M_TRIG_DEG'] = np.degrees(segments['M_TRIG_RAD'])

#triggers['X0'] = - triggers['Q'] / triggers['M']
#segments['X0'] = - segments['Q'] / segments['M']
#segments['X0_TRIG'] = - segments['Q_TRIG'] / segments['M_TRIG']

################################################## PART 7: Write output

if args.verbose >= 2:
    print("Events:\n", events.head(50), "\n")
    print("Missing hits:\n", missinghits.head(10), "\n")
    print("Segments head:\n", segments.head(10), "\n")
    print("Segments tail:\n", segments.tail(10), "\n")
    print("Times:\n", times.head(10), "\n")
    print("Triggers:\n", triggers.head(10), "\n")

# Output to csv files
if len(args.outputdir) > 0 and not args.nosave:
    events.to_csv(args.outputdir + runname + "_csv/events" + args.suffix + ".csv", header=True, index=False)
    missinghits.to_csv(args.outputdir + runname + "_csv/missinghits" + args.suffix + ".csv", header=True, index=False)
    segments.to_csv(args.outputdir + runname + "_csv/segments" + args.suffix + ".csv", header=True, index=False)
    triggers.to_csv(args.outputdir + runname + "_csv/triggers" + args.suffix + ".csv", header=True, index=False)
    occupancy.to_csv(args.outputdir + runname + "_csv/occupancy" + args.suffix + ".csv", header=True, index=False)
    times.to_csv(args.outputdir + runname + "_csv/times" + args.suffix + ".csv", header=True, index=False)
    
    if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Output files saved in directory", args.outputdir + runname + "_csv/")
else:
    if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Output directory not specified, no file saved")


################################################## PART 8: Event display

# Event display
import bokeh

evs = events.groupby(['ORBIT'])

ndisplays = 0

# Loop on events (same orbit)
for orbit, hitlist in evs:
    ndisplays += 1
    if ndisplays > args.display: break
    if args.verbose >= 1: print("Drawing event", orbit, "with", len(hitlist), "hits and", len(triggers[triggers['ORBIT'] == orbit]), "triggers ...")
    # Creating figures of the chambers
    figs = {}
    figs['sl'] = plot.book_chambers_figure(G)
    figs['global'] = plot.book_global_figure(G, GLOBAL_VIEW_SLs)
    # Draw chamber
    for iSL, sl in SLs.items():
        # Hits
        hitsl = hitlist[hitlist['CHAMBER'] == iSL]
        m_hitsl = missinghits[(missinghits['ORBIT'] == orbit) & (missinghits['CHAMBER'] == iSL)]
        figs['sl'][iSL].circle(x=hitsl['X_LEFT'], y=hitsl['Z'], size=5, fill_color='red', fill_alpha=0.5, line_width=0)
        figs['sl'][iSL].circle(x=hitsl['X_RIGHT'], y=hitsl['Z'], size=5, fill_color='green', fill_alpha=0.5, line_width=0)
        figs['sl'][iSL].circle(x=hitsl['X'], y=hitsl['Z'], size=5, fill_color='black', fill_alpha=1., line_width=0)
        if len(m_hitsl) > 0: figs['sl'][iSL].cross(x=m_hitsl['X'], y=m_hitsl['Z'], size=10, line_color='royalblue', fill_alpha=0.7, line_width=2, angle=0.785398)
        
        # Segments
        if len(segments) <= 0: continue
        segsl = segments[(segments['VIEW'] == '0') & (segments['ORBIT'] == orbit) & (segments['CHAMBER'] == iSL)]
        for index, seg in segsl.iterrows():
            #col = config.TRACK_COLORS[iR]
            segz = [G.SL_FRAME['b'], G.SL_FRAME['t']]
            segx = [z * seg['M'] + seg['Q'] for z in segz]
            figs['sl'][iSL].line(x=np.array(segx), y=np.array(segz), line_color='black', line_alpha=0.7, line_width=3)

        # Triggers
        trisl = triggers[(triggers['ORBIT'] == orbit) & (triggers['CHAMBER'] == iSL)]
        if len(trisl) <= 0: continue
        for index, tri in trisl.iterrows():
            #col = config.TRACK_COLORS[iR]
            triz = [G.SL_FRAME['b'], G.SL_FRAME['t']]
            trix = [z * tri['M'] + tri['Q'] for z in triz]
            figs['sl'][iSL].line(x=np.array(trix), y=np.array(triz), line_color='magenta', line_alpha=0.7, line_width=3)


    # Global points
    for view, sls in GLOBAL_VIEW_SLs.items():
        sl_ids = [sl.id for sl in sls]
        viewhits = hitlist[hitlist['CHAMBER'].isin(sl_ids)]
        m_viewhits = missinghits[(missinghits['ORBIT'] == orbit) & (missinghits['CHAMBER'].isin(sl_ids))]
        figs['global'][view].circle(x=viewhits[view[0].upper() + '_LEFT_GLOB'], y=viewhits[view[1].upper() + '_LEFT_GLOB'], fill_color='red', fill_alpha=0.5, line_width=0)
        figs['global'][view].circle(x=viewhits[view[0].upper() + '_RIGHT_GLOB'], y=viewhits[view[1].upper() + '_RIGHT_GLOB'], fill_color='green', fill_alpha=0.5, line_width=0)
        figs['global'][view].circle(x=viewhits[view[0].upper() + '_GLOB'], y=viewhits[view[1].upper() + '_GLOB'], fill_color='black', fill_alpha=1., line_width=0)
        if len(m_viewhits) > 0: figs['global'][view].cross(x=m_viewhits[view[0].upper() + '_GLOB'], y=m_viewhits[view[1].upper() + '_GLOB'], size=10, line_color='royalblue', fill_alpha=0.7, line_width=2, angle=0.785398)

        # Tracks
        if len(segments) <= 0: continue
        tracks = segments[(segments['VIEW'] == view.upper()) & (segments['ORBIT'] == orbit) & (segments['CHAMBER'].str.len() > (1 if len(sls) > 1 else 0))]
        for index, trk in tracks.iterrows():
            trkz = [config.PLOT_RANGE['y'][0] + 1, config.PLOT_RANGE['y'][1] - 1]
            trkxy = [z * trk['M'] + trk['Q'] for z in trkz]
            figs['global'][view].line(x=np.array(trkxy), y=np.array(trkz), line_color='black', line_alpha=0.7, line_width=3)

        # Segments
        for isl in sl_ids:
            segsl = segments[(segments['VIEW'] == view.upper()) & (segments['ORBIT'] == orbit) & (segments['CHAMBER'] == str(isl))]
            for index, seg in segsl.iterrows():
                segz = [config.PLOT_RANGE['y'][0] + 1, config.PLOT_RANGE['y'][1] - 1]
                segx = [z * seg['M'] + seg['Q'] for z in segz]
                figs['global'][view].line(x=np.array(segx), y=np.array(segz), line_color='gray', line_alpha=0.3, line_width=2, line_dash='dashed')

        # Triggers
        trigs = triggers[(triggers['ORBIT'] == orbit) & (triggers['VIEW'] == view.upper())]
        for index, tri in trigs.iterrows():
            #q_local = - tri['Q'] / tri['M'] # local q coordinate in the standard x-z reference frame
            #q_global = sls[ sl_ids.index(tri['CHAMBER']) ].coor_to_global(np.array([[0., 0., q_local]]))[0][2] # transform to global standard reference frame
            #q_global = -tri['M'] * q_global # transform back to segment reference frame
            triz = [config.PLOT_RANGE['y'][0] + 1, config.PLOT_RANGE['y'][1] - 1]
            trixy = [z * tri['M'] + tri['Q_GLOB'] for z in triz]
            figs['global'][view].line(x=np.array(trixy), y=np.array(triz), line_color='magenta', line_alpha=0.7, line_width=3)


    plots = [[figs['sl'][l]] for l in [3, 2, 1, 0]]
    plots.append([figs['global'][v] for v in ['xz', 'yz']])
    if len(args.outputdir) > 0:
        bokeh.io.output_file(args.outputdir + "/" + runname + "_display/orbit_%d.html" % orbit, mode='cdn') #args.outputdir
        bokeh.io.save(bokeh.layouts.layout(plots))
        #bokeh.io.export_png(bokeh.layouts.layout(plots), filename="output/" + runname + "_display/orbit_%d.png" % orbit)
        if args.verbose >= 2: print("Event display number", orbit, "saved in", args.outputdir + runname + "_display/")


if args.verbose >= 1: print("[", datetime.now() - itime, "]", "Done.")


# python3 analysis.py -i data/Run000966/output_raw.dat -m 1 -v
