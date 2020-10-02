#!/usr/bin/env python
# coding: utf-8

import os, sys, math
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

from modules.mapping.config import TDRIFT, VDRIFT, DURATION, TIME_WINDOW
from modules.mapping import *
from modules.analysis.patterns import PATTERNS, PATTERN_NAMES, ACCEPTANCE_CHANNELS, MEAN_TZERO_DIFF, MEANTIMER_ANGLES, meantimereq, mean_tzero, tzero_clusters

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-e", "--eventdisplay", action="store_true", default=False, dest="eventdisplay", help="Run event display (loop on events, takes time)")
parser.add_option("-i", "--inputfile", action="store", type="string", dest="filename", default="data/Run000966/output_raw.dat", help="Provide input file (either binary or txt)")
parser.add_option("-o", "--outputdir", action="store", type="string", dest="outputdir", default="./output/", help="Specify output directory")
parser.add_option("-m", "--max", action="store", type=int, default=-1, dest="max", help="Maximum number of words to be read")
parser.add_option("-x", "--meantimer", action="store_true", default=False, dest="meantimer", help="Force application of the meantimer algorithm (override BX assignment)")
parser.add_option("-v", "--verbose", action="store_true", default=False, dest="verbose", help="Increase verbosity")
(options, args) = parser.parse_args()

runname = [x for x in options.filename.split('/') if 'Run' in x][0] if "Run" in options.filename else "Run000000"

if not os.path.exists(options.outputdir): os.makedirs(options.outputdir)
if not os.path.exists(options.outputdir + runname + "_plots/"): os.makedirs(options.outputdir + runname + "_plots/")
if not os.path.exists(options.outputdir + runname + "_events/"): os.makedirs(options.outputdir + runname + "_events/")

# Copy data from LNL to CERN
# scp /data/Run001089/*.dat zucchett@lxplus.cern.ch:/afs/cern.ch/work/z/zucchett/public/FortyMHz/Run001089/
# Copy data from CERN to local
# mkdir data/Run001089
# scp lxplus.cern.ch:/afs/cern.ch/work/z/zucchett/public/FortyMHz/Run001089/* data/Run001089/


# Layer    # Parameters

#          +--+--+--+--+--+--+--+--+
# 1        |  1  |  5  |  9  |  13 | 17 ...
#          +--+--+--+--+--+--+--+--+
# 2           |  3  |  7  |  11 | 15 ...
#          +--+--+--+--+--+--+--+--+
# 3        |  2  |  6  |  10 |  14 | 18 ...
#          +--+--+--+--+--+--+--+--+
# 4           |  4  |  8  |  12 | 16 ...
#          +--+--+--+--+--+--+


def meantimer(adf):
    adf = adf.drop_duplicates() 
    tzeros, angles = meantimer_results(adf)
    adf['T0'] = np.mean(tzeros) if len(tzeros) > 0 else np.nan
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
def meantimer_results(df_hits, verbose=False):
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
    grp = grp.cumsum().astype(np.uint16)
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
                        if verbose:
                            print('{4:d} {0:s}: {1:.0f}  {2:+.2f}  {3}'.format(pattern, mean_time, angle, triplet, sl))
                        # print(triplet, pattern, mean_time, angle)
                        #if not MEANTIMER_ANGLES[sl][0] < angle < MEANTIMER_ANGLES[sl][1]: # Override requirement as long as SL are swapped
                        if not -0.3 < angle < 0.3:
                            continue
                        tzeros.append(mean_time)
                        angles.append(angle)

    return tzeros, angles


itime = datetime.now()
if options.verbose: print("Starting script [", itime, "]")

### Open file ###

if options.verbose: print("Importing dataset...")

if options.filename.endswith('.dat'):
    from modules.unpacker import *
    unpk = Unpacker()
    inputFile = open(options.filename, 'rb')
    dt = unpk.unpack(inputFile, options.max)

    if options.verbose: print("Read %d lines from binary file %s" % (len(dt), options.filename))
    df = pd.DataFrame.from_dict(dt)
    
elif options.filename.endswith('.txt') or options.filename.endswith('.csv'):
    # force 7 fields
    # in case of forcing, for some reason, the first raw is not interpreted as columns names
    df = pd.read_csv(options.filename, \
        names=['HEAD', 'FPGA', 'TDC_CHANNEL', 'ORBIT_CNT', 'BX_COUNTER', 'TDC_MEAS', 'TRG_QUALITY'], \
        dtype={'HEAD' : 'int32', 'FPGA' : 'int32', 'TDC_CHANNEL' : 'int32', 'ORBIT_CNT' : 'int32', 'BX_COUNTER' : 'int32', 'TDC_MEAS' : 'int32', 'TRG_QUALITY' : 'float64'}, \
        low_memory=False, \
        skiprows=1, \
        nrows=options.max*1024 + 1, \
    )
    if options.verbose: print("Read %d lines from txt file %s" % (len(df), options.filename))

else:
    print("File format not recognized, exiting...")
    exit()


if options.verbose: print(df.head(50))

# remove tdc_channel = 139 since they are not physical events
df = df[df['TDC_CHANNEL']<136]
#df = df[df['FPGA']==0]
# remove double hits
##df['TDC_MEAS'] = df.groupby(['HEAD', 'FPGA', 'ORBIT_CNT', 'TDC_CHANNEL'])['TDC_MEAS'].transform(np.max)
##df = df.drop_duplicates(subset=['HEAD', 'FPGA', 'ORBIT_CNT', 'TDC_CHANNEL'], keep='last')
#df = df.drop_duplicates(subset=['HEAD', 'FPGA', 'ORBIT_CNT', 'TDC_CHANNEL'], keep='first')

if options.verbose: print("Mapping channels...")

# Map TDC_CHANNEL, FPGA to SL, LAYER, WIRE_NUM, WIRE_POS
mapconverter = Mapping()
df = mapconverter.virtex7(df)
#df = mapconverter.virtex7obdt(df)


# FIXME: SL 0 and 1 are swapped
pd.options.mode.chained_assignment = None
df.loc[df['SL'] == 0, 'SL'] = -1
df.loc[df['SL'] == 1, 'SL'] = 0
df.loc[df['SL'] == -1, 'SL'] = 1

if options.verbose: print("Determining BX0...")


# Determine BX0 either using meantimer or the trigger BX assignment
if options.meantimer: # still to be developed
    mitime = datetime.now()

    # Use only hits
    df = df[df['HEAD']==1]
    
    # Add necessary columns
    #df['TDC_CHANNEL_NORM'] = (df['TDC_CHANNEL'] - 64 * (df['SL']%2)).astype(np.uint8)
    df['TIME_ABS'] = (df['ORBIT_CNT'].astype(np.float64)*DURATION['orbit'] + df['BX_COUNTER'].astype(np.float64)*DURATION['bx'] + df['TDC_MEAS'].astype(np.float64)*DURATION['tdc']).astype(np.float64)

    
    if options.verbose: print("+ Running meantimer...")
    # Group by orbit counter (event) and SL    
    df = df.groupby(['ORBIT_CNT', 'SL'], as_index=False).apply(meantimer)
    
    '''
    grpbyorbit = df.groupby(['ORBIT_CNT', 'SL'])
    norbit, norbits = 0, df['ORBIT_CNT'].nunique()
    for (iorbit, isl), adf in grpbyorbit:
        norbit += 1
        if options.verbose and norbit % 100 == 0:
            print " + Running meantimer... (%3.2f %%)\r" % (100.*float(norbit)/float(norbits)),
            sys.stdout.flush()
        adf = adf.drop_duplicates() 
        tzeros = meantimer_results(adf)[0]
        if len(tzeros) > 0: df.loc[(df['ORBIT_CNT'] == iorbit) & (df['SL'] == isl), 'T0'] = np.mean(tzeros)
    '''

    mftime = datetime.now()
    if options.verbose: print("\nMeantimer completed [", mftime - mitime, "]")
    
    # Calculate drift time
    hits = df[df['T0'].notna()].copy()
    hits['TDRIFT'] = (hits['TIME_ABS'] - hits['T0']).astype(np.float32)

else:
    # Take the minimum BX selected among the macro-cells, and propagate it to the other rows in the same orbit
    df['T0'] = df[df['HEAD']==3].groupby('ORBIT_CNT')['TDC_MEAS'].transform(np.min)
    df['T0'] = df.groupby('ORBIT_CNT')['T0'].transform(np.max)

    # Select only valid hits
    sparhits = df.loc[df['T0'].isnull()].copy()
    trighits = df.loc[df['T0'] >= 0].copy()
    hits = trighits.loc[df['HEAD']==1].copy()

    # Create column TDRIFT
    hits['TDRIFT'] = (hits['BX_COUNTER']-hits['T0'])*DURATION['bx'] + hits['TDC_MEAS']*DURATION['tdc']

if options.verbose: print("Assigning positions...")

# Find events
hits = hits[(hits['TDRIFT']>TIME_WINDOW[0]) & (hits['TDRIFT']<TIME_WINDOW[1])]

# Count hits in each event
hits['NHITS'] = hits.groupby('ORBIT_CNT')['TDC_CHANNEL'].transform(np.size)

# Conversion from time to position
mapconverter.addXleftright(hits)

# Cosmetic changes to be compliant with common format
hits.rename(columns={'ORBIT_CNT': 'ORBIT', 'BX_COUNTER': 'BX', 'SL' : 'CHAMBER', 'WIRE_NUM' : 'WIRE', 'Z_POS' : 'Z', 'TDRIFT' : 'TIMENS'}, inplace=True)

if options.verbose: print(hits[hits['TDC_CHANNEL'] >= -128].head(50))

ftime = datetime.now()
if options.verbose: print("Ending analysis [", ftime, "],", "time elapsed [", ftime - itime, "]")


# General plots

if options.verbose: print("Producing general plots...")

import matplotlib.pyplot as plt

if options.verbose: print("Writing plots in directory %s" % (options.outputdir + runname))

# Timebox
plt.figure(figsize=(15,10))
for chamber in range(4):
    hist, bins = np.histogram(hits.loc[hits['CHAMBER']==chamber, 'TIMENS'], density=False, bins=80, range=(-150, 650))
    plt.subplot(2, 2, chamber+1)
    plt.title("Timebox [chamber %d]" % chamber)
    plt.xlabel("Time (ns)")
    plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))
plt.savefig(options.outputdir + runname + "_plots/timebox.png")
plt.savefig(options.outputdir + runname + "_plots/timebox.pdf")


# Space boxes
plt.figure(figsize=(15,10))
for chamber in range(4):
    hist, bins = np.histogram(hits.loc[hits['CHAMBER']==chamber, 'X_LEFT'], density=False, bins=70, range=(-5, 30))
    plt.subplot(2, 2, chamber+1)
    plt.title("Space box LEFT [chamber %d]" % chamber)
    plt.xlabel("Position (mm)")
    plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))
plt.savefig(options.outputdir + runname + "_plots/spacebox_left.png")
plt.savefig(options.outputdir + runname + "_plots/spacebox_left.pdf")

plt.figure(figsize=(15,10))
for chamber in range(4):
    hist, bins = np.histogram(hits.loc[hits['CHAMBER']==chamber, 'X_RIGHT'], density=False, bins=70, range=(-5, 30))
    plt.subplot(2, 2, chamber+1)
    plt.title("Space box RIGHT [chamber %d]" % chamber)
    plt.xlabel("Position (mm)")
    plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))
plt.savefig(options.outputdir + runname + "_plots/spacebox_right.png")
plt.savefig(options.outputdir + runname + "_plots/spacebox_right.pdf")


# Occupancy
plt.figure(figsize=(15,20))
occupancy = hits.groupby(["CHAMBER", "LAYER", "WIRE"])['HEAD'].count() # count performed a random column
occupancy = occupancy.reset_index().rename(columns={'HEAD' : 'COUNTS'}) # reset the indices and make new ones, because the old indices are needed for selection
for chamber in range(4):
    for layer in range(4):
        x = np.array( occupancy.loc[((occupancy['CHAMBER'] == chamber) & (occupancy['LAYER'] == layer+1)), 'WIRE'] )
        y = np.array( occupancy.loc[((occupancy['CHAMBER'] == chamber) & (occupancy['LAYER'] == layer+1)), 'COUNTS'] )
        plt.subplot(4, 4, chamber*4 + layer + 1)
        plt.title("Occupancy [SL %d, LAYER %d]" % (chamber, layer+1))
        plt.xlabel("Wire number")
        plt.bar(x, y)
plt.savefig(options.outputdir + runname + "_plots/occupancy.png")
plt.savefig(options.outputdir + runname + "_plots/occupancy.pdf")


# Event display
if not options.eventdisplay:
    if options.verbose: print("Skipping event display...\nDone.")
    exit()

if options.verbose: print("Producing event displays...")

from pdb import set_trace as br
from operator import itemgetter
from numpy.polynomial.polynomial import Polynomial

from modules.utils import OUT_CONFIG
from modules.geometry.hit import HitManager
from modules.geometry.sl import SL
from modules.geometry.segment import Segment
from modules.geometry import Geometry, COOR_ID
from modules.reco import config, plot
from modules.analysis import config as CONFIGURATION

import os
import itertools
import bokeh
import numpy as np

G = Geometry(CONFIGURATION)
H = HitManager()
SLs = {}
for iSL in config.SL_SHIFT.keys():
    SLs[iSL] = SL(iSL, config.SL_SHIFT[iSL], config.SL_ROTATION[iSL])
# Defining which SLs should be plotted in which global view
GLOBAL_VIEW_SLs = {
    'xz': [SLs[0], SLs[2]],
    'yz': [SLs[1], SLs[3]]
}

ev = hits[['ORBIT', 'BX', 'NHITS', 'CHAMBER', 'LAYER', 'WIRE', 'X_LEFT', 'X_RIGHT', 'Z', 'TIMENS']]

chi2s = {}
for iSL, sl in SLs.items(): chi2s[iSL] = np.array([])

events = ev.groupby(['ORBIT'])
# Loop on events (same orbit)
for orbit, hitlist in events:
    if options.verbose: print("Drawing event", orbit, "...")
    H.reset()
    hits_lst = []
    # Loop over hits and fill H object
    for names, hit in hitlist.iterrows():
        # format: 'sl', 'layer', 'wire', 'time'
        hits_lst.append([hit['CHAMBER'], hit['LAYER'], hit['WIRE'], hit['TIMENS']]) 
    
    H.add_hits(hits_lst)
    # Calculating local+global hit positions
    H.calc_pos(SLs)
    # Creating figures of the chambers
    figs = {}
    figs['sl'] = plot.book_chambers_figure(G)
    figs['global'] = plot.book_global_figure(G, GLOBAL_VIEW_SLs)
    # Analyzing hits in each SL
    sl_fit_results = {}
    
    for iSL, sl in SLs.items():
        
        hits_sl = H.hits.loc[H.hits['sl'] == iSL].sort_values('layer')

        if True: #args.plot:
            # Drawing the left and right hits in local frame
            figs['sl'][iSL].square(x=hits_sl['lposx'], y=hits_sl['posz'], size=5, fill_color='red', fill_alpha=0.7, line_width=0)
            figs['sl'][iSL].square(x=hits_sl['rposx'], y=hits_sl['posz'], size=5, fill_color='blue', fill_alpha=0.7, line_width=0)
        # Performing track reconstruction in the local frame
        sl_fit_results[iSL] = []
        layer_groups = hits_sl.groupby('layer').groups
        n_layers = len(layer_groups)
        # Stopping if lass than 3 layers of hits
        if n_layers < config.NHITS_MIN_LOCAL:
            continue
        hitid_layers = [gr.to_numpy() for gr_name, gr in layer_groups.items()]
        # Building the list of all possible hit combinations with 1 hit from each layer
        hits_layered = list(itertools.product(*hitid_layers))
        # Building more combinations using only either left or right position of each hit
        for hit_ids in hits_layered:
            # print('- -', hit_ids)
            posz = hits_sl.loc[hits_sl.index.isin(hit_ids), 'posz'].values
            posx = hits_sl.loc[hits_sl.index.isin(hit_ids), ['lposx', 'rposx']].values
            posx_combs = list(itertools.product(*posx))
            # Fitting each combination
            fit_results_lr = []
            fit_range = (min(posz), max(posz))
            for iC, posx_comb in enumerate(posx_combs):
                pfit, stats = Polynomial.fit(posz, posx_comb, 1, full=True, window=fit_range, domain=fit_range)
                chi2 = stats[0][0] / n_layers
                if chi2 < config.FIT_CHI2_MAX:
                    a0, a1 = pfit
                    fit_results_lr.append((chi2, hit_ids, pfit))
                    if options.verbose: print("Track found in SL", iSL, "with parameters:", a0, a1, ", chi2:", chi2)
            # Keeping only the best fit result from the given set of physical hits
            fit_results_lr.sort(key=itemgetter(0))
            if fit_results_lr:
                sl_fit_results[iSL].append(fit_results_lr[0])
                bestp0, bestp1 = fit_results_lr[0][2]
                print("+ Orbit", orbit, "best segment in SL", iSL, "with angle =", bestp1, "and offset =", bestp0, " ( chi2 =", fit_results_lr[0][0], ")")
                chi2s[iSL] = np.append(chi2s[iSL], fit_results_lr[0][0])
        # Sorting the fit results of a SL by Chi2
        sl_fit_results[iSL].sort(key=itemgetter(0))
        if sl_fit_results[iSL]:
            # Drawing fitted tracks
            posz = np.array([G.SL_FRAME['b']+1, G.SL_FRAME['t']-1], dtype=np.float32)
            for iR, res in enumerate(sl_fit_results[iSL][:5]):
                col = config.TRACK_COLORS[iR]
                posx = res[2](posz)
                figs['sl'][iSL].line(x=posx, y=posz,
                                     line_color=col, line_alpha=0.7, line_width=3)

    if True: #args.plot:
        # Drawing the left and right hits in global frame
        for view, sls in GLOBAL_VIEW_SLs.items():
            sl_ids = [sl.id for sl in sls]
            hits_sls = H.hits.loc[H.hits['sl'].isin(sl_ids)]
            figs['global'][view].square(x=hits_sls['glpos'+view[0]], y=hits_sls['glpos'+view[1]],
                                        fill_color='red', fill_alpha=0.7, line_width=0)
            figs['global'][view].square(x=hits_sls['grpos'+view[0]], y=hits_sls['grpos'+view[1]],
                                        fill_color='blue', fill_alpha=0.7, line_width=0)
            # Building 3D segments from the fit results in each SL
            posz = np.array([G.SL_FRAME['b'], G.SL_FRAME['t']], dtype=np.float32)
            for sl in sls:
                for iR, res in enumerate(sl_fit_results[sl.id][:5]):
                    posx = res[2](posz)
                    start = (posx[0], 0, posz[0])
                    end = (posx[1], 0, posz[1])
                    segL = Segment(start, end)
                    segG = segL.fromSL(sl)
                    segG.calc_vector()
                    # Extending the global segment to the full height of the view
                    start = segG.pointAtZ(plot.PLOT_RANGE['y'][0])
                    end = segG.pointAtZ(plot.PLOT_RANGE['y'][1])
                    # Getting XY coordinates of the global segment for the current view
                    iX = COOR_ID[view[0]]
                    posx = [start[iX], end[iX]]
                    posy = [start[2], end[2]]
                    # Drawing the segment
                    col = config.TRACK_COLORS[sl.id]
                    figs['global'][view].line(x=posx, y=posy,
                                         line_color=col, line_alpha=0.7, line_width=3)



    # Storing the figures to an HTML file
    if True: #args.plot:
        plots = [[figs['sl'][l]] for l in [3, 2, 1, 0]]
        plots.append([figs['global'][v] for v in ['xz', 'yz']])
        bokeh.io.output_file(options.outputdir + runname + "_events/orbit_%d.html" % orbit, mode='cdn')
        bokeh.io.save(bokeh.layouts.layout(plots))


# Chi2
plt.figure(figsize=(15,10))
for chamber in range(4):
    hist, bins = np.histogram(chi2s[chamber], density=False, bins=20, range=(0, 2))
    plt.subplot(2, 2, chamber+1)
    plt.title("Chi2 [chamber %d]" % chamber)
    plt.xlabel("Chi2")
    plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))
plt.savefig(options.outputdir + runname + "_plots/chi2.png")
plt.savefig(options.outputdir + runname + "_plots/chi2.pdf")



#print ev.head(60)

if options.verbose: print("Done.")



# python analysis.py -i data/Run000966/output_raw.dat -m 1 -v
