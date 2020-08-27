#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-i", "--inputfile", action="store", type="string", dest="filename", default="LNL/Run000966/output_raw.dat", help="Provide input file (either binary or txt)")
parser.add_option("-o", "--outputdir", action="store", type="string", dest="outputdir", default="./output/", help="Specify output directory")
parser.add_option("-m", "--max", action="store", type=int, default=-1, dest="max", help="Maximum number of words to be read")
parser.add_option("-x", "--meantimer", action="store_true", default=False, dest="meantimer", help="Force application of the meantimer algorithm (override BX assignment)")
parser.add_option("-v", "--verbose", action="store_true", default=False, dest="verbose", help="Increase verbosity")
(options, args) = parser.parse_args()

runname = [x for x in options.filename.split('/') if 'Run' in x][0] if "Run" in options.filename else "Run000000"

if not os.path.exists(options.outputdir): os.makedirs(options.outputdir)
if not os.path.exists(options.outputdir + runname + "_plots/"): os.makedirs(options.outputdir + runname + "_plots/")
if not os.path.exists(options.outputdir + runname + "_events/"): os.makedirs(options.outputdir + runname + "_events/")


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


DURATION = {                         
    'orbit:bx': 3564,
    'orbit': 3564*25,
    'bx': 25.,
    'tdc': 25./30
}

TIME_WINDOW = (-50, 500)

### Chamber configuration
NCHANNELS  = 64    # channels per SL
NSL        = 4     # number of SL
### Cell parameters
XCELL     = 42.                      # cell width in mm
ZCELL     = 13.                      # cell height in mm
ZCHAMB    = 550.                     # spacing betweeen chambers in mm

TDRIFT    = 15.6*DURATION['bx']    # drift time in ns
VDRIFT    = XCELL*0.5 / TDRIFT     # drift velocity in mm/ns 

layer_z     = [  1,            3,            2,            4,         ]
chanshift_x = [  0,            -1,           0,            -1,        ]
posshift_z  = [  ZCELL*1.5,    -ZCELL*0.5,   ZCELL*0.5,    -ZCELL*1.5 ]
posshift_x  = [  -7.5*XCELL,   -7.5*XCELL,   -7.0*XCELL,   -7.0*XCELL ]


# Unpacker
import struct

word_size = 8 # one 64-bit word
num_words = 128 + 1 # 1 DMA data transfer = 1 kB = 1024 B = 128 words (hits)

def hit_unpacker(word):
    # hit masks
    hmaskTDC_MEAS     = 0x1F
    hmaskBX_COUNTER   = 0xFFF
    hmaskORBIT_CNT    = 0xFFFFFFFF
    hmaskTDC_CHANNEL  = 0x1FF
    hmaskFPGA         = 0xF
    hmaskHEAD         = 0x3

    hfirstTDC_MEAS    = 0
    hfirstBX_COUNTER  = 5
    hfirstORBIT_CNT   = 17
    hfirstTDC_CHANNEL = 49
    hfirstFPGA        = 58
    hfirstHEAD        = 62
    
    TDC_MEAS     =      int(( word >> hfirstTDC_MEAS    ) & hmaskTDC_MEAS   )
    BX_COUNTER   =      int(( word >> hfirstBX_COUNTER  ) & hmaskBX_COUNTER )
    ORBIT_CNT    =      int(( word >> hfirstORBIT_CNT   ) & hmaskORBIT_CNT  )
    TDC_CHANNEL  =  1 + int(( word >> hfirstTDC_CHANNEL ) & hmaskTDC_CHANNEL)
    FPGA         =      int(( word >> hfirstFPGA        ) & hmaskFPGA       )
    HEAD         =      int(( word >> hfirstHEAD        ) & hmaskHEAD       )
    
    if((TDC_CHANNEL!=137) and (TDC_CHANNEL!=138)):
            TDC_MEAS -= 1

    unpacked  = {
        'HEAD': HEAD,
        'FPGA': FPGA,
        'TDC_CHANNEL': TDC_CHANNEL,
        'ORBIT_CNT': ORBIT_CNT,
        'BX_COUNTER': BX_COUNTER,
        'TDC_MEAS': TDC_MEAS,
        'TRG_QUALITY': np.NaN
    }
    
    return unpacked #Row(**unpacked)

def trigger_unpacker(word):
    # Trigger masks
    tmaskQUAL    = 0x0000000000000001
    tmaskBX      = 0x0000000000001FFE
    tmaskTAGBX   = 0x0000000001FFE000
    tmaskTAGORB  = 0x01FFFFFFFE000000
    tmaskMCELL   = 0x0E00000000000000
    tmaskSL      = 0x3000000000000000
    tmaskHEAD    = 0xC000000000000000

    tfirstQUAL   = 0
    tfirstBX     = 1
    tfirstTAGBX  = 13
    tfirstTAGORB = 25
    tfirstMCELL  = 57
    tfirstSL     = 60
    tfirstHEAD   = 62
    
    storedTrigHead     = int(( word & tmaskHEAD   ) >> tfirstHEAD  )
    storedTrigMiniCh   = int(( word & tmaskSL     ) >> tfirstSL    )
    storedTrigMCell    = int(( word & tmaskMCELL  ) >> tfirstMCELL )
    storedTrigTagOrbit = int(( word & tmaskTAGORB ) >> tfirstTAGORB)
    storedTrigTagBX    = int(( word & tmaskTAGBX  ) >> tfirstTAGBX )
    storedTrigBX       = int(( word & tmaskBX     ) >> tfirstBX    )
    storedTrigQual     = int(( word & tmaskQUAL   ) >> tfirstQUAL  )
    
    unpacked = {
        'HEAD': storedTrigHead,
        'FPGA': storedTrigMiniCh,
        'TDC_CHANNEL': storedTrigMCell,
        'ORBIT_CNT': storedTrigTagOrbit,
        'BX_COUNTER': storedTrigTagBX,
        'TDC_MEAS': storedTrigBX,
        'TRG_QUALITY': storedTrigQual
    }
    
    return unpacked #Row(**unpacked)


def unpacker(hit):
    
    rows = []
    
    for i in range(0, num_words*word_size, word_size):
        
        buffer = struct.unpack('<Q', hit[i:i+word_size])[0]
        head = (buffer >> 62) & 0x3
        
        if head == 1 or head == 2:
            rows.append(hit_unpacker(buffer))

        elif head == 3:
            rows.append(trigger_unpacker(buffer))
        
    return rows


def meanTimer(obj):
    #print len(obj), obj.values
    return obj.sum()


### Open file ###

if options.filename.endswith('.dat'):
    dt = []
    word_count = 0
    inputFile = open(options.filename, 'rb')
    while (word_count < 0 or word_count < options.max):
        word = inputFile.read(num_words*word_size)
        if word:
              d = unpacker(word)
              dt += d
              word_count += 1
              #print len(dt)
        else: break

    if options.verbose: print("Read %d lines from binary file %s" % (len(dt), options.filename))
    df = pd.DataFrame.from_dict(dt)
    
elif options.filename.endswith('.txt') or options.filename.endswith('.csv'):
    # force 7 fields
    # in case of forcing, for some reason, the first raw is not interpreted as columns names
    df = pd.read_csv(options.filename, \
        names=['HEAD', 'FPGA', 'TDC_CHANNEL', 'ORBIT_CNT', 'BX_COUNTER', 'TDC_MEAS', 'TRG_QUALITY'], \
        dtype={'HEAD' : 'int32', 'FPGA' : 'int32', 'TDC_CHANNEL' : 'int32', 'ORBIT_CNT' : 'int32', 'BX_COUNTER' : 'int32', 'TDC_MEAS' : 'int32', 'TRG_QUALITY' : 'float64'}, \
        low_memory=False, \
        skiprows=0, \
    )
    if options.verbose: print("Read %d lines from txt file %s" % (len(df), options.filename))

else:
    print("File format not recognized, exiting...")
    exit()


if options.verbose: print df.head(50)

# remove tdc_channel = 139 since they are not physical events
df = df.loc[ df['TDC_CHANNEL']!=139 ]

df['T0'] = df[df['HEAD']==3].groupby('ORBIT_CNT')['TDC_MEAS'].transform(np.min)
df['T0'] = df.groupby('ORBIT_CNT')['T0'].transform(np.max)

sparhits = df.loc[df['T0'].isnull()].copy()
trighits = df.loc[df['T0'] >= 0].copy()
hits = trighits.loc[df['HEAD']==1].copy()

if options.meantimer: # still to be developed
    if options.verbose: print("Running meantimer")
    hits['T0'] = hits.groupby('ORBIT_CNT')['TDC_MEAS'].transform(meanTimer)


# Create column TDRIFT
hits['TDRIFT'] = (hits['BX_COUNTER']-hits['T0'])*DURATION['bx'] + hits['TDC_MEAS']*DURATION['tdc']

# Find events
hits = hits[(hits['TDRIFT']>TIME_WINDOW[0]) & (hits['TDRIFT']<TIME_WINDOW[1])]

# Count hits in each event
hits['NHITS'] = hits.groupby('ORBIT_CNT')['TDC_CHANNEL'].transform(np.size)

hits.loc[(hits['FPGA'] == 0) & (hits['TDC_CHANNEL'] <= NCHANNELS), 'SL'] = 0
hits.loc[(hits['FPGA'] == 0) & (hits['TDC_CHANNEL'] > NCHANNELS) & (hits['TDC_CHANNEL'] <= 2*NCHANNELS), 'SL'] = 1
hits.loc[(hits['FPGA'] == 1) & (hits['TDC_CHANNEL'] <= NCHANNELS), 'SL'] = 2
hits.loc[(hits['FPGA'] == 1) & (hits['TDC_CHANNEL'] > NCHANNELS) & (hits['TDC_CHANNEL'] <= 2*NCHANNELS), 'SL'] = 3

hits.loc[hits['TDC_CHANNEL'] % 4 == 1, 'LAYER'] = 1
hits.loc[hits['TDC_CHANNEL'] % 4 == 2, 'LAYER'] = 3
hits.loc[hits['TDC_CHANNEL'] % 4 == 3, 'LAYER'] = 2
hits.loc[hits['TDC_CHANNEL'] % 4 == 0, 'LAYER'] = 4

hits.loc[hits['TDC_CHANNEL'] % 4 == 1, 'X_POSSHIFT'] = posshift_x[0]
hits.loc[hits['TDC_CHANNEL'] % 4 == 2, 'X_POSSHIFT'] = posshift_x[1]
hits.loc[hits['TDC_CHANNEL'] % 4 == 3, 'X_POSSHIFT'] = posshift_x[2]
hits.loc[hits['TDC_CHANNEL'] % 4 == 0, 'X_POSSHIFT'] = posshift_x[3]

hits.loc[hits['TDC_CHANNEL'] % 4 == 1, 'Z_POS'] = posshift_z[0]
hits.loc[hits['TDC_CHANNEL'] % 4 == 2, 'Z_POS'] = posshift_z[1]
hits.loc[hits['TDC_CHANNEL'] % 4 == 3, 'Z_POS'] = posshift_z[2]
hits.loc[hits['TDC_CHANNEL'] % 4 == 0, 'Z_POS'] = posshift_z[3]


hits['TDC_CHANNEL_NORM'] = ( hits['TDC_CHANNEL'] - NCHANNELS*(hits['SL']%2) ).astype(np.uint8) # TDC_CHANNEL from 0 to 127 -> TDC_CHANNEL_NORM from 0 to 63
hits['WIRE_NUM'] = ( (hits['TDC_CHANNEL_NORM'] - 1) / 4 + 1 ).astype(np.uint8)
hits['WIRE_POS'] = (hits['WIRE_NUM'] - 1)*XCELL + hits['X_POSSHIFT']

hits['X_LEFT']  = hits['WIRE_POS'] - np.maximum(hits['TDRIFT'], 0)*VDRIFT
hits['X_RIGHT'] = hits['WIRE_POS'] + np.maximum(hits['TDRIFT'], 0)*VDRIFT
#hits['X_POS_DELTA'] = np.abs(hits['X_POS_RIGHT'] -hits['X_POS_LEFT'])

# Cosmetic changes to be compliant to common format
hits = hits.astype({'SL' : 'int8', 'LAYER' : 'int8'})
hits.rename(columns={'ORBIT_CNT': 'ORBIT', 'BX_COUNTER': 'BX', 'SL' : 'CHAMBER', 'WIRE_NUM' : 'WIRE', 'Z_POS' : 'Z', 'TDRIFT' : 'TIMENS'}, inplace=True)

if options.verbose: print hits[hits['TDC_CHANNEL'] >= -128].head(50)


# General plots
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
plt.figure(figsize=(15,10))
occupancy = hits.groupby(["CHAMBER", "TDC_CHANNEL_NORM"])['HEAD'].count() # count performed a random column
occupancy = occupancy.reset_index().rename(columns={'HEAD' : 'COUNTS'}) # reset the indices and make new ones, because the old indices are needed for selection
for chamber in range(4):
    x = np.array( occupancy.loc[occupancy['CHAMBER'] == chamber, 'TDC_CHANNEL_NORM'] )
    y = np.array( occupancy.loc[occupancy['CHAMBER'] == chamber, 'COUNTS'] )
    plt.subplot(2, 2, chamber+1)
    plt.title("Occupancy [chamber %d]" % chamber)
    plt.xlabel("Channel")
    plt.bar(x, y)
plt.savefig(options.outputdir + runname + "_plots/occupancy.png")
plt.savefig(options.outputdir + runname + "_plots/occupancy.pdf")



# Event display
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

# FIXME: SL 0 and 1 are swapped
pd.options.mode.chained_assignment = None
ev.loc[ev['CHAMBER'] == 0, 'CHAMBER'] = -1
ev.loc[ev['CHAMBER'] == 1, 'CHAMBER'] = 0
ev.loc[ev['CHAMBER'] == -1, 'CHAMBER'] = 1


events = ev.groupby(['ORBIT'])
# Loop on events (same orbit)
for orbit, hitlist in events:
    if options.verbose: print "Drawing event", orbit, "..."
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
        # print('- SL', iSL)
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
                    if options.verbose: print "Track found in SL", iSL, "with parameters:", a0, a1, ", chi2:", chi2
            # Keeping only the best fit result from the given set of physical hits
            fit_results_lr.sort(key=itemgetter(0))
            if fit_results_lr:
                sl_fit_results[iSL].append(fit_results_lr[0])
                bestp0, bestp1 = fit_results_lr[0][2]
                print "+ Orbit", orbit, "best segment in SL", iSL, "with angle =", bestp1, "and offset =", bestp0, " ( chi2 =", fit_results_lr[0][0], ")"
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
        plots = [[figs['sl'][l]] for l in [3, 1, 2, 0]]
        plots.append([figs['global'][v] for v in ['xz', 'yz']])
        bokeh.io.output_file(options.outputdir + runname + "_events/orbit_%d.html" % orbit, mode='cdn')
        bokeh.io.save(bokeh.layouts.layout(plots))

#print ev.head(60)

if options.verbose: print("Done.")



# python analysis.py -i LNL/Run000966/output_raw.dat -m 1 -v
