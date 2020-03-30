#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-i", "--inputfile", action="store", type="string", dest="filename", default="LNL/Run000810/data_000000.dat", help="Provide input file (either binary or txt)")
parser.add_option("-o", "--outputdir", action="store", type="string", dest="outputdir", default="plots/", help="Specify output directory")
parser.add_option("-m", "--meantimer", action="store_true", default=False, dest="meantimer", help="Force application of the meantimer algorithm (override BX assignment)")
parser.add_option("-v", "--verbose", action="store_true", default=False, dest="verbose", help="Increase verbosity")
(options, args) = parser.parse_args()

fileName = options.filename
isBinary = not fileName.endswith('.txt')
outputDir = options.outputdir
forceMeantimer = options.meantimer
verbose = options.verbose


word_size = 8 # one 64-bit word
num_words = 128 # 1 DMA data transfer = 1 kB = 1024 B = 128 words (hits)

import struct

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
        
        if head <= 2:
            rows.append(hit_unpacker(buffer))

        elif head == 3:
            rows.append(trigger_unpacker(buffer))
        
    return rows


def meanTimer(obj):
    #print len(obj), obj.values
    return obj.sum()


if isBinary:
    dt = []
    word_count = 0
    inputFile = open(fileName, 'rb')
    while True:
        word = inputFile.read(num_words*word_size)
        if word:
              d = unpacker(word)
              dt += d
              word_count += 1
              #print len(dt)
        else: break

    if verbose: print("Read %d lines from binary file %s" % (len(dt), fileName))
    df = pd.DataFrame.from_dict(dt)
    
else:
    # force 7 fields
    # in case of forcing, for some reason, the first raw is not interpreted as columns names
    df = pd.read_csv(fileName, \
        names=['HEAD', 'FPGA', 'TDC_CHANNEL', 'ORBIT_CNT', 'BX_COUNTER', 'TDC_MEAS', 'TRG_QUALITY'], \
        dtype={'HEAD' : 'int32', 'FPGA' : 'int32', 'TDC_CHANNEL' : 'int32', 'ORBIT_CNT' : 'int32', 'BX_COUNTER' : 'int32', 'TDC_MEAS' : 'int32', 'TRG_QUALITY' : 'float64'}, \
        low_memory=False, \
        skiprows=1, \
    )
    if verbose: print("Read %d lines from txt file %s" % (len(df), fileName))

#else:
#    print("File format not recognized, exiting...")
#    exit()


# remove tdc_channel = 139 since they are not physical events
df = df.loc[ df['TDC_CHANNEL']!=139 ]

df['T0'] = df[df['HEAD']==3].groupby('ORBIT_CNT')['TDC_MEAS'].transform(np.min)
df['T0'] = df.groupby('ORBIT_CNT')['T0'].transform(np.max)

sparhits = df.loc[df['T0'].isnull()].copy()
trighits = df.loc[df['T0'] >= 0].copy()
hits = trighits.loc[df['HEAD']==1].copy()

if forceMeantimer: # still to be developed
    if verbose: print("Running meantimer")
    hits['T0'] = hits.groupby('ORBIT_CNT')['TDC_MEAS'].transform(meanTimer)


## Parameters

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
pos_z       = [  ZCELL*3.5,    ZCELL*1.5,    ZCELL*2.5,    ZCELL*0.5, ]
posshift_x  = [  0,            0,            0.5,          0.5,       ]

# Create column TDRIFT
hits['TDRIFT'] = (hits['BX_COUNTER']-hits['T0'])*DURATION['bx'] + hits['TDC_MEAS']*DURATION['tdc']

# Find events
hits = hits[(hits['TDRIFT']>TIME_WINDOW[0]) & (hits['TDRIFT']<TIME_WINDOW[1])]

hits.loc[hits['TDC_CHANNEL'] % 4 == 1, 'X_POSSHIFT'] = posshift_x[0]
hits.loc[hits['TDC_CHANNEL'] % 4 == 2, 'X_POSSHIFT'] = posshift_x[1]
hits.loc[hits['TDC_CHANNEL'] % 4 == 3, 'X_POSSHIFT'] = posshift_x[2]
hits.loc[hits['TDC_CHANNEL'] % 4 == 0, 'X_POSSHIFT'] = posshift_x[3]

hits.loc[(hits['FPGA'] == 0) & (hits['TDC_CHANNEL'] <= NCHANNELS), 'SL'] = 0
hits.loc[(hits['FPGA'] == 0) & (hits['TDC_CHANNEL'] > NCHANNELS) & (hits['TDC_CHANNEL'] <= 2*NCHANNELS), 'SL'] = 1
hits.loc[(hits['FPGA'] == 1) & (hits['TDC_CHANNEL'] <= NCHANNELS), 'SL'] = 2
hits.loc[(hits['FPGA'] == 1) & (hits['TDC_CHANNEL'] > NCHANNELS) & (hits['TDC_CHANNEL'] <= 2*NCHANNELS), 'SL'] = 3

hits['TDC_CHANNEL_NORM'] = hits['TDC_CHANNEL'] - NCHANNELS*(hits['SL']%2)

hits['X_POS_LEFT']  = np.int32((hits['TDC_CHANNEL_NORM']-0.5)/4) + hits['X_POSSHIFT']*XCELL + XCELL/2 - np.maximum(hits['TDRIFT'], 0)*VDRIFT
hits['X_POS_RIGHT'] = np.int32((hits['TDC_CHANNEL_NORM']-0.5)/4) + hits['X_POSSHIFT']*XCELL + XCELL/2 + np.maximum(hits['TDRIFT'], 0)*VDRIFT
#hits['X_POS_DELTA'] = np.abs(hits['X_POS_RIGHT'] -hits['X_POS_LEFT'])

hits.loc[hits['TDC_CHANNEL'] % 4 == 1, 'Z_POS'] = pos_z[0]
hits.loc[hits['TDC_CHANNEL'] % 4 == 2, 'Z_POS'] = pos_z[1]
hits.loc[hits['TDC_CHANNEL'] % 4 == 3, 'Z_POS'] = pos_z[2]
hits.loc[hits['TDC_CHANNEL'] % 4 == 0, 'Z_POS'] = pos_z[3]

if verbose: print("Writing plots in directory %s" % (outputDir))

import matplotlib.pyplot as plt

# Timebox
plt.figure(figsize=(15,10))
for chamber in range(4):
    hist, bins = np.histogram(hits.loc[hits['SL']==chamber, 'TDRIFT'], density=False, bins=80, range=(-150, 650))
    plt.subplot(2, 2, chamber+1)
    plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))
plt.savefig(outputDir + "timebox.png")
plt.savefig(outputDir + "timebox.pdf")


# Space boxes
plt.figure(figsize=(15,10))
for chamber in range(4):
    hist, bins = np.histogram(hits.loc[hits['SL']==chamber, 'X_POS_LEFT'], density=False, bins=70, range=(-5, 30))
    plt.subplot(2, 2, chamber+1)
    plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))
plt.savefig(outputDir + "spacebox_left.png")
plt.savefig(outputDir + "spacebox_left.pdf")

plt.figure(figsize=(15,10))
for chamber in range(4):
    hist, bins = np.histogram(hits.loc[hits['SL']==chamber, 'X_POS_RIGHT'], density=False, bins=70, range=(-5, 30))
    plt.subplot(2, 2, chamber+1)
    plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))
plt.savefig(outputDir + "spacebox_right.png")
plt.savefig(outputDir + "spacebox_right.pdf")


# Occupancy
plt.figure(figsize=(15,10))
occupancy = hits.groupby(["SL", "TDC_CHANNEL_NORM"])['HEAD'].count() # count performed a random column
occupancy = occupancy.reset_index().rename(columns={'HEAD' : 'COUNTS'}) # reset the indices and make new ones, because the old indices are needed for selection
for chamber in range(4):
    x = np.array( occupancy.loc[occupancy['SL'] == chamber, 'TDC_CHANNEL_NORM'] )
    y = np.array( occupancy.loc[occupancy['SL'] == chamber, 'COUNTS'] )
    plt.subplot(2, 2, chamber+1)
    plt.bar(x, y)
plt.savefig(outputDir + "occupancy.png")
plt.savefig(outputDir + "occupancy.pdf")

if verbose: print("Done.")

