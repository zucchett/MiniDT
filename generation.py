#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import random as rnd
import math

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-n", "--number", action="store", type="int", dest="number", default=100000, help="Number of muons to be generated")
parser.add_option("-o", "--outputfile", action="store", type="string", dest="outputfile", default="generation.csv", help="Specify name of the output .csv file")
parser.add_option("-p", "--plot", action="store_true", default=False, dest="plot", help="Produce control plots")
parser.add_option("-v", "--verbose", action="store_true", default=False, dest="verbose", help="Increase verbosity")
(options, args) = parser.parse_args()

number = options.number
outputFile = options.outputfile
doPlots = options.plot
verbose = options.verbose


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


# Generate muons
rnd.seed(123456)

# starting orbit counter (arbitrary)
orbit_cnt = 1000000

# dict of tracks, [m, q]
dc = []
for i in range(number):
	sl = rnd.randint(0, NSL-1)
	orbit_cnt += 1
	bx0 = rnd.randint(1 + 30, DURATION['orbit:bx'] - 30) # avoid first and final BX to avoid cross-orbit muons
	angle = rnd.uniform(math.pi*1./4., math.pi*3./4.) # angle in radians between 45 and 135 degrees
	offset = rnd.uniform(0. - 5*ZCELL, NCHANNELS/4.*XCELL + 5*ZCELL)
	#tracks.append([sl, angle, inter])
	m = math.tan(angle) # slope
	q = -m * offset # intercept

	for l in range(4):
		x = (pos_z[l] - q)/m # invert formula z = mx + q
		nw = int(math.floor((x - posshift_x[l] * XCELL - XCELL/2.) / XCELL) + 1) # number of wire in the same layer
		if nw < 1 or nw > 16: continue
		dx = x - posshift_x[l] * XCELL - nw * XCELL
		t = abs(dx) / VDRIFT
		bx = t // DURATION['bx']
		dt = t % DURATION['bx']
		
		tdc_meas = int(math.floor(dt * DURATION['tdc']))
		bx_counter = int(bx0 + bx)
		tdc_channel_norm = (layer_z[l] - 1) + nw * 4
		tdc_channel = tdc_channel_norm + NCHANNELS * (sl % 2)
		fpga = 0 if sl < 2 else 1
		
		dc.append({"HEAD" : 1, 'FPGA' : fpga, 'TDC_CHANNEL' : tdc_channel, 'ORBIT_CNT' : orbit_cnt, 'BX_COUNTER' : bx_counter, 'TDC_MEAS' : tdc_meas, 'TRG_QUALITY' : 0, 'TDC_CHANNEL_NORM' : tdc_channel_norm, 'X' : x, 'DX' : dx, 'DT' : dt, 'T' : t, 'NWIRE' : nw, 'SL' : sl})
		#dc.append({"HEAD" : 1, 'FPGA' : fpga, 'TDC_CHANNEL' : tdc_channel, 'ORBIT_CNT' : orbit_cnt, 'BX_COUNTER' : bx_counter, 'TDC_MEAS' : tdc_meas, 'TRG_QUALITY' : 0})



import pandas as pd
import numpy as np

columns=['HEAD', 'FPGA', 'TDC_CHANNEL', 'ORBIT_CNT', 'BX_COUNTER', 'TDC_MEAS', 'TRG_QUALITY']

df = pd.DataFrame.from_dict(dc)


if doPlots:

	import matplotlib.pyplot as plt

	# Timebox
	#plt.figure(figsize=(15,10))
	#
	#hist, bins = np.histogram(df['ANGLE'], density=False, bins=100, range=(0., 3.15))
	#plt.subplot(2, 2, 1)
	#plt.xlabel("angle (rad)")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	#hist, bins = np.histogram(df['OFFSET'], density=False, bins=100, range=(0., NCHANNELS/4.*XCELL))
	#plt.subplot(2, 2, 2)
	#plt.xlabel("offset (mm)")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	#hist, bins = np.histogram(df['M'], density=False, bins=100, range=(-1.2, 1.2))
	#plt.subplot(2, 2, 3)
	#plt.xlabel("m")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	#hist, bins = np.histogram(df['Q'], density=False, bins=100, range=(-10000., 10000.))
	#plt.subplot(2, 2, 4)
	#plt.xlabel("q (mm)")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))


	plt.figure(figsize=(15,10))

	hist, bins = np.histogram(df['X'], density=False, bins=100, range=(-100., 1000))
	plt.subplot(3, 3, 1)
	plt.xlabel("position in the layer (mm)")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(df['DX'], density=False, bins=100, range=(-25., 25))
	plt.subplot(3, 3, 2)
	plt.xlabel("position in the cell (mm)")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(df['NWIRE'], density=False, bins=22, range=(-1.5, 20.5))
	plt.subplot(3, 3, 3)
	plt.xlabel("number of wire in the layer")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))


	hist, bins = np.histogram(df['T'], density=False, bins=100, range=(-10, 900))
	plt.subplot(3, 3, 4)
	plt.xlabel("drift time (ns)")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(df['DT'], density=False, bins=100, range=(-10, 40))
	plt.subplot(3, 3, 5)
	plt.xlabel("drift time in the same BX (ns)")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(df['TDC_MEAS'], density=False, bins=40, range=(-3.5, 36.5))
	plt.subplot(3, 3, 6)
	plt.xlabel("TDC measurement")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))


	hist, bins = np.histogram(df['TDC_CHANNEL_NORM'], density=False, bins=NCHANNELS+1, range=(-0.5, NCHANNELS+0.5))
	plt.subplot(3, 3, 7)
	plt.xlabel("TDC CHANNEL NORM")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(df['TDC_CHANNEL'], density=False, bins=NCHANNELS*2+1, range=(-0.5, NCHANNELS*2+0.5))
	plt.subplot(3, 3, 8)
	plt.xlabel("TDC CHANNEL")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	#hist, bins = np.histogram(df['BX_COUNTER'], density=False, bins=DURATION['orbit:bx']+1, range=(-0.5, DURATION['orbit:bx']+0.5))
	#plt.subplot(3, 3, 9)
	#plt.xlabel("bunch crossing number")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(df['FPGA'], density=False, bins=5, range=(-0.5, 4.5))
	plt.subplot(3, 3, 9)
	plt.xlabel("FPGA")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))


	plt.savefig("generation.png")
	plt.savefig("generation.pdf")


# filter and save
df = df[columns]
df.to_csv(outputFile, columns=columns)

