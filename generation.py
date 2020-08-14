#!/usr/bin/env python
# coding: utf-8

import os, math
import pandas as pd
import numpy as np
import random as rnd

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-n", "--number", action="store", type="int", dest="number", default=100000, help="Number of muons to be generated")
parser.add_option("-o", "--outputfile", action="store", type="string", dest="outputfile", default="output_unpacked.txt", help="Specify name of the output .csv file")
parser.add_option("-p", "--plot", action="store_true", default=False, dest="plot", help="Produce control plots")
parser.add_option("-r", "--run", action="store", type=int, dest="run", default=1, help="Specify Run number")
parser.add_option("-s", "--seed", action="store", type=int, dest="seed", default=123456, help="Generation seed")
parser.add_option("-t", "--trigger", action="store_true", default=False, dest="trigger", help="Simulate trigger response (simplified)")
parser.add_option("-v", "--verbose", action="store_true", default=False, dest="verbose", help="Increase verbosity")
(options, args) = parser.parse_args()


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
posshift_z  = [  ZCELL*1.5,    -ZCELL*0.5,   ZCELL*0.5,    -ZCELL*1.5 ]
posshift_x  = [  -7.5*XCELL,   -7.5*XCELL,   -7.0*XCELL,   -7.0*XCELL ]



# Generate muons
rnd.seed(options.seed)

# starting orbit counter (arbitrary)
orbit_cnt = 1000000

# dict of tracks, [m, q]
dc = []
# start with run number
dc.append({"HEAD" : 0, 'FPGA' : 0, 'TDC_CHANNEL' : 0, 'ORBIT_CNT' : 0, 'BX_COUNTER' : 0, 'TDC_MEAS' : 0, 'TRG_QUALITY' : options.run})

for i in range(options.number):
	sl = rnd.randint(0, NSL-1)
	orbit_cnt += 1
	bx0 = rnd.randint(1 + 30, DURATION['orbit:bx'] - 30) # avoid first and final BX to avoid cross-orbit muons
	angle = rnd.uniform(math.pi*1./4., math.pi*3./4.) # angle in radians between 45 and 135 degrees
	offset = rnd.uniform(-10*XCELL, 10*XCELL) # x position with respect to the center (x, z) of the chamber
	#tracks.append([sl, angle, inter])
	m = math.tan(angle) # slope
	q = 0. - m * offset # intercept = z - mx

	print "+ Orbit", orbit_cnt, " gen muon in SL", sl, "with angle =", angle, "and offset =", offset

	# Loop on layers
	for l in range(NSL):
		x = (posshift_z[l] - q)/m # invert formula z = mx + q
		wire_num = int(round((x - posshift_x[l]) / XCELL) + 1) # number of wire in the same layer
		wire_pos = (wire_num - 1) * XCELL + posshift_x[l]
		dx = x - wire_pos
		tdrift = abs(dx) / VDRIFT
		bx = tdrift // DURATION['bx'] # number of bunch crossings
		dt = tdrift % DURATION['bx'] # time between bx and the hit
		tdc_meas = int(math.floor(dt * DURATION['tdc']))
		bx_counter = int(bx0 + bx)
		tdc_channel_norm = (wire_num - 1)*4 + (l+1) #layer_z[l] FIXME
		tdc_channel = tdc_channel_norm + NCHANNELS * (sl % 2)
		fpga = 0 if sl < 2 else 1

		if options.verbose: print "LAYER:", l+1, ", X:", x, ", WIRE_NUM:", wire_num, ", DX:", dx, ", TDC_CHANNEL_NORM:", tdc_channel_norm
		
		if wire_num < 1 or wire_num > 16: continue
		dc.append({"HEAD" : 1, 'FPGA' : fpga, 'TDC_CHANNEL' : tdc_channel, 'ORBIT_CNT' : orbit_cnt, 'BX_COUNTER' : bx_counter, 'TDC_MEAS' : tdc_meas, 'TRG_QUALITY' : 0, 'TDC_CHANNEL_NORM' : tdc_channel_norm, 'X' : x, 'DX' : dx, 'DT' : dt, 'TDRIFT' : tdrift, 'WIRE_NUM' : wire_num, 'SL' : sl})
		#dc.append({"HEAD" : 1, 'FPGA' : fpga, 'TDC_CHANNEL' : tdc_channel, 'ORBIT_CNT' : orbit_cnt, 'BX_COUNTER' : bx_counter, 'TDC_MEAS' : tdc_meas, 'TRG_QUALITY' : 0})

	# Simple trigger implementation to determine BX0
	if options.trigger:
		dc.append({"HEAD" : 3, 'FPGA' : 2, 'TDC_CHANNEL' : 0, 'ORBIT_CNT' : orbit_cnt, 'BX_COUNTER' : bx0, 'TDC_MEAS' : bx0, 'TRG_QUALITY' : 1})


columns=['HEAD', 'FPGA', 'TDC_CHANNEL', 'ORBIT_CNT', 'BX_COUNTER', 'TDC_MEAS', 'TRG_QUALITY']

df = pd.DataFrame.from_dict(dc)

if options.verbose: print df.head(50)

if options.plot:

	import matplotlib.pyplot as plt

	# Keep only hits
	hits = df.loc[ df['HEAD']==1 ]

	# Timebox
	#plt.figure(figsize=(15,10))
	#
	#hist, bins = np.histogram(hits['ANGLE'], density=False, bins=100, range=(0., 3.15))
	#plt.subplot(2, 2, 1)
	#plt.xlabel("angle (rad)")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	#hist, bins = np.histogram(hits['OFFSET'], density=False, bins=100, range=(0., NCHANNELS/4.*XCELL))
	#plt.subplot(2, 2, 2)
	#plt.xlabel("offset (mm)")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	#hist, bins = np.histogram(hits['M'], density=False, bins=100, range=(-1.2, 1.2))
	#plt.subplot(2, 2, 3)
	#plt.xlabel("m")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	#hist, bins = np.histogram(hits['Q'], density=False, bins=100, range=(-10000., 10000.))
	#plt.subplot(2, 2, 4)
	#plt.xlabel("q (mm)")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))


	plt.figure(figsize=(15,10))

	hist, bins = np.histogram(hits['X'], density=False, bins=100, range=(-100., 1000))
	plt.subplot(3, 3, 1)
	plt.xlabel("position in the layer (mm)")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(hits['DX'], density=False, bins=100, range=(-25., 25))
	plt.subplot(3, 3, 2)
	plt.xlabel("position in the cell (mm)")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(hits['WIRE_NUM'], density=False, bins=22, range=(-1.5, 20.5))
	plt.subplot(3, 3, 3)
	plt.xlabel("number of wire in the layer")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))


	hist, bins = np.histogram(hits['TDRIFT'], density=False, bins=100, range=(-10, 900))
	plt.subplot(3, 3, 4)
	plt.xlabel("drift time (ns)")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(hits['DT'], density=False, bins=100, range=(-10, 40))
	plt.subplot(3, 3, 5)
	plt.xlabel("drift time in the same BX (ns)")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(hits['TDC_MEAS'], density=False, bins=40, range=(-3.5, 36.5))
	plt.subplot(3, 3, 6)
	plt.xlabel("TDC measurement")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))


	hist, bins = np.histogram(hits['TDC_CHANNEL_NORM'], density=False, bins=NCHANNELS+1, range=(-0.5, NCHANNELS+0.5))
	plt.subplot(3, 3, 7)
	plt.xlabel("TDC CHANNEL NORM")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(hits['TDC_CHANNEL'], density=False, bins=NCHANNELS*2+1, range=(-0.5, NCHANNELS*2+0.5))
	plt.subplot(3, 3, 8)
	plt.xlabel("TDC CHANNEL")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	#hist, bins = np.histogram(hits['BX_COUNTER'], density=False, bins=DURATION['orbit:bx']+1, range=(-0.5, DURATION['orbit:bx']+0.5))
	#plt.subplot(3, 3, 9)
	#plt.xlabel("bunch crossing number")
	#plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))

	hist, bins = np.histogram(hits['FPGA'], density=False, bins=5, range=(-0.5, 4.5))
	plt.subplot(3, 3, 9)
	plt.xlabel("FPGA")
	plt.bar((bins[:-1] + bins[1:]) / 2, hist, align='center', width=np.diff(bins))


	plt.savefig(os.path.dirname(options.outputfile) + "/generation.png")
	plt.savefig(os.path.dirname(options.outputfile) + "/generation.pdf")
	if options.verbose: print "Control plots saved in", os.path.dirname(options.outputfile)


# filter and save
df = df[columns]
outName = options.outputfile
#outName = outName.replace(".csv", "%6.0f.csv" % options.run)
df.to_csv(outName, columns=columns, header=False, index=False)


