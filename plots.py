#!/usr/bin/env python
# coding: utf-8

import os, sys, math, time, glob
import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
from modules.mapping.config import TDRIFT, VDRIFT, DURATION, TIME_WINDOW, XCELL, ZCELL
from modules.reco import config_1_2_1 as config

def gaus(x, n, mean, sigma):
    return (n/np.sqrt(2.*np.pi)) * np.exp(-(x - mean)**2.0 / (2 * sigma**2))

def gausc(x, n, mean, sigma, bkg):
    return (n/np.sqrt(2.*np.pi)) * np.exp(-(x - mean)**2.0 / (2 * sigma**2)) + bkg


def fitGaus(axs, ybins, xbins, patches):
	if np.sum(ybins) <= 0.: return
	cbins = xbins[:-1] + np.diff(xbins) / 2.
	xmin, xmax = axs.get_xlim()
	cfbins = cbins[(cbins < xmax) & (cbins > xmin)]
	yfbins = ybins[int((len(ybins) - len(cfbins)) / 2) + 1 : -int((len(ybins) - len(cfbins)) / 2)]
	mean = np.average(cbins, weights=ybins)
	rms = np.sqrt(np.average((cbins - mean)**2, weights=ybins))
	(gaus_norm, gaus_mean, gaus_sigma), cov = curve_fit(gaus, cbins, ybins, p0=[np.sum(ybins), mean, rms])
	gaus_x = np.linspace(xmin, xmax, 1000)
	gaus_y = gaus(gaus_x, gaus_norm, gaus_mean, gaus_sigma)
	axs.plot(gaus_x, gaus_y, '-', linewidth=2, color='red', label=("gaus"))
	axs.text(axs.get_xlim()[1]*1/2, axs.get_ylim()[1]*3/4, "$x_0 = %.4f$" % (gaus_mean))
	axs.text(axs.get_xlim()[1]*1/2, axs.get_ylim()[1]*5/8, "$\sigma = %.4f$" % (gaus_sigma))
	return gaus_norm, gaus_mean, gaus_sigma


import argparse
parser = argparse.ArgumentParser(description='Command line arguments')
parser.add_argument("-i", "--inputfile", action="store", type=str, dest="inputdir", default="./output/Run000967_csv/", help="Provide directory of the input files (csv)")
parser.add_argument("-o", "--outputdir", action="store", type=str, dest="outputdir", default="./output/", help="Specify output directory")
parser.add_argument("-p", "--plot", action="store", type=str, dest="plot", default="occupancy,timebox,parameters,alignment,residuals,missinghits", help="Specify output directory")
parser.add_argument("-v", "--verbose", action="store", type=int, default=0, dest="verbose", help="Specify verbosity level")
args = parser.parse_args()

runname = [x for x in args.inputdir.split('/') if 'Run' in x][0].replace("_csv", "") if "Run" in args.inputdir else "Run000000"


if not os.path.exists(args.inputdir + "/events.csv") or not os.path.exists(args.inputdir + "/segments.csv"):
	print("One or more input files not found.")
	exit()

if args.verbose >= 1: print("Writing plots in directory %s" % (args.outputdir + runname))


#hits = pd.read_csv(args.inputdir + "/events.csv", skiprows=0, low_memory=False)
#segments = pd.read_csv(args.inputdir + "/segments.csv", dtype={'CHAMBER': object}, skiprows=0, low_memory=False)
#missinghits = pd.read_csv(args.inputdir + "/missinghits.csv", skiprows=0, low_memory=False)
#occupancy = pd.read_csv(args.inputdir + "/occupancy.csv", skiprows=0, low_memory=False)

hits = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "events*.csv"))))
segments = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "segments*.csv"))))
missinghits = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "missinghits*.csv"))))
occupancy = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "occupancy*.csv"))))



if args.verbose >= 1: print("Read %d lines from events.csv" % (len(hits), ))
if args.verbose >= 2: print(hits.head(50))
if args.verbose >= 1: print("Read %d lines from segments.csv" % (len(segments), ))
if args.verbose >= 2:
	print(segments.head(50))
	print(segments.tail(50))

# Recalculate rate in case of multiple files
runtime = (hits['ORBIT'].max() - hits['ORBIT'].min()) * DURATION['orbit'] * 1.e-9 # Approximate acquisition time in seconds
occupancy['RATE'] = occupancy['COUNTS'] / runtime

hits = hits[hits['T0'].notnull()]

hits['X_DRIFT'] = hits['TIMENS'] * VDRIFT

# Residuals
hits['RESIDUES_SEG'] = np.abs(hits['X'] - hits['X_WIRE']) - np.abs(hits['X_SEG'] - hits['X_WIRE'])
hits['DIST_SEG_WIRE'] = hits['X_SEG'] - hits['X_WIRE']
hits["ABS_DIST_SEG_WIRE"] = np.abs(hits["DIST_SEG_WIRE"])

# Residuals from tracks
hits['RESIDUES_TRACK'] = np.abs(hits['X'] - hits['X_WIRE']) - np.abs(hits['X_TRACK'] - hits['X_WIRE'])
hits['DIST_TRACK_WIRE'] = hits['X_TRACK'] - hits['X_WIRE']
hits["ABS_DIST_TRACK_WIRE"] = np.abs(hits["DIST_TRACK_WIRE"])

# Disentangle tracks and segments
parlabel = {'ANGLE_RAD' : 'Angle (rad)', 'ANGLE_DEG' : 'Angle ($^o$)', 'Q' : 'Intercept', 'M' : 'Angular coefficient', 'X0' : '$x_{0}$ (mm)', 'NHITS' : 'number of hits'}
parbins = {'ANGLE_RAD' : np.arange(-np.pi, np.pi, 0.1), 'ANGLE_DEG' : np.arange(-50., 50., 1.), 'Q' : np.arange(-10000., 10000., 100.), 'M' : np.arange(-1000., 1000., 10.), 'X0' : np.arange(-336., 336., 8.), 'NHITS' : np.arange(-0.5, 12.5, 1)}
segments['ANGLE_RAD'] = np.arctan(segments['M'])
segments.loc[segments['ANGLE_RAD'] < 0., 'ANGLE_RAD'] = segments.loc[segments['ANGLE_RAD'] < 0., 'ANGLE_RAD'] + np.pi # shift angle by pi
segments['ANGLE_DEG'] = np.degrees(segments['ANGLE_RAD']) - 90.
segments['X0'] = - segments['Q'] / segments['M']

tracks = segments.loc[segments['VIEW'] != '0']
segments = segments.loc[segments['VIEW'] == '0']

print("Numer of global tracks:", len(tracks[tracks['CHAMBER'].str.len() > 1]))

segments['XL1'] = -(-19.5 + segments['Q']) / segments['M']
segments['XL2'] = -(- 6.5 + segments['Q']) / segments['M']
segments['XL3'] = -(+ 6.5 + segments['Q']) / segments['M']
segments['XL4'] = -(+19.5 + segments['Q']) / segments['M']


# Subplot indices
iax = [(i, j) for i in range(2) for j in range(2)]
isl = range(4)
ila = range(1, 4+1)

chi2bins = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])

# Occupancy
if 'occupancy' in args.plot:
	fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
	for chamber in range(4):
		axs[-chamber-1].set_title("Occupancy [SL %d]" % (chamber))
		axs[-chamber-1].set_xlabel("Wire number")
		axs[-chamber-1].set_xticks(range(1, 17))
		axs[-chamber-1].set_ylabel("Layer")
		axs[-chamber-1].set_yticks(range(1, 5))
		h = axs[-chamber-1].hist2d(occupancy.loc[occupancy['SL'] == chamber, 'WIRE_NUM'], occupancy.loc[occupancy['SL'] == chamber, 'LAYER'], weights=occupancy.loc[occupancy['SL'] == chamber, 'COUNTS'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]]) #, vmin=0, vmax=np.max(occupancy['COUNTS']))
		cbar = fig.colorbar(h[3], ax=axs[-chamber-1], pad=0.01)
		cbar.set_label("Counts")
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/occupancy.png")
	fig.savefig(args.outputdir + runname + "_plots/occupancy.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for chamber in range(4):
		for layer in range(4):
			occ = occupancy[((occupancy['SL'] == chamber) & (occupancy['LAYER'] == layer+1) & (occupancy['WIRE_NUM'] <= 16))]
			axs[chamber][layer].set_title("Occupancy [SL %d, LAYER %d]" % (chamber, layer+1))
			axs[chamber][layer].set_xlabel("Wire number")
			axs[chamber][layer].set_ylabel("Counts")
			axs[chamber][layer].set_xticks(range(1, 17))
			axs[chamber][layer].bar(occ['WIRE_NUM'], occ['COUNTS'])
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/occupancy_wire.png")
	fig.savefig(args.outputdir + runname + "_plots/occupancy_wire.pdf")
	plt.close(fig)


	fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
	for chamber in range(4):
		axs[-chamber-1].set_title("Rate [SL %d]" % (chamber))
		axs[-chamber-1].set_xlabel("Wire number")
		axs[-chamber-1].set_xticks(range(1, 17))
		axs[-chamber-1].set_ylabel("Layer")
		axs[-chamber-1].set_yticks(range(1, 5))
		h = axs[-chamber-1].hist2d(occupancy.loc[occupancy['SL'] == chamber, 'WIRE_NUM'], occupancy.loc[occupancy['SL'] == chamber, 'LAYER'], weights=occupancy.loc[occupancy['SL'] == chamber, 'RATE'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]]) #, vmin=0, vmax=np.max(occupancy['RATE']))
		cbar = fig.colorbar(h[3], ax=axs[-chamber-1], pad=0.01)
		cbar.set_label("Rate (Hz)")
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/rate.png")
	fig.savefig(args.outputdir + runname + "_plots/rate.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for chamber in range(4):
		for layer in range(4):
			occ = occupancy[((occupancy['SL'] == chamber) & (occupancy['LAYER'] == layer+1) & (occupancy['WIRE_NUM'] <= 16))]
			axs[chamber][layer].set_title("Rate [SL %d, LAYER %d]" % (chamber, layer+1))
			axs[chamber][layer].set_xlabel("Wire number")
			axs[chamber][layer].set_ylabel("Rate (Hz)")
			axs[chamber][layer].set_xticks(range(1, 17))
			axs[chamber][layer].bar(occ['WIRE_NUM'], occ['RATE'])
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/rate_wire.png")
	fig.savefig(args.outputdir + runname + "_plots/rate_wire.pdf")
	plt.close(fig)
	'''
	fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
	nmax = occ.groupby(['SL', 'WIRE_NUM', 'LAYER'])['COUNTS'].max().max()
	print(nmax)
	for chamber in range(4):
		axs[chamber].set_title("Occupancy [SL %d]" % (chamber))
		axs[chamber].set_xlabel("Wire number")
		axs[chamber].set_xticks(range(1, 17))
		axs[chamber].set_ylabel("Layer")
		axs[chamber].set_yticks(range(1, 5))
		h = axs[chamber].hist2d(occ.loc[occ['SL'] == chamber, 'WIRE_NUM'], occ.loc[occ['SL'] == chamber, 'LAYER'], weights=occ.loc[occ['SL'] == chamber, 'COUNTS'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]], vmin=0, vmax=nmax)
		#fig.colorbar(h[3], ax=axs[chamber])
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/occupancy2d.png")
	fig.savefig(args.outputdir + runname + "_plots/occupancy2d.pdf")
	plt.close(fig)
	'''
	'''
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		axs[idx].set_title("Multiplicity [SL %d]" % chamber)
		axs[idx].set_xlabel("Number of hits per SL")
		axs[idx].set_ylabel("Counts")
		axs[idx].set_xticks(range(0, 20+1))
		axs[idx].hist(hits[(hits['CHAMBER'] == chamber)].groupby('ORBIT')['ORBIT'].count(), bins=np.arange(0, 20+1, 1))
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/multiplicity.png")
	fig.savefig(args.outputdir + runname + "_plots/multiplicity.pdf")
	plt.close(fig)
	'''

# Timebox
if 'timebox' in args.plot:

	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		axs[idx].set_title("Timebox [chamber %d]" % chamber)
		axs[idx].set_xlabel("Time (ns)")
		axs[idx].set_ylabel("Counts")
		axs[idx].hist(hits.loc[hits['CHAMBER']==chamber, 'TIMENS'], bins=np.arange(-50, 550+1, 5), label='Drift time (ns)')
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/timebox.png")
	fig.savefig(args.outputdir + runname + "_plots/timebox.pdf")
	plt.close(fig)

	'''
	# Space boxes
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		axs[idx].set_title("Space box LEFT [chamber %d]" % chamber)
		axs[idx].set_xlabel("Position (mm)")
		axs[idx].set_ylabel("Counts")
		axs[idx].hist(hits.loc[hits['CHAMBER']==chamber, 'X_LEFT'], bins=np.arange(-50, 50+1, 1), color='green', alpha=0.5, label='LEFT')
		axs[idx].hist(hits.loc[hits['CHAMBER']==chamber, 'X_RIGHT'], bins=np.arange(-50, 50+1, 1), color='red', alpha=0.5, label='RIGHT')
		axs[idx].legend()
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/spacebox_left_right.png")
	fig.savefig(args.outputdir + runname + "_plots/spacebox_left_right.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		axs[idx].set_title("Space box [chamber %d]" % chamber)
		axs[idx].set_xlabel("Position (mm)")
		axs[idx].set_ylabel("Counts")
		axs[idx].hist(hits.loc[hits['CHAMBER']==chamber, 'X'], bins=np.arange(-378, +378+1, 1), label='')
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/spacebox.png")
	fig.savefig(args.outputdir + runname + "_plots/spacebox.pdf")
	plt.close(fig)
	'''
	'''
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		axs[idx].set_title("Hit LEFT position [chamber %d]" % chamber)
		axs[idx].set_xlabel("X position (mm)")
		axs[idx].set_ylabel("Z position (mm)")
		axs[idx].scatter(hits.loc[hits['CHAMBER']==chamber, 'X_LEFT'], hits.loc[hits['CHAMBER']==chamber, 'Z']+1, s=0.2, c='green', alpha=0.5, label='LEFT')
		axs[idx].scatter(hits.loc[hits['CHAMBER']==chamber, 'X_RIGHT'], hits.loc[hits['CHAMBER']==chamber, 'Z']-1, s=0.2, c='red', alpha=0.5, label='RIGHT')
		axs[idx].legend()
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/x_left_right_position.png")
	fig.savefig(args.outputdir + runname + "_plots/x_left_right_position.pdf")
	plt.close(fig)
	'''
	'''
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		axs[idx].set_title("Hit position (after L/R disambiguation) [chamber %d]" % chamber)
		axs[idx].set_xlabel("X position (mm)")
		axs[idx].set_ylabel("Z position (mm)")
		axs[idx].scatter(hits.loc[hits['CHAMBER']==chamber, 'X'], hits.loc[hits['CHAMBER']==chamber, 'Z'], s=1)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/x_position.png")
	fig.savefig(args.outputdir + runname + "_plots/x_position.pdf")
	plt.close(fig)
	'''
	nmax = hits.groupby(['CHAMBER', 'WIRE', 'LAYER']).size().max()
	fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
	for chamber in range(4):
		axs[-chamber-1].set_title("Hit position (after L/R disambiguation) [chamber %d]" % chamber)
		axs[-chamber-1].set_xlabel("X position (mm)")
		axs[-chamber-1].set_ylabel("Z position (mm)")
		#axs[-chamber-1].scatter(hits.loc[hits['CHAMBER']==chamber, 'X'], hits.loc[hits['CHAMBER']==chamber, 'Z'], s=1)
		h = axs[-chamber-1].hist2d(hits.loc[hits['CHAMBER']==chamber, 'X'], hits.loc[hits['CHAMBER']==chamber, 'Z'], bins=[17*42, 4], range=[[-8.5*42., +8.5*42.], [-26., 26.]], vmin=0, vmax=nmax/42)
		cbar = fig.colorbar(h[3], ax=axs[-chamber-1], pad=0.01)
		cbar.set_label("Counts")
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/hits_position.png")
	fig.savefig(args.outputdir + runname + "_plots/hits_position.pdf")
	plt.close(fig)


# Parameters
if 'parameters' in args.plot:
	
	# Chi2
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		axs[idx].set_title("Segments $\chi^2$ [SL %d]" % chamber)
		axs[idx].set_xlabel("$\chi^2$")
		axs[idx].set_ylabel("Counts")
		axs[idx].hist(segments.loc[segments['CHAMBER'].astype('int32') == chamber, 'CHI2'], bins=chi2bins)
		axs[idx].set_xscale("log")
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/segment_chi2.png")
	fig.savefig(args.outputdir + runname + "_plots/segment_chi2.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
	for w, view in enumerate(['XZ', 'YZ']):
		axs[w].set_title("Track $\chi^2$ [%s]" % view)
		axs[w].set_xlabel("$\chi^2$")
		axs[w].set_ylabel("Counts")
		axs[w].hist(tracks.loc[(tracks['VIEW'] == view), 'CHI2'], bins=chi2bins)
		axs[w].set_xscale("log")
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/track_chi2.png")
	fig.savefig(args.outputdir + runname + "_plots/track_chi2.pdf")
	plt.close(fig)


	# Parameters
	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for i, p in enumerate(['ANGLE_DEG', 'Q', 'X0', 'NHITS']):
	    for chamber in range(4):
	    	axs[i][chamber].set_title("Segments %s [SL %d]" % (p, chamber))
	    	axs[i][chamber].set_xlabel(parlabel[p])
	    	axs[i][chamber].hist(segments.loc[segments['CHAMBER'].astype('int32') == chamber, p], bins=parbins[p])
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/segment_par.png")
	fig.savefig(args.outputdir + runname + "_plots/segment_par.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 10))
	for w, view in enumerate(['XZ', 'YZ']):
		for i, p in enumerate(['ANGLE_DEG', 'Q', 'X0', 'NHITS']):
			axs[i][w].set_title("Tracks %s [%s]" % (p, view))
			axs[i][w].set_xlabel(parlabel[p])
			axs[i][w].hist(tracks[p], bins=parbins[p])
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/tracks_par.png")
	fig.savefig(args.outputdir + runname + "_plots/tracks_par.pdf")
	plt.close(fig)
	
	# X0
	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
		chamber = int(cl % 4)
		layer = cl // 4 + 1
		axs[idx].set_title("Segment $x_{l}$ [CHAMBER %d, LAYER %d]" % (chamber, layer))
		axs[idx].set_xlabel(parlabel[p])
		axs[idx].hist(segments.loc[(segments['CHAMBER'] == str(chamber)), 'XL%d' % layer].dropna(), bins=parbins['X0'])
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/segment_x0_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/segment_x0_chamber_layer.pdf")
	plt.close(fig)


if 'residuals' in args.plot:
	'''
	# Residuals from Segments
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		axs[idx].set_title("Local residues [SL %d]" % chamber)
		axs[idx].set_xlabel("|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
		ybins, xbins, patches = axs[idx].hist(hits.loc[hits['CHAMBER'] == chamber, 'RESIDUES_SEG'].dropna(), bins=np.arange(-4., 4., 0.05))
		fitGaus(axs[idx], ybins, xbins, patches)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_chamber.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_chamber.pdf")
	plt.close(fig)


	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, layer in zip(iax, ila):
		axs[idx].set_title("Local residues [LAYER %d]" % layer)
		axs[idx].set_xlabel("|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
		ybins, xbins, patches = axs[idx].hist(hits.loc[hits['LAYER'] == layer, 'RESIDUES_SEG'].dropna(), bins=np.arange(-4., 4., 0.05))
		fitGaus(axs[idx], ybins, xbins, patches)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_layer.pdf")
	plt.close(fig)


	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
		chamber = int(cl % 4)
		layer = cl // 4 + 1
		hitsl = hits[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer)]
		axs[idx].set_title("Local residues [CHAMBER %d, LAYER %d]" % (chamber, layer))
		axs[idx].set_xlabel("|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
		ybins, xbins, patches = axs[idx].hist(hitsl['RESIDUES_SEG'].dropna(), bins=np.arange(-4., 4., 0.05))
		fitGaus(axs[idx], ybins, xbins, patches)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_chamber_layer.pdf")
	plt.close(fig)


	# Residuals from Tracks
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		axs[idx].set_title("Global residues [SL %d]" % chamber)
		axs[idx].set_xlabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
		ybins, xbins, patches = axs[idx].hist(hits.loc[hits['CHAMBER'] == chamber, 'RESIDUES_TRACK'].dropna(), bins=np.arange(-4., 4., 0.05))
		fitGaus(axs[idx], ybins, xbins, patches)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_track_chamber.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_track_chamber.pdf")
	plt.close(fig)


	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, layer in zip(iax, ila):
		axs[idx].set_title("Global residues [LAYER %d]" % layer)
		axs[idx].set_xlabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
		ybins, xbins, patches = axs[idx].hist(hits.loc[hits['LAYER'] == layer, 'RESIDUES_TRACK'].dropna(), bins=np.arange(-4., 4., 0.05))
		fitGaus(axs[idx], ybins, xbins, patches)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_track_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_track_layer.pdf")
	plt.close(fig)


	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
		chamber = int(cl % 4)
		layer = cl // 4 + 1
		axs[idx].set_title("Global residues [CHAMBER %d, LAYER %d]" % (chamber, layer))
		axs[idx].set_xlabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
		ybins, xbins, patches = axs[idx].hist(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'RESIDUES_TRACK'].dropna(), bins=np.arange(-4., 4., 0.05))
		fitGaus(axs[idx], ybins, xbins, patches)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_track_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_track_chamber_layer.pdf")
	plt.close(fig)


	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
		chamber = int(cl % 4)
		layer = cl // 4 + 1
		axs[idx].set_title("Global LEFT residues [CHAMBER %d, LAYER %d]" % (chamber, layer))
		axs[idx].set_xlabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
		ybins, xbins, patches = axs[idx].hist(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer) & (hits['X_LABEL'] == 1), 'RESIDUES_TRACK'].dropna(), bins=np.arange(-4., 4., 0.05))
		fitGaus(axs[idx], ybins, xbins, patches)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_track_left_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_track_left_chamber_layer.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
		chamber = int(cl % 4)
		layer = cl // 4 + 1
		axs[idx].set_title("Global RIGHT residues [CHAMBER %d, LAYER %d]" % (chamber, layer))
		axs[idx].set_xlabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
		ybins, xbins, patches = axs[idx].hist(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer) & (hits['X_LABEL'] == 2), 'RESIDUES_TRACK'].dropna(), bins=np.arange(-4., 4., 0.05))
		fitGaus(axs[idx], ybins, xbins, patches)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_track_right_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_track_right_chamber_layer.pdf")
	plt.close(fig)
	'''

	'''
	# Residues from Segments 2D
	jgrid = sns.jointplot(data=hits, dropna=True, x="TIMENS", y="RESIDUES_SEG", kind="reg", scatter_kws={'s': 0.1})
	jgrid.set_axis_labels("$t_{drift}$ (ns)", "|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
	jgrid.ax_joint.set_ylim(-2., 2)
	jgrid.ax_marg_y.set_ylim(-2., 2)
	jgrid.savefig(args.outputdir + runname + "_plots/residues_seg_tdrift_joint.png")
	jgrid.savefig(args.outputdir + runname + "_plots/residues_seg_tdrift_joint.pdf")

	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		jgrid = sns.jointplot(ax=axs[idx], data=hits.loc[hits['CHAMBER'] == chamber, ['RESIDUES_SEG', 'TIMENS']].dropna(how="any"), x="TIMENS", y="RESIDUES_SEG", kind="reg", scatter_kws={'s': 0.1})
		jgrid.set_axis_labels("$t_{drift}$ (ns)", "$x$ - $x_{fit}$ (mm)")
		jgrid.ax_joint.set_xlabel("$t_{drift}$ (ns)")
		jgrid.ax_joint.set_ylabel("|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
		jgrid.ax_joint.set_xlim(-50., 500)
		jgrid.ax_joint.set_ylim(-2.5, +2.5)
		jgrid.ax_marg_y.set_ylim(-2.5, +2.5)
		jgrid = sns.jointplot(ax=axs[idx], data=hits.loc[hits['CHAMBER'] == chamber, ['RESIDUES_SEG', 'TIMENS']].dropna(how="any"), x="TIMENS", y="RESIDUES_SEG", kind="hist")
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_tdrift.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_tdrift.pdf")
	plt.close(fig)


	#jgrid = sns.jointplot(x=hits["ABS_DIST_SEG_WIRE"], y=hits["RESIDUES_SEG"], dropna=True, kind="reg", scatter_kws={'s': 0.1})
	#plt.errorbar(hits["ABS_DIST_SEG_WIRE"], hits["RESIDUES_SEG"], color="red", linestyle = 'none', marker="o")
	jgrid = sns.jointplot(x=hits["ABS_DIST_SEG_WIRE"], y=hits["RESIDUES_SEG"], dropna=True, kind="hist")
	jgrid.plot_joint(sns.regplot, order=1, scatter=False) #, scatter_kws=dict(alpha=0)
	jgrid.set_axis_labels("|$d_{seg}$| (mm)", "|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
	jgrid.ax_joint.set_ylim(-2., 2)
	jgrid.ax_marg_y.set_ylim(-2., 2)
	jgrid.ax_joint.set_xlim(0., 21.)
	jgrid.ax_marg_x.set_xlim(0., 21.)
	jgrid.savefig(args.outputdir + runname + "_plots/residues_seg_pos_joint.png")
	jgrid.savefig(args.outputdir + runname + "_plots/residues_seg_pos_joint.pdf")


	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, chamber in zip(iax, isl):
		jgrid = sns.jointplot(ax=axs[idx], x=hits["ABS_DIST_SEG_WIRE"], y=hits["RESIDUES_SEG"], dropna=True, kind="hist")
		jgrid.plot_joint(sns.regplot, ax=axs[idx], order=1, scatter=False)
		axs[idx].set_title("Residues vs position [SL %d]" % chamber)
		axs[idx].set_xlabel("|$d_{seg}$| (mm)")
		axs[idx].set_ylabel("|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
		axs[idx].set_xlim(0., 21.)
		axs[idx].set_xlim(0., 21.)
		axs[idx].set_ylim(-2., +2.)
		axs[idx].set_ylim(-2., +2.)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_pos_chamber.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_pos_chamber.pdf")
	plt.close(fig)
	'''
	'''
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	for idx, layer in zip(iax, ila):
		hitsl = hits[hits['LAYER'] == layer]
		jgrid = sns.jointplot(ax=axs[idx], x=hitsl["ABS_DIST_SEG_WIRE"], y=hitsl["RESIDUES_SEG"], dropna=True, kind="hist")
		jgrid.plot_joint(sns.regplot, ax=axs[idx], order=1, scatter=False, color=".3")
		axs[idx].set_title("Residues vs position [LAYER %d]" % layer)
		axs[idx].set_xlabel("|$d_{seg}$| (mm)")
		axs[idx].set_ylabel("|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
		axs[idx].set_xlim(0., 21.)
		axs[idx].set_ylim(-2., +2.)
		hitsl = hitsl[(hitsl["ABS_DIST_SEG_WIRE"] > 4.) & (hitsl["ABS_DIST_SEG_WIRE"] < 18.)]
		p0, p1, rval, pval, p0err = sp.linregress(hitsl["ABS_DIST_SEG_WIRE"], hitsl["RESIDUES_SEG"])
		sns.lineplot(x=np.linspace(4, 18), y=p1 + p0 * np.linspace(4, 18), ax=axs[idx], linewidth=2, color="red", ci=None)
		#print("Layer", layer, ": p0 = %.4f +- %.4f" % (p0, p0err), ", p1 = %.4f" % p1, ", p0 std. dev. = %.1f" % (p0/p0err))
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_pos_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_pos_layer.pdf")
	plt.close(fig)
	'''
	'''
	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
		chamber = int(cl % 4)
		layer = cl // 4 + 1
		hitsl = hits[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer)]
		jgrid = sns.jointplot(ax=axs[idx], x=hitsl["ABS_DIST_SEG_WIRE"], y=hitsl["RESIDUES_SEG"], dropna=True, kind="hist")
		jgrid.plot_joint(sns.regplot, ax=axs[idx], order=1, scatter=False, color=".3")
		axs[idx].set_title("Residues vs d [CHAMBER %d, LAYER %d]" % (chamber, layer))
		axs[idx].set_xlabel("|$d_{seg}$| (mm)")
		axs[idx].set_ylabel("|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
		axs[idx].set_xlim(0., 21.)
		axs[idx].set_ylim(-2., +2.)
		hitsl = hitsl[(hitsl["ABS_DIST_SEG_WIRE"] > 4.) & (hitsl["ABS_DIST_SEG_WIRE"] < 18.)]
		#p0, p1 = np.polyfit(hitsl["ABS_DIST_SEG_WIRE"], hitsl["RESIDUES_SEG"], 1, full=False)
		p0, p1, rval, pval, p0err = sp.linregress(hitsl["ABS_DIST_SEG_WIRE"], hitsl["RESIDUES_SEG"])
		sns.lineplot(x=np.linspace(4, 18), y=p1 + p0 * np.linspace(4, 18), ax=axs[idx], linewidth=2, color="red", ci=None)
		#print("Chamber", chamber, ", Layer", layer, ": p0 = %.4f +- %.4f" % (p0, p0err), ", p1 = %.4f" % (p1), ", p0 std. dev. = %.1f" % (p0/p0err))
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_pos_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_seg_pos_chamber_layer.pdf")
	plt.close(fig)


	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
		chamber = int(cl % 4)
		layer = cl // 4 + 1
		hitsl = hits[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer)]
		jgrid = sns.jointplot(ax=axs[idx], x=hitsl["ABS_DIST_TRACK_WIRE"], y=hitsl["RESIDUES_TRACK"], dropna=True, kind="hist")
		jgrid.plot_joint(sns.regplot, ax=axs[idx], order=1, scatter=False, color=".3")
		axs[idx].set_title("Global residues vs d [CHAMBER %d, LAYER %d]" % (chamber, layer))
		axs[idx].set_xlabel("|$d_{track}$| (mm)")
		axs[idx].set_ylabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
		axs[idx].set_xlim(0., 21.)
		axs[idx].set_ylim(-2., +2.)
		if chamber == 2: axs[idx].set_ylim(-10., +10.)
		hitsl = hitsl[(hitsl["ABS_DIST_TRACK_WIRE"] > 4.) & (hitsl["ABS_DIST_TRACK_WIRE"] < 18.)]
		#p0, p1 = np.polyfit(hitsl["ABS_DIST_SEG_WIRE"], hitsl["RESIDUES_SEG"], 1, full=False)
		p0, p1, rval, pval, p0err = sp.linregress(hitsl["ABS_DIST_TRACK_WIRE"], hitsl["RESIDUES_TRACK"])
		sns.lineplot(x=np.linspace(4, 18), y=p1 + p0 * np.linspace(4, 18), ax=axs[idx], linewidth=2, color="red", ci=None)
		#print("Chamber", chamber, ", Layer", layer, ": p0 = %.4f +- %.4f" % (p0, p0err), ", p1 = %.4f" % (p1), ", p0 std. dev. = %.1f" % (p0/p0err))
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_track_pos_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_track_pos_chamber_layer.pdf")
	plt.close(fig)
	'''
	'''
	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
		chamber = int(cl % 4)
		layer = cl // 4 + 1
		hitsl = hits[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer) & (hits['X_LABEL'] == 1)]
		jgrid = sns.jointplot(ax=axs[idx], x=hitsl["ABS_DIST_TRACK_WIRE"], y=hitsl["RESIDUES_TRACK"], dropna=True, kind="hist")
		jgrid.plot_joint(sns.regplot, ax=axs[idx], order=1, scatter=False, color=".3")
		axs[idx].set_title("Global LEFT residues vs d [CHAMBER %d, LAYER %d]" % (chamber, layer))
		axs[idx].set_xlabel("|$d_{track}$| (mm)")
		axs[idx].set_ylabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
		axs[idx].set_xlim(0., 21.)
		axs[idx].set_ylim(-2., +2.)
		if chamber == 2: axs[idx].set_ylim(-10., +10.)
		hitsl = hitsl[(hitsl["ABS_DIST_TRACK_WIRE"] > 4.) & (hitsl["ABS_DIST_TRACK_WIRE"] < 18.)]
		#p0, p1 = np.polyfit(hitsl["ABS_DIST_SEG_WIRE"], hitsl["RESIDUES_SEG"], 1, full=False)
		p0, p1, rval, pval, p0err = sp.linregress(hitsl["ABS_DIST_TRACK_WIRE"], hitsl["RESIDUES_TRACK"])
		sns.lineplot(x=np.linspace(4, 18), y=p1 + p0 * np.linspace(4, 18), ax=axs[idx], linewidth=2, color="red", ci=None)
		#print("Chamber", chamber, ", Layer", layer, ": p0 = %.4f +- %.4f" % (p0, p0err), ", p1 = %.4f" % (p1), ", p0 std. dev. = %.1f" % (p0/p0err))
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_track_pos_left_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_track_pos_left_chamber_layer.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
		chamber = int(cl % 4)
		layer = cl // 4 + 1
		hitsl = hits[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer) & (hits['X_LABEL'] == 2)]
		jgrid = sns.jointplot(ax=axs[idx], x=hitsl["ABS_DIST_TRACK_WIRE"], y=hitsl["RESIDUES_TRACK"], dropna=True, kind="hist")
		jgrid.plot_joint(sns.regplot, ax=axs[idx], order=1, scatter=False, color=".3")
		axs[idx].set_title("Global RIGHT residues vs d [CHAMBER %d, LAYER %d]" % (chamber, layer))
		axs[idx].set_xlabel("|$d_{track}$| (mm)")
		axs[idx].set_ylabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
		axs[idx].set_xlim(0., 21.)
		axs[idx].set_ylim(-2., +2.)
		if chamber == 2: axs[idx].set_ylim(-10., +10.)
		hitsl = hitsl[(hitsl["ABS_DIST_TRACK_WIRE"] > 4.) & (hitsl["ABS_DIST_TRACK_WIRE"] < 18.)]
		#p0, p1 = np.polyfit(hitsl["ABS_DIST_SEG_WIRE"], hitsl["RESIDUES_SEG"], 1, full=False)
		p0, p1, rval, pval, p0err = sp.linregress(hitsl["ABS_DIST_TRACK_WIRE"], hitsl["RESIDUES_TRACK"])
		sns.lineplot(x=np.linspace(4, 18), y=p1 + p0 * np.linspace(4, 18), ax=axs[idx], linewidth=2, color="red", ci=None)
		#print("Chamber", chamber, ", Layer", layer, ": p0 = %.4f +- %.4f" % (p0, p0err), ", p1 = %.4f" % (p1), ", p0 std. dev. = %.1f" % (p0/p0err))
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_track_pos_right_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_track_pos_right_chamber_layer.pdf")
	plt.close(fig)
	'''
	'''
	fig, axs = plt.subplots(nrows=16, ncols=1, figsize=(20, 10))
	for cl in range(4*4 -1, 0 -1, -1):
		layer = int(cl % 4) + 1
		chamber = cl // 4
		if cl==15: axs[-cl-1].set_title("Global residues vs position")
		if cl==0: axs[-cl-1].set_xlabel("$x_{hit}$ (mm)")
		if layer==2 and chamber==1: axs[-cl-1].set_ylabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
		h = axs[-cl-1].hist2d(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X'], hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'RESIDUES_TRACK'], bins=[33*21, 40], range=[[-8.0*42., +8.5*42.], [-2., 2.]])
		#cbar = fig.colorbar(h[3], ax=axs[-cl-1], pad=0.01)
		#cbar.set_label("Counts")
		for c in range(-8, +9):
			axs[-cl-1].axvline(c * 42. + (21. if layer % 2 ==1 else 0.), axs[-cl-1].get_ylim()[0], axs[-cl-1].get_ylim()[1], linewidth=2, color='white')
	#fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/residues_track_x_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/residues_track_x_chamber_layer.pdf")
	plt.close(fig)
	'''
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
	axs[0][0].set_title("LEFT residues 100 < X < 200")
	axs[0][0].hist(hits.loc[(hits['CHAMBER'] == 3) & (hits['LAYER'] == 1) & (hits['X_LABEL'] == 1) & (hits['X'] < 200) & (hits['X'] > 100), 'RESIDUES_TRACK'], bins=np.arange(-4., 4., 0.05))
	axs[0][1].set_title("RIGHT residues 100 < X < 200")
	axs[0][1].hist(hits.loc[(hits['CHAMBER'] == 3) & (hits['LAYER'] == 1) & (hits['X_LABEL'] == 2) & (hits['X'] < 200) & (hits['X'] > 100), 'RESIDUES_TRACK'], bins=np.arange(-4., 4., 0.05))
	axs[1][0].set_title("LEFT residues X < 50")
	axs[1][0].hist(hits.loc[(hits['CHAMBER'] == 3) & (hits['LAYER'] == 1) & (hits['X_LABEL'] == 1) & (hits['X'] < 50), 'RESIDUES_TRACK'], bins=np.arange(-4., 4., 0.05))
	axs[1][1].set_title("RIGHT residues X < 50")
	axs[1][1].hist(hits.loc[(hits['CHAMBER'] == 3) & (hits['LAYER'] == 1) & (hits['X_LABEL'] == 2) & (hits['X'] < 50), 'RESIDUES_TRACK'], bins=np.arange(-4., 4., 0.05))
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/deleteme.png")
	fig.savefig(args.outputdir + runname + "_plots/deleteme.pdf")
	plt.close(fig)

	'''
	fig, axs = plt.subplots(nrows=16, ncols=1, figsize=(20, 10))
	for cl in range(4*4 -1, 0 -1, -1):
		layer = int(cl % 4) + 1
		chamber = cl // 4
		if cl==15: axs[-cl-1].set_title("Distance hit - wire vs position")
		if cl==0: axs[-cl-1].set_xlabel("$x_{hit}$ (mm)")
		if layer==2 and chamber==1: axs[-cl-1].set_ylabel("|$x_{hit}$ - $x_{wire}$| (mm)")
		h = axs[-cl-1].hist2d(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X'], np.abs(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X'] - hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X_WIRE']), bins=[17*21, 40], range=[[-8.5*42., +8.5*42.], [0., 21.]])
		#cbar = fig.colorbar(h[3], ax=axs[-cl-1], pad=0.01)
		#cbar.set_label("Counts")
	#fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/dxwire_track_x_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/dxwire_track_x_chamber_layer.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=16, ncols=1, figsize=(20, 10))
	for cl in range(4*4 -1, 0 -1, -1):
		layer = int(cl % 4) + 1
		chamber = cl // 4
		if cl==15: axs[-cl-1].set_title("Distance track - wire vs position")
		if cl==0: axs[-cl-1].set_xlabel("$x_{hit}$ (mm)")
		if layer==2 and chamber==1: axs[-cl-1].set_ylabel("|$x_{track}$ - $x_{wire}$| (mm)")
		h = axs[-cl-1].hist2d(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X'], np.abs(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X_TRACK'] - hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X_WIRE']), bins=[17*21, 40], range=[[-8.5*42., +8.5*42.], [0., 21.]])
		#cbar = fig.colorbar(h[3], ax=axs[-cl-1], pad=0.01)
		#cbar.set_label("Counts")
	#fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/dfitwire_track_x_chamber_layer.png")
	fig.savefig(args.outputdir + runname + "_plots/dfitwire_track_x_chamber_layer.pdf")
	plt.close(fig)
	'''	

# Missing hits
if 'missinghits' in args.plot:
	nmax = missinghits.groupby(['CHAMBER', 'WIRE', 'LAYER']).size().max()

	fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
	for chamber in range(4):
		axs[-chamber-1].set_title("Missing hit position [SL %d]" % chamber)
		axs[-chamber-1].set_xlabel("X position (mm)")
		axs[-chamber-1].set_ylabel("Z position (mm)")
		h = axs[-chamber-1].hist2d(missinghits.loc[missinghits['CHAMBER']==chamber, 'X'], missinghits.loc[missinghits['CHAMBER']==chamber, 'Z'], bins=[17*42, 4], range=[[-8.5*42., +8.5*42.], [-26., 26.]], vmin=0, vmax=nmax/42)
		cbar = fig.colorbar(h[3], ax=axs[-chamber-1], pad=0.01)
		cbar.set_label("Counts")
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/missinghits_position.png")
	fig.savefig(args.outputdir + runname + "_plots/missinghits_position.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for chamber in range(4):
		for layer in range(4):
			mis = missinghits[((missinghits['CHAMBER'] == chamber) & (missinghits['LAYER'] == layer+1))]
			axs[-chamber-1][layer].set_title("Missing hits [SL %d, LAYER %d]" % (chamber, layer+1))
			axs[-chamber-1][layer].set_xlabel("Wire number")
			axs[-chamber-1][layer].set_xticks(range(1, 17))
			axs[-chamber-1][layer].set_ylabel("Counts")
			axs[-chamber-1][layer].set_xlim(0, 17)
			axs[-chamber-1][layer].bar(mis['WIRE'].value_counts().index, mis['WIRE'].value_counts().values)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/missinghits_wire.png")
	fig.savefig(args.outputdir + runname + "_plots/missinghits_wire.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
	for chamber in range(4):
		axs[-chamber-1].set_title("Missing hits [SL %d]" % (chamber))
		axs[-chamber-1].set_xlabel("Wire number")
		axs[-chamber-1].set_xticks(range(1, 17))
		axs[-chamber-1].set_ylabel("Layer")
		axs[-chamber-1].set_yticks(range(1, 5))
		h = axs[-chamber-1].hist2d(missinghits.loc[missinghits['CHAMBER'] == chamber, 'WIRE'], missinghits.loc[missinghits['CHAMBER'] == chamber, 'LAYER'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]], vmin=0, vmax=nmax)
		cbar = fig.colorbar(h[3], ax=axs[-chamber-1], pad=0.01)
		cbar.set_label("Counts")
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/missinghits_cell.png")
	fig.savefig(args.outputdir + runname + "_plots/missinghits_cell.pdf")
	plt.close(fig)

	nmax = hits.groupby(['CHAMBER', 'WIRE', 'LAYER']).size().max()
	fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
	for chamber in range(4):
		axs[-chamber-1].set_title("Position of hits included in the fit [SL %d]" % chamber)
		axs[-chamber-1].set_xlabel("X position (mm)")
		axs[-chamber-1].set_ylabel("Z position (mm)")
		h = axs[-chamber-1].hist2d(hits.loc[((hits['X_SEG'].notna()) & (hits['CHAMBER']==chamber)), 'X'], hits.loc[((hits['X_SEG'].notna()) & (hits['CHAMBER']==chamber)), 'Z'], bins=[17*42, 4], range=[[-8.5*42., +8.5*42.], [-26., 26.]], vmin=0, vmax=nmax/42)
		cbar = fig.colorbar(h[3], ax=axs[-chamber-1], pad=0.01)
		cbar.set_label("Counts")
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/segmenthits_position.png")
	fig.savefig(args.outputdir + runname + "_plots/segmenthits_position.pdf")
	plt.close(fig)
	'''
	# Wire efficiency
	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for chamber in range(4):
		for layer in range(4):
			hit = hits.loc[((hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer+1)), 'WIRE'].value_counts().sort_index()
			mis = missinghits.loc[((missinghits['CHAMBER'] == chamber) & (missinghits['LAYER'] == layer+1) & (missinghits['WIRE'] <= 16)), 'WIRE'].value_counts().sort_index()
			axs[-chamber-1][layer].set_title("Wire efficiency [SL %d, LAYER %d]" % (chamber, layer+1))
			axs[-chamber-1][layer].set_xlabel("Efficiency")
			axs[-chamber-1][layer].set_ylabel("Counts")
			axs[-chamber-1][layer].set_xticks(range(1, 17))
			eff = hit / (hit + mis)
			axs[-chamber-1][layer].bar(eff.index, eff.values)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/efficiency_wire.png")
	fig.savefig(args.outputdir + runname + "_plots/efficiency_wire.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
	for chamber in range(4):
		for layer in range(4):
			hit = hits.loc[((hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer+1)), 'WIRE'].value_counts().sort_index()
			mis = missinghits.loc[((missinghits['CHAMBER'] == chamber) & (missinghits['LAYER'] == layer+1) & (missinghits['WIRE'] <= 16)), 'WIRE'].value_counts().sort_index()
			axs[-chamber-1][layer].set_title("Wire efficiency [SL %d, LAYER %d]" % (chamber, layer+1))
			axs[-chamber-1][layer].set_xlabel("Inefficiency")
			axs[-chamber-1][layer].set_ylabel("Counts")
			axs[-chamber-1][layer].set_xticks(range(1, 17))
			axs[-chamber-1][layer].set_ylim(0., 1.)
			eff = mis / (hit + mis)
			axs[-chamber-1][layer].bar(eff.index, eff.values)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/inefficiency_wire.png")
	fig.savefig(args.outputdir + runname + "_plots/inefficiency_wire.pdf")
	plt.close(fig)
	'''


if 'alignment' in args.plot:
	#Zchambers = {'02' : 816.2, '03' : 1599.7, '23' : 783.5}
	adf = pd.DataFrame(columns=['ORBIT'])
	gp = tracks.groupby('ORBIT')
	iEv, nEv = 0, len(gp)
	for iorbit, ev in gp:
		iEv += 1
		if args.verbose >= 1 and iEv % 100 == 0: print("Track extrapolation... [%.2f %%]" % (100.*iEv/nEv), end='\r')
		for i, sl in enumerate([['0', '2'], ['0', '3'], ['2', '3']]):
			if len(ev[ev['CHAMBER'] == sl[0]]) > 0 and len(ev[ev['CHAMBER'] == sl[1]]) > 0:
				if ev.loc[ev['CHAMBER'] == sl[0], 'NHITS'].values[0] >= 4 and ev.loc[ev['CHAMBER'] == sl[1], 'NHITS'].values[0] >= 4:
					sli, isli = '%s%s' % (sl[0], sl[1]), '%s%s' % (sl[1], sl[0])
					a0, m0, q0, x0 = ev.loc[ev['CHAMBER'] == sl[0], 'ANGLE_DEG'].values[0], ev.loc[ev['CHAMBER'] == sl[0], 'M'].values[0], ev.loc[ev['CHAMBER'] == sl[0], 'Q'].values[0], ev.loc[ev['CHAMBER'] == sl[0], 'X0'].values[0]
					a1, m1, q1, x1 = ev.loc[ev['CHAMBER'] == sl[1], 'ANGLE_DEG'].values[0], ev.loc[ev['CHAMBER'] == sl[1], 'M'].values[0], ev.loc[ev['CHAMBER'] == sl[1], 'Q'].values[0], ev.loc[ev['CHAMBER'] == sl[1], 'X0'].values[0]
					if abs(a0) < 15. and abs(a1) < 15.:
						da = a0 - a1
						dx01 = (config.SL_SHIFT[int(sl[1])][2] - q0) / m0 - (config.SL_SHIFT[int(sl[1])][2] - q1) / m1
						dx10 = (config.SL_SHIFT[int(sl[0])][2] - q0) / m0 - (config.SL_SHIFT[int(sl[0])][2] - q1) / m1
						adf = adf.append(pd.DataFrame.from_dict({'ORBIT' : [iorbit], 'ANGLE_DEG_' + sl[0] : [a0], 'ANGLE_DEG_' + sl[1] : [a1], 'Q_' + sl[0] : [q0], 'Q_' + sl[1] : [q1], 'X0_' + sl[0] : [x0], 'X0_' + sl[1] : [x1], 'deltaA_' + sli : [da], 'deltaX_' + sli : [dx01], 'deltaX_' + isli : [dx10]}), ignore_index=True)

	if args.verbose >= 1: print("Segment extrapolation completed.")

	# Delta angles
	fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
	for i, sl in enumerate([['0', '2'], ['0', '3'], ['2', '3']]):
		#segs = segments[((segments['VIEW'] == '0') & (segments['NHITS'] >= 4) & ((segments['CHAMBER'] == sl[0]) | (segments['CHAMBER'] == sl[1])))]
		#diff = (segs.groupby(['ORBIT'])['ANGLE_DEG']).diff().to_numpy()
		#diff = diff[np.isfinite(diff)]
		axs[0][i].set_title("Angle difference [SL %s vs SL %s]" % (sl[0], sl[1]))
		axs[0][i].set_xlabel("Angle difference [SL %s - SL %s] (deg)" % (sl[0], sl[1]))
		ybins, xbins, patches = axs[0][i].hist(adf['deltaA_%s%s' % (sl[0], sl[1])], bins=np.arange(-10., 10., 0.5))
		fitGaus(axs[0][i], ybins, xbins, patches)
		# 2D
		axs[1][i].set_title("Angle difference [SL %s vs SL %s]" % (sl[0], sl[1]))
		axs[1][i].set_ylabel("Angle [SL %s] (deg)" % (sl[0]))
		axs[1][i].set_xlabel("Angle difference [SL %s - SL %s] (deg)" % (sl[0], sl[1]))
		h = axs[1][i].hist2d(adf['deltaA_%s%s' % (sl[0], sl[1])], adf['ANGLE_DEG_%s' % (sl[0])], bins=[40, 50], range=[[-10., 10.], [-5., 5.]],)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/segment_delta_angles.png")
	fig.savefig(args.outputdir + runname + "_plots/segment_delta_angles.pdf")
	plt.close(fig)

	# Delta positions (extrapolated)
	fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
	for i, sl in enumerate([['0', '2'], ['0', '3'], ['2', '3']]):
		axs[0][i].set_title("Position difference [SL %s extrapolated from SL %s]" % (sl[1], sl[0]))
		axs[0][i].set_xlabel("Position difference [extrapolated from SL %s - SL %s] (mm)" % (sl[0], sl[1]))
		ybins, xbins, patches = axs[0][i].hist(adf['deltaX_%s%s' % (sl[0], sl[1])], bins=np.arange(-100., 100., 5.))
		fitGaus(axs[0][i], ybins, xbins, patches)
		axs[1][i].set_title("Position difference [SL %s extrapolated from SL %s]" % (sl[0], sl[1]))
		axs[1][i].set_xlabel("Position difference [extrapolated from SL %s - SL %s] (mm)" % (sl[1], sl[0]))
		ybins, xbins, patches = axs[1][i].hist(adf['deltaX_%s%s' % (sl[1], sl[0])], bins=np.arange(-100., 100., 5.))
		fitGaus(axs[1][i], ybins, xbins, patches)
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/segment_delta_positions.png")
	fig.savefig(args.outputdir + runname + "_plots/segment_delta_positions.pdf")
	plt.close(fig)

	fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
	for i, p in enumerate(['ANGLE_DEG', 'Q', 'X0']):
	    for j, chamber in enumerate([0, 2, 3]):
	    	axs[i][chamber].set_title("Segments %s [SL %d]" % (p, chamber))
	    	axs[i][chamber].set_xlabel(parlabel[p])
	    	axs[i][chamber].hist(adf['%s_%d' % (p, chamber)], bins=parbins[p])
	fig.tight_layout()
	fig.savefig(args.outputdir + runname + "_plots/segment_delta_par.png")
	fig.savefig(args.outputdir + runname + "_plots/segment_delta_par.pdf")
	plt.close(fig)


if args.verbose >= 1: print("Done.")