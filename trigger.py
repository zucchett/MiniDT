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

import argparse
parser = argparse.ArgumentParser(description='Command line arguments')
parser.add_argument("-i", "--inputfile", action="store", type=str, dest="inputdir", default="./output/Run000967_csv/", help="Provide directory of the input files (csv)")
parser.add_argument("-o", "--outputdir", action="store", type=str, dest="outputdir", default="./output/", help="Specify output directory")
parser.add_argument("-v", "--verbose", action="store", type=int, default=0, dest="verbose", help="Specify verbosity level")
args = parser.parse_args()

runname = [x for x in args.inputdir.split('/') if 'Run' in x][0].replace("_csv", "") if "Run" in args.inputdir else "Run000000"

def gaus(x, n, mean, sigma):
    return (n/np.sqrt(2.*np.pi)) * np.exp(-(x - mean)**2.0 / (2 * sigma**2))

def gausc(x, n, mean, sigma, bkg):
    return (n/np.sqrt(2.*np.pi)) * np.exp(-(x - mean)**2.0 / (2 * sigma**2)) + bkg


def fitGaus(axs, ybins, xbins, patches):
	cbins = xbins[:-1] + np.diff(xbins) / 2.
	xmin, xmax = axs.get_xlim()
	cfbins = cbins[(cbins < xmax) & (cbins > xmin)]
	yfbins = ybins[int((len(ybins) - len(cfbins)) / 2) + 1 : -int((len(ybins) - len(cfbins)) / 2)]
	mean = np.average(cbins, weights=ybins)
	rms = np.sqrt(np.average((cbins - mean)**2, weights=ybins))
	try:
		(gaus_norm, gaus_mean, gaus_sigma), cov = curve_fit(gaus, cbins, ybins, p0=[np.sum(ybins), mean, rms])
		gaus_x = np.linspace(xmin, xmax, 1000)
		gaus_y = gaus(gaus_x, gaus_norm, gaus_mean, gaus_sigma)
		axs.plot(gaus_x, gaus_y, '-', linewidth=2, color='red', label=("gaus"))
		axs.text(axs.get_xlim()[1]*1/2, axs.get_ylim()[1]*3/4, "$x_0 = %.2f$" % (gaus_mean))
		axs.text(axs.get_xlim()[1]*1/2, axs.get_ylim()[1]*5/8, "$\sigma = %.2f$" % (gaus_sigma))
		return gaus_norm, gaus_mean, gaus_sigma
	except:
		print("[ WARNING ] Fit unsuccessful")
		return 0, 0, 0


if args.verbose >= 1: print("Writing plots in directory %s" % (args.outputdir + runname))

triggers = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "triggers*.csv"))))
#triggers = pd.read_csv(args.inputdir + "/triggers.csv", skiprows=0, low_memory=False)
if args.verbose >= 1: print("Read %d lines from triggers.csv" % (len(triggers), ))
if args.verbose >= 2: print(triggers)

segments = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "segments*.csv"))))
#segments = pd.read_csv(args.inputdir + "/segments.csv", skiprows=0, low_memory=False)
if args.verbose >= 1: print("Read %d lines from segments.csv" % (len(segments), ))
if args.verbose >= 2: print(segments)


z0 = 1035.6 # z coordinate of the triggering SL

triggers['M_RAD'] = np.arctan(triggers['M'])
segments['M_RAD'] = np.arctan(segments['M'])

triggers['M_DEG'] = np.degrees(triggers['M_RAD']) - 90.
segments['M_DEG'] = np.degrees(segments['M_RAD']) - 90.

nTot4, nTrig4, nTot3, nTrig3 = 0, 0, 0, 0
df = pd.DataFrame(columns=['deltaM_global_local', 'deltaQ_global_local', 'deltaT_global_local', 'deltaM_global_trigger', 'deltaQ_global_trigger', 'deltaT_global_trigger', 'deltaM_local_trigger', 'deltaQ_local_trigger', 'deltaT_local_trigger'])

gp = segments.groupby('ORBIT')
iEv, nEv = 0, len(gp)
for iorbit, ev in gp:
	iEv += 1
	if args.verbose >= 1 and iEv % 100 == 0: print("Trigger-segment matching... [%.2f %%]" % (100.*iEv/nEv), end='\r')
	glt = ev[ev['VIEW'] == 'XZ']
	glb = ev[(ev['VIEW'] == 'YZ') & ((ev['CHAMBER'] == '0,3') | (ev['CHAMBER'] == '0,2,3'))]
	loc = ev[(ev['VIEW'] == 'YZ') & (ev['CHAMBER'] == '2')]
	if len(loc) > 0 and len(glb) > 0: # and len(glt) > 0:
		delta_gm, delta_gx0, delta_gt, delta_lm, delta_lx0, delta_lt = 9999., 9999., 9999., 9999., 9999., 9999.
		if glb['NHITS'].values[0] >= 7:
			if loc['NHITS'].values[0] >= 4: # and abs(glb['M_DEG'].values[0]) > 85. and abs(glt['M_DEG'].values[0]) > 85.:
				nTot4 += 1
				gm_rad, gm, gq, gt = glb['M_RAD'].values[0], glb['M'].values[0], glb['Q'].values[0], glb['T0'].values[0]
				gx0 = (z0 - gq) / gm # global fit has global coordinates
				lm_rad, lm, lq, lt = loc['M_RAD'].values[0], loc['M'].values[0], loc['Q'].values[0], loc['T0'].values[0]
				lx0 = (z0 - lq) / lm # local fit has local coordinates
				# Trigger
				if len(triggers[triggers['ORBIT'] == iorbit]) > 0:
					nTrig4 += 1
					tm_rad, tm, tq, tt = triggers.loc[triggers['ORBIT'] == iorbit, 'M_RAD'].values, triggers.loc[triggers['ORBIT'] == iorbit, 'M'].values, triggers.loc[triggers['ORBIT'] == iorbit, 'Q'].values, triggers.loc[triggers['ORBIT'] == iorbit, 'T0'].values
					tx0 = (- tq) / tm if not np.any(tm[np.isnan(tm) | np.isinf(tm)]) else np.zeros(len(tm)) # trigger has local coordinates
					# Comparison with global track
					for m_rad, x0, t in zip(tm_rad, tx0, tt):
						if abs(gm_rad - m_rad) < delta_gm: delta_gm = (gm_rad - m_rad)
						if abs(gx0 - x0) < delta_gx0: delta_gx0 = (gx0 - x0)
						if abs(gt - t) < delta_gt: delta_gt = (gt - t)
					# Comparison with local track
					for m_rad, x0 in zip(tm_rad, tx0):
						if abs(lm_rad - m_rad) < delta_lm: delta_lm = (lm_rad - m_rad)
						if abs(lx0 - x0) < delta_lx0: delta_lx0 = (lx0 - x0)
						if abs(lt - t) < delta_lt: delta_lt = (lt - t)

				df = df.append(pd.DataFrame.from_dict({ \
					'deltaM_global_local' : [gm_rad - lm_rad], 'deltaQ_global_local' : [gx0 - lx0], 'deltaT_global_local' : [gt - lt], \
					'deltaM_global_trigger' : [delta_gm], 'deltaQ_global_trigger' : [delta_gx0], 'deltaT_global_trigger' : [delta_gt], \
					'deltaM_local_trigger' : [delta_lm], 'deltaQ_local_trigger' : [delta_lx0], 'deltaT_local_trigger' : [delta_lt], \
				}), ignore_index=True)
			if loc['NHITS'].values[0] == 3:
				nTot3 += 1
				if len(triggers[triggers['ORBIT'] == iorbit]) > 0:
					nTrig3 += 1

if args.verbose >= 1: print("Trigger-segment matching completed.")

print("Trigger efficiency (4 hits): %d / %d = %.3f +- %.3f" % (nTrig4, nTot4, nTrig4/nTot4, nTrig4/nTot4*math.sqrt(1./nTrig4 + 1./nTot4)))
print("Trigger efficiency (3 hits): %d / %d = %.3f +- %.3f" % (nTrig3, nTot3, nTrig3/nTot3, nTrig3/nTot3*math.sqrt(1./nTrig3 + 1./nTot3)))
print("Trigger efficiency (total) : %d / %d = %.3f +- %.3f" % ((nTrig4+nTrig3), (nTot4+nTot3), (nTrig4+nTrig3)/(nTot4+nTot3), (nTrig4+nTrig3)/(nTot4+nTot3)*math.sqrt(1./(nTrig4+nTrig3) + 1./(nTot4+nTot3))))



# Difference between local and global fit
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axs[0].set_title("Angle")
axs[0].set_xlabel("$\\alpha_{local} - \\alpha_{global}$ (mrad)")
ybins, xbins, patches = axs[0].hist(df['deltaM_global_local']*1.e3, bins=np.arange(-100., 100., 2.))
fitGaus(axs[0], ybins, xbins, patches)
axs[1].set_title("Position")
axs[1].set_xlabel("$x^0_{local} - x^0_{global}$ (mm)")
ybins, xbins, patches = axs[1].hist(df['deltaQ_global_local'], bins=np.arange(-25., 25., 0.5))
fitGaus(axs[1], ybins, xbins, patches)
axs[2].set_title("Angle vs position")
axs[2].set_xlabel("$\\alpha_{local} - \\alpha_{global}$ (mrad)")
axs[2].set_ylabel("$x^0_{local} - x^0_{global}$ (mm)")
axs[2].hist2d(x=df['deltaM_global_local']*1.e3, y=df['deltaQ_global_local'], bins=[np.arange(-50., 50., 2.), np.arange(-10., 10., 0.5)])
#fitGaus(axs[2], ybins, xbins, patches)

fig.tight_layout()
fig.savefig(args.outputdir + runname + "_plots/global_vs_local.png")
fig.savefig(args.outputdir + runname + "_plots/global_vs_local.pdf")
plt.close(fig)

# Difference between trigger and local fit
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axs[0].set_title("Angle")
axs[0].set_xlabel("$\\alpha_{trigger} - \\alpha_{local}$ (mrad)")
ybins, xbins, patches = axs[0].hist(df['deltaM_local_trigger']*1.e3, bins=np.arange(-100., 100., 2.))
fitGaus(axs[0], ybins, xbins, patches)
axs[1].set_title("Position")
axs[1].set_xlabel("$x^0_{trigger} - x^0_{local}$ ($\mu$m)")
ybins, xbins, patches = axs[1].hist(df['deltaQ_local_trigger']*1.e3, bins=np.arange(-250., 250., 5.))
fitGaus(axs[1], ybins, xbins, patches)
axs[2].set_title("Time")
axs[2].set_xlabel("$t^0_{scintillator} - t^0_{trigger}$ (ns)")
ybins, xbins, patches = axs[2].hist(df['deltaT_local_trigger'], bins=25, range=(-25./30.*50, 25./30.*50))
fitGaus(axs[2], ybins, xbins, patches)
fig.tight_layout()
fig.savefig(args.outputdir + runname + "_plots/trigger_vs_local.png")
fig.savefig(args.outputdir + runname + "_plots/trigger_vs_local.pdf")
plt.close(fig)

# Difference between trigger and global fit
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axs[0].set_title("Angle")
axs[0].set_xlabel("$\\alpha_{trigger} - \\alpha_{global}$ (mrad)")
ybins, xbins, patches = axs[0].hist(df['deltaM_global_trigger']*1.e3, bins=np.arange(-100., 100., 2.))
fitGaus(axs[0], ybins, xbins, patches)
axs[1].set_title("Position")
axs[1].set_xlabel("$x^0_{trigger} - x^0_{global}$ (mm)")
ybins, xbins, patches = axs[1].hist(df['deltaQ_global_trigger'], bins=np.arange(-25., 25., 0.5))
fitGaus(axs[1], ybins, xbins, patches)
axs[2].set_title("Time")
axs[2].set_xlabel("$t^0_{scintillator} - t^0_{trigger}$ (ns)")
ybins, xbins, patches = axs[2].hist(df['deltaT_global_trigger'], bins=25, range=(-25./30.*50, 25./30.*50))
fitGaus(axs[2], ybins, xbins, patches)
fig.tight_layout()
fig.savefig(args.outputdir + runname + "_plots/trigger_vs_global.png")
fig.savefig(args.outputdir + runname + "_plots/trigger_vs_global.pdf")
plt.close(fig)


if args.verbose >= 1: print("Done.")