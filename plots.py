#!/usr/bin/env python
# coding: utf-8

import os, sys, math, time, glob, itertools
import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from scipy.stats import norm
from scipy.optimize import curve_fit
from modules.mapping.config import TDRIFT, VDRIFT, DURATION, XCELL, ZCELL
from modules.mapping import *
#from modules.reco import config_1_2_1 as config
from modules.reco import config_3_1 as config

import argparse
parser = argparse.ArgumentParser(description='Command line arguments')
parser.add_argument("-a", "--all", action="store_true", default=False, dest="all", help="Plot all categories")
parser.add_argument("-i", "--inputfile", action="store", type=str, dest="inputdir", default="./output/Run000967_csv/", help="Provide directory of the input files (csv)")
parser.add_argument("-o", "--outputdir", action="store", type=str, dest="outputdir", default="./output/", help="Specify output directory")
parser.add_argument("-p", "--plot", action="store", type=str, dest="plot", default="boxes,residues,res2D,resolution,tappini,efficiency,occupancy,parameters,alignment,trigger", help="Specify plot category")
parser.add_argument("-v", "--verbose", action="store", type=int, default=0, dest="verbose", help="Specify verbosity level")
args = parser.parse_args()

hep.set_style(hep.style.CMS)

mapconverter = Mapping()

'''
risoluzione in tempo trigger rispetto lo scintillatore (tutte le primitive / solo 3pletti / solo 4pletti)
risoluzione in angolo trigger rispetto la traccia globale (tutte le primitive / solo 3pletti / solo 4pletti)
risoluzione in angolo traccia locale rispetto la traccia globale
risoluzione in angolo trigger rispetto la traccia globale solo per le tracce con angolo ricostruito entro +/-15 gradi (tutte le primitive / solo 3pletti / solo 4pletti)
efficienza trigger vs angolo della traccia locale ricostruita [denominatore=traccia locale in accettanza / numeratore=traccia locale in accettanta & trigger] (tutte le primitive / solo 3pletti / solo 4pletti)
'''

isDensity = False
colors = {0 : '#000000', 3 : '#0072B2', 4 : '#D7301F'}
markers = {0 : 'o', 3 : 's', 4 : 'o'}
effRange = np.linspace(-1*math.pi/4, math.pi/4, 10)
#effRange = [-math.pi/4, -math.pi/8, 0., math.pi/8, math.pi/4]

chi2bins = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])

# Subplot indices
iax = [(i, j) for i in range(2) for j in range(2)]
isl = range(4)
ila = range(1, 4+1)

parlabel = {'M_RAD' : 'Angle (rad)', 'M_DEG' : 'Angle ($^o$)', 'Q' : 'Intercept', 'M' : 'Angular coefficient', 'X0' : '$x_{0}$ (mm)', 'NHITS' : 'number of hits'}
parunits = {'M_RAD' : 'rad', 'M_DEG' : '^o', 'Q' : 'mm', 'M' : '', 'X0' : 'mm', 'NHITS' : ''}
parbins = {'M_RAD' : np.arange(-np.pi, np.pi, 0.1), 'M_DEG' : np.arange(-50., 50., 1.), 'Q' : np.arange(-10000., 10000., 100.), 'M' : np.arange(-1000., 1000., 10.), 'X0' : np.arange(-336., 336., 8.), 'NHITS' : np.arange(-0.5, 12.5, 1)}

trigger_plots = {
    "time_trigger_vs_scint" : {
        "var" : "deltaT_local_trigger",
        "scale" : 1,
        "xlabel" : "$t^0_{scintillator} - t^0_{trigger}$ [ns]",
        "ylabel" : "Events / 1 ns",
        "unit" : "ns",
        "hrange" : [-25./30.*50, 25./30.*50],
        "bins" : 25,
        "xlim" : (-25./30.*50, 25./30.*50),
    },
    "time_trigger_vs_scint_vertical" : {
        "var" : "deltaT_local_trigger",
        "scale" : 1,
        "xlabel" : "$t^0_{scintillator} - t^0_{trigger}$ [ns]",
        "ylabel" : "Events / 1 ns",
        "unit" : "ns",
        "hrange" : [-25./30.*50, 25./30.*50],
        "bins" : 25,
        "xlim" : (-25./30.*50, 25./30.*50),
    },
    "angle_global_vs_trigger" : {
        "var" : "deltaM_global_trigger",
        "scale" : 1000.,
        "xlabel" : "$\\phi_{trigger} - \\phi_{global}$ [mrad]",
        "ylabel" : "Events / 1 mrad",
        "unit" : "mrad",
        "hrange" : [-100, 100],
        "bins" : 100,
        "xlim" : (-100, 100),
    },
    "angle_local_vs_trigger" : {
        "var" : "deltaM_local_trigger",
        "scale" : 1000.,
        "xlabel" : "$\\phi_{trigger} - \\phi_{local}$ [mrad]",
        "ylabel" : "Events / 1 mrad",
        "unit" : "mrad",
        "hrange" : [-100, 100],
        "bins" : 100,
        "xlim" : (-100, 100),
    },
    "angle_global_vs_local" : {
        "var" : "deltaM_global_local",
        "scale" : 1000.,
        "xlabel" : "$\\phi_{local} - \\phi_{global}$ [mrad]",
        "ylabel" : "Events / 1 mrad",
        "unit" : "mrad",
        "hrange" : [-100, 100],
        "bins" : 100,
        "xlim" : (-100, 100),
    },
    "angle_global_vs_trigger_vertical" : {
        "var" : "deltaM_global_trigger",
        "scale" : 1000.,
        "xlabel" : "$\\phi_{trigger} - \\phi_{global}$ [mrad]",
        "ylabel" : "Events / 1 mrad",
        "unit" : "mrad",
        "hrange" : [-100, 100],
        "bins" : 100,
        "xlim" : (-100, 100),
    },
}


def gaus(x, n, mean, sigma):
    return (n/np.sqrt(2.*np.pi)) * np.exp(-(x - mean)**2.0 / (2 * sigma**2))

def gausc(x, n, mean, sigma, bkg):
    return (n/np.sqrt(2.*np.pi)) * np.exp(-(x - mean)**2.0 / (2 * sigma**2)) + bkg

def gaus2(x, mean, n1, sigma1, n2, sigma2):
    return (n1/np.sqrt(2.*np.pi)) * np.exp(-(x - mean)**2.0 / (2 * sigma1**2)) + (n2/np.sqrt(2.*np.pi)) * np.exp(-(x - mean)**2.0 / (2 * sigma2**2))

def fitGaus(axs, ybins, xbins, patches):
    try:
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
        axs.text(axs.get_xlim()[1]*1/2, axs.get_ylim()[1]*3/4, "$x_0 = %.2f$" % (gaus_mean))
        axs.text(axs.get_xlim()[1]*1/2, axs.get_ylim()[1]*5/8, "$\sigma = %.2f$" % (gaus_sigma))
        return gaus_norm, gaus_mean, gaus_sigma
    except:
        print("[ WARNING ] Fit unsuccessful")
        return 0, 0, 0

def plotHist(axs, data, name, title, unit, bins, label, linecolor, markercolor, markerstyle, norm=False, fit="Gaus"):
    axs.set_title(name)
    axs.set_xlabel(title + " [" + unit + "]")
    if isinstance(data, np.ndarray): bin_heights, bin_borders = data, bins
    else: bin_heights, bin_borders = np.histogram(data.dropna(), range=(bins[0], bins[-1]), bins=bins)
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    integral = np.sum(bin_heights)
    if not integral > 0.: return
    mean = np.average(bin_centers, weights=bin_heights)
    rms = np.sqrt(np.average((bin_centers - mean)**2, weights=bin_heights))
    bin_errors = np.sqrt(bin_heights)
    if norm: bin_heights, bin_errors = bin_heights / integral, bin_errors / integral
    if fit=="Gaus" and integral > 100:
        popt, pcov = curve_fit(gaus, bin_centers, bin_heights, maxfev=10000, p0=[integral, mean, rms])
        x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 1000)
        axs.plot(x_interval_for_fit, gaus(x_interval_for_fit, *popt), color=linecolor, lw=1)
        nargs = label.count('%')
        if nargs == 2: label_string = label % (abs(popt[2]), unit)
        elif nargs == 4: label_string = label % (popt[1], unit, abs(popt[2]), unit)
        else: label_string = label
    else:
        popt, pcov, label_string = [], [], label
    line = axs.errorbar(bin_centers, bin_heights, yerr=bin_errors, fmt=markerstyle, color=markercolor, alpha=1, markersize=6, zorder=1, label=label_string)[0]
    line.set_clip_on(False)
    axs.legend()
    return popt


def plotChannels(axs, sl, xbins, ybins):
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            axs.text(xbins[j]+0.5, ybins[i]+0.5, mapconverter.getChannel(sl, i + 1, j + 1), color="w", ha="center", va="center", fontweight="bold")
    return

def plotWires(axs):
    x, y = [], []
    for layer in range(1, 4+1):
        for wire in range(1, 16+1):
            x.append(mapconverter.getWirePosition(layer, wire))
            y.append(mapconverter.getZlayer(layer))
    axs.scatter(x, y, marker='x', c='w')
    return





runname = [x for x in args.inputdir.split('/') if 'Run' in x][0].replace("_csv", "") if "Run" in args.inputdir else "Run000000"

if args.verbose >= 1: print("Writing plots in directory %s_plots/" % (args.outputdir + runname))

# Read files in blocks
ex, mt, nA, nT = None, None, None, None

#if len(args.outputdir + runname) > 0:
#    if not os.path.exists(args.outputdir + runname): os.makedirs(args.outputdir + runname)
#    for d in ["occupancy", "timebox", "parameters", "residuals", "missinghits", "alignment", "trigger"]:
#        if not os.path.exists(args.outputdir + runname + "/" + d + "/"): os.makedirs(args.outputdir + runname + "/" + d + "/")


#hits = pd.read_csv(args.inputdir + "/events.csv", skiprows=0, low_memory=False)
#segments = pd.read_csv(args.inputdir + "/segments.csv", dtype={'CHAMBER': object}, skiprows=0, low_memory=False)
#missinghits = pd.read_csv(args.inputdir + "/missinghits.csv", skiprows=0, low_memory=False)
#occupancy = pd.read_csv(args.inputdir + "/occupancy.csv", skiprows=0, low_memory=False)

hits = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "events*.csv"))))
segments = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "segments*.csv"))))
missinghits = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "missinghits*.csv"))))
occupancy = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "occupancy*.csv"))))
times = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "times*.csv"))))


if args.verbose >= 1: print("Read %d lines from events.csv" % (len(hits), ))
if args.verbose >= 2: print(hits.head(50))

# Recalculate rate in case of multiple files
runtime = (hits['ORBIT'].max() - hits['ORBIT'].min()) * DURATION['orbit'] * 1.e-9 # Approximate acquisition time in seconds

#hits = hits[hits['T0'].notnull()]

hits['X_DRIFT'] = hits['TIMENS'] * VDRIFT

# Residuals
hits['RESIDUES_SEG'] = np.abs(hits['X'] - hits['X_WIRE']) - np.abs(hits['X_SEG'] - hits['X_WIRE'])
hits['DIST_SEG_WIRE'] = hits['X_SEG'] - hits['X_WIRE']
hits["ABS_DIST_SEG_WIRE"] = np.abs(hits["DIST_SEG_WIRE"])

# Residuals from tracks
hits['RESIDUES_TRACK'] = np.abs(hits['X'] - hits['X_WIRE']) - np.abs(hits['X_TRACK'] - hits['X_WIRE'])
hits['DIST_TRACK_WIRE'] = hits['X_TRACK'] - hits['X_WIRE']
hits["ABS_DIST_TRACK_WIRE"] = np.abs(hits["DIST_TRACK_WIRE"])

# time refit
hits['DELTAT_REFIT_VS_SCINTPMT'] = hits['T0'] - hits['T_SCINTPMT']
hits[hits['DELTAT_REFIT_VS_SCINTPMT'] == 0.] = np.nan

#segments['M_RAD'] = np.arctan(segments['M'])
#segments['M_RAD'] = np.where(segments['M_RAD'] < 0, segments['M_RAD'] + math.pi/2., segments['M_RAD'] - math.pi/2.)
#segments['M_DEG'] = np.degrees(segments['M_RAD'])
#segments['X0'] = - segments['Q'] / segments['M']

# Calculate the difference between the angles of the segments in the same orbit, and report it to the hits df for further filtering
angle_df = segments.loc[(segments['VIEW'] == '0') & (segments['CHAMBER'].isin([str(x) for x in config.SL_VIEW[config.PHI_VIEW]])), ['ORBIT', 'CHAMBER', 'M_DEG']].copy()
angle_df['DELTA_ANGLE'] = angle_df.groupby('ORBIT')['M_DEG'].transform(np.ptp)
angle_df = angle_df[angle_df['DELTA_ANGLE'] != 0]
angle_dict = angle_df[['ORBIT', 'DELTA_ANGLE']].drop_duplicates().set_index('ORBIT').T.to_dict('records')[0] if len(angle_df) > 0 else {}
hits['DELTA_ANGLE'] = hits['ORBIT'].map(angle_dict)

# Calculate the difference between the x0 position interpolated from the track, and the one in the test layer, for further filtering
pos_df = segments.loc[(segments['VIEW'] == config.PHI_VIEW.upper()) & ((segments['CHAMBER'] == ",".join([str(x) for x in config.SL_AUX])) | (segments['CHAMBER'] == str(config.SL_TEST))), ['ORBIT', 'CHAMBER', 'X0']].copy()
pos_df['DELTA_X0'] = pos_df.groupby('ORBIT')['X0'].transform(np.ptp)
pos_df = pos_df[pos_df['DELTA_X0'] != 0]
pos_dict = pos_df[['ORBIT', 'DELTA_X0']].drop_duplicates().set_index('ORBIT').T.to_dict('records')[0] if len(pos_df) > 0 else {}
hits['DELTA_X0'] = hits['ORBIT'].map(pos_dict)

# Associate the track chi2 to each hit
chi2_df = segments[(segments['VIEW'] == config.PHI_VIEW.upper()) & (segments['CHAMBER'] == ",".join([str(x) for x in config.SL_AUX]))].copy()
chi2_dict = chi2_df[['ORBIT', 'CHI2']].drop_duplicates().set_index('ORBIT').T.to_dict('records')[0] if len(chi2_df) > 0 else {}
hits['TRACK_CHI2'] = hits['ORBIT'].map(chi2_dict)

# Split by number of segment hits
nhits = {}
for num_hits in [0, 3, 4]: nhits[num_hits] = hits.loc[(hits['NHITS_SEG'] == num_hits)] if num_hits != 0 else hits.loc[(hits['NHITS_SEG'] > 0)]#hits.loc[(hits['NHITS_SEG'] == 3) & (hits['NHITS_SEG'] == 4)]



# Timebox
def plotBoxes():
    if not os.path.exists(args.outputdir + runname + "_plots/boxes/"): os.makedirs(args.outputdir + runname + "_plots/boxes/")
    '''
    print(hits.columns)
    chamber = 1
    dk = hits.groupby(['ORBIT', 'CHAMBER', 'LAYER', 'WIRE'])['TIMENS'].count() # count performed a random column
    dk = dk.reset_index().rename(columns={'TIMENS' : 'COUNTS'})
    macs = dk['COUNTS'].max()
    #dfh = hits.loc[(hits['CHAMBER']==chamber)]
    #dk = .groupby(['ORBIT', 'CHAMBER', 'LAYER', 'WIRE_NUM'])['TIMENS'].diff()#.transform(lambda x: x.diff() if x > 0 else 0.)
    print(dk)
    print(macs)
    exit()
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 20))
    for idx, chamber in zip(iax, isl):
        axs[idx].set_title("Timebox [SL %d]" % chamber)
        axs[idx].set_xlabel("Time (ns)")
        axs[idx].set_ylabel("Counts")
        axs[idx].hist(hits.loc[hits['CHAMBER']==chamber, 'TIMENS'], bins=np.arange(-50, 550+1, 5), label='Drift time (ns)')
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/boxes/timediff_vs_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/boxes/timediff_vs_chamber.pdf")
    plt.close(fig)
    exit()
    '''
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 20))
    for idx, chamber in zip(iax, isl):
        axs[idx].set_title("Timebox [SL %d]" % chamber)
        axs[idx].set_xlabel("Time (ns)")
        axs[idx].set_ylabel("Counts")
        axs[idx].hist(hits.loc[hits['CHAMBER']==chamber, 'TIMENS'], bins=np.arange(-50, 550+1, 5), label='Drift time (ns)')
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/boxes/timebox_vs_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/boxes/timebox_vs_chamber.pdf")
    plt.close(fig)
    
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for chamber in range(4):
        for layer in range(4):
            axs[-chamber-1][layer].set_title("Timebox [SL %d, LAYER %d]" % (chamber, layer+1))
            axs[-chamber-1][layer].set_xlabel("Time (ns)")
            axs[-chamber-1][layer].set_ylabel("Counts")
            axs[-chamber-1][layer].hist(hits.loc[(hits['CHAMBER']==chamber) & (hits['LAYER']==layer+1), 'TIMENS'], bins=np.arange(-50, 550+1, 5), label='Drift time (ns)')
            axs[-chamber-1][layer].legend()
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/boxes/timebox_vs_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/boxes/timebox_vs_chamber_layer.pdf")
    plt.close(fig)
    
    fig, axs = plt.subplots(nrows=4 * 4, ncols=16, figsize=(80, 60))
    for cell in range(16):
        for chamber in range(4):
            for layer in range(4):
                axs[-chamber*4-layer-1][cell].set_title("[SL %d, LAYER %d, WIRE %d]" % (chamber, layer+1, cell+1))
                #axs[-chamber*4-layer-1][cell].set_xlabel("Time (ns)")
                #axs[-chamber*4-layer-1][cell].set_ylabel("Counts")
                axs[-chamber*4-layer-1][cell].hist(hits.loc[(hits['CHAMBER']==chamber) & (hits['LAYER']==layer+1) & (hits['WIRE']==cell+1), 'TIMENS'], bins=np.arange(-50, 550+1, 5), label='Drift time (ns)')
                #axs[-chamber*4-layer-1][cell].legend()
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/boxes/timebox_vs_wire.png")
    fig.savefig(args.outputdir + runname + "_plots/boxes/timebox_vs_wire.pdf")
    plt.close(fig)
    
    # Space boxes
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for chamber in range(4):
        for layer in range(4):
            axs[-chamber-1][layer].set_title("Timebox [SL %d, LAYER %d]" % (chamber, layer))
            axs[-chamber-1][layer].set_xlabel("Position (mm)")
            axs[-chamber-1][layer].set_ylabel("Counts")
            axs[-chamber-1][layer].hist(hits.loc[(hits['CHAMBER']==chamber) & (hits['LAYER']==layer+1), 'TIMENS']*VDRIFT, bins=np.arange(-5., 35., 1), label='position (mm)')
            axs[-chamber-1][layer].legend()
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/boxes/spacebox_vs_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/boxes/spacebox_vs_chamber_layer.pdf")
    plt.close(fig)


    nmax = hits.groupby(['CHAMBER', 'WIRE', 'LAYER']).size().max()
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(40, 20))
    for chamber in range(4):
        axs[-chamber-1].set_title("Hit position (after L/R disambiguation) [chamber %d]" % chamber)
        axs[-chamber-1].set_xlabel("X position (mm)")
        axs[-chamber-1].set_ylabel("Z position (mm)")
        #axs[-chamber-1].scatter(hits.loc[hits['CHAMBER']==chamber, 'X'], hits.loc[hits['CHAMBER']==chamber, 'Z'], s=1)
        h, xbins, ybins, im = axs[-chamber-1].hist2d(hits.loc[hits['CHAMBER']==chamber, 'X'], hits.loc[hits['CHAMBER']==chamber, 'Z'], bins=[17*42, 4], range=[[-8.5*42., +8.5*42.], [-26., 26.]], vmin=0, vmax=nmax/42)
        cbar = fig.colorbar(im, ax=axs[-chamber-1], pad=0.01)
        cbar.set_label("Counts")
        plotWires(axs[-chamber-1])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/boxes/spacebox.png")
    fig.savefig(args.outputdir + runname + "_plots/boxes/spacebox.pdf")
    plt.close(fig)
    
    



def plotResidues():
    if not os.path.exists(args.outputdir + runname + "_plots/residues/"): os.makedirs(args.outputdir + runname + "_plots/residues/")
    
    # Residuals from Segments
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
    for idx, chamber in zip(iax, isl):
        for num_hits in [0, 3, 4]:
            pars = plotHist(axs[idx], data=nhits[num_hits].loc[nhits[num_hits]['CHAMBER'] == chamber, 'RESIDUES_SEG'], name="Local residues [SL %d]" % chamber, title="|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
            if num_hits == 0: print("Time calibration CHAMBER %d:\t%.1f ns" % (chamber, -pars[1]/VDRIFT *2.))
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_seg_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_seg_chamber.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
    for idx, layer in zip(iax, ila):
        for num_hits in [0, 3, 4]:
            plotHist(axs[idx], data=nhits[num_hits].loc[nhits[num_hits]['LAYER'] == layer, 'RESIDUES_SEG'], name="Local residues [LAYER %d]" % layer, title="|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_seg_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_seg_layer.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
        chamber, layer = 4 - cl // 4 - 1, int(cl % 4) + 1
        for num_hits in [0, 3, 4]:
            plotHist(axs[idx], data=nhits[num_hits].loc[(nhits[num_hits]['CHAMBER'] == chamber) & (nhits[num_hits]['LAYER'] == layer), 'RESIDUES_SEG'], name="Local residues [CHAMBER %d, LAYER %d]" % (chamber, layer), title="|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_seg_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_seg_chamber_layer.pdf")
    plt.close(fig)
    

    # Residuals from Tracks
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
    for idx, chamber in zip(iax, isl):
        for num_hits in [0, 3, 4]:
            plotHist(axs[idx], data=nhits[num_hits].loc[nhits[num_hits]['CHAMBER'] == chamber, 'RESIDUES_TRACK'], name="Global residues [SL %d]" % chamber, title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_chamber.pdf")
    plt.close(fig)


    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
    for idx, layer in zip(iax, ila):
        for num_hits in [0, 3, 4]:
            plotHist(axs[idx], data=nhits[num_hits].loc[nhits[num_hits]['LAYER'] == layer, 'RESIDUES_TRACK'], name="Global residues [LAYER %d]" % layer, title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_layer.pdf")
    plt.close(fig)


    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
        chamber, layer = 4 - cl // 4 - 1, int(cl % 4) + 1
        for num_hits in [0, 3, 4]:
            plotHist(axs[idx], data=nhits[num_hits].loc[(nhits[num_hits]['CHAMBER'] == chamber) & (nhits[num_hits]['LAYER'] == layer), 'RESIDUES_TRACK'], name="Global residues [CHAMBER %d, LAYER %d]" % (chamber, layer), title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_chamber_layer.pdf")
    plt.close(fig)

    # Tracks L/R
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
    for idx, chamber in zip(iax, isl):
        for num_hits in [0, 3, 4]:
            plotHist(axs[idx], data=nhits[num_hits].loc[(nhits[num_hits]['CHAMBER'] == chamber) & (nhits[num_hits]['X_LABEL'] == 1), 'RESIDUES_TRACK'], name="Global LEFT residues [CHAMBER %d]" % (chamber), title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_left_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_left_chamber.pdf")
    plt.close(fig)
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
    for idx, chamber in zip(iax, isl):
        for num_hits in [0, 3, 4]:
            plotHist(axs[idx], data=nhits[num_hits].loc[(nhits[num_hits]['CHAMBER'] == chamber) & (nhits[num_hits]['X_LABEL'] == 2), 'RESIDUES_TRACK'], name="Global RIGHT residues [CHAMBER %d]" % (chamber), title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_right_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_track_right_chamber.pdf")
    plt.close(fig)
    
    # Symmetric combination: (sx + dx) / 2
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
    for idx, chamber in zip(iax, isl):
        for num_hits in [0, 3, 4]:
            sx, sx_borders = np.histogram(nhits[num_hits].loc[(nhits[num_hits]['CHAMBER'] == chamber) & (nhits[num_hits]['X_LABEL'] == 1), 'RESIDUES_TRACK'].dropna(), bins=np.arange(-4., 4., 0.05))
            dx, dx_borders = np.histogram(nhits[num_hits].loc[(nhits[num_hits]['CHAMBER'] == chamber) & (nhits[num_hits]['X_LABEL'] == 2), 'RESIDUES_TRACK'].dropna(), bins=np.arange(-4., 4., 0.05))
            symm = (sx + dx) * 0.5
            plotHist(axs[idx], data=symm, name="Global (RIGHT+LEFT)/2 residues [CHAMBER %d]" % (chamber), title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_symm_track_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_symm_track_chamber.pdf")
    plt.close(fig)
    
    # Antisymmetric combination: (sx - dx) / 2
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
    for idx, chamber in zip(iax, isl):
        for num_hits in [0, 3, 4]:
            sx, sx_borders = np.histogram(nhits[num_hits].loc[(nhits[num_hits]['CHAMBER'] == chamber) & (nhits[num_hits]['X_LABEL'] == 1), 'RESIDUES_TRACK'].dropna(), bins=np.arange(-4., 4., 0.05))
            dx, dx_borders = np.histogram(nhits[num_hits].loc[(nhits[num_hits]['CHAMBER'] == chamber) & (nhits[num_hits]['X_LABEL'] == 2), 'RESIDUES_TRACK'].dropna()*-1., bins=np.arange(-4., 4., 0.05))
            symm = (sx + dx) * 0.5
            pars = plotHist(axs[idx], data=symm, name="Global (RIGHT-LEFT)/2 residues [CHAMBER %d]" % (chamber), title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
            if chamber == config.SL_TEST and num_hits == 0 and len(pars) > 1: print("Offset calibration CHAMBER %d:\t%.2f mm" % (chamber, pars[1]))
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_asym_track_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/residues/residues_asym_track_chamber.pdf")
    plt.close(fig)


def plotResidues2D():

    if not os.path.exists(args.outputdir + runname + "_plots/residues2D/"): os.makedirs(args.outputdir + runname + "_plots/residues2D/")
    
    # Residues from Segments 2D
    jgrid = sns.jointplot(data=hits, dropna=True, x="TIMENS", y="RESIDUES_SEG", kind="reg", scatter_kws={'s': 0.1})
    jgrid.set_axis_labels("$t_{drift}$ (ns)", "|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$| (mm)")
    jgrid.ax_joint.set_ylim(-2., 2)
    jgrid.ax_marg_y.set_ylim(-2., 2)
    jgrid.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_tdrift_joint.png")
    jgrid.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_tdrift_joint.pdf")

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 20))
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
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_tdrift.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_tdrift.pdf")
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
    jgrid.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_pos_joint.png")
    jgrid.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_pos_joint.pdf")
    exit()

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
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_pos_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_pos_chamber.pdf")
    plt.close(fig)
    
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    for idx, layer in zip(iax, ila):
        hitsl = hits[hits['LAYER'] == layer].dropna()
        if not len(hitsl) > 0: continue
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
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_pos_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_pos_layer.pdf")
    plt.close(fig)
    
    
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
    for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
        chamber, layer = 4 - cl // 4 - 1, int(cl % 4) + 1
        hitsl = hits[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer)].dropna()
        if not len(hitsl) > 0: continue
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
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_pos_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_seg_pos_chamber_layer.pdf")
    plt.close(fig)
    
    
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
    for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
        chamber, layer = 4 - cl // 4 - 1, int(cl % 4) + 1
        hitsl = hits[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer)].dropna()
        if not len(hitsl) > 0: continue
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
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_track_pos_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_track_pos_chamber_layer.pdf")
    plt.close(fig)
    
    
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
    for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
        chamber, layer = 4 - cl // 4 - 1, int(cl % 4) + 1
        hitsl = hits[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer) & (hits['X_LABEL'] == 1)].dropna()
        if not len(hitsl) > 0: continue
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
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_track_pos_left_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_track_pos_left_chamber_layer.pdf")
    plt.close(fig)
    
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
    for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
        chamber, layer = 4 - cl // 4 - 1, int(cl % 4) + 1
        hitsl = hits[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer) & (hits['X_LABEL'] == 2)].dropna()
        if not len(hitsl) > 0: continue
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
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_track_pos_right_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_track_pos_right_chamber_layer.pdf")
    plt.close(fig)
    
    
    fig, axs = plt.subplots(nrows=16, ncols=1, figsize=(40, 30))
    for cl in range(4*4 -1, 0 -1, -1):
        chamber, layer = 4 - cl // 4 - 1, int(cl % 4) + 1
        if cl==15: axs[-cl-1].set_title("Global residues vs position")
        if cl==0: axs[-cl-1].set_xlabel("$x_{hit}$ (mm)")
        if layer==2 and chamber==1: axs[-cl-1].set_ylabel("|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$| (mm)")
        h = axs[-cl-1].hist2d(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X'], hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'RESIDUES_TRACK'], bins=[33*21, 40], range=[[-8.0*42., +8.5*42.], [-2., 2.]])
        #cbar = fig.colorbar(h[3], ax=axs[-cl-1], pad=0.01)
        #cbar.set_label("Counts")
        for c in range(-8, +9):
            axs[-cl-1].axvline(c * 42. + (21. if layer % 2 ==1 else 0.), axs[-cl-1].get_ylim()[0], axs[-cl-1].get_ylim()[1], linewidth=2, color='white')
    #fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_track_x_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/residues_track_x_chamber_layer.pdf")
    plt.close(fig)
    
    
    fig, axs = plt.subplots(nrows=16, ncols=1, figsize=(40, 30))
    for cl in range(4*4 -1, 0 -1, -1):
        chamber, layer = 4 - cl // 4 - 1, int(cl % 4) + 1
        if cl==15: axs[-cl-1].set_title("Distance hit - wire vs position")
        if cl==0: axs[-cl-1].set_xlabel("$x_{hit}$ (mm)")
        if layer==2 and chamber==1: axs[-cl-1].set_ylabel("|$x_{hit}$ - $x_{wire}$| (mm)")
        h = axs[-cl-1].hist2d(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X'], np.abs(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X'] - hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X_WIRE']), bins=[17*21, 40], range=[[-8.5*42., +8.5*42.], [0., 21.]])
        #cbar = fig.colorbar(h[3], ax=axs[-cl-1], pad=0.01)
        #cbar.set_label("Counts")
    #fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues2D/dxwire_track_x_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/dxwire_track_x_chamber_layer.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(nrows=16, ncols=1, figsize=(40, 30))
    for cl in range(4*4 -1, 0 -1, -1):
        chamber, layer = 4 - cl // 4 - 1, int(cl % 4) + 1
        if cl==15: axs[-cl-1].set_title("Distance track - wire vs position")
        if cl==0: axs[-cl-1].set_xlabel("$x_{hit}$ (mm)")
        if layer==2 and chamber==1: axs[-cl-1].set_ylabel("|$x_{track}$ - $x_{wire}$| (mm)")
        h = axs[-cl-1].hist2d(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X'], np.abs(hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X_TRACK'] - hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer), 'X_WIRE']), bins=[17*21, 40], range=[[-8.5*42., +8.5*42.], [0., 21.]])
        #cbar = fig.colorbar(h[3], ax=axs[-cl-1], pad=0.01)
        #cbar.set_label("Counts")
    #fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/residues2D/dfitwire_track_x_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/residues2D/dfitwire_track_x_chamber_layer.pdf")
    plt.close(fig)
    


def plotResolution():
    if not os.path.exists(args.outputdir + runname + "_plots/resolution/"): os.makedirs(args.outputdir + runname + "_plots/resolution/")
    
    # Refit
    refhits = pd.concat(map(pd.read_csv, glob.glob(os.path.join(args.inputdir, "residues*.csv"))))
    if len(refhits) <= 0:
        print("Resolution file is empty, exiting...")
        exit()
    
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for chamber in range(4):
        for layer in range(1, 4+1):
            plotHist(axs[-chamber-1][layer-1], data=refhits.loc[(refhits['CHAMBER'] == chamber) & (refhits['LAYER'] == layer), 'RESIDUES_SEG'], name="Residues [SL %d, LAYER %d]" % (chamber, layer), title="|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$|", unit="mm", bins=np.arange(-2, 2, 0.05), label="$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/resolution/refit_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/resolution/refit_chamber_layer.pdf")
    plt.close(fig)
    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    plotHist(axs, data=refhits.loc[(refhits['CHAMBER'] > -1) & (refhits['LAYER'].isin([1, 4])), 'RESIDUES_SEG'], name="", title="|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="External layers:\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$", linecolor=colors[3], markercolor=colors[3], markerstyle=markers[3])
    plotHist(axs, data=refhits.loc[(refhits['CHAMBER'] > -1) & (refhits['LAYER'].isin([2, 3])), 'RESIDUES_SEG'], name="", title="|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="Internal layers:\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$", linecolor=colors[4], markercolor=colors[4], markerstyle=markers[4])
    plotHist(axs, data=refhits.loc[(refhits['CHAMBER'] > -1) & (refhits['LAYER'] >= -1)       , 'RESIDUES_SEG'], name="", title="|$x_{hit}$ - $x_{wire}$| - |$x_{seg}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="Combined:\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/resolution/refit_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/resolution/refit_layer.pdf")
    plt.close(fig)
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    plotHist(axs[0], data=hits['DELTA_ANGLE'].drop_duplicates(), name="$\Delta \phi$ between the segments", title="$\Delta \phi$", unit="deg", bins=np.arange(0., 10., 0.1), label=None, linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0], fit="None")
    plotHist(axs[1], data=hits['DELTA_X0'].drop_duplicates(), name="$\Delta x_0$ between the segments", title="$\Delta x_0$", unit="mm", bins=np.arange(0., 50., 1.), label=None, linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0], fit="None")
    plotHist(axs[2], data=hits['TRACK_CHI2'].drop_duplicates(), name="$\chi^2$ / n.d.f.", title="$\chi^2$ / n.d.f.", unit="", bins=np.arange(0., 5., 0.1), label=None, linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0], fit="None")
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/resolution/deltas.png")
    fig.savefig(args.outputdir + runname + "_plots/resolution/deltas.pdf")
    plt.close(fig)
    
    reshits = hits[(hits['TRACK_CHI2'] > 1.)].copy() #(hits['DELTA_ANGLE'] < 1.) & (hits['DELTA_X0'] < 10.) & 
    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    plotHist(axs, data=reshits.loc[(reshits['CHAMBER'] == config.SL_TEST) & (reshits['LAYER'].isin([1, 4])), 'RESIDUES_TRACK'], name="", title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.1), label="External layers:\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$", linecolor=colors[3], markercolor=colors[3], markerstyle=markers[3])
    plotHist(axs, data=reshits.loc[(reshits['CHAMBER'] == config.SL_TEST) & (reshits['LAYER'].isin([2, 3])), 'RESIDUES_TRACK'], name="", title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.1), label="Internal layers:\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$", linecolor=colors[4], markercolor=colors[4], markerstyle=markers[4])
    plotHist(axs, data=reshits.loc[(reshits['CHAMBER'] == config.SL_TEST) & (reshits['LAYER'] >= -1)       , 'RESIDUES_TRACK'], name="", title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.1), label="Combined:\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/resolution/interpolation.png")
    fig.savefig(args.outputdir + runname + "_plots/resolution/interpolation.pdf")
    plt.close(fig)
    
    

# Tappini
def plotTappini():
    if not os.path.exists(args.outputdir + runname + "_plots/tappini/"): os.makedirs(args.outputdir + runname + "_plots/tappini/")
    
    wires = {
        0 : [1, 2],
        1 : [3, 4, 5, 6],
        2 : [7, 8, 9, 10],
        3 : [11, 12, 13, 14],
        4 : [15, 16],
    }
    chamber = config.SL_TEST
    
    # Tappini check
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(40, 30))
    for layer in range(1, 4+1):
        for group in range(0, 5):
    #for idx, cl in zip([(i, j) for i in range(5) for j in range(4)], range(0, 4*5)):
    #    chamber, layer, group = config.SL_TEST, 4 - cl // 4 - 1, int(cl % 4) + 1
            plotHist(axs[-layer][group], data=hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer) & (hits['WIRE'].isin(wires[group])), 'RESIDUES_TRACK'], name="Global residues [LAYER %d, TAPPINO %d]" % (layer, group), title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""3/4 + 4/4,\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/tappini/residues_track_tappino.png")
    fig.savefig(args.outputdir + runname + "_plots/tappini/residues_track_tappino.pdf")
    plt.close(fig)
    
    # Tappini check LEFT
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(40, 30))
    for layer in range(1, 4+1):
        for group in range(0, 5):
            plotHist(axs[-layer][group], data=hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer) & (hits['WIRE'].isin(wires[group]) & (hits['X_LABEL'] == 1)), 'RESIDUES_TRACK'], name="Global residues LEFT [LAYER %d, TAPPINO %d]" % (layer, group), title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""3/4 + 4/4,\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/tappini/residues_track_left_tappino.png")
    fig.savefig(args.outputdir + runname + "_plots/tappini/residues_track_left_tappino.pdf")
    plt.close(fig)
    
    # Tappini check RIGHT
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(40, 30))
    for layer in range(1, 4+1):
        for group in range(0, 5):
            plotHist(axs[-layer][group], data=hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer) & (hits['WIRE'].isin(wires[group]) & (hits['X_LABEL'] == 2)), 'RESIDUES_TRACK'], name="Global residues RIGHT [LAYER %d, TAPPINO %d]" % (layer, group), title="|$x_{hit}$ - $x_{wire}$| - |$x_{track}$ - $x_{wire}$|", unit="mm", bins=np.arange(-4., 4., 0.05), label="""3/4 + 4/4,\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/tappini/residues_track_right_tappino.png")
    fig.savefig(args.outputdir + runname + "_plots/tappini/residues_track_right_tappino.pdf")
    plt.close(fig)


# Efficiency aka Missing hits
def plotEfficiency():
    
    if not os.path.exists(args.outputdir + runname + "_plots/efficiency/"): os.makedirs(args.outputdir + runname + "_plots/efficiency/")
    
    nmax = missinghits.groupby(['CHAMBER', 'WIRE', 'LAYER']).size().max()

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(40, 20))
    for chamber in range(4):
        axs[-chamber-1].set_title("Missing hit position [SL %d]" % chamber)
        axs[-chamber-1].set_xlabel("X position (mm)")
        axs[-chamber-1].set_ylabel("Z position (mm)")
        h, xbins, ybins, im = axs[-chamber-1].hist2d(missinghits.loc[missinghits['CHAMBER']==chamber, 'X'], missinghits.loc[missinghits['CHAMBER']==chamber, 'Z'], bins=[17*42, 4], range=[[-8.5*42., +8.5*42.], [-26., 26.]], vmin=0, vmax=nmax/42)
        cbar = fig.colorbar(im, ax=axs[-chamber-1], pad=0.01)
        cbar.set_label("Counts")
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/efficiency/missinghits_position.png")
    fig.savefig(args.outputdir + runname + "_plots/efficiency/missinghits_position.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 20))
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
    fig.savefig(args.outputdir + runname + "_plots/efficiency/missinghits_wire.png")
    fig.savefig(args.outputdir + runname + "_plots/efficiency/missinghits_wire.pdf")
    plt.close(fig)


    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(40, 20))
    for chamber in range(4):
        axs[-chamber-1].set_title("Missing hits [SL %d]" % (chamber))
        axs[-chamber-1].set_xlabel("Wire number")
        axs[-chamber-1].set_xticks(range(1, 17))
        axs[-chamber-1].set_ylabel("Layer")
        axs[-chamber-1].set_yticks(range(1, 5))
        h, xbins, ybins, im = axs[-chamber-1].hist2d(missinghits.loc[missinghits['CHAMBER'] == chamber, 'WIRE'], missinghits.loc[missinghits['CHAMBER'] == chamber, 'LAYER'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]], vmin=0, vmax=nmax)
        cbar = fig.colorbar(im, ax=axs[-chamber-1], pad=0.01)
        cbar.set_label("Counts")
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/efficiency/missinghits_cell.png")
    fig.savefig(args.outputdir + runname + "_plots/efficiency/missinghits_cell.pdf")
    plt.close(fig)


    nmax = hits.groupby(['CHAMBER', 'WIRE', 'LAYER']).size().max()
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(40, 20))
    for chamber in range(4):
        axs[-chamber-1].set_title("Position of hits included in the fit [SL %d]" % chamber)
        axs[-chamber-1].set_xlabel("X position (mm)")
        axs[-chamber-1].set_ylabel("Z position (mm)")
        h, xbins, ybins, im = axs[-chamber-1].hist2d(hits.loc[((hits['X_SEG'].notna()) & (hits['CHAMBER']==chamber)), 'X'], hits.loc[((hits['X_SEG'].notna()) & (hits['CHAMBER']==chamber)), 'Z'], bins=[17*42, 4], range=[[-8.5*42., +8.5*42.], [-26., 26.]], vmin=0, vmax=nmax/42)
        cbar = fig.colorbar(im, ax=axs[-chamber-1], pad=0.01)
        cbar.set_label("Counts")
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/efficiency/segmenthits_position.png")
    fig.savefig(args.outputdir + runname + "_plots/efficiency/segmenthits_position.pdf")
    plt.close(fig)
    
    
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(40, 20))
    for chamber in range(4):
        axs[-chamber-1].set_title("Efficiency [SL %d]" % (chamber))
        axs[-chamber-1].set_xlabel("Wire number")
        axs[-chamber-1].set_xticks(range(1, 17))
        axs[-chamber-1].set_ylabel("Layer")
        axs[-chamber-1].set_yticks(range(1, 5))
        h, xbins, ybins, im = axs[-chamber-1].hist2d(missinghits.loc[missinghits['CHAMBER'] == chamber, 'WIRE'], missinghits.loc[missinghits['CHAMBER'] == chamber, 'LAYER'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]], vmin=0, vmax=nmax)
        cbar = fig.colorbar(im, ax=axs[-chamber-1], pad=0.01)
        cbar.set_label("Counts")
        plotChannels(axs[-chamber-1], chamber, xbins, ybins)
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/efficiency/missinghits_cell.png")
    fig.savefig(args.outputdir + runname + "_plots/efficiency/missinghits_cell.pdf")
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
    
    
    # Efficiency
    fire = hits[hits['X_LABEL'] != 0].groupby(['CHAMBER', 'LAYER', 'WIRE'])['X_LABEL'].count().reset_index().rename(columns={'X_LABEL' : 'COUNTS'}) # count performed a random column
    miss = missinghits.groupby(['CHAMBER', 'LAYER', 'WIRE'])['X'].count().reset_index().rename(columns={'X' : 'COUNTS'})
    
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(40, 20))
    for chamber in range(4):
        h_fire, x_fire, y_fire = np.histogram2d(fire.loc[fire['CHAMBER'] == chamber, 'WIRE'], fire.loc[fire['CHAMBER'] == chamber, 'LAYER'], weights=fire.loc[fire['CHAMBER'] == chamber, 'COUNTS'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]])
        h_miss, x_miss, y_miss = np.histogram2d(miss.loc[miss['CHAMBER'] == chamber, 'WIRE'], miss.loc[miss['CHAMBER'] == chamber, 'LAYER'], weights=miss.loc[miss['CHAMBER'] == chamber, 'COUNTS'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]])
        h_deno = h_fire + h_miss
        h_eff = np.divide(h_fire, h_deno, where=(h_deno != 0))
        h_eff[h_deno == 0] = 0
        h_err = np.divide(h_fire * np.sqrt(h_deno) + h_deno * np.sqrt(h_fire), np.power(h_deno, 2), where=(h_deno != 0))

        axs[-chamber-1].set_title("[SL %d]" % (chamber))
        axs[-chamber-1].set_xlabel("Wire number")
        axs[-chamber-1].set_xticks(range(1, 17))
        axs[-chamber-1].set_ylabel("Layer")
        axs[-chamber-1].set_yticks(range(1, 5))
        im = axs[-chamber-1].imshow(h_eff.T, origin='lower', extent=[x_fire[0], x_fire[-1], y_fire[0], y_fire[-1]], interpolation='nearest', aspect='auto', vmin=0., vmax=1.)
        cbar = fig.colorbar(im, ax=axs[-chamber-1], pad=0.01)
        cbar.set_label("Efficiency")
        #plotChannels(axs[-chamber-1], chamber, x_fire, y_fire)
        for i in range(len(y_fire)-1):
            for j in range(len(x_fire)-1):
                axs[-chamber-1].text(x_fire[j]+0.5, y_fire[i]+0.5, "%.3f" % h_eff.T[i,j], color="w", ha="center", va="center", fontweight="bold")
        
        t_eff = h_fire.sum()/h_deno.sum()
        print("Chamber", chamber, "overall efficiency: %.4f +- %.4f" % (t_eff, t_eff * np.sqrt(1./h_fire.sum() + 1./h_deno.sum()) ) )
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/efficiency/efficiency.png")
    fig.savefig(args.outputdir + runname + "_plots/efficiency/efficiency.pdf")
    plt.close(fig)
    


# Occupancy
def plotOccupancy():
    if not os.path.exists(args.outputdir + runname + "_plots/occupancy/"): os.makedirs(args.outputdir + runname + "_plots/occupancy/")
    
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(40, 20))
    for chamber in range(4):
        axs[-chamber-1].set_title("Occupancy [SL %d]" % (chamber))
        axs[-chamber-1].set_xlabel("Wire number")
        axs[-chamber-1].set_xticks(range(1, 17))
        axs[-chamber-1].set_ylabel("Layer")
        axs[-chamber-1].set_yticks(range(1, 5))
        h, xbins, ybins, im = axs[-chamber-1].hist2d(occupancy.loc[occupancy['SL'] == chamber, 'WIRE_NUM'], occupancy.loc[occupancy['SL'] == chamber, 'LAYER'], weights=occupancy.loc[occupancy['SL'] == chamber, 'COUNTS'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]]) #, vmin=0, vmax=np.max(occupancy['COUNTS']))
        cbar = fig.colorbar(im, ax=axs[-chamber-1], pad=0.01)
        cbar.set_label("Counts")
        plotChannels(axs[-chamber-1], chamber, xbins, ybins)
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/occupancy/counts.png")
    fig.savefig(args.outputdir + runname + "_plots/occupancy/counts.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for chamber in range(4):
        for layer in range(4):
            occ = occupancy[((occupancy['SL'] == chamber) & (occupancy['LAYER'] == layer+1) & (occupancy['WIRE_NUM'] <= 16))]
            axs[-chamber-1][layer].set_title("Occupancy [SL %d, LAYER %d]" % (chamber, layer+1))
            axs[-chamber-1][layer].set_xlabel("Wire number")
            axs[-chamber-1][layer].set_ylabel("Counts")
            axs[-chamber-1][layer].set_xticks(range(1, 17))
            axs[-chamber-1][layer].bar(occ['WIRE_NUM'], occ['COUNTS'])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/occupancy/counts_vs_wire.png")
    fig.savefig(args.outputdir + runname + "_plots/occupancy/counts_vs_wire.pdf")
    plt.close(fig)


    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(40, 20))
    for chamber in range(4):
        axs[-chamber-1].set_title("Rate [SL %d]" % (chamber))
        axs[-chamber-1].set_xlabel("Wire number")
        axs[-chamber-1].set_xticks(range(1, 17))
        axs[-chamber-1].set_ylabel("Layer")
        axs[-chamber-1].set_yticks(range(1, 5))
        h, xbins, ybins, im = axs[-chamber-1].hist2d(occupancy.loc[occupancy['SL'] == chamber, 'WIRE_NUM'], occupancy.loc[occupancy['SL'] == chamber, 'LAYER'], weights=occupancy.loc[occupancy['SL'] == chamber, 'RATE'], bins=[16, 4], range=[[0.5, 16.5], [0.5, 4.5]]) #, vmin=0, vmax=np.max(occupancy['RATE']))
        cbar = fig.colorbar(im, ax=axs[-chamber-1], pad=0.01)
        cbar.set_label("Rate (Hz)")
        plotChannels(axs[-chamber-1], chamber, xbins, ybins)
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/occupancy/rate.png")
    fig.savefig(args.outputdir + runname + "_plots/occupancy/rate.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for chamber in range(4):
        for layer in range(4):
            occ = occupancy[((occupancy['SL'] == chamber) & (occupancy['LAYER'] == layer+1) & (occupancy['WIRE_NUM'] <= 16))]
            axs[-chamber-1][layer].set_title("Rate [SL %d, LAYER %d]" % (chamber, layer+1))
            axs[-chamber-1][layer].set_xlabel("Wire number")
            axs[-chamber-1][layer].set_ylabel("Rate (Hz)")
            axs[-chamber-1][layer].set_xticks(range(1, 17))
            axs[-chamber-1][layer].bar(occ['WIRE_NUM'], occ['RATE'])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/occupancy/rate_vs_wire.png")
    fig.savefig(args.outputdir + runname + "_plots/occupancy/rate_vs_wire.pdf")
    plt.close(fig)
    


# Parameters
def plotParameters():
    if not os.path.exists(args.outputdir + runname + "_plots/parameters/"): os.makedirs(args.outputdir + runname + "_plots/parameters/")
    
    if args.verbose >= 1: print("Read %d lines from segments.csv" % (len(segments), ))
    if args.verbose >= 2:
        print(segments.head(50))
        print(segments.tail(50))
    
    # Disentangle tracks and segments
    #segments['ANGLE_RAD'] = np.arctan(segments['M'])
    #segments['ANGLE_RAD'] = np.where(segments['ANGLE_RAD'] < 0, segments['ANGLE_RAD'] + math.pi/2., segments['ANGLE_RAD'] - math.pi/2.)
    ##segments.loc[segments['ANGLE_RAD'] < 0., 'ANGLE_RAD'] = segments.loc[segments['ANGLE_RAD'] < 0., 'ANGLE_RAD'] + np.pi # shift angle by pi
    #segments['ANGLE_DEG'] = np.degrees(segments['ANGLE_RAD']) #- 90.
    #segments['X0'] = - segments['Q'] / segments['M']

    tracks = segments.loc[segments['VIEW'] != '0'].copy()
    stubs = segments.loc[segments['VIEW'] == '0'].copy()

    if args.verbose >= 1: print("Numer of global tracks:", len(tracks[tracks['CHAMBER'].str.len() > 1]))

    stubs['XL1'] = -(-19.5 + stubs['Q']) / stubs['M']
    stubs['XL2'] = -(- 6.5 + stubs['Q']) / stubs['M']
    stubs['XL3'] = -(+ 6.5 + stubs['Q']) / stubs['M']
    stubs['XL4'] = -(+19.5 + stubs['Q']) / stubs['M']
    
    # Chi2
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(30, 20))
    for idx, chamber in zip(iax, isl):
        axs[idx].set_title("Segments $\chi^2$ [SL %d]" % chamber)
        axs[idx].set_xlabel("$\chi^2$/n.d.f.")
        axs[idx].set_ylabel("Counts")
        axs[idx].hist(stubs.loc[stubs['CHAMBER'] == str(chamber), 'CHI2'], bins=chi2bins)
        axs[idx].set_xscale("log")
        print("SL %d chi2 mean: %.2f" % (chamber, stubs.loc[stubs['CHAMBER'] == str(chamber), 'CHI2'].mean()))
    for w, view in enumerate(['XZ', 'YZ']):
        axs[(2, w)].set_title("Track $\chi^2$ [%s]" % view)
        axs[(2, w)].set_xlabel("$\chi^2$/n.d.f.")
        axs[(2, w)].set_ylabel("Counts")
        axs[(2, w)].hist(tracks.loc[(tracks['VIEW'] == view), 'CHI2'], bins=chi2bins)
        axs[(2, w)].set_xscale("log")
        print("View %s chi2 mean: %.2f" % (view, tracks.loc[(tracks['VIEW'] == view), 'CHI2'].mean()))
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/parameters/chi2.png")
    fig.savefig(args.outputdir + runname + "_plots/parameters/chi2.pdf")
    plt.close(fig)


    # Parameters
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for i, p in enumerate(['M_DEG', 'Q', 'X0', 'NHITS']):
        for chamber in range(4):
            axs[i][chamber].set_title("Segments %s [SL %d]" % (p, chamber))
            axs[i][chamber].set_xlabel(parlabel[p])
            axs[i][chamber].hist(stubs.loc[stubs['CHAMBER'] == str(chamber), p], bins=parbins[p])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/parameters/segment_par.png")
    fig.savefig(args.outputdir + runname + "_plots/parameters/segment_par.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(40, 30))
    for w, view in enumerate(['XZ', 'YZ']):
        for i, p in enumerate(['M_DEG', 'Q', 'X0', 'NHITS']):
            axs[i][w].set_title("Tracks %s [%s]" % (p, view))
            axs[i][w].set_xlabel(parlabel[p])
            axs[i][w].hist(tracks[p], bins=parbins[p])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/parameters/tracks_par.png")
    fig.savefig(args.outputdir + runname + "_plots/parameters/tracks_par.pdf")
    plt.close(fig)
    
    # X0
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for idx, cl in zip([(i, j) for i in range(4) for j in range(4)], range(0, 4*4)):
        chamber = int(cl % 4)
        layer = cl // 4 + 1
        axs[idx].set_title("Segment $x_{l}$ [CHAMBER %d, LAYER %d]" % (chamber, layer))
        axs[idx].set_xlabel(parlabel[p])
        axs[idx].hist(stubs.loc[(stubs['CHAMBER'] == str(chamber)), 'XL%d' % layer].dropna(), bins=parbins['X0'])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/parameters/segment_x0_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/parameters/segment_x0_chamber_layer.pdf")
    plt.close(fig)
    



def plotAlignment():
    if not os.path.exists(args.outputdir + runname + "_plots/alignment/"): os.makedirs(args.outputdir + runname + "_plots/alignment/")
    
    # Delta angles
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(40, 20))
    for i, sl in enumerate(list(itertools.combinations(config.SL_VIEW[config.PHI_VIEW], 2))):
        for num_hits in [0]:
            deltaT = segments[segments['CHAMBER'].isin([str(x) for x in sl])].groupby('ORBIT')['M_DEG'].diff().dropna()
            plotHist(axs[0][i], data=deltaT, name="Angle difference [SL %d vs SL %d]" % (sl[0], sl[1]), title="Angle difference [SL %d - SL %d]" % (sl[0], sl[1]), unit="deg", bins=np.arange(-10., 10., 0.5), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
            # 2D
            axs[1][i].set_title("Angle difference [SL %d vs SL %d]" % (sl[0], sl[1]))
            #axs[1][i].set_xlabel("Angle difference [SL %d - SL %d] (deg)" % (sl[0], sl[1]))
            axs[1][i].set_xlabel("Angle [SL %d] (deg)" % (sl[0]))
            axs[1][i].set_ylabel("Angle [SL %d] (deg)" % (sl[1]))
            segev = segments.loc[(segments['VIEW'] == '0') & (segments['CHAMBER'].isin([str(x) for x in sl]))].copy()
            segev['NCH'] = segev.groupby(['ORBIT'])['CHAMBER'].transform('count')
            segev = segev[segev['NCH'] == 2].sort_values('ORBIT')
            h = axs[1][i].hist2d(segev.loc[segev['CHAMBER'] == str(sl[0]), 'M_DEG'], segev.loc[segev['CHAMBER'] == str(sl[1]), 'M_DEG'], bins=[120, 120], range=[[-30., 30.], [-30., 30.]],)
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/alignment/segment_delta_angles.png")
    fig.savefig(args.outputdir + runname + "_plots/alignment/segment_delta_angles.pdf")
    plt.close(fig)
    
    # Delta positions (extrapolated)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(40, 20))
    for i, sl in enumerate(list(itertools.combinations(config.SL_VIEW[config.PHI_VIEW], 2))):
        for num_hits in [0]:
            segex = segments.loc[(segments['VIEW'] == config.PHI_VIEW.upper()) & (segments['CHAMBER'].isin([str(x) for x in sl]))].copy()
            segex['NCH'] = segex.groupby(['ORBIT'])['CHAMBER'].transform('count')
            segex = segex[segex['NCH'] == 2].sort_values('ORBIT')
            mask0, mask1 = segex['CHAMBER'] == str(sl[0]), segex['CHAMBER'] == str(sl[1])
            # The DELTA_EXTR is the difference between the extrapolated position from the OTHER chamber to the corresponding CHAMBER
            segex.loc[mask1, 'DELTA_EXTR'] = (config.SL_SHIFT[sl[1]][2] - segex.loc[mask0, 'Q'].to_numpy()) / segex.loc[mask0, 'M'].to_numpy() - (config.SL_SHIFT[sl[1]][2] - segex.loc[mask1, 'Q'].to_numpy()) / segex.loc[mask1, 'M'].to_numpy()
            segex.loc[mask0, 'DELTA_EXTR'] = (config.SL_SHIFT[sl[0]][2] - segex.loc[mask0, 'Q'].to_numpy()) / segex.loc[mask0, 'M'].to_numpy() - (config.SL_SHIFT[sl[0]][2] - segex.loc[mask1, 'Q'].to_numpy()) / segex.loc[mask1, 'M'].to_numpy()
            plotHist(axs[0][i], data=segex.loc[mask1, 'DELTA_EXTR'], name="Position difference [SL %d extrapolated from SL %d]" % (sl[1], sl[0]), title="Position difference [extrapolated from SL %d - SL %d]" % (sl[0], sl[1]), unit="mm", bins=np.arange(-50., 50., 1.), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
            plotHist(axs[1][i], data=segex.loc[mask0, 'DELTA_EXTR'], name="Position difference [SL %d extrapolated from SL %d]" % (sl[0], sl[1]), title="Position difference [extrapolated from SL %d - SL %d]" % (sl[1], sl[0]), unit="mm", bins=np.arange(-50., 50., 1.), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/alignment/segment_delta_positions.png")
    fig.savefig(args.outputdir + runname + "_plots/alignment/segment_delta_positions.pdf")
    plt.close(fig)
    
    # Delta parameters
    #fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(40, 30))
    #for i, p in enumerate(['M_DEG', 'Q', 'X0']):
    #    for j, chamber in enumerate(config.SL_VIEW[config.PHI_VIEW]):
    #        for num_hits in [0]:
    #            plotHist(axs[i][chamber], data=segments.loc[segments['CHAMBER'] == str(chamber), p], name="Segments %s [SL %d]" % (p, chamber), title=parlabel[p], unit=parunits[p], bins=parbins[p], label="3/4 + 4/4", linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits], fit="None")
    #fig.tight_layout()
    #fig.savefig(args.outputdir + runname + "_plots/alignment/segment_delta_par.png")
    #fig.savefig(args.outputdir + runname + "_plots/alignment/segment_delta_par.pdf")
    #plt.close(fig)
    


def plotTrigger():
    if not os.path.exists(args.outputdir + runname + "_plots/trigger/"): os.makedirs(args.outputdir + runname + "_plots/trigger/")
    
    #mt = pd.read_csv(args.inputdir + "matching.csv")
    #nA = pd.read_csv(args.inputdir + "denominator.csv")
    #nT = pd.read_csv(args.inputdir + "numerator.csv")
    
    #nmt = {}
    #for num_hits in [0, 3, 4]: nmt[num_hits] = mt.loc[(mt['n_hits_local'] == num_hits)] if num_hits != 0 else mt

    '''
    for p in trigger_plots.keys():

        plt.figure(figsize=(8, 6))
        for num_hits in [3, 4]:
            dfp = mt[mt['n_hits_local'] == num_hits]
            if 'vertical' in p: dfp = dfp[(dfp['angleXZ_global'] < 5.) & (dfp['angleYZ_global'] < 5.)]
            bin_heights, bin_borders = np.histogram(dfp[trigger_plots[p]["var"]] * trigger_plots[p]["scale"], range=trigger_plots[p]["hrange"], bins=trigger_plots[p]["bins"], density=isDensity)
            bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
            integral = np.sum(bin_heights)
            bin_heights, bin_errors = bin_heights / integral, np.sqrt(bin_heights) / integral
            #print(p, num_hits)
            popt, pcov = curve_fit(gaus, bin_centers, bin_heights, maxfev=10000, p0=[0., 1., 5.]) #p0= [0., 1., 3., 0.1, 20.]
            #print( *popt)
            x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 1000)
            plt.plot(x_interval_for_fit, gaus(x_interval_for_fit, *popt), color=colors[num_hits], lw=1)
            label_string = """{}/4,\n$\sigma={:.2f}\,{}$""".format(num_hits, abs(popt[2]), trigger_plots[p]["unit"])
            line = plt.errorbar(bin_centers, bin_heights, yerr=bin_errors, fmt=markers[num_hits], color=colors[num_hits], alpha=1, markersize=6, zorder=1, label=label_string)[0]
            line.set_clip_on(False)
        plt.ylabel("a. u.", size=20) #plots[p]["ylabel"]
        plt.xlabel(trigger_plots[p]["xlabel"], size=20)
        plt.legend(fontsize=15, frameon=True)
        plt.ylim(0, )
        plt.xlim(trigger_plots[p]["xlim"])
        plt.tight_layout()
        plt.savefig(args.outputdir + runname + "_plots/trigger/" + p + ".pdf")
        plt.savefig(args.outputdir + runname + "_plots/trigger/" + p + ".png")
    
    
        plt.figure(figsize=(8, 6))
        bin_heights, bin_borders = np.histogram(dfp[trigger_plots[p]["var"]] * trigger_plots[p]["scale"], range=trigger_plots[p]["hrange"], bins=trigger_plots[p]["bins"], density=isDensity)
        bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
        integral = np.sum(bin_heights)
        bin_heights, bin_errors = bin_heights / integral, np.sqrt(bin_heights) / integral
        #print(p)
        popt, pcov = curve_fit(gaus, bin_centers, bin_heights, maxfev=10000, p0=[0., 1., 5.])
        #print( *popt)
        x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 1000)
        plt.plot(x_interval_for_fit, gaus(x_interval_for_fit, *popt), color=colors[0], lw=1)
        label_string = """$\mu={:.2f}\,{}$\n$\sigma={:.2f}\,{}$""".format(abs(popt[0]), trigger_plots[p]["unit"], abs(popt[2]), trigger_plots[p]["unit"])
        line = plt.errorbar(bin_centers, bin_heights, yerr=bin_errors, fmt=markers[0], color=colors[0], alpha=1, markersize=6, zorder=1, label=label_string)[0]
        line.set_clip_on(False)
        plt.ylabel("a. u.", size=20) #plots[p]["ylabel"]
        plt.xlabel(trigger_plots[p]["xlabel"], size=20)
        plt.legend(fontsize=15, frameon=True)
        plt.ylim(0, )
        plt.xlim(trigger_plots[p]["xlim"])
        plt.tight_layout()
        plt.savefig(args.outputdir + runname + "_plots/trigger/" + p + "_all.pdf")
        plt.savefig(args.outputdir + runname + "_plots/trigger/" + p + "_all.png")
    '''
    
    
    # Difference between local and global fit
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    for num_hits in [0, 3, 4]:
        #trk = segments[segments['NHITS'] == num_hits].copy() if not num_hits == 0 else segments.copy()
        trk = segments[(segments['VIEW'] == config.PHI_VIEW.upper()) & ((segments['CHAMBER'] == str(config.SL_AUX_STR)) | (segments['CHAMBER'] == str(config.SL_TEST)))].copy()
        trk = trk[((trk['CHAMBER'] == config.SL_AUX_STR) & (trk['NHITS'] >= 7)) | ((trk['CHAMBER'] == str(config.SL_TEST)) & (trk['NHITS'] == num_hits if not num_hits == 0 else trk['NHITS'] >= 0))]
        trk['NCH'] = trk.groupby(['ORBIT'])['CHAMBER'].transform('count')
        trk = trk[trk['NCH'] == 2].sort_values('ORBIT')
        maskL, maskG = trk['CHAMBER'] == str(config.SL_TEST), trk['CHAMBER'] == str(config.SL_AUX_STR)
        deltaM = trk.loc[maskL, 'M_RAD'].to_numpy() - trk.loc[maskG, 'M_RAD'].to_numpy()
        deltaX = (config.SL_SHIFT[config.SL_TEST][2] - trk.loc[maskL, 'Q'].to_numpy()) / trk.loc[maskL, 'M'].to_numpy() - (config.SL_SHIFT[config.SL_TEST][2] - trk.loc[maskG, 'Q'].to_numpy()) / trk.loc[maskG, 'M'].to_numpy()
        deltaT = trk.loc[maskL, 'T_SCINTPMT'].to_numpy() - trk.loc[maskG, 'T0'].to_numpy()
        plotHist(axs[0], data=pd.Series(deltaM)*1.e3, name="Angle", title="$\\alpha_{local} - \\alpha_{global}$", unit="mrad", bins=np.arange(-100., 100., 5.), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
        plotHist(axs[1], data=pd.Series(deltaX), name="Position", title="$x^0_{local} - x^0_{global}$", unit="mm", bins=np.arange(-5., 5., 0.1), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
        plotHist(axs[2], data=pd.Series(deltaT), name="Time", title="$t^0_{scintillator} - t^0_{refit}$", unit="ns", bins=np.arange(-25./30.*50, 50./30.*50, 2./30.*50.), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    #axs[2].set_title("Angle vs position")
    #axs[2].set_xlabel("$\\alpha_{local} - \\alpha_{global}$ (mrad)")
    #axs[2].set_ylabel("$x^0_{local} - x^0_{global}$ (mm)")
    #axs[2].hist2d(x=mt['deltaM_global_local']*1.e3, y=mt['deltaQ_global_local'], bins=[np.arange(-50., 50., 2.), np.arange(-2., 2., 0.1)])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/trigger/global_vs_local.png")
    fig.savefig(args.outputdir + runname + "_plots/trigger/global_vs_local.pdf")
    plt.close(fig)
    
    # Difference between trigger and local fit
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    for num_hits in [0, 3, 4]:
        trk = segments[(segments['VIEW'] == '0') & (segments['CHAMBER'] == str(config.SL_TEST)) & (segments['NHITS'] == num_hits if not num_hits == 0 else segments['NHITS'] >= 0) & (segments['M_TRIG'].notna())].copy()
        plotHist(axs[0], data=(trk['M_TRIG_RAD'] - trk['M_RAD'])*1.e3, name="Angle", title="$\\alpha_{trigger} - \\alpha_{local}$", unit="mrad", bins=np.arange(-50., 50., 1.), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
        plotHist(axs[1], data=(trk['X0_TRIG'] - trk['X0']), name="Position", title="$x^0_{trigger} - x^0_{local}$", unit="mm", bins=np.arange(-2., 2., 0.01), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
        plotHist(axs[2], data=(trk['T_TRIG'] - trk['T_SCINTPMT']), name="Time", title="$t^0_{trigger} - t^0_{scintillator}$", unit="ns", bins=np.arange(-50./30.*50, 25./30.*50, 2./30.*50.), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/trigger/trigger_vs_local.png")
    fig.savefig(args.outputdir + runname + "_plots/trigger/trigger_vs_local.pdf")
    plt.close(fig)

    # Difference between trigger and global fit
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    for num_hits in [0, 3, 4]:
        trk = segments[(segments['VIEW'] == config.PHI_VIEW.upper()) & (segments['CHAMBER'] == str(config.SL_AUX_STR)) & (segments['NHITS'] >= num_hits if not num_hits == 0 else segments['NHITS'] >= 0) & (segments['M_TRIG'].notna())].copy()
        plotHist(axs[0], data=(trk['M_TRIG_RAD'] - trk['M_RAD'])*1.e3, name="Angle", title="$\\alpha_{trigger} - \\alpha_{global}$", unit="mrad", bins=np.arange(-100., 100., 1.), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
        plotHist(axs[1], data=(trk['X0_TRIG'] - trk['X0']), name="Position", title="$x^0_{trigger} - x^0_{global}$", unit="mm", bins=np.arange(-5., 5., 0.1), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
        plotHist(axs[2], data=(trk['T_TRIG'] - trk['T_SCINTPMT']), name="Time", title="$t^0_{trigger} - t^0_{scintillator}$", unit="ns", bins=np.arange(-50./30.*50, 25./30.*50, 2./30.*50.), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/trigger/trigger_vs_global.png")
    fig.savefig(args.outputdir + runname + "_plots/trigger/trigger_vs_global.pdf")
    plt.close(fig)
    

    plt.figure(figsize=(8, 6))
    for num_hits in [0, 3, 4]:
        atrk = segments[(segments['VIEW'] == '0') & (segments['CHAMBER'] == str(config.SL_TEST)) & (segments['NHITS']== num_hits if not num_hits == 0 else segments['NHITS'] >= 0)].copy()
        ttrk = atrk[atrk['M_TRIG'].notna()]#nT.loc[nT['n_hits_local'] == num_hits] if num_hits != 0 else nT
        eff, sigma_eff = len(ttrk)/len(atrk), len(ttrk)/len(atrk)*math.sqrt(1./len(ttrk) + 1./len(atrk))
        print("Trigger efficiency (%s hits): %d / %d = %.3f +- %.3f" % (str(num_hits) if num_hits else "all", len(ttrk), len(atrk), eff, sigma_eff))
        num, bin_borders = np.histogram(ttrk['M_RAD'], bins=effRange)
        den, bin_borders = np.histogram(atrk['M_RAD'], bins=effRange)
        den[den <= 0.] = 1
        effic = np.divide(num, den)
        num[num <= 0.] = 1
        errDown = effic * np.sqrt(1./num + 1./den)
        errUp = np.where(effic + errDown < 1., effic + errDown, 1.) - effic
        bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
        label_string = """{},\neff.=${:.3f}\pm{:.3f}$""".format(("%d/4" % num_hits if num_hits else "3/4 + 4/4"), eff, sigma_eff)
        line = plt.errorbar(bin_centers, effic, yerr=[errDown, errUp], fmt=markers[num_hits], color=colors[num_hits], alpha=1, markersize=6, zorder=1, label=label_string)[0]
        line.set_clip_on(False)
    plt.ylabel("Trigger efficiency", size=20)
    plt.xlabel("$\\phi$ [rad]", size=20)
    plt.legend(fontsize=15, frameon=True)
    #plt.ylim(0.8, 1.05)
    plt.xlim(effRange[0], effRange[-1])
    plt.tight_layout()
    plt.savefig(args.outputdir + runname + "_plots/trigger/efficiency.pdf")
    plt.savefig(args.outputdir + runname + "_plots/trigger/efficiency.png")
    
    
    
def plotTimes():
    if not os.path.exists(args.outputdir + runname + "_plots/times/"): os.makedirs(args.outputdir + runname + "_plots/times/")
    
    times['DELTAT_SCINTAND_VS_SCINTEXT'] = times['T_SCINTAND'] - times['T_SCINTEXT']
    times['DELTAT_SCINTAND_VS_SCINTINT'] = times['T_SCINTAND'] - times['T_SCINTINT']
    times['DELTAT_SCINTEXT_VS_SCINTINT'] = times['T_SCINTEXT'] - times['T_SCINTINT']
    times['DELTAT_SCINTAND_VS_SCINTPMT'] = times['T_SCINTAND'] - times['T_SCINTPMT']
    times['DELTAT_TRIG_VS_SCINTAND'] = times['T_TRIG'] - times['T_SCINTAND']
    times['DELTAT_TRIG_VS_SCINTPMT'] = times['T_TRIG'] - times['T_SCINTPMT']
    times['DELTAT_REFIT_VS_SCINTAND'] = times['T_REFIT'] - times['T_SCINTAND']
    times['DELTAT_REFIT_VS_SCINTPMT'] = times['T_REFIT'] - times['T_SCINTPMT']
    times['DELTAT_REFIT_VS_TRIG'] = times['T_REFIT'] - times['T_TRIG']
    
    times['TIMEMIN'] = ( (times['ORBIT'] - times['ORBIT'].min()) * DURATION['orbit'] * 1.e-9 / 60 ).astype(int) # Absolute time from the start of the run in minutes
    rate = times.groupby('TIMEMIN')[['T_TRIG', 'T_SCINTAND', 'T_SCINTEXT', 'T_SCINTINT', 'T_SCINTPMT']].count()
    rate[['T_TRIG', 'T_SCINTAND', 'T_SCINTEXT', 'T_SCINTINT', 'T_SCINTPMT']] = rate[['T_TRIG', 'T_SCINTAND', 'T_SCINTEXT', 'T_SCINTINT', 'T_SCINTPMT']] / 60.
    rate = rate.reset_index()
    
    # Rate
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(40, 10))
    axs.set_yscale("log")
    axs.plot(rate['TIMEMIN'], rate['T_TRIG'], 'o--', color='grey', alpha=1., label="FPGA")
    axs.plot(rate['TIMEMIN'], rate['T_SCINTAND'], 'o--', color='tab:purple', alpha=1., label="SiPM VME")
    axs.plot(rate['TIMEMIN'], rate['T_SCINTEXT'], 'o--', color='tab:cyan', alpha=1., label="SiPM EXT")
    axs.plot(rate['TIMEMIN'], rate['T_SCINTINT'], 'o--', color='tab:blue', alpha=1., label="SiPM NIM")
    axs.plot(rate['TIMEMIN'], rate['T_SCINTPMT'], 'o--', color='tab:red', alpha=1., label="PMT")
    axs.set_xlabel('Time (minutes)')
    axs.set_ylabel('Rate (Hz)')
    axs.legend(title='Trigger rates')
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/times/rates.png")
    fig.savefig(args.outputdir + runname + "_plots/times/rates.pdf")
    plt.close(fig)
    
    # Efficiencies
    nPMT = len(times[times['T_SCINTPMT'] > 0]) # Number of orbits with 129 trigger
    nAND = len(times[times['T_SCINTAND'] > 0]) # Number of orbits with 128 trigger
    nBOTH = len(times[(times['T_SCINTPMT'] > 0) & (times['T_SCINTAND'] > 0)]) # Number of events with both 128 and 129
    print("Trigger efficiency of SiPM AND against PMT: %d / %d = %.3f" % (nBOTH, nPMT, nBOTH/nPMT))
    print("Trigger efficiency of PMT against SiPM AND: %d / %d = %.3f" % (nBOTH, nAND, nBOTH/nAND))
    
    # Difference between trigger and global fit
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    
    plotHist(axs[(0, 0)], data=times['DELTAT_SCINTAND_VS_SCINTEXT'].dropna(), name="Time $\Delta$ between SiPM VME and SiPM EXT", title="$t_{VME} - t_{EXT}$", unit="ns", bins=np.arange(-100, 100, 5.), label="""$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    plotHist(axs[(0, 1)], data=times['DELTAT_SCINTAND_VS_SCINTINT'].dropna(), name="Time $\Delta$ between SiPM VME and SiPM NIM", title="$t_{VME} - t_{NIM}$", unit="ns", bins=np.arange(-100, 100, 5.), label="""$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    plotHist(axs[(1, 0)], data=times['DELTAT_SCINTEXT_VS_SCINTINT'].dropna(), name="Time $\Delta$ between SiPM EXT and SiPM NIM", title="$t_{EXT} - t_{NIM}$", unit="ns", bins=np.arange(-100, 100, 5.), label="""$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    plotHist(axs[(1, 1)], data=times['DELTAT_SCINTAND_VS_SCINTPMT'].dropna(), name="Time $\Delta$ between SiPM VME and PMT", title="$t_{SiPM} - t_{PMT}$", unit="ns", bins=np.arange(-100, 100, 5.), label="""$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/times/delta_scint.png")
    fig.savefig(args.outputdir + runname + "_plots/times/delta_scint.pdf")
    plt.close(fig)
    
    # Difference between trigger and scintillators
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    
    plotHist(axs[0], data=times['DELTAT_TRIG_VS_SCINTAND'].dropna(), name="Time $\Delta$ between trigger and SiPM scintillators", title="$t_{trigger} - t_{SiPM}$", unit="ns", bins=np.arange(-200, 100, 10.), label="""$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    plotHist(axs[1], data=times['DELTAT_TRIG_VS_SCINTPMT'].dropna(), name="Time $\Delta$ between trigger and PMT scintillators", title="$t_{trigger} - t_{PMT}$", unit="ns", bins=np.arange(-200, 100, 10.), label="""$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/times/delta_triggers.png")
    fig.savefig(args.outputdir + runname + "_plots/times/delta_triggers.pdf")
    plt.close(fig)
    
    # Difference between global fit and triggers and scintillators
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    
    plotHist(axs[0], data=times['DELTAT_REFIT_VS_SCINTAND'].dropna(), name="Time $\Delta$ between fitted T0 and SiPM scintillators", title="$t_{fit} - t_{SiPM}$", unit="ns", bins=np.arange(-100, 100, 2), label="""$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    plotHist(axs[1], data=times['DELTAT_REFIT_VS_SCINTPMT'].dropna(), name="Time $\Delta$ between fitted T0 and PMT scintillators", title="$t_{fit} - t_{PMT}$", unit="ns", bins=np.arange(-100, 100, 2), label="""$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    plotHist(axs[2], data=times['DELTAT_REFIT_VS_TRIG'].dropna(), name="Time $\Delta$ between fitted T0 and trigger", title="$t_{fit} - t_{trigger}$", unit="ns", bins=np.arange(-100, 100, 2), label="""$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/times/delta_refit.png")
    fig.savefig(args.outputdir + runname + "_plots/times/delta_refit.pdf")
    plt.close(fig)
    
    # Segment refit
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
    for idx, chamber in zip(iax, isl):
        for num_hits in [0, 3, 4]:
            pars = plotHist(axs[idx], data=nhits[num_hits].loc[nhits[num_hits]['CHAMBER'] == chamber, 'DELTAT_REFIT_VS_SCINTPMT'], name="Refit $\Delta t$ [SL %d]" % (chamber), title="$\Delta t$", unit="ns", bins=np.arange(-100, 100, 2.), label="""{},\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
            if num_hits == 0: print("Time calibration CHAMBER %d:\t%.1f ns" % (chamber, pars[1]))
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/times/delta_refit_chamber.png")
    fig.savefig(args.outputdir + runname + "_plots/times/delta_refit_chamber.pdf")
    plt.close(fig)
    
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 30))
    for chamber in range(4):
        for layer in range(4):
            pars = plotHist(axs[-chamber-1][layer], data=hits.loc[(hits['CHAMBER'] == chamber) & (hits['LAYER'] == layer+1), 'DELTAT_REFIT_VS_SCINTPMT'], name="Refit $\Delta t$ [SL %d, LAYER %d]" % (chamber, layer+1), title="$\Delta t$", unit="ns", bins=np.arange(-100, 100, 2.), label="$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$", linecolor=colors[0], markercolor=colors[0], markerstyle=markers[0])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/times/delta_refit_chamber_layer.png")
    fig.savefig(args.outputdir + runname + "_plots/times/delta_refit_chamber_layer.pdf")
    plt.close(fig)
    
    '''
    print(segments.columns)
    
    segments['DELTAT_T0_SCINT'] = segments['T0'] - segments['T0_SCINT']
    segments['DELTAT_SCINT_TRIG'] = segments['T0_SCINT'] - segments['T_TRIG']
    print(segments[['ORBIT', 'CHAMBER', 'T0', 'T0_SCINT', 'T_TRIG']].head(50))
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 20))
    for idx, chamber in zip(iax, isl):
        axs[idx].set_title("Timebox [SL %d]" % chamber)
        axs[idx].set_xlabel("Time (ns)")
        axs[idx].set_ylabel("Counts")
        for num_hits in [0]:
            plotHist(axs[idx], data=segments.loc[segments['CHAMBER'] == str(chamber), 'DELTAT_T0_SCINT'].dropna(), name="Time difference [CHAMBER %d]" % (chamber), title="$t_{0}$ - $t_{scint}$ (ns)", unit="ns", bins=np.arange(-50., 50., 0.25), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/times/deltaT_t0_scint.png")
    fig.savefig(args.outputdir + runname + "_plots/times/deltaT_t0_scint.pdf")
    plt.close(fig)
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 20))
    for idx, chamber in zip(iax, isl):
        axs[idx].set_title("Timebox [SL %d]" % chamber)
        axs[idx].set_xlabel("Time (ns)")
        axs[idx].set_ylabel("Counts")
        for num_hits in [0]:
            plotHist(axs[idx], data=segments.loc[segments['CHAMBER'] == str(chamber), 'DELTAT_SCINT_TRIG'].dropna(), name="Time difference [CHAMBER %d]" % (chamber), title="$t_{scint}$ - $t_{trigger}$ (ns)", unit="ns", bins=np.arange(-50., 50., 0.5), label="""{},\n$x_0=%.2f\,%s$,\n$\sigma=%.2f\,%s$""".format("%d/4" % num_hits if num_hits else "3/4 + 4/4"), linecolor=colors[num_hits], markercolor=colors[num_hits], markerstyle=markers[num_hits])
    fig.tight_layout()
    fig.savefig(args.outputdir + runname + "_plots/times/deltaT_scint_trig.png")
    fig.savefig(args.outputdir + runname + "_plots/times/deltaT_scint_trig.pdf")
    plt.close(fig)
    '''

if 'boxes' in args.plot or args.all: plotBoxes()
if 'residues' in args.plot or args.all: plotResidues()
#if 'res2D' in args.plot or args.all: plotResidues2D()
if 'resolution' in args.plot or args.all: plotResolution()
if 'tappini' in args.plot or args.all: plotTappini()
if 'efficiency' in args.plot or args.all: plotEfficiency()
if 'occupancy' in args.plot or args.all: plotOccupancy()
if 'parameters' in args.plot or args.all: plotParameters()
if 'alignment' in args.plot: plotAlignment()
if 'trigger' in args.plot: plotTrigger()
if 'times' in args.plot: plotTimes()


if args.verbose >= 1: print("Done.")