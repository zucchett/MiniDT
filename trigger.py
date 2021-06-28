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
parser.add_argument("-i", "--inputdir", action="store", type=str, dest="inputdir", default="./output/Run000967_csv/", help="Provide directory of the input files (csv)")
parser.add_argument("-o", "--outputdir", action="store", type=str, dest="outputdir", default="./output/", help="Specify output directory")
parser.add_argument("-v", "--verbose", action="store", type=int, default=0, dest="verbose", help="Specify verbosity level")
args = parser.parse_args()

if len(args.outputdir) == 0: args.outputdir = args.inputdir

runname = [x for x in args.inputdir.split('/') if 'Run' in x][0].replace("_csv", "") if "Run" in args.inputdir else "Run000000"



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

# Use the convention where 0 is vertical
triggers['M_RAD'] = np.arctan(triggers['M'])
segments['M_RAD'] = np.arctan(segments['M'])

triggers['M_RAD'] = np.where(triggers['M_RAD'] < 0, triggers['M_RAD'] + math.pi/2., triggers['M_RAD'] - math.pi/2.)
segments['M_RAD'] = np.where(segments['M_RAD'] < 0, segments['M_RAD'] + math.pi/2., segments['M_RAD'] - math.pi/2.)

triggers['M_DEG'] = np.degrees(triggers['M_RAD'])
segments['M_DEG'] = np.degrees(segments['M_RAD'])

#triggers['M_DEG'] = np.where(triggers['M_DEG'] < 0, triggers['M_DEG'] + 90, triggers['M_DEG'] - 90)
#segments['M_DEG'] = np.where(segments['M_DEG'] < 0, segments['M_DEG'] + 90, segments['M_DEG'] - 90)


df = pd.DataFrame(columns=['n_hits_local', 'angleXZ_global', 'angleYZ_global', 'deltaM_global_local', 'deltaQ_global_local', 'deltaT_global_local', 'deltaM_global_trigger', 'deltaQ_global_trigger', 'deltaT_global_trigger', 'deltaM_local_trigger', 'deltaQ_local_trigger', 'deltaT_local_trigger'])
nA = pd.DataFrame(columns=['n_hits_local', 'angleXZ_global', 'angleYZ_global', 'radXZ_global', 'radYZ_global'])
nT = pd.DataFrame(columns=['n_hits_local', 'angleXZ_global', 'angleYZ_global', 'radXZ_global', 'radYZ_global'])

gp = segments.groupby('ORBIT')
iEv, nEv = 0, len(gp)
for iorbit, ev in gp:
    iEv += 1
    #if iEv > 10000: break
    if args.verbose >= 1 and iEv % 100 == 0: print("Trigger-segment matching... [%.2f %%]" % (100.*iEv/nEv), end='\r')
    glt = ev[ev['VIEW'] == 'XZ']
    glb = ev[(ev['VIEW'] == 'YZ') & ((ev['CHAMBER'] == '0,3') | (ev['CHAMBER'] == '0,2,3'))]
    loc = ev[(ev['VIEW'] == 'YZ') & (ev['CHAMBER'] == '2')]
    if len(loc) > 0 and len(glb) > 0 and len(glt) > 0:
        n_hits_local, angleXZ_global, angleYZ_global, radXZ_global, radYZ_global = loc['NHITS'].values[0], glb['M_DEG'].values[0], glt['M_DEG'].values[0], glb['M_RAD'].values[0], glt['M_RAD'].values[0]
        delta_gm, delta_gx0, delta_gt, delta_lm, delta_lx0, delta_lt = 9999., 9999., 9999., 9999., 9999., 9999.
        if glb['NHITS'].values[0] >= 7:
            if loc['NHITS'].values[0] > 2: # and abs(glb['M_DEG'].values[0]) > 85. and abs(glt['M_DEG'].values[0]) > 85.:
                nA = nA.append(pd.DataFrame.from_dict({'n_hits_local' : [n_hits_local], 'angleXZ_global' : [angleXZ_global], 'angleYZ_global' : [angleYZ_global], 'radXZ_global' : [radXZ_global], 'radYZ_global' : [radYZ_global]}), ignore_index=True)

                gm_rad, gm, gq, gt = glb['M_RAD'].values[0], glb['M'].values[0], glb['Q'].values[0], glb['T0'].values[0]
                gx0 = (z0 - gq) / gm # global fit has global coordinates
                lm_rad, lm, lq, lt = loc['M_RAD'].values[0], loc['M'].values[0], loc['Q'].values[0], loc['T0'].values[0]
                lx0 = (z0 - lq) / lm # local fit has local coordinates
                # Trigger
                if len(triggers[triggers['ORBIT'] == iorbit]) > 0:
                    nT = nT.append(pd.DataFrame.from_dict({'n_hits_local' : [n_hits_local], 'angleXZ_global' : [angleXZ_global], 'angleYZ_global' : [angleYZ_global], 'radXZ_global' : [radXZ_global], 'radYZ_global' : [radYZ_global]}), ignore_index=True)
                    
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
                
#                delta_gmV, delta_gx0V, delta_gtV = (delta_gm, delta_gx0, delta_gt) if (abs(angleXZ_global) < 5. or abs(angleXZ_global + 180.) < 5.) and (abs(angleYZ_global) < 5. or abs(angleYZ_global + 180.) < 5.) else (np.nan, np.nan, np.nan)
                
                df = df.append(pd.DataFrame.from_dict({ 'n_hits_local' : [n_hits_local], 'angleXZ_global' : [angleXZ_global], 'angleYZ_global' : [angleYZ_global], \
                    'deltaM_global_local' : [gm_rad - lm_rad], 'deltaQ_global_local' : [gx0 - lx0], 'deltaT_global_local' : [gt - lt], \
                    'deltaM_global_trigger' : [delta_gm], 'deltaQ_global_trigger' : [delta_gx0], 'deltaT_global_trigger' : [delta_gt], \
                    'deltaM_local_trigger' : [delta_lm], 'deltaQ_local_trigger' : [delta_lx0], 'deltaT_local_trigger' : [delta_lt], \
                }), ignore_index=True)

if args.verbose >= 1: print("Trigger - segment matching completed.")


if len(args.outputdir) > 0:
    df.to_csv(args.outputdir + "_csv/matching.csv", header=True, index=False)
    nA.to_csv(args.outputdir + "_csv/denominator.csv", header=True, index=False)
    nT.to_csv(args.outputdir + "_csv/numerator.csv", header=True, index=False)

if args.verbose >= 1: print("Done.")