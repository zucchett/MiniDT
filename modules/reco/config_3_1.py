"""Configuration for the reconstruction code"""

import numpy as np
from math import pi, sqrt

# Shifts of each SL along the (X,Y,Z) axis
SL_SHIFT = {
    0: (0.,   0.,   338. + 53.5/2.),
    1: (1.23,   0.,   338. + 53.5 + 111.5 + 53.5/2.),
    2: (0.,   0.,   338. + 53.5 + 111.5 + 53.5 + 112.5 + 53.5/2.),
    3: (0.,   0.,   338. + 53.5 + 111.5 + 53.5 + 112.5 + 53.5 + 41.0 + 53.5/2.),
}

# Rotation of each SL around the (X,Y,Z) axis
SL_ROTATION = {
    0: (0,        0,          0),
    1: (0,        0,          0),
    2: (0,        0,          0),
    3: (0,        0,     0.5*pi),
}

# View of each SL (XZ, YZ)
SL_VIEW = {
    'xz': [0, 1, 2],
    'yz': [3],
}
SL_FITS = {
    'xz': [[0], [1], [2], [0, 2]],
    'yz': [[3]],
}

# Definition of the vies (phi, theta)
PHI_VIEW = 'xz'
THETA_VIEW = 'yz'

# Number of the triggering chamber to be tested
SL_TEST = 1
SL_AUX = max(SL_FITS[PHI_VIEW], key=len)
SL_AUX_STR = ','.join(map(str, SL_AUX))

# Scintillator time offset
TIME_OFFSET_SCINTAND = -236.0+34
TIME_OFFSET_SCINTEXT = -236.0+34
TIME_OFFSET_SCINTINT = -236.0+73.0+34
TIME_OFFSET_SCINTPMT = -120.0-25.0+34

# Time offset for each SL
TIME_OFFSET_SL = {
    0 : -3.7 + 0.7,
    1 : -5.4,
    2 : -4.0,
    3 : -2.9 + 0.6,
}

# Time offset for each LAYER (all SL)
TIME_OFFSET_LAYER = {
    1 : 0.,
    2 : 0.,
    3 : 0.,
    4 : 0.,
}

# Selection of hits to use for the fits
TIMEBOX = (-5, 400)

# Fit quality cuts
FIT_ANGLES = (-1.1, +1.1)
FIT_M_MIN = 1./sqrt(3.)
FIT_CHI2NDF_MAX = 2.0
FIT_GLOBAL_CHI2NDF_MAX = 10.0
FIT_GLOBAL_MINHITS = 7
FIT_LOCAL_MINHITS = 3

# Noise cuts
NHITS_LOCAL_MIN = 3
NHITS_LOCAL_MAX = 10

# Trigger
TRIGGER_CELL = 7
TRIGGER_CELL_OFFSET = +0.5 # in number of cells
N_TRIGGER_CELLS = 5

# Plot
PLOT_RANGE = {'x': (-400, 400), 'y': (0, 1000)}
