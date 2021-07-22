"""Configuration for the reconstruction code"""

import numpy as np
from math import pi, sqrt

# Shifts of each SL along the (X,Y,Z) axis
SL_SHIFT = {
    0: (0.,   0.,   338. + 53.5/2.),
    1: (1.21,   0.,   338. + 53.5 + 111.5 + 53.5/2.),
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

# Scintillator time offset
TIME_OFFSET_SCINT = -120.0

# Time offset for each SL
TIME_OFFSET_SL = {
    0 : -0.2,
    1 : -1.7,
    2 : -8.9,
    3 : -10.0,
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
FIT_CHI2NDF_MAX = 4.0
FIT_GLOBAL_CHI2NDF_MAX = 4.0
FIT_GLOBAL_MINHITS = 7
FIT_LOCAL_MINHITS = 3

# Noise cuts
NHITS_LOCAL_MIN = 3
NHITS_LOCAL_MAX = 20

# Trigger
TRIGGER_CELL = 7
TRIGGER_CELL_OFFSET = +0.