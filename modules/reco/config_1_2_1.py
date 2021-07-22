"""Configuration for the reconstruction code"""

import numpy as np
from math import pi, sqrt

# Shifts of each SL along the (X,Y,Z) axis
SL_SHIFT = {
    0: (0.,   2.5,   219.8),
    1: (0.,   0.,   977.3),
    2: (0.,   0.,   1035.6),
    3: (0.,   -2.,   1819.8),
}

# Rotation of each SL around the (X,Y,Z) axis
SL_ROTATION = {
    0: (0,        0.013,     0.5*pi), #-0.0125
    1: (0,        0,          0),
    2: (0,        0,     0.5*pi),
    3: (0,        0,     0.5*pi)
}

# View of each SL (XZ, YZ)
SL_VIEW = {
    'xz': [1, ],
    'yz': [0, 2, 3],
}
SL_FITS = {
    'xz': [[1], ],
    'yz': [[0, 3], [0], [2], [3]],
}

# Definition of the vies (phi, theta)
PHI_VIEW = 'yz'
THETA_VIEW = 'xz'

# Number of the triggering chamber to be tested
SL_TEST = 2
SL_AUX = max(SL_FITS[PHI_VIEW], key=len)

# Scintillator time offset
TIME_OFFSET_SCINT = -95.0

# Run 1209+
TIME_OFFSET_SL = {
    0 : -1.1,
    1 : 6.4,
    2 : 0.5,
    3 : -2.6,
}

# Time offset for each LAYER (all SL)
TIME_OFFSET_LAYER = {
    1 : 0.,
    2 : 0.,
    3 : 0.,
    4 : 0.,
}

# Selection of hits to use for the fits
TIMEBOX = (-5, 380)

# Fit quality cuts
FIT_ANGLES = (-1.1, +1.1)
FIT_M_MIN = 1./sqrt(3.)
FIT_CHI2NDF_MAX = 10.0
FIT_GLOBAL_CHI2NDF_MAX = 10.0
FIT_GLOBAL_MINHITS = 7
FIT_LOCAL_MINHITS = 3

# Noise cuts
NHITS_LOCAL_MIN = 3
NHITS_LOCAL_MAX = 20

# Trigger
TRIGGER_CELL = 0
TRIGGER_CELL_OFFSET = 3