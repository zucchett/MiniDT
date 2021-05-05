"""Configuration for the reconstruction code"""

import numpy as np
from math import pi, sqrt

# Shifts of each SL along the (X,Y,Z) axis
SL_SHIFT = {
    0: (0.,   0.,   338. + 53.5/2.),
    1: (0.,   0.,   338. + 53.5 + 111.5 + 53.5/2.),
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
    'xz': [[0, 1, 2]],
    'yz': [[3]],
}

# Scintillator time offset
TIME_OFFSET_SCINT = 100.0

# Time offset for each SL
TIME_OFFSET_SL = {
    0 : 13.0,
    1 : 13.1,
    2 : 12.7,
    3 : 12.,
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

# Fit parameters
NHITS_LOCAL_MIN = 3
NHITS_LOCAL_MAX = 20
