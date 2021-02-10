"""Configuration for the reconstruction code"""

import numpy as np
from math import pi, sqrt

# Shifts of each SL along the (X,Y,Z) axis
SL_SHIFT = {
    0: (0.,   0.,   219.8),
    1: (0.,   0.,   977.7),
    2: (0.,   0.,   1036.0),
    3: (0.,   0.,   1819.5),
}

# Rotation of each SL around the (X,Y,Z) axis
SL_ROTATION = {
    0: (0,        0,     0.5*pi),
    1: (0,        0,          0),
    2: (0,        0,     0.5*pi),
    3: (0,        0,     0.5*pi)
}

# View of each SL (XZ, YZ)
SL_VIEW = {
    'xz': [1, ],
    'yz': [0, 2, 3],
}

# Selection of hits to use for the fits
TIMEBOX = (-5, 380)

# Fit quality cuts
FIT_ANGLES = (-1.1, +1.1)
FIT_M_MIN = 1./sqrt(3.)
FIT_CHI2NDF_MAX = 10.0

# Fit parameters
NHITS_LOCAL_MIN = 3
NHITS_LOCAL_MAX = 20