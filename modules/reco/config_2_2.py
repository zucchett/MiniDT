"""Configuration for the reconstruction code"""

from math import pi, sqrt

sl_w = 700     # chamber width (along X)
sl_l = 700     # chamber length (along Y)
sl_h = 62      # chamber height
dsl = 782      # Z of chamber 2
hsl = 951      # height of the stand from the floor level

# Shifts of each SL along the (X,Y,Z) axis
SL_SHIFT = {
    0: (0.,   0.,   hsl + 0.5*sl_h),
    1: (0.,   0.,   hsl + 1.5*sl_h),
    2: (0.,   0.,   hsl + dsl + 0.5*sl_h),
    3: (0.,   0.,   hsl + dsl + 1.5*sl_h),
}

# Rotation of each SL around the (X,Y,Z) axis
SL_ROTATION = {
    0: (0,        0,          0),
    1: (0,        0,     0.5*pi),
    2: (0,        0,          0),
    3: (0,        0,     0.5*pi)
}

# View of each SL (XZ, YZ)
SL_VIEW = {
    'xz': [0, 2],
    'yz': [1, 3],
}

# Fit quality cuts
FIT_ANGLES = (-1.1, +1.1)
FIT_M_MIN = 1./sqrt(3.)
FIT_CHI2NDF_MAX = 10.0

# Fit parameters
NHITS_LOCAL_MIN = 3
NHITS_LOCAL_MAX = 20
