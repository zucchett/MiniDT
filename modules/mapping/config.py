"""Configuration for the reconstruction code"""


# Layer    # Parameters

#          +--+--+--+--+--+--+--+--+
# 1        |  1  |  5  |  9  |  13 | 17 ...
#          +--+--+--+--+--+--+--+--+
# 2           |  3  |  7  |  11 | 15 ...
#          +--+--+--+--+--+--+--+--+
# 3        |  2  |  6  |  10 |  14 | 18 ...
#          +--+--+--+--+--+--+--+--+
# 4           |  4  |  8  |  12 | 16 ...
#          +--+--+--+--+--+--+


DURATION = {                         
    'orbit:bx': 3564,
    'orbit': 3564*25,
    'bx': 25.,
    'tdc': 25./30
}

TIME_WINDOW = (-50, 500)

### Chamber configuration
NCHANNELS  = 64    # channels per SL
NSL        = 4     # number of SL
### Cell parameters
XCELL     = 42.                      # cell width in mm
ZCELL     = 13.                      # cell height in mm
ZCHAMB    = 550.                     # spacing betweeen chambers in mm

TDRIFT    = 15.6*DURATION['bx']    # drift time in ns
VDRIFT    = XCELL*0.5 / TDRIFT     # drift velocity in mm/ns 

layer_z     = [  1,            3,            2,            4,         ]
chanshift_x = [  0,            -1,           0,            -1,        ]
posshift_z  = [  ZCELL*1.5,    -ZCELL*0.5,   ZCELL*0.5,    -ZCELL*1.5 ]
posshift_x  = [  -7.5*XCELL,   -7.5*XCELL,   -7.0*XCELL,   -7.0*XCELL ]

