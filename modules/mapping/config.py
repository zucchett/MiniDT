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

#TIME_WINDOW = (-50, 500)

### Chamber configuration
NCHANNELS  = 64    # channels per SL
NSL        = 4     # number of SL
### Cell parameters
XCELL     = 42.                      # cell width in mm
ZCELL     = 13.                      # cell height in mm
ZCHAMB    = 550.                     # spacing betweeen chambers in mm

TDRIFT    = 15.6*DURATION['bx']    # drift time in ns
VDRIFT    = XCELL*0.5 / TDRIFT     # drift velocity in mm/ns 

#layer_z     = [  1,            3,            2,            4,         ]
#chanshift_x = [  0,            -1,           0,            -1,        ]
#posshift_z  = [  ZCELL*1.5,    -ZCELL*0.5,   ZCELL*0.5,    -ZCELL*1.5 ]
#posshift_x  = [  -7.5*XCELL,   -7.5*XCELL,   -7.0*XCELL,   -7.0*XCELL ]

layer_z     = [  4,            2,            3,            1,         ]
posshift_z  = [  -ZCELL*1.5,   -ZCELL*0.5,   ZCELL*0.5,    +ZCELL*1.5 ]
posshift_x  = [  -7.0*XCELL,   -7.5*XCELL,   -7.0*XCELL,   -7.5*XCELL ]

tappino = {
    0 : 0,
    1 : 1,
    2 : 1,
    3 : 2,
    4 : 2, 
    5 : 2,
    6 : 2,
    7 : 3,
    8 : 3,
    9 : 3,
    10 : 3,
    11 : 4,
    12 : 4,
    13 : 4,
    14 : 4,
    15 : 5,
    16 : 5,
    17 : 0,
}


# keys: FPGA, TDC_CHANNEL
VIRTEX7_LAYER = {
	1  : 1,
	2  : 3,
	3  : 2,
	4  : 4,
	5  : 1,
	6  : 3,
	7  : 2,
	8  : 4,
	9  : 1,
	10 : 3,
	11 : 2,
	12 : 4,
	13 : 1,
	14 : 3,
	15 : 2,
	16 : 4,
	17 : 1,
	18 : 3,
	19 : 2,
	20 : 4,
	21 : 1,
	22 : 3,
	23 : 2,
	24 : 4,
	25 : 1,
	26 : 3,
	27 : 2,
	28 : 4,
	29 : 1,
	30 : 3,
	31 : 2,
	32 : 4,
	33 : 1,
	34 : 3,
	35 : 2,
	36 : 4,
	37 : 1,
	38 : 3,
	39 : 2,
	40 : 4,
	41 : 1,
	42 : 3,
	43 : 2,
	44 : 4,
	45 : 1,
	46 : 3,
	47 : 2,
	48 : 4,
	49 : 1,
	50 : 3,
	51 : 2,
	52 : 4,
	53 : 1,
	54 : 3,
	55 : 2,
	56 : 4,
	57 : 1,
	58 : 3,
	59 : 2,
	60 : 4,
	61 : 1,
	62 : 3,
	63 : 2,
	64 : 4,

	65 : 1,
	66 : 3,
	67 : 2,
	68 : 4,
	69 : 1,
	70 : 3,
	71 : 2,
	72 : 4,
	73 : 1,
	74 : 3,
	75 : 2,
	76 : 4,
	77 : 1,
	78 : 3,
	79 : 2,
	80 : 4,
	81 : 1,
	82 : 3,
	83 : 2,
	84 : 4,
	85 : 1,
	86 : 3,
	87 : 2,
	88 : 4,
	89 : 1,
	90 : 3,
	91 : 2,
	92 : 4,
	93 : 1,
	94 : 3,
	95 : 2,
	96 : 4,
	97 : 1,
	98 : 3,
	99 : 2,
	100 : 4,
	101 : 1,
	102 : 3,
	103 : 2,
	104 : 4,
	105 : 1,
	106 : 3,
	107 : 2,
	108 : 4,
	109 : 1,
	110 : 3,
	111 : 2,
	112 : 4,
	113 : 1,
	114 : 3,
	115 : 2,
	116 : 4,
	117 : 1,
	118 : 3,
	119 : 2,
	120 : 4,
	121 : 1,
	122 : 3,
	123 : 2,
	124 : 4,
	125 : 1,
	126 : 3,
	127 : 2,
	128 : 4,
}

VIRTEX7_WIRE = {
	1  : 1,
	2  : 1,
	3  : 1,
	4  : 1,
	5  : 2,
	6  : 2,
	7  : 2,
	8  : 2,
	9  : 3,
	10 : 3,
	11 : 3,
	12 : 3,
	13 : 4,
	14 : 4,
	15 : 4,
	16 : 4,
	17 : 5,
	18 : 5,
	19 : 5,
	20 : 5,
	21 : 6,
	22 : 6,
	23 : 6,
	24 : 6,
	25 : 7,
	26 : 7,
	27 : 7,
	28 : 7,
	29 : 8,
	30 : 8,
	31 : 8,
	32 : 8,
	33 : 9,
	34 : 9,
	35 : 9,
	36 : 9,
	37 : 10,
	38 : 10,
	39 : 10,
	40 : 10,
	41 : 11,
	42 : 11,
	43 : 11,
	44 : 11,
	45 : 12,
	46 : 12,
	47 : 12,
	48 : 12,
	49 : 13,
	50 : 13,
	51 : 13,
	52 : 13,
	53 : 14,
	54 : 14,
	55 : 14,
	56 : 14,
	57 : 15,
	58 : 15,
	59 : 15,
	60 : 15,
	61 : 16,
	62 : 16,
	63 : 16,
	64 : 16,

	65 : 1,
	66 : 1,
	67 : 1,
	68 : 1,
	69 : 2,
	70 : 2,
	71 : 2,
	72 : 2,
	73 : 3,
	74 : 3,
	75 : 3,
	76 : 3,
	77 : 4,
	78 : 4,
	79 : 4,
	80 : 4,
	81 : 5,
	82 : 5,
	83 : 5,
	84 : 5,
	85 : 6,
	86 : 6,
	87 : 6,
	88 : 6,
	89 : 7,
	90 : 7,
	91 : 7,
	92 : 7,
	93 : 8,
	94 : 8,
	95 : 8,
	96 : 8,
	97 : 9,
	98 : 9,
	99 : 9,
	100 : 9,
	101 : 10,
	102 : 10,
	103 : 10,
	104 : 10,
	105 : 11,
	106 : 11,
	107 : 11,
	108 : 11,
	109 : 12,
	110 : 12,
	111 : 12,
	112 : 12,
	113 : 13,
	114 : 13,
	115 : 13,
	116 : 13,
	117 : 14,
	118 : 14,
	119 : 14,
	120 : 14,
	121 : 15,
	122 : 15,
	123 : 15,
	124 : 15,
	125 : 16,
	126 : 16,
	127 : 16,
	128 : 16,
}


VIRTEX7_MAP = {
	1  : [1, 0],
	2  : [3, 0],
	3  : [2, 0],
	4  : [4, 0],
	5  : [1, 0],
	6  : [3, 0],
	7  : [2, 0],
	8  : [4, 0],
	9  : [1, 0],
	10 : [3, 0],
	11 : [2, 0],
	12 : [4, 0],
	13 : [1, 0],
	14 : [3, 0],
	15 : [2, 0],
	16 : [4, 0],
	17 : [1, 0],
	18 : [3, 0],
	19 : [2, 0],
	20 : [4, 0],
	21 : [1, 0],
	22 : [3, 0],
	23 : [2, 0],
	24 : [4, 0],
	25 : [1, 0],
	26 : [3, 0],
	27 : [2, 0],
	28 : [4, 0],
	29 : [1, 0],
	30 : [3, 0],
	31 : [2, 0],
	32 : [4, 0],
	33 : [1, 0],
	34 : [3, 0],
	35 : [2, 0],
	36 : [4, 0],
	37 : [1, 0],
	38 : [3, 0],
	39 : [2, 0],
	40 : [4, 0],
	41 : [1, 0],
	42 : [3, 0],
	43 : [2, 0],
	44 : [4, 0],
	45 : [1, 0],
	46 : [3, 0],
	47 : [2, 0],
	48 : [4, 0],
	49 : [1, 0],
	50 : [3, 0],
	51 : [2, 0],
	52 : [4, 0],
	53 : [1, 0],
	54 : [3, 0],
	55 : [2, 0],
	56 : [4, 0],
	57 : [1, 0],
	58 : [3, 0],
	59 : [2, 0],
	60 : [4, 0],
	61 : [1, 0],
	62 : [3, 0],
	63 : [2, 0],
	64 : [4, 0],

	65 : [1, 0],
	66 : [3, 0],
	67 : [2, 0],
	68 : [4, 0],
	69 : [1, 0],
	70 : [3, 0],
	71 : [2, 0],
	72 : [4, 0],
	73 : [1, 0],
	74 : [3, 0],
	75 : [2, 0],
	76 : [4, 0],
	77 : [1, 0],
	78 : [3, 0],
	79 : [2, 0],
	80 : [4, 0],
	81 : [1, 0],
	82 : [3, 0],
	83 : [2, 0],
	84 : [4, 0],
	85 : [1, 0],
	86 : [3, 0],
	87 : [2, 0],
	88 : [4, 0],
	89 : [1, 0],
	90 : [3, 0],
	91 : [2, 0],
	92 : [4, 0],
	93 : [1, 0],
	94 : [3, 0],
	95 : [2, 0],
	96 : [4, 0],
	97 : [1, 0],
	98 : [3, 0],
	99 : [2, 0],
	100 : [4, 0],
	101 : [1, 0],
	102 : [3, 0],
	103 : [2, 0],
	104 : [4, 0],
	105 : [1, 0],
	106 : [3, 0],
	107 : [2, 0],
	108 : [4, 0],
	109 : [1, 0],
	110 : [3, 0],
	111 : [2, 0],
	112 : [4, 0],
	113 : [1, 0],
	114 : [3, 0],
	115 : [2, 0],
	116 : [4, 0],
	117 : [1, 0],
	118 : [3, 0],
	119 : [2, 0],
	120 : [4, 0],
	121 : [1, 0],
	122 : [3, 0],
	123 : [2, 0],
	124 : [4, 0],
	125 : [1, 0],
	126 : [3, 0],
	127 : [2, 0],
	128 : [4, 0],
}

OBDT_MAP = {
	# Flat  9 -> K C
	119: 0 ,
	115: 1,
	96: 2,
	29: 3,
	32: 4,
	19: 5,
	34: 6,
	22: 7,
	142: 8,
	105: 9,
	137: 10,
	120: 11,
	124: 12,
	152: 13,
	149: 14,
	165: 15,
	# Flat 10 -> K D
	40: 16,
	62: 17,
	73: 18,
	153: 19,
	148: 20,
	155: 21,
	156: 22,
	157: 23,
	159: 24,
	166: 25,
	61: 26,
	82: 27,
	77: 28,
	68: 29,
	146: 30,
	147: 31,
	# Flat 11 -> K E
	33: 32,
	27: 33,
	10: 34,
	231: 35,
	232: 36,
	109: 37,
	213: 38,
	239: 39,
	226: 40,
	223: 41,
	7: 42,
	227: 43,
	2: 44,
	8: 45,
	114: 46,
	113: 47,
	# Flat 12 -> K F
	229: 48,
	139: 49,
	141: 50,
	143: 51,
	161: 52,
	59: 53,
	63: 54,
	65: 55,
	58: 56,
	60: 57,
	175: 58,
	171: 59,
	177: 60,
	181: 61,
	185: 62,
	184: 63,
	# Flat 13 -> K I
	6: 64,
	47: 65,
	207: 66,
	54: 67,
	208: 68,
	205: 69,
	206: 70,
	212: 71,
	225: 72,
	18: 73,
	101: 74,
	12: 75,
	110: 76,
	111: 77,
	126: 78,
	125: 79,
	# Flat 14 -> K J
	112: 80,
	188: 81,
	187: 82,
	154: 83,
	170: 84,
	179: 85,
	180: 86,
	182: 87,
	183: 88,
	199: 89,
	195: 90,
	193: 91,
	189: 92,
	191: 93,
	203: 94,
	190: 95,
	# Flat 15 -> K A
	222: 96,
	45: 97,
	49: 98,
	52: 99,
	43: 100,
	209: 101,
	53: 102,
	51: 103,
	215: 104,
	235: 105,
	238: 106,
	28: 107,
	30: 108,
	31: 109,
	20: 110,
	17: 111,
	# Flat 16 -> K B
	14: 112,
	16: 113,
	108: 114,
	13: 115,
	221: 116,
	89: 117,
	90: 118,
	57: 119,
	88: 120,
	198: 121,
	42: 122,
	38: 123,
	37: 124 ,
	39: 125 ,
	186: 126 ,
	0: 127,
	#240 : 1,
	#172 : 1,
}