import pandas as pd
import numpy as np

import config

class Mapping:
    '''Maps channel numbering from the KCU to detector-related quantities'''

    #def __init__(self):
        

    '''Adds SL, LAYER, WIRE_NUM, WIRE_POS columns to dataframe according to the double Virtex7 setup'''
    def virtex7(self, hits):
        hits.loc[(hits['FPGA'] == 0) & (hits['TDC_CHANNEL'] <= config.NCHANNELS), 'SL'] = 0
        hits.loc[(hits['FPGA'] == 0) & (hits['TDC_CHANNEL'] > config.NCHANNELS) & (hits['TDC_CHANNEL'] <= 2*config.NCHANNELS), 'SL'] = 1
        hits.loc[(hits['FPGA'] == 1) & (hits['TDC_CHANNEL'] <= config.NCHANNELS), 'SL'] = 2
        hits.loc[(hits['FPGA'] == 1) & (hits['TDC_CHANNEL'] > config.NCHANNELS) & (hits['TDC_CHANNEL'] <= 2*config.NCHANNELS), 'SL'] = 3

        hits.loc[hits['TDC_CHANNEL'] % 4 == 1, 'LAYER'] = 1
        hits.loc[hits['TDC_CHANNEL'] % 4 == 2, 'LAYER'] = 3
        hits.loc[hits['TDC_CHANNEL'] % 4 == 3, 'LAYER'] = 2
        hits.loc[hits['TDC_CHANNEL'] % 4 == 0, 'LAYER'] = 4

        hits.loc[hits['TDC_CHANNEL'] % 4 == 1, 'X_POSSHIFT'] = config.posshift_x[0]
        hits.loc[hits['TDC_CHANNEL'] % 4 == 2, 'X_POSSHIFT'] = config.posshift_x[1]
        hits.loc[hits['TDC_CHANNEL'] % 4 == 3, 'X_POSSHIFT'] = config.posshift_x[2]
        hits.loc[hits['TDC_CHANNEL'] % 4 == 0, 'X_POSSHIFT'] = config.posshift_x[3]

        hits.loc[hits['TDC_CHANNEL'] % 4 == 1, 'Z_POS'] = config.posshift_z[0]
        hits.loc[hits['TDC_CHANNEL'] % 4 == 2, 'Z_POS'] = config.posshift_z[1]
        hits.loc[hits['TDC_CHANNEL'] % 4 == 3, 'Z_POS'] = config.posshift_z[2]
        hits.loc[hits['TDC_CHANNEL'] % 4 == 0, 'Z_POS'] = config.posshift_z[3]


        hits['TDC_CHANNEL_NORM'] = ( hits['TDC_CHANNEL'] - config.NCHANNELS*(hits['SL']%2) ).astype(np.uint8) # TDC_CHANNEL from 0 to 127 -> TDC_CHANNEL_NORM from 0 to 63
        hits['WIRE_NUM'] = ( (hits['TDC_CHANNEL_NORM'] - 1) / 4 + 1 ).astype(np.uint8)
        hits['WIRE_POS'] = (hits['WIRE_NUM'] - 1)*config.XCELL + hits['X_POSSHIFT']


