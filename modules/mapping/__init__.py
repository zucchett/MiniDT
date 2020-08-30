import pandas as pd
import numpy as np

import config

class Mapping:
    '''Maps channel numbering from the KCU to detector-related quantities'''

    #def __init__(self):
        

    '''Adds SL, LAYER, WIRE_NUM, WIRE_POS columns to dataframe according to the double Virtex7 setup'''
    def virtex7(self, df):
        df.loc[(df['FPGA'] == 0) & (df['TDC_CHANNEL'] <= config.NCHANNELS), 'SL'] = 0
        df.loc[(df['FPGA'] == 0) & (df['TDC_CHANNEL'] > config.NCHANNELS) & (df['TDC_CHANNEL'] <= 2*config.NCHANNELS), 'SL'] = 1
        df.loc[(df['FPGA'] == 1) & (df['TDC_CHANNEL'] <= config.NCHANNELS), 'SL'] = 2
        df.loc[(df['FPGA'] == 1) & (df['TDC_CHANNEL'] > config.NCHANNELS) & (df['TDC_CHANNEL'] <= 2*config.NCHANNELS), 'SL'] = 3

        df['TDC_CHANNEL_NORM'] = ( df['TDC_CHANNEL'] - config.NCHANNELS*(df['SL']%2) ).astype(np.uint8) # TDC_CHANNEL from 0 to 127 -> TDC_CHANNEL_NORM from 0 to 63

        df.loc[df['TDC_CHANNEL'] % 4 == 1, 'LAYER'] = 1
        df.loc[df['TDC_CHANNEL'] % 4 == 2, 'LAYER'] = 3
        df.loc[df['TDC_CHANNEL'] % 4 == 3, 'LAYER'] = 2
        df.loc[df['TDC_CHANNEL'] % 4 == 0, 'LAYER'] = 4

        df.loc[df['TDC_CHANNEL'] % 4 == 1, 'X_POSSHIFT'] = config.posshift_x[0]
        df.loc[df['TDC_CHANNEL'] % 4 == 2, 'X_POSSHIFT'] = config.posshift_x[1]
        df.loc[df['TDC_CHANNEL'] % 4 == 3, 'X_POSSHIFT'] = config.posshift_x[2]
        df.loc[df['TDC_CHANNEL'] % 4 == 0, 'X_POSSHIFT'] = config.posshift_x[3]

        df.loc[df['TDC_CHANNEL'] % 4 == 1, 'Z_POS'] = config.posshift_z[0]
        df.loc[df['TDC_CHANNEL'] % 4 == 2, 'Z_POS'] = config.posshift_z[1]
        df.loc[df['TDC_CHANNEL'] % 4 == 3, 'Z_POS'] = config.posshift_z[2]
        df.loc[df['TDC_CHANNEL'] % 4 == 0, 'Z_POS'] = config.posshift_z[3]


        
        df['WIRE_NUM'] = ( (df['TDC_CHANNEL_NORM'] - 1) / 4 + 1 ).astype(np.uint8)
        df['WIRE_POS'] = (df['WIRE_NUM'] - 1)*config.XCELL + df['X_POSSHIFT']

        df = df.astype({'SL' : 'int8', 'LAYER' : 'int8'})
        return df


    '''Adds SL, LAYER, WIRE_NUM, WIRE_POS columns to dataframe according to the double Virtex7 setup using direct channel mapping and lambda functions'''
    def virtex7lambda(self, df):
        df['SL'] = df.apply(lambda x: x['FPGA']*2 + int(x['TDC_CHANNEL'] > config.NCHANNELS), axis=1)
        df['LAYER'] = df['TDC_CHANNEL'].map(config.VIRTEX7_LAYER)
        df['WIRE_NUM'] = df['TDC_CHANNEL'].map(config.VIRTEX7_WIRE)
        
        df['X_POSSHIFT'] = df['LAYER'].apply(lambda x: config.posshift_x[int(x)-1])
        df['Z_POS'] = df['LAYER'].apply(lambda x: config.posshift_z[int(x)-1])

        df['WIRE_POS'] = (df['WIRE_NUM'] - 1)*config.XCELL + df['X_POSSHIFT']

        df = df.astype({'SL' : 'int8', 'LAYER' : 'int8'})
        return df


    def addXleftright(self, df):
        df['X_LEFT']  = df['WIRE_POS'] - np.maximum(df['TDRIFT'], 0)*config.VDRIFT
        df['X_RIGHT'] = df['WIRE_POS'] + np.maximum(df['TDRIFT'], 0)*config.VDRIFT