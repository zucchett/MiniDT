import pandas as pd
import numpy as np

import modules.mapping.config

class Mapping:
    '''Maps channel numbering from the KCU to detector-related quantities'''

    #def __init__(self):
        

    '''Adds SL, LAYER, WIRE_NUM, WIRE_POS columns to dataframe according to the double Virtex7 setup'''
    def virtex7(self, df):
        df['SL'] = -1
        df.loc[(df['FPGA'] == 0) & (df['TDC_CHANNEL'] < config.NCHANNELS), 'SL'] = 0
        df.loc[(df['FPGA'] == 0) & (df['TDC_CHANNEL'] >= config.NCHANNELS) & (df['TDC_CHANNEL'] < 2*config.NCHANNELS), 'SL'] = 1
        df.loc[(df['FPGA'] == 1) & (df['TDC_CHANNEL'] < config.NCHANNELS), 'SL'] = 2
        df.loc[(df['FPGA'] == 1) & (df['TDC_CHANNEL'] >= config.NCHANNELS) & (df['TDC_CHANNEL'] < 2*config.NCHANNELS), 'SL'] = 3

        df['TDC_CHANNEL_NORM'] = ( df['TDC_CHANNEL'] - config.NCHANNELS*(df['SL'] % 2) ).astype(np.uint8) # TDC_CHANNEL from 0 to 127 -> TDC_CHANNEL_NORM from 0 to 63
        #df.loc[(df['FPGA'] >= 2), 'TDC_CHANNEL_NORM'] = -1

        #df.loc[df['TDC_CHANNEL'] % 4 == 1, 'LAYER'] = 1
        #df.loc[df['TDC_CHANNEL'] % 4 == 2, 'LAYER'] = 3
        #df.loc[df['TDC_CHANNEL'] % 4 == 3, 'LAYER'] = 2
        #df.loc[df['TDC_CHANNEL'] % 4 == 0, 'LAYER'] = 4

        df.loc[df['TDC_CHANNEL'] % 4 == 1, 'LAYER'] = 2
        df.loc[df['TDC_CHANNEL'] % 4 == 2, 'LAYER'] = 3
        df.loc[df['TDC_CHANNEL'] % 4 == 3, 'LAYER'] = 1
        df.loc[df['TDC_CHANNEL'] % 4 == 0, 'LAYER'] = 4

        df.loc[df['TDC_CHANNEL'] % 4 == 1, 'X_POSSHIFT'] = config.posshift_x[1]
        df.loc[df['TDC_CHANNEL'] % 4 == 2, 'X_POSSHIFT'] = config.posshift_x[2]
        df.loc[df['TDC_CHANNEL'] % 4 == 3, 'X_POSSHIFT'] = config.posshift_x[0]
        df.loc[df['TDC_CHANNEL'] % 4 == 0, 'X_POSSHIFT'] = config.posshift_x[3]

        df.loc[df['TDC_CHANNEL'] % 4 == 1, 'Z_POS'] = config.posshift_z[1]
        df.loc[df['TDC_CHANNEL'] % 4 == 2, 'Z_POS'] = config.posshift_z[2]
        df.loc[df['TDC_CHANNEL'] % 4 == 3, 'Z_POS'] = config.posshift_z[0]
        df.loc[df['TDC_CHANNEL'] % 4 == 0, 'Z_POS'] = config.posshift_z[3]
        
        df['WIRE_NUM'] = (df['TDC_CHANNEL_NORM'] / 4 + 1 ).astype(np.uint8)
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


    '''Adds SL, LAYER, WIRE_NUM, WIRE_POS columns to dataframe according to the hybrid OBDT/Virtex7 setup'''
    def virtex7obdt(self, df):
        df.loc[df['FPGA'] == 0, 'TDC_CHANNEL'] = df.loc[df['FPGA'] == 0, 'TDC_CHANNEL'].map(config.OBDT_MAP, na_action=-1)
        df['TDC_CHANNEL'] = df['TDC_CHANNEL'] + 1
        df = self.virtex7(df)
        return df


    def addXleftright(self, df):
        df['X_LEFT']  = df['WIRE_POS'] - np.maximum(df['TDRIFT'], 0)*config.VDRIFT
        df['X_RIGHT'] = df['WIRE_POS'] + np.maximum(df['TDRIFT'], 0)*config.VDRIFT


    '''Get single values'''
    def getZlayer(self, layer): # Input Layer is between [1, 4]
        #l = config.layer_z[layer - 1]
        #return config.posshift_z[l - 1] #if layer < len(config.posshift_z) else -99.
        return config.posshift_z[layer - 1]

    def getWireNumber(self, x, layer): # Input Layer is between [1, 4]
        #l = config.layer_z[layer - 1]
        return int(round((x - config.posshift_x[layer - 1]) / config.XCELL) + 1)
    
    def getWirePosition(self, layer, wire_num): # Input Layer is between [1, 4]
        #l = config.layer_z[layer - 1]
        return (wire_num - 1) * config.XCELL + config.posshift_x[ layer - 1 ]

    def getChannelNorm(self, layer, wire_num):
        return (wire_num - 1) * 4 + config.layer_z[ (layer - 1) % 4 ] - 1

    def getChannel(self, sl, layer, wire_num):
        if sl % 2 == 0:
            return self.getChannelNorm(layer, wire_num)
        elif sl % 2 == 1:
            return self.getChannelNorm(layer, wire_num) + config.NCHANNELS
        else:
            return -1

    def getLayerZ(self, z):
        for i in range(3):
            if config.posshift_z[i] - config.ZCELL/2. <= z < config.posshift_z[i + 1] - config.ZCELL/2.: return i + 1
        if config.posshift_z[-1] - config.ZCELL/2. <= z <= config.posshift_z[-1] + config.ZCELL/2.: return 3 + 1
        return -1

    def getFPGA(self, sl):
        if 0 <= sl <= 1: return 0
        elif 2 <= sl <= 3: return 1
        return -1
    #

    def getSL(self, fpga, channel): # Input channel starts from 0
        if fpga == 0 and channel < config.NCHANNELS: return 0
        elif fpga == 0 and channel >= config.NCHANNELS and channel < 2*config.NCHANNELS: return 1
        elif fpga == 1 and channel < config.NCHANNELS: return 2
        elif fpga == 1 and channel >= config.NCHANNELS and channel < 2*config.NCHANNELS: return 3
        return -1

    def getLayer(self, channel): # Input channel starts from 0
        return config.layer_z[ channel % 4 ]
    
    def getXposshift(self, channel): # Input channel starts from 0
        return config.posshift_x[ config.layer_z[ channel % 4 ] - 1 ]

    def getZpos(self, channel): # Input channel starts from 0
        return config.posshift_z[ config.layer_z[ channel % 4 ] - 1 ]

    def getWireNum(self, channel): # Input channel starts from 0
        return channel // 4 + 1
        
    def addTappinoNum(self, df):
        df['TAPPINO'] = df['WIRE'].map( config.tappino )
        return df
        