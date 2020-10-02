import pandas as pd
import numpy as np
import struct
#import config

class Unpacker:
    '''Maps channel numbering from the KCU to detector-related quantities'''

    def __init__(self):
    
        self.word_size = 8 # one 64-bit word
        self.num_words = 128 + 1 # 1 DMA data transfer = 1 kB = 1024 B = 128 words (hits)


    def unpack(self, inputfile, maxwords=-1):
        dt = []
        word_count = 0
        while (word_count < 0 or word_count < maxwords):
            word = inputfile.read(self.num_words*self.word_size)
            if word:
                  d = self.unpacker(word)
                  dt += d
                  word_count += 1
                  #print len(dt)
            else: break
        return dt


    def unpacker(self, hit):
        
        rows = []
        
        for i in range(0, self.num_words*self.word_size, self.word_size):
            
            buffer = struct.unpack('<Q', hit[i:i+self.word_size])[0]
            head = (buffer >> 62) & 0x3
            
            if head == 1 or head == 2:
                rows.append(self.hit_unpacker(buffer))

            elif head == 3:
                rows.append(self.trigger_unpacker(buffer))
            
        return rows

    def hit_unpacker(self, word):
        # hit masks
        hmaskTDC_MEAS     = 0x1F
        hmaskBX_COUNTER   = 0xFFF
        hmaskORBIT_CNT    = 0xFFFFFFFF
        hmaskTDC_CHANNEL  = 0x1FF
        hmaskFPGA         = 0xF
        hmaskHEAD         = 0x3

        hfirstTDC_MEAS    = 0
        hfirstBX_COUNTER  = 5
        hfirstORBIT_CNT   = 17
        hfirstTDC_CHANNEL = 49
        hfirstFPGA        = 58
        hfirstHEAD        = 62
        
        TDC_MEAS     =      int(( word >> hfirstTDC_MEAS    ) & hmaskTDC_MEAS   )
        BX_COUNTER   =      int(( word >> hfirstBX_COUNTER  ) & hmaskBX_COUNTER )
        ORBIT_CNT    =      int(( word >> hfirstORBIT_CNT   ) & hmaskORBIT_CNT  )
        TDC_CHANNEL  =  1 + int(( word >> hfirstTDC_CHANNEL ) & hmaskTDC_CHANNEL)
        FPGA         =      int(( word >> hfirstFPGA        ) & hmaskFPGA       )
        HEAD         =      int(( word >> hfirstHEAD        ) & hmaskHEAD       )
        
        if((TDC_CHANNEL!=137) and (TDC_CHANNEL!=138)):
            TDC_MEAS -= 1

        unpacked  = {
            'HEAD': HEAD,
            'FPGA': FPGA,
            'TDC_CHANNEL': TDC_CHANNEL,
            'ORBIT_CNT': ORBIT_CNT,
            'BX_COUNTER': BX_COUNTER,
            'TDC_MEAS': TDC_MEAS,
            'TRG_QUALITY': np.NaN
        }
        
        return unpacked #Row(**unpacked)

    def trigger_unpacker(self, word):
        # Trigger masks
        tmaskQUAL    = 0x0000000000000001
        tmaskBX      = 0x0000000000001FFE
        tmaskTAGBX   = 0x0000000001FFE000
        tmaskTAGORB  = 0x01FFFFFFFE000000
        tmaskMCELL   = 0x0E00000000000000
        tmaskSL      = 0x3000000000000000
        tmaskHEAD    = 0xC000000000000000

        tfirstQUAL   = 0
        tfirstBX     = 1
        tfirstTAGBX  = 13
        tfirstTAGORB = 25
        tfirstMCELL  = 57
        tfirstSL     = 60
        tfirstHEAD   = 62
        
        storedTrigHead     = int(( word & tmaskHEAD   ) >> tfirstHEAD  )
        storedTrigMiniCh   = int(( word & tmaskSL     ) >> tfirstSL    )
        storedTrigMCell    = int(( word & tmaskMCELL  ) >> tfirstMCELL )
        storedTrigTagOrbit = int(( word & tmaskTAGORB ) >> tfirstTAGORB)
        storedTrigTagBX    = int(( word & tmaskTAGBX  ) >> tfirstTAGBX )
        storedTrigBX       = int(( word & tmaskBX     ) >> tfirstBX    )
        storedTrigQual     = int(( word & tmaskQUAL   ) >> tfirstQUAL  )
        
        unpacked = {
            'HEAD': storedTrigHead,
            'FPGA': storedTrigMiniCh,
            'TDC_CHANNEL': storedTrigMCell,
            'ORBIT_CNT': storedTrigTagOrbit,
            'BX_COUNTER': storedTrigTagBX,
            'TDC_MEAS': storedTrigBX,
            'TRG_QUALITY': storedTrigQual
        }
        
        return unpacked #Row(**unpacked)

