import pandas as pd
import numpy as np
import struct
#import config

class Unpacker:
    '''Maps channel numbering from the KCU to detector-related quantities'''

    def __init__(self):
    
        self.word_size = 8 # one 64-bit word
        self.num_words = 128 + 1 # 1 DMA data transfer = 1 kB = 1024 B = 128 words (hits)
        self.num_words_transfer = 1024

    def unpack(self, inputfile, maxwords=-1, skipFlush=False):
        dt = []
        word_count = 0
        while (maxwords < 0 or word_count - (self.num_words_transfer if skipFlush else 0) < maxwords):
            word = inputfile.read(self.num_words*self.word_size)
            
            if word and len(word) == self.num_words*self.word_size:
                d = self.unpacker_v3(word)
                if not (skipFlush and word_count < self.num_words_transfer): dt += d
                word_count += 1

            else: break
        return dt


    def unpacker_fht(self, hit):
        
        rows = []
        
        for i in range(0, self.num_words*self.word_size, self.word_size):
            
            buffer = struct.unpack('<Q', hit[i:i+self.word_size])[0]
            head = (buffer >> 62) & 0x3
            
            if head == 1 or head == 2:
                rows.append(self.hit_unpacker_v1(buffer))

            elif head == 3:
                rows.append(self.trigger_unpacker(buffer))
            
        return rows


    def unpacker_v2(self, hit):
        
        rows = []
        
        for i in range(0, self.num_words*self.word_size, self.word_size):
            
            buffer = struct.unpack('<Q', hit[i:i+self.word_size])[0]
            head = (buffer >> 61) & 0x7
            
            if head == 2:
                rows.append(self.hit_unpacker_v2(buffer))

            elif head == 0:
                rows.append(self.eq_unpacker(buffer))

            elif head == 4 or head == 5:
                rows.append(self.param_unpacker_v2(buffer))
            
        return rows
        
    
    def unpacker_v3(self, hit):
        
        rows = []
        
        for i in range(0, self.num_words*self.word_size, self.word_size):
            
            buffer = struct.unpack('<Q', hit[i:i+self.word_size])[0]
            head = (buffer >> 61) & 0x7
            
            if head == 2:
                rows.append(self.hit_unpacker_v2(buffer))
            elif head == 0:
                rows.append(self.eq_unpacker_v3(buffer))
            elif head == 4 or head == 5:
                rows.append(self.param_unpacker_v3(buffer))
            
        return rows


    def hit_unpacker_v1(self, word):
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
        TDC_CHANNEL  =      int(( word >> hfirstTDC_CHANNEL ) & hmaskTDC_CHANNEL)
        FPGA         =      int(( word >> hfirstFPGA        ) & hmaskFPGA       )
        HEAD         =      int(( word >> hfirstHEAD        ) & hmaskHEAD       )
        
        #if((TDC_CHANNEL!=137) and (TDC_CHANNEL!=138)):
        #    TDC_MEAS -= 1

        unpacked  = {
            'HEAD': HEAD,
            'FPGA': FPGA,
            'TDC_CHANNEL': TDC_CHANNEL,
            'ORBIT_CNT': ORBIT_CNT,
            'BX_COUNTER': BX_COUNTER,
            'TDC_MEAS': TDC_MEAS - 1,
        }
        
        return unpacked #Row(**unpacked)


    def hit_unpacker_v2(self, word):
        # hit masks
        hmaskTDC_MEAS     = 0x1F
        hmaskBX_COUNTER   = 0xFFF
        hmaskORBIT_CNT    = 0xFFFFFFFF
        hmaskTDC_CHANNEL  = 0x1FF
        hmaskFPGA         = 0x7
        hmaskHEAD         = 0x7

        hfirstTDC_MEAS    = 0
        hfirstBX_COUNTER  = 5
        hfirstORBIT_CNT   = 17
        hfirstTDC_CHANNEL = 49
        hfirstFPGA        = 58
        hfirstHEAD        = 61
        
        TDC_MEAS     =      int(( word >> hfirstTDC_MEAS    ) & hmaskTDC_MEAS   )
        BX_COUNTER   =      int(( word >> hfirstBX_COUNTER  ) & hmaskBX_COUNTER )
        ORBIT_CNT    =      int(( word >> hfirstORBIT_CNT   ) & hmaskORBIT_CNT  )
        TDC_CHANNEL  =      int(( word >> hfirstTDC_CHANNEL ) & hmaskTDC_CHANNEL)
        FPGA         =      int(( word >> hfirstFPGA        ) & hmaskFPGA       )
        HEAD         =      int(( word >> hfirstHEAD        ) & hmaskHEAD       )

        unpacked  = {
            'HEAD': HEAD,
            'FPGA': FPGA,
            'TDC_CHANNEL': TDC_CHANNEL,
            'ORBIT_CNT': ORBIT_CNT,
            'BX_COUNTER': BX_COUNTER,
            'TDC_MEAS': TDC_MEAS - 1,
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


    def eq_unpacker(self, word):
        # EQ-hits masks
        emaskEQ_LABEL     = 0x1F
        emaskTDC_MEAS     = 0x1F
        emaskBX_COUNTER   = 0xFFF
        emaskORBIT_CNT    = 0xFFFFFFFF
        emaskFPGA         = 0x7
        emaskHEAD         = 0x7

        efirstEQ_LABEL    = 0
        efirstTDC_MEAS    = 7
        efirstBX_COUNTER  = 12
        efirstORBIT_CNT   = 24
        efirstFPGA        = 58
        efirstHEAD        = 61

        TDC_MEAS     =      int(( word >> efirstTDC_MEAS    ) & emaskTDC_MEAS   )
        BX_COUNTER   =      int(( word >> efirstBX_COUNTER  ) & emaskBX_COUNTER )
        ORBIT_CNT    =      int(( word >> efirstORBIT_CNT   ) & emaskORBIT_CNT  )
        TDC_CHANNEL  =      int(( word >> efirstTDC_MEAS    ) & emaskTDC_MEAS   )
        FPGA         =      int(( word >> efirstFPGA        ) & emaskFPGA       )
        HEAD         =      int(( word >> efirstHEAD        ) & emaskHEAD       )

        unpacked  = {
            'HEAD': HEAD,
            'FPGA': FPGA,
            'TDC_CHANNEL': TDC_CHANNEL,
            'ORBIT_CNT': ORBIT_CNT,
            'BX_COUNTER': BX_COUNTER,
            'TDC_MEAS': TDC_MEAS - 1,
        }
        
        return unpacked #Row(**unpacked)

    def eq_unpacker_v3(self, word):
        # EQ-hits masks
        emaskMACROCELL    = 0xF
        emaskTDC0         = 0x1F
        emaskBX0          = 0xFFF
        emaskEQ_LABEL     = 0x1F
        emaskTRIG_ORBIT    = 0xFFFFFFFF
        emaskFPGA         = 0x7
        emaskHEAD         = 0x7

        efirstMACROCELL   = 54
        efirstTDC0        = 5
        efirstBX0         = 10
        efirstEQ_LABEL    = 0
        efirstTRIG_ORBIT  = 22
        efirstFPGA        = 58
        efirstHEAD        = 61

        MACROCELL    =      int(( word >> efirstMACROCELL   ) & emaskMACROCELL  )
        TDC0         =      int(( word >> efirstTDC0        ) & emaskTDC0       )
        BX0          =      int(( word >> efirstBX0         ) & emaskBX0        )
        EQ_LABEL     =      int(( word >> efirstEQ_LABEL    ) & emaskEQ_LABEL   )
        TRIG_ORBIT   =      int(( word >> efirstTRIG_ORBIT  ) & emaskTRIG_ORBIT )
        FPGA         =      int(( word >> efirstFPGA        ) & emaskFPGA       )
        HEAD         =      int(( word >> efirstHEAD        ) & emaskHEAD       )
        
        unpacked  = {
            'HEAD': HEAD,
            'FPGA': FPGA,
            'TDC_CHANNEL': MACROCELL,
            'ORBIT_CNT': TRIG_ORBIT,
            'BX_COUNTER': BX0,
            'TDC_MEAS': TDC0 - 1,
        }
        
        return unpacked #Row(**unpacked)


    def param_unpacker_v1(self, word):
        # hit masks
        pmaskHEAD         = 0x7
        pmaskFPGA         = 0x7
        pmaskPARAM        = 0xFFFFFFFF

        pfirstHEAD        = 61
        pfirstFPGA        = 58
        pfirstPARAM       = 0
        
        PARAMINT     =      int(( word >> pfirstPARAM    ) & pmaskPARAM   )
        FPGA         =      int(( word >> pfirstFPGA     ) & pmaskFPGA    )
        HEAD         =      int(( word >> pfirstHEAD     ) & pmaskHEAD    )

        PARAM = struct.unpack('!f', struct.pack('!I', PARAMINT))[0] # Equivalent to the C code: float fvalue = *(float*)&value;
        
        unpacked  = {
            'HEAD': HEAD,
            'FPGA': FPGA,
            'TDC_CHANNEL': 0,
            'ORBIT_CNT': 0,
            'BX_COUNTER': 0,
            'TDC_MEAS': VALUE,
        }
        
        return unpacked #Row(**unpacked)



    def param_unpacker_v2(self, word):
        # hit masks
        pmaskHEAD         = 0x7
        pmaskFPGA         = 0x7
        pmaskORBIT_CNT    = 0xFFFFFFFF
        pmaskVALUE        = 0xFFFF

        pfirstHEAD        = 61
        pfirstFPGA        = 58
        pfirstORBIT_CNT   = 24
        pfirstVALUE       = 0
        
        VALUEINT     =      int(( word >> pfirstVALUE    ) & pmaskVALUE     )
        ORBIT_CNT    =      int(( word >> pfirstORBIT_CNT) & pmaskORBIT_CNT )
        FPGA         =      int(( word >> pfirstFPGA     ) & pmaskFPGA      )
        HEAD         =      int(( word >> pfirstHEAD     ) & pmaskHEAD      )
        
        VALUE = struct.unpack('!e', struct.pack('!H', VALUEINT))[0] # Equivalent to the C code: float fvalue = *(float*)&value;
        
        unpacked  = {
            'HEAD': HEAD,
            'FPGA': FPGA,
            'TDC_CHANNEL': 0,
            'ORBIT_CNT': ORBIT_CNT,
            'BX_COUNTER': 0,
            'TDC_MEAS': VALUE,
        }
        
        return unpacked #Row(**unpacked)



    def param_unpacker_v3(self, word):
        # hit masks
        pmaskHEAD         = 0x7
        pmaskFPGA         = 0x7
        pmaskMACROCELL    = 0xF
        pmaskORBIT_CNT    = 0xFFFFFFFF
        pmaskVALUE        = 0xFFFF

        pfirstHEAD        = 61
        pfirstFPGA        = 58
        pfirstMACROCELL   = 54
        pfirstORBIT_CNT   = 22
        pfirstVALUE       = 0
        
        FPVALUE      =      int(( word >> pfirstVALUE    ) & pmaskVALUE     )
        ORBIT_CNT    =      int(( word >> pfirstORBIT_CNT) & pmaskORBIT_CNT )
        MACROCELL    =      int(( word >> pfirstMACROCELL) & pmaskMACROCELL )
        FPGA         =      int(( word >> pfirstFPGA     ) & pmaskFPGA      )
        HEAD         =      int(( word >> pfirstHEAD     ) & pmaskHEAD      )
        
        VALUE = 0.
        if HEAD == 4: VALUE = self.float_16_2(FPVALUE)
        elif HEAD == 5: VALUE = self.float_16_8(FPVALUE)
        
        unpacked  = {
            'HEAD': HEAD,
            'FPGA': FPGA,
            'TDC_CHANNEL': MACROCELL,
            'ORBIT_CNT': ORBIT_CNT,
            'BX_COUNTER': 0,
            'TDC_MEAS': VALUE,
        }
    
        return unpacked #Row(**unpacked)
    
    
    def float_16_2(self, x):
        sign = (x >> 15) & 0x1
        rest = x & 0x7FFF
        if sign > 0: rest = (~rest + 1) & 0x7FFF
        inte = float((rest & 0x4000) >> 14)
        deci = float(rest & 0x3FFF) / (2**14)
        fp = inte + deci
        if sign > 0: fp *= -1
        return fp
    
    def float_16_8(self, x):
        sign = (x >> 15) & 0x1
        rest = x & 0x7FFF
        if sign > 0: rest = (~rest + 1) & 0x7FFF
        inte = float((rest & 0x7F00) >> 8)
        deci = float(rest & 0x00FF) / (2**8)
        fp = inte + deci
        if sign > 0: fp *= -1
        return fp
    