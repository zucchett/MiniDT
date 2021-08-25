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
        
        # Hits
        self.hmaskHEAD         = 0x7
        self.hmaskHEADv1       = 0x3
        self.hmaskFPGA         = 0x7
        self.hmaskFPGAv1       = 0xF
        self.hmaskTDC_CHANNEL  = 0x1FF
        self.hmaskORBIT_CNT    = 0xFFFFFFFF
        self.hmaskBX_COUNTER   = 0xFFF
        self.hmaskTDC_MEAS     = 0x1F
        
        # Trigger
        self.tmaskHEAD         = 0xC000000000000000
        self.tmaskSL           = 0x3000000000000000
        self.tmaskMCELL        = 0x0E00000000000000
        self.tmaskTAGORB       = 0x01FFFFFFFE000000
        self.tmaskTAGBX        = 0x0000000001FFE000
        self.tmaskBX           = 0x0000000000001FFE
        self.tmaskQUAL         = 0x0000000000000001

        # Equations
        self.emaskHEAD         = 0x7
        self.emaskFPGA         = 0x7
        self.emaskTRIG_ORBIT   = 0xFFFFFFFF
        self.emaskORBIT_CNT    = 0xFFFFFFFF
        self.emaskBX_COUNTER   = 0xFFF
        self.emaskBX0          = 0xFFF
        self.emaskEQ_LABEL     = 0x1F
        self.emaskMACROCELL    = 0xF
        self.emaskTDC_MEAS     = 0x1F
        self.emaskTDC0         = 0x1F
        
        # Parameters
        self.pmaskHEAD         = 0x7
        self.pmaskFPGA         = 0x7
        self.pmaskMACROCELL    = 0xF
        self.pmaskORBIT_CNT    = 0xFFFFFFFF
        self.pmaskVALUE        = 0xFFFF
        self.pmaskVALUEv1      = 0xFFFFFFFF
        
        
        ### SHIFTS
        
        # Hits
        self.hfirstHEAD        = 61
        self.hfirstHEADv1      = 62
        self.hfirstFPGA        = 58
        self.hfirstFPGAv1      = 58
        self.hfirstTDC_CHANNEL = 49
        self.hfirstORBIT_CNT   = 17
        self.hfirstBX_COUNTER  = 5
        self.hfirstTDC_MEAS    = 0
        
        # Trigger
        self.tfirstHEAD        = 62
        self.tfirstSL          = 60
        self.tfirstMCELL       = 57
        self.tfirstTAGORB      = 25
        self.tfirstTAGBX       = 13
        self.tfirstBX          = 1
        self.tfirstQUAL        = 0
        
        # Equations
        self.efirstHEAD        = 61
        self.efirstFPGA        = 58
        self.efirstTRIG_ORBIT  = 22
        self.efirstORBIT_CNT   = 24
        self.efirstBX_COUNTER  = 12
        self.efirstBX0         = 10
        self.efirstEQ_LABEL    = 0
        self.efirstMACROCELL   = 54
        self.efirstTDC_MEAS    = 7
        self.efirstTDC0        = 5

        # Parameters
        self.pfirstHEAD        = 61
        self.pfirstFPGA        = 58
        self.pfirstMACROCELL   = 54
        self.pfirstORBIT_CNT   = 22
        self.pfirstORBIT_CNTv2 = 24
        self.pfirstVALUE       = 0
        

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
        #hmaskTDC_MEAS     = 0x1F
        #hmaskBX_COUNTER   = 0xFFF
        #hmaskORBIT_CNT    = 0xFFFFFFFF
        #hmaskTDC_CHANNEL  = 0x1FF
        #hmaskFPGA         = 0xF
        #hmaskHEAD         = 0x3

        #hfirstTDC_MEAS    = 0
        #hfirstBX_COUNTER  = 5
        #hfirstORBIT_CNT   = 17
        #hfirstTDC_CHANNEL = 49
        #hfirstFPGA        = 58
        #hfirstHEAD        = 62
        
        TDC_MEAS     =      int(( word >> self.hfirstTDC_MEAS    ) & self.hmaskTDC_MEAS   )
        BX_COUNTER   =      int(( word >> self.hfirstBX_COUNTER  ) & self.hmaskBX_COUNTER )
        ORBIT_CNT    =      int(( word >> self.hfirstORBIT_CNT   ) & self.hmaskORBIT_CNT  )
        TDC_CHANNEL  =      int(( word >> self.hfirstTDC_CHANNEL ) & self.hmaskTDC_CHANNEL)
        FPGA         =      int(( word >> self.hfirstFPGAv1      ) & self.hmaskFPGAv1     )
        HEAD         =      int(( word >> self.hfirstHEADv1      ) & self.hmaskHEADv1     )
        
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
        #hmaskTDC_MEAS     = 0x1F
        #hmaskBX_COUNTER   = 0xFFF
        #hmaskORBIT_CNT    = 0xFFFFFFFF
        #hmaskTDC_CHANNEL  = 0x1FF
        #hmaskFPGA         = 0x7
        #hmaskHEAD         = 0x7

        #hfirstTDC_MEAS    = 0
        #hfirstBX_COUNTER  = 5
        #hfirstORBIT_CNT   = 17
        #hfirstTDC_CHANNEL = 49
        #hfirstFPGA        = 58
        #hfirstHEAD        = 61
        
        TDC_MEAS     =      int(( word >> self.hfirstTDC_MEAS    ) & self.hmaskTDC_MEAS   )
        BX_COUNTER   =      int(( word >> self.hfirstBX_COUNTER  ) & self.hmaskBX_COUNTER )
        ORBIT_CNT    =      int(( word >> self.hfirstORBIT_CNT   ) & self.hmaskORBIT_CNT  )
        TDC_CHANNEL  =      int(( word >> self.hfirstTDC_CHANNEL ) & self.hmaskTDC_CHANNEL)
        FPGA         =      int(( word >> self.hfirstFPGA        ) & self.hmaskFPGA       )
        HEAD         =      int(( word >> self.hfirstHEAD        ) & self.hmaskHEAD       )

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
        #tmaskQUAL    = 0x0000000000000001
        #tmaskBX      = 0x0000000000001FFE
        #tmaskTAGBX   = 0x0000000001FFE000
        #tmaskTAGORB  = 0x01FFFFFFFE000000
        #tmaskMCELL   = 0x0E00000000000000
        #tmaskSL      = 0x3000000000000000
        #tmaskHEAD    = 0xC000000000000000

        #tfirstQUAL   = 0
        #tfirstBX     = 1
        #tfirstTAGBX  = 13
        #tfirstTAGORB = 25
        #tfirstMCELL  = 57
        #tfirstSL     = 60
        #tfirstHEAD   = 62
        
        storedTrigHead     = int(( word & self.tmaskHEAD   ) >> self.tfirstHEAD  )
        storedTrigMiniCh   = int(( word & self.tmaskSL     ) >> self.tfirstSL    )
        storedTrigMCell    = int(( word & self.tmaskMCELL  ) >> self.tfirstMCELL )
        storedTrigTagOrbit = int(( word & self.tmaskTAGORB ) >> self.tfirstTAGORB)
        storedTrigTagBX    = int(( word & self.tmaskTAGBX  ) >> self.tfirstTAGBX )
        storedTrigBX       = int(( word & self.tmaskBX     ) >> self.tfirstBX    )
        storedTrigQual     = int(( word & self.tmaskQUAL   ) >> self.tfirstQUAL  )
        
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
        #emaskEQ_LABEL     = 0x1F
        #emaskTDC_MEAS     = 0x1F
        #emaskBX_COUNTER   = 0xFFF
        #emaskORBIT_CNT    = 0xFFFFFFFF
        #emaskFPGA         = 0x7
        #emaskHEAD         = 0x7

        #efirstEQ_LABEL    = 0
        #efirstTDC_MEAS    = 7
        #efirstBX_COUNTER  = 12
        #efirstORBIT_CNT   = 24
        #efirstFPGA        = 58
        #efirstHEAD        = 61

        TDC_MEAS     =      int(( word >> self.efirstTDC_MEAS    ) & self.emaskTDC_MEAS   )
        BX_COUNTER   =      int(( word >> self.efirstBX_COUNTER  ) & self.emaskBX_COUNTER )
        ORBIT_CNT    =      int(( word >> self.efirstORBIT_CNT   ) & self.emaskORBIT_CNT  )
        TDC_CHANNEL  =      int(( word >> self.efirstTDC_MEAS    ) & self.emaskTDC_MEAS   )
        FPGA         =      int(( word >> self.efirstFPGA        ) & self.emaskFPGA       )
        HEAD         =      int(( word >> self.efirstHEAD        ) & self.emaskHEAD       )

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
        #emaskMACROCELL    = 0xF
        #emaskTDC0         = 0x1F
        #emaskBX0          = 0xFFF
        #emaskEQ_LABEL     = 0x1F
        #emaskTRIG_ORBIT    = 0xFFFFFFFF
        #emaskFPGA         = 0x7
        #emaskHEAD         = 0x7

        #efirstMACROCELL   = 54
        #efirstTDC0        = 5
        #efirstBX0         = 10
        #efirstEQ_LABEL    = 0
        #efirstTRIG_ORBIT  = 22
        #efirstFPGA        = 58
        #efirstHEAD        = 61

        MACROCELL    =      int(( word >> self.efirstMACROCELL   ) & self.emaskMACROCELL  )
        TDC0         =      int(( word >> self.efirstTDC0        ) & self.emaskTDC0       )
        BX0          =      int(( word >> self.efirstBX0         ) & self.emaskBX0        )
        EQ_LABEL     =      int(( word >> self.efirstEQ_LABEL    ) & self.emaskEQ_LABEL   )
        TRIG_ORBIT   =      int(( word >> self.efirstTRIG_ORBIT  ) & self.emaskTRIG_ORBIT )
        FPGA         =      int(( word >> self.efirstFPGA        ) & self.emaskFPGA       )
        HEAD         =      int(( word >> self.efirstHEAD        ) & self.emaskHEAD       )
        
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
        #pmaskHEAD         = 0x7
        #pmaskFPGA         = 0x7
        #pmaskVALUE        = 0xFFFFFFFF

        #pfirstHEAD        = 61
        #pfirstFPGA        = 58
        #pfirstVALUE       = 0
        
        PARAMINT     =      int(( word >> self.pfirstVALUE    ) & self.pmaskVALUEv1   )
        FPGA         =      int(( word >> self.pfirstFPGA     ) & self.pmaskFPGA    )
        HEAD         =      int(( word >> self.pfirstHEAD     ) & self.pmaskHEAD    )

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
        #pmaskHEAD         = 0x7
        #pmaskFPGA         = 0x7
        #pmaskORBIT_CNT    = 0xFFFFFFFF
        #pmaskVALUE        = 0xFFFF

        #pfirstHEAD        = 61
        #pfirstFPGA        = 58
        #pfirstORBIT_CNT   = 24
        #pfirstVALUE       = 0
        
        VALUEINT     =      int(( word >> self.pfirstVALUE      ) & self.pmaskVALUE     )
        ORBIT_CNT    =      int(( word >> self.pfirstORBIT_CNTv2) & self.pmaskORBIT_CNT )
        FPGA         =      int(( word >> self.pfirstFPGA       ) & self.pmaskFPGA      )
        HEAD         =      int(( word >> self.pfirstHEAD       ) & self.pmaskHEAD      )
        
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
        #pmaskHEAD         = 0x7
        #pmaskFPGA         = 0x7
        #pmaskMACROCELL    = 0xF
        #pmaskORBIT_CNT    = 0xFFFFFFFF
        #pmaskVALUE        = 0xFFFF

        #pfirstHEAD        = 61
        #pfirstFPGA        = 58
        #pfirstMACROCELL   = 54
        #pfirstORBIT_CNT   = 22
        #pfirstVALUE       = 0
        
        FPVALUE      =      int(( word >> self.pfirstVALUE    ) & self.pmaskVALUE     )
        ORBIT_CNT    =      int(( word >> self.pfirstORBIT_CNT) & self.pmaskORBIT_CNT )
        MACROCELL    =      int(( word >> self.pfirstMACROCELL) & self.pmaskMACROCELL )
        FPGA         =      int(( word >> self.pfirstFPGA     ) & self.pmaskFPGA      )
        HEAD         =      int(( word >> self.pfirstHEAD     ) & self.pmaskHEAD      )
        
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
    