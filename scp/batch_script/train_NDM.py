######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

import os
import sys
from configTemplate import *
from hpc_submit import *
import os.path
import operator

seed = [str(i) for i in range(1,6)]

dataset     = 'CamRest676'
learn_mode  = 'encdec'
h           = '50'
b           = 'summary'

# decoder trick
token_reward, repeat_penalty, alpha = 'False','inf','1.0'                              
shuffle, lengthen = 'static', '1'


policy = ['normal','attention']
decoder= ['lstm_lm','lstm_cond','lstm_mix']
lrs    = ['0.008','0.01','0.012']
regs   = ['0.0','0.0001','0.00001','0.000001']
snaps  = ['True','False']

for p in policy:
    for l in decoder:
        for lr in lrs:
            for reg in regs:
                for sn in snaps:
                    for s in seed:
                        # identifier
                        identifier = '.'.join([p,l,sn,lr,reg,'s'+s])

                        # tracker
                        trk_model= 'model/CamRest.tracker.model'

                        # files
                        model   = 'model/batch/'+identifier+'.model'
                        config  = 'config/batch/'+identifier+'.cfg'
                        log     = 'log/batch/'+identifier+'.log'

                        scp     = 'scp/batch/'+identifier+'.scp'
                        slurm   = 'scp/batch/slurm-'+identifier+'.darwin'
                        
                        # write config
                        fout = file(config,'w')
                        fout.write( configLearn(s,lr=lr,lr_divide='2',beta=reg)  )
                        fout.write( configFile(dataset,model)   )
                        fout.write( configData(shuffle,lengthen)  )
                        fout.write( configMode('encdec')   )
                        fout.write( configN2N(arc['enc'],arc['dec'],arc['trk'])  )
                        fout.write( configEnc(h)   )
                        fout.write( configTrk(arc['inf'],arc['req'],b,arc['trkenc'])  )
                        fout.write( configPly(p,'100')   )
                        fout.write( configDec(h,l,sn)  )
                        fout.write( configGen(token_reward, repeat_penalty,alpha,topk='1'))
                        fout.close()
                        
                        # write scripts
                        fout = file(scp,'w')
                        fout.write('#!/bin/sh\n')
                        fout.write('#$ -S /bin/bash\n')
                        fout.write('cp '+trk_model+' '+model+'\n')
                        fout.write('rm '+log+'\n')
                        fout.write('rm '+log+'-valid\n')
                        fout.write('THEANO_FLAGS="optimizer=fast_compile" \
                                    python nndial.py -config '+config+' -mode adjust\n')
                        fout.write('python nndial.py -config '+config+\
                                ' -mode valid > '+log+'-valid\n')
                        fout.write('python nndial.py -config '+config+\
                                ' -mode test > '+log+'\n')
                        
                        os.chmod(scp,0755)
                        
                        # TODO: Modify here to run parallel experiments
                        # submit job
                        os.system('qsub '+scp)

