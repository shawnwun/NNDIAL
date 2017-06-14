######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

import os
import sys
from configTemplate import *
import os.path
import operator
import numpy as np
import re

seed = [str(i) for i in range(1,6)]

dataset     = 'CamRest676'
learn_mode  = 'encdec'
h           = '50'
b           = 'summary'

# decoder trick
token_reward, repeat_penalty = 'False','inf'                               
shuffle, lengthen = 'static', '1'

policy = ['normal','attention']
decoder= ['lstm_lm','lstm_cond','lstm_mix']
lrs    = ['0.008','0.01','0.012']
regs   = ['0.0','0.0001','0.00001','0.000001']
snaps  = ['False','True']

print '%40s\tsuc\tbleu' % ''
for p in policy:
    print '-'*80
    for l in decoder:
        for sn in snaps:
            ids = []
            metrics = np.array([[0.0 for x in range(
                        len(seed)*len(lrs)*len(regs))] for i in range(4)])
            i = -1
            for lr in lrs:
                for reg in regs:
                    for j in range(len(seed)):
 
                        i+=1
                        # identifier
                        s = seed[j]
                        identifier = '.'.join([p,l,sn,lr,reg,'s'+s])
                        # files
                        log     = 'log/batch/'+identifier+'.log-valid'
                        ids.append([lr,reg,seed[j]])

                        try:
                            # read log file
                            fout    = file(log)
                            lines   = fout.readlines()[-22:]
                            # generator performance
                            vmc     = re.findall(r'[0-9\.]+',lines[0])[0]
                            suc     = re.findall(r'[0-9\.]+',lines[1])[0]
                            bleu1,bleu2 = re.findall(r'[0-9\.]+',lines[2])
                            metrics[0,i] = float(vmc)
                            metrics[1,i] = float(suc)
                            metrics[2,i] = float(bleu1)
                            metrics[3,i] = float(suc)+float(bleu1)*100*0.5
                            fout.close()
                        except: pass
            choice = np.argmax(metrics[3,:])
            lr,reg,s = ids[choice]
            
            #print '%40s\t' % ' '.join([p,l,sn,lr,reg,s]),
            identifier = '.'.join([p,l,sn,lr,reg,'s'+s])
            log = 'log/batch/'+identifier+'.log'
            metric = [0. for i in range(4)]
            try:
                # read log file
                fout    = file(log)
                lines   = fout.readlines()[-22:]
                # generator performance
                vmc     = re.findall(r'[0-9\.]+',lines[0])[0]
                suc     = re.findall(r'[0-9\.]+',lines[1])[0]
                bleu1,bleu2 = re.findall(r'[0-9\.]+',lines[2])
                metric[0] = float(vmc)
                metric[1] = float(suc)
                metric[2] = float(bleu1)
                metric[3] = float(suc)+float(bleu1)*100*0.5
                fout.close()
            except: pass
            for i in range(len(metric)):
                if i==2 or i==3:
                    print '%.4f\t' % metric[i],
                else:
                    print '%2.2f%%\t' % metric[i],
            print


