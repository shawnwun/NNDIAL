######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import sys
import os

from utils.commandparser import NNSDSOptParser
from nn.NNDialogue import NNDial

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

if __name__ == '__main__':
    
    args = NNSDSOptParser()
    config = args.config

    model = NNDial(config,args)
    if args.mode=='train' or args.mode=='adjust': 
        model.trainNet()    
    elif args.mode=='test' or args.mode=='valid':
        model.testNet()
    elif args.mode=='interact':
        while True: model.dialog()
    elif args.mode=='rl':
        model.trainNetRL()


