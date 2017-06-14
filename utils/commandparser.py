######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

import argparse


def NNSDSOptParser():
    
    parser = argparse.ArgumentParser(\
            description='Default E2E SDS opt parser.')

    parser.add_argument('-mode',  help='modes: train|test|gen|adapt|obj|sds')
    parser.add_argument('-config', help='config file to set.')
    
    return parser.parse_args()

