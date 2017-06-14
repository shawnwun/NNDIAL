######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

import numpy as np
import math

def softmax(w):
    maxW = np.max(w)
    e = np.exp(w-maxW)
    dist = e/np.sum(e)
    return dist

def sigmoid(w):
    e = np.exp(-w)
    acti = 1/(1+e)
    return acti

def tanh(w):
    return np.tanh(w)


