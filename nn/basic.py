######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import numpy as np
import theano
import theano.tensor as T
from utils.mathUtil import softmax, sigmoid, tanh

# gradient clipping 
class GradClip(theano.compile.ViewOp):
    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) 
                for g_out in g_outs]

def clip_gradient(x, bound):
    grad_clip = GradClip(-bound, bound)
    try:
        T.opt.register_canonicalize(
            theano.gof.OpRemove(grad_clip), name='grad_clip_%.1f' % (bound))
    except ValueError:
        pass
    return grad_clip(x)


# Base class for all neural network modules
class BaseNNModule(object):

    # default parameters manipulation functions
    def setParams(self,params):
        for i in range(len(self.params)):
            self.params[i].set_value(params[i])

    def getParams(self):
        return [p.get_value() for p in self.params]

    def numOfParams(self):
        return sum([p.get_value().size for p in self.params])



