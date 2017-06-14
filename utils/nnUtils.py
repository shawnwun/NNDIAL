######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import random


theano_rng = RandomStreams(np.random.RandomState(1234).randint(2**30))
np.random.seed(1234)
random.seed(1234)

theano.config.floatX='float32'
def initial_weights(*argv):
	"""
	return np.asarray(
		np.random.uniform(
			low  = -np.sqrt(6. / sum(argv)),
			high =  np.sqrt(6. / sum(argv)),
			size =  argv
		),
		dtype=theano.config.floatX
	)
	"""
	return 0.0001 * 2 * (np.random.rand(*argv) - 0.5)

def create_shared(array, dtype=theano.config.floatX, name=None):
	return theano.shared(
			value = np.asarray(
				array,
				dtype = dtype
			),
			name = name
		)

def vector_softmax(vec):
	return T.nnet.softmax(vec.reshape((1,vec.shape[0])))[0]


