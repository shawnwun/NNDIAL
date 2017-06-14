######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

from itertools import izip
import theano
import theano.tensor as T
import numpy         as np
import nnUtils       as U


def adadelta(parameters,gradients,rho=np.float32(0.999),eps=np.float32(1e-6)):
	gradients_sq = [ U.create_shared(np.zeros(p.get_value().shape,dtype=np.float32)) for p in parameters ]
	deltas_sq    = [ U.create_shared(np.zeros(p.get_value().shape,dtype=np.float32)) for p in parameters ]

	gradients_sq_new = [ rho*g_sq + (np.float32(1)-rho)*(g**2)      for g_sq,g         in izip(gradients_sq,gradients) ]
	deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]
	deltas_sq_new = [ rho*d_sq + (np.float32(1)-rho)*(d**2)         for d_sq,d         in izip(deltas_sq,deltas) ]

	gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
	deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
	parameters_updates = [ (p,p - d) for p,d in izip(parameters,deltas) ]
	return gradient_sq_updates + deltas_sq_updates + parameters_updates

def momentum(parameters,gradients,mu,eps):
	t = U.create_shared(1)
	m = (1 - 3.0/(t+5) < mu)
	mu = m * (1 - 3.0/(t+5)) + (1-m) * mu
	deltas = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
	delta_nexts = [ mu*delta + eps*grad for delta,grad in zip(deltas,gradients) ]
	delta_updates = [ (delta, delta_next) for delta,delta_next in zip(deltas,delta_nexts) ]
	param_updates = [ (param, param - delta_next) for param,delta_next in zip(parameters,delta_nexts) ]
	return delta_updates + param_updates + [ (t,t + 1) ]


def rmsprop(parameters,gradients,discount=0.95,momentum=0.9,learning_rate=1e-4,epsilon=1e-4):
	#gradients = [ (g < -clip)*(-clip) + (g > clip)*(clip) + (abs(g) <= clip) * g for g in gradients ]
	sq_acc    = [ U.create_shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in parameters ]
	acc       = [ U.create_shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in parameters ]
	delta_acc = [ U.create_shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in parameters ]

	sq_avg = [ discount * sq_a + (1 - discount) * g**2 for sq_a,g in izip(sq_acc,gradients) ]
	avg    = [ discount * a    + (1 - discount) * g    for a,   g in izip(acc,gradients) ]
	scaled_grads = [ g / T.sqrt(sq_a - a**2 + epsilon) for g,a,sq_a in izip(gradients,acc,sq_acc) ]
	deltas = [ momentum * d_a + learning_rate * s_g for d_a,s_g in izip(delta_acc,scaled_grads) ]


	sq_acc_updates = [ (sq_a, sq_aa) for sq_a,sq_aa in izip(sq_acc,sq_avg) ]
	acc_updates    = [ (a,    aa)    for a,   aa    in izip(acc,avg) ]
	delta_updates  = [ (d_a,d) for d_a,d in izip(delta_acc,deltas) ]
	parameters_updates = [ (p, p - d) for p,d in izip(parameters,deltas) ]

	return parameters_updates + acc_updates + sq_acc_updates + delta_updates

