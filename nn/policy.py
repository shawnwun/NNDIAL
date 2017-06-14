######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import numpy as np
import theano
import theano.tensor as T
from utils.mathUtil import softmax, sigmoid, tanh
import operator
from math import pow
import sys
import theano.gradient as G
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


from basic import *
from encoder import *

def computeBeleifDim(trk, inf, req, bef, iseg, rseg):

    # compute belief vector dimension
    belief_size = 0
    if trk=='rnn' and req==True: # requestable
        if bef=='full' or bef=='summary':
            belief_size += 2*(len(rseg)-1)
            belief_size += 2
        elif bef=='simplified':
            belief_size += len(rseg)-1
            belief_size += 1
    if trk=='rnn' and inf==True: # informable
        if bef=='full':
            belief_size += iseg[-1]
        elif bef=='summary':
            belief_size += 3*(len(iseg)-1)
        elif bef=='simplified':
            belief_size += 2*(len(iseg)-1)
    return belief_size


class Policy(BaseNNModule):
    
    # Policy network takes three inputs and produces a single 
    # system action embedding. Its use is heavily coupled with decoder.
    
    def __init__(self, belief_size, degree_size, ihidden_size, ohidden_size):

        # belief to action parameter
        self.Ws1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (belief_size,ohidden_size)).astype(theano.config.floatX))
        # matching degree to action parameter
        self.Ws2 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (degree_size,ohidden_size)).astype(theano.config.floatX))
        # intent to action parameter
        self.Ws3 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*2,ohidden_size)).astype(theano.config.floatX))
        # all parameters
        self.params = [self.Ws1,    self.Ws2,   self.Ws3    ]

    def encode(self, belief_t, degree_t, intent_t):
        belief_t = T.concatenate(belief_t,axis=0)
        return T.tanh(  T.dot(belief_t,self.Ws1)+
                        T.dot(degree_t,self.Ws2)+
                        T.dot(intent_t,self.Ws3)).dimshuffle('x',0)

    def decide(self, belief_t, degree_t, intent_t):
        # numpy function called during test time
        belief_t = np.concatenate(belief_t,axis=0)
        return np.expand_dims( tanh(
                    np.dot(belief_t,self.Ws1_backup)+
                    np.dot(degree_t,self.Ws2_backup)+
                    np.dot(intent_t,self.Ws3_backup)    ),axis=0)
    def loadConverseParams(self):
        self.Ws1_backup  = self.params[0].get_value()
        self.Ws2_backup  = self.params[1].get_value()
        self.Ws3_backup  = self.params[2].get_value()

 
class LatentPolicy(BaseNNModule):
    
    # Policy network takes three inputs and produces a single 
    # system action embedding. Its use is heavily coupled with decoder.
    
    def __init__(self, latent_size, learn_mode, 
            belief_size, degree_size, ihidden_size, ohidden_size,
            tfEncoder, tbEncoder, sfEncoder, sbEncoder ):

        # latent variable dimension
        self.dl = latent_size
        hidden_size = 100
        # set default sampling mode: posterior, from all actions
        if learn_mode=='rl':self.setSampleMode('prior',5)
        else:               self.setSampleMode('posterior',latent_size)

        # random seed
        self.srng = RandomStreams(seed=234)
        
        # decoder input embedding
        self.Wd1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (latent_size,hidden_size)).astype(theano.config.floatX))
        self.Wd2 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (latent_size,hidden_size)).astype(theano.config.floatX))
        self.bd1 = theano.shared(2.*np.ones(
                (hidden_size)).astype(theano.config.floatX))
        self.Wd3 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (hidden_size*2,ohidden_size)).astype(theano.config.floatX))
        # for state construction
        # belief to state
        self.Ws1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (belief_size,hidden_size)).astype(theano.config.floatX))
        # matching degree to state
        self.Ws2 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (degree_size,hidden_size)).astype(theano.config.floatX))
        # intent to state
        self.Ws3 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*2,hidden_size)).astype(theano.config.floatX))
        # latent policy parameterisation, state -> action
        self.Wp1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (hidden_size,hidden_size)).astype(theano.config.floatX))
        self.Wp2 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (hidden_size,latent_size)).astype(theano.config.floatX))
        self.bp1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (hidden_size)).astype(theano.config.floatX))
        # prior parameters P(z_t|S_t) and P(R_t|z_t)
        self.params = [ self.Wd1,   self.Wd2,   self.bd1, self.Wd3, 
                        self.Ws1,   self.Ws2,   self.Ws3,  
                        self.Wp1,   self.Wp2,   self.bp1  ]
        
        # approximated posterior parameters Q(z_t|S_t,R_t)
        # sentence encoders
        self.sfEncoder, self.sbEncoder = sfEncoder, sbEncoder
        self.tfEncoder, self.tbEncoder = tfEncoder, tbEncoder
        # belief to posterior
        self.Wq1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (belief_size,hidden_size)).astype(theano.config.floatX))
        # matching degree to posterior
        self.Wq2 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (degree_size,hidden_size)).astype(theano.config.floatX))
        # intent to posterior
        self.Wq3 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*2,hidden_size)).astype(theano.config.floatX))
        # response to posterior
        self.Wq4 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*2,hidden_size)).astype(theano.config.floatX))
        # MLP 2nd layer
        self.Wq5 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (hidden_size,latent_size)).astype(theano.config.floatX))
        #posterior parameters Q(z_t|S_t,R_t)
        self.Qparams = [self.Wq1,  self.Wq2,   self.Wq3,   self.Wq4, self.Wq5]
        self.Qparams.extend(self.tfEncoder.params + self.tbEncoder.params +
                            self.sfEncoder.params + self.sbEncoder.params )
        # add posterior also into parameter set
        self.params.extend(self.Qparams)
        
        # Reinforce baseline
        self.baseline = ReinforceBaseline(belief_size, degree_size, ihidden_size)

    def setSampleMode(self,sample_mode,topN):
        self.sample_mode = sample_mode
        self.topN = topN

    def encode(self, belief_t, degree_t, intent_t,
            masked_source_t, masked_source_len_t,
            masked_target_t, masked_target_len_t, 
            utt_group_t, sample_t=None):
        
        # prepare belief state vector
        belief_t = G.disconnected_grad(T.concatenate(belief_t,axis=0))
        ##########################
        # prior parameterisarion #
        ##########################
        hidden_t = T.tanh( T.dot(belief_t,self.Ws1)+
                    T.dot(degree_t,self.Ws2)+
                    T.dot(intent_t,self.Ws3))
        prior_t  = T.nnet.softmax(
                    T.dot( T.tanh(
                        T.dot(hidden_t,self.Wp1)+self.bp1),
                        self.Wp2) )
       
        ##############################
        # posterior parameterisation #
        ##############################
        # response encoding
        target_intent_t = bidirectional_encode(
                self.tfEncoder, self.tbEncoder,
                masked_target_t, masked_target_len_t  )
        source_intent_t = bidirectional_encode(
                self.sfEncoder, self.sbEncoder,
                masked_source_t, masked_source_len_t  )
        # scores before softmax layer
        q_logit_t = T.dot(T.tanh( T.dot(belief_t,self.Wq1)+
                        T.dot(degree_t,self.Wq2)+
                        T.dot(source_intent_t,self.Wq3)+
                        T.dot(target_intent_t,self.Wq4)),
                    self.Wq5 )

        # sampling from a scaled posterior
        if self.sample_mode=='posterior':
            print '\t\tSampling from posterior ...'
            posterior_t= T.nnet.softmax(q_logit_t) 
            z_t = T.switch( T.lt(utt_group_t,self.dl-1),
                    utt_group_t,
                    G.disconnected_grad( T.argmax( 
                      self.srng.multinomial(
                      pvals=posterior_t,dtype='float32')[0])  )
                  )
        else:
            # choose to use the current sample or ground truth
            print '\t\tSampling from prior ...'
            z_t = T.switch( T.lt(utt_group_t,self.dl-1),
                    utt_group_t, sample_t)
        
        # put sample into decoder to decode 
        hidden_t = T.nnet.sigmoid(self.Wd2[z_t,:]+self.bd1)*hidden_t
        actEmb_t = T.tanh(T.dot(
                            T.concatenate( [T.tanh(self.Wd1[z_t,:]),hidden_t],axis=0 ),
                          self.Wd3)).dimshuffle('x',0)
        
        # return the true posterior
        posterior_t= T.nnet.softmax(q_logit_t)
        
        # compute baseline estimate
        b_t = self.baseline.encode(belief_t,degree_t,source_intent_t,target_intent_t)

        return actEmb_t, prior_t[0], posterior_t[0], z_t, b_t, posterior_t


    def decide(self,belief_t, degree_t, intent_t, 
            masked_source_t, masked_target_t, forced_sample=None):
        # prepare belief state vector
        belief_t = np.concatenate(belief_t,axis=0)
        # sample how many actions
        n = 1
        # forced sampling
        if forced_sample!=None: 
            z_t = [forced_sample]
            prob_t = None
        # different sampling mode
        elif self.sample_mode=='posterior' and masked_target_t!=None:
            # training time, sample from posterior
            z_t, prob_t = self._sample_from_posterior(
                belief_t, degree_t, intent_t, masked_source_t, masked_target_t)
        elif self.sample_mode=='prior':
            # testing time, sample from prior
            z_t, prob_t = self._sample_from_prior(belief_t, degree_t, intent_t)
       
        # state representation
        hidden_t = tanh(np.dot(belief_t,self.Ws1_backup)+
                        np.dot(degree_t,self.Ws2_backup)+
                        np.dot(intent_t,self.Ws3_backup) )
      
        # put sample into decoder to decode 
        hidden_t = np.multiply(sigmoid(self.Wd2_backup[z_t,:]+self.bd1_backup),hidden_t)
        hidden_t = np.repeat(hidden_t,n,axis=0)
        actEmb_t = tanh(np.dot(
                          np.concatenate([tanh(self.Wd1_backup[z_t,:]),hidden_t],axis=1),
                        self.Wd3_backup))
        
        return actEmb_t, z_t, prob_t

    def _sample_from_prior(self, belief_t, degree_t, intent_t):

        # prior parameterisarion
        hidden_t = tanh(np.dot(belief_t,self.Ws1_backup)+
                        np.dot(degree_t,self.Ws2_backup)+
                        np.dot(intent_t,self.Ws3_backup) )
        p_logit_t = np.dot(
                        tanh(np.dot(hidden_t,self.Wp1_backup)+self.bp1_backup),
                    self.Wp2_backup)
        
        # sampling from prior
        sortedIndex = np.argsort(p_logit_t)[::-1][:self.topN]
        topN_prior_t= softmax(p_logit_t[sortedIndex])
        z_t = sortedIndex[ np.argmax( np.random.multinomial(n=1,
                    pvals=topN_prior_t))    ]
        z_t = np.expand_dims(z_t,axis=0)
        # choose the top N samples 
        print 'Sample     : %s' % z_t
        print 'Prior dist.: %s' % sortedIndex
        print 'probability: %s' % topN_prior_t
        print
        return z_t, softmax(p_logit_t)

    def _sample_from_posterior(self, belief_t, degree_t, intent_t, 
            masked_source_t, masked_target_t):
        
        # Posterior
        # response encoding
        target_intent_t = bidirectional_read(
                self.tfEncoder, self.tbEncoder, masked_target_t)
        source_intent_t = bidirectional_read(
                self.sfEncoder, self.sbEncoder, masked_source_t)
        # posterior parameterisation
        q_logit_t = np.dot(tanh( 
                        np.dot(belief_t,self.Wq1_backup)+
                        np.dot(degree_t,self.Wq2_backup)+
                        np.dot(source_intent_t,self.Wq3_backup)+
                        np.dot(target_intent_t,self.Wq4_backup)),
                    self.Wq5_backup )

        # sampling from a scaled posterior
        sortedIndex = np.argsort(q_logit_t)[::-1][:self.topN]
        topN_posterior_t= softmax(q_logit_t[sortedIndex])
        z_t = sortedIndex[ np.argmax( np.random.multinomial(n=1,
                    pvals=topN_posterior_t))    ]
        #z_t = sortedIndex[0]
        z_t = np.expand_dims(z_t,axis=0)
        print sortedIndex[:3]
        print softmax(q_logit_t)[sortedIndex][:3]
        print 'Posterior  : %s' % sortedIndex
        print 'probability: %s' % topN_posterior_t
        
        return z_t, softmax(q_logit_t)

    def loadConverseParams(self):
        # decoder
        self.Wd1_backup = self.params[0].get_value()
        self.Wd2_backup = self.params[1].get_value()
        self.bd1_backup = self.params[2].get_value()
        self.Wd3_backup = self.params[3].get_value() 
        # state
        self.Ws1_backup = self.params[4].get_value()
        self.Ws2_backup = self.params[5].get_value()
        self.Ws3_backup = self.params[6].get_value()
        # latent policy (conditional prior)
        self.Wp1_backup = self.params[7].get_value()
        self.Wp2_backup = self.params[8].get_value()
        self.bp1_backup = self.params[9].get_value()
        # posterior
        self.Wq1_backup = self.params[10].get_value()
        self.Wq2_backup = self.params[11].get_value()  
        self.Wq3_backup = self.params[12].get_value() 
        self.Wq4_backup = self.params[13].get_value()
        self.Wq5_backup = self.params[14].get_value()
        # posterior sentence encoder
        self.tfEncoder.loadConverseParams()
        self.tbEncoder.loadConverseParams()
        self.sfEncoder.loadConverseParams()
        self.sbEncoder.loadConverseParams()
 

class ReinforceBaseline(BaseNNModule):
    
    def __init__(self, belief_size, degree_size, ihidden_size):

        self.Ws1= theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (belief_size,ihidden_size)).astype(theano.config.floatX))
        self.Ws2 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (degree_size,ihidden_size)).astype(theano.config.floatX))
        self.Ws3= theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*2,ihidden_size)).astype(theano.config.floatX))
        self.Wb1= theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*2,ihidden_size)).astype(theano.config.floatX))
        self.Wb2= theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size)).astype(theano.config.floatX))
        self.bb1= theano.shared(np.float32(-60))
        # all parameters
        self.params = [ self.Ws1,   self.Ws2,   self.Ws3, self.Wb1,    
                        self.Wb2,   self.bb1  ]

    def encode(self, belief_t, degree_t, source_intent_t, target_intent_t):
        h_t = T.tanh(   T.dot(belief_t,self.Ws1)+
                        T.dot(degree_t,self.Ws2)+
                        T.dot(source_intent_t,self.Ws3)+
                        T.dot(target_intent_t,self.Wb1))#+
        v_t = T.dot(h_t,self.Wb2) + self.bb1
        return v_t


class AttentivePolicy(BaseNNModule):
    
    # Policy network takes three inputs and produces a single 
    # system action embedding. Its use is heavily coupled with decoder.
    
    def __init__(self, belief_size, degree_size, ihidden_size, ohidden_size):

        # belief to action parameter
        self.Ws1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (belief_size,ohidden_size)).astype(theano.config.floatX))
        # matching degree to action parameter
        self.Ws2 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (degree_size,ohidden_size)).astype(theano.config.floatX))
        # intent to action parameter
        self.Ws3 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*2,ohidden_size)).astype(theano.config.floatX))
   
        # for attention scoring
        self.Wa1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ohidden_size,ohidden_size)).astype(theano.config.floatX))
        self.Wa2 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ohidden_size,ohidden_size)).astype(theano.config.floatX))
        self.Wa3 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ohidden_size,ohidden_size)).astype(theano.config.floatX))
        self.Va1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ohidden_size)).astype(theano.config.floatX))

        # all parameters
        self.params = [
                self.Ws1,   self.Ws2,   self.Ws3,
                self.Wa1,   self.Wa2,   self.Wa3,   self.Va1 ]

    def prepareBelief(self,belief_t):
        # belief vectors
        beliefs_t = []
        bn = 0
        for bvec in belief_t:
            size = bvec.shape[0]
            beliefs_t.append( T.tanh(T.dot(bvec,self.Ws1[bn:bn+size,:])).\
                    dimshuffle('x',0) )
            bn += size
        beliefs_t = T.concatenate(beliefs_t,axis=0)
        return beliefs_t 

    def encode(self, belief_t, degree_t, intent_t, ohidden_tjm1, wemb_tj):
        # embed 
        degree_t= T.tanh(T.dot(degree_t,self.Ws2))
        intent_t= T.tanh(T.dot(intent_t,self.Ws3))

        # score bias
        score_t = T.dot(belief_t,self.Wa3)+\
                    np.repeat(
                        T.dot(ohidden_tjm1,self.Wa1)+\
                        T.dot(wemb_tj,self.Wa2), 
                    10, axis=0)

        # attention mechanism
        atten_t= T.nnet.softmax(T.dot(T.nnet.sigmoid(score_t),self.Va1))[0]
        actEmb = T.tanh(T.dot(atten_t,belief_t)+degree_t+intent_t)
        return actEmb.dimshuffle('x',0)

    def _decideBelief(self,belief_t):
        # belief vectors
        beliefs_t = []
        bn = 0
        for bvec in belief_t:
            size = bvec.shape[0]
            beliefs_t.append( tanh(np.dot(bvec,self.Ws1_backup[bn:bn+size,:])))
            bn += size
        beliefs_t = np.concatenate(np.expand_dims(beliefs_t,axis=0),axis=0)
        return beliefs_t 

    def decide(self, belief_t, degree_t, intent_t, ohidden_tjm1, wemb_tj):
        # embed 
        degree_t = tanh(np.dot(degree_t,self.Ws2_backup))
        intent_t = tanh(np.dot(intent_t,self.Ws3_backup))

        # score bias
        score_t=np.dot(ohidden_tjm1,self.Wa1_backup)+\
                np.dot(wemb_tj, self.Wa2_backup)+\
                np.dot(belief_t,self.Wa3_backup)

        # attention mechanism
        atten_t= softmax(np.dot(sigmoid(score_t),self.Va1_backup))
        actEmb = tanh(np.dot(atten_t,belief_t)+degree_t+intent_t)
        return np.expand_dims(actEmb,axis=0)


    def loadConverseParams(self):
        self.Ws1_backup  = self.params[0].get_value()
        self.Ws2_backup  = self.params[1].get_value()
        self.Ws3_backup  = self.params[2].get_value()
        self.Wa1_backup  = self.params[3].get_value()
        self.Wa2_backup  = self.params[4].get_value()
        self.Wa3_backup  = self.params[5].get_value()
        self.Va1_backup  = self.params[6].get_value()
     

