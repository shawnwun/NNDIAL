######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import numpy as np
import theano
import theano.tensor as T
from utils.mathUtil import softmax, sigmoid, tanh
from Queue import PriorityQueue
import operator
from math import pow
from copy import deepcopy
import sys
import theano.gradient as G

from policy import *
from basic  import *

# Node class for performing beam search
class BeamSearchNode(object):

    def __init__(self,h,c,prevNode,wordid,logp,leng,record):
        self.h = h
        self.c = c
        self.prevNode = prevNode
        self.wordid = wordid
        self.logp   = logp
        self.leng   = leng
        self.record = record
    
    def eval(self, repeatPenalty, tokenReward, scoreTable, alpha=1.0):
        reward = 0
        # repeat penalty
        if repeatPenalty=='inf':
            # value repeat is not allowed
            for k,v in self.record['v'].iteritems():
                if v>1: reward -= 1000
            # slot repeat is slightly allowed
            for k,v in self.record['s'].iteritems():
                if v>1: reward -= pow(v-1,2)*0.5
        # special token reward
        if tokenReward and scoreTable!=None:
            for k,v in self.record['v'].iteritems():
                if v>0 and scoreTable.has_key(k):
                    reward += scoreTable[k]
        
        return self.logp/float(self.leng-1+1e-6)+alpha*reward 


def initParamsHelper(doh, do, struct):
    if struct=='lstm_lm':
        # params
        oWgate  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (doh,doh*4)).astype(theano.config.floatX))
        oUgate  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (doh,doh*4)).astype(theano.config.floatX))
        Wzh     = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (doh,doh*4)).astype(theano.config.floatX))
        b       = np.random.uniform(-0.3,0.3,(doh*3))
        b[doh:doh*2] = 2.0
        b       = theano.shared(b.astype(theano.config.floatX))
    elif struct=='lstm_cond':
        # params
        oWgate  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (doh,doh*5)).astype(theano.config.floatX))
        oUgate  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (doh,doh*5)).astype(theano.config.floatX))
        Wzh     = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (doh,doh*5)).astype(theano.config.floatX))
        b       = np.random.uniform(-0.3,0.3,(doh*4))
        b[doh:doh*2] = 2.0
        b       = theano.shared(b.astype(theano.config.floatX))
    elif struct=='lstm_mix':
        # params
        oWgate  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (doh,doh*5)).astype(theano.config.floatX))
        oUgate  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (doh,doh*5)).astype(theano.config.floatX))
        Wzh     = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (doh,doh*5)).astype(theano.config.floatX))
        b       = np.random.uniform(-0.3,0.3,(doh*4))
        b[doh:doh*2] = 2.0
        b       = theano.shared(b.astype(theano.config.floatX))
  
    Who = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
            (doh,do)).astype(theano.config.floatX))
    # initial memory cell
    oc0 = theano.shared(np.zeros((1,doh),dtype=theano.config.floatX))
    oh0 = theano.shared(np.zeros((1,doh),dtype=theano.config.floatX))
    return oWgate, oUgate, Wzh, Who, b, oh0, oc0


# LSTM decoder
class Decoder(BaseNNModule):
    
    def __init__(self, policy, vocab_size, ohidden_size,
            struct, ply, use_snap=False, snap_size=0):

        # parameters for decoder LSTM
        self.use_snapshot = use_snap
        self.dsp    = snap_size
        self.struct = struct
        self.ply    = ply
        self.doh    = ohidden_size
        self.do     = vocab_size
        self.policy = policy

        # init parameters
        self.oWgate, self.oUgate, self.Wzh, self.Who, self.b, self.oh0,self.oc0=\
                initParamsHelper( self.doh, self.do, self.struct)
        self.Wemb   = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.do,self.doh)).astype(theano.config.floatX))
        self.params = [ self.oWgate,self.oUgate,self.Wemb,
                        self.Wzh,   self.Who,   self.b  ]
    
    def setDecodeConfig(self, verbose, topk, beamwidth,
            vocab, repeat_penalty, token_reward, alpha ):
        self.vocab          = vocab
        self.verbose        = verbose
        self.beamwidth      = beamwidth
        self.topk           = topk
        self.token_reward   = token_reward
        self.repeat_penalty = repeat_penalty
        self.alpha          = alpha
        self.q_limit        = 10000

        # special token accumulation table
        self.recordTable = {'s':{},'v':{}}
        for idx in range(len(self.vocab)):
            w = self.vocab[idx]
            if w.startswith('[VALUE_'):
                self.recordTable['v'][idx] = 0.0
            elif w.startswith('[SLOT_'):
                self.recordTable['s'][idx] = 0.0
    
    # decoder LSTM recurrence
    def recur(self, w_j, y_j, sshot_tj, oh_jm1, oc_jm1, 
            intent_t, degree_t, belief_t ,actEmb_t ): 
        epsln = 10e-6
        # current input
        in_j = T.nnet.sigmoid( self.Wemb[w_j] )

        # policy embedding
        if self.ply=='attention':
            actEmb_tj = self.policy.encode(belief_t, degree_t,
                    intent_t, oh_jm1, in_j.dimshuffle('x',0))
        else: # normal or latent policy 
            actEmb_tj = actEmb_t

        # snapshot 
        if self.use_snapshot:
            prdtshot_j = actEmb_tj[:,:self.dsp]/2.0+0.5
            snapcost_j = \
                T.mean(sshot_tj*T.log10(prdtshot_j+epsln))+\
                T.mean((1.0-sshot_tj)*T.log10(1.0-prdtshot_j+epsln))
        else: snapcost_j = T.sum(sshot_tj)
        
        # syntatic memory cell and gate
        # compute i, f, o, c together and slice it
        bundle_j =  T.dot(in_j,     self.oWgate).dimshuffle('x',0)+\
                    T.dot(oh_jm1,   self.oUgate)
        bundle_aj=  T.dot(actEmb_tj,self.Wzh)
        # input gate
        ig = T.nnet.sigmoid(bundle_j[:,:self.doh]+
                            bundle_aj[:,:self.doh]+
                            self.b[:self.doh])
        # use forget bias or not
        fg = T.nnet.sigmoid(bundle_j[:,self.doh:self.doh*2]+
                            bundle_aj[:,self.doh:self.doh*2]+
                            self.b[self.doh:self.doh*2])
        # output gate
        og = T.nnet.sigmoid(bundle_j[:,self.doh*2:self.doh*3]+
                            bundle_aj[:,self.doh*2:self.doh*3]+
                            self.b[self.doh*2:self.doh*3])
        # proposed memory cell
        # reading gate, memory cell, hidden layer
        if self.struct=='lstm_cond': # reading gate control signal
            rg   =  T.nnet.sigmoid( bundle_j[:,self.doh*4:self.doh*5]+
                                    bundle_aj[:,self.doh*4:self.doh*5]+
                                    self.b[self.doh*3:])
            cx_j =  T.tanh(bundle_j[:,self.doh*3:self.doh*4])
            oc_j =  ig*cx_j + fg*oc_jm1 +\
                    rg*T.tanh(bundle_aj[:,self.doh*3:self.doh*4])
            oh_j =  og*T.tanh(oc_j)
            o_j  =  T.nnet.softmax( T.dot(oh_j,self.Who) )
        elif self.struct=='lstm_mix':# two signals
            rg   =  T.nnet.sigmoid( bundle_j[:,self.doh*4:self.doh*5]+
                                    bundle_aj[:,self.doh*4:self.doh*5]+
                                    self.b[self.doh*3:])
            cx_j =  T.tanh(bundle_j[:,self.doh*3:self.doh*4])
            oc_j =  ig*cx_j + fg*oc_jm1
            oh_j =  og*T.tanh(oc_j) + \
                    rg*T.tanh(bundle_aj[:,self.doh*3:self.doh*4])
            o_j  = T.nnet.softmax( T.dot(oh_j,self.Who) )
        elif self.struct=='lstm_lm': # lm style
            cx_j =  T.tanh( bundle_j[:,self.doh*3:self.doh*4]+
                            bundle_aj[:,self.doh*3:self.doh*4])
            oc_j =  ig*cx_j + fg*oc_jm1
            oh_j =  og*T.tanh(oc_j)
            o_j  =  T.nnet.softmax( T.dot(oh_j,self.Who) )
        else:
            sys.exit('[ERROR]: Unseen decoder structure '+self.struct)
        # compute output distribution and cross entropy error
        p_j = o_j[:,y_j] 

        return oh_j, oc_j, p_j, snapcost_j

    # decoding function
    def decode(self, masked_source_t, masked_source_len_t,
            masked_target_t, masked_target_len_t,
            intent_t, belief_t, degree_t, utt_group_t,
            snapshot_t, sample_t):
        # decide policy type
        if self.ply=='attention':
            # set dummpy, not used
            actEmb_t = T.concatenate(belief_t,axis=0)
            prior_t = posterior_t = z_t = base_t = debug_t = None
            # attentive policy
            belief_t = self.policy.prepareBelief(belief_t)
        elif self.ply=='normal':
            # deterministic policy
            actEmb_t = self.policy.encode(belief_t,degree_t,intent_t)
            # set dummpy, not used
            prior_t = posterior_t= z_t = base_t = debug_t = None
            belief_t = T.concatenate(belief_t,axis=0)
        elif self.ply=='latent':
            # variational policy, return: actEmb, prior, posterior, sample
            actEmb_t, prior_t, posterior_t, z_t, base_t, debug_t =\
                self.policy.encode(
                    belief_t,degree_t,intent_t,
                    masked_source_t,masked_source_len_t,
                    masked_target_t,masked_target_len_t,
                    utt_group_t, sample_t)
            # set dummy, not used
            belief_t = T.concatenate(belief_t,axis=0)

        # recurrence
        [oh_t,oc_t,p_t,scost_t],_= theano.scan(fn=self.recur,\
                sequences=[
                    masked_target_t[:-1],masked_target_t[1:],snapshot_t[1:]],\
                outputs_info=[self.oh0,self.oc0,None,None],
                non_sequences=[intent_t,degree_t,belief_t,actEmb_t])

        return p_t, scost_t, prior_t, posterior_t, z_t, base_t, debug_t
  
    # function for testing
    def talk(self, masked_intent_t, belief_t, degree_t, 
            masked_source_t=None, masked_target_t=None,
            scoreTable=None, forced_sample=None):

        # decide belief vector
        if self.ply=='attention':
            # attentive policy
            belief_t = self.policy._decideBelief(belief_t)
            actEmb_t = np.zeros((1,1))
            sample_t = prob_t = None
        elif self.ply=='normal':
            # deterministic policy
            actEmb_t = self.policy.decide(belief_t,degree_t,masked_intent_t)
            sample_t = prob_t = None
        elif self.ply=='latent':
            # variational policy
            actEmb_t, sample_t, prob_t = self.policy.decide(
                    belief_t,degree_t,masked_intent_t,
                    masked_source_t,masked_target_t,forced_sample)
        
        # stored end node
        endnodes = []

        # iterate through actions
        for a in range(actEmb_t.shape[0]):
            # compute how many sentences we want to generate for this act
            prev_endnode = len(endnodes)
            number_required = min(  (self.topk+1)/actEmb_t.shape[0],
                                    self.topk-len(endnodes))

            # hidden layers to be stored
            h0 = np.zeros(self.doh)
            c0 = np.zeros(self.doh)
            # starting node
            node = BeamSearchNode(h0,c0,None,1,0,1,\
                    record=deepcopy(self.recordTable))
            nodes= PriorityQueue()
            # put it in the queue
            nodes.put(( -node.eval(self.repeat_penalty,self.token_reward,\
                    scoreTable,self.alpha), node))
            qsize = 1
            # start beam search
            while True:
                
                # give up when decoding takes too long 
                if qsize>self.q_limit: break
                
                # fetch the best node
                score, n = nodes.get()

                # if end of sentence token 
                if n.wordid==1 and n.prevNode!=None:
                    endnodes.append((score,n))
                    # if reach maximum # of sentences required
                    if len(endnodes)-prev_endnode>=number_required:break
                    else:   continue
                
                # decode for one step using decoder 
                nextnodes = self._forwardpass(n, 
                        masked_intent_t, belief_t, degree_t, actEmb_t[a,:],
                        scoreTable)
                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put( (score,nn) )
                # increase qsize
                qsize += len(nextnodes)-1
        
        # choose nbest paths, back trace them
        if len(endnodes)==0:
            endnodes = [nodes.get() for n in range(self.topk)]
        utts = []
        for score,n in sorted(endnodes,key=operator.itemgetter(0)):
            utt,att,gates = [],[],[]
            utt.append(n.wordid)
            # back trace
            while n.prevNode!=None:
                n = n.prevNode
                utt.append(n.wordid)
            
            utt,att,gates = utt[::-1],att[::-1],gates[::-1]
            utts.append([utt,att,gates])
        return utts, sample_t, prob_t
    
    def _forwardpass(self, n, intent_t, belief_vec_t, degree_t, actEmb_t,
            scoreTable):
        
        # forward pass
        in_j    = sigmoid( self.Wemb_backup[n.wordid] )
        
        # action embedding
        if self.ply=='attention':
            actEmb_tj = self.policy.decide(belief_vec_t, 
                    degree_t, intent_t, n.h, in_j)[0]
        else: # fixed action embedding
            actEmb_tj = actEmb_t

        # syntatic memory cell and gate
        # compute i, f, o, c together and slice it
        bundle_j =  np.dot(in_j,self.oWgate_backup) +\
                    np.dot(n.h, self.oUgate_backup)
        bundle_aj=  np.dot(actEmb_tj,self.Wzh_backup)
        # input gate
        ig   = sigmoid( bundle_j[:self.doh]+
                        bundle_aj[:self.doh]+
                        self.b_backup[:self.doh])
        # use forget bias or not
        fg = sigmoid(   bundle_j[self.doh:self.doh*2]+
                        bundle_aj[self.doh:self.doh*2]+
                        self.b_backup[self.doh:self.doh*2])
        # output gate
        og   = sigmoid( bundle_j[self.doh*2:self.doh*3]+
                        bundle_aj[self.doh*2:self.doh*3]+
                        self.b_backup[self.doh*2:self.doh*3])
        # proposed memory cell
        # reading gate, memory cell, hidden layer
        if self.struct=='lstm_cond': # reading gate control signal
            rg   = sigmoid( bundle_j[self.doh*4:self.doh*5]+
                            bundle_aj[self.doh*4:self.doh*5]+
                            self.b_backup[self.doh*3:])
            cx_j =  tanh(bundle_j[self.doh*3:self.doh*4])
            oc_j =  np.multiply(ig,cx_j)+\
                    np.multiply(fg,n.c)+\
                    np.multiply(rg,tanh(bundle_aj[self.doh*3:self.doh*4]))
            oh_j = np.multiply(og,tanh(oc_j))
            o_j  = softmax( np.dot(oh_j, self.Who_backup) )
        elif self.struct=='lstm_mix':# two signals   
            rg   = sigmoid( bundle_j[self.doh*4:self.doh*5]+
                            bundle_aj[self.doh*4:self.doh*5]+
                            self.b_backup[self.doh*3:])
            cx_j =  tanh(bundle_j[self.doh*3:self.doh*4])
            oc_j =  np.multiply(ig,cx_j)+\
                    np.multiply(fg,n.c)
            oh_j =  np.multiply(og,tanh(oc_j))+\
                    np.multiply(rg,tanh(bundle_aj[self.doh*3:self.doh*4]))
            o_j  = softmax( np.dot(oh_j, self.Who_backup) )
        elif self.struct=='lstm_lm': # lm style
            cx_j =  tanh(   bundle_j[self.doh*3:self.doh*4]+
                            bundle_aj[self.doh*3:self.doh*4])
            oc_j =  np.multiply(ig,cx_j)+\
                    np.multiply(fg,n.c)
            oh_j =  np.multiply(og,tanh(oc_j))
            o_j  = softmax( np.dot(oh_j, self.Who_backup) )
        else:
            sys.exit('[ERROR]: Unseen decoder structure '+self.struct)
       
        # compute output distribution, logp, and sample
       
        # make sure we won't sample unknown word
        o_j[0] = 0.0
        selected_words = np.argsort(o_j)[::-1][:self.beamwidth]
 
        # expand nodes and add additional reward
        nextnodes = []
        for wid in selected_words: # ignore <unk> token
            # loglikelihood of current word
            logp = np.log10(o_j[wid])

            # update record for new node
            new_record = deepcopy(n.record)
            if new_record['s'].has_key(wid):
                new_record['s'][wid] += 1
            if new_record['v'].has_key(wid):
                new_record['v'][wid] += 1
            
            # create new node and score it
            node = BeamSearchNode(oh_j,oc_j,n,wid,\
                    n.logp+logp,n.leng+1,new_record)
            
            # store nodes
            nextnodes.append( \
                    (-node.eval(self.repeat_penalty,self.token_reward,\
                    scoreTable,self.alpha), node))

        return nextnodes

    def loadConverseParams(self):
        self.oWgate_backup  = self.params[0].get_value()
        self.oUgate_backup  = self.params[1].get_value()
        self.Wemb_backup    = self.params[2].get_value()
        self.Wzh_backup     = self.params[3].get_value() 
        self.Who_backup     = self.params[4].get_value()
        self.b_backup       = self.params[5].get_value()

