######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import numpy as np
import theano
import theano.tensor as T
import theano.gradient as G
from utils.mathUtil import softmax, sigmoid, tanh
from encoder import *

# Informable slot tracker
class CNNInformableTracker(BaseNNModule):

    def __init__(self, belief_size, 
            ivocab_size, ihidden_size,
            ovocab_size, ohidden_size):

        # parameters for RNN tracker
        # tracker specific CNN encoder
        self.sCNN = CNNEncoder(ivocab_size, ihidden_size,
                pool=(False,False,True), level=3)
        self.tCNN = CNNEncoder(ovocab_size, ihidden_size, 
                pool=(False,False,True), level=3)

        # handling feature to belief mapping
        self.dbm1 = belief_size-1
        self.db   = belief_size
        self.dh   = int(belief_size/1.5)
        self.Wfbs = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*5,self.dh)).astype(theano.config.floatX))
        self.Wfbt = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*5,self.dh)).astype(theano.config.floatX))
        self.Whb  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))

        # handling belief self recurrence
        self.Wrec = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))
        self.Wnon = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))
        self.B    = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (1)).astype(theano.config.floatX))
        
        # bias term
        self.B0 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX)) 

        self.params = [
                self.Wfbs,  self.Wfbt,  self.Whb,   self.B,
                self.Wrec,  self.Wnon,  self.B0 ] +\
                self.sCNN.params + self.tCNN.params

        # initial state
        self.b0 = theano.shared(np.zeros((belief_size),\
                dtype=theano.config.floatX))
   
    def value_recur(self, vsrcpos_jsv, vtarpos_jsv, ssrcpos_jsv, starpos_jsv,
            b_jm1v, b_jm1N, ngms_j, ngmt_jm1, uttms_j, uttmt_jm1):

        # source features
        ssrcemb_jsv = T.sum(ngms_j[ssrcpos_jsv,:],axis=0)
        vsrcemb_jsv = T.sum(ngms_j[vsrcpos_jsv,:],axis=0)
        src_jsv = T.concatenate([ssrcemb_jsv,vsrcemb_jsv,uttms_j],axis=0)
        # target features
        staremb_jsv = T.sum(ngmt_jm1[starpos_jsv,:],axis=0)
        vtaremb_jsv = T.sum(ngmt_jm1[vtarpos_jsv,:],axis=0)
        tar_jsv = T.concatenate([staremb_jsv,vtaremb_jsv,uttmt_jm1],axis=0)
        # update g_jv 
        g_jv =  T.dot( self.Whb, T.nnet.sigmoid(
                T.dot(src_jsv,self.Wfbs) + T.dot(tar_jsv,self.Wfbt)+ 
                G.disconnected_grad(b_jm1v)*self.Wrec +
                G.disconnected_grad(b_jm1N)*self.Wnon + self.B0 ))
        
        return g_jv

    def recur(self, b_jm1, ms_j, mt_jm1, mscut_j, mtcut_jm1,
            ssrcpos_js, vsrcpos_js, starpos_js, vtarpos_js ):
      
        # cnn encoding
        ngms_j,  uttms_j   = self.sCNN.encode(ms_j,  mscut_j)
        ngmt_jm1,uttmt_jm1 = self.tCNN.encode(mt_jm1,mtcut_jm1)

        # padding dummy vector
        ngms_j   = T.concatenate([ngms_j,T.zeros_like(ngms_j[-1:,:])],axis=0)
        ngmt_jm1 = T.concatenate([ngmt_jm1,T.zeros_like(ngmt_jm1[-1:,:])],axis=0)

        # new belief
        g_j,_  = theano.scan(fn=self.value_recur,\
                sequences=[ vsrcpos_js, vtarpos_js,\
                            ssrcpos_js, starpos_js, b_jm1[:-1]],\
                non_sequences=[ b_jm1[-1], ngms_j, ngmt_jm1,\
                                uttms_j, uttmt_jm1],\
                outputs_info=None)
        # produce new belief b_j
        g_j = T.concatenate([g_j,self.B],axis=0)
        b_j = T.nnet.softmax( g_j )[0,:]
        
        return b_j#, g_j

    def track(self, b_jm1, ms_j, mt_jm1,
            ssrcpos_js, vsrcpos_js, starpos_js, vtarpos_js ):
        
        # cnn encoding
        ngms_j,  uttms_j   = self.sCNN.read(ms_j)
        ngmt_jm1,uttmt_jm1 = self.tCNN.read(mt_jm1)

        # padding dummy vector 
        ngms_j = np.concatenate([ngms_j,np.zeros_like(ngms_j[-1:,:])],axis=0)
        ngmt_jm1 = np.concatenate([ngmt_jm1,np.zeros_like(ngmt_jm1[-1:,:])],axis=0)
        
        # new belief
        g_j = np.zeros(self.dbm1)
        for v in range(self.dbm1):
            # source features
            ssrcemb_jsv = np.sum(ngms_j[ssrcpos_js[v],:],axis=0)
            vsrcemb_jsv = np.sum(ngms_j[vsrcpos_js[v],:],axis=0)
            src_jsv = np.concatenate([ssrcemb_jsv,vsrcemb_jsv,uttms_j],axis=0)
            # target features
            staremb_jsv = np.sum(ngmt_jm1[starpos_js[v],:],axis=0)
            vtaremb_jsv = np.sum(ngmt_jm1[vtarpos_js[v],:],axis=0)
            tar_jsv = np.concatenate([staremb_jsv,vtaremb_jsv,uttmt_jm1],axis=0)
            
            # update g_jv value
            g_jv = np.dot( self.Whb_backup, sigmoid(
                np.dot(src_jsv,self.Wfbs_backup) + 
                np.dot(tar_jsv,self.Wfbt_backup) + 
                b_jm1[v] *self.Wrec_backup +
                b_jm1[-1]*self.Wnon_backup +
                self.B0_backup ))
            g_j[v] = g_jv

        # produce new belief b_j
        g_j = np.concatenate([g_j,self.B_backup],axis=0)
        b_j = softmax( g_j )
        
        return b_j

    def setParams(self,params):
        for i in range(len(self.params)):
            self.params[i].set_value(params[i])
        self.sCNN.setParams(params[7:7+len(self.sCNN.params)])
        self.tCNN.setParams(params[-len(self.tCNN.params):])

    def loadConverseParams(self):
        self.Wfbs_backup = self.params[0].get_value()
        self.Wfbt_backup = self.params[1].get_value()
        self.Whb_backup  = self.params[2].get_value()
        self.B_backup    = self.params[3].get_value()
        self.Wrec_backup = self.params[4].get_value()
        self.Wnon_backup = self.params[5].get_value()
        self.B0_backup   = self.params[6].get_value()
        self.sCNN.loadConverseParams()
        self.tCNN.loadConverseParams()


# Requestable slot tracker
class CNNRequestableTracker(BaseNNModule):

    def __init__(self, 
            ivocab_size, ihidden_size,
            ovocab_size, ohidden_size):

        # parameters for RNN tracker
        # handling feature to belief mapping
        
        # tracker specific CNN encoder
        self.sCNN = CNNEncoder(ivocab_size, ihidden_size,
                pool=(False,False,True), level=3)
        self.tCNN = CNNEncoder(ovocab_size, ohidden_size,
                pool=(False,False,True), level=3)

        self.dh   = 4
        self.Wfbs = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*5,self.dh)).astype(theano.config.floatX))
        self.Wfbt = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ihidden_size*5,self.dh)).astype(theano.config.floatX))
        self.Whb  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))
        self.B    = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (1)).astype(theano.config.floatX))
        
        # bias term
        self.B0 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))

        self.params = [
                self.Wfbs,  self.Wfbt,  self.Whb,   
                self.B,     self.B0 ] + \
                self.sCNN.params + self.tCNN.params

    
    def recur(self, ms_j, mt_jm1, mscut_j, mtcut_jm1,
            ssrcpos_js, vsrcpos_js, starpos_js, vtarpos_js ):
        
         # cnn encoding
        ngms_j,  uttms_j   = self.sCNN.encode(ms_j,  mscut_j)
        ngmt_jm1,uttmt_jm1 = self.tCNN.encode(mt_jm1,mtcut_jm1)
        
        # padding dummy vector
        ngms_j   = T.concatenate([ngms_j,T.zeros_like(ngms_j[-1:,:])],axis=0)
        ngmt_jm1 = T.concatenate([ngmt_jm1,T.zeros_like(ngmt_jm1[-1:,:])],axis=0)

        # source features
        ssrcemb_js = T.sum(ngms_j[ssrcpos_js,:],axis=0)
        vsrcemb_js = T.sum(ngms_j[vsrcpos_js,:],axis=0)
        src_js = T.concatenate([ssrcemb_js,vsrcemb_js,uttms_j],axis=0)
        
        # target features
        staremb_js = T.sum(ngmt_jm1[starpos_js,:],axis=0)
        vtaremb_js = T.sum(ngmt_jm1[vtarpos_js,:],axis=0)
        tar_js = T.concatenate([staremb_js,vtaremb_js,uttmt_jm1],axis=0)
       
        # update g_j
        g_j   = T.dot( self.Whb, T.nnet.sigmoid( 
                T.dot(src_js,self.Wfbs) + 
                T.dot(tar_js,self.Wfbt) +
                self.B0)).dimshuffle('x')
        # update b_j
        g_j = T.concatenate([g_j,self.B],axis=0)
        b_j = T.nnet.softmax( g_j )[0,:]
        
        return b_j
    
    def track(self, ms_j, mt_jm1, 
            ssrcpos_js, vsrcpos_js, starpos_js, vtarpos_js ):
        
        # cnn encoding
        ngms_j,  uttms_j   = self.sCNN.read(ms_j)
        ngmt_jm1,uttmt_jm1 = self.tCNN.read(mt_jm1)
        
        # padding dummy vector 
        ngms_j = np.concatenate([ngms_j,np.zeros_like(ngms_j[-1:,:])],axis=0)
        ngmt_jm1 = np.concatenate([ngmt_jm1,np.zeros_like(ngmt_jm1[-1:,:])],axis=0)
        
        # source features
        ssrcemb_js = np.sum(ngms_j[ssrcpos_js,:],axis=0)
        vsrcemb_js = np.sum(ngms_j[vsrcpos_js,:],axis=0)
        src_js = np.concatenate([ssrcemb_js,vsrcemb_js,uttms_j],axis=0)
        
        # target features
        staremb_js = np.sum(ngmt_jm1[starpos_js,:],axis=0)
        vtaremb_js = np.sum(ngmt_jm1[vtarpos_js,:],axis=0)
        tar_js = np.concatenate([staremb_js,vtaremb_js,uttmt_jm1],axis=0)
       
        # update g_j 
        g_j = np.dot( self.Whb_backup, sigmoid(
                np.dot(src_js,self.Wfbs_backup) +
                np.dot(tar_js,self.Wfbt_backup) +
                self.B0_backup ))
        
        # update b_j
        g_j = np.array([g_j,self.B_backup])
        b_j = softmax( g_j )
        
        return b_j

    def loadConverseParams(self):
        self.Wfbs_backup = self.params[0].get_value()
        self.Wfbt_backup = self.params[1].get_value()
        self.Whb_backup  = self.params[2].get_value()
        self.B_backup    = self.params[3].get_value()
        self.B0_backup   = self.params[4].get_value()
        self.sCNN.loadConverseParams()
        self.tCNN.loadConverseParams()


# Informable slot tracker
class NgramInformableTracker(BaseNNModule):

    def __init__(self, ng_size, belief_size):

        # parameters for RNN tracker
        # handling feature to belief mapping
        self.dbm1 = belief_size-1
        self.db   = belief_size
        self.dh   = int(belief_size/1.5)
        self.Wfbs = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ng_size,self.dh)).astype(theano.config.floatX))
        self.Wfbt = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ng_size,self.dh)).astype(theano.config.floatX))

        self.Whb  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))

        # handling belief self recurrence
        self.Wrec = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))
        self.Wnon = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))
        self.B    = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (1)).astype(theano.config.floatX))
        
        # bias term
        self.B0 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))

        self.params = [
                self.Wfbs,  self.Wfbt,  self.Whb,   self.B,
                self.Wrec,  self.Wnon,  self.B0]

        # initial state
        self.b0 = theano.shared(np.zeros((belief_size),\
                dtype=theano.config.floatX))
   
    def value_recur(self, ngs_jv, ngt_jv, b_jm1v, b_jm1N):
        
        # padding dummy
        Wfbs = T.concatenate([self.Wfbs,T.zeros_like(self.Wfbs[-1:,:])],
                axis=0)
        Wfbt = T.concatenate([self.Wfbt,T.zeros_like(self.Wfbt[-1:,:])],
                axis=0)
        # get ngram embedding
        fembs_v= T.sum(Wfbs[ngs_jv,:],axis=0)
        fembt_v= T.sum(Wfbt[ngt_jv,:],axis=0)
        # calculate g value
        g_jv =  T.dot( self.Whb, T.nnet.sigmoid(
                fembs_v + fembt_v + 
                G.disconnected_grad(b_jm1v)*self.Wrec +
                G.disconnected_grad(b_jm1N)*self.Wnon +
                self.B0 ))
        
        return g_jv

    def recur(self, b_jm1, ngs_j, ngt_j):
        
        # new belief
        g_j,_  = theano.scan(fn=self.value_recur,\
                sequences=[ ngs_j[:-1],ngt_j[:-1],\
                            b_jm1[:-1]],\
                non_sequences=[ b_jm1[-1]],\
                outputs_info=None)

        g_j = T.concatenate([g_j,self.B],axis=0)
        b_j = T.nnet.softmax( g_j )[0,:]
        
        return b_j, g_j

    def track(self, b_jm1, ngs_j, ngt_j):
        
        # padding dummy
        Wfbs = np.concatenate([self.Wfbs_backup,\
                np.zeros_like(self.Wfbs_backup[-1:,:])],axis=0)
        Wfbt = np.concatenate([self.Wfbt_backup,\
                np.zeros_like(self.Wfbt_backup[-1:,:])],axis=0)

        # new belief
        g_j = np.zeros(self.dbm1)
        for v in range(self.dbm1):
            ngsidx = ngs_j[v]
            ngtidx = ngt_j[v]
            
            fembs_v = np.sum(Wfbs[ngsidx,:],axis=0)
            fembt_v = np.sum(Wfbt[ngtidx,:],axis=0)
            
            g_jv = np.dot( self.Whb_backup, sigmoid(
                fembs_v + fembt_v + 
                b_jm1[v] *self.Wrec_backup +
                b_jm1[-1]*self.Wnon_backup +
                self.B0_backup ))
            g_j[v] = g_jv

        g_j = np.concatenate([g_j,self.B_backup],axis=0)
        b_j = softmax( g_j )
        
        return b_j

    def loadConverseParams(self):
        self.Wfbs_backup = self.params[0].get_value()
        self.Wfbt_backup = self.params[1].get_value()
        self.Whb_backup  = self.params[2].get_value()
        self.B_backup    = self.params[3].get_value()
        self.Wrec_backup = self.params[4].get_value()
        self.Wnon_backup = self.params[5].get_value()
        self.B0_backup   = self.params[6].get_value()

# Requestable slot tracker
class NgramRequestableTracker(BaseNNModule):

    def __init__(self, ng_size):

        # parameters for RNN tracker
        # handling feature to belief mapping
        self.dh   = 4
        self.Wfbs = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ng_size,self.dh)).astype(theano.config.floatX))
        self.Wfbt = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (ng_size,self.dh)).astype(theano.config.floatX))

        self.Whb  = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))
        self.B    = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (1)).astype(theano.config.floatX))
        
        # bias term
        self.B0 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh)).astype(theano.config.floatX))

        self.params = [
                self.Wfbs,  self.Wfbt,  self.Whb,   
                self.B,     self.B0 ]

    
    def recur(self, ngs_j, ngt_j ):
        
        # padding dummy
        Wfbs = T.concatenate([self.Wfbs,T.zeros_like(self.Wfbs[-1:,:])],
                axis=0)
        Wfbt = T.concatenate([self.Wfbt,T.zeros_like(self.Wfbt[-1:,:])],
                axis=0)

        # new belief
        fembs = T.sum(Wfbs[ngs_j,:],axis=0)
        fembt = T.sum(Wfbt[ngt_j,:],axis=0)
        g_j   = T.dot( self.Whb, T.nnet.sigmoid( 
                fembs + fembt + self.B0)).dimshuffle('x')

        g_j = T.concatenate([g_j,self.B],axis=0)
        b_j = T.nnet.softmax( g_j )[0,:]
        
        return b_j

    def track(self, ngs_j, ngt_j):
        
        # padding dummy
        Wfbs = np.concatenate([self.Wfbs_backup,\
                np.zeros_like(self.Wfbs_backup[-1:,:])],axis=0)
        Wfbt = np.concatenate([self.Wfbt_backup,\
                np.zeros_like(self.Wfbt_backup[-1:,:])],axis=0)

        # new belief
        fembs_v = np.sum(Wfbs[ngs_j,:],axis=0)
        fembt_v = np.sum(Wfbt[ngt_j,:],axis=0)
            
        g_j = np.dot( self.Whb_backup, sigmoid(
                fembs_v + fembt_v + self.B0_backup ))

        g_j = np.array([g_j,self.B_backup])
        b_j = softmax( g_j )
        
        return b_j

    def loadConverseParams(self):
        self.Wfbs_backup = self.params[0].get_value()
        self.Wfbt_backup = self.params[1].get_value()
        self.Whb_backup  = self.params[2].get_value()
        self.B_backup    = self.params[3].get_value()
        self.B0_backup   = self.params[4].get_value()



