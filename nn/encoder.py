######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import numpy as np
import theano
import theano.tensor as T
from utils.mathUtil import softmax, sigmoid, tanh
from math import ceil, floor
from theano.tensor.signal import pool
from basic import BaseNNModule

def max_pool(data, factor=(3,1),ignore_border=False):

    # calculate new shape
    shape = data.shape
    fw, fh = factor
    w = float(data.shape[0])/float(fw)
    h = float(data.shape[1])/float(fh)
    if ignore_border:
        w,h = int(floor(w)), int(floor(h))
    else:
        w,h = int(ceil(w)),  int(ceil(h))
    
    # pooling
    pool_out = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            wend = min((i+1)*fw,data.shape[0])
            hend = min((j+1)*fh,data.shape[1])
            pool_out[i,j] = np.max( data[i*fw:wend,j*fh:hend] )
    return pool_out


# CNN Encoder
class CNNEncoder(BaseNNModule):

    def __init__(self, vocab_size, hidden_size, pool, level=3):
        self.dh     = hidden_size
        self.di     = vocab_size
        self.level  = level
        # embedding to hidden transform
        self.Wemb   = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.di,self.dh)).astype(theano.config.floatX))
        # conv weights
        self.Wcv1 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh*3,self.dh)).astype(theano.config.floatX))
        self.Wcv2 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh*3,self.dh)).astype(theano.config.floatX))
        self.Wcv3 = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dh*3,self.dh)).astype(theano.config.floatX))
        self.pool = pool
    
        self.params = [ self.Wemb,  self.Wcv1,  self.Wcv2 ]
        if level>=3:
            self.params.append(self.Wcv3)

    def conv(self,w_tm1,w_t,w_tp1,W):
        wv_t = T.concatenate([w_tm1,w_t,w_tp1],axis=0)
        return T.dot(wv_t,W)

    def convolution(self,cv_input,Wcv,mode='T'):

        if mode=='T': # theano mode
            # padding dummy vec
            dummyvec = T.zeros_like(cv_input[:1,:])
            cv_input = T.concatenate([dummyvec,cv_input,dummyvec],axis=0)
            # convolution
            cv_output,_= theano.scan(fn=self.conv,\
                sequences=[cv_input[:-2],cv_input[1:-1],cv_input[2:]],\
                outputs_info=None,
                non_sequences=Wcv)
            return cv_output
        else: # numpy mode
            # padding dummy vec
            dummyvec = np.zeros_like(cv_input[:1,:])
            cv_input = np.concatenate([dummyvec,cv_input,dummyvec],axis=0)
            # 1st convolution
            cv_output = []
            for i in range(1,cv_input.shape[0]-1):
                w3g = np.concatenate(
                    [cv_input[i-1:i,:],cv_input[i:i+1,:],cv_input[i+1:i+2,:]],
                    axis=1)
                cv_output.append(np.dot(w3g,Wcv))
            cv_output = np.concatenate(cv_output,axis=0)
            return cv_output

    def encode(self, utt_j, uttcut_j):
       
        # transform word embedding to hidden size
        emb_j = T.tanh( self.Wemb[utt_j[:uttcut_j],:] )
        
        # 1st convolution
        wh1_j = self.convolution(emb_j,self.Wcv1)
        if self.pool[0]: # pooling
            wh1_j = pool.max_pool(input=wh1_j,ds=(3,1),ignore_border=False)
        wh1_j = T.tanh(wh1_j)

        # 2nd convolution
        wh2_j = self.convolution(wh1_j, self.Wcv2)
        if self.pool[1]: # pooling
            wh2_j = pool.max_pool(input=wh2_j,ds=(3,1),ignore_border=False)
        wh2_j = T.tanh(wh2_j)
        
        if self.level>=3:
            # 3nd convolution
            wh3_j = self.convolution(wh2_j, self.Wcv3)
            if self.pool[2]:
                wh3_j = pool.pool_2d(input=wh3_j,ds=(3,1),
                        ignore_border=False)
            # average pooling
            wh3_j = T.tanh(T.sum(wh3_j,axis=0))
        else: # level < 3
            wh3_j = None
        
        if self.pool==(True,True,True):
            return _, wh3_j
        else:
            return T.concatenate([wh1_j,wh2_j],axis=1), wh3_j
            #return wh2_j,wh3_j

    def read(self,utt_j):
        
        # transformation
        emb_j = tanh( self.Wemb_backup[utt_j,:] )

        # 1st convolution
        wh1_j = self.convolution(emb_j,self.Wcv1_backup,mode='np')
        if self.pool[0]: # max pooling
            wh1_j = max_pool(wh1_j,(3,1),ignore_border=False)
        wh1_j = tanh(wh1_j)
        
        # 2nd convolution
        wh2_j = self.convolution(wh1_j, self.Wcv2_backup,mode='np')
        if self.pool[1]: # max pooling
            wh2_j = max_pool(wh2_j,(3,1),ignore_border=False)
        wh2_j = tanh(wh2_j)
        
        if self.level>=3:
            # 3rd convolution and pooling
            wh3_j = self.convolution(wh2_j, self.Wcv3_backup,mode='np')
            if self.pool[2]: # max pooling
                wh3_j = max_pool(wh3_j,(3,1),ignore_border=False)
            # average pooling
            wh3_j = tanh(np.sum(wh3_j,axis=0))
        else: # level < 3
            wh3_j = None
        
        if self.pool==(True,True,True):
            return None, wh3_j
        else:
            return np.concatenate([wh1_j,wh2_j],axis=1), wh3_j
            #return wh2_j, wh3_j

    def loadConverseParams(self):
        self.Wemb_backup    = self.params[0].get_value()
        self.Wcv1_backup    = self.params[1].get_value()
        self.Wcv2_backup    = self.params[2].get_value()
        if self.level>=3:
            self.Wcv3_backup    = self.params[3].get_value()
   

# LSTM Encoder
class LSTMEncoder(BaseNNModule):

    def __init__(self, vocab_size, ihidden_size):
        
        # parameters for encoder biLSTM
        self.dih = ihidden_size
        self.di  = vocab_size
        self.iWgate = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dih,self.dih*4)).astype(theano.config.floatX))
        self.iUgate = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.dih,self.dih*4)).astype(theano.config.floatX))
        self.Wemb   = theano.shared(0.3 * np.random.uniform(-1.0,1.0,\
                (self.di,self.dih)).astype(theano.config.floatX))
        self.bf = theano.shared(2.0*np.ones((ihidden_size)).\
                astype(theano.config.floatX))

        self.params = [
                self.iWgate, self.iUgate,   self.Wemb,  self.bf ]

        # initial states
        self.ih0 = theano.shared(np.zeros((self.dih),\
                dtype=theano.config.floatX))
        self.ic0 = theano.shared(np.zeros((self.dih),\
                dtype=theano.config.floatX))
        
    # encoder LSTM recurrence
    def recur(self, w_t, ih_tm1, ic_tm1):
        
        in_t = T.nnet.sigmoid( self.Wemb[w_t] )
        
        # syntatic memory cell and gate
        # compute i, f, o, c together and slice it
        bundle_t =  T.dot(in_t,  self.iWgate) +\
                    T.dot(ih_tm1,self.iUgate)
        ig   = T.nnet.sigmoid(bundle_t[:self.dih])
        fg = T.nnet.sigmoid(bundle_t[self.dih:self.dih*2]+self.bf)
        og   = T.nnet.sigmoid(bundle_t[self.dih*2:self.dih*3])
        cx_t = T.tanh(bundle_t[self.dih*3:])
        
        # cell value and hidden layer
        ic_t = ig*cx_t + fg*ic_tm1 
        ih_t = og*T.tanh(ic_t)
        return ih_t, ic_t 
    
    # encoder function
    def encode(self, source_t, scut_t):
        [ihf,icf],_= theano.scan(fn=self.recur,\
            sequences=[source_t],\
            outputs_info=[self.ih0,self.ic0])
        return ihf[scut_t-1,:]

    # function for testing 
    def read(self, s_j):
        
        # hidden layers to be stored
        h = [np.zeros(self.dih)]
        c_tm1 = np.zeros(self.dih)
        # forward encoder
        for w_t in s_j:
            # lstm step
            in_t = sigmoid( self.Wemb_backup[w_t] )
            bundle_t =  np.dot(in_t,  self.iWgate_backup) +\
                        np.dot(h[-1],self.iUgate_backup)
            ig  = sigmoid(bundle_t[:self.dih])
            fg  = sigmoid(bundle_t[self.dih:self.dih*2]+self.bf_backup)
            og  = sigmoid(bundle_t[self.dih*2:self.dih*3])
            cx_t= tanh(bundle_t[self.dih*3:])

            # compute cell activation and hidden layer
            c_t =   np.multiply(ig,cx_t) + \
                    np.multiply(fg,c_tm1)
            h_t =   np.multiply(og,tanh(c_t))
            
            # store current step vectors
            h.append(h_t)
            c_tm1 = c_t
        
        return np.array(h)[-1]
   
    def loadConverseParams(self):
        self.iWgate_backup  = self.params[0].get_value()
        self.iUgate_backup  = self.params[1].get_value()
        self.Wemb_backup    = self.params[2].get_value()
        self.bf_backup      = self.params[3].get_value()


# Bidirectional encoding
def bidirectional_encode(fEncoder,bEncoder,sent,leng):
    fw_intent_t = fEncoder.encode(sent,leng)
    bw_intent_t = bEncoder.encode(T.concatenate(
        [ sent[:leng][::-1],sent[leng:] ],axis=0),leng)
    intent_t = T.concatenate(
            [fw_intent_t,bw_intent_t],axis=0)
    return intent_t

def bidirectional_read(fEncoder, bEncoder, sent):
    fw_intent_t = fEncoder.read(sent)
    bw_intent_t = bEncoder.read(sent[::-1])
    intent_t = np.concatenate([fw_intent_t,bw_intent_t],axis=0)
    return intent_t


