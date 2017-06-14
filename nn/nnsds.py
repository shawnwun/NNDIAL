######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import theano
import theano.tensor as T
import sys
from collections import OrderedDict
from utils.updates import *
from theano.ifelse import ifelse
import theano.gradient as G
from copy import deepcopy

from basic   import *
from encoder import *
from tracker import *
from decoder import *
from policy  import *
from utils.adam import adam


class NNSDS(BaseNNModule):

    def __init__(self, enc, dec, ply, trk, inf, req, bef, 
            trkenc, use_snap , decstruct, voc_size,
            ih_size, oh_size, inf_size, req_size, gradcut='1', 
            learn_mode='all', snap_size=0, latent_size=1):
        
        print 'init n2n SDS ...'
        # memorise some params for future use
        self.dl     = latent_size
        self.db     = 0
        self.dsp    = snap_size
        self.iseg   = inf_size
        self.rseg   = req_size
        self.enc    = enc
        self.dec    = dec
        self.trk    = trk
        self.inf    = inf
        self.req    = req
        self.bef    = bef
        self.trkenc = trkenc
        self.ply    = ply
        self.learn_mode = learn_mode
        self.gradcut= gradcut
        self.decstruct = decstruct
        self.use_snap = use_snap

        # trainable parameters
        self.params = {
            'enc':      [],
            'inftrk':   [],
            'reqtrk':   [],
            'dec':      [],
            'ply':      []
        }

        ##############################################################
        # init encoder
        if enc=='lstm':
            print '\tinit lstm encoder ...'
            self.fEncoder = LSTMEncoder( voc_size, ih_size)
            self.bEncoder = LSTMEncoder( voc_size, ih_size)
            self.params['enc'].extend(self.fEncoder.params+self.bEncoder.params)
        
        ##############################################################
        # init requestable tracker
        if trk=='rnn' and req==True: # track requestable slots
            print '\tinit rnn requestable trackers ...'
            self.reqtrackers = []
            for i in range(len(req_size)-1):
                if trkenc=='cnn':# taking cnn features
                    t = CNNRequestableTracker(
                        voc_size,ih_size,voc_size,oh_size)
                self.reqtrackers.append(t)
                self.params['reqtrk'].extend(t.params)
            
            # offer change tracker
            print '\tinit OfferChange tracker ...'
            self.changeTracker = CNNRequestableTracker(
                    voc_size,ih_size,voc_size,oh_size)
            self.params['reqtrk'].extend(self.changeTracker.params)
       
        # init informable tracker
        if trk=='rnn' and inf==True: # track informable slots  
            print '\tinit rnn informable trackers ...'
            self.infotrackers= []
            for i in range(len(inf_size)-1):
                b_size = inf_size[i+1]-inf_size[i]
                if trkenc=='cnn':# taking cnn features
                    t = CNNInformableTracker(b_size,
                        voc_size,ih_size,voc_size,oh_size)
                self.infotrackers.append(t)
                self.params['inftrk'].extend(t.params)
        
        ##############################################################
        # init policy network
        belief_size = computeBeleifDim(trk, inf, req, bef, self.iseg, self.rseg)
        if self.ply=='attention':
            print '\tinit attentive policy network ...'
            self.policy = AttentivePolicy( belief_size, 6, ih_size, oh_size )
        elif self.ply=='normal':
            print '\tinit normal policy network ...'
            self.policy = Policy( belief_size, 6, ih_size, oh_size )
        elif self.ply=='latent':
            print '\tinit latent policy network ...'
            self.policy = LatentPolicy( 
                    latent_size, learn_mode, belief_size, 6, ih_size, oh_size, 
                    LSTMEncoder(voc_size, ih_size), LSTMEncoder(voc_size, ih_size),        
                    LSTMEncoder(voc_size, ih_size), LSTMEncoder(voc_size, ih_size))
        
        self.params['ply'].extend(self.policy.params) 
         
        ##############################################################
        # init decoder
        if dec=='lstm':
            print '\tinit lstm decoder ...'
            # init decoder, select decoder type
            self.decoder = Decoder( self.policy, voc_size, oh_size, 
                    self.decstruct, self.ply, self.use_snap, self.dsp)
            self.params['dec'].extend(self.decoder.params)
        
    def config_theano(self):
        ##################################################################
        ########################### NOT USING NOW ########################
        ##################################################################
        # snapshot and change
        snapshot    = T.itensor3('snapshot')
        change_label= T.fmatrix('change_label')
        
        ##################################################################
        ##################################################################
        ##################################################################
        # trade-off hyperparameters
        _lambda = 0.1
        _alpha  = 0.1
        _avgLen = 20.

        # regularization and learning rate
        lr   = T.scalar('lr')
        reg  = T.scalar('reg')
        beta = T.scalar('beta')

        # semantics
        inf_trk_labels = T.fmatrix('inf_trk_labels')
        req_trk_labels = T.fmatrix('req_trk_labels')
        
        # DB matching degree
        db_degrees  = T.fmatrix('db_degrees')

        # source and target utts
        source      = T.imatrix('source')
        target      = T.imatrix('target')
        source_len  = T.ivector('source_len')
        target_len  = T.ivector('target_len')
        utt_group   = T.ivector('utt_group')
        
        # masked source and target utts
        masked_source       = T.imatrix('masked_source')
        masked_target       = T.imatrix('masked_target')
        masked_source_len   = T.ivector('masked_source_len')
        masked_target_len   = T.ivector('masked_target_len')
        
        # tracker features, either n-grams or delexicalised position
        srcfeat = T.itensor4('srcfeat')
        tarfeat = T.itensor4('tarfeat')

        # external samples
        success_rewards = T.fvector('success_reward')
        samples = T.ivector('samples')

        # for numerical stability
        epsln = 1e-10

        # dialog level recurrence
        def dialog_recur(source_t, target_t, source_len_t, target_len_t,
                    masked_source_t, masked_target_t, 
                    masked_source_len_t, masked_target_len_t,
                    utt_group_t, snapshot_t, success_reward_t, sample_t,
                    change_label_t, db_degree_t, 
                    inf_label_t, req_label_t, source_feat_t, target_feat_t, 
                    belief_tm1, masked_target_tm1, masked_target_len_tm1, 
                    target_feat_tm1, posterior_tm1): 
            
            ##############################################################
            ##################### Intent encoder #########################
            ##############################################################
            # Intent encoder
            if self.enc=='lstm':
                masked_intent_t = bidirectional_encode(
                        self.fEncoder,self.bEncoder,
                        masked_source_t,masked_source_len_t)
            
            ##############################################################
            ########## Belief tracker, informable + requestable ##########
            ##############################################################
            # cost placeholder for accumulation
            print '\tloss function'
            loss_t          = theano.shared(np.zeros((1),dtype=theano.config.floatX))[0]
            companion_loss_t= theano.shared(np.zeros((1),dtype=theano.config.floatX))[0]
            prior_loss_t    = theano.shared(np.zeros((1),dtype=theano.config.floatX))[0]
            posterior_loss_t= theano.shared(np.zeros((1),dtype=theano.config.floatX))[0]
            base_loss_t     = theano.shared(np.zeros((1),dtype=theano.config.floatX))[0]
            # other information to store
            dtmp = 1 #if self.vae_train=='sample' else self.dl
            reward_t        = theano.shared(np.zeros((dtmp),dtype=theano.config.floatX))
            baseline_t      = theano.shared(np.zeros((1),dtype=theano.config.floatX))[0]
            posterior_t     = theano.shared(np.zeros((self.dl),dtype=theano.config.floatX))[0]
            
            # Informable slot belief tracker
            # belief vector
            belief_t = []

            if self.trk=='rnn' and self.inf==True:
                for i in range(len(self.infotrackers)):
                    # slice the current belief tracker output
                    cur_belief_tm1  = belief_tm1[self.iseg[i]:self.iseg[i+1]]
                    if self.trkenc=='cnn': # cnn, position features
                        ssrcpos_js  = source_feat_t[0,self.iseg[i]:self.iseg[i+1],:]
                        vsrcpos_js  = source_feat_t[1,self.iseg[i]:self.iseg[i+1],:]
                        starpos_jm1s= target_feat_tm1[0,self.iseg[i]:self.iseg[i+1],:]
                        vtarpos_jm1s= target_feat_tm1[1,self.iseg[i]:self.iseg[i+1],:]
                    
                        # tracking 
                        cur_belief_t = self.infotrackers[i].recur( cur_belief_tm1, 
                                masked_source_t, masked_target_tm1,
                                masked_source_len_t, masked_target_len_tm1,
                                ssrcpos_js, vsrcpos_js, starpos_jm1s, vtarpos_jm1s)
                
                    # semi label
                    cur_label_t = inf_label_t[self.iseg[i]:self.iseg[i+1]]
                    # include cost if training tracker
                    if self.learn_mode=='all' or self.learn_mode=='trk':
                        print '\t\tincluding informable  tracker loss ...'
                        loss_t += -T.sum( cur_label_t*T.log10(cur_belief_t+epsln) )

                    # accumulate belief vector
                    if self.bef=='full':
                        belief_t.append(cur_label_t)
                    else:
                        # summary belief
                        tmp  = [T.sum(  cur_label_t[:-2],axis=0).dimshuffle('x'),\
                                        cur_label_t[-2].dimshuffle('x')]
                        tmp  = tmp + [cur_label_t[-1].dimshuffle('x')] if\
                                self.bef=='summary' else tmp
                        cur_sum_belief_t = T.concatenate( tmp,axis=0 )
                        belief_t.append(cur_sum_belief_t)
            
            inf_belief_t = inf_label_t

            # Requestable slot belief tracker
            if self.trk=='rnn' and self.req==True:
                for i in range(len(self.rseg)-1):
                    # current feature index
                    bn = self.iseg[-1]+2*i
                    if self.trkenc=='cnn': # cnn, position features
                        ssrcpos_js  = source_feat_t[0,bn,:]
                        vsrcpos_js  = source_feat_t[1,bn,:]
                        starpos_jm1s= target_feat_tm1[0,bn,:]
                        vtarpos_jm1s= target_feat_tm1[1,bn,:]
                        # tracking 
                        cur_belief_t = self.reqtrackers[i].recur(
                            masked_source_t, masked_target_tm1,
                            masked_source_len_t, masked_target_len_tm1,
                            ssrcpos_js,vsrcpos_js,starpos_jm1s,vtarpos_jm1s )
                    
                    # semi label
                    cur_label_t = req_label_t[2*i:2*(i+1)]
                    # include cost if training tracker
                    if self.learn_mode=='all' or self.learn_mode=='trk':
                        print '\t\tincluding requestable tracker loss ...'
                        loss_t += -T.sum( cur_label_t*T.log10(cur_belief_t+epsln) )
                    # accumulate belief vector
                    if self.bef=='full':
                        belief_t.append(cur_label_t)
                    else:
                        tmp = cur_label_t if self.bef=='summary' else cur_label_t[:1]
                        belief_t.append(tmp)
            
                # offer-change tracker
                minus1 = -T.ones((1),dtype='int32')
                cur_belief_t = self.changeTracker.recur(
                        masked_source_t, masked_target_tm1,
                        masked_source_len_t, masked_target_len_tm1,
                        minus1, minus1, minus1, minus1)
                # cost function
                if self.learn_mode=='trk' or self.learn_mode=='all': 
                    print '\t\tincluding OfferChange tracker loss ...'
                    loss_t += -T.sum( change_label_t*T.log10(cur_belief_t+epsln) )
                # accumulate belief vector
                if self.bef=='full':
                    belief_t.append(change_label_t)
                else:
                    tmp = change_label_t[:1] if self.bef=='simplified' \
                            else change_label_t
                    belief_t.append(tmp)

            ##############################################################
            ######################## LSTM decoder ########################
            ##############################################################
            bef_t = T.concatenate(belief_t,axis=0)
            # LSTM decoder
            if self.dec=='lstm' and self.learn_mode!='trk':
                prob_t, snapCost_t, prior_t, posterior_t, z_t, base_t, debugX = \
                    self.decoder.decode( 
                        masked_source_t, masked_source_len_t,
                        masked_target_t, masked_target_len_t,
                        masked_intent_t, belief_t, db_degree_t[-6:],
                        utt_group_t, snapshot_t, sample_t)
                debug_t = prior_t
               
                # decoder loss 
                if self.ply!='latent': # deterministic policy 
                    print '\t\tincluding decoder loss ...'
                    loss_t += -T.sum(T.log10(prob_t+epsln))
                else: # variational policy
                    # disconnet gradient flow
                    P = G.disconnected_grad(prior_t)
                    Q = G.disconnected_grad(posterior_t)
                    Qtm1 = G.disconnected_grad(posterior_tm1)
                    
                    # prior network loss
                    if self.learn_mode=='rl': # rl fine-tuning
                        print '\t\tincluding RL success reward for fine-tine policy ...'
                        prior_loss_t = -success_reward_t*T.log10(prior_t+epsln)[z_t]
                    else: # neural variational inference 
                        # encoder loss, minimising KL(Q|P) and self-supervised action
                        print '\t\tinclding KL(Q|Pi) to train policy network Pi ...'
                        prior_loss_t = -T.switch( T.lt(utt_group_t,self.dl-1),
                            T.log10(prior_t+epsln)[z_t],
                            _alpha*T.sum( Q*(T.log10(prior_t+epsln)-T.log10(Q+epsln)))  
                        )

                        # decoder loss for current sample/ground truth
                        print '\t\tincluding decoder loss ...' 
                        loss_t = -T.sum(T.log10(prob_t+epsln))
                        
                        # define reward function for Q
                        print '\t\tincluding reinforce loss to train inference network Q ...'
                        r_t =  G.disconnected_grad(
                            _avgLen*T.mean(T.log10(prob_t+epsln))+ # decoder loglikelihood
                            -_lambda*T.sum(Q*(T.log10(Q+epsln)-T.log10(P+epsln)))+ # KL(P|Q)
                            -_lambda*T.sum(Qtm1*(T.log10(Qtm1+epsln)-T.log10(Q+epsln))) # KL(Qt|Qtm1)
                        )
                        
                        # actual reward after deducting baseline
                        reward_t= G.disconnected_grad( r_t-base_t )
                        baseline_t = base_t
                        #debug_t = r_t-base_t
                        
                        # Q network loss: reinforce objective 
                        posterior_loss_t = -T.switch( T.lt(utt_group_t,self.dl-1),
                            T.log10(posterior_t+epsln)[z_t],# self-sup
                            _alpha*reward_t*T.log10(posterior_t+epsln)[z_t] # reinforce
                        )

                        # baseline loss
                        print '\t\tincluding baseline loss ...'
                        base_loss_t = T.switch( T.lt(utt_group_t,self.dl-1),
                                0., (r_t-baseline_t)**2)
                    
                # snapshot objective
                if self.use_snap:
                    print'\t\tincluding decoder snapshot loss ...'
                    companion_loss_t += -T.sum(snapCost_t[:masked_target_len_t-1])

            # dummy, TODO: change it
            if self.ply!='latent': 
                posterior_t = posterior_tm1
                z_t = posterior_tm1
                reward_t = posterior_tm1
                prior_t = posterior_tm1
                debug_t = posterior_tm1
                        
            # take the semi label for next input - like LM
            return inf_belief_t, masked_target_t, masked_target_len_t, \
                    target_feat_t, posterior_t, z_t,\
                    loss_t, companion_loss_t, prior_loss_t, posterior_loss_t, base_loss_t,\
                    reward_t, baseline_t, debug_t

        # initial belief state
        belief_0 = T.zeros((self.iseg[-1]),dtype=theano.config.floatX)
        belief_0 = T.set_subtensor(belief_0[[x-1 for x in self.iseg[1:]]],1.0)
        # initial target jm1
        masked_target_tm1    = T.ones_like(masked_target[0])
        masked_target_len_tm1= T.ones_like(masked_target_len[0])
        # initial target jm1 position features
        tarfeat_tm1 = -T.ones_like(tarfeat[0])
        # initial posterior
        p0 = np.ones((self.dl))/float(self.dl)
        posterior_0 = theano.shared(p0.astype(theano.config.floatX))

        # Dialogue level forward propagation
        [_,_,_,_,posterior,sample,loss,companion_loss,prior_loss,posterior_loss,base_loss,
                reward,baseline,debug], updates= \
                theano.scan( fn=dialog_recur,
                sequences=[source,target,source_len,target_len,
                        masked_source,masked_target,
                        masked_source_len,masked_target_len,
                        utt_group, snapshot, success_rewards, samples,
                        change_label, db_degrees,
                        inf_trk_labels, req_trk_labels,
                        srcfeat, tarfeat],\
                outputs_info=[belief_0,masked_target_tm1,masked_target_len_tm1,tarfeat_tm1,
                        posterior_0,None,None,None,None,None,None,None,None,None])
        
        # Theano validation function
        self.valid = theano.function(
                inputs=[source, target, source_len, target_len,
                        masked_source, masked_target, 
                        masked_source_len, masked_target_len,
                        utt_group, snapshot, success_rewards, samples, 
                        change_label, inf_trk_labels, req_trk_labels, 
                        db_degrees, srcfeat, tarfeat],\
                outputs=[loss,prior_loss,posterior],\
                updates=updates,\
                on_unused_input='warn')
      
        # RL validation function
        self.validRL = theano.function(
                inputs=[source, target, source_len, target_len,
                        masked_source, masked_target, 
                        masked_source_len, masked_target_len,
                        utt_group, snapshot, success_rewards, samples, 
                        change_label, inf_trk_labels, req_trk_labels, 
                        db_degrees, srcfeat, tarfeat],\
                outputs=[prior_loss, debug],\
                updates=updates,\
                on_unused_input='warn')
      
        # for deterministic case, just loglikelihood
        if self.ply=='attention' or self.ply=='normal':

            # flatten parameters
            self.flatten_params = []
            for k in ['inftrk','reqtrk','dec','ply','enc']:
                ws = self.params[k]
                if self.learn_mode=='all': 
                    # train whole model
                    print '\tgradient w.r.t %s' % (k)
                    self.flatten_params += ws
                elif self.learn_mode=='trk' and 'trk' in k: 
                    # pretrain tracker
                    print '\tgradient w.r.t %s' % (k)
                    self.flatten_params += ws
                elif self.learn_mode=='encdec':
                    # train * apart from tracker
                    if 'trk' in k:  continue # tracker
                    else:           
                        print '\tgradient w.r.t %s' % (k)
                        self.flatten_params += ws
            
            # loss function
            self.cost = T.sum(loss) + 0.1*T.sum(companion_loss) 
            # gradients and updates
            updates = adam(self.cost, self.flatten_params, lr=lr, reg=reg)
            # default value for function output
            prior_loss = posterior_loss = baseline_loss = self.cost

        # for NVI
        elif self.ply=='latent':
           
            # flatten parameters
            self.flatten_params = []
            for k in ['ply','enc','dec']:
                # train encoder decoder
                if self.learn_mode=='encdec':
                    print '\tgradient w.r.t %s' %(k)
                    self.flatten_params += self.params[k] 
                # fine-tune policy network by RL
                elif self.learn_mode=='rl':
                    if k=='ply':
                        print '\tgradient w.r.t %s prior network' %(k)
                        self.flatten_params += self.params[k][7:10]

            # loss function
            if self.learn_mode=='rl':
                self.cost = T.sum(prior_loss)
            elif self.learn_mode=='encdec':
                self.cost = T.sum(loss) + 0.1*T.sum(companion_loss) +\
                            T.sum(prior_loss) + T.sum(posterior_loss)
            # gradients and updates
            for p,q in adam(self.cost, self.flatten_params, lr=lr, reg=reg):
                updates.update({ p : q })

            if self.learn_mode=='encdec':
                # baseline objective
                for p,q in adam(T.sum(base_loss), self.policy.baseline.params, 
                        lr=lr*10., reg=0.):
                    updates.update({ p : q })
                self.flatten_params.extend(self.policy.baseline.params)
        
        # theano training function
        self.train = theano.function(
                inputs= [source, target, source_len, target_len,
                        masked_source, masked_target, 
                        masked_source_len, masked_target_len,
                        utt_group, snapshot, success_rewards, samples, 
                        change_label, inf_trk_labels, req_trk_labels, 
                        db_degrees, srcfeat, tarfeat, lr, reg],\
                outputs=[loss,prior_loss,posterior_loss,base_loss,
                        posterior,sample,reward,baseline,debug],\
                updates=updates,\
                on_unused_input='warn')

        # RL training function
        self.trainRL = theano.function(
                inputs= [source, target, source_len, target_len,
                        masked_source, masked_target, 
                        masked_source_len, masked_target_len,
                        utt_group, snapshot, success_rewards, samples, 
                        change_label, inf_trk_labels, req_trk_labels, 
                        db_degrees, srcfeat, tarfeat, lr, reg],\
                outputs=[prior_loss,sample, debug],\
                updates=updates,\
                on_unused_input='warn')
            
    # function for testing
    def read(self, masked_source_t):
        ##############################################################
        # encoder
        if self.enc=='lstm':
            masked_intent_t = bidirectional_read(
                    self.fEncoder, self.bEncoder, masked_source_t)
        else:
            masked_intent_t = []
        return masked_intent_t

    def track(self, belief_tm1, masked_source_t, masked_target_tm1, 
            srcfeat_t, tarfeat_tm1):
        
        ##############################################################
        # Informable slot belief tracker
        # belief vector
        belief_t = []
        # full_belief
        full_belief_t = []
        if self.trk=='rnn' and self.inf==True:
            for i in range(len(self.infotrackers)):
                # belief
                cur_belief_tm1  = belief_tm1[self.iseg[i]:self.iseg[i+1]]
                if self.trkenc=='cnn':
                    # cnn encoded tracker, pos features
                    ssrcpos_ts  = srcfeat_t[0][self.iseg[i]:self.iseg[i+1]]
                    vsrcpos_ts  = srcfeat_t[1][self.iseg[i]:self.iseg[i+1]]
                    starpos_tm1s= tarfeat_tm1[0][self.iseg[i]:self.iseg[i+1]]
                    vtarpos_tm1s= tarfeat_tm1[1][self.iseg[i]:self.iseg[i+1]]

                    # tracking 
                    cur_belief_t = self.infotrackers[i].track(cur_belief_tm1, 
                            masked_source_t, masked_target_tm1,
                            ssrcpos_ts, vsrcpos_ts, starpos_tm1s, vtarpos_tm1s)
                
                # accumulating belief states
                full_belief_t.append(cur_belief_t)
                if self.bef=='full':
                    belief_t.append(cur_belief_t)
                else:
                    tmp = [np.sum(cur_belief_t[:-2],axis=0),cur_belief_t[-2]]
                    tmp = tmp + [cur_belief_t[-1]] if self.bef=='summary' else tmp
                    belief_t.append( np.array(tmp) )
        

        # Requestable slot belief tracker
        if self.trk=='rnn' and self.req==True:
            for i in range(len(self.reqtrackers)):
                # current step feature
                
                bn = self.iseg[-1]+2*i
                if self.trkenc=='cnn': # cnn, position features
                    ssrcpos_ts  = srcfeat_t[0][bn]
                    vsrcpos_ts  = srcfeat_t[1][bn]
                    starpos_tm1s= tarfeat_tm1[0][bn]
                    vtarpos_tm1s= tarfeat_tm1[1][bn]
                
                    # tracking 
                    cur_belief_t = self.reqtrackers[i].track(
                            masked_source_t, masked_target_tm1,
                            ssrcpos_ts,vsrcpos_ts,starpos_tm1s,vtarpos_tm1s)
                
                # accumulating belief states
                full_belief_t.append(cur_belief_t)
                if self.bef=='full':
                    belief_t.append(cur_belief_t)
                else:
                    tmp = cur_belief_t if self.bef=='summary' else cur_belief_t[:1]
                    belief_t.append( tmp )
            
            # offer change tracker
            cur_belief_t = self.changeTracker.track(
                    masked_source_t, masked_target_tm1,
                    [-1],[-1],[-1],[-1])
            full_belief_t.append(cur_belief_t)
            tmp = cur_belief_t[:1] if self.bef=='simplified' else cur_belief_t
            belief_t.append(tmp)
        
        return full_belief_t, belief_t

    def talk(self, masked_intent_t, belief_t, degree_t, 
            masked_source_t=None, masked_target_t=None, 
            scoreTable=None, forced_sample=None):
        
        ##############################################################
        # decoding
        if self.dec=='lstm' and self.learn_mode!='trk': 
            if self.trk!='none': # include tracker
                responses, sample, prob = self.decoder.talk( 
                        masked_intent_t,belief_t,degree_t[-6:], 
                        masked_source_t, masked_target_t, 
                        scoreTable, forced_sample) 
        else: # no decoder
            responses,sample,prob = [[[],[],[]]], [], None
        return responses, sample, prob
    
    
    def setParams(self,update_params):
        for k in self.params.iterkeys():
            # number of parameters are not the same => use a different model, re-init
            for i in range(len(self.params[k])):
                try:
                    # if parameter size is not the same , skip it
                    if self.params[k][i].get_value().shape!= update_params[k][i].shape:
                        print '\t\tshape mismatch, re-initialise Matrix ... (%s,%d)' % (k,i)
                        continue
                    # try to set the parameters, if fail skip it
                    self.params[k][i].set_value(update_params[k][i])
                except: print '\t\tre-initialise Matrix ... (%s,%d)' % (k,i)
    
    def getParams(self):
        return_params = {}
        for k, ws in self.params.iteritems():
            return_params[k] = [p.get_value() for p in ws]
        return return_params

    def numOfParams(self):
        num = 0
        for k, ws in self.params.iteritems():
            num += sum([p.get_value().size for p in ws])
        return num, sum([p.get_value().size for p in self.flatten_params])

    def loadConverseParams(self):

        ##############################################################
        # intent encoder
        if self.enc=='lstm' or self.enc=='cnn':
            self.fEncoder.loadConverseParams()
            self.bEncoder.loadConverseParams()

        ##############################################################
        # beleif tracker
        if self.trk=='rnn' and self.req==True:
            for t in self.reqtrackers:  t.loadConverseParams()

        if self.trk=='rnn' and self.inf==True:
            for t in self.infotrackers: t.loadConverseParams()
            self.changeTracker.loadConverseParams()
       
        ##############################################################
        # policy
        self.policy.loadConverseParams()

        ##############################################################
        # decoder
        if self.dec=='lstm':
            self.decoder.loadConverseParams()
    
    # for speed up external sampling
    def genPseudoBelief(self, inf_trk_labels, req_trk_labels, change):
        belief_t = []
        full_belief_t = []
        # informable trackers
        for i in range(len(self.infotrackers)):
            cur_belief_t = inf_trk_labels[self.iseg[i]:self.iseg[i+1]]
            full_belief_t.append(cur_belief_t)
            if self.bef=='full':
                belief_t.append(cur_belief_t)
            else:
                tmp = [np.sum(cur_belief_t[:-2],axis=0),cur_belief_t[-2]]
                tmp = tmp + [cur_belief_t[-1]] if self.bef=='summary' else tmp
                belief_t.append( np.array(tmp) )
        # requestable trackers
        for i in range(len(self.reqtrackers)): 
            bn = self.iseg[-1]+2*i
            cur_belief_t = req_trk_labels[2*i:2*(i+1)]
            full_belief_t.append(cur_belief_t)
            if self.bef=='full':
                belief_t.append(cur_belief_t)
            else:
                tmp = cur_belief_t if self.bef=='summary' else cur_belief_t[:1]
                belief_t.append( tmp )
        # change tracker
        cur_belief_t = change
        full_belief_t.append(cur_belief_t)
        tmp = cur_belief_t[:1] if self.bef=='simplified' else cur_belief_t
        belief_t.append(tmp)
        
        return full_belief_t, belief_t

       
