
def configLearn(random_seed, lr='0.001', lr_divide='2', \
        beta='0.', grad_cutoff='1'):
    text = ('[learn] // parameters for training\n' +\
            'lr             = %s\n'    +\
            'lr_decay       = 0.5\n'   +\
            'stop_count     = %s\n'    +\
            'cur_stop_count = 0\n'     +\
            'l2             = %s\n'    +\
            'random_seed    = %s\n'    +\
            'min_impr       = 1.003\n' +\
            'debug          = True\n'  +\
            'llogp          = -1e7\n' +\
            'grad_clip      = %s\n\n') %\
            (lr,lr_divide,beta,random_seed,grad_cutoff)
    return text 

def configFile(dataset,model,semidict='CamRestHDCSemiDict'):
    text = ('[file] // file paths\n'                    +\
            'db             = db/CamRest.json\n'            +\
            'ontology       = resource/CamRestOTGY.json\n'  +\
            'corpus         = data/%s.json\n'               +\
            'semi           = resource/%s.json\n'           +\
            'model          = %s\n\n') %\
            (dataset,semidict,model)
    return text

def configData(shuffle, lengthen, split='(3,1,1)',percent='100'):
    text = ('[data] // data manipulation, shuffle: dynamic|static\n'+\
            'split          = %s\n'+\
            'percent        = %s\n'+\
            'shuffle        = %s\n'+\
            'lengthen       = %s\n\n'  ) %\
            (split,percent,shuffle,lengthen)
    return text

def configMode(mode):
    text = ('[mode] // training mode: trk|encdec|all\n'+\
            'learn_mode     = %s\n\n') % mode
    return text
        
def configN2N(enc,dec,trk):
    text = ('[n2n] // components of network\n'+\
            'encoder        = %s\n'+\
            'tracker        = %s\n'+\
            'decoder        = %s\n\n') %\
            (enc,trk,dec)
    return text

def configEnc(h):
    text = ('[enc] // structure of encoder\n'+\
            'ihidden        = %s\n\n') % (h)
    return text

def configTrk(inf,req,bef,trkenc,wvec='none'):
    text = ('[trk] // structure of tracker summary|full, cnn|ng\n'+\
            'informable     = %s\n'+\
            'requestable    = %s\n'+\
            'belief         = %s\n'+\
            'trkenc         = %s\n'+\
            'wvec           = %s\n\n') %\
            (inf,req,bef,trkenc,wvec)
    return text

def configDec(h,struct,sshot,wvec='none'):
    text = ('[dec] // structure of decoder\n'+\
            'ohidden        = %s\n'+\
            'struct         = %s\n'+\
            'snapshot       = %s\n'+\
            'wvec           = %s\n\n') %\
            (h,struct,sshot,wvec)
    return text

def configPly(policy,latent):
    text = ('[ply] // structure of policy network\n'+\
            'policy         = %s\n'+\
            'latent         = %s\n\n') %\
            (policy,latent)
    return text
 
def configGen(token_reward, repeat_penalty, alpha='0.0', 
                verbose='2', topk='1',beamwidth='10'):
    text = ('[gen] // generation, repeat penalty: inf|none\n'+\
            'alpha          = %s\n'+\
            'verbose        = %s\n'+\
            'topk           = %s\n'+\
            'beamwidth      = %s\n'+\
            'repeat_penalty = %s\n'+\
            'token_reward   = %s\n\n') %\
            (alpha,verbose,topk,beamwidth,repeat_penalty,token_reward)
    return text

if __name__ == '__main__':
    print configEnc('50')

