######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

import random
import os
import sys
import operator
import math
import numpy as np
from copy import deepcopy 

#sys.path.append('/home/dial/thw28/research/theano-rnnlm/dialact/')
#from FeatParser import DialogActDelexicalizedParser


def findSubList(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

# TODO: need to modify
def dumpReprs(vocab,wvec):
    for i in range(len(vocab)):
        word = vocab[i][0]
        vec  = [str(x) for x in wvec[i,:].tolist()]
        print '%s %s' % (word,' '.join(vec)) 
    for x in range(len(self.sems)):
        for y in range(len(self.sems[x])):
            if self.sems[x][y] != 'N/A':
                dact = self.sems[x][y]
                idx  = x*self.dcl + y
                vec  = [str(k) for k in self.Wfh_backup[idx,:].tolist()]
                print '%s %s' % (dact,' '.join(vec))

def setWordVector(filename,vocab,wvec):
    fin = file(filename)
    count = 0
    for line in fin.readlines():
        tokens = line.replace('\n','').split()
        word = tokens[0]
        vec = [float(x) for x in tokens[1:]]
        try:
            wordid = vocab.index(word)
            wvec[wordid,:] = np.array(vec)
            count +=1
        except:
            pass
    c = 100*float(count)/float(wvec.shape[0])
    print 'Wordvec coverage : %2.1f%%' % c
    print '==============='
    return wvec
     

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
 
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
 
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]


def loadmask(name,lexicon):

    mask = {}
    slots = {}
    for word, tup in lexicon.iteritems():
        if word.startswith('SLOT_'):
            slots[word] = tup

    fin = file(name)
    for line in fin.readlines():
        tokens = line.replace('\n','').split()
        slot = tokens[0]
        val  = '-'.join(tokens[1:])
        if slots.has_key(slot):
            del slots[slot]
        if lexicon.has_key(slot):
            if mask.has_key(slot):
                mask[lexicon[slot][0]].append(val)
            else:
                mask[lexicon[slot][0]] = [val]
    for word in slots.iterkeys():
        print 'Missing mask for slot %s' % word
    return mask


def loadlex(lex,num_class=1,cutoff=0):
    # load vocab, frequncy binning
    fin = file(lex)
    word2cn = []
    lexicon = {'</s>':(0,0)}
    totalcn = 0
    for line in fin.readlines():
        word, cnt = line.replace('\n','').split('\t')
        if int(cnt)<=cutoff:
            continue
        word2cn.append( [word,int(cnt)] )
        lexicon[word] = (0,0)
        totalcn += int(cnt)
    fin.close()

    if num_class==1:
        bin_size = totalcn
    else:
        bin_size = totalcn/(num_class-1)+1
    accum = bin_size
    classidx = 1
    wordidx = 1
    for word, cn in sorted(word2cn,key=operator.itemgetter(1),\
            reverse=True):
        if classidx>=num_class:
            lexicon[word] = (wordidx,num_class-1)
            wordidx+=1
            continue
        accum+=cn
        lexicon[word] = (wordidx,classidx)
        if( accum>=bin_size*(classidx+1) ):
            classidx += 1
        wordidx+=1
    # append <unk> to make it close vocab
    if not lexicon.has_key('<unk>'):
        lexicon['<unk>'] = (wordidx,num_class-1)
    return lexicon
     

def loaddata(train,valid,test,lexicon):

    # load train, valid, and test data from file 
    train_dat = readfile(train,lexicon)
    valid_dat = readfile(valid,lexicon)
    test_dat  = readfile(test, lexicon)

    return train_dat, valid_dat, test_dat, lexicon


def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [-1] + l + win/2 * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out


def idlize(sent,lexicon, max=-1):
    if not isinstance(sent,list):
        print 'idlize take list as input not string'
        return -1
    sen = []
    #if sent[0]!='</s>':
    #    sen = [lexicon['</s>'][0]]
    for word in sent:
        if lexicon.has_key(word):
            sen.append(lexicon[word][0])
        else:
            sen.append(lexicon['<unk>'][0])
    #if sent[-1]!='</s>':
    #    sen.append(lexicon['</s>'][0])
    #while len(sen)<max:
    #    sen.append(lexicon['</s>'][0])
    return sen 

if __name__ == '__main__':
    s1 = 'indian food and indian takeaway'.split()
    s1x= 'and indian takeaway'.split()
    s2 = 'indian and indian takeaway'.split()
    print levenshtein(s1, s2)   
    print levenshtein(s1x, s2)   

