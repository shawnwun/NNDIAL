######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
import os
import json
import sys
import math
import operator

from collections import Counter
from nltk.util import ngrams
#from nltk.align.bleu import BLEU


class ERRScorer():

    ## Scorer for calculating the slot errors
    ## it scores utterances one by one
    ## using two levels of matching 
    ## 1. exact match for categorical values
    ## 2. multiple keyword matching for binary values
    ## 3. cannot deal with don't care values
    def __init__(self,detectfile):

        self.detectPairs = []
        fin = file(detectfile)
        self.detectPairs = json.load(fin)
        fin.close()

    def countSlots(self,dataset,reader):
        count = 0
        for t in dataset:
            feat = reader.hardformatter.format(t[0])[0]
            for f in feat:
                if '=VALUE' in f or '=none' in f \
                    or '=yes' in f or '=no' in f:
                    count +=1
        return count

    def score(self,feat,gen):
        exact_slot_error = 0
        apprx_slot_error = 0
        
        for f,tok in self.detectPairs['general'].iteritems():
            f = f+'=VALUE'
            fcnt = feat.count(f)
            gcnt = gen.count(tok)
            if fcnt!=gcnt:  exact_slot_error += 1
        for f,toks in self.detectPairs['binary'].iteritems():
            fcnt =  feat.count(f+'=yes')+feat.count(f+'=no')+\
                    feat.count(f+'=none')+feat.count(f+'=dontcare')
            gcnt = sum([gen.count(tok) for tok in toks]) 
            if fcnt<gcnt:   apprx_slot_error += 1
            elif fcnt>gcnt and '?select' not in feat[0] and \
                'suggest' not in feat[0]:
                apprx_slot_error += 1
        total_slot_error = exact_slot_error + apprx_slot_error
        return total_slot_error, exact_slot_error

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass
    def score(self,parallel_corpus):
        
        # containers
        count = [0,0,0,0]
        clip_count = [0,0,0,0]
        r = 0
        c = 0
        weights=[0.25,0.25,0.25,0.25]

        # accumulate ngram statistics
        for hyps,refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:
                
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i+1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts 
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i+1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0),refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                            for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000,1000]
                for ref in refs:
                    if bestmatch[0]==0: break
                    diff = abs(len(ref)-len(hyp))
                    if diff<bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c>r else math.exp(1-float(r)/float(c))
        p_ns = [float(clip_count[i])/float(count[i]+p0)+p0 \
                for i in range(4)]
        s = math.fsum(w*math.log(p_n) \
                for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp*math.exp(s)
        return bleu

class GentScorer(object):
    ## main Scorer interfaces for all scorers
    ## it can do 
    ## 1. Compute bleu score
    ## 2. Compute slot error rate
    ## 3. Detailed illustraction of how differet split 
    ##    of data affect performance
    def __init__(self,detectfile):
        self.errscorer = ERRScorer(detectfile)
        self.bleuscorer= BLEUScorer()

    def scoreERR( self,feat,gen):
        return self.errscorer.score(feat,gen)
        
    def countSlots(self,dataset,reader):
        return self.errscorer.countSlots(dataset,reader)

    def scoreBLEU(self,parallel_corpus):
        return self.bleuscorer.score(parallel_corpus)


