######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

import math
from collections import Counter
from nltk.util import ngrams
#from nltk.align.bleu import BLEU

def sentence_bleu_4(hyp,refs,weights=[0.25,0.25,0.25,0.25]):
    # input : single sentence, multiple references
    count = [0,0,0,0]
    clip_count = [0,0,0,0]
    r = 0
    c = 0
        
    for i in range(4):
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
        
    bestmatch = [1000,1000]
    for ref in refs:
        if bestmatch[0]==0:
            break
        diff = abs(len(ref)-len(hyp))
        if diff<bestmatch[0]:
            bestmatch[0] = diff
            bestmatch[1] = len(ref)
    r = bestmatch[1] 
    c = len(hyp)

    p0 = 1e-7
    bp = math.exp(-abs(1.0-float(r)/float(c+p0)))

    p_ns = [float(clip_count[i])/float(count[i]+p0)+p0 for i in range(4)]
    s = math.fsum(w*math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
    bleu_hyp = bp*math.exp(s)
    """
    print 
    print hyp
    print refs

    print clip_count
    print count
    print r, c
    print bp
    print p_ns
    print math.exp(s)
    print bleu_hyp
    raw_input()
    """
    return bleu_hyp

def corupus_bleu_4(refhypbase,weights=[0.25,0.25,0.25,0.25]):
    # list of tuple lists: (ref,hyp,base,dact)
    # each of them with multiple candidates 
    count = [0,0,0,0]
    bcount= [0,0,0,0]
    clip_count = [0,0,0,0]
    bclip_count= [0,0,0,0]
    r = 0
    c = 0
    rb= 0
    b = 0
    for refs,hyps,bases,dact in refhypbase:
        
        for hyp in hyps:
            for i in range(4):
                cnt = sum(Counter(ngrams(hyp, i+1)).values())
                count[i] += cnt
                clipcnt = BLEU.modified_precision(hyp,refs,i+1)*cnt
                clip_count[i] += clipcnt
                
            bestmatch = [1000,1000]
            for ref in refs:
                if bestmatch[0]==0:
                    break
                diff = abs(len(ref)-len(hyp))
                if diff<bestmatch[0]:
                    bestmatch[0] = diff
                    bestmatch[1] = len(ref)
            r += bestmatch[1] 
            c += len(hyp)
        
        for base in bases:
            
            # special case for base bleu score
            if base==['goodbye','.']:
                for i in range(4):
                    bcount[i] += 1.0
                    bclip_count[i] += 1.0
            else: 
                for i in range(4):
                    cnt = sum(Counter(ngrams(base, i+1)).values())
                    bcount[i] += cnt
                    clipcnt = BLEU.modified_precision(base,refs,i+1)*cnt
                    bclip_count[i] += clipcnt
                
            bestmatch = [1000,1000]
            for ref in refs:
                if bestmatch[0]==0:
                    break
                diff = abs(len(ref)-len(base))
                if diff<bestmatch[0]:
                    bestmatch[0] = diff
                    bestmatch[1] = len(ref)
            rb+= bestmatch[1] 
            b += len(base) 

    p0 = 1e-7

    bp = 1 if c>r else math.exp(1-float(r)/float(c))
    p_ns = [float(clip_count[i])/float(count[i]+p0)+p0 for i in range(4)]
    s = math.fsum(w*math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
    bleu_hyp = bp*math.exp(s)
    
    bp = 1 if b>rb else math.exp(1-float(rb)/float(b))
    p_ns = [float(bclip_count[i])/float(bcount[i]+p0)+p0 for i in range(4)]
    s = math.fsum(w*math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
    bleu_base = bp*math.exp(s)
   
    return bleu_hyp,bleu_base

if __name__ == '__main__':
    
    s1 = 'red door cafe is in the cathedral hill and does not allow kids and is good for breakfast , yes , does not allow kids and is good for a meal that allow kids .'.split()
    s2 = 'red door cafe is good for breakfast and does not allow children in the cathedral hill area .'.split()
    ref = 'red door cafe is in the area of cathedral hill and serves breakfast but does not allow kids .'.split()
    
    print BLEU.compute(s1,[ref],[0.25,0.25,0.25,0.25])
    print BLEU.compute(s2,[ref],[0.25,0.25,0.25,0.25])
