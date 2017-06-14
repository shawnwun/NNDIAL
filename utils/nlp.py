######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################

import os
import json
import sys
import operator
import re
#import enchant
from tools import levenshtein
sys.path.append('./loader/')
#from DataMasker import DataMasker

fin = file('utils/nlp/mapping.pair')
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n','').split('\t')
    replacements.append((' '+tok_from+' ',' '+tok_to+' '))

def insertSpace(token,text):
    sidx = 0
    while True:
        sidx = text.find(token,sidx)
        if sidx==-1:
            break
        if sidx+1<len(text) and re.match('[0-9]',text[sidx-1]) and \
                re.match('[0-9]',text[sidx+1]):
            sidx += 1
            continue
        if text[sidx-1]!=' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx +=1
        if sidx+len(token)<len(text) and text[sidx+len(token)]!=' ':
            text = text[:sidx+1] + ' ' + text[sidx+1:]
        sidx+=1
    return text

def normalize(text):

    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$','',text)
    
    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0],sidx)
            if text[sidx-1]=='(':
                sidx -= 1
            eidx = text.find(m[-1],sidx)+len(m[-1])
            text = text.replace(text[sidx:eidx],''.join(m))
    
    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m,sidx)
            eidx = sidx + len(m)
            text = text[:sidx]+re.sub('[,\. ]','',m)+text[eidx:]

    # replace st.
    text = text.replace(';',',')
    text = re.sub('$\/','',text)
    text = text.replace('/',' and ')

    # replace other special characters
    text = text.replace('-',' ')
    text = re.sub('[\":\<>@\(\)]','',text)

    # insert white space before and after tokens:
    for token in ['?','.',',','!']:
        text = insertSpace(token,text)
    
    # insert white space for 's
    text = insertSpace('\'s',text)
     
    # replace it's, does't, you'd ... etc
    text = re.sub('^\'','',text)
    text = re.sub('\'$','',text)
    text = re.sub('\'\s',' ',text)
    text = re.sub('\s\'',' ',text)
    for fromx, tox in replacements:
		text = ' '+text+' '
		text = text.replace(fromx,tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +',' ',text)
    
    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i<len(tokens):
        if  re.match(u'^\d+$',tokens[i]) and \
            re.match(u'\d+$',tokens[i-1]):
            tokens[i-1]+=tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)
    
    return text
"""
def spell_correction(words,data):
    d = enchant.Dict("en_US")
    ws = []
    rs = []
    for word,cnt in sorted(words.iteritems(),\
            key=operator.itemgetter(1),reverse=True):
        if cnt<10 and not d.check(word):
            ws.append(word)
        else:
            rs.append(word)
    reps = []
    print len(ws)
    for w in ws:
        if not d.check(w):
            print 
            print '%s\t\t\t%d'%(w,words[w])
            print d.suggest(w)
            input = raw_input('select:\t')
            while True:
                try:
                    if re.match('^[0-9]+$',input):
                        reps.append(d.suggest(w)[int(input)])
                    else:
                        reps.append(input)
                    break
                except:
                    input = raw_input("error, re-enter:\t")
                    pass
        else:
            reps.append(w)
    for i in range(len(data)):
        sent = data[i][1]
        for j in range(len(reps)):
            sent = sent.replace(' '+ws[j]+' ',\
                                ' '+reps[j]+' ')
        data[i][1] = sent
    return data
"""

def compute_words(data):
	lex = {}
	processed = []
	for dact,sent,base in data:
		for word in sent.split():
			if lex.has_key(word):
				lex[word] +=1
			else:
				lex[word] = 1
	return lex,processed

"""
if __name__ == '__main__':
    fin = file(sys.argv[1])
    data = json.load(fin)
    fin.close()
    masker = DataMasker()
    normalised = []
    for obj in data:
        if sys.argv[2]=='raw':
            text = normalize(obj['text'])
            base = normalize(obj['action']['base'])
            slot = [[normalize(e) for e in sv] \
                    for sv in obj['action']['slot']]
            s2vs = []
            for s2v in slot:
                sv = s2v[0]+'='+s2v[1] if len(s2v)>1 else s2v[0]
                s2vs.append(sv)
            dact = obj['action']['act'] +'('+';'.join(s2vs)+')'
            sent = masker.mask(text,dact)
            normalised.append([dact,sent,base])
        else:
            dact,sent,base = obj
            sent = normalize(sent)
            base = normalize(base)
            sent = masker.mask(sent,dact)
            normalised.append([dact,sent,base])
	
    lex,data = compute_words(normalised)
    for w,c in sorted(lex.iteritems(),key=operator.itemgetter(1),\
            reverse=True):
        print w, c
    
    data = spell_correction(lex,data)
    
    fout = file('tmp.dump','w')
    fout.write(json.dumps( data, indent=4, separators=(',', ': ')) )
    fout.close()
"""

if __name__ == '__main__':
    text = 'restaurant one seven'
    text = normalize(text)
    print text
