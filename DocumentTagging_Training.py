#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, string, random, numpy, re
import nltk
from optparse import OptionParser
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from utils.llda import LLDA
import pandas as pd
import numpy as np
from Reuters import *
from utils.TextUtils import *

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=25)
parser.add_option("-s", dest="seed", type="int", help="random seed", default=None)
parser.add_option("-n", dest="samplesize", type="int", help="dataset sample size", default=100)
(options, args) = parser.parse_args()

if __name__ == '__main__':
    data_streamer = ReutersStreamReader('reuters').iterdocs()
    data = get_minibatch(data_streamer, 50000)
    labels_r = data.tags
    corpus = data.text
    
    stoplist = stopwords.words('english')
    reuters_corpus = []
    for corp in corpus:
	content = TextUtils().clean_text(corp)
	tokens = nltk.word_tokenize(content.lower().translate(None, string.punctuation))
	tokens = tokens[0: len(tokens) - 1]
	stems = []
	for word in tokens:
	    if word not in stoplist:
		stems.append(word)
	reuters_corpus.append(stems)
    labels = []
    for labs in labels_r:
	labels.append([x for x in labs if len(x)>0])

    labelset = list(set(reduce(list.__add__, labels)))

    llda = LLDA(options.K, options.alpha, options.beta)
    llda.set_corpus(labelset, reuters_corpus, labels)
    print "M=%d, V=%d, L=%d, K=%d" % (len(reuters_corpus), len(llda.vocas), len(labelset), options.K)

    for i in range(options.iteration):
        sys.stderr.write("-- %d : %.4f\n" % (i, llda.perplexity()))
        llda.inference()


    print "perplexity : %.4f" % llda.perplexity()
    #phi = llda.phi()
    with open('toddler_reuters_llda_training_data_25', 'wb') as handle:
        pickle.dump(labelset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(llda.vocas, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(llda.n_m_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(llda.n_z_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(llda.n_z, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Store labelset
# Store llda.vocas
# Store llda.
# Store llda.n_m_z
# Store llda.n_z_t
# Store llda.n_z


#print corpus[10]
#print labels[10]

#print llda.complement_label(labels[10])

#test_doc = corpus[10]

#testdoc = [llda.term_to_id(term) for term in test_doc]
#testlabel = llda.complement_label(None)

#test_n_z = numpy.zeros(len(labelset) , dtype=int)
#test_z_n = [numpy.random.multinomial(1, testlabel / testlabel.sum()).argmax() for x in range(len(testdoc))]

#for i in range(options.iteration):
    #print i
#    testlabels = llda.inference_query(testdoc)
#print testlabels
#res =  sorted(range(len(testlabels)), key=lambda i: testlabels[i])[-3:]
#print res
#print labelset[res[0]]
#print labelset[res[1]]
#print labelset[res[2]]

#test_labels = llda.inference_query(['indonesia', 'rejected', 'world', 'bank', 'recommendations', 'for', 'sweeping', 'reforms', 'to', 'its', 'farm', 'economy', 'as', 'the', 'country'])

#print phi
#sys.exit()
#for k, label in enumerate(labelset):
#    print "\n-- label %d : %s" % (k, label)
#    for w in numpy.argsort(-phi[k])[:20]:
#        print "%s: %.4f" % (llda.vocas[w], phi[k,w])
