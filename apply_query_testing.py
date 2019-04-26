#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, sys, string, random, numpy, re
import nltk
from optparse import OptionParser
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
from Reuters import *
from utils.TextUtils import *
import pickle
import math
from gensim import corpora, models, similarities
#from utils.llda import LLDA
import collections
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=1)
parser.add_option("-s", dest="seed", type="int", help="random seed", default=None)
parser.add_option("-n", dest="samplesize", type="int", help="dataset sample size", default=100)
(options, args) = parser.parse_args()

class LLDA:
    def __init__(self, K, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def test_term_to_id(self, term):
	if term in self.vocas_id:
	    return self.vocas_id[term]
	else:
	    return -1

    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
	    self.vocas_id[term] = voca_id
	    self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def complement_label(self, label):
        if not label: return numpy.ones(len(self.labelmap))
        vec = numpy.zeros(len(self.labelmap))
        for x in label: vec[self.labelmap[x]] = 1.0
        return vec

    def set_corpus(self, labelset, corpus, labels):
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)
	self.vocas = []
        self.vocas_id = dict()
        self.labels = numpy.array([self.complement_label(label) for label in labels])
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]
	M = len(corpus)
        V = len(self.vocas)
        self.z_m_n = []
        self.n_m_z = numpy.zeros((M, self.K), dtype=int)
        self.n_z_t = numpy.zeros((self.K, V), dtype=int)
        self.n_z = numpy.zeros(self.K, dtype=int)
        for m, doc, label in zip(range(M), self.docs, self.labels):
            N_m = len(doc)
            z_n = [numpy.random.multinomial(1, label / label.sum()).argmax() for x in range(N_m)]
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
	print self.n_z_t.shape

    def inference_query(self, testdoc):
        V = len(self.vocas)
	#for t, z in zip(testdoc, self.test_z_n):
	#    self.test_n_z[z] += 1  
	for n in range(len(testdoc)):
            t = testdoc[n]
	    if (t< V):
                z = self.test_z_n[n]
                denom_a = self.test_n_z.sum() + self.K * self.alpha
                denom_b = self.n_z_t.sum(axis=1) + V * self.beta
		p_z = self.testlabel*(self.n_z_t[:, t] + self.beta) / denom_b
		#print numpy.random.multinomial(1, p_z / p_z.sum())
                #new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
		new_z = p_z.argmax()
		newvalue = p_z[new_z]
	    self.test_z_n[n] = new_z
	return self.test_z_n 

    def inference(self):
        V = len(self.vocas)
        for m, doc, label in zip(range(len(self.docs)), self.docs, self.labels):
            for n in range(len(doc)):
                t = doc[n]	
                z = self.z_m_n[m][n]
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1
                denom_a = self.n_m_z[m].sum() + self.K * self.alpha
                denom_b = self.n_z_t.sum(axis=1) + V * self.beta
                p_z = label * (self.n_z_t[:, t] + self.beta) / denom_b * (self.n_m_z[m] + self.alpha) / denom_a
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                self.z_m_n[m][n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1
	    
    def phi(self):
        V = len(self.vocas)
        return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + V * self.beta)

    def theta(self):
        """document-topic distribution"""
        n_alpha = self.n_m_z + self.labels * self.alpha
        return n_alpha / n_alpha.sum(axis=1)[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.phi()
        thetas = self.theta()
	
        log_per = N = 0
        for doc, theta in zip(docs, thetas):
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

def do_training():
    with open('corpus_training', 'rb') as handle:
        corpus = pickle.load(handle)
        labels = pickle.load(handle)
    labelset = list(set(reduce(list.__add__, labels)))
    print len(corpus)
    print len(labels)
    print len(labelset)
    llda = LLDA(options.K, options.alpha, options.beta)
    llda.set_corpus(labelset, corpus, labels)
    print "M=%d, V=%d, L=%d, K=%d" % (len(corpus), len(llda.vocas), len(labelset), options.K)

    for i in range(options.iteration):
        sys.stderr.write("-- %d : %.4f\n" % (i, llda.perplexity()))
        llda.inference()
    print "perplexity : %.4f" % llda.perplexity()
    #phi = llda.phi()
    with open('toddler_llda_training_data', 'wb') as handle:
        pickle.dump(labelset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(llda.vocas, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(llda.n_m_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(llda.n_z_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(llda.n_z, handle, protocol=pickle.HIGHEST_PROTOCOL)

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def apply_query():
    print "Start Time " + time.strftime("%c")

    data_streamer = ReutersStreamReader('reuters').iterdocs()
    data = get_minibatch(data_streamer, 50000)
    labels_r_ = data.tags
    corpus_ = data.text
    corpus =  corpus_[0:100]
    labels_r = labels_r_[0:100]

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
    with open('toddler_reuters_llda_training_data_25', 'rb') as handle:
        labelset   = pickle.load(handle)
        llda.vocas = pickle.load(handle)
        llda.n_m_z = pickle.load(handle)
        llda.n_z_t = pickle.load(handle)
        llda.n_z   = pickle.load(handle)
    print llda.n_z_t.shape
    print llda.n_z_t[0]

    print "Loaded Time " + time.strftime("%c")
    test_doc = reuters_corpus[1000]
 
    testdoc = []
    for term in test_doc:
        termid = llda.test_term_to_id(term)
        if termid > -1:
            testdoc.append(termid)

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for corp in corpus:
        for token in corp:
            frequency[token] += 1

    texts = [[token for token in corp if frequency[token] > 1]
              for corp in reuters_corpus]
    text_s = [[token for token in corp if is_ascii(token)]
              for corp in reuters_corpus]

    dictionary = corpora.Dictionary(text_s)
    corpus_ds = [dictionary.doc2bow(text) for text in text_s]

    lsi = models.LsiModel(corpus_ds, id2word=dictionary, num_topics = len(labelset)) # initialize an LSI transformation

    vec_bow = dictionary.doc2bow(test_doc)
    #vec_bow = dictionary.doc2bow(test_doc.lower().split())
    vec_lsi = lsi[vec_bow] # convert the query to LSI space
    #print(vec_lsi)

    index = similarities.MatrixSimilarity(lsi[corpus_ds]) # transform corpus to LSI space and index it

    sims = index[vec_lsi] # perform a similarity query against the corpus
    #print(list(enumerate(sims)))

    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    testlabel_s = []
    for k in range(10):
	for l in labels[sims[k][0]]:
            testlabel_s.append(l)
    #print testlabel_s
    testlabelset = list(set(testlabel_s))
    print testlabelset
    llda.testlabel = llda.complement_label(testlabelset)

    #llda.testlabel = llda.complement_label(None)
    llda.test_n_z = numpy.zeros(len(labelset) , dtype=int)
    llda.test_z_n = numpy.zeros(len(test_doc))
    #llda.test_z_n = [numpy.random.multinomial(1, llda.testlabel / llda.testlabel.sum()).argmax() for x in range(len(testdoc))]
    #print llda.test_z_n
    #for i in range(options.iteration):
    print "Inference for query started"
    testlabels = llda.inference_query(testdoc)
    #nonzerindices = numpy.nonzero(testlabels)[0]
    print "Testlabels" , testlabels
    resu =  collections.Counter(testlabels)

    unique_testlabels = list(set(testlabels))
    testlabels_list = testlabels.tolist()
    for i in unique_testlabels:
        if(max(resu.values()) - min(resu.values()))>0:
            norm_val = float((testlabels_list.count(i)- min(resu.values())))/float((max(resu.values()) - min(resu.values())))
	    print labelset[int(round(i))], norm_val
            if(norm_val > 0.3):
                print labelset[int(round(i))]
	else:
	    print labelset[int(round(i))]

if __name__ == '__main__':
    print "Test labelling"
    apply_query()
