#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Labeled LDA using nltk.corpus.reuters as dataset
# This code is available under the MIT License.
# (c)2013 Nakatani Shuyo / Cybozu Labs Inc.

from Reuters import *
import sys, string, random, numpy
import nltk
from optparse import OptionParser
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
import pickle
from utils.llda import LLDA
from gensim import corpora, models, similarities
import numpy as np 
from utils.TextUtils import *
import itertools

parser = OptionParser()
parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=5)
parser.add_option("-s", dest="seed", type="int", help="random seed", default=None)
parser.add_option("-n", dest="samplesize", type="int", help="dataset sample size", default=100)
(options, args) = parser.parse_args()

class LLDA:
    def __init__(self, K, alpha, beta):
        self.alpha = alpha
        self.beta = beta

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

# reuters corpus
data_streamer = ReutersStreamReader('reuters').iterdocs()
data = get_minibatch(data_streamer, 50000)
labels_r_ = data.tags
corpus_ = data.text

corpus =  corpus_[0:100]
labels_ = labels_r_[0:100]

stoplist = stopwords.words('english')
reuters_corpus = []
labels_r = []
i = 0
for corp in corpus:
    content = TextUtils().clean_text(corp)
    tokens = nltk.word_tokenize(content.lower().translate(None, string.punctuation))
    tokens = tokens[0: len(tokens) - 1]
    if (len(tokens) > 30):
        stems = []
        for word in tokens:
            if word not in stoplist and len(word) > 2:
                stems.append(word)
        reuters_corpus.append(stems)
        labels_r.append(labels_[i]) 
    i = i + 1 

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

phi = llda.phi()

for k, label in enumerate(labelset):
    print "\n-- label %d : %s" % (k, label)
    for w in numpy.argsort(-phi[k])[:50]:
        print "%s: %.4f" % (llda.vocas[w], phi[k,w])

#  New Code for topic representation in terms of frequency patterns
words = []
documents = []
topics = []

for top in range(len(labelset)):
    for m, doc in zip(range(len(reuters_corpus)), reuters_corpus):
	for n in range(len(doc)):
            if ( top == llda.z_m_n[m][n]):
                words.append(reuters_corpus[m][n])
	documents.append(words)
	words = []
    topics.append(documents)
    documents = []

print len(reuters_corpus)
count = 0
patterns = []
topic_patterns = []
topic_patterns_list = []
for tap in topics:
    tapset = list(set(reduce(list.__add__, tap)))
    for kl in [1,2,3]:
        patterns.append(list(itertools.combinations(tapset,kl)))
    patterns_list = list(itertools.chain(*patterns))
    topic_patterns_list.append(patterns_list)
    for patns in patterns_list:
        for patdocs in tap:
	    if (set(patns).issubset(set(patdocs))):
	        count = count + 1
            if (count > 2):
	        topic_patterns.append(patns)
		count = 0
		break
    topic_patterns_list.append(topic_patterns)
    topic_patterns = []

